use axum::{
    body::Body,
    extract::Request,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct Client {
    child: Child,
    messages: mpsc::Receiver<Value>,
    server: tokio::task::JoinHandle<()>,
    workspace: std::path::PathBuf,
    log: std::path::PathBuf,
}
impl Client {
    async fn start(router: Router, native: bool, version: u64) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let workspace = std::env::temp_dir().join(format!("acp-upstream-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&workspace).unwrap();
        let log = workspace.join("agent.log");
        let endpoint = format!("http://{address}{}", if native { "" } else { "/v1" });
        let mut child = Command::new(env!("CARGO_BIN_EXE_acp-bridge"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(std::fs::File::create(&log).unwrap()))
            .env("LLM_BASE_URL", endpoint)
            .env("LLM_MODEL", "test-model")
            .env("LLM_API_KEY", "test-key")
            .env("LLM_TIMEOUT", "10")
            .env("LLM_MODEL_CONTEXT", "32768")
            .env("RUST_LOG", "acp_bridge=debug")
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, messages) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(value) = serde_json::from_str(&line) {
                    if tx.send(value).is_err() {
                        break;
                    }
                }
            }
        });
        let mut client = Self {
            child,
            messages,
            server,
            workspace,
            log,
        };
        client.send(json!({"jsonrpc":"2.0", "id":0, "method":"initialize", "params":{"protocolVersion":version}}));
        let (_, response) = client.response(0);
        assert!(response.get("error").is_none(), "{response}");
        client
    }
    fn send(&mut self, value: Value) {
        let stdin = self.child.stdin.as_mut().unwrap();
        writeln!(stdin, "{value}").unwrap();
        stdin.flush().unwrap();
    }
    fn response(&self, id: u64) -> (Vec<Value>, Value) {
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut notifications = Vec::new();
        loop {
            let value = self
                .messages
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|error| {
                    panic!(
                        "response {id}: {error}; log: {}",
                        std::fs::read_to_string(&self.log).unwrap_or_default()
                    )
                });
            if value.get("id") == Some(&json!(id)) {
                return (notifications, value);
            }
            notifications.push(value);
        }
    }
    fn session(&mut self) -> String {
        self.send(json!({"jsonrpc":"2.0", "id":1, "method":"session/new", "params":{"cwd":self.workspace}}));
        let (_, response) = self.response(1);
        response["result"]["sessionId"]
            .as_str()
            .unwrap()
            .to_string()
    }
    fn prompt(&mut self, sid: &str, id: u64) {
        self.send(json!({"jsonrpc":"2.0", "id":id, "method":"session/prompt", "params":{"sessionId":sid,"prompt":[{"type":"text","text":"hello"}]}}));
    }
}
impl Drop for Client {
    fn drop(&mut self) {
        self.child.stdin.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.server.abort();
        let _ = std::fs::remove_dir_all(&self.workspace);
    }
}
fn router() -> Router {
    Router::new()
        .route(
            "/v1/models",
            get(|| async { axum::Json(json!({"data":[{"id":"test-model"}]})) }),
        )
        .route(
            "/api/tags",
            get(|| async { axum::Json(json!({"models":[{"name":"test-model"}]})) }),
        )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutation_diffs_preserve_text_and_respect_cap_on_both_wires() {
    use acp_bridge::tools::MAX_DIFF_TEXT_BYTES;
    for version in [1, 2] {
        let scenarios = [
            (
                "write_file",
                json!({"path":"new.txt","content":"first"}),
                Some((None, "first".to_string())),
            ),
            (
                "write_file",
                json!({"path":"new.txt","content":"second"}),
                Some((Some("first"), "second".to_string())),
            ),
            (
                "edit",
                json!({"path":"new.txt","old_text":"second","new_text":"third"}),
                Some((Some("second"), "third".to_string())),
            ),
            (
                "edit",
                json!({"path":"new.txt","old_text":"missing","new_text":"x"}),
                None,
            ),
            (
                "write_file",
                json!({"path":"boundary.txt","content":"а".repeat(MAX_DIFF_TEXT_BYTES/2)}),
                Some((None, "а".repeat(MAX_DIFF_TEXT_BYTES / 2))),
            ),
            (
                "write_file",
                json!({"path":"large.txt","content":"x".repeat(MAX_DIFF_TEXT_BYTES+1)}),
                None,
            ),
            (
                "write_file",
                json!({"path":"binary.txt","content":"valid text"}),
                None,
            ),
        ];
        let requests: Vec<_> = scenarios
            .iter()
            .map(|(name, args, _)| (*name, args.clone()))
            .collect();
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let app = router().route("/v1/chat/completions", post(move |request: Request<Body>| {
            let requests = requests.clone(); let count=count.clone();
            async move {
                let body: Value = serde_json::from_slice(&axum::body::to_bytes(request.into_body(),1024*1024).await.unwrap()).unwrap();
                let n=count.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
                let message=if n.is_multiple_of(2) {
                    let (name,args)=&requests[n/2];
                    json!({"role":"assistant","content":"","tool_calls":[{"id":format!("diff-{n}"),"type":"function","function":{"name":name,"arguments":args.to_string()}}]})
                } else {
                    let tool=body["messages"].as_array().unwrap().iter().rev().find(|m| m["role"]=="tool").unwrap();
                    assert!(tool["content"].as_str().is_some());
                    json!({"role":"assistant","content":"finished"})
                };
                axum::Json(json!({"choices":[{"message":message,"finish_reason":"stop"}]}))
            }
        }));
        let mut client = Client::start(app, false, version).await;
        std::fs::write(client.workspace.join("binary.txt"), [0xff, 0xfe]).unwrap();
        let sid = client.session();
        for (index, (_, args, expected)) in scenarios.iter().enumerate() {
            let id = 2 + index as u64;
            client.prompt(&sid, id);
            let (updates, response) = client.response(id);
            assert!(response.get("error").is_none());
            let update = updates
                .iter()
                .find(|m| {
                    m["params"]["update"]["sessionUpdate"] == "tool_call_update"
                        && m["params"]["update"]["status"] == "completed"
                        && m["params"]["update"]["toolCallId"] == format!("diff-{}", index * 2)
                })
                .unwrap();
            let content = &update["params"]["update"]["content"];
            match expected {
                Some((old, new)) => {
                    let diff = &content[0];
                    assert_eq!(
                        diff["type"], "diff",
                        "version={version} scenario={index} update={update}"
                    );
                    assert_eq!(diff["oldText"], json!(old));
                    assert_eq!(diff["newText"], *new);
                    let path = std::path::Path::new(diff["path"].as_str().unwrap());
                    assert!(path.is_absolute());
                    assert_eq!(
                        path,
                        client
                            .workspace
                            .canonicalize()
                            .unwrap()
                            .join(args["path"].as_str().unwrap())
                    );
                }
                None => assert!(content.is_null(), "unexpected diff: {content}"),
            }
        }
    }
}
