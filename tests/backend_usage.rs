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
async fn usage_uses_last_round_and_falls_back_when_absent() {
    for (native, reports) in [(false, true), (true, true), (false, false)] {
        let app = router().route(if native { "/api/chat" } else { "/v1/chat/completions" }, post(move |request: Request<Body>| async move {
            let body: Value = serde_json::from_slice(&axum::body::to_bytes(request.into_body(), 1024*1024).await.unwrap()).unwrap();
            assert_eq!(body["stream"], false);
            assert!(body.get("stream_options").is_none());
            let after_tool = body["messages"].as_array().unwrap().iter().any(|m| m["role"] == "tool");
            let message = if after_tool { json!({"role":"assistant","content":"finished"}) } else {
                json!({"role":"assistant","content":"","tool_calls":[{"id":"usage-tool","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"missing.txt\"}"}}]})
            };
            let mut response = if native { json!({"message":message,"done":true}) } else { json!({"choices":[{"message":message,"finish_reason":"stop"}]}) };
            if reports {
                let count = if after_tool { 1200 } else { 900 };
                if native { response["prompt_eval_count"]=json!(count); response["eval_count"]=json!(23); }
                else { response["usage"]=json!({"prompt_tokens":count,"completion_tokens":23}); }
            }
            axum::Json(response)
        }));
        let mut client = Client::start(app, native, 1).await;
        let sid = client.session();
        client.prompt(&sid, 2);
        let (updates, response) = client.response(2);
        assert_eq!(response["result"]["status"], "completed");
        let usage = updates
            .iter()
            .find(|m| m["params"]["update"]["sessionUpdate"] == "usage_update")
            .unwrap();
        if reports {
            assert_eq!(usage["params"]["update"]["used"], 1223);
        } else {
            assert!(usage["params"]["update"]["used"].as_u64().unwrap() > 0);
            assert_ne!(usage["params"]["update"]["used"], 1223);
        }
    }
}
