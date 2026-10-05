//! acp-bridge — Minimal ACP adapter for local AI.
//!
//! Single transport: stdin/stdout JSON-RPC 2.0 (ACP).
//! Spawns by openab, Zed, JetBrains, or any ACP harness.

use acp_bridge::acp;
use acp_bridge::bench;
use acp_bridge::config::ConfigFile;
use acp_bridge::engine::{self, AppState, Notification};
use acp_bridge::hardware;
use acp_bridge::llm;
use acp_bridge::protocol::{AcpError, JsonRpcRequest, ProtocolVersion, RequestId};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Run mode
// ---------------------------------------------------------------------------

enum RunMode {
    /// stdin/stdout ACP (default)
    Acp,
    /// Benchmark mode — run fixture prompts against the configured LLM, print stats, exit.
    Bench,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Parse CLI flags before anything else
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("acp-bridge {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "acp-bridge {} — Minimal ACP adapter for local AI",
            env!("CARGO_PKG_VERSION")
        );
        println!();
        println!("USAGE:");
        println!("  acp-bridge [OPTIONS] [config.toml]");
        println!();
        println!("MODES:");
        println!("  (default)    ACP mode — stdin/stdout JSON-RPC (act as agent)");
        println!(
            "  --bench      Benchmark mode — run fixture prompts against LLM, print stats, exit"
        );
        println!();
        println!("OPTIONS:");
        println!("  --version    Print version");
        println!("  --help       Print this help");
        println!();
        println!("ENVIRONMENT:");
        println!("  LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_TIMEOUT, ...");
        return;
    }

    let mode = if args.iter().any(|a| a == "--bench") {
        RunMode::Bench
    } else {
        RunMode::Acp
    };

    // Initialize tracing — stderr by default (stdout is the ACP wire),
    // or an append-only log file when ACP_LOG_FILE is set. bb's host
    // daemon swallows agent stderr (bridge INFO/DEBUG lines never reach
    // ~/.bb logs), so under bb the per-round override lines (#40/#43)
    // need a file the bridge owns. Zero config-file surface: one env var.
    let default_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "acp_bridge=info".parse().unwrap());
    match std::env::var("ACP_LOG_FILE") {
        Ok(path) => {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|e| panic!("ACP_LOG_FILE '{path}' unwritable: {e}"));
            tracing_subscriber::fmt()
                .with_env_filter(default_filter)
                .with_target(true)
                .with_writer(std::sync::Mutex::new(file))
                .init();
        }
        Err(_) => {
            tracing_subscriber::fmt()
                .with_env_filter(default_filter)
                .with_target(true)
                .with_writer(std::io::stderr)
                .init();
        }
    }
    // Binary identity (reports should be actionable): the git short
    // hash stamped at build time, logged after tracing init so it
    // lands in the ACP_LOG_FILE under stderr-swallowing hosts.
    info!(
        version = env!("CARGO_PKG_VERSION"),
        build_hash = env!("ACP_BUILD_HASH"),
        "binary identity"
    );

    // Load config: CLI arg (optional TOML path) → env vars → defaults
    let config_path = args.iter().skip(1).find(|a| !a.starts_with('-')).cloned();

    let config_file = config_path
        .as_ref()
        .map(|path| ConfigFile::load(std::path::Path::new(path)));

    let config = match config_file {
        Some(file) => file.into_llm_config(),
        None => llm::LlmConfig::from_env(),
    };

    if let RunMode::Bench = mode {
        for line in hardware::detect().report_lines() {
            info!("{line}");
        }
        info!(
            base_url = %config.base_url,
            model = %config.model,
            "Running benchmark"
        );
        let results = bench::run(&config, &bench::default_fixtures()).await;
        bench::print_report(&config, &results);
        return;
    }

    info!(
        version = env!("CARGO_PKG_VERSION"),
        model = %config.model,
        base_url = %config.base_url,
        backend = ?config.backend(),
        max_history_turns = config.max_history_turns,
        max_tool_rounds = config.max_tool_rounds,
        max_sessions = config.max_sessions,
        session_idle_timeout_secs = config.session_idle_timeout_secs,
        "Starting acp-bridge"
    );

    for line in hardware::detect().report_lines() {
        info!("{line}");
    }

    // Probe backend and capture the model list (issue #40): the
    // startup probe's result feeds the `model` config option so
    // Clients can switch models per session.
    let mut config = config;
    config.available_models = {
        let mut list = probe_backend(&config).await;
        // The configured model is always offered and stays the default.
        if !list.iter().any(|m| m == &config.model) {
            list.insert(0, config.model.clone());
        }
        list
    };

    // Build shared state
    // Session persistence (issue #17): SQLite store unless disabled via
    // ACP_PERSISTENCE=off. Opening or pruning failures degrade to no
    // persistence — the loadSession capability is simply not advertised,
    // so spec-compliant Clients never attempt restore.
    let persistence_disabled = matches!(
        std::env::var("ACP_PERSISTENCE").as_deref(),
        Ok("off") | Ok("false") | Ok("0")
    );
    let store = if persistence_disabled {
        None
    } else {
        // Issue #22: derive the default DB filename from the config
        // file stem (`cometapi.toml` → `sessions-cometapi.db`) so
        // multiple agent configs get isolated stores by default. Bare
        // invocations keep `sessions.db`; explicit `ACP_SESSION_DB`
        // always wins.
        let db_path = std::env::var("ACP_SESSION_DB")
            .map(PathBuf::from)
            .unwrap_or_else(|_| acp_bridge::session_store::default_db_path(config_path.as_deref()));
        match acp_bridge::session_store::SessionStore::open(&db_path) {
            Ok(s) => {
                let keep = std::env::var("ACP_SESSION_RETENTION")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(100);
                match s.prune(keep) {
                    Ok(n) if n > 0 => {
                        info!(deleted = n, keep, "Pruned old persisted sessions")
                    }
                    Ok(_) => {}
                    Err(e) => warn!(error = %e, "Session retention prune failed"),
                }
                Some(Arc::new(s))
            }
            Err(e) => {
                warn!(error = %e, path = %db_path.display(), "Failed to open session store; persistence disabled");
                None
            }
        }
    };

    // Issue #22 strict mode: `ACP_SESSION_STRICT_MODELS=1` makes
    // session/load refuse to restore a session persisted by a
    // different agent identity. Default is permissive (restore
    // anything in the store).
    let strict_models = matches!(
        std::env::var("ACP_SESSION_STRICT_MODELS").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    );
    let agent_identity = config_path
        .as_ref()
        .and_then(|p| {
            std::path::Path::new(p)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .or_else(|| Some(config.model.clone()));

    let mut state = AppState::with_store(config, store, agent_identity);
    // Fresh Arc — no clones exist yet, so direct mutation is sound.
    // (The protocol_version negotiation uses the same pattern later.)
    Arc::get_mut(&mut state)
        .expect("AppState not yet shared")
        .strict_models = strict_models;

    // Spawn idle session cleanup task
    let idle_timeout = state.config.session_idle_timeout_secs;
    if idle_timeout > 0 {
        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let interval = Duration::from_secs(idle_timeout.min(60));
            loop {
                tokio::time::sleep(interval).await;
                state_clone.evict_idle_sessions(idle_timeout);
            }
        });
    }

    // Run ACP stdin/stdout loop
    run_acp_loop(state).await;
}

// ---------------------------------------------------------------------------
// Backend probing (shared by both modes)
// ---------------------------------------------------------------------------

async fn probe_backend(config: &llm::LlmConfig) -> Vec<String> {
    let available = match llm::probe_backend(config).await {
        Ok(models) if models.is_empty() => {
            info!("Connected to backend (no models listed)");
            Vec::new()
        }
        Ok(models) => {
            info!(count = models.len(), "Available models:");
            for m in &models {
                info!("  - {m}");
            }
            if !models.iter().any(|m| {
                m.starts_with(&config.model)
                    || config.model.starts_with(m.split(':').next().unwrap_or(""))
            }) {
                warn!(configured = %config.model, "Configured model not found in available models");
            }
            models
        }
        Err(reason) => {
            warn!(
                base_url = %config.base_url,
                error = %reason,
                "Cannot reach backend — will retry on first request"
            );
            Vec::new()
        }
    };

    // Query model info (Ollama native only)
    if let Some(info) = llm::query_model_info(config).await {
        info!(
            context_length = info.context_length,
            "Model info from /api/show"
        );
    }

    // Check running models (Ollama)
    if let Some(running) = llm::query_running_models(config).await {
        if running.is_empty() {
            warn!(
                model = %config.model,
                "No models loaded in VRAM — first request may be slow. Run: ollama run {}",
                config.model
            );
        } else {
            info!(count = running.len(), "Running models (loaded in VRAM):");
            for m in &running {
                info!("  - {m}");
            }
        }
    }

    available
}

// ---------------------------------------------------------------------------
// ACP mode — stdin/stdout JSON-RPC loop
// ---------------------------------------------------------------------------

/// Negotiate the ACP wire-format version from a Client's
/// `initialize` request parameters.
///
/// The Client's `protocolVersion` is an integer (1 or 2 today).
/// Negotiation rule: pick the highest version that both sides
/// support. We support v1 and v2; anything newer than v2 falls back to
/// v2 with a warning. Missing field → v1 (conservative default that
/// works with every existing Client).
///
/// This is the *exact* protocol rule described in
/// <https://agentclientprotocol.com/protocol/initialization> and
/// mirrored by the official Rust / TypeScript / Elixir SDKs.
fn negotiate_protocol_version(params: Option<Value>) -> ProtocolVersion {
    let requested = params
        .as_ref()
        .and_then(|p| p.get("protocolVersion"))
        .and_then(|v| v.as_u64())
        .map(|n| ProtocolVersion(n as u16));
    match requested {
        None => {
            debug!("Client omitted protocolVersion, defaulting to v1");
            ProtocolVersion::V1
        }
        Some(v) if v.as_u16() <= ProtocolVersion::V1.as_u16() => {
            debug!(requested = %v, "Negotiating v1");
            ProtocolVersion::V1
        }
        Some(v) if v.as_u16() == ProtocolVersion::V2.as_u16() => {
            debug!("Negotiating v2");
            ProtocolVersion::V2
        }
        Some(v) => {
            warn!(
                requested = %v,
                "Client requested a version newer than we support; falling back to v2"
            );
            ProtocolVersion::V2
        }
    }
}

async fn run_acp_loop(mut state: Arc<AppState>) {
    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    loop {
        tokio::select! {
            line_result = lines.next_line() => {
                match line_result {
                    Ok(Some(line)) => {
                        let trimmed = line.trim().to_string();
                        if trimmed.is_empty() {
                            continue;
                        }

                        let msg: JsonRpcRequest = match serde_json::from_str(&trimmed) {
                            Ok(m) => m,
                            Err(e) => {
                                debug!(error = %e, "Skipping invalid JSON-RPC line");
                                continue;
                            }
                        };

                        let id_opt = msg.id;
                        let method = msg.method.as_str();
                        let params = msg.params.clone().unwrap_or(json!({}));

                        debug!(?id_opt, method, "Received message");

                        // Notifications (no id, no response expected per JSON-RPC 2.0).
                        let id = match id_opt {
                            Some(id) => id,
                            None => {
                                match method {
                                    "session/cancel" => {
                                        let sid = params
                                            .get("sessionId")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        // Flip the in-flight turn's cancel
                                        // channel (issue #3). The turn's
                                        // response task observes it, aborts
                                        // the engine, and answers with
                                        // stopReason "cancelled". No turn
                                        // in flight → nothing to do.
                                        match state
                                            .turn_registry
                                            .lock()
                                            .expect("turn registry lock")
                                            .get(sid)
                                        {
                                            Some((gen, cancel_tx)) => {
                                                info!(
                                                    session_id = %sid,
                                                    generation = gen,
                                                    "Cancelling in-flight turn"
                                                );
                                                let _ = cancel_tx.send(true);
                                            }
                                            None => {
                                                debug!(
                                                    session_id = %sid,
                                                    "session/cancel for session without in-flight turn"
                                                );
                                            }
                                        }
                                    }
                                    _ => {
                                        debug!(method, "Ignoring unknown notification");
                                    }
                                }
                                continue;
                            }
                        };

                        match method {
                            "initialize" => {
                                // Negotiate the wire-format version. The Client
                                // sends its preferred version; we choose the
                                // highest version we both support. If the
                                // Client omits the field we fall back to v1
                                // (the conservative default).
                                let negotiated = negotiate_protocol_version(Some(params.clone()));
                                // Arc::make_mut gives us &mut AppState when
                                // there's only one strong reference; if the
                                // Arc is shared, it clones. Inside
                                // run_acp_loop the Arc has refcount 1, so
                                // make_mut never actually clones.
                                Arc::make_mut(&mut state).protocol_version = negotiated;
                                let result = engine::initialize(
                                    &state.config,
                                    negotiated,
                                    state.store.is_some(),
                                );
                                acp::send_response(&id, result);
                            }
                            "session/new" => {
                                let raw_cwd = params.get("cwd").and_then(|v| v.as_str()).unwrap_or("/tmp");
                                if let Some(servers) = params.get("mcpServers").and_then(|v| v.as_array()) {
                                    if !servers.is_empty() {
                                        debug!(count = servers.len(), "Ignoring mcpServers param (not supported in v0.7)");
                                    }
                                }
                                match engine::session_new(&state, raw_cwd, state.protocol_version) {
                                    Ok(session_id) => {
                                        // Post-creation notifications first so
                                        // Clients that want to render the
                                        // slash-command menu / session title
                                        // as soon as the sessionId arrives
                                        // see them attached to the same
                                        // session. Notifications carry the
                                        // sessionId they apply to, so sending
                                        // them before the response is safe
                                        // (Client binds them on receipt of
                                        // the response anyway). Routed via
                                        // the v1/v2 dispatcher so v2 Clients
                                        // get the v2 wire shape (e.g.
                                        // plan_update with planId).
                                        engine::session_new_post_create_notifications(
                                            &session_id,
                                            raw_cwd,
                                            state.protocol_version,
                                        );
                                        acp::send_response(
                                            &id,
                                            engine::session_new_response(&state, &session_id),
                                        );
                                    }
                                    Err(e) => {
                                        acp::send_error(&id, e.code(), &e.to_string());
                                    }
                                }
                            }
                            "session/prompt" => {
                                // Runs as its own task: the main loop must
                                // keep reading stdin during a turn, or
                                // `session/cancel` would sit unread until
                                // the turn completes (issue #3).
                                let sid = params
                                    .get("sessionId")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                // One turn per session at a time: a second
                                // prompt while a turn runs would race the
                                // session history (issue #3). The Client
                                // cancels first if it wants to redirect.
                                if state
                                    .turn_registry
                                    .lock()
                                    .expect("turn registry lock")
                                    .contains_key(&sid)
                                {
                                    // Steering (issue #30): a mid-turn
                                    // prompt is queued and injected as a
                                    // user message at the next tool-round
                                    // boundary — NOT rejected. Its
                                    // JSON-RPC request is held open
                                    // (answered by SteerAck on injection,
                                    // or by the turn-end flush if the
                                    // turn never reaches a boundary).
                                    // Validation happens now: an invalid
                                    // steer errors immediately, before
                                    // anything is queued.
                                    let raw = engine::extract_user_text_from_prompt(
                                        &params.get("prompt").cloned().unwrap_or(Value::Null),
                                    );
                                    let (text, _) = engine::strip_sender_context(&raw);
                                    let images = engine::extract_user_images_from_prompt(
                                        &params.get("prompt").cloned().unwrap_or(Value::Null),
                                    );
                                    if text.trim().is_empty() && images.is_empty() {
                                        acp::send_error(
                                            &id,
                                            -32602,
                                            "prompt (expected non-empty text or image content)",
                                        );
                                        continue;
                                    }
                                    let entry = engine::SteerEntry {
                                        text,
                                        images,
                                        request_id: id.clone(),
                                        message_id: uuid::Uuid::new_v4().to_string(),
                                    };
                                    let queued = state.queue_steer(&sid, entry);
                                    info!(session_id = %sid, queued, "Queued steer for in-flight turn");
                                    continue;
                                }
                                let generation = state
                                    .turn_registry
                                    .lock()
                                    .expect("turn registry lock")
                                    .get(&sid)
                                    .map(|(g, _)| *g + 1)
                                    .unwrap_or(0);
                                let (cancel_tx, cancel_rx) =
                                    tokio::sync::watch::channel(false);
                                state
                                    .turn_registry
                                    .lock()
                                    .expect("turn registry lock")
                                    .insert(sid, (generation, cancel_tx));
                                let state_task = Arc::clone(&state);
                                let params_task = params.clone();
                                tokio::spawn(async move {
                                    handle_acp_prompt(
                                        id,
                                        &params_task,
                                        &state_task,
                                        generation,
                                        cancel_rx,
                                    )
                                    .await;
                                });
                            }
                            "session/set_config_option" => {
                                // Issue #13 (bb's dynamic-config surface):
                                // the Client picks a thought level from the
                                // `configOptions` advertised at `session/new`;
                                // the choice applies to this session's
                                // subsequent rounds.
                                let session_id =
                                    params.get("sessionId").and_then(|v| v.as_str()).unwrap_or("");
                                let config_id =
                                    params.get("configId").and_then(|v| v.as_str()).unwrap_or("");
                                let value = params.get("value").cloned().unwrap_or(Value::Null);
                                if session_id.is_empty() || config_id.is_empty() {
                                    let err = AcpError::MissingParam {
                                        field: "sessionId, configId".into(),
                                    };
                                    acp::send_error(&id, err.code(), &err.to_string());
                                } else {
                                    match engine::session_set_config_option(
                                        &state, session_id, config_id, &value,
                                    ) {
                                        Ok(updated) => acp::send_response(&id, updated),
                                        Err(e) => acp::send_error(&id, e.code(), &e.to_string()),
                                    }
                                }
                            }
                            "session/end" | "session/close" => {
                                let session_id = params.get("sessionId").and_then(|v| v.as_str()).unwrap_or("");
                                if session_id.is_empty() {
                                    let err = AcpError::MissingParam { field: "sessionId".into() };
                                    acp::send_error(&id, err.code(), &err.to_string());
                                } else {
                                    // `session/end` is the v1 method;
                                    // `session/close` is the v2 baseline
                                    // alias (see schema/v2 SessionCapabilities).
                                    // Both share the same implementation:
                                    // remove the in-memory session. No
                                    // persistence, so re-opening is a no-op.
                                    match engine::session_end(&state, session_id) {
                                        Ok(()) => acp::send_response(&id, json!({"status": "ended"})),
                                        Err(e) => acp::send_error(&id, e.code(), &e.to_string()),
                                    }
                                }
                            }
                            "session/list" => {
                                // v2 baseline session method. Lists currently
                                // active sessions. Returns an empty `sessions`
                                // array (and cursor: null) when there are none.
                                let sessions = engine::session_list(&state);
                                acp::send_response(
                                    &id,
                                    json!({
                                        "sessions": sessions,
                                        "nextCursor": null,
                                    }),
                                );
                            }
                            "session/load" | "session/resume" => {
                                // Both methods restore a persisted session.
                                // `session/load` additionally replays the
                                // conversation as session/update
                                // notifications BEFORE the response (spec
                                // order); `session/resume` restores context
                                // without replay. Requires persistence —
                                // without it the historical `-32001
                                // no_persistence` rejection stands.
                                let Some(store) = &state.store else {
                                    acp::send_error_with_data(
                                        &id,
                                        -32001,
                                        "acp-bridge has no persistence layer; session/load and session/resume are unavailable",
                                        json!({ "reason": "no_persistence" }),
                                    );
                                    continue;
                                };
                                let Some(session_id) = params.get("sessionId").and_then(|v| v.as_str()) else {
                                    acp::send_error(&id, AcpError::MissingParam { field: "sessionId".into() }.code(), &AcpError::MissingParam { field: "sessionId".into() }.to_string());
                                    continue;
                                };
                                let Some(cwd) = params.get("cwd").and_then(|v| v.as_str()) else {
                                    acp::send_error(&id, AcpError::MissingParam { field: "cwd".into() }.code(), &AcpError::MissingParam { field: "cwd".into() }.to_string());
                                    continue;
                                };
                                let replay = method == "session/load";
                                match engine::session_restore(&state, replay, session_id, cwd, store) {
                                    Ok(events) => {
                                        // Spec order: the entire replay
                                        // precedes the response.
                                        for event in &events {
                                            acp::notify_replay_event(
                                                state.protocol_version,
                                                session_id,
                                                event,
                                            );
                                        }
                                        acp::send_response(&id, json!({}));
                                    }
                                    Err(e) => acp::send_error(&id, e.code(), &e.to_string()),
                                }
                            }
                            "session/delete" => {
                                // v2 optional session method (advertised via
                                // `capabilities.session.delete: {}`). acp-bridge
                                // does not implement deletion from `session/list`
                                // — `session/close` is the equivalent.
                                acp::send_error_with_data(
                                    &id,
                                    -32601,
                                    "session/delete is not supported by acp-bridge; use session/close instead",
                                    json!({ "reason": "not_implemented" }),
                                );
                            }
                            "session/set_mode" => {
                                // session/new does not include a `modes` array, so
                                // per ACP spec this method is not applicable to
                                // sessions created by acp-bridge. Spec-compliant
                                // Clients detect this via the missing `modes` array
                                // and skip the call entirely; this branch is the
                                // safety net for non-compliant Clients.
                                acp::send_error_with_data(
                                    &id,
                                    -32602,
                                    "session/set_mode is not applicable; sessions created by acp-bridge have no modes",
                                    json!({ "reason": "no_modes" }),
                                );
                            }
                            "auth/login" | "auth/logout" => {
                                // acp-bridge advertises `authMethods: []` at
                                // initialize, so a spec-compliant Client should not
                                // call auth methods. Return a clear reason so the
                                // Client can log it.
                                acp::send_error_with_data(
                                    &id,
                                    -32601,
                                    "acp-bridge does not implement authentication; authMethods is empty at initialize",
                                    json!({ "reason": "no_auth_methods" }),
                                );
                            }
                            method if method.starts_with("fs/") => {
                                // The Client-side fs surface is for agents that
                                // want to read/write files via the editor.
                                // acp-bridge uses its own built-in tool set
                                // (`read_file` / `write_file` / `list_dir` / …)
                                // and never calls these. Spec-compliant Clients
                                // detect the agent's lack of need via
                                // `agentCapabilities`; this is the safety net.
                                acp::send_error_with_data(
                                    &id,
                                    -32601,
                                    "acp-bridge does not call client-side filesystem methods",
                                    json!({ "reason": "agent_does_not_call_client_fs" }),
                                );
                            }
                            method if method.starts_with("terminal/") => {
                                acp::send_error_with_data(
                                    &id,
                                    -32601,
                                    "acp-bridge does not call client-side terminal methods",
                                    json!({ "reason": "agent_does_not_call_client_terminal" }),
                                );
                            }
                            _ => {
                                let err = AcpError::MethodNotFound { method: method.to_string() };
                                acp::send_error(&id, err.code(), &err.to_string());
                            }
                        }
                    }
                    Ok(None) => {
                        info!("stdin closed, shutting down gracefully");
                        break;
                    }
                    Err(e) => {
                        error!(error = %e, "Error reading stdin");
                        break;
                    }
                }
            }
            _ = shutdown_signal() => {
                info!("Received shutdown signal, exiting");
                // Signal-initiated shutdown must not return through the
                // runtime drop: tokio's stdin reader is a blocking-pool
                // task parked in an OS read on stdin, and BlockingPool's
                // Drop waits for it — forever, when a Client still holds
                // the pipe open. Exit the process explicitly instead;
                // the OS closes the read end (issue #18).
                let session_count = state.cleanup();
                if session_count > 0 {
                    info!(sessions = session_count, "Cleaned up sessions on exit");
                }
                std::process::exit(0);
            }
        }
    }

    // Cleanup — stdin-close path: the blocking stdin read has already
    // returned, so the runtime drops cleanly here.
    let session_count = state.cleanup();
    if session_count > 0 {
        info!(sessions = session_count, "Cleaned up sessions on exit");
    }
}

/// Completes on SIGINT (Ctrl-C) or SIGTERM — the two termination signals
/// supervisors and update tooling actually send. Previously only SIGINT
/// was handled; SIGTERM killed the process abruptly, skipping the
/// graceful-shutdown path and exiting non-zero (issue #18).
async fn shutdown_signal() {
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut sigterm) => {
                    sigterm.recv().await;
                }
                Err(e) => {
                    // Unix-only API; on other platforms this arm simply
                    // never fires and Ctrl-C remains the shutdown path.
                    debug!(error = %e, "SIGTERM handler unavailable on this platform");
                    std::future::pending::<()>().await;
                }
            }
        } => {}
    }
}

/// Handle ACP session/prompt — runs engine and streams notifications to stdout.
///
/// Runs as its own task (spawned by the main loop) so stdin keeps being
/// read while the turn executes. Cancellation (issue #3): when the
/// Client sends `session/cancel`, the cancel handler flips the
/// session's watch channel; the notification drain selects on it,
/// aborts the engine task, and responds with `stopReason: "cancelled"`
/// (spec: the Agent MUST answer the original prompt request with the
/// cancelled stop reason — Clients must not see cancellation as an
/// error).
///
/// The registry entry created by the main loop before spawning is
/// removed on every exit path via [`RegistryGuard`].
async fn handle_acp_prompt(
    id: RequestId,
    params: &Value,
    state: &Arc<AppState>,
    generation: u64,
    mut cancel_rx: tokio::sync::watch::Receiver<bool>,
) {
    struct RegistryGuard<'a> {
        state: &'a AppState,
        session_id: &'a str,
        generation: u64,
    }
    impl Drop for RegistryGuard<'_> {
        fn drop(&mut self) {
            let mut registry = self.state.turn_registry.lock().expect("turn registry lock");
            // Remove only if we still own the entry — a newer turn for
            // the same session would have replaced it.
            if registry.get(self.session_id).map(|(g, _)| *g) == Some(self.generation) {
                registry.remove(self.session_id);
            }
        }
    }
    let _registry_guard = RegistryGuard {
        state,
        session_id: params
            .get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        generation,
    };

    let session_id = match params.get("sessionId").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => {
            let err = AcpError::MissingParam {
                field: "sessionId".into(),
            };
            acp::send_error(&id, err.code(), &err.to_string());
            return;
        }
    };

    let prompt_value = params.get("prompt").cloned().unwrap_or(Value::Null);
    let prompt_kind = match &prompt_value {
        Value::Null => "null",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
        _ => "other",
    };
    debug!(prompt_kind, "session/prompt input shape");
    // Selection-intent carrier (bb#4433 research): bb's composer may
    // attach model/reasoning to the prompt request as fields. Log the
    // full top-level key set plus any non-standard field values, so
    // intent delivery is verifiable regardless of field naming.
    if let Some(fields) = params.as_object() {
        let mut keys: Vec<&String> = fields.keys().collect();
        keys.sort();
        let standard = ["sessionId", "prompt", "mcpServers", "_meta"];
        let extras: serde_json::Map<String, Value> = fields
            .iter()
            .filter(|(k, _)| !standard.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        debug!(
            keys = %keys.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","),
            ?extras,
            "session/prompt top-level fields"
        );
    }

    let raw_user_text = engine::extract_user_text_from_prompt(&prompt_value);
    let (user_text, sender_context) = engine::strip_sender_context(&raw_user_text);
    if let Some(ctx) = &sender_context {
        debug!(
            sender_context_len = ctx.len(),
            "Stripped <sender_context> block from user text"
        );
    }
    let user_images = engine::extract_user_images_from_prompt(&prompt_value);

    if user_text.trim().is_empty() && user_images.is_empty() {
        let err = AcpError::MissingParam {
            field: "prompt (expected non-empty text or image content)".into(),
        };
        acp::send_error(&id, err.code(), &err.to_string());
        return;
    }

    // Mint a messageId for this prompt. v2's PromptResponse schema requires
    // it (`required: ["messageId"]`); v1 Clients ignore the field but we
    // send it on every wire version to keep the response shape
    // identical between versions. The id is opaque to the Client — we use
    // a UUID so two concurrent turns in the same session can never
    // collide.
    let message_id = uuid::Uuid::new_v4().to_string();

    // Compute a session title from the user's first non-empty line so
    // Clients can surface a meaningful name in their session list. Done
    // before spawning the engine task because `user_text` is moved into
    // the task below.
    let derived_title = user_text
        .lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().chars().take(80).collect::<String>())
        .filter(|s: &String| !s.is_empty());

    // Set up notification channel for ACP streaming
    let (notify_tx, mut notify_rx) = mpsc::unbounded_channel::<Notification>();

    // Spawn the engine prompt in a task so we can drain notifications
    let state_clone = Arc::clone(state);
    let sid = session_id.clone();
    let mid = message_id.clone();
    let handle = tokio::spawn(async move {
        engine::session_prompt(
            &state_clone,
            &sid,
            &user_text,
            &user_images,
            Some(notify_tx),
            &mid,
        )
        .await
    });

    // Drain notifications to ACP stdout, racing the cancel channel.
    // Each notification is routed through the `_for` dispatcher so v2
    // Clients receive v2-shaped payloads (e.g. `tool_call_update`
    // instead of `tool_call`) and v1 Clients receive the legacy shapes
    // they were written against.
    let mut cancelled = false;
    loop {
        tokio::select! {
            notif = notify_rx.recv() => {
                match notif {
                    Some(Notification::Thinking) => {
                        acp::notify_thinking_for(state.protocol_version, &session_id)
                    }
                    Some(Notification::ThinkingText { text }) => {
                        acp::notify_thinking_text(&session_id, &text)
                    }
                    Some(Notification::ToolStart { id, name, args }) => {
                        acp::notify_tool_start_for(
                            state.protocol_version,
                            &session_id,
                            &id,
                            &name,
                            &args,
                        )
                    }
                    Some(Notification::ToolDone { id, status, result, diff, .. }) => {
                        acp::notify_tool_done_for(
                            state.protocol_version,
                            &session_id,
                            &id,
                            &status,
                            result.as_deref(),
                            diff.as_ref(),
                        )
                    }
                    Some(Notification::TextChunk(text)) => {
                        acp::notify_text_for(state.protocol_version, &session_id, &text)
                    }
                    Some(Notification::SteerAck {
                        request_id,
                        message_id,
                    }) => {
                        // Ack-on-injection (issue #30): the steer just
                        // reached session history; answer the held
                        // request now so the Client's pending state
                        // clears exactly when the model can see it.
                        if state.protocol_version == ProtocolVersion::V2 {
                            acp::send_response(&request_id, json!({ "messageId": message_id }));
                        } else {
                            acp::send_response(
                                &request_id,
                                json!({
                                    "stopReason": "end_turn",
                                    "status": "steered",
                                    "text": "",
                                }),
                            );
                        }
                    }
                    None => break, // engine task finished; all notifications seen
                }
            }
            _ = cancel_rx.changed() => {
                if *cancel_rx.borrow() {
                    cancelled = true;
                    // Abort the engine at its current await point. Any
                    // in-flight LLM request or tool execution dies here;
                    // partial round state is discarded (per-round
                    // persistence means completed rounds survive).
                    handle.abort();
                    // Drain whatever the engine managed to send before
                    // dying so notifications stay on the wire.
                    while let Ok(notif) = notify_rx.try_recv() {
                        match notif {
                            Notification::Thinking => {
                                acp::notify_thinking_for(
                                    state.protocol_version,
                                    &session_id,
                                )
                            }
                            Notification::ThinkingText { text } => {
                                acp::notify_thinking_text(&session_id, &text)
                            }
                            Notification::ToolStart { id, name, args } => {
                                acp::notify_tool_start_for(
                                    state.protocol_version,
                                    &session_id,
                                    &id,
                                    &name,
                                    &args,
                                )
                            }
                            Notification::ToolDone { id, status, result, diff, .. } => {
                                acp::notify_tool_done_for(
                                    state.protocol_version,
                                    &session_id,
                                    &id,
                                    &status,
                                    result.as_deref(),
                                    diff.as_ref(),
                                )
                            }
                            Notification::TextChunk(text) => {
                                acp::notify_text_for(
                                    state.protocol_version,
                                    &session_id,
                                    &text,
                                )
                            }
                            Notification::SteerAck {
                                request_id,
                                message_id,
                            } => {
                                // Ack-on-injection during the cancel
                                // drain too (issue #30): a steer that
                                // was injected in the instant before
                                // cancellation still gets its response.
                                if state.protocol_version == ProtocolVersion::V2 {
                                    acp::send_response(
                                        &request_id,
                                        json!({ "messageId": message_id }),
                                    );
                                } else {
                                    acp::send_response(
                                        &request_id,
                                        json!({
                                            "stopReason": "end_turn",
                                            "status": "steered",
                                            "text": "",
                                        }),
                                    );
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // De-register the turn regardless of outcome.
    state
        .turn_registry
        .lock()
        .expect("turn registry lock")
        .remove(&session_id);

    // Steering (issue #30), no-loss guarantee: the turn is over —
    // anything still queued either arrived in the final race window
    // (queued after the last injection point but before deregistration)
    // or the turn was cancelled with steers pending. Nothing may be
    // silently dropped: each residual steer is answered HERE as a
    // rejected/aborted response — the Client knows its message never
    // reached the model and can resend it as a fresh prompt.
    for steer in state.drain_steers(&session_id) {
        warn!(session_id = %session_id, "Turn ended with steers still queued; answering as aborted");
        acp::send_error(
            &steer.request_id,
            -32001,
            "Steer not delivered: the turn ended before a tool-round boundary; resend as a new prompt",
        );
    }

    if cancelled {
        // Spec (issue #3): respond with the cancelled stop reason —
        // cancellation is not an error, and the Client MUST see the
        // prompt request answered. Version dispatch mirrors the normal
        // response path: v1 keeps the legacy body, v2 reports the turn
        // end via state_update and answers with messageId only.
        if state.protocol_version == ProtocolVersion::V2 {
            acp::notify_state_idle_for(state.protocol_version, &session_id, Some("cancelled"));
            acp::send_response(&id, json!({ "messageId": message_id }));
        } else {
            acp::send_response(
                &id,
                json!({
                    "stopReason": "cancelled",
                    "status": "cancelled",
                    "text": "",
                }),
            );
        }
        return;
    }

    let result = handle.await.unwrap_or_else(|_| engine::PromptResult {
        status: "failed".into(),
        text: "Internal error".into(),
        error: None,
        usage: engine::UsageReport {
            used: 0,
            size: state.config.context_size,
        },
        error_class: Some(acp_bridge::llm::LlmErrorKind::Unknown),
        error_retryable: false,
        message_id: message_id.clone(),
    });

    // If the engine returned a protocol error (e.g. unknown session), send JSON-RPC error
    if let Some(err) = &result.error {
        acp::send_error(&id, err.code(), &err.to_string());
        return;
    }

    // A successful prompt ends a turn, which ACP signals via `stopReason:
    // "end_turn"`. The generated text is delivered through the streamed
    // `agent_message_chunk` notifications.
    //
    // The legacy `status`/`text` fields are kept alongside `stopReason` so
    // existing OpenAB pipelines that rely on the final response body keep
    // working; standard ACP clients read `stopReason` and ignore the extra
    // fields. Once OpenAB migrates off the legacy format this can be trimmed.
    // ACP v1 StopReason values: end_turn / max_tokens / max_turn_requests /
    // refusal / cancelled. There is no generic "error" reason, so a turn that
    // hits the tool-call round limit maps to `max_turn_requests` (the model
    // made the maximum number of requests in a single turn). The actual error
    // text is already delivered via the streamed `agent_message_chunk` and the
    // legacy `status`/`text` fields below.
    // The legacy `status`/`text` fields are kept alongside `stopReason` so
    // existing OpenAB pipelines that rely on the final response body keep
    // working; standard ACP clients read `stopReason` and ignore the extra
    // fields. Once OpenAB migrates off the legacy format this can be trimmed.
    // ACP v1 StopReason values: end_turn / max_tokens / max_turn_requests /
    // refusal / cancelled. There is no generic "error" reason, so a turn that
    // hits the tool-call round limit maps to `max_turn_requests` (the model
    // made the maximum number of requests in a single turn). The actual error
    // text is already delivered via the streamed `agent_message_chunk` and the
    // legacy `status`/`text` fields below. The stop_reason value lives
    // inside the response-body dispatch below; v1 puts it on the
    // response directly, v2 puts it on `state_update`.

    // If the turn failed due to a classified LLM error, attach the
    // classification to the response so Clients can branch on
    // `error.category` and `error.retryable` without parsing the prose
    // `text` field. We only emit this when `error_class` is set; pure
    // `max_turn_requests` (tool-loop exhaustion) does not classify.
    let error_meta = result.error_class.as_ref().map(|k| {
        json!({
            "category": k.as_str(),
            "retryable": k.is_retryable(),
        })
    });

    // Post-turn notifications come BEFORE the response so Clients that
    // collect both (e.g. via `read_until_response`) see them bound to
    // this turn. The notification `sessionId` field already tells the
    // Client which session they apply to. Routed through the v1/v2
    // dispatcher so v2 Clients get the v2 wire shape.
    acp::notify_usage_for(
        state.protocol_version,
        &session_id,
        result.usage.used,
        result.usage.size,
        None,
    );
    if let Some(title) = derived_title.as_deref() {
        acp::notify_session_info_for(state.protocol_version, &session_id, title, None);
    }

    // v2 Clients expect a `state_update` with `stopReason` so they can
    // render the prompt input affordably. v1 Clients ignore unknown
    // sessionUpdate discriminators per the JSON-RPC spec, so it's safe
    // to always call the dispatcher (it no-ops on v1).
    if state.protocol_version == ProtocolVersion::V2 {
        let stop_reason = if result.status == "completed" {
            "end_turn"
        } else {
            "max_turn_requests"
        };
        acp::notify_state_idle_for(state.protocol_version, &session_id, Some(stop_reason));
    }

    // Wire-shape dispatch on the response itself. v1 Clients keep the
    // legacy `{stopReason, status, text, error?}` shape (ACP v1 has no
    // concept of messageId; the OpenAB pipeline reads `text`). v2
    // Clients receive the spec-compliant `{messageId}` shape —
    // `stopReason` lives on `state_update` per the v2 schema, and
    // messageId is the only required field on PromptResponse.
    let response_body = if state.protocol_version == ProtocolVersion::V2 {
        json!({
            "messageId": result.message_id,
        })
    } else {
        let stop_reason = if result.status == "completed" {
            "end_turn"
        } else {
            "max_turn_requests"
        };
        let mut body = json!({
            "stopReason": stop_reason,
            "status": result.status,
            "text": result.text,
        });
        if let Some(meta) = error_meta {
            body["error"] = meta;
        }
        body
    };
    acp::send_response(&id, response_body);
}
