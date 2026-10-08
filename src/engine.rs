//! Transport-agnostic engine — shared business logic for ACP and A2A transports.
//!
//! All session management, LLM interaction, and tool execution lives here.
//! Transport layers (stdin/stdout ACP, HTTP A2A) call these functions and
//! deliver results in their own format.

use crate::llm::{self, ImageBlock, LlmConfig, DEFAULT_IMAGE_MIME};
use crate::protocol::{AcpError, Session};
use crate::tools;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Default maximum number of tool call rounds per prompt, used when neither
/// the config file nor the environment specifies a value. The limit is a
/// backstop against degenerate models that never stop requesting tools — it
/// bounds API spend, process lifetime, and the time until the Client gets a
/// final response (surfaced as ACP `stopReason: "max_turn_requests"`).
/// Strong agentic models routinely need 6–15 rounds for real tasks, so the
/// default is deliberately generous; tune it down via
/// `LLM_MAX_TOOL_ROUNDS` / `[llm] max_tool_rounds` for small local models.
pub const DEFAULT_MAX_TOOL_ROUNDS: usize = 25;

/// Extract concatenated text from a slice of ACP/A2A content parts.
///
/// Each part is expected to be an object with `"type": "text"` and `"text": "..."`.
/// `ContentBlock::ResourceLink` parts (type == "resource_link") are converted
/// to a `[Attached resource: <uri>]` pseudo-line so the LLM has at least a
/// textual hint that a resource was attached. This satisfies ACP v1's
/// "MUST support ContentBlock::ResourceLink in session/prompt" requirement
/// without acp-bridge having to dereference the URI itself — that is the
/// responsibility of the Client per spec.
pub fn extract_text_parts(parts: &[Value]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for p in parts {
        let Some(t) = p.get("type").and_then(|v| v.as_str()) else {
            continue;
        };
        match t {
            "text" => {
                if let Some(text) = p.get("text").and_then(|v| v.as_str()) {
                    lines.push(text.to_string());
                }
            }
            "resource_link" => {
                if let Some(uri) = p.get("uri").and_then(|v| v.as_str()) {
                    let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    if name.is_empty() {
                        lines.push(format!("[Attached resource: {uri}]"));
                    } else {
                        lines.push(format!("[Attached resource: {name} ({uri})]"));
                    }
                }
            }
            _ => {}
        }
    }
    lines.join("\n")
}

/// Extract base64 image content blocks from a slice of ACP/A2A content parts.
/// Preserves the per-block `mimeType` so multimodal LLM payloads round-trip
/// correctly (PNG, WebP, GIF, etc. — not just JPEG).
pub fn extract_image_parts(parts: &[Value]) -> Vec<ImageBlock> {
    parts
        .iter()
        .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("image"))
        .filter_map(|p| {
            let data = p.get("data").and_then(|d| d.as_str())?.to_string();
            let mime_type = p
                .get("mimeType")
                .and_then(|m| m.as_str())
                .unwrap_or(DEFAULT_IMAGE_MIME)
                .to_string();
            Some(ImageBlock { data, mime_type })
        })
        .collect()
}

/// Pull user text out of a `session/prompt` `prompt` parameter, tolerating
/// the three shapes we have seen in the wild:
///
/// 1. ACP spec: `Array<ContentBlock>` — handled by `extract_text_parts`.
/// 2. A single ContentBlock object (some clients send the block directly,
///    not wrapped in a one-element array). ResourceLink is converted to a
///    pseudo-line in the same way as in the array case.
/// 3. A plain string (legacy or simplified clients that put the whole
///    prompt directly in the `prompt` field).
///
/// Returns an empty string when the shape matches none of the above. The
/// caller is responsible for rejecting an empty result with a clear error
/// rather than handing an empty message to the LLM.
pub fn extract_user_text_from_prompt(prompt: &Value) -> String {
    match prompt {
        Value::String(s) => s.clone(),
        Value::Array(arr) => extract_text_parts(arr),
        Value::Object(_) => {
            // ResourceLink-as-the-whole-prompt: turn into a single pseudo-line.
            if prompt.get("type").and_then(|v| v.as_str()) == Some("resource_link") {
                let uri = prompt.get("uri").and_then(|v| v.as_str()).unwrap_or("");
                let name = prompt.get("name").and_then(|v| v.as_str()).unwrap_or("");
                if uri.is_empty() {
                    String::new()
                } else if name.is_empty() {
                    format!("[Attached resource: {uri}]")
                } else {
                    format!("[Attached resource: {name} ({uri})]")
                }
            } else {
                prompt
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(str::to_owned)
                    .unwrap_or_default()
            }
        }
        _ => String::new(),
    }
}

/// Strip a leading `<sender_context>…</sender_context>` block from user
/// text, returning the cleaned text plus the captured inner string for
/// observability.
///
/// OpenAB and similar harnesses prepend a metadata block of this shape:
///
/// ```text
/// <sender_context>
/// {"schema":"openab.sender.v1","user_id":"…",…}
/// </sender_context>
/// <actual user message>
/// ```
///
/// Some LLMs interpret the XML tag as a directive and stall trying to
/// reconcile the wrapper with their tool-calling instructions — the
/// observed symptom is the model returning empty content with no tool
/// calls. Strip the block before forwarding so the model sees only the
/// real user text; the captured inner string is returned alongside so
/// callers can log it at debug level for traceability.
pub fn strip_sender_context(text: &str) -> (String, Option<String>) {
    const OPEN: &str = "<sender_context>";
    const CLOSE: &str = "</sender_context>";

    let Some(start) = text.find(OPEN) else {
        return (text.to_string(), None);
    };
    let after_open = start + OPEN.len();
    let Some(rel_end) = text[after_open..].find(CLOSE) else {
        return (text.to_string(), None);
    };
    let close_start = after_open + rel_end;
    let close_end = close_start + CLOSE.len();

    let inner = text[after_open..close_start].trim().to_string();
    let cleaned = format!("{}{}", &text[..start], &text[close_end..])
        .trim()
        .to_string();
    (cleaned, Some(inner))
}

/// Pull image content blocks out of a `session/prompt` `prompt` parameter
/// across the same three shapes recognized by
/// [`extract_user_text_from_prompt`]. MIME type is preserved per block.
pub fn extract_user_images_from_prompt(prompt: &Value) -> Vec<ImageBlock> {
    match prompt {
        Value::Array(arr) => extract_image_parts(arr),
        Value::Object(_) => {
            if prompt.get("type").and_then(|t| t.as_str()) == Some("image") {
                let Some(data) = prompt.get("data").and_then(|d| d.as_str()) else {
                    return Vec::new();
                };
                let mime_type = prompt
                    .get("mimeType")
                    .and_then(|m| m.as_str())
                    .unwrap_or(DEFAULT_IMAGE_MIME)
                    .to_string();
                vec![ImageBlock {
                    data: data.to_string(),
                    mime_type,
                }]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Notification — transport-agnostic events emitted during prompt processing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Notification {
    Thinking,
    /// Model reasoning ("thinking") text captured from the backend for the
    /// current round — `message.reasoning_content` (OpenAI-compatible,
    /// DeepSeek/GLM-style), `message.thinking` (Ollama native), or streamed
    /// reasoning deltas. Display-only: never appended to session history
    /// and never treated as the final answer.
    ThinkingText {
        text: String,
    },
    ToolStart {
        /// ACP v1 `toolCallId` — required for clients to pair tool_call /
        /// tool_call_update updates. The LLM assigns this id per call.
        id: String,
        /// Programmatic tool name (`read_file`, `bash`, …).
        name: String,
        /// Parsed arguments the model sent — surfaced as `rawInput` on the
        /// wire and baked into the human-readable title (issue #14).
        args: Value,
    },
    ToolDone {
        id: String,
        name: String,
        status: String,
        /// Tool result text — surfaced as `rawOutput` + a text `content`
        /// block on the completion update (issue #14). `None` for
        /// pseudo-tools with nothing to show.
        result: Option<String>,
        /// Before/after capture from file-mutating tools (issue #26) —
        /// emitted as an ACP `diff` content block so Clients can render
        /// the change. `None` for everything else.
        diff: Option<crate::tools::ToolDiff>,
    },
    TextChunk(String),
    /// A queued steer (issue #30) was injected into session history at a
    /// tool-round boundary — the model can now see it. The transport
    /// must answer the held `session/prompt` request (`request_id`) at
    /// this point: ack-on-injection means the Client's pending state
    /// clears exactly when the content reached the model.
    SteerAck {
        request_id: crate::protocol::RequestId,
        message_id: String,
    },
}

// ---------------------------------------------------------------------------
// AppState — shared state for all transports
// ---------------------------------------------------------------------------

pub struct AppState {
    /// Sessions are wrapped in `Arc` so that `AppState: Clone` works
    /// cheaply — the clone shares the same inner map. This is what
    /// lets `main::run_acp_loop` call `Arc::make_mut` to swap in the
    /// negotiated `protocol_version` after `initialize` without
    /// disturbing the live-spawned session map.
    pub sessions: Arc<RwLock<HashMap<String, Session>>>,
    pub config: LlmConfig,
    /// ACP wire-format protocol version negotiated at `initialize`.
    /// All session(s) opened by this Client inherit this version; the
    /// emit helpers (`acp::notify_*`) branch on it so v1 Clients see
    /// v1 notifications and v2 Clients see v2.
    pub protocol_version: crate::protocol::ProtocolVersion,
    /// Disk-backed session persistence. `None` when persistence is
    /// disabled (`ACP_PERSISTENCE=off`): capabilities are not
    /// advertised and `session/load` / `session/resume` keep the
    /// `no_persistence` rejection. Shared via `Arc` so `AppState` stays
    /// cheaply cloneable.
    pub store: Option<Arc<crate::session_store::SessionStore>>,
    /// This agent's identity for the session store (issue #22): the
    /// config file stem (`cometapi`), or the model name for bare
    /// invocations. Stamped onto persisted rows; compared in
    /// `ACP_SESSION_STRICT_MODELS` strict mode on restore. `None` only
    /// in tests that build an AppState without one.
    pub agent_identity: Option<String>,
    /// Issue #22 strict mode: when set (via `main` from
    /// `ACP_SESSION_STRICT_MODELS`), `session_restore` refuses
    /// sessions persisted by a different `agent_identity`.
    pub strict_models: bool,
    /// In-flight turn registry for `session/cancel` (issue #3): one
    /// entry per session with a running prompt turn. The value is the
    /// turn's generation counter plus a `watch` channel that the
    /// cancel handler flips; the turn's response task selects on it.
    /// Turns are keyed by session — a session runs at most one turn.
    pub turn_registry: Arc<std::sync::Mutex<HashMap<String, TurnEntry>>>,
    /// Mid-turn Client prompts ("steering", issue #30). When a turn is
    /// in flight, a second `session/prompt` is queued here instead of
    /// being rejected: the engine injects the content as a real `user`
    /// message at the next tool-round boundary — the model sees it
    /// in-context — and only then is the steer's JSON-RPC request
    /// answered (ack-on-injection: the Client's "pending" indicator
    /// clears exactly when the model can see the content).
    pub pending_steers: Arc<std::sync::Mutex<HashMap<String, Vec<SteerEntry>>>>,
}

/// A queued mid-turn prompt (issue #30). Carries everything needed to
/// (a) append the content to session history on injection and (b)
/// answer the held JSON-RPC request exactly once — on injection, or on
/// turn end if the turn finished without an injection point.
#[derive(Debug, Clone)]
pub struct SteerEntry {
    pub text: String,
    pub images: Vec<ImageBlock>,
    /// The held `session/prompt` JSON-RPC request id.
    pub request_id: crate::protocol::RequestId,
    /// Freshly minted messageId for the v2 ack response.
    pub message_id: String,
}

/// One in-flight turn: generation counter + cancel flag (issue #3).
pub type TurnEntry = (u64, tokio::sync::watch::Sender<bool>);

impl Clone for AppState {
    fn clone(&self) -> Self {
        Self {
            sessions: Arc::clone(&self.sessions),
            config: self.config.clone(),
            protocol_version: self.protocol_version,
            store: self.store.clone(),
            agent_identity: self.agent_identity.clone(),
            strict_models: self.strict_models,
            turn_registry: Arc::clone(&self.turn_registry),
            pending_steers: Arc::clone(&self.pending_steers),
        }
    }
}

impl AppState {
    /// Queue a mid-turn prompt for injection at the next tool-round
    /// boundary (issue #30). Returns the queue length after the push.
    pub fn queue_steer(&self, session_id: &str, entry: SteerEntry) -> usize {
        let mut map = self.pending_steers.lock().expect("pending_steers lock");
        let queue = map.entry(session_id.to_string()).or_default();
        queue.push(entry);
        queue.len()
    }

    /// Drain all queued steers for a session (issue #30). Called by the
    /// engine at tool-round boundaries and by the transport at turn end
    /// (to acknowledge anything the turn finished without injecting —
    /// nothing queued may be silently dropped).
    pub fn drain_steers(&self, session_id: &str) -> Vec<SteerEntry> {
        let mut map = self.pending_steers.lock().expect("pending_steers lock");
        map.remove(session_id).unwrap_or_default()
    }
}

impl AppState {
    pub fn new(config: LlmConfig) -> Arc<Self> {
        Self::with_store(config, None, None)
    }

    /// Real constructor. `store` is `Some` when persistence is enabled;
    /// `agent_identity` is the config stem (issue #22) — stamped onto
    /// persisted sessions and compared in strict mode.
    pub fn with_store(
        config: LlmConfig,
        store: Option<Arc<crate::session_store::SessionStore>>,
        agent_identity: Option<String>,
    ) -> Arc<Self> {
        Arc::new(Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            // Default to V1 for safety. `main::run_acp_loop` overwrites
            // this with whatever the Client negotiated during
            // `initialize`. We pick V1 here so unit tests that build an
            // AppState directly get v1 wire format without ceremony.
            protocol_version: crate::protocol::ProtocolVersion::V1,
            store,
            agent_identity,
            strict_models: false,
            turn_registry: Arc::new(std::sync::Mutex::new(HashMap::new())),
            pending_steers: Arc::new(std::sync::Mutex::new(HashMap::new())),
        })
    }

    fn sessions_write(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Session>> {
        match self.sessions.write() {
            Ok(s) => s,
            Err(p) => {
                warn!("Session lock poisoned, recovering");
                p.into_inner()
            }
        }
    }

    fn sessions_read(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Session>> {
        match self.sessions.read() {
            Ok(s) => s,
            Err(p) => {
                warn!("Session lock poisoned, recovering");
                p.into_inner()
            }
        }
    }

    /// Evict sessions that have been idle longer than the timeout.
    pub fn evict_idle_sessions(&self, timeout_secs: u64) {
        if timeout_secs == 0 {
            return;
        }
        let timeout = Duration::from_secs(timeout_secs);
        let mut sessions = self.sessions_write();
        let before = sessions.len();
        sessions.retain(|_id, session| session.last_active.elapsed() < timeout);
        let evicted = before - sessions.len();
        if evicted > 0 {
            info!(evicted, remaining = sessions.len(), "Evicted idle sessions");
        }
    }

    /// Clean up all sessions. Returns the number of sessions cleaned.
    pub fn cleanup(&self) -> usize {
        let mut s = self.sessions_write();
        let n = s.len();
        s.clear();
        n
    }
}

// ---------------------------------------------------------------------------
// Engine functions — called by both ACP and A2A transports
// ---------------------------------------------------------------------------

/// Handle `initialize` — returns agent info.
///
/// ACP v1 spec treats capability flags as the only signal a Client has for
/// whether to send images / audio / embedded resources. Per-spec the default
/// for all three is `false`, and acp-bridge historically advertised
/// `image: true` unconditionally — which caused Clients (Meuxe, ACP UI, …)
/// to forward image attachments to local backends that had no vision model
/// loaded, surfacing upstream as confusing "agent didn't respond" errors.
///
/// We now advertise `image` only when the operator opts in via the
/// `LLM_SUPPORTS_IMAGE` env var or the equivalent field in the config file.
/// `audio` and `embeddedContext` remain `false` — acp-bridge does not yet
/// process them.
///
/// The `protocol_version` parameter controls which wire-shape the
/// response uses:
/// - `ProtocolVersion::V1` → `agentCapabilities` + `agentInfo` shape
///   (every existing Client speaks this today)
/// - `ProtocolVersion::V2` → unified `capabilities` + `info` shape, with
///   `promptCapabilities.image` expressed as `{}` (capability marker)
///   rather than `true`
pub fn initialize(
    config: &LlmConfig,
    protocol_version: crate::protocol::ProtocolVersion,
    persistence_enabled: bool,
) -> Value {
    info!(
        model = %config.model,
        base_url = %config.base_url,
        protocol_version = %protocol_version,
        "Initialize"
    );

    // v2 marks capability support with the present-of-an-object marker
    // (`{}` = "supported") rather than `true`. We still want to
    // distinguish image support from audio / embeddedContext, so:
    //   - image enabled   → `{"image": {}}` (present, supported)
    //   - image disabled  → omit `image` entirely (not advertised)
    // audio / embeddedContext stay omitted regardless.
    let prompt_capabilities_v2 = if config.prompt_supports_image {
        json!({ "image": {} })
    } else {
        json!({})
    };

    match protocol_version {
        crate::protocol::ProtocolVersion::V2 => json!({
            "protocolVersion": 2,
            "info": {
                "name": format!("acp-bridge ({})", config.model),
                "title": "acp-bridge",
                "version": env!("CARGO_PKG_VERSION")
            },
            "capabilities": {
                "loadSession": true,
                "session": {
                    "prompt": prompt_capabilities_v2,
                    "resume": {},
                }
            },
            "authMethods": []
        }),
        _ => {
            // v1 (default)
            let prompt_capabilities_v1 = if config.prompt_supports_image {
                json!({
                    "image": true,
                    "audio": false,
                    "embeddedContext": false
                })
            } else {
                json!({
                    "image": false,
                    "audio": false,
                    "embeddedContext": false
                })
            };
            let mut agent_capabilities = json!({
                "loadSession": persistence_enabled,
                "promptCapabilities": prompt_capabilities_v1
            });
            // Spec: omitted capabilities are treated as UNSUPPORTED — with
            // persistence off, `loadSession` must be absent so Clients never
            // attempt `session/load` (issue #17).
            if !persistence_enabled {
                agent_capabilities
                    .as_object_mut()
                    .expect("object")
                    .remove("loadSession");
            }
            json!({
                "protocolVersion": 1,
                "agentInfo": {
                    "name": format!("acp-bridge ({})", config.model),
                    "version": env!("CARGO_PKG_VERSION")
                },
                "agentCapabilities": agent_capabilities,
                "authMethods": []
            })
        }
    }
}

/// Handle `session/new` — creates a new session, returns session ID.
///
/// `protocol_version` is the version negotiated at `initialize` for the
/// Client calling us. All subsequent notifications in this session use
/// the v1 or v2 wire shape accordingly.
pub fn session_new(
    state: &AppState,
    cwd: &str,
    protocol_version: crate::protocol::ProtocolVersion,
) -> Result<String, AcpError> {
    // Enforce max_sessions limit
    if state.config.max_sessions > 0 {
        let count = state.sessions_read().len();
        if count >= state.config.max_sessions {
            return Err(AcpError::SessionLimitReached {
                max: state.config.max_sessions,
            });
        }
    }

    // Sanitize cwd
    let cwd: String = cwd
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '/' | '-' | '_' | '.' | ' ' | '~'))
        .collect();

    let session_id = Uuid::new_v4().to_string();

    let system_prompt = state.config.system_prompt.clone().unwrap_or_else(|| {
        std::env::var("LLM_SYSTEM_PROMPT").unwrap_or_else(|_| {
            format!("You are a helpful coding assistant. The user's working directory is: {cwd}")
        })
    });

    let mut session = Session::new(
        json!({"role": "system", "content": system_prompt}),
        PathBuf::from(&cwd),
        protocol_version,
    );
    // Issue #13: if the Client never sends `session/set_config_option`,
    // the session uses the config default — which today means the
    // `[llm.request_overrides]` reasoning_effort when present.
    if let Some(effort) = state
        .config
        .request_overrides
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
    {
        if state.config.thought_levels.iter().any(|l| l == effort) {
            session.thought_level = Some(effort.to_string());
        }
    }
    state.sessions_write().insert(session_id.clone(), session);

    // Persist the (empty) session immediately: a Client that restores
    // after a restart must be able to load a session even when the
    // restart happened before the first prompt (issue #17).
    persist_session_snapshot(state, &session_id);

    info!(session_id = %session_id, max_history = state.config.max_history_turns, "New session");

    // NOTE: We return `Ok(session_id)` from here BEFORE emitting the
    // post-session-creation notifications below. Callers (e.g.
    // `main::run_acp_loop`) send the JSON-RPC response immediately on
    // receiving the session id, then emit the notifications. This ordering
    // matches what other ACP agents do (OpenClaw, OpenCode) — the Client
    // learns the new sessionId first, then receives the post-session
    // notifications (commands, info update) bound to that id.
    Ok(session_id)
}

/// Post-creation notifications to send after `session/new` returns.
/// Exposed so `main::run_acp_loop` can emit them with the right wire
/// framing after the response is on the wire. The `protocol_version`
/// argument controls which wire shape (v1 vs v2) the dispatchers emit.
/// The `thought_level` config option advertised in `session/new`
/// (issue #13, bb's `configOptions` extension). `None` = feature off
/// (empty `thought_levels`): nothing is advertised and the wire is
/// byte-identical to before.
pub fn thought_level_config_option(state: &AppState, current: Option<&str>) -> Option<Value> {
    if state.config.thought_levels.is_empty() {
        return None;
    }
    // The current value defaults to the first advertised level (what
    // bb renders as selected before any `set_config_option`).
    // `current` is Option<&str> already; the default is the first
    // advertised level.
    let current = current.unwrap_or(&state.config.thought_levels[0]);
    Some(json!({
        "id": "thought_level",
        "name": "Thought level",
        "category": "thought_level",
        "type": "select",
        "currentValue": current,
        "options": state
            .config
            .thought_levels
            .iter()
            .map(|level| {
                json!({
                    "value": level,
                    // bb's picker maps these onto effort descriptions
                    // ("max" → "Maximum reasoning effort", …); unknown
                    // levels still render by their raw value.
                    "name": level,
                })
            })
            .collect::<Vec<_>>(),
    }))
}

/// The `model` config option (issue #13): bb's UI builds its model
/// list from a `category: "model"` option and then PROBES reasoning
/// support by sending `session/set_config_option(configId: "model",
/// value: <model>)` and reading the `thought_level` option back from
/// the response. Without this option bb takes an early return in its
/// discovery path and never builds the reasoning-effort picker —
/// advertising `thought_level` alone is not enough. acp-bridge offers
/// exactly one model: the configured one.
fn model_config_option(state: &AppState, current: Option<&str>) -> Value {
    // Issue #40: advertise the real backend list (startup probe);
    // configured model first when present. Empty list = probe failed
    // → single configured model, as before.
    let mut models = state.config.available_models.clone();
    if !models.iter().any(|m| m == &state.config.model) {
        models.insert(0, state.config.model.clone());
    }
    let current = current.unwrap_or(&state.config.model);
    json!({
        "id": "model",
        "name": "Model",
        "category": "model",
        "type": "select",
        "currentValue": current,
        "options": models
            .iter()
            .map(|m| json!({ "value": m, "name": m }))
            .collect::<Vec<_>>(),
    })
}

/// Both options bb's picker pipeline needs, in a fixed order (model
/// first, thought level second). Called only when the feature is on.
fn config_options_body(
    state: &AppState,
    current_model: Option<&str>,
    current_level: Option<&str>,
) -> Value {
    // The model option is always advertised (issue #40): the list
    // comes from the startup probe, so model switching works with
    // zero configuration. The thought_level option (issue #13) only
    // when levels are configured.
    let mut options = vec![model_config_option(state, current_model)];
    if let Some(thought) = thought_level_config_option(state, current_level) {
        options.push(thought);
    }
    json!({ "configOptions": options })
}

/// Build the `session/new` success response (issue #13): `sessionId`
/// plus — when the thought-level feature is configured — the
/// `configOptions` array bb's picker reads (model option + thought
/// level). Without the feature the response is byte-identical to the
/// historical `{"sessionId"}` shape.
pub fn session_new_response(state: &AppState, session_id: &str) -> Value {
    let (current_model, current_level) = state
        .sessions_read()
        .get(session_id)
        .map(|s| (s.model_override.clone(), s.thought_level.clone()))
        .unwrap_or((None, None));
    let mut body = json!({ "sessionId": session_id });
    body["configOptions"] =
        config_options_body(state, current_model.as_deref(), current_level.as_deref())
            ["configOptions"]
            .clone();
    body
}

/// Handle `session/set_config_option` (issue #13): validate and apply
/// the request, returning the updated `configOptions` array (bb reads
/// it back — including from the `model` probe responses). Errors for
/// unknown sessions / invalid values.
pub fn session_set_config_option(
    state: &AppState,
    session_id: &str,
    config_id: &str,
    value: &Value,
) -> Result<Value, AcpError> {
    match config_id {
        // The Client's model picker (and bb's reasoning probe) select
        // the model through configId "model". Accept anything the
        // startup probe advertised (issue #40): store a per-session
        // override applied via the existing model_override plumbing.
        "model" => {
            let requested = value.as_str().ok_or_else(|| AcpError::InvalidParam {
                field: "value: expected a string model id".into(),
            })?;
            let mut offered = state.config.available_models.clone();
            if !offered.iter().any(|m| m == &state.config.model) {
                offered.insert(0, state.config.model.clone());
            }
            if !offered.iter().any(|m| m == requested) {
                return Err(AcpError::InvalidParam {
                    field: format!(
                        "value: '{requested}' is not offered; available: {}",
                        offered.join(", ")
                    ),
                });
            }
            let mut sessions = state.sessions_write();
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| AcpError::UnknownSession {
                    session_id: session_id.to_string(),
                })?;
            session.model_override = Some(requested.to_string());
        }
        "thought_level" => {
            if state.config.thought_levels.is_empty() {
                return Err(AcpError::InvalidParam {
                    field: "configId: no thought levels configured".into(),
                });
            }
            let Some(level) = value.as_str() else {
                return Err(AcpError::InvalidParam {
                    field: "value: expected a string thought level".into(),
                });
            };
            if !state.config.thought_levels.iter().any(|l| l == level) {
                return Err(AcpError::InvalidParam {
                    field: format!(
                        "value: '{level}' is not one of the advertised thought levels: {}",
                        state.config.thought_levels.join(", ")
                    ),
                });
            }
            let mut sessions = state.sessions_write();
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| AcpError::UnknownSession {
                    session_id: session_id.to_string(),
                })?;
            session.thought_level = Some(level.to_string());
        }
        other => {
            return Err(AcpError::InvalidParam {
                field: format!("configId: unknown config option '{other}'"),
            });
        }
    }

    let (current_model, current_level) = state
        .sessions_read()
        .get(session_id)
        .map(|s| (s.model_override.clone(), s.thought_level.clone()))
        .unwrap_or((None, None));
    Ok(config_options_body(
        state,
        current_model.as_deref(),
        current_level.as_deref(),
    ))
}

pub fn session_new_post_create_notifications(
    session_id: &str,
    cwd: &str,
    protocol_version: crate::protocol::ProtocolVersion,
) {
    // Advertise the slash commands this agent recognises. Clients use this
    // to populate the "/…" shortcut menu. The list is intentionally small:
    // commands are shortcuts, not advertised features. See `tools.rs` for
    // the tool surface that backs the longer-tail capabilities.
    crate::acp::notify_available_commands_for(
        protocol_version,
        session_id,
        &[
            crate::acp::AvailableCommand::new(
                "read",
                "Read a file (alias for read_file tool)",
                Some::<&str>("path"),
            ),
            crate::acp::AvailableCommand::new(
                "ls",
                "List directory contents (alias for list_dir tool)",
                Some::<&str>("path"),
            ),
            crate::acp::AvailableCommand::new(
                "search",
                "Search code (alias for search tool)",
                Some::<&str>("pattern"),
            ),
            crate::acp::AvailableCommand::new(
                "edit",
                "Patch a file via write_file tool",
                Some::<&str>("path"),
            ),
            crate::acp::AvailableCommand::new(
                "shell",
                "Run a shell command via the shell tool",
                Some::<&str>("command"),
            ),
        ],
    );

    // Initial session_info_update — title defaults to the cwd basename so
    // the Client has something to show in its session list before any
    // prompt runs.
    let title = std::path::Path::new(cwd)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(cwd);
    crate::acp::notify_session_info_for(protocol_version, session_id, title, None);
}

/// Handle `session/prompt` — runs the LLM with tool loop.
///
/// Sends `Notification` events through `notify_tx` as they happen (for ACP streaming).
/// Returns the final status ("completed" or "failed") and accumulated text.
///
/// `message_id` is the opaque identifier acp-bridge mints for this
/// prompt. It is required on the v2 `PromptResponse` and used by v2
/// Clients to correlate the response with `user_message` and
/// `agent_message` session updates. v1 Clients ignore it.
/// Steering (issue #30): drain queued mid-turn prompts and append them
/// to session history as real `user` messages. Called at every
/// tool-round boundary AND at the final-response guard — the model must
/// see the steer before the turn may end. Each injected steer's held
/// JSON-RPC request is acked here (ack-on-injection, pinned decision on
/// #30): the Client's pending state clears exactly when the content is
/// visible to the model. Returns the number of steers injected.
fn inject_steers(
    state: &Arc<AppState>,
    session_id: &str,
    round: usize,
    notify: &impl Fn(Notification),
) -> usize {
    let steers = state.drain_steers(session_id);
    let count = steers.len();
    if count == 0 {
        return 0;
    }
    {
        let mut sessions = state.sessions_write();
        if let Some(session) = sessions.get_mut(session_id) {
            let backend = state.config.backend();
            for steer in &steers {
                session
                    .messages
                    .push(backend.format_user_message(&steer.text, &steer.images));
            }
        }
    }
    persist_session_snapshot(state, session_id);
    for steer in steers {
        info!(session_id, round, "Injecting queued steer");
        notify(Notification::SteerAck {
            request_id: steer.request_id,
            message_id: steer.message_id,
        });
    }
    count
}

pub async fn session_prompt(
    state: &Arc<AppState>,
    session_id: &str,
    user_text: &str,
    user_images: &[ImageBlock],
    notify_tx: Option<mpsc::UnboundedSender<Notification>>,
    message_id: &str,
) -> PromptResult {
    let notify = |n: Notification| {
        if let Some(tx) = &notify_tx {
            let _ = tx.send(n);
        }
    };

    // Context compaction (issue #25): before appending this turn's
    // user message, check whether the previous round's reported prompt
    // tokens crossed the configured fraction of the context window. If
    // so, summarize everything older than the recent tail into a
    // rolling note — the scissors (trim_history) become the fallback,
    // not the primary defense. Failures degrade to the plain trim.
    let compaction_split = {
        let sessions = state.sessions_read();
        let session = sessions.get(session_id);
        match session {
            Some(s) => should_compact(&state.config, s.last_used_tokens.map(|u| (u, 0)), s),
            None => None,
        }
    };
    if let Some((used, threshold)) = compaction_split {
        // Cut point: keep the recent half of history (same tail size
        // trim_history would keep) so the model retains immediate
        // context; everything before it gets summarized.
        let split = {
            let sessions = state.sessions_read();
            sessions
                .get(session_id)
                .and_then(|s| s.compaction_split(state.config.max_history_turns.max(4)))
                .unwrap_or(1)
        };
        match compact_session(state, session_id, split).await {
            Ok(summary) => {
                info!(
                    session_id,
                    used_tokens = used,
                    threshold_tokens = threshold,
                    summary_chars = summary.len(),
                    "Context compacted: older rounds summarized"
                );
                notify(Notification::ThinkingText {
                    text: format!(
                        "[acp-bridge: context compacted — older rounds summarized \
({} chars) to stay within the model window]",
                        summary.len()
                    ),
                });
            }
            Err(e) => {
                warn!(error = %e, session_id, "Compaction failed; falling back to trim");
            }
        }
    }

    // Add user message, touch session, and trim history
    {
        let mut sessions = state.sessions_write();
        let session = match sessions.get_mut(session_id) {
            Some(s) => s,
            None => {
                return PromptResult {
                    status: "failed".into(),
                    text: format!("Unknown session: {session_id}"),
                    error: Some(AcpError::UnknownSession {
                        session_id: session_id.into(),
                    }),
                    usage: UsageReport {
                        used: 0,
                        size: state.config.context_size,
                    },
                    error_class: None,
                    error_retryable: false,
                    // Caller never wired a prompt through this branch —
                    // the session id doesn't exist — so we emit an
                    // empty messageId. The Client will see a JSON-RPC
                    // error from the outer send_error path anyway.
                    message_id: String::new(),
                };
            }
        };
        session.touch();
        session.messages.push(
            state
                .config
                .backend()
                .format_user_message(user_text, user_images),
        );

        // Scissors only when compaction did NOT just run (or is
        // disabled): compacted sessions already lost their old tail,
        // so trimming again would cut fresh context.
        if state.config.max_history_turns > 0 && compaction_split.is_none() {
            let before = session.messages.len();
            session.trim_history(state.config.max_history_turns);
            let after = session.messages.len();
            if before != after {
                debug!(before, after, "Trimmed conversation history");
            }
        }
    }

    // NOTE: no turn-level synthetic tool_call here — `tool_call`
    // notifications represent model-invoked tools, and the real per-round
    // tool calls (with specifics, since #14) plus the streamed chunks
    // already tell the Client what is happening. The old synthetic
    // `llm_chat` wrapper rendered as bare noise in Clients.

    let mut had_error = false;
    let mut got_final_response = false;
    let mut final_text = String::new();
    // Backend-reported usage for the most recent round (issue #4) —
    // `None` until a backend actually reports; the final usage report
    // prefers this over the chars/4 estimate.
    let mut turn_usage: Option<(u64, u64)> = None;
    let mut last_error_class: Option<crate::llm::LlmErrorKind> = None;
    let tool_defs = tools::tool_definitions();

    // Per-turn tool-call round budget. `0` disables the cap entirely —
    // symmetric with `max_history_turns` / `max_sessions`. Only for
    // trusted setups: a degenerate model that never stops requesting
    // tools will then loop (and spend) until the client disconnects.
    let max_tool_rounds = state.config.max_tool_rounds;

    // Tool call loop. Structured as a `loop` (not a ranged `for`) so the
    // budget can distinguish "capped at N rounds" from `0` = unlimited,
    // symmetric with `max_history_turns` / `max_sessions`.
    let mut round = 0usize;
    loop {
        if max_tool_rounds > 0 && round >= max_tool_rounds {
            break;
        }
        // Steering (issue #30): drain queued mid-turn prompts at the
        // round boundary and append them as real `user` messages, so
        // the model addresses them in-context this round. Each injected
        // steer's held JSON-RPC request is acked exactly here
        // (ack-on-injection): the Client's pending state clears the
        // moment the content is visible to the model. NOTE: this runs
        // BEFORE the `messages` snapshot below, so an injected steer is
        // part of THIS round's LLM request, not the next one.
        inject_steers(state, session_id, round, &notify);

        let (messages, working_dir) = {
            let sessions = state.sessions_read();
            match sessions.get(session_id) {
                Some(s) => (s.messages.clone(), s.working_dir.clone()),
                None => {
                    warn!(session_id, "Session removed during prompt processing");
                    had_error = true;
                    final_text = format!("Session {session_id} was removed during processing");
                    break;
                }
            }
        };
        let backend = state.config.backend();

        // Open + drain the round's stream, with one bounded silent retry
        // when the round fails before notifying anything (issue #15).
        // Once a chunk has reached the Client, fail fast instead — a
        // retry would re-generate and duplicate already-visible output.
        let mut notified_any = false;
        let mut content = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut round_error: Option<crate::llm::LlmError> = None;
        // Session-scoped overrides ride every round's request body:
        // thought level (issue #13) and the model pick (issue #40) —
        // both chosen by the Client through set_config_option.
        let (model_override, thought_effort) = {
            let sessions = state.sessions_read();
            sessions
                .get(session_id)
                .map(|s| (s.model_override.clone(), s.thought_level.clone()))
                .unwrap_or((None, None))
        };
        // Live-verifiable record of the per-session overrides actually
        // riding this round (issues #40/#43): at debug level, so the
        // picker → request chain can be confirmed with
        // RUST_LOG=acp_bridge=debug without touching the gateway.
        debug!(
            session_id = %session_id,
            model = %model_override.as_ref().unwrap_or(&state.config.model),
            reasoning_effort = ?thought_effort,
            "round request overrides"
        );
        for attempt in 0..2 {
            // The receiver is loop-local: once the round's stream ends
            // (normally or by error) it has no further use.
            let (mut rx, round_err) = match llm::chat_streamed(
                &state.config,
                &messages,
                model_override.as_deref(),
                Some(&tool_defs),
                thought_effort.as_deref(),
            )
            .await
            {
                Ok(rx) => (Some(rx), None),
                Err(e) => (None, Some(e)),
            };
            round_error = round_err;
            if round_error.is_some() {
                break;
            }

            // Drain this round's stream. Parsers guarantee Thinking
            // before Content before ToolCall and exactly one terminal
            // event (Error or Done). Reasoning and text are notified as
            // they arrive — the point of the streaming loop (issue #11)
            // — while the accumulated forms feed the session history and
            // the turn result.
            if let Some(rx) = rx.as_mut() {
                while let Some(chunk) = rx.recv().await {
                    match chunk {
                        llm::StreamChunk::Thinking(t) => {
                            notified_any = true;
                            notify(Notification::ThinkingText { text: t });
                        }
                        llm::StreamChunk::Content(t) => {
                            notified_any = true;
                            content.push_str(&t);
                            notify(Notification::TextChunk(t));
                        }
                        llm::StreamChunk::ToolCall(call) => {
                            notified_any = true;
                            tool_calls.push(call)
                        }
                        llm::StreamChunk::Error(message, kind) => {
                            round_error = Some(crate::llm::LlmError {
                                kind,
                                message,
                                status: None,
                            });
                        }
                        llm::StreamChunk::Usage {
                            prompt_tokens,
                            completion_tokens,
                        } => {
                            // Issue #4: backend-reported usage. Last round
                            // wins — `prompt_tokens` is the cumulative
                            // context for THIS round's request, so it
                            // already includes everything before it.
                            turn_usage = Some((prompt_tokens, completion_tokens));
                        }
                        llm::StreamChunk::Done => break,
                    }
                }
            }

            if let Some(err) = round_error.take() {
                if attempt == 0 && !notified_any {
                    // Nothing reached the Client — a single silent retry
                    // is invisible and strictly better than killing a
                    // long turn (issue #15). Reset accumulators.
                    warn!(
                        attempt,
                        kind = err.kind.as_str(),
                        "Round failed before notifying anything; retrying once"
                    );
                    notified_any = false;
                    content.clear();
                    tool_calls.clear();
                    continue;
                }
                round_error = Some(err);
                break;
            }
            break;
        }

        if let Some(e) = round_error {
            // Classify the LLM error so the response can carry a
            // structured `data.category` field that Clients can
            // switch on without parsing prose. See
            // `LlmErrorKind::as_str()` for the stable category names.
            // Covers both setup-time failures (request rejected before
            // any chunk) and mid-stream failures — the latter fail the
            // turn instead of retrying, because a retry would re-generate
            // chunks the client already saw (issue #11 fallback policy).
            let kind = e.kind.clone();
            let retryable = e.kind.is_retryable();
            let err_msg = format!(
                "\n\n**Error ({}):** {}\n{}",
                e.kind.as_str(),
                e.message,
                if retryable {
                    "_Hint: this is a transient error; the same prompt may succeed on retry._"
                } else {
                    "_Hint: this error is not retryable; check the model name, base URL, or prompt shape._"
                }
            );
            notify(Notification::TextChunk(err_msg.clone()));
            final_text = err_msg;
            had_error = true;
            last_error_class = Some(kind);
            error!(
                kind = e.kind.as_str(),
                status = ?e.status,
                retryable,
                detail = %e.message,
                "LLM communication failed"
            );
            break;
        }

        if tool_calls.is_empty() {
            // Steering (issue #30): steers that arrived while the final
            // round was streaming would otherwise be stranded — the turn
            // ends here and the round-boundary drain never runs again.
            // Inject them and loop for one more round; the model must
            // see the steer before the turn may end. (Next round the
            // queue is empty and this guard passes through.)
            if inject_steers(state, session_id, round, &notify) > 0 {
                continue;
            }
            got_final_response = true;
            if !content.is_empty() {
                final_text = content.clone();
                {
                    let mut sessions = state.sessions_write();
                    if let Some(session) = sessions.get_mut(session_id) {
                        session
                            .messages
                            .push(json!({"role": "assistant", "content": &content}));
                    }
                }
                // The text was already notified delta-by-delta above; no
                // additional aggregate chunk here (the client would see
                // the answer twice).
            }
            persist_session_snapshot(state, session_id);
            break;
        }

        // Execute tool calls
        info!(round, count = tool_calls.len(), "Executing tool calls");

        {
            let mut sessions = state.sessions_write();
            if let Some(session) = sessions.get_mut(session_id) {
                // No raw response exists in the streaming path — assemble
                // the assistant message from the accumulated round data in
                // the shape each backend expects. OpenAI-compatible
                // upstreams carry `content: null` on tool-call turns.
                let content_value = if content.is_empty() {
                    Value::Null
                } else {
                    json!(content)
                };
                let synthetic = json!({
                    "choices": [{"message": {
                        "role": "assistant",
                        "content": content_value,
                        "tool_calls": tool_calls,
                    }}]
                });
                let assistant_msg =
                    backend.format_assistant_message(&content, &tool_calls, &synthetic);
                session.messages.push(assistant_msg);
            }
        }

        for tc in &tool_calls {
            let func = &tc["function"];
            let name = func["name"].as_str().unwrap_or("unknown");
            // `function.arguments` can be either a JSON-encoded
            // string (OpenAI-compatible backends) or a JSON
            // object (Ollama native /api/chat). Handle both —
            // the previous `as_str().unwrap_or("{}")` silently
            // dropped every Ollama native tool call's actual
            // arguments (see review §"既有 bug 未修").
            let args: Value = match func.get("arguments") {
                Some(Value::String(s)) => {
                    serde_json::from_str(s).unwrap_or_else(|_| Value::Object(Default::default()))
                }
                Some(Value::Object(_)) | Some(Value::Array(_)) => func["arguments"].clone(),
                Some(_) | None => Value::Object(Default::default()),
            };
            let tool_call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");

            notify(Notification::ToolStart {
                id: tool_call_id.to_string(),
                name: name.into(),
                args: args.clone(),
            });
            let outcome = tools::execute_tool(&working_dir, name, &args);
            // Tool failures are signaled by the "Error" prefix of the
            // result text (see tools.rs) — surface them as a real
            // `failed` status instead of unconditional `completed`
            // (issue #14).
            let status = if outcome.text.starts_with("Error") {
                "failed"
            } else {
                "completed"
            };
            notify(Notification::ToolDone {
                id: tool_call_id.to_string(),
                name: name.into(),
                status: status.into(),
                result: Some(outcome.text.clone()),
                diff: outcome.diff,
            });

            debug!(
                tool = name,
                result_len = outcome.text.len(),
                "Tool executed"
            );

            {
                let mut sessions = state.sessions_write();
                if let Some(session) = sessions.get_mut(session_id) {
                    session
                        .messages
                        .push(backend.format_tool_result(tool_call_id, &outcome.text));
                }
            }

            // Persist after each completed tool round (issue #17): a
            // crash or SIGTERM mid-turn keeps every completed round; only
            // the in-flight round is lost. Off the notification path —
            // the client-visible stream never waits for disk.
            persist_session_snapshot(state, session_id);
        }

        round += 1;
    }

    // The loop can also exit by exhausting the per-turn round budget while
    // the model keeps requesting tools. Without this the turn would be
    // reported as a successful "completed" with empty text, silently
    // swallowing the fact that the model never produced a final answer.
    // (When `max_tool_rounds` is 0 the loop has no cap and can only exit
    // via a final response or an LLM error.)
    if !had_error && !got_final_response {
        warn!(
            max_rounds = max_tool_rounds,
            "Tool-call loop hit round limit without a final response"
        );
        let msg = format!(
            "\n\n**Error:** reached the tool-call limit ({max_tool_rounds} rounds) without a final answer\n_Hint: raise `LLM_MAX_TOOL_ROUNDS` (env) or `[llm] max_tool_rounds` (config) if your model needs more rounds._\n"
        );
        notify(Notification::TextChunk(msg.clone()));
        final_text = msg;
        had_error = true;
    }

    let status = if had_error { "failed" } else { "completed" };

    // Context utilization for `usage_update` (issue #4): prefer the
    // backend's own reported token counts from the most recent round —
    // `prompt_tokens` is the real context occupancy of that request and
    // `completion_tokens` its generation. Backends that never report
    // usage fall back to the chars/4 estimate over session history
    // (deliberately approximate; clients use this for progress bars,
    // not cost attribution).
    let used_tokens = match turn_usage {
        Some((prompt, completion)) => {
            debug!(prompt, completion, "Using backend-reported usage");
            prompt + completion
        }
        None => {
            let sessions = state.sessions_read();
            sessions
                .get(session_id)
                .map(|s| estimate_tokens(&s.messages))
                .unwrap_or(0)
        }
    };
    // Record for the next turn's compaction trigger (issue #25).
    if let Some(sessions) = state.sessions.write().ok().as_mut() {
        if let Some(session) = sessions.get_mut(session_id) {
            session.last_used_tokens = Some(used_tokens as u64);
        }
    }

    PromptResult {
        status: status.into(),
        text: final_text,
        error: None,
        usage: UsageReport {
            used: used_tokens,
            size: state.config.context_size,
        },
        error_class: last_error_class.clone(),
        error_retryable: last_error_class.as_ref().is_some_and(|k| k.is_retryable()),
        message_id: message_id.to_string(),
    }
}

/// Rough token estimate from a list of chat messages. Sums the textual
/// content of every message and divides by 4 chars/token. We do NOT try
/// to be precise — local backends do not stream stable per-turn token
/// counts, and clients use this only for progress visualisation.
pub fn estimate_tokens(messages: &[Value]) -> u64 {
    let total_chars: usize = messages
        .iter()
        .map(|m| {
            // `content` may be a string, an array of ContentBlocks, or
            // null; extract any text we can find.
            match m.get("content") {
                Some(Value::String(s)) => s.len(),
                Some(Value::Array(arr)) => arr
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|v| v.as_str()))
                    .map(|s| s.len())
                    .sum(),
                _ => 0,
            }
        })
        .sum();
    ((total_chars / 4) as u64).max(1)
}

/// Handle `session/end` — removes a session.
pub fn session_end(state: &AppState, session_id: &str) -> Result<(), AcpError> {
    let removed = state.sessions_write().remove(session_id).is_some();
    if removed {
        // The Client explicitly ended the session — drop the persisted
        // snapshot too, otherwise a later session/load would resurrect
        // it (issue #17).
        if let Some(store) = &state.store {
            if let Err(e) = store.delete(session_id) {
                warn!(error = %e, session_id, "Failed to delete persisted session");
            }
        }
        info!(session_id = %session_id, "Session ended");
        Ok(())
    } else {
        Err(AcpError::UnknownSession {
            session_id: session_id.into(),
        })
    }
}

/// Snapshot a live session into the persistence store (issue #17).
///
/// Synchronous by design: a WAL upsert of one small row is sub-
/// millisecond, and keeping saves inline between rounds guarantees
/// ordering — concurrent saves could persist a stale snapshot over a
/// newer one. No-op when persistence is disabled or the session is
/// already gone.
pub fn persist_session_snapshot(state: &AppState, session_id: &str) {
    let Some(store) = &state.store else {
        return;
    };
    let record = {
        let sessions = state.sessions_read();
        let Some(session) = sessions.get(session_id) else {
            return;
        };
        crate::session_store::SessionRecord {
            session_id: session_id.to_string(),
            cwd: session.working_dir.display().to_string(),
            protocol_version: session.protocol_version.as_u16(),
            title: None,
            created_at: now_ms(),
            updated_at: now_ms(),
            messages: session.messages.clone(),
            agent: state.agent_identity.clone(),
        }
    };
    if let Err(e) = store.save(&record) {
        warn!(error = %e, session_id, "Failed to persist session snapshot");
    }
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

// -- Context compaction (issue #25) -----------------------------------------

/// Effective compaction trigger: `Some(fraction)` when enabled, `None`
/// when disabled (threshold unset → default 0.75; explicit 0 → off;
/// out-of-range values already filtered at config load).
fn compaction_trigger_fraction(config: &LlmConfig) -> Option<f64> {
    const DEFAULT_TRIGGER: f64 = 0.75;
    match config.compaction_threshold {
        Some(0.0) => None,
        Some(t) => Some(t),
        None => Some(DEFAULT_TRIGGER),
    }
}

/// Decide whether compaction should run, from the most recent round's
/// backend-reported prompt tokens (the real context occupancy) or the
/// chars/4 estimate as fallback (same precedence as `usage_update`).
fn should_compact(
    config: &LlmConfig,
    turn_usage: Option<(u64, u64)>,
    session: &Session,
) -> Option<(u64, u64)> {
    let trigger = compaction_trigger_fraction(config)?;
    if config.context_size == 0 {
        return None;
    }
    let threshold = (config.context_size as f64 * trigger) as u64;
    let used = match turn_usage {
        Some((prompt, _completion)) => prompt,
        None => estimate_tokens(&session.messages),
    };
    (used >= threshold).then_some((used, threshold))
}

/// Build the summarization request: a strict single-message chat call
/// over the OLDER half of history (everything before `split`), with a
/// system prompt that pins the output contract. No tools, no history —
/// one shot.
fn summarization_messages(older: &[Value]) -> Vec<Value> {
    vec![
        json!({
            "role": "system",
            "content": "You summarize coding-agent conversation history for context reuse. \
        Input is a JSON array of messages (system, user, assistant, tool). \
        Produce a dense technical summary in English: what the session is about, \
        what was already done (with file paths, commands, and outcomes), decisions made, \
        and anything unfinished. No preamble, no markdown headers, 400 words max. \
        You are writing notes for the next model turn, not for a human."
        }),
        json!({
            "role": "user",
            "content": serde_json::to_string(&older).unwrap_or_default(),
        }),
    ]
}

/// Compact the session: summarize everything before `split` into a
/// rolling note and splice it in as a single system-level context
/// message. On summarizer failure the session is left untouched —
/// compaction is best-effort; the next turn will retry.
pub async fn compact_session(
    state: &AppState,
    session_id: &str,
    split: usize,
) -> Result<String, crate::llm::LlmError> {
    let older: Vec<Value> = {
        let sessions = state.sessions_read();
        let session = sessions
            .get(session_id)
            .ok_or_else(|| crate::llm::LlmError {
                kind: crate::llm::LlmErrorKind::Unknown,
                message: "session vanished during compaction".into(),
                status: None,
            })?;
        session.messages[1..split.max(1)].to_vec()
    };
    if older.is_empty() {
        return Ok(String::new());
    }

    // Summarizer call: same client/timeouts, dedicated model when
    // configured, no tools (chat() sends none when tools=None).
    let mut config = state.config.clone();
    if let Some(model) = &state.config.compaction_model {
        config.model = model.clone();
    }
    let messages = summarization_messages(&older);
    let summary_value = llm::chat(&config, &messages, None, None, None).await?;
    let summary = state
        .config
        .backend()
        .extract_response_text(&summary_value)
        .trim()
        .to_string();

    let mut sessions = state.sessions_write();
    let session = sessions
        .get_mut(session_id)
        .ok_or_else(|| crate::llm::LlmError {
            kind: crate::llm::LlmErrorKind::Unknown,
            message: "session vanished during compaction splice".into(),
            status: None,
        })?;
    // Re-validate the boundary: the turn that ran concurrently may
    // have appended messages; cut only at a message that is not part
    // of an unfinished tool round.
    let split = split.min(session.messages.len());
    let system = session.messages[0].clone();
    let tail: Vec<Value> = session.messages[split..].to_vec();
    session.messages = vec![
        system,
        json!({
            "role": "user",
            "content": format!(
                "[Context compacted by acp-bridge to stay within the model window. \
        Summary of the conversation so far: {summary}]"
            ),
        }),
        json!({
            "role": "assistant",
            "content": "Understood — continuing from that summary."
        }),
    ];
    session.messages.extend(tail);
    Ok(summary)
}

/// One conversation-history entry mapped back to a Client-visible
/// replay notification (issue #17). Derived from the persisted
/// OpenAI-style history at `session/load` time — nothing extra is
/// stored for this.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayEvent {
    UserText(String),
    AssistantText(String),
    ToolCall {
        id: String,
        name: String,
        args: Value,
    },
    ToolResult {
        id: String,
        result: String,
    },
}

/// Map the persisted session history to replay events, in order. The
/// system prompt is skipped (it is re-created, not replayed); thinking
/// text is display-only and never persisted (#5), so it never replays.
pub fn replay_updates(messages: &[Value]) -> Vec<ReplayEvent> {
    let mut events = Vec::new();
    for m in messages {
        match m.get("role").and_then(|v| v.as_str()) {
            Some("user") => {
                let text = match m.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(parts)) => extract_text_parts(parts),
                    _ => continue,
                };
                if !text.is_empty() {
                    events.push(ReplayEvent::UserText(text));
                }
            }
            Some("assistant") => {
                if let Some(text) = m.get("content").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        events.push(ReplayEvent::AssistantText(text.to_string()));
                    }
                }
                if let Some(calls) = m.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in calls {
                        let id = tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        // OpenAI-compatible carries arguments as a JSON
                        // string; Ollama native as an object. Normalize to
                        // a Value for the rawInput field.
                        let args: Value = match tc.get("function").and_then(|f| f.get("arguments"))
                        {
                            Some(Value::String(s)) => {
                                serde_json::from_str(s).unwrap_or(Value::Null)
                            }
                            Some(v @ Value::Object(_)) => v.clone(),
                            _ => Value::Null,
                        };
                        events.push(ReplayEvent::ToolCall { id, name, args });
                    }
                }
            }
            Some("tool") => {
                let id = m
                    .get("tool_call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let result = m
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                events.push(ReplayEvent::ToolResult { id, result });
            }
            _ => {} // system and anything unrecognized
        }
    }
    events
}

/// Restore a persisted session into the live session map and return the
/// replay timeline. `replay = false` is `session/resume` (spec: restore
/// context, respond without streaming history); `replay = true` is
/// `session/load`.
///
/// The request cwd must match the persisted cwd — the spec makes cwd the
/// base for relative-path resolution, so silently re-anchoring a session
/// to a different tree would break tool sandboxing.
pub fn session_restore(
    state: &AppState,
    replay: bool,
    session_id: &str,
    request_cwd: &str,
    store: &crate::session_store::SessionStore,
) -> Result<Vec<ReplayEvent>, AcpError> {
    let record = store
        .load(session_id)
        .map_err(|e| AcpError::LlmError {
            reason: format!("persistence store error: {e}"),
        })?
        .ok_or_else(|| AcpError::UnknownSession {
            session_id: session_id.to_string(),
        })?;

    // Issue #22 strict mode: refuse cross-agent restores. Rows without
    // an agent identity (legacy v1 rows) and test AppStates without an
    // identity stay permissive.
    if state.strict_models {
        if let (Some(stored), Some(current)) = (&record.agent, &state.agent_identity) {
            if stored != current {
                return Err(AcpError::AgentMismatch {
                    session_id: session_id.to_string(),
                    stored: stored.clone(),
                    current: current.clone(),
                });
            }
        }
    }

    if record.cwd != request_cwd {
        return Err(AcpError::LlmError {
            reason: format!(
                "cwd mismatch for session {}: persisted '{}', requested '{}'",
                session_id, record.cwd, request_cwd
            ),
        });
    }

    let session = Session {
        messages: record.messages.clone(),
        last_active: std::time::Instant::now(),
        working_dir: PathBuf::from(&record.cwd),
        protocol_version: state.protocol_version,
        // Pinned decisions on #13/#40: overrides are in-memory only —
        // a restored session resets to the config defaults (level and
        // model), and occupancy restarts from zero until a round runs.
        thought_level: None,
        model_override: None,
        last_used_tokens: None,
    };
    state
        .sessions_write()
        .insert(session_id.to_string(), session);

    info!(
        session_id = %session_id,
        replay = replay,
        messages = record.messages.len(),
        "Session restored from persistence"
    );

    Ok(if replay {
        replay_updates(&record.messages)
    } else {
        Vec::new()
    })
}

/// Handle `session/list` (v2 baseline) — returns a list of currently
/// active sessions. The wire shape follows the v2 schema's
/// `SessionListResponse`: `{sessions: SessionInfo[], nextCursor: null}`.
///
/// We do not currently include `cwd` or `title` in `SessionInfo` (we
/// only track `messages`, `working_dir`, `protocol_version`, and
/// `last_active`); `cwd` is reachable via the working_dir field so
/// emitting it costs nothing.
pub fn session_list(state: &AppState) -> Vec<Value> {
    state
        .sessions_read()
        .iter()
        .map(|(session_id, session)| {
            json!({
                "sessionId": session_id,
                "cwd": session.working_dir.display().to_string(),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

pub struct PromptResult {
    pub status: String,
    pub text: String,
    pub error: Option<AcpError>,
    pub usage: UsageReport,
    /// Classification of the most recent LLM error (if any). `None` means
    /// the turn completed without a backend failure. Carries a stable
    /// `LlmErrorKind` so the JSON-RPC response can attach a structured
    /// `error.data.category` field Clients can switch on.
    pub error_class: Option<crate::llm::LlmErrorKind>,
    /// True if the last LLM failure was a transient error the Client
    /// could retry by re-issuing the same prompt.
    pub error_retryable: bool,
    /// ACP v2 `PromptResponse.messageId` — opaque, unique within the
    /// session. Required by the v2 schema. acp-bridge mints this when it
    /// accepts the prompt (before invoking the LLM) and echoes it on
    /// the response; v2 Clients use it to correlate the response with
    /// `user_message` / `agent_message` session updates.
    pub message_id: String,
}

/// Estimated token usage for a completed prompt turn. The wire shape
/// consumed by `acp::notify_usage` (`used`, `size`) — no cost is
/// reported because acp-bridge runs against local backends where cost
/// is unknown.
#[derive(Debug, Clone, Copy)]
pub struct UsageReport {
    pub used: u64,
    pub size: u64,
}

// ---------------------------------------------------------------------------
// Backend-specific message formatting, response extraction, and tool-call handling
// now live in `crate::llm::Backend` so `engine` stays transport-agnostic.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Backend;
    use crate::protocol::ProtocolVersion;
    use axum::body::Body;
    use axum::routing::post;
    use axum::Router;

    #[test]
    fn extract_user_text_handles_acp_array_shape() {
        let prompt = serde_json::json!([
            {"type": "text", "text": "查大腦"},
            {"type": "text", "text": "查 X 的看法"}
        ]);
        let text = extract_user_text_from_prompt(&prompt);
        assert_eq!(text, "查大腦\n查 X 的看法");
    }

    #[test]
    fn extract_user_text_handles_single_block_object() {
        let prompt = serde_json::json!({"type": "text", "text": "hello"});
        assert_eq!(extract_user_text_from_prompt(&prompt), "hello");
    }

    #[test]
    fn extract_user_text_handles_resource_link_in_array() {
        let prompt = serde_json::json!([
            {"type": "text", "text": "describe this"},
            {"type": "resource_link", "uri": "file:///etc/hostname", "name": "hostname"}
        ]);
        let text = extract_user_text_from_prompt(&prompt);
        assert!(text.contains("describe this"));
        assert!(text.contains("[Attached resource: hostname (file:///etc/hostname)]"));
    }

    #[test]
    fn extract_user_text_handles_pure_resource_link_object() {
        // Some Clients send a ResourceLink as the entire prompt.
        let prompt = serde_json::json!({
            "type": "resource_link",
            "uri": "file:///etc/hostname",
            "name": "hostname"
        });
        assert_eq!(
            extract_user_text_from_prompt(&prompt),
            "[Attached resource: hostname (file:///etc/hostname)]"
        );
    }

    #[test]
    fn extract_user_text_handles_resource_link_without_name() {
        let prompt = serde_json::json!({
            "type": "resource_link",
            "uri": "file:///x"
        });
        assert_eq!(
            extract_user_text_from_prompt(&prompt),
            "[Attached resource: file:///x]"
        );
    }

    #[test]
    fn extract_user_text_handles_plain_string() {
        let prompt = serde_json::json!("hello world");
        assert_eq!(extract_user_text_from_prompt(&prompt), "hello world");
    }

    #[test]
    fn extract_user_text_returns_empty_on_null() {
        let prompt = serde_json::Value::Null;
        assert_eq!(extract_user_text_from_prompt(&prompt), "");
    }

    #[test]
    fn extract_user_text_ignores_non_text_array_entries() {
        let prompt = serde_json::json!([
            {"type": "image", "data": "iVBORw0K..."},
            {"type": "text", "text": "describe this"}
        ]);
        assert_eq!(extract_user_text_from_prompt(&prompt), "describe this");
    }

    #[test]
    fn extract_user_images_handles_acp_array_shape() {
        let prompt = serde_json::json!([
            {"type": "text", "text": "describe"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
            {"type": "image", "data": "BBBB"}
        ]);
        let images = extract_user_images_from_prompt(&prompt);
        assert_eq!(
            images,
            vec![
                ImageBlock {
                    data: "AAAA".into(),
                    mime_type: "image/png".into()
                },
                ImageBlock {
                    data: "BBBB".into(),
                    mime_type: "image/jpeg".into()
                },
            ]
        );
    }

    #[test]
    fn extract_user_images_handles_single_image_object() {
        let prompt = serde_json::json!({
            "type": "image",
            "data": "AAAA",
            "mimeType": "image/webp"
        });
        let images = extract_user_images_from_prompt(&prompt);
        assert_eq!(
            images,
            vec![ImageBlock {
                data: "AAAA".into(),
                mime_type: "image/webp".into()
            }]
        );
    }

    #[test]
    fn extract_user_images_defaults_to_jpeg_when_mime_missing() {
        let prompt = serde_json::json!([{"type": "image", "data": "AAAA"}]);
        let images = extract_user_images_from_prompt(&prompt);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime_type, "image/jpeg");
    }

    #[test]
    fn extract_user_images_empty_on_string_prompt() {
        let prompt = serde_json::json!("just text");
        assert!(extract_user_images_from_prompt(&prompt).is_empty());
    }

    #[test]
    fn strip_sender_context_pulls_leading_block_and_returns_clean_text() {
        let input = "<sender_context>\n{\"schema\":\"openab.sender.v1\",\"user_id\":\"729\"}\n</sender_context>查一下大腦";
        let (cleaned, ctx) = strip_sender_context(input);
        assert_eq!(cleaned, "查一下大腦");
        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert!(ctx.contains("openab.sender.v1"));
        assert!(ctx.contains("729"));
    }

    #[test]
    fn strip_sender_context_passthrough_when_no_block() {
        let (cleaned, ctx) = strip_sender_context("hello world");
        assert_eq!(cleaned, "hello world");
        assert!(ctx.is_none());
    }

    #[test]
    fn strip_sender_context_handles_open_tag_without_close() {
        let input = "<sender_context>unterminated";
        let (cleaned, ctx) = strip_sender_context(input);
        assert_eq!(cleaned, input);
        assert!(ctx.is_none());
    }

    #[test]
    fn strip_sender_context_when_block_is_whole_input_yields_empty_text() {
        let input = "<sender_context>just metadata</sender_context>";
        let (cleaned, ctx) = strip_sender_context(input);
        assert_eq!(cleaned, "");
        assert_eq!(ctx.unwrap(), "just metadata");
    }

    #[test]
    fn response_helpers_support_ollama_and_openai_formats() {
        let tool_call = json!({"id": "call-1", "function": {"name": "read_file"}});
        let ollama = json!({
            "message": {"content": "ollama", "tool_calls": [tool_call.clone()]}
        });
        let openai = json!({
            "choices": [{"message": {"content": "openai", "tool_calls": [tool_call.clone()]}}]
        });

        assert_eq!(Backend::Ollama.extract_response_text(&ollama), "ollama");
        assert_eq!(
            Backend::Ollama.extract_tool_calls(&ollama),
            vec![tool_call.clone()]
        );
        assert_eq!(Backend::OpenAi.extract_response_text(&openai), "openai");
        assert_eq!(Backend::OpenAi.extract_tool_calls(&openai), vec![tool_call]);
    }

    fn cfg_with_image(supports_image: bool) -> LlmConfig {
        LlmConfig {
            base_url: "http://localhost:11434/v1".into(),
            model: "m".into(),
            api_key: "k".into(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            timeout_secs: 5,
            max_history_turns: 50,
            max_tool_rounds: 25,
            max_sessions: 0,
            session_idle_timeout_secs: 0,
            prompt_supports_image: supports_image,
            context_size: 32768,
            available_models: Vec::new(),
            compaction_threshold: None,
            compaction_model: None,
            thought_levels: Vec::new(),
            thought_levels_set: false,
            request_overrides: serde_json::Map::new(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("client"),
        }
    }

    // -- thought_level config option (issue #13) -----------------------------

    #[test]
    fn thought_level_option_absent_when_unconfigured() {
        let state = AppState::new(cfg_with_image(false));
        assert!(thought_level_config_option(&state, None).is_none());
    }

    #[test]
    fn thought_level_option_shape_and_default() {
        let mut cfg = cfg_with_image(false);
        cfg.thought_levels = vec!["low".into(), "high".into(), "max".into()];
        let state = AppState::new(cfg);

        let option = thought_level_config_option(&state, None).expect("configured → advertised");
        assert_eq!(option["id"], "thought_level");
        assert_eq!(option["category"], "thought_level");
        assert_eq!(option["type"], "select");
        // bb renders the FIRST level as selected before any set_config_option.
        assert_eq!(option["currentValue"], "low");
        assert_eq!(
            option["options"],
            serde_json::json!([
                {"value": "low", "name": "low"},
                {"value": "high", "name": "high"},
                {"value": "max", "name": "max"}
            ])
        );

        // Session override (set via session_new's request_overrides
        // default) wins over the first-level default.
        let option = thought_level_config_option(&state, Some("max")).expect("present");
        assert_eq!(option["currentValue"], "max");
    }

    #[test]
    fn session_new_advertises_model_and_thought_level() {
        let mut cfg = cfg_with_image(false);
        cfg.thought_levels = vec!["low".into(), "high".into(), "max".into()];
        let state = AppState::new(cfg);
        state.sessions_write().insert(
            "s1".into(),
            Session::new(
                json!({"role": "system", "content": "x"}),
                "/tmp".into(),
                ProtocolVersion::V1,
            ),
        );

        let body = session_new_response(&state, "s1");
        // bb's pipeline needs BOTH: a model option to build the model
        // list from, and the thought_level option the probe reads back.
        assert_eq!(body["configOptions"][0]["id"], "model");
        assert_eq!(body["configOptions"][0]["category"], "model");
        assert_eq!(body["configOptions"][0]["currentValue"], "m");
        assert_eq!(body["configOptions"][0]["options"][0]["value"], "m");
        assert_eq!(body["configOptions"][1]["id"], "thought_level");
        assert_eq!(body["configOptions"][1]["currentValue"], "low");
    }

    #[test]
    fn model_option_advertises_backend_list() {
        // Issue #40: the startup probe's list is advertised; the
        // configured model stays first/default even when the probe
        // omitted it; probe failure degrades to the single model.
        let mut cfg = cfg_with_image(false);
        cfg.model = "gemma4".into();
        cfg.available_models = vec!["claude-x".into(), "gemma4".into(), "gpt-y".into()];
        let state = AppState::new(cfg);

        let option = model_config_option(&state, None);
        assert_eq!(option["currentValue"], "gemma4");
        let values: Vec<&str> = option["options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| o["value"].as_str().unwrap())
            .collect();
        assert_eq!(values, vec!["claude-x", "gemma4", "gpt-y"]);

        // Session override reflected as currentValue.
        let option = model_config_option(&state, Some("claude-x"));
        assert_eq!(option["currentValue"], "claude-x");

        // Probe failed → configured model only.
        let mut cfg = cfg_with_image(false);
        cfg.model = "gemma4".into();
        cfg.available_models = Vec::new();
        let state = AppState::new(cfg);
        let option = model_config_option(&state, None);
        assert_eq!(option["options"].as_array().unwrap().len(), 1);
        assert_eq!(option["options"][0]["value"], "gemma4");
    }

    #[test]
    fn model_option_works_without_thought_levels() {
        // Issue #40: model switching is zero-config — the option must
        // be advertised even when thought_levels is empty.
        let cfg = cfg_with_image(false);
        let state = AppState::new(cfg);
        state.sessions_write().insert(
            "s1".into(),
            Session::new(
                json!({"role": "system", "content": "x"}),
                "/tmp".into(),
                ProtocolVersion::V1,
            ),
        );
        let body = session_new_response(&state, "s1");
        assert_eq!(body["configOptions"][0]["id"], "model");
        assert_eq!(body["configOptions"].as_array().unwrap().len(), 1);

        // …and set_config_option on "model" is accepted without
        // thought_levels configured.
        let updated = session_set_config_option(&state, "s1", "model", &json!("m")).unwrap();
        assert_eq!(updated["configOptions"][0]["currentValue"], "m");
        assert_eq!(
            state.sessions_read().get("s1").unwrap().model_override,
            Some("m".to_string())
        );
    }

    #[test]
    fn set_config_option_validates_and_stores() {
        let mut cfg = cfg_with_image(false);
        cfg.thought_levels = vec!["low".into(), "high".into(), "max".into()];
        let state = AppState::new(cfg);

        state.sessions_write().insert(
            "s1".into(),
            Session::new(
                json!({"role": "system", "content": "x"}),
                "/tmp".into(),
                ProtocolVersion::V1,
            ),
        );

        // Unknown configId → rejected.
        let err = session_set_config_option(&state, "s1", "fast", &json!("true")).unwrap_err();
        assert!(err.to_string().contains("unknown config option"));

        // Unknown session → rejected.
        let err =
            session_set_config_option(&state, "nope", "thought_level", &json!("max")).unwrap_err();
        assert!(matches!(err, AcpError::UnknownSession { .. }));

        // Value outside the advertised set → rejected with the list.
        let err =
            session_set_config_option(&state, "s1", "thought_level", &json!("ultra")).unwrap_err();
        assert!(err.to_string().contains("ultra"), "got: {err}");

        // Non-string value → rejected.
        let err = session_set_config_option(&state, "s1", "thought_level", &json!(3)).unwrap_err();
        assert!(err.to_string().contains("expected a string"));

        // bb's model probe: configId "model" with the configured model
        // is accepted (no-op selection) and echoes the full option set.
        let updated = session_set_config_option(&state, "s1", "model", &json!("m")).unwrap();
        assert_eq!(updated["configOptions"][0]["id"], "model");
        assert_eq!(updated["configOptions"][0]["currentValue"], "m");
        assert_eq!(updated["configOptions"][1]["id"], "thought_level");

        // …but a model we never offered is rejected.
        let err =
            session_set_config_option(&state, "s1", "model", &json!("other-model")).unwrap_err();
        assert!(err.to_string().contains("not offered"), "got: {err}"); // Valid thought level → stored, response carries the updated
                                                                        // option (index 1: the model option is first).
        let updated =
            session_set_config_option(&state, "s1", "thought_level", &json!("max")).unwrap();
        assert_eq!(updated["configOptions"][1]["currentValue"], "max");
        assert_eq!(
            state.sessions_read().get("s1").unwrap().thought_level,
            Some("max".to_string())
        );
    }

    #[test]
    fn strict_mode_refuses_cross_agent_restore() {
        // Issue #22: ACP_SESSION_STRICT_MODELS — a session persisted by
        // agent 'ollama' must not restore into agent 'cometapi' in
        // strict mode; permissive mode (default) allows it.
        let store = crate::session_store::SessionStore::open_in_memory().unwrap();
        let mut record = crate::session_store::SessionRecord {
            session_id: "s_x".into(),
            cwd: "/tmp".into(),
            protocol_version: 1,
            title: None,
            created_at: 1,
            updated_at: 1,
            messages: vec![json!({"role": "user", "content": "hi"})],
            agent: Some("ollama".into()),
        };
        store.save(&record).unwrap();

        let mut cfg = cfg_with_image(false);
        cfg.model = "comet-model".into();
        let mut state = AppState::with_store(cfg, Some(Arc::new(store)), Some("cometapi".into()));

        // Permissive (default): restores fine. Fresh Arc — get_mut is
        // sound until the first clone.
        Arc::get_mut(&mut state)
            .expect("AppState not yet shared")
            .strict_models = false;
        session_restore(&state, false, "s_x", "/tmp", state.store.as_ref().unwrap()).unwrap();

        // Strict: same-agent still fine, cross-agent refused.
        Arc::get_mut(&mut state)
            .expect("AppState not yet shared")
            .strict_models = true;
        record.session_id = "s_same".into();
        record.agent = Some("cometapi".into());
        state.store.as_ref().unwrap().save(&record).unwrap();
        session_restore(
            &state,
            false,
            "s_same",
            "/tmp",
            state.store.as_ref().unwrap(),
        )
        .unwrap();

        let err = session_restore(&state, false, "s_x", "/tmp", state.store.as_ref().unwrap())
            .unwrap_err();
        assert!(
            err.to_string().contains("persisted by agent 'ollama'"),
            "got: {err}"
        );
        // A refused session must never reach memory: fresh cross-agent
        // row, strict restore, then assert absence.
        record.session_id = "s_never".into();
        record.agent = Some("ollama".into());
        state.store.as_ref().unwrap().save(&record).unwrap();
        assert!(session_restore(
            &state,
            false,
            "s_never",
            "/tmp",
            state.store.as_ref().unwrap()
        )
        .is_err());
        assert!(state.sessions_read().get("s_never").is_none());
    }

    // -- compaction (issue #25) ------------------------------------------------

    #[test]
    fn compaction_trigger_matrix() {
        // None (unset) → default 0.75; Some(0.0) → disabled;
        // Some(0.5) → honored.
        assert_eq!(
            compaction_trigger_fraction(&cfg_with_image(false)),
            Some(0.75)
        );
        let mut cfg = cfg_with_image(false);
        cfg.compaction_threshold = Some(0.0);
        assert_eq!(compaction_trigger_fraction(&cfg), None);
        cfg.compaction_threshold = Some(0.5);
        assert_eq!(compaction_trigger_fraction(&cfg), Some(0.5));
    }

    #[test]
    fn should_compact_uses_reported_tokens_over_estimate() {
        let mut cfg = cfg_with_image(false);
        cfg.context_size = 1000;
        cfg.compaction_threshold = Some(0.5);
        let session = Session::new(
            json!({"role":"system","content":"s"}),
            "/tmp".into(),
            ProtocolVersion::V1,
        );

        // Backend-reported 600/1000 ≥ 50% → compact, threshold 500.
        assert_eq!(
            should_compact(&cfg, Some((600, 10)), &session),
            Some((600, 500))
        );

        // Reported 400 → below threshold, no compaction — even though
        // a huge history would estimate higher (reported wins).
        let mut big = session.clone();
        big.messages = vec![
            json!({"role":"system","content":"s"}),
            json!({"role":"user","content":"x".repeat(100000)}),
        ];
        assert_eq!(should_compact(&cfg, Some((400, 10)), &big), None);

        // No report → falls back to chars/4 estimate; big history compacts.
        assert!(should_compact(&cfg, None, &big).is_some());

        // Zero context size → never compacts (no window to exceed).
        cfg.context_size = 0;
        assert_eq!(should_compact(&cfg, Some((9999, 0)), &session), None);
    }

    #[test]
    fn compaction_split_respects_turn_boundary() {
        let mut session = Session::new(
            json!({"role":"system","content":"s"}),
            "/tmp".into(),
            ProtocolVersion::V1,
        );
        // 6 messages after the system prompt = 3 turns.
        for i in 0..3 {
            session
                .messages
                .push(json!({"role":"user","content":format!("u{i}")}));
            session
                .messages
                .push(json!({"role":"assistant","content":format!("a{i}")}));
        }
        // keep = 2 turns (4 messages) → split at 1+6-4 = 3.
        assert_eq!(session.compaction_split(2), Some(3));
        // Everything fits → no split.
        assert_eq!(session.compaction_split(10), None);

        // Round-boundary snap: a naive cut that lands on a `tool`
        // result must walk back to the assistant that owns it — an
        // orphaned tool result (its tool_calls summarized away) is
        // rejected by OpenAI-compatible backends.
        let mut session = Session::new(
            json!({"role":"system","content":"s"}),
            "/tmp".into(),
            ProtocolVersion::V1,
        );
        for m in [
            json!({"role":"user","content":"u1"}),
            json!({"role":"assistant","content":"a1","tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}]}),
            json!({"role":"tool","content":"r1","tool_call_id":"c1"}),
            json!({"role":"tool","content":"r2","tool_call_id":"c1"}),
            json!({"role":"assistant","content":"a2"}),
            json!({"role":"user","content":"u2"}),
            json!({"role":"assistant","content":"a3","tool_calls":[{"id":"c2","type":"function","function":{"name":"bash","arguments":"{}"}}]}),
            json!({"role":"tool","content":"r3","tool_call_id":"c2"}),
            json!({"role":"tool","content":"r4","tool_call_id":"c2"}),
            json!({"role":"assistant","content":"a4"}),
        ] {
            session.messages.push(m);
        }
        // len = 11 (system + 10), keep = 2 → naive split = 9, which is
        // tool r4 → snap back to the owning assistant a3 (index 7).
        assert_eq!(
            session.compaction_split(1),
            Some(7),
            "messages: {:?}",
            session.messages
        );
        // And the tail starts with the assistant, not a tool result.
        let split = session.compaction_split(1).unwrap();
        assert_ne!(
            session.messages[split]["role"].as_str(),
            Some("tool"),
            "tail must not start with an orphaned tool result"
        );
    }

    #[tokio::test]
    async fn compaction_summarizes_into_rolling_note() {
        // End-to-end: threshold crossed → summarizer called with the
        // OLDER messages only → session spliced to system + summary
        // pair + recent tail; tail and system preserved verbatim.
        let router: Router = Router::new().route(
            "/v1/chat/completions",
            post(|req: axum::extract::Request<Body>| async move {
                let body = axum::body::to_bytes(req.into_body(), 1024 * 1024)
                    .await
                    .unwrap();
                let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                // The summarizer sends exactly two messages: system
                // contract + the older history as a JSON string.
                let msgs = body["messages"].as_array().unwrap();
                assert_eq!(msgs.len(), 2, "summarizer shape: {msgs:?}");
                assert!(msgs[0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("summarize coding-agent"));
                let older = msgs[1]["content"].as_str().unwrap();
                assert!(older.contains("OLD_FACT"), "older history must be summarized, got: {older}");
                assert!(!older.contains("RECENT"), "recent tail must NOT be summarized, got: {older}");
                axum::Json(json!({
                    "choices": [{"message": {"role": "assistant", "content": "OLD_FACT noted; work half done."}}]
                }))
            }),
        );
        // split: max_history_turns=2 → keep 4 messages → split at 5-4=1... 5 messages total → split = 1: everything but system is "older", so include the marker via tail check below.
        // Serve the mock on an ephemeral port and point the agent at it.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let mut cfg = cfg_with_image(false);
        cfg.base_url = format!("http://127.0.0.1:{port}/v1");
        cfg.context_size = 1000;
        cfg.compaction_threshold = Some(0.1); // trigger immediately
        let state = AppState::new(cfg);
        state.sessions_write().insert(
            "s1".into(),
            Session::new(
                json!({"role":"system","content":"SYS"}),
                "/tmp".into(),
                ProtocolVersion::V1,
            ),
        );
        {
            let mut sessions = state.sessions_write();
            let s = sessions.get_mut("s1").unwrap();
            s.messages = vec![
                json!({"role":"system","content":"SYS"}),
                json!({"role":"user","content":"remember OLD_FACT"}),
                json!({"role":"assistant","content":"noted"}),
                json!({"role":"user","content":"RECENT question"}),
                json!({"role":"assistant","content":"RECENT answer"}),
            ];
        }
        // Split = boundary between "older" (summarized) and the recent
        // tail (kept verbatim): older = messages[1..3] (the OLD_FACT
        // exchange), tail = messages[3..] (the RECENT pair). System
        // (index 0) is never summarized.
        let split = 3;
        let summary = compact_session(&state, "s1", split).await.unwrap();
        assert!(summary.contains("OLD_FACT"), "summary: {summary}");

        let sessions = state.sessions_read();
        let s = sessions.get("s1").unwrap();
        assert_eq!(s.messages[0]["content"], "SYS", "system preserved");
        assert!(
            serde_json::to_string(&s.messages[1])
                .unwrap()
                .contains("OLD_FACT noted"),
            "rolling note present: {:?}",
            s.messages[1]
        );
        assert!(
            serde_json::to_string(&s.messages[2])
                .unwrap()
                .contains("continuing from that summary"),
            "ack present"
        );
        // Recent tail preserved verbatim after the note.
        let tail_json = serde_json::to_string(&s.messages[3..]).unwrap();
        assert!(tail_json.contains("RECENT"), "tail kept: {tail_json}");
    }

    #[test]
    fn initialize_does_not_advertise_image_by_default() {
        let caps = initialize(&cfg_with_image(false), ProtocolVersion::V1, false);
        assert_eq!(caps["protocolVersion"], 1);
        assert_eq!(
            caps["agentCapabilities"]["promptCapabilities"]["image"], false,
            "image must default to false to avoid clients forwarding image \
             attachments to local backends without vision"
        );
        assert_eq!(
            caps["agentCapabilities"]["promptCapabilities"]["audio"],
            false
        );
        assert_eq!(
            caps["agentCapabilities"]["promptCapabilities"]["embeddedContext"],
            false
        );
    }

    #[test]
    fn initialize_advertises_image_when_opted_in() {
        let caps = initialize(&cfg_with_image(true), ProtocolVersion::V1, false);
        assert_eq!(
            caps["agentCapabilities"]["promptCapabilities"]["image"], true,
            "image must be true when prompt_supports_image is set"
        );
    }

    #[test]
    fn initialize_v2_uses_unified_capabilities_and_info_shape() {
        let caps = initialize(&cfg_with_image(true), ProtocolVersion::V2, false);
        assert_eq!(caps["protocolVersion"], 2);
        // v2 collapses agentCapabilities / clientCapabilities into a single
        // `capabilities`, and `agentInfo` / `clientInfo` into `info`.
        // Spec deliberately forbids the v1-style aliases — only one of
        // each pair is permitted on the wire.
        assert!(caps.get("info").is_some(), "v2 must emit info");
        assert!(
            caps.get("capabilities").is_some(),
            "v2 must emit capabilities"
        );
        assert!(
            caps.get("agentInfo").is_none(),
            "v1 agentInfo must not appear on v2 wire"
        );
        assert!(
            caps.get("agentCapabilities").is_none(),
            "v1 agentCapabilities must not appear on v2 wire"
        );
        // Image is advertised as `{}` (capability marker) on v2, not `true`.
        let image = &caps["capabilities"]["session"]["prompt"]["image"];
        assert!(
            image.is_object(),
            "v2 image capability should be an object marker, got: {image}"
        );
    }

    #[test]
    fn initialize_v2_omits_image_when_disabled() {
        let caps = initialize(&cfg_with_image(false), ProtocolVersion::V2, false);
        let prompt_caps = &caps["capabilities"]["session"]["prompt"];
        assert!(
            prompt_caps.get("image").is_none(),
            "v2 image must be omitted (not `false`) when not supported; \
             clients interpret present-as-object as 'supported' and absent \
             as 'not advertised'"
        );
    }

    #[test]
    fn initialize_falls_back_to_v1_for_unknown_versions() {
        // Default Default for ProtocolVersion is V1, so omitting the arg
        // should produce the v1 wire shape — Clients that omit
        // protocolVersion still get a working session on the most widely
        // deployed wire shape.
        let caps = initialize(&cfg_with_image(false), ProtocolVersion::default(), false);
        assert_eq!(caps["protocolVersion"], 1);
    }

    #[test]
    fn notification_tool_start_carries_id() {
        let n = Notification::ToolStart {
            id: "tc_42".into(),
            name: "read_file".into(),
            args: json!({"path": "src/main.rs"}),
        };
        match n {
            Notification::ToolStart { id, name, args } => {
                assert_eq!(id, "tc_42");
                assert_eq!(name, "read_file");
                assert_eq!(args["path"], "src/main.rs");
            }
            _ => panic!("expected ToolStart"),
        }
    }

    #[test]
    fn notification_tool_done_carries_id_and_status() {
        let n = Notification::ToolDone {
            id: "tc_42".into(),
            name: "read_file".into(),
            status: "completed".into(),
            result: Some("file contents".into()),
            diff: None,
        };
        match n {
            Notification::ToolDone {
                id,
                name,
                status,
                result,
                ..
            } => {
                assert_eq!(id, "tc_42");
                assert_eq!(name, "read_file");
                assert_eq!(status, "completed");
                assert_eq!(result.as_deref(), Some("file contents"));
            }
            _ => panic!("expected ToolDone"),
        }
    }
}
