use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Instant;

/// ACP wire-format protocol version negotiated at `initialize` time.
///
/// The ACP working group uses a single integer that is bumped only for
/// breaking changes. acp-bridge supports v1 (stable, the version most
/// Clients speak today) and v2 (released 2026, see
/// <https://agentclientprotocol.com/protocol/v2/initialization>).
///
/// On `initialize` the Client's requested version is compared against the
/// set we support. We echo the negotiated value back so it knows whether
/// to stay on its preferred version or follow our fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProtocolVersion(pub u16);

impl ProtocolVersion {
    /// v1: the version Zed, JetBrains, ACP UI, ACP Inspector, Meuxe,
    /// and Codex CLI adapter all speak today. Stable since 2025.
    pub const V1: ProtocolVersion = ProtocolVersion(1);
    /// v2: released 2026. Unifies `agentCapabilities`/`clientCapabilities`
    /// into a single `capabilities`, adds `messageId`-keyed message
    /// upserts, and removes the `tool_call` notification in favour of
    /// `tool_call_update` as an upsert keyed by `toolCallId`.
    pub const V2: ProtocolVersion = ProtocolVersion(2);

    /// Highest version we support. Negotiated as the fallback when the
    /// Client requests something newer.
    pub const LATEST: ProtocolVersion = Self::V2;

    pub fn as_u16(self) -> u16 {
        self.0
    }
}

impl Default for ProtocolVersion {
    fn default() -> Self {
        // Defaulting to V1 is the conservative choice — Clients that
        // don't bother sending a protocolVersion still get a working
        // session on the wire shape every existing Client speaks.
        Self::V1
    }
}

impl std::fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// JSON-RPC 2.0 request/response ID. Per the spec an `id` may be a
/// string or a number; a `null`/absent `id` marks a notification.
///
/// acp-bridge must preserve the original type and value so clients that use
/// UUID/string IDs (e.g. Meuxe) get their exact ID echoed back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    Number(u64),
    String(String),
}

impl RequestId {
    /// Render the ID back into its JSON representation.
    pub fn as_value(&self) -> Value {
        match self {
            RequestId::Number(n) => Value::Number((*n).into()),
            RequestId::String(s) => Value::String(s.clone()),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    /// Present for requests, absent for notifications (e.g. `session/cancel`).
    pub id: Option<RequestId>,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub messages: Vec<Value>,
    /// Last activity timestamp for idle timeout.
    pub last_active: Instant,
    /// Working directory for this session (used for tool sandboxing).
    pub working_dir: PathBuf,
    /// ACP wire-format protocol version this session uses. Inherited
    /// from the AppState at session creation; the emit helpers branch
    /// on it so a v2 Client gets v2-shaped notifications even if a v1
    /// Client later connects to the same process.
    pub protocol_version: ProtocolVersion,
    /// Per-session reasoning-effort override (issue #13): set by
    /// `session/set_config_option` (bb's thought-level picker), applied
    /// to every round's request body as `reasoning_effort` — session
    /// choice beats `[llm.request_overrides]`. In-memory only: resets
    /// to the config default on `session/load` after a restart (pinned
    /// decision on #13).
    pub thought_level: Option<String>,
    /// Context occupancy of the most recent round, in prompt tokens as
    /// reported by the backend (issue #4/#25): the compaction trigger
    /// reads this instead of re-estimating. `None` until a backend
    /// reports usage; compaction then falls back to the chars/4
    /// estimate. Updated at turn end.
    pub last_used_tokens: Option<u64>,
    /// Per-session model override (issue #40): set by
    /// `session/set_config_option(configId: "model")` from the
    /// Client's model picker, applied to every round's request body
    /// via the existing `model_override` plumbing. In-memory only,
    /// same policy as `thought_level`.
    pub model_override: Option<String>,
}

impl Session {
    pub fn new(
        system_message: Value,
        working_dir: PathBuf,
        protocol_version: ProtocolVersion,
    ) -> Self {
        Self {
            messages: vec![system_message],
            last_active: Instant::now(),
            working_dir,
            protocol_version,
            thought_level: None,
            model_override: None,
            last_used_tokens: None,
        }
    }

    pub fn touch(&mut self) {
        self.last_active = Instant::now();
    }
}

impl Session {
    /// Trim conversation history to keep the system prompt + last `max_turns` pairs.
    /// Each "turn" = one user message + one assistant message.
    /// The system prompt (first message) is always preserved.
    pub fn trim_history(&mut self, max_turns: usize) {
        // messages[0] = system prompt, then alternating user/assistant
        let keep = max_turns * 2; // user + assistant per turn
        if self.messages.len() > keep + 1 {
            let system = self.messages[0].clone();
            let tail = self.messages.split_off(self.messages.len() - keep);
            self.messages = vec![system];
            self.messages.extend(tail);
        }
    }

    /// Compat boundary (issue #25): compaction replaces `trim_history`'s
    /// scissors when enabled — split index for the messages older than
    /// the last N turns, or `None` when history fits without cutting.
    /// Only tool-aware rounds (the agentic shape: user → tool-calling
    /// assistant → tool results → … → final assistant) start before
    /// the boundary; a cut is only *placed* at round boundaries, so
    /// tool results are never orphaned from their assistant request.
    pub fn compaction_split(&self, max_turns: usize) -> Option<usize> {
        let keep = max_turns * 2;
        if self.messages.len() <= keep + 1 {
            return None;
        }
        Some(self.messages.len() - keep)
    }
}

/// ACP-layer error codes following JSON-RPC 2.0 conventions.
#[derive(Debug, thiserror::Error)]
pub enum AcpError {
    #[error("Missing required parameter: {field}")]
    MissingParam { field: String },

    #[error("Unknown session: {session_id}")]
    UnknownSession { session_id: String },

    #[error("Method not found: {method}")]
    MethodNotFound { method: String },

    #[error("LLM communication error: {reason}")]
    LlmError { reason: String },

    #[error("Session limit reached (max: {max})")]
    SessionLimitReached { max: usize },

    #[error("Invalid parameter: {field}")]
    InvalidParam { field: String },

    #[error(
        "Session {session_id} was persisted by agent '{stored}', not '{current}' (strict mode: ACP_SESSION_STRICT_MODELS)"
    )]
    AgentMismatch {
        session_id: String,
        stored: String,
        current: String,
    },
}

impl AcpError {
    /// JSON-RPC error code for this variant.
    pub fn code(&self) -> i64 {
        match self {
            AcpError::MissingParam { .. } => -32602,   // Invalid params
            AcpError::UnknownSession { .. } => -32001, // Application error
            AcpError::MethodNotFound { .. } => -32601, // Method not found
            AcpError::LlmError { .. } => -32003,       // Application error
            AcpError::SessionLimitReached { .. } => -32004, // Application error
            AcpError::InvalidParam { .. } => -32602,   // Invalid params
            AcpError::AgentMismatch { .. } => -32001,  // Application error
        }
    }
}
