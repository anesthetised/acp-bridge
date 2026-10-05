//! Configuration — TOML file with env var override.
//!
//! Priority: env var > config file > default.
//! When spawned by openab, env vars are sufficient (no config file needed).
//! For standalone deployment, use a config file.

use serde::Deserialize;
use std::path::Path;
use tracing::{info, warn};

use crate::llm::LlmConfig;
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;

/// On-disk config file structure.
#[derive(Debug, Deserialize, Default)]
pub struct ConfigFile {
    #[serde(default)]
    pub llm: LlmSection,
}

#[derive(Debug, Deserialize, Default)]
pub struct LlmSection {
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub system_prompt: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub timeout_secs: Option<u64>,
    pub max_history_turns: Option<usize>,
    /// Maximum tool call rounds per prompt (0 = unlimited). Mirrors the
    /// `LLM_MAX_TOOL_ROUNDS` env var; the env var takes precedence.
    pub max_tool_rounds: Option<usize>,
    pub max_sessions: Option<usize>,
    pub session_idle_timeout_secs: Option<u64>,
    /// Whether the backend accepts image content blocks. Mirrors the
    /// `LLM_SUPPORTS_IMAGE` env var; the env var takes precedence.
    #[serde(default)]
    pub supports_image: Option<bool>,
    /// Model context window in tokens. Mirrors the `LLM_MODEL_CONTEXT` env
    /// var; the env var takes precedence. Reported as `size` in
    /// `usage_update` notifications.
    #[serde(default)]
    pub model_context: Option<u64>,
    /// Arbitrary passthrough fields merged into the top level of every
    /// upstream request body (issue #2) — e.g. `reasoning_effort`,
    /// `top_p`. Applied last, so overrides win over built-in sampling
    /// fields; reserved engine-owned keys (`model`, `messages`,
    /// `stream`, `tools`) are ignored with a warning. No env var:
    /// structural config, not a secret.
    #[serde(default)]
    pub request_overrides: Option<serde_json::Map<String, Value>>,
    /// Reasoning-effort levels advertised to Clients as a
    /// `thought_level` config option (issue #13). Empty = off. The env
    /// var `LLM_THOUGHT_LEVELS` (comma-separated) takes precedence.
    #[serde(default)]
    pub thought_levels: Option<Vec<String>>,
}

impl ConfigFile {
    /// Try to load from a TOML file path. Returns default if file doesn't exist.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(cfg) => {
                    info!(path = %path.display(), "Loaded config file");
                    cfg
                }
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Failed to parse config file, using defaults");
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    /// Merge config file values into LlmConfig. Env vars always take precedence.
    pub fn into_llm_config(self) -> LlmConfig {
        let file = self.llm;

        // Helper: env var wins, then config file, then default
        let base_url = std::env::var("LLM_BASE_URL")
            .or_else(|_| std::env::var("OLLAMA_BASE_URL"))
            .ok()
            .or(file.base_url)
            .unwrap_or_else(|| "http://localhost:11434/v1".into());

        let model = std::env::var("LLM_MODEL")
            .or_else(|_| std::env::var("OLLAMA_MODEL"))
            .ok()
            .or(file.model)
            .unwrap_or_else(|| "gemma4:26b".into());

        let api_key = std::env::var("LLM_API_KEY")
            .or_else(|_| std::env::var("OLLAMA_API_KEY"))
            .ok()
            .or(file.api_key)
            .unwrap_or_else(|| "local-ai".into());

        let system_prompt = std::env::var("LLM_SYSTEM_PROMPT")
            .ok()
            .or(file.system_prompt);

        let temperature = std::env::var("LLM_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .or(file.temperature)
            .filter(|t| t.is_finite());

        let max_tokens = std::env::var("LLM_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(file.max_tokens);

        let timeout_secs = std::env::var("LLM_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(file.timeout_secs)
            .unwrap_or(300);

        let max_history_turns = std::env::var("LLM_MAX_HISTORY_TURNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(file.max_history_turns)
            .unwrap_or(50);

        let max_tool_rounds = std::env::var("LLM_MAX_TOOL_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(file.max_tool_rounds)
            .unwrap_or(crate::engine::DEFAULT_MAX_TOOL_ROUNDS);

        let max_sessions = std::env::var("LLM_MAX_SESSIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(file.max_sessions)
            .unwrap_or(0);

        let session_idle_timeout_secs = std::env::var("LLM_SESSION_IDLE_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(file.session_idle_timeout_secs)
            .unwrap_or(0);

        let prompt_supports_image = match std::env::var("LLM_SUPPORTS_IMAGE").as_deref() {
            Ok("1") | Ok("true") | Ok("yes") | Ok("on") => true,
            Ok("0") | Ok("false") | Ok("no") | Ok("off") => false,
            _ => file.supports_image.unwrap_or(false),
        };

        let context_size = std::env::var("LLM_MODEL_CONTEXT")
            .ok()
            .and_then(|v| v.parse().ok())
            .or(file.model_context)
            .unwrap_or(32768);

        // Thought levels (issue #13): env wins over config file; empty
        // = feature off (no picker advertised).
        let thought_levels = std::env::var("LLM_THOUGHT_LEVELS")
            .ok()
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or(file.thought_levels.unwrap_or_default());

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .pool_max_idle_per_host(4)
            .build()
            .expect("Failed to create HTTP client");

        LlmConfig {
            base_url,
            model,
            api_key,
            system_prompt,
            temperature,
            max_tokens,
            timeout_secs,
            max_history_turns,
            max_tool_rounds,
            max_sessions,
            session_idle_timeout_secs,
            prompt_supports_image,
            context_size,
            thought_levels,
            // Issue #2: request overrides are config-only (structural
            // passthrough, not a secret) — no env var by design.
            request_overrides: file.request_overrides.unwrap_or_default(),
            client,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ConfigFile, LlmSection};
    use std::sync::{Mutex, MutexGuard};

    /// Serializes tests that mutate process-global environment variables.
    /// Cargo runs a test binary's unit tests in parallel threads, and
    /// `std::env::set_var` / `remove_var` are process-global — without
    /// this lock the two `LLM_SYSTEM_PROMPT` tests race and fail roughly
    /// every other `cargo test --lib` run (observed on clean v0.9.1).
    /// Poison-tolerant so a panicking test does not cascade into the next.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_guard() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn llm_config_uses_system_prompt_from_file_when_env_missing() {
        let _env = env_guard();
        std::env::remove_var("LLM_SYSTEM_PROMPT");

        let cfg = ConfigFile {
            llm: LlmSection {
                system_prompt: Some("from file".into()),
                ..LlmSection::default()
            },
        };

        let llm = cfg.into_llm_config();
        assert_eq!(llm.system_prompt.as_deref(), Some("from file"));
    }

    #[test]
    fn llm_config_env_system_prompt_overrides_file() {
        let _env = env_guard();
        std::env::set_var("LLM_SYSTEM_PROMPT", "from env");

        let cfg = ConfigFile {
            llm: LlmSection {
                system_prompt: Some("from file".into()),
                ..LlmSection::default()
            },
        };

        let llm = cfg.into_llm_config();
        assert_eq!(llm.system_prompt.as_deref(), Some("from env"));

        std::env::remove_var("LLM_SYSTEM_PROMPT");
    }

    #[test]
    fn llm_config_max_tool_rounds_precedence() {
        // Mutates LLM_MAX_TOOL_ROUNDS — must hold ENV_LOCK like the other
        // env-touching tests in this module.
        let _env = env_guard();

        // 1. Default (25) when neither env nor file is set.
        std::env::remove_var("LLM_MAX_TOOL_ROUNDS");
        let llm = ConfigFile::default().into_llm_config();
        assert_eq!(llm.max_tool_rounds, 25);

        // 2. File value used when the env var is absent. (A closure, not a
        // value: `into_llm_config` takes ownership, and the same fixture is
        // reused across the precedence scenarios below.)
        let file_only = || ConfigFile {
            llm: LlmSection {
                max_tool_rounds: Some(7),
                ..LlmSection::default()
            },
        };
        let llm = file_only().into_llm_config();
        assert_eq!(llm.max_tool_rounds, 7);

        // 3. Env wins over the file.
        std::env::set_var("LLM_MAX_TOOL_ROUNDS", "3");
        let llm = file_only().into_llm_config();
        assert_eq!(llm.max_tool_rounds, 3);

        // 0 = unlimited is preserved through the precedence chain.
        std::env::set_var("LLM_MAX_TOOL_ROUNDS", "0");
        let llm = file_only().into_llm_config();
        assert_eq!(llm.max_tool_rounds, 0);

        std::env::remove_var("LLM_MAX_TOOL_ROUNDS");
    }
}
