//! Configuration — TOML file with env var override.
//!
//! Priority: env var > config file > default.
//! When spawned by openab, env vars are sufficient (no config file needed).
//! For standalone deployment, use a config file.

use serde::Deserialize;
use std::path::Path;
use tracing::{info, warn};

use crate::llm::{Backend, LlmConfig};
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
    /// Allowlist filter for the advertised model list (issue #48):
    /// advertised = fetched ∩ `models`, config order wins, the
    /// configured model is always included. Unset = advertise every
    /// model the backend reports. A curation preference, not
    /// deployment-specific — TOML only, no env twin.
    pub models: Option<Vec<String>>,
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
    pub prompt_supports_image: Option<bool>,
    /// Compaction trigger as a fraction of the context window (issue
    /// #25): when a round's reported prompt tokens cross this share of
    /// `context_size`, older rounds are summarized into a rolling
    /// note before the next request. `0` disables compaction. Default
    /// 0.75 when unset.
    pub compaction_threshold: Option<f64>,
    /// Model used for the summarization call (issue #25). Default:
    /// the configured model. Intended for a cheaper dedicated
    /// summarizer on gateways where one exists.
    pub compaction_model: Option<String>,
    /// Whether the backend accepts image content blocks.
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

        // Thought levels (issue #13): env wins over config file. When
        // neither is set, apply the safe default — low/medium/high/max
        // for OpenAI-compatible backends, none for Ollama-native (no
        // reasoning_effort parameter there). An explicit empty list
        // (`thought_levels = []` or `LLM_THOUGHT_LEVELS=""`) forces
        // the picker off. Env counts as "set" even when it parses to
        // an empty list (an empty env var is a deliberate off).
        let (thought_levels, thought_levels_set) =
            if let Ok(v) = std::env::var("LLM_THOUGHT_LEVELS") {
                (
                    v.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>(),
                    true,
                )
            } else {
                match file.thought_levels {
                    Some(levels) => (levels, true),
                    None => {
                        let backend = Backend::from_url(&base_url);
                        if backend.is_ollama_native() {
                            (Vec::new(), false)
                        } else {
                            (
                                vec![
                                    "low".to_string(),
                                    "medium".to_string(),
                                    "high".to_string(),
                                    "max".to_string(),
                                ],
                                false,
                            )
                        }
                    }
                }
            };

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
            // Issue #25: compaction. 0 = disabled; None = default 0.75.
            // The summarizer model defaults to the configured model.
            compaction_threshold: file.compaction_threshold.filter(|t| *t >= 0.0 && *t <= 1.0),
            compaction_model: file.compaction_model,
            thought_levels,
            thought_levels_set,
            // Issue #40: filled in by main after the startup probe.
            available_models: Vec::new(),
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
    fn models_allowlist_parses_from_toml() {
        // Issue #48: [llm] models = [...] parses; omitted = None.
        let cfg: ConfigFile =
            toml::from_str("model = 'glm-a'\n[llm]\nmodels = ['glm-a', 'claude-x']\n").unwrap();
        assert_eq!(
            cfg.llm.models,
            Some(vec!["glm-a".to_string(), "claude-x".to_string()])
        );

        let cfg: ConfigFile = toml::from_str("model = 'glm-a'\n[llm]\n").unwrap();
        assert_eq!(cfg.llm.models, None);
    }

    #[test]
    fn thought_levels_safe_default_when_unset() {
        let _env = env_guard();
        std::env::remove_var("LLM_THOUGHT_LEVELS");

        // Default base_url is OpenAI-compatible → safe four.
        let llm = ConfigFile::default().into_llm_config();
        assert_eq!(llm.thought_levels, vec!["low", "medium", "high", "max"]);
        assert!(!llm.thought_levels_set);

        // Ollama-native (no /v1) → none: no reasoning_effort there.
        let cfg = ConfigFile::default();
        let llm = ConfigFile {
            llm: LlmSection {
                base_url: Some("http://localhost:11434".into()),
                ..cfg.llm
            },
        }
        .into_llm_config();
        assert!(llm.thought_levels.is_empty());
        assert!(!llm.thought_levels_set);
    }

    #[test]
    fn thought_levels_explicit_empty_forces_off() {
        let _env = env_guard();
        std::env::remove_var("LLM_THOUGHT_LEVELS");

        // `thought_levels = []` in TOML → off even on OpenAI-compatible.
        let llm = ConfigFile {
            llm: LlmSection {
                thought_levels: Some(Vec::new()),
                ..Default::default()
            },
        }
        .into_llm_config();
        assert!(llm.thought_levels.is_empty());
        assert!(llm.thought_levels_set);

        // `LLM_THOUGHT_LEVELS=""` → off too (deliberate empty).
        std::env::set_var("LLM_THOUGHT_LEVELS", "");
        let llm = ConfigFile::default().into_llm_config();
        assert!(llm.thought_levels.is_empty());
        assert!(llm.thought_levels_set);
        std::env::remove_var("LLM_THOUGHT_LEVELS");
    }

    #[test]
    fn thought_levels_env_and_config_override_default() {
        let _env = env_guard();

        // Config file wins over the default...
        std::env::remove_var("LLM_THOUGHT_LEVELS");
        let llm = ConfigFile {
            llm: LlmSection {
                thought_levels: Some(vec!["none".into(), "xhigh".into()]),
                ..Default::default()
            },
        }
        .into_llm_config();
        assert_eq!(llm.thought_levels, vec!["none", "xhigh"]);
        assert!(llm.thought_levels_set);

        // ...and env wins over the config file.
        std::env::set_var("LLM_THOUGHT_LEVELS", "minimal,ultra");
        let llm = ConfigFile {
            llm: LlmSection {
                thought_levels: Some(vec!["none".into(), "xhigh".into()]),
                ..Default::default()
            },
        }
        .into_llm_config();
        assert_eq!(llm.thought_levels, vec!["minimal", "ultra"]);
        std::env::remove_var("LLM_THOUGHT_LEVELS");
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
