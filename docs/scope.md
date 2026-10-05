# acp-bridge Scope

This document is the source of truth for what acp-bridge supports, what it
supports with caveats, and what it deliberately does not implement. Use it
to set expectations when a Client picks acp-bridge from the ACP registry or
launches it as a subprocess.

acp-bridge positions itself as a **minimal ACP adapter for local AI**, not
a full agent runtime. Anything that would require an external auth
provider or a hosted control-plane surface is intentionally out of scope.

## Protocol versions

acp-bridge **supports both ACP v1 and v2** on the same code base. The
Client negotiates the version at `init` time by sending its preferred
`protocolVersion`. acp-bridge picks the highest version both sides support
and emits the corresponding wire shape for every notification and method.

| Client | Wire emitted by acp-bridge |
|---|---|
| Zed, JetBrains, ACP UI, Meuxe, ACP Inspector, Codex CLI adapter (v1) | ACP v1 |
| OpenCode v2 preview, future v2-only Clients | ACP v2 |

The two wire surfaces are not a separate feature flag — every notification
goes through `acp::notify_*_for(version, …)` dispatchers. v1 Clients
see no behavior change from previous versions; v2 Clients receive
spec-compliant v2 payloads.

## Supported

### Methods (v1 + v2 baseline)

| ACP surface | Status | Notes |
|---|---|---|
| `initialize` | ✅ | Negotiates v1/v2; emits the corresponding `agentCapabilities` + `agentInfo` shape (v1) or `capabilities` + `info` shape (v2) |
| `session/new` | ✅ | Multi-session with per-session conversation history. `mcpServers` accepted but ignored (acp-bridge does not relay MCP) |
| `session/prompt` | ✅ | The tool loop runs on streamed rounds (issue #11): reasoning and answer text arrive as incremental chunk notifications; streamed tool-call fragments are assembled and executed like non-streaming calls; backends that reject `stream: true` fall back to non-streaming; a stream that dies mid-round fails the turn. v1 final response carries `stopReason` (`end_turn` / `max_turn_requests`). v2 final response carries `messageId` and the turn end is reported via `state_update` (per schema) |
| `session/cancel` notification | ✅ | Acknowledged with a log line; in-flight cancellation is not yet implemented |
| `session/end` (v1) / `session/close` (v2 baseline) | ✅ | Removes the session and frees its history. Both methods share the same implementation |
| `session/list` (v2 baseline) | ✅ | Returns currently active sessions as `{sessions: [{sessionId, cwd}], nextCursor: null}` |
| Streaming `agent_message_chunk` | ✅ | Typed `content: {type: "text", text: …}`. The engine's tool loop runs on streamed rounds, so answer text arrives incrementally. v1 and v2 use the same discriminator |
| `session/load` / `session/resume` | ✅ | Sessions persist in SQLite (bundled, WAL) — one row per session, saved after every completed tool round, pruned to last 100 at startup. `session/load` restores and **replays the full timeline** (user chunks → `tool_call`/`tool_call_update` pairs → agent chunks) before responding, per spec; `session/resume` restores without replay. `loadSession` + `sessionCapabilities.resume` advertised; `ACP_SESSION_DB` overrides the DB path, `ACP_PERSISTENCE=off` disables and reverts to the `no_persistence` rejections |
| `agent_thought_chunk` | ✅ | Typed `content: {type: "text", text: ""}` — emitted so Clients render the thought bubble. Model reasoning streams delta-by-delta while thinking (backend `message.reasoning_content` / `message.thinking` / `delta.reasoning_content`); display-only, never appended to session history |
| `tool_call` (v1) / `tool_call_update` (v1 + v2) | ✅ | Carries `toolCallId`, `name`, human-readable `title`, `kind`, `status`, plus optional specifics: `rawInput` (model's arguments), `locations` (path-carrying tools), and on completion `rawOutput` + text `content` (preview-capped). Real `failed` status when the tool errored. v1 Clients see `tool_call`; v2 Clients see only `tool_call_update` with `status: "in_progress"` |
| `diff` content blocks on file mutations | ✅ | `edit` / `write_file` completions append a spec `diff` content block (`path` absolute, `oldText` → `newText`; `oldText: null` for new files, snippet-based for `edit`) so Clients render real diffs (issue #26). Size-capped — oversized changes emit the text block only |
| `plan` (v1) / `plan_update` (v2) | ✅ | `plan_update` carries `plan: {type: "items", planId, entries[]}` so v2 Clients can track multiple plans |
| `available_commands_update` (v1 + v2) | ✅ | Slash-command menu: `/read`, `/ls`, `/search`, `/edit`, `/shell` |
| `session_info_update` (v1 + v2) | ✅ | Title defaults to cwd basename, updated to first line of user prompt after each turn |
| `usage_update` (v1 + v2) | ✅ | Estimated `used` (chars / 4 across session history) + `size` from `LLM_MODEL_CONTEXT` (default 32768) |
| `state_update` (v2 only) | ✅ | Emitted at end of each prompt with `state: "idle"` + `stopReason`. v2 Clients ignore unknown discriminators so a v1 Client never sees it |
| Request ID echo | ✅ | Numeric AND string/UUID request ids are echoed verbatim — fixes the issue where Meuxe's UUID ids were silently dropped |
| Stdout framing | ✅ | Newline-delimited JSON, one object per line |

### Built-in tools

| Tool | Description |
|---|---|
| `read_file` | Read a file (sandboxed, max 1 MB) |
| `list_dir` | List directory contents (max 4 levels, max 200 entries) |
| `search_code` | Grep for a literal string in files under the working dir (max 50 matches). Symlinks are skipped; depth is bounded |
| `write_file` | Create or overwrite a file (max 5 MB). Sandbox rejects `..` escapes, symlinked ancestors, and symlinked final components — resolved target must stay inside the working dir. Result reports the absolute path written |
| `edit` | Replace a unique substring in a file. Fails on missing / ambiguous / empty matches |
| `web_fetch` | HTTP/HTTPS GET with HTML reduction. **Off by default** — requires `LLM_WEB_ALLOWLIST`. Redirects are limited to 5 hops and re-validated against the allowlist on every hop |
| `bash` | Run a bash command. Note: no OS-level sandbox, no timeout enforcement (see "Things acp-bridge is not") |
| `git_status` | Compact `git status --short --branch` |
| `git_diff` | Unified diff, optional `path` filter, optional `staged: true` (`--cached`) |
| `git_log` | One-line-per-commit log, `max_count` clamped to 1–200 |
| `git_commit` | Stage listed paths (or `git add -u`) and commit with the supplied message |

## Supported with caveats

| ACP surface | Caveat | How to enable |
|---|---|---|
| Image content blocks (`ContentBlock::Image`) | Off by default. Turning it on without a vision-capable backend causes Clients (Meuxe, ACP UI, …) to forward images that the LLM cannot parse | Set `LLM_SUPPORTS_IMAGE=true` or `[llm].supports_image = true` |
| `session/prompt` with audio / embedded resource content | Not parsed; audio/resource blocks are silently dropped from the prompt | Out of scope; document if you need it |
| `mcpServers` on `session/new` | Accepted for spec compatibility but **not relayed** to the underlying LLM. The agent uses its own built-in tool set | Future work; tracked but not scheduled |

## Deliberately not implemented

These methods are part of the ACP surface but acp-bridge intentionally
does not implement them. When called, the agent responds with `code: -32601`
(JSON-RPC `Method not found`) carrying a `data: {reason: …}` field that
explains *why*, so Clients can log a clear root cause instead of a generic
error.

| Method | Reason | Error code |
|---|---|---|
| `session/delete` (v2 optional) | No persistence to delete from; use `session/close` instead | `-32601` (`data.reason: "not_implemented"`) |
| `session/set_mode` | `session/new` does not return a `modes` array | `-32602` (`data.reason: "no_modes"`) |
| `session/load` / `session/resume` with cwd mismatch | Persisted sessions re-anchor to their original working dir; silently re-anchoring would break tool sandboxing | `-32001` (message names both cwds) |
| `session/load` / `session/resume` with persistence disabled (`ACP_PERSISTENCE=off`) | Legacy opt-out behavior | `-32001` (`data.reason: "no_persistence"`) |
| `auth/login` | `initialize` returns `authMethods: []` | `-32601` (`data.reason: "no_auth_methods"`) |
| `auth/logout` | Same | `-32601` (`data.reason: "no_auth_methods"`) |
| `fs/read_text_file`, `fs/write_text_file` | acp-bridge never requests client-side file operations | `-32601` (`data.reason: "agent_does_not_call_client_fs"`) |
| `terminal/*` | Same | `-32601` (`data.reason: "agent_does_not_call_client_terminal"`) |

The Client should consult `agentCapabilities` (v1) / `capabilities`
(v2) to detect these gaps before calling — spec-compliant Clients
(Zed, JetBrains, ACP Inspector) already do this. The error responses
above are a safety net for Clients that don't.

## Things acp-bridge is not

- **Not a hosted session database.** Sessions persist locally in SQLite
  (issue #17) — they survive agent restarts on the same machine. There
  is no hosted control plane, no cross-machine sync, no sharing between
  different Client instances beyond what the local store allows.

### Why persistence lives in this layer (design rationale)

The ACP protocol gives Clients **no method to inject conversation
history** — the only restore mechanisms are `session/load` /
`session/resume`, and both place the burden on the Agent. The state
being persisted is also Agent-owned: the assembled model context
(system prompt, tool results in backend shape), which the Client never
sees and could not reconstruct from its own transcript. Within the
protocol as specified, agent-side persistence is therefore the only
possible home for restore. Revisit if ACP ever adds client-side
history injection.
- **Not an auth provider.** No login flow, no token storage, no user
  identity. If you need per-user sessions, route through a Client that
  gates on its own identity.
- **Not an MCP server.** acp-bridge ignores `mcpServers` in `session/new`.
  Built-in tools are the only tool surface.
- **Not a v1-only agent.** acp-bridge supports both ACP v1 and v2
  on the same code base. See "Protocol versions" above.
- **Not a web transport.** Transport is stdio JSON-RPC only — there is no
  HTTP / WebSocket / SSE surface. Use a proxy (e.g. `acp2api`) if you
  need to expose acp-bridge to a browser.
- **Not a hardened `bash` sandbox.** `bash` runs whatever it is told with
  the working dir as cwd. There is no `seccomp` / `firecracker` / `bubblewrap`
  boundary. Don't run acp-bridge as root; don't trust untrusted prompts.

## Roadmap

- **`session/cancel` cancellation.** Acknowledge the notification and
  propagate cancellation into the in-flight LLM request (currently the
  in-flight prompt is allowed to run to completion).
- **Optional MCP relay.** Opt-in via config to honor `mcpServers` in
  `session/new`.
- **ACP v2 `agent_message` upsert.** Currently acp-bridge emits the
  `agent_message_chunk` variant only. The v2 RFD allows chunk-only
  emission; an upsert path can be added later without breaking
  spec-compliant Clients.
- **ACP v2 `terminal_update` / `terminal_output_chunk`.** Agent-owned
  terminal output is not currently emitted — acp-bridge returns tool
  results as text content.

## How to verify a Client works against this scope

The `tests/clients/` integration tests spawn acp-bridge as a subprocess
and drive it through the protocol sequences emitted by named Clients.
Init payloads are pulled directly from each Client's source code, not
guessed:

- `tests/clients/zed_style.rs` — Zed / JetBrains style: full client
  capabilities declared, image + text in the prompt, `session/load` and
  `session/set_mode` called and gracefully rejected.
- `tests/clients/inspector_style.rs` — ACP Inspector style: minimal
  capabilities, request-id fuzz (numeric + UUID + null), prompt with
  `ContentBlock::ResourceLink`.
- `tests/clients/minimal_style.rs` — Codex CLI adapter style: minimal
  capabilities, bare tool-call round-trip, session-end at finish.
- `tests/clients/notifications.rs` — spec notification shapes
  (`available_commands_update`, `session_info_update`, `usage_update`).
- `tests/clients/protocol_version.rs` — v1 / v2 wire-shape dispatch.

Run them with:

```sh
cargo test --test clients -- --test-threads=1
```

These tests are the contract. If you find a real Client whose
expectations diverge from what the tests assert, the test is wrong, not
the Client — open an issue and we will update both.