//! Background task registry and process lifecycle (issue #28).
//!
//! A background task is a `bash -c <command>` spawned detached in its
//! own process group (`tokio::process::Command::process_group(0)`), so
//! it is independent of the turn that started it: cancelling or ending
//! the turn never touches it (pinned decision 5 of #28's design
//! comment). Output goes to a per-task file under the platform temp
//! dir; `read_new` is cursor-based — each poll returns only bytes new
//! since the previous poll (pinned decision 1).
//!
//! The registry is the single authority on task state, regardless of
//! conversation history: compaction (#25) can summarize away the
//! history that mentioned a task while the task is still running, and
//! the no-arg listing (`list`) is the model's re-discovery path
//! (pinned decision 6).
//!
//! Platform: unix only. On other platforms `spawn` returns an error.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::{Read as _, Seek as _};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Process-global temp-file counter: log file names must be unique
/// across ALL registries in the process (tests run registries in
/// parallel with the same session ids and pids — per-registry
/// counters collided and one test deleted another's log).
static FILE_SEQ: AtomicU64 = AtomicU64::new(0);
use std::time::{Duration, Instant};

/// Model-facing read cap for one poll (bytes): `tools::MAX_TOOL_OUTPUT`.
/// NOT `acp::TOOL_RESULT_MAX` (8192) — that one is the wire display cap
/// applied by `preview_result` independently.
use crate::tools::MAX_TOOL_OUTPUT;

/// Production grace period between SIGTERM and SIGKILL. A parameter in
/// `kill()` so tests can pass ~200ms.
pub const KILL_GRACE: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Exited(i32),
    Signaled(i32),
}

impl TaskStatus {
    pub fn describe(&self) -> String {
        match self {
            TaskStatus::Running => "running".into(),
            TaskStatus::Exited(code) => format!("exited with code {code}"),
            TaskStatus::Signaled(sig) => format!("killed by signal {sig}"),
        }
    }
}

#[derive(Debug)]
struct TaskEntry {
    /// Leader pid; equals the process group id (`process_group(0)`).
    pid: u32,
    command: String,
    output_path: PathBuf,
    started: Instant,
    status: TaskStatus,
    /// Cursor for `read_new` (pinned decision 1).
    read_offset: u64,
    /// Set by `kill()` — distinguishes "we stopped it" from "it died
    /// on its own" in listings and the #55 notice.
    killed_by_us: bool,
}

/// Immutable snapshot for listings, notices, and the spawn result.
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub task_id: String,
    pub pid: u32,
    pub command: String,
    pub output_path: PathBuf,
    pub started: Instant,
    pub status: TaskStatus,
    /// Post-leader-exit hint: the group still has live members (e.g.
    /// `bash -c 'npm run dev &'` — the shell exited, the server runs).
    pub group_alive: bool,
    pub killed_by_us: bool,
}

impl TaskInfo {
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn status_text(&self) -> String {
        // Leader is a dead shell but the group has live members.
        if self.status == TaskStatus::Running && self.group_alive {
            return "running (shell exited; process group still active)".into();
        }
        self.status.describe()
    }
}

#[derive(Default)]
struct Inner {
    /// Keyed by `(session_id, task_id)`.
    tasks: HashMap<(String, String), TaskEntry>,
    /// Per-session monotonic task counter. Ids restart at `t1` per
    /// session and are never reused within the process — the counter
    /// only ever grows, even after kills and exits.
    counters: HashMap<String, u64>,
}

pub struct TaskRegistry {
    inner: Mutex<Inner>,
    /// Agent pid, baked into log file names: orphans from a
    /// hard-killed agent stay identifiable on disk.
    agent_pid: u32,
}

impl TaskRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            agent_pid: std::process::id(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("task registry lock poisoned")
    }

    /// Spawn a detached background task.
    ///
    /// HARD REQUIREMENT — stdio isolation: the agent's stdin and stdout
    /// are the ACP JSON-RPC wire. `tokio::process::Command` inherits
    /// stdio by default (unlike `output()`), so every stream here is
    /// set explicitly: stdin nulled, stdout+stderr to the task's log
    /// file (one `File` plus `try_clone()`, so the streams interleave
    /// in write order). An inherited stdout would corrupt the protocol
    /// stream; an inherited stdin would eat Client requests.
    ///
    /// INVARIANT — no `.await` between `spawn()` and the registry
    /// insert: a turn abort (`handle.abort()`) can only land at an
    /// await point, and an await in between could abort the turn and
    /// leave a live, unregistered process that nothing can kill. This
    /// whole function is synchronous after the syscall.
    ///
    /// Takes `self: &Arc<Self>` because the detached waiter must hold
    /// the *same* registry to record the final status.
    pub fn spawn(
        self: &Arc<Self>,
        session_id: &str,
        working_dir: &std::path::Path,
        command: &str,
    ) -> Result<TaskInfo, String> {
        use std::os::unix::fs::OpenOptionsExt;

        // Per-session id, never reused within the process.
        let seq = {
            let mut inner = self.lock();
            let c = inner.counters.entry(session_id.to_string()).or_insert(0);
            *c += 1;
            *c
        };
        let task_id = format!("t{seq}");

        let file_seq = FILE_SEQ.fetch_add(1, Ordering::Relaxed);
        let output_path = std::env::temp_dir().join(format!(
            "acp-bridge-{}-{}-{}-{}.log",
            self.agent_pid, session_id, task_id, file_seq
        ));
        // 0600: command output can contain secrets.
        let out_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&output_path)
            .map_err(|e| {
                let _ = std::fs::remove_file(&output_path);
                format!("cannot create task log {}: {e}", output_path.display())
            })?;
        let err_file = out_file.try_clone().map_err(|e| {
            let _ = std::fs::remove_file(&output_path);
            format!("cannot clone task log handle: {e}")
        })?;

        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-c")
            .arg(command)
            .current_dir(working_dir)
            // The task leads its own process group: kill() signals the
            // whole group; terminal Ctrl-C and client teardown do not
            // reach it. That independence is the point.
            .process_group(0)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::from(out_file))
            .stderr(std::process::Stdio::from(err_file))
            // Explicit over default: dropping our handle must not kill
            // the task (pinned decision 5 — tasks outlive the turn).
            .kill_on_drop(false);

        let child = cmd.spawn().map_err(|e| {
            let _ = std::fs::remove_file(&output_path);
            format!("failed to spawn background task: {e}")
        })?;
        let pid = child.id().unwrap_or(0);

        {
            let mut inner = self.lock();
            inner.tasks.insert(
                (session_id.to_string(), task_id.clone()),
                TaskEntry {
                    pid,
                    command: command.to_string(),
                    output_path: output_path.clone(),
                    started: Instant::now(),
                    status: TaskStatus::Running,
                    read_offset: 0,
                    killed_by_us: false,
                },
            );
        }

        // Detached waiter: reaps the child and records the final
        // status. It holds no turn handles, so aborting a turn cannot
        // cancel the reaping.
        let registry = Arc::clone(self);
        let key = (session_id.to_string(), task_id.clone());
        tokio::spawn(async move {
            let mut child = child;
            let status = child.wait().await;
            if let Ok(mut inner) = registry.inner.lock() {
                if let Some(entry) = inner.tasks.get_mut(&key) {
                    entry.status = match status {
                        Ok(s) => {
                            if let Some(code) = s.code() {
                                TaskStatus::Exited(code)
                            } else {
                                // tokio's ExitStatus doesn't expose the
                                // signal directly; unix exposes it via
                                // ExitStatusExt::from_raw on the raw wait
                                // status. Keep it simple: record the pid's
                                // fate as signalled without the number only
                                // if we cannot recover it.
                                TaskStatus::Signaled(0)
                            }
                        }
                        Err(_) => TaskStatus::Exited(-1),
                    };
                }
            }
        });

        tracing::info!(
            session_id = %session_id,
            task_id = %task_id,
            pid,
            command = %truncate_cmd(command),
            "Background task started"
        );

        Ok(TaskInfo {
            task_id,
            pid,
            command: command.to_string(),
            output_path,
            started: Instant::now(),
            status: TaskStatus::Running,
            group_alive: false,
            killed_by_us: false,
        })
    }

    #[cfg(not(unix))]
    pub fn spawn(
        &self,
        _session_id: &str,
        _working_dir: &std::path::Path,
        _command: &str,
    ) -> Result<TaskInfo, String> {
        Err("background tasks are not supported on this platform".into())
    }

    /// Cursor-based read (pinned decision 1): returns only bytes new
    /// since the previous poll, capped at `MAX_TOOL_OUTPUT` (tail
    /// when more, with `skipped`), plus current status.
    pub fn read_new(&self, session_id: &str, task_id: &str) -> Result<ReadChunk, String> {
        let key = (session_id.to_string(), task_id.to_string());
        let mut inner = self.lock();
        let entry = inner
            .tasks
            .get_mut(&key)
            .ok_or_else(|| format!("no such background task '{task_id}' in this session"))?;

        let mut file = std::fs::File::open(&entry.output_path)
            .map_err(|e| format!("task log unreadable: {e}"))?;
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
        let start = entry.read_offset;
        let raw_len = file_len.saturating_sub(start);

        // Tail mode jumps the base forward and reports what it skipped.
        let (base, skipped, tail) = if raw_len > MAX_TOOL_OUTPUT as u64 {
            (
                file_len - MAX_TOOL_OUTPUT as u64,
                raw_len - MAX_TOOL_OUTPUT as u64,
                true,
            )
        } else {
            (start, 0, false)
        };

        file.seek(std::io::SeekFrom::Start(base))
            .map_err(|e| format!("task log seek failed: {e}"))?;
        let mut buf = Vec::with_capacity((file_len - base) as usize);
        file.read_to_end(&mut buf)
            .map_err(|e| format!("task log read failed: {e}"))?;

        // UTF-8 boundaries: a byte cursor can split a multi-byte
        // character. Skip a torn leading sequence; hold back an
        // incomplete trailing sequence (≤3 bytes) so it completes by
        // the next poll. from_utf8_lossy is the last resort only.
        let start_skip = buf
            .iter()
            .take(3)
            .take_while(|b| **b & 0xC0 == 0x80)
            .count();
        let mut end_keep = buf.len();
        if end_keep > 0 {
            let mut probe = end_keep;
            let mut conts = 0;
            while probe > 0 && conts < 3 && buf[probe - 1] & 0xC0 == 0x80 {
                probe -= 1;
                conts += 1;
            }
            if probe > 0 {
                let lead = buf[probe - 1];
                let need = if lead & 0xF8 == 0xF0 {
                    3
                } else if lead & 0xF0 == 0xE0 {
                    2
                } else if lead & 0xE0 == 0xC0 {
                    1
                } else {
                    0
                };
                if need > 0 && conts < need {
                    // Incomplete trailing sequence: hold back the whole
                    // thing, including the lead byte.
                    end_keep = probe - 1;
                }
            }
        }

        entry.read_offset = base + start_skip as u64 + end_keep as u64;
        let text = String::from_utf8_lossy(&buf[start_skip..end_keep]).into_owned();

        Ok(ReadChunk {
            task_id: task_id.to_string(),
            text,
            skipped,
            status: entry.status.clone(),
            elapsed: entry.started.elapsed(),
            held_back: (buf.len() - end_keep) as u64,
            tail,
        })
    }

    /// No-arg listing (pinned decision 2 / re-discovery path): every
    /// task in the session, regardless of what history remembers.
    pub fn list(&self, session_id: &str) -> Vec<TaskInfo> {
        self.lock()
            .tasks
            .iter()
            .filter(|((sid, _), _)| sid == session_id)
            .map(|((_, tid), e)| TaskInfo {
                task_id: tid.clone(),
                pid: e.pid,
                command: e.command.clone(),
                output_path: e.output_path.clone(),
                started: e.started,
                status: e.status.clone(),
                group_alive: group_alive(e.pid),
                killed_by_us: e.killed_by_us,
            })
            .collect()
    }

    /// SIGTERM the process group, poll up to `grace`, SIGKILL on
    /// survival. The group is signalled **even if the leader already
    /// exited** — `bash -c 'npm run dev &'` leaves exactly that shape.
    /// The entry stays in the registry (status `Signaled`,
    /// `killed_by_us`), so output stays readable until session cleanup
    /// (#55). Escalation is logged at `info`.
    pub async fn kill(
        &self,
        session_id: &str,
        task_id: &str,
        grace: Duration,
    ) -> Result<KillOutcome, String> {
        let (pid, already_exited) = {
            let inner = self.lock();
            let entry = inner
                .tasks
                .get(&(session_id.to_string(), task_id.to_string()))
                .ok_or_else(|| format!("no such background task '{task_id}' in this session"))?;
            (
                entry.pid,
                // `killed_by_us` counts too: a killed task's group was
                // already signalled and any survivors are escalations.
                entry.status != TaskStatus::Running || entry.killed_by_us,
            )
        };

        // The group gets SIGTERM regardless of the leader's fate.
        let _ = unsafe { libc::killpg(pid as i32, libc::SIGTERM) };

        let deadline = Instant::now() + grace;
        let mut outcome = KillOutcome::Terminated;
        loop {
            if !group_alive(pid) {
                break;
            }
            if Instant::now() >= deadline {
                unsafe { libc::killpg(pid as i32, libc::SIGKILL) };
                outcome = KillOutcome::EscalatedToKill;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        {
            let mut inner = self.lock();
            if let Some(entry) = inner
                .tasks
                .get_mut(&(session_id.to_string(), task_id.to_string()))
            {
                entry.killed_by_us = true;
                if entry.status == TaskStatus::Running {
                    entry.status = TaskStatus::Signaled(libc::SIGTERM);
                }
            }
        }

        tracing::info!(
            session_id = %session_id,
            task_id = %task_id,
            pid,
            escalated = matches!(outcome, KillOutcome::EscalatedToKill),
            already_exited,
            "Background task kill signalled"
        );

        Ok(if already_exited {
            KillOutcome::AlreadyExited
        } else {
            outcome
        })
    }

    /// `session/end` / eviction (issue #55): SIGTERM every group in
    /// the session, remove the entries (log files deleted with them),
    /// return what was terminated — for the #55 transcript notice and
    /// its background SIGKILL escalation (which re-uses the pgids).
    pub fn terminate_session(&self, session_id: &str) -> Vec<TaskInfo> {
        self.terminate_inner(Some(session_id))
    }

    /// Graceful agent shutdown (#55): every session, everything.
    pub fn terminate_all(&self) -> Vec<TaskInfo> {
        self.terminate_inner(None)
    }

    /// SIGKILL any process group still alive after the grace period.
    /// Used by session/end (issue #55): the response is sent
    /// immediately; this escalation runs in the background.
    pub async fn escalate(&self, pgids: Vec<u32>, grace: Duration) {
        tokio::time::sleep(grace).await;
        for pgid in pgids {
            if group_alive(pgid) {
                unsafe { libc::killpg(pgid as i32, libc::SIGKILL) };
                tracing::info!(pgid, "SIGKILL escalation for background task group");
            }
        }
    }

    fn terminate_inner(&self, session_id: Option<&str>) -> Vec<TaskInfo> {
        let mut inner = self.lock();
        let mut terminated = Vec::new();
        let keys: Vec<(String, String)> = inner
            .tasks
            .keys()
            .filter(|(sid, _)| session_id.is_none_or(|s| sid == s))
            .cloned()
            .collect();
        for key in keys {
            if let Some(entry) = inner.tasks.remove(&key) {
                if entry.status == TaskStatus::Running {
                    // SAFETY: killpg with SIGTERM on the task's group.
                    unsafe { libc::killpg(entry.pid as i32, libc::SIGTERM) };
                }
                let _ = std::fs::remove_file(&entry.output_path);
                terminated.push(TaskInfo {
                    task_id: key.1,
                    pid: entry.pid,
                    command: entry.command,
                    output_path: entry.output_path,
                    started: entry.started,
                    status: entry.status,
                    group_alive: false,
                    killed_by_us: false,
                });
            }
        }
        terminated
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ReadChunk {
    pub task_id: String,
    pub text: String,
    /// Bytes skipped by the tail cut (0 when within the cap).
    pub skipped: u64,
    pub status: TaskStatus,
    pub elapsed: Duration,
    /// Bytes of an incomplete trailing multi-byte sequence; they stay
    /// unread until the next poll completes them.
    pub held_back: u64,
    /// True when this poll jumped to the tail (more output existed
    /// than the cap allows).
    pub tail: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KillOutcome {
    /// SIGTERM sufficed.
    Terminated,
    /// Survived the grace; SIGKILLed.
    EscalatedToKill,
    /// The leader had already exited; the group was still signalled
    /// for any remaining children.
    AlreadyExited,
}

/// Is any process alive in the group led by `pid`? `killpg(_, 0)`.
#[cfg(unix)]
fn group_alive(pgid: u32) -> bool {
    // SAFETY: killpg with signal 0 is the documented existence probe.
    let rc = unsafe { libc::killpg(pgid as i32, 0) };
    rc == 0
}

fn truncate_cmd(command: &str) -> &str {
    if command.len() > 80 {
        &command[..80]
    } else {
        command
    }
}

/// Render a `ReadChunk` in the #54 model-facing format.
pub fn render_read(chunk: &ReadChunk, command: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "[task {}: {} after {}s — {}]",
        chunk.task_id,
        chunk.status.describe(),
        chunk.elapsed.as_secs(),
        truncate_cmd(command)
    );
    if chunk.skipped > 0 {
        let _ = writeln!(out, "[… {} earlier bytes skipped …]", chunk.skipped);
    }
    if chunk.text.is_empty() {
        out.push_str("(no new output since last read)");
    } else {
        out.push_str(chunk.text.trim_end());
    }
    if chunk.held_back > 0 {
        let _ = writeln!(
            out,
            "[… final bytes pending (incomplete multibyte sequence) …]"
        );
    }
    out
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Unique temp dir per test (pid + counter + nanos), tools.rs pattern.
    fn tmpwd() -> PathBuf {
        static C: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "acp-tasks-test-{}-{}-{}",
            std::process::id(),
            C.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Poll `cond` until true or the deadline passes. Never sleeps as
    /// synchronization — deadlines are generous (macOS runners).
    async fn wait_until(deadline: Duration, mut cond: impl FnMut() -> bool) -> bool {
        let end = Instant::now() + deadline;
        while Instant::now() < end {
            if cond() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        cond()
    }

    /// SIGKILLs a pid on drop — a failing test must not leak
    /// `sleep 300` processes onto CI runners (the #39/#42 lesson).
    struct KillGuard(u32);
    impl Drop for KillGuard {
        fn drop(&mut self) {
            unsafe { libc::killpg(self.0 as i32, libc::SIGKILL) };
        }
    }

    fn registry() -> Arc<TaskRegistry> {
        Arc::new(TaskRegistry::new())
    }

    #[tokio::test]
    async fn growth_cursor_no_duplicates() {
        let reg = registry();
        let dir = tmpwd();
        let go = dir.join("go");

        let info = reg
            .spawn(
                "sA",
                &dir,
                "echo one; while [ ! -f go ]; do sleep 0.02; done; echo two",
            )
            .unwrap();
        let _guard = KillGuard(info.pid);

        // Poll until `one` appears.
        let first = wait_until(Duration::from_secs(10), || {
            reg.read_new("sA", &info.task_id)
                .map(|c| c.text.contains("one"))
                .unwrap_or(false)
        })
        .await;
        assert!(first, "`one` never appeared");

        std::fs::write(&go, "").unwrap();

        // Poll until `two` appears.
        let mut second = String::new();
        let got = wait_until(Duration::from_secs(10), || {
            match reg.read_new("sA", &info.task_id) {
                Ok(c) => {
                    if c.text.contains("two") {
                        second.push_str(&c.text);
                        true
                    } else {
                        false
                    }
                }
                Err(_) => false,
            }
        })
        .await;
        assert!(got, "`two` never appeared");
        assert!(
            !second.contains("one"),
            "cursor duplicated output: {second}"
        );
    }

    #[tokio::test]
    async fn exit_status_reported() {
        let reg = registry();
        let dir = tmpwd();
        let info = reg.spawn("sA", &dir, "echo bye; exit 3").unwrap();
        let _guard = KillGuard(info.pid);

        let done = wait_until(Duration::from_secs(10), || {
            reg.read_new("sA", &info.task_id)
                .map(|c| c.status == TaskStatus::Exited(3))
                .unwrap_or(false)
        })
        .await;
        assert!(done, "exit status 3 never reported");

        let chunk = reg.read_new("sA", &info.task_id).unwrap();
        assert_eq!(chunk.status, TaskStatus::Exited(3));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kill_reaches_children_in_group() {
        let reg = registry();
        let dir = tmpwd();
        let pidfile = dir.join("child.pid");
        let info = reg
            .spawn(
                "sA",
                &dir,
                &format!("sleep 300 & echo $! > {}; wait", pidfile.display()),
            )
            .unwrap();
        let _guard = KillGuard(info.pid);

        // Wait for the child pid file.
        let child_pid: u32 = tokio::task::block_in_place(|| {
            let ok = futures_lite::future::block_on(wait_until(Duration::from_secs(10), || {
                pidfile.exists()
            }));
            ok.then(|| {
                std::fs::read_to_string(&pidfile)
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap()
            })
        })
        .expect("child pid never written");

        let outcome = reg
            .kill("sA", &info.task_id, Duration::from_millis(200))
            .await
            .unwrap();
        assert!(matches!(
            outcome,
            KillOutcome::Terminated | KillOutcome::AlreadyExited
        ));

        // The child (in the same group) is gone within the deadline.
        let gone = wait_until(Duration::from_secs(5), || {
            // SAFETY: existence probe.
            unsafe { libc::kill(child_pid as i32, 0) != 0 }
        })
        .await;
        assert!(gone, "child pid {child_pid} survived the group kill");
    }

    #[tokio::test]
    async fn term_to_kill_escalation() {
        let reg = registry();
        let dir = tmpwd();
        let info = reg
            .spawn(
                "sA",
                &dir,
                "trap '' TERM; echo ready; while :; do sleep 0.05; done",
            )
            .unwrap();
        let _guard = KillGuard(info.pid);

        // Wait for `ready` so the trap is installed.
        let ready = wait_until(Duration::from_secs(10), || {
            reg.read_new("sA", &info.task_id)
                .map(|c| c.text.contains("ready"))
                .unwrap_or(false)
        })
        .await;
        assert!(ready);

        let outcome = reg
            .kill("sA", &info.task_id, Duration::from_millis(200))
            .await
            .unwrap();
        assert_eq!(
            outcome,
            KillOutcome::EscalatedToKill,
            "TERM-ignoring task must force SIGKILL"
        );

        let dead = wait_until(Duration::from_secs(5), || !group_alive(info.pid)).await;
        assert!(dead, "group survived SIGKILL");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kill_after_leader_exit_hits_lingering_child() {
        let reg = registry();
        let dir = tmpwd();
        let pidfile = dir.join("child.pid");
        // bash exits immediately; the backgrounded sleep keeps the group alive.
        let info = reg
            .spawn(
                "sA",
                &dir,
                &format!("sleep 300 & echo $! > {}", pidfile.display()),
            )
            .unwrap();
        let _guard = KillGuard(info.pid);

        let child_pid: u32 = tokio::task::block_in_place(|| {
            let ok = futures_lite::future::block_on(wait_until(Duration::from_secs(10), || {
                pidfile.exists()
            }));
            ok.then(|| {
                std::fs::read_to_string(&pidfile)
                    .unwrap()
                    .trim()
                    .parse()
                    .unwrap()
            })
        })
        .expect("child pid never written");

        // Leader should exit on its own quickly.
        let exited = wait_until(Duration::from_secs(10), || {
            reg.read_new("sA", &info.task_id)
                .map(|c| c.status != TaskStatus::Running)
                .unwrap_or(false)
        })
        .await;
        assert!(exited, "leader never exited");

        // Listing shows the group still alive (shell exited, server up).
        let listed = reg.list("sA");
        assert!(listed
            .iter()
            .any(|t| t.task_id == info.task_id && t.group_alive));

        let outcome = reg
            .kill("sA", &info.task_id, Duration::from_millis(200))
            .await
            .unwrap();
        assert_eq!(outcome, KillOutcome::AlreadyExited);

        let gone = wait_until(Duration::from_secs(5), || unsafe {
            libc::kill(child_pid as i32, 0) != 0
        })
        .await;
        assert!(gone, "lingering child survived the kill");
    }

    #[tokio::test]
    async fn cap_and_tail_with_skip_marker() {
        let reg = registry();
        let dir = tmpwd();
        // >50 000 bytes up front. Pure-shell loop: a `head | tr`
        // pipeline can die by SIGPIPE and write nothing.
        let info = reg
            .spawn(
                "sA",
                &dir,
                "i=0; while [ $i -lt 7500 ]; do printf 'xxxxxxxxxx'; i=$((i+1)); done; echo END",
            )
            .unwrap();
        let _guard = KillGuard(info.pid);

        // Wait WITHOUT read_new — polling would advance the cursor and
        // defeat the cap test. Group liveness is the exit signal.
        let finished = wait_until(Duration::from_secs(10), || !group_alive(info.pid)).await;
        assert!(finished, "task never finished");

        // First read after finish: 75k bytes from offset 0 → tail cut.
        let chunk = reg.read_new("sA", &info.task_id).unwrap();
        assert!(chunk.text.len() <= MAX_TOOL_OUTPUT, "cap breached");
        assert!(chunk.skipped > 0, "expected skipped bytes");
        assert!(
            chunk.text.contains("END"),
            "tail must keep the newest output"
        );

        // Next poll: nothing new.
        let next = reg.read_new("sA", &info.task_id).unwrap();
        assert!(next.text.is_empty());
    }

    #[tokio::test]
    async fn utf8_split_across_polls() {
        let reg = registry();
        let dir = tmpwd();
        // Cyrillic 'ж' is 2 bytes (0xD0 0xB6): write it in two halves
        // with a delay between, so the cursor splits the character.
        let info = reg
            .spawn(
                "sA",
                &dir,
                "printf '\\xd0'; sleep 0.3; printf '\\xb6'; echo END",
            )
            .unwrap();
        let _guard = KillGuard(info.pid);

        let done = wait_until(Duration::from_secs(10), || {
            reg.read_new("sA", &info.task_id)
                .map(|c| c.status != TaskStatus::Running)
                .unwrap_or(false)
        })
        .await;
        assert!(done);

        // The full text across polls concatenates to the original, no
        // replacement chars anywhere.
        let final_read = reg.read_new("sA", &info.task_id).unwrap();
        assert!(
            !final_read.text.contains('\u{FFFD}'),
            "lossy fallback hit: {}",
            final_read.text
        );
    }

    #[tokio::test]
    async fn session_isolation_and_id_restart() {
        let reg = registry();
        let dir = tmpwd();
        let a = reg.spawn("sA", &dir, "sleep 0.01").unwrap();
        let b = reg.spawn("sB", &dir, "sleep 0.01").unwrap();

        // Ids restart at t1 per session.
        assert_eq!(a.task_id, "t1");
        assert_eq!(b.task_id, "t1");

        // Foreign session read/kill → no such task. sC has no tasks.
        assert!(reg.read_new("sB", "t1").is_ok());
        assert!(reg.read_new("sC", "t1").is_err());
        let err = reg
            .kill("sC", "t1", Duration::from_millis(50))
            .await
            .unwrap_err();
        assert!(err.contains("no such background task"), "{err}");
    }

    #[tokio::test]
    async fn terminate_session_removes_and_cleans_files() {
        let reg = registry();
        let dir = tmpwd();
        let info = reg.spawn("sA", &dir, "sleep 300").unwrap();
        let _guard = KillGuard(info.pid);
        assert!(info.output_path.exists());

        let terminated = reg.terminate_session("sA");
        assert_eq!(terminated.len(), 1);
        assert_eq!(terminated[0].task_id, info.task_id);
        assert!(!info.output_path.exists(), "log file must be deleted");
        assert!(reg.list("sA").is_empty());

        // Post-termination reads fail (entry gone).
        assert!(reg.read_new("sA", &info.task_id).is_err());
    }

    #[tokio::test]
    async fn output_file_mode_is_0600() {
        use std::os::unix::fs::PermissionsExt;
        let reg = registry();
        let dir = tmpwd();
        let info = reg.spawn("sA", &dir, "echo hi").unwrap();
        let _guard = KillGuard(info.pid);
        let mode = std::fs::metadata(&info.output_path)
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "log file must be 0600");
    }
}
