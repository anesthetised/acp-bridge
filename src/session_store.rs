//! Disk-backed session persistence (issue #17).
//!
//! SQLite via `rusqlite` with the `bundled` feature — the database
//! compiles into the binary, keeping the single-static-binary
//! deployment property. Sessions are saved **after each completed tool
//! round** (not per turn), so a crash or SIGTERM mid-turn keeps every
//! completed round; only the in-flight round is lost.
//!
//! The store is deliberately boring: one table, one row per session,
//! messages as a JSON payload (the OpenAI-style history is
//! heterogeneous — string/array content, tool_calls — and Value-shaped
//! throughout the engine; forcing it into typed columns buys nothing).
//! Everything queryable lives in typed columns instead.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::Connection;
use serde_json::{json, Value};

/// Default database location following platform conventions:
/// macOS `~/Library/Application Support/acp-bridge/sessions.db`,
/// Linux `~/.local/state/acp-bridge/sessions.db` (XDG state home).
pub fn default_db_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    if cfg!(target_os = "macos") {
        Path::new(&home)
            .join("Library")
            .join("Application Support")
            .join("acp-bridge")
            .join("sessions.db")
    } else {
        let state_home = std::env::var("XDG_STATE_HOME")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(&home).join(".local").join("state"));
        state_home.join("acp-bridge").join("sessions.db")
    }
}

/// A persisted session snapshot. `messages` is the engine's OpenAI-style
/// history verbatim; the replay timeline (issue #17: user chunks, agent
/// chunks, tool_call / tool_call_update pairs) is derived from it at
/// load time rather than stored separately.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionRecord {
    pub session_id: String,
    pub cwd: String,
    pub protocol_version: u16,
    pub title: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub messages: Vec<Value>,
}

#[derive(Debug)]
pub enum StoreError {
    Io(std::io::Error),
    Db(rusqlite::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "session store I/O error: {e}"),
            StoreError::Db(e) => write!(f, "session store database error: {e}"),
            StoreError::Json(e) => write!(f, "session store serialization error: {e}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<rusqlite::Error> for StoreError {
    fn from(e: rusqlite::Error) -> Self {
        StoreError::Db(e)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        StoreError::Json(e)
    }
}

pub struct SessionStore {
    // `Connection` is `Send` but not `Sync`; saves come from the engine
    // loop and loads from the request handler — serialize behind a mutex.
    conn: Mutex<Connection>,
}

impl SessionStore {
    /// Open (creating if needed) the session database at `path`,
    /// creating parent directories and running migrations.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(StoreError::Io)?;
        }
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// Open an in-memory store — for tests.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self, StoreError> {
        // WAL: the per-round saves become appends against a durable
        // log instead of page rewrites in the main file.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                session_id       TEXT PRIMARY KEY,
                cwd              TEXT NOT NULL,
                protocol_version INTEGER NOT NULL,
                title            TEXT,
                created_at       INTEGER NOT NULL,
                updated_at       INTEGER NOT NULL,
                messages         TEXT NOT NULL
            );",
        )?;
        // Future schema migrations branch on this.
        conn.pragma_update(None, "user_version", 1)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Upsert a session snapshot.
    pub fn save(&self, record: &SessionRecord) -> Result<(), StoreError> {
        let messages = serde_json::to_string(&record.messages)?;
        let conn = self.conn.lock().expect("session store lock poisoned");
        conn.execute(
            "INSERT INTO sessions (session_id, cwd, protocol_version, title,
                                   created_at, updated_at, messages)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(session_id) DO UPDATE SET
                cwd = excluded.cwd,
                protocol_version = excluded.protocol_version,
                title = excluded.title,
                updated_at = excluded.updated_at,
                messages = excluded.messages",
            rusqlite::params![
                record.session_id,
                record.cwd,
                record.protocol_version,
                record.title,
                record.created_at,
                record.updated_at,
                messages
            ],
        )?;
        Ok(())
    }

    /// Load a session snapshot; `Ok(None)` when unknown.
    pub fn load(&self, session_id: &str) -> Result<Option<SessionRecord>, StoreError> {
        let conn = self.conn.lock().expect("session store lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT session_id, cwd, protocol_version, title,
                    created_at, updated_at, messages
             FROM sessions WHERE session_id = ?1",
        )?;
        let mut rows = stmt.query(rusqlite::params![session_id])?;
        match rows.next()? {
            None => Ok(None),
            Some(row) => {
                let messages_json: String = row.get(6)?;
                let messages: Vec<Value> = serde_json::from_str(&messages_json)?;
                Ok(Some(SessionRecord {
                    session_id: row.get(0)?,
                    cwd: row.get(1)?,
                    protocol_version: row.get(2)?,
                    title: row.get(3)?,
                    created_at: row.get(4)?,
                    updated_at: row.get(5)?,
                    messages,
                }))
            }
        }
    }

    /// Delete a session snapshot (session/end).
    pub fn delete(&self, session_id: &str) -> Result<(), StoreError> {
        let conn = self.conn.lock().expect("session store lock poisoned");
        conn.execute(
            "DELETE FROM sessions WHERE session_id = ?1",
            rusqlite::params![session_id],
        )?;
        Ok(())
    }

    /// Keep the `keep` most recently updated sessions, delete the rest.
    /// Returns the number of deleted rows.
    pub fn prune(&self, keep: usize) -> Result<usize, StoreError> {
        let conn = self.conn.lock().expect("session store lock poisoned");
        let deleted = conn.execute(
            "DELETE FROM sessions WHERE session_id NOT IN (
                 SELECT session_id FROM sessions
                 ORDER BY updated_at DESC
                 LIMIT ?1
             )",
            rusqlite::params![keep as i64],
        )?;
        Ok(deleted)
    }

    /// List persisted sessions, most recently updated first — for
    /// diagnostics and future `session/list` extension.
    pub fn list(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.conn.lock().expect("session store lock poisoned");
        let mut stmt = conn.prepare(
            "SELECT session_id, cwd, title, updated_at
             FROM sessions ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(json!({
                "sessionId": row.get::<_, String>(0)?,
                "cwd": row.get::<_, String>(1)?,
                "title": row.get::<_, Option<String>>(2)?,
                "updatedAt": row.get::<_, i64>(3)?,
            }))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, updated_at: i64) -> SessionRecord {
        SessionRecord {
            session_id: id.to_string(),
            cwd: "/tmp".to_string(),
            protocol_version: 1,
            title: Some("test session".to_string()),
            created_at: 1000,
            updated_at,
            messages: vec![
                json!({"role": "user", "content": "hello"}),
                json!({"role": "assistant", "content": "world"}),
            ],
        }
    }

    #[test]
    fn save_and_load_roundtrip() {
        let store = SessionStore::open_in_memory().unwrap();
        store.save(&record("sess_a", 1000)).unwrap();

        let loaded = store.load("sess_a").unwrap().expect("session must exist");
        assert_eq!(loaded.session_id, "sess_a");
        assert_eq!(loaded.cwd, "/tmp");
        assert_eq!(loaded.protocol_version, 1);
        assert_eq!(loaded.messages.len(), 2);
        assert_eq!(loaded.messages[0]["content"], "hello");
    }

    #[test]
    fn load_unknown_returns_none() {
        let store = SessionStore::open_in_memory().unwrap();
        assert!(store.load("nope").unwrap().is_none());
    }

    #[test]
    fn save_upserts_by_id() {
        let store = SessionStore::open_in_memory().unwrap();
        store.save(&record("sess_a", 1000)).unwrap();

        let mut updated = record("sess_a", 2000);
        updated.messages = vec![json!({"role": "user", "content": "new"})];
        store.save(&updated).unwrap();

        let loaded = store.load("sess_a").unwrap().unwrap();
        assert_eq!(loaded.updated_at, 2000);
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0]["content"], "new");
    }

    #[test]
    fn delete_removes_session() {
        let store = SessionStore::open_in_memory().unwrap();
        store.save(&record("sess_a", 1000)).unwrap();
        store.delete("sess_a").unwrap();
        assert!(store.load("sess_a").unwrap().is_none());
        // Deleting again is a no-op, not an error.
        store.delete("sess_a").unwrap();
    }

    #[test]
    fn prune_keeps_most_recent() {
        let store = SessionStore::open_in_memory().unwrap();
        for i in 0..10 {
            store.save(&record(&format!("sess_{i}"), i)).unwrap();
        }

        let deleted = store.prune(3).unwrap();
        assert_eq!(deleted, 7);

        let list = store.list().unwrap();
        assert_eq!(list.len(), 3);
        // Newest survive: 9, 8, 7.
        let ids: Vec<&str> = list
            .iter()
            .map(|s| s["sessionId"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["sess_9", "sess_8", "sess_7"]);
    }

    #[test]
    fn prune_with_large_keep_is_noop() {
        let store = SessionStore::open_in_memory().unwrap();
        store.save(&record("sess_a", 1000)).unwrap();
        assert_eq!(store.prune(100).unwrap(), 0);
        assert!(store.load("sess_a").unwrap().is_some());
    }

    #[test]
    fn open_creates_parent_directories() {
        let dir = std::env::temp_dir().join(format!(
            "acp_store_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("nested").join("deeper").join("sessions.db");
        let store = SessionStore::open(&path).unwrap();
        store.save(&record("sess_a", 1)).unwrap();
        assert!(store.load("sess_a").unwrap().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
