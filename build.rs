//! Build metadata: stamp the git commit short hash so every binary can
//! self-identify at runtime. Reports and logs become actionable without
//! forensics ("which binary was running?" → grep the banner).
//!
//! Best effort by design: a missing or broken git (tarball builds,
//! stripped worktrees, CI checkouts without history) yields
//! "unknown" — the build must never fail over metadata.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads/");

    let hash = git_commit().unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=ACP_BUILD_HASH={hash}");
}

fn git_commit() -> Option<String> {
    let short = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()?;
    if !short.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&short.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}
