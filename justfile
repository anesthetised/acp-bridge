# Local gate mirroring CI exactly: Format -> Clippy(-D warnings) -> Test.
# Run before every push: `just gate`. CI gates everything on Format
# first, so a missed local `cargo fmt` wastes a full CI cycle (seen on
# PR #43) — this makes the local order impossible to get wrong.

default:
    @just --list

gate:
    cargo fmt --all
    RUSTFLAGS="-D warnings" cargo clippy --all-targets
    cargo test
