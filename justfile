# Local gate mirroring CI exactly: Format -> Clippy(-D warnings) -> Test.
# Run before every push: `just gate`. CI gates everything on Format
# first, so a missed local `cargo fmt` wastes a full CI cycle (seen on
# PR #43) — this makes the local order impossible to get wrong.

default:
    @just --list

# reformat in place (CI checks with --check; run this before committing)
fmt:
    cargo fmt --all

# clippy with CI's deny-warnings flag
lint:
    RUSTFLAGS="-D warnings" cargo clippy --all-targets

# full test suite
test:
    cargo test

# full gate: fmt, then clippy -D warnings, then tests. CI-identical order.
gate:
    just fmt
    just lint
    just test
