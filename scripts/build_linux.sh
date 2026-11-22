#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
OUT="dist/percystar_v${VERSION}_linux_x86_64"
TARGET_CPU="${TARGET_CPU:-x86-64-v3}"

rustup target add x86_64-unknown-linux-gnu >/dev/null

RUSTFLAGS="-C target-cpu=${TARGET_CPU}" cargo build --release --locked --target x86_64-unknown-linux-gnu

mkdir -p dist
cp target/x86_64-unknown-linux-gnu/release/PercyStar dist/PercyStar
command -v strip >/dev/null 2>&1 && strip dist/PercyStar || true

rm -f "${OUT}.zip"
zip -j -q "${OUT}.zip" dist/PercyStar
echo "Built ${OUT}.zip ($(du -sh "${OUT}.zip" | cut -f1))"
