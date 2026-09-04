#!/usr/bin/env bash
# Build the browser bundle and drop it into the portfolio's static assets.
#
# /usr/bin first: a Nix gcc shadows Apple clang on this machine, and host
# proc-macros fail to link against it.
set -euo pipefail

OUT="${1:-/Users/Programming/richie-portfolio/static/wasm/burn}"
TMP="$(mktemp -d)"

PATH=/usr/bin:$PATH wasm-pack build \
	--release \
	--target web \
	--out-dir "$TMP" \
	--out-name burn \
	--no-typescript \
	--no-pack

mkdir -p "$OUT"
cp "$TMP/burn.js" "$TMP/burn_bg.wasm" "$OUT/"
rm -rf "$TMP"

ls -lh "$OUT"
