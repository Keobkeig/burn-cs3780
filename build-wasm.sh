#!/usr/bin/env bash
# Build the browser bundle.
#
# By default it writes into the sibling portfolio checkout, which is where the
# demos are served from. Override with an argument or $BURN_WASM_OUT:
#
#   ./build-wasm.sh                      # ../richie-portfolio/static/wasm/burn
#   ./build-wasm.sh /path/to/out
#   BURN_WASM_OUT=/path/to/out ./build-wasm.sh
#
# /usr/bin goes first because a Nix gcc can shadow Apple clang on macOS, and
# host proc-macros fail to link against it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUT="$(dirname "$REPO")/richie-portfolio/static/wasm/burn"
OUT="${1:-${BURN_WASM_OUT:-$DEFAULT_OUT}}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PATH=/usr/bin:$PATH wasm-pack build \
	--release \
	--target web \
	--out-dir "$TMP" \
	--out-name burn \
	--no-typescript \
	--no-pack

mkdir -p "$OUT"
cp "$TMP/burn.js" "$TMP/burn_bg.wasm" "$OUT/"

echo "wrote to $OUT"
ls -lh "$OUT"
