#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs/pdf"

mkdir -p "$OUTPUT_DIR"

xelatex -interaction=nonstopmode -halt-on-error -output-directory "$OUTPUT_DIR" "$ROOT_DIR/docs/ari_founder_handbook.tex"
xelatex -interaction=nonstopmode -halt-on-error -output-directory "$OUTPUT_DIR" "$ROOT_DIR/docs/ari_founder_handbook.tex"

echo "Built PDF at $OUTPUT_DIR/ari_founder_handbook.pdf"