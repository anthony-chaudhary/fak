#!/usr/bin/env bash
# Render fleet visuals from Mermaid (.mmd) sources to SVG + PNG.
#
# WHITE BACKGROUND IS THE DEFAULT (operator preference, 2026-06-17): a transparent
# SVG looks broken on dark backgrounds (Slack / email / dark-mode docs). The mermaid
# CLI only takes the canvas background as a flag (-b), not via .puppeteer.json, so this
# script is where the default lives. Override with: BG=transparent ./render.sh ...
#
# Usage:
#   ./render.sh                 # render every *.mmd in this dir
#   ./render.sh 06-context-mmu  # render one figure (with or without the .mmd suffix)
#   BG=transparent ./render.sh  # opt out of the white default
#
# Run from a shell with node/npx on PATH (Git Bash on Windows is fine).
set -euo pipefail
cd "$(dirname "$0")"

BG="${BG:-white}"
SCALE="${SCALE:-2}"
MMDC=(npx -y @mermaid-js/mermaid-cli -p .puppeteer.json -b "$BG")

if [ "$#" -gt 0 ]; then
  targets=("$@")
else
  targets=(*.mmd)
fi

for src in "${targets[@]}"; do
  base="${src%.mmd}"
  if [ ! -f "$base.mmd" ]; then
    echo "skip  $base (no $base.mmd)" >&2
    continue
  fi
  echo "render $base  (bg=$BG)"
  "${MMDC[@]}" -i "$base.mmd" -o "$base.svg"
  "${MMDC[@]}" -i "$base.mmd" -o "$base.png" -s "$SCALE"
done
