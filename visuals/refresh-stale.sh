#!/usr/bin/env bash
# One-shot: re-render the deck SVGs that preview as blank whitespace in non-browser
# viewers (mermaid default htmlLabels:true → all labels in <foreignObject>, text=0,
# root width="100%") to the RENDERING-NOTE.md recipe:
#   SVG: htmlLabels:false (labels become real <text>/<tspan>) + pin width/height
#        from the viewBox so the image always has an intrinsic size.
#   PNG: chromium rasterizes the foreignObject HTML → text baked into pixels
#        (the "never whitespace anywhere" canonical artifact).
#
# Idempotent: only re-renders bases passed as args (or the STALE list below).
# Run from a shell with node/npx on PATH and the puppeteer chromium cache present.
set -euo pipefail
cd "$(dirname "$0")"
export PUPPETEER_CACHE_DIR="${PUPPETEER_CACHE_DIR:-$HOME/.cache/puppeteer}"

echo '{"flowchart":{"htmlLabels":false},"securityLevel":"loose"}' > cfg.json
MMDC=(npx -y @mermaid-js/mermaid-cli -p .puppeteer.json -b white)
SCALE="${SCALE:-2}"

# The whitespace-defect set: text=0, all-foreignObject, root width="100%"
# (measured 2026-07-03). The newer 15-24 / 39-67 charts already carry real text.
STALE=(
  00-legend 01-master 02-boundaries 03-syscall-mapping 04-syscall-path
  05-preflight 06-context-mmu 07-kv-hierarchy 08-residency-fsm 09-shared-prefix
  10-compute-tiers 11-stewards 12-rsi 13-kpis 14-economics
  25-three-lane-stacking-map 26-claims-vs-reality 27-sound-not-complete-not-precise
  29-session-core-dump 30-readmit-ladder 31-machine-facts 34-fleet-advantage-activation
  35-permission-options-map 37-permission-two-gate-path 38-api-host-bridge-proof-stack
  51-webbench-prefill-elimination 52-getting-started-journey
  68-adoption-seat-map 69-adoption-onramp 70-adoption-value-split
  71-adoption-shape-picker 72-adoption-perf-gate
  fleet-sweep-01-two-knobs fleet-sweep-02-shared-whiteboard fleet-sweep-03-honest-test
  fleet-sweep-04-the-law fleet-sweep-05-the-catch
  recall-01-coredump recall-02-survives-boundary recall-03-rung4-gates
  recall-04-rosetta recall-07-ladder recall-08-toctou-rescreen
  recall-09-floor-moved recall-10-taint recall-11-driver-stack
)

if [ "$#" -gt 0 ]; then
  targets=("$@")
else
  targets=("${STALE[@]}")
fi

for base in "${targets[@]}"; do
  base="${base%.mmd}"; base="${base%.svg}"
  if [ ! -f "$base.mmd" ]; then echo "skip  $base (no .mmd)" >&2; continue; fi
  echo "render $base"
  # SVG with real text
  "${MMDC[@]}" -c cfg.json -i "$base.mmd" -o "$base.svg"
  # Pin width/height from the viewBox (replace the lone width="100%").
  python3 - "$base.svg" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
m = re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', s)
if m:
    w = str(round(float(m.group(1))))
    h = str(round(float(m.group(2))))
    # Only touch the root <svg ... width="100%"> once.
    s2 = re.sub(r'(<svg[^>]*?)width="100%"', rf'\1width="{w}" height="{h}"', s, count=1)
    if s2 != s:
        open(p, "w", encoding="utf-8").write(s2)
        print(f"  pinned {w}x{h}")
PY
  # PNG (rasterized, never whitespace)
  "${MMDC[@]}" -c cfg.json -i "$base.mmd" -o "$base.png" -s "$SCALE"
done
echo "done."
