#!/usr/bin/env bash
# Rasterize the lab-style HERO svgs to PNG — the bullet-proof "renders anywhere,
# including GitHub markdown" artifact (see RENDERING-NOTE.md: a standalone .svg can
# preview as whitespace in a non-browser renderer; a PNG never does).
#
# The .svg is the SOURCE OF TRUTH (generated + --check-gated from tools/hero_*_gen.py);
# these PNGs are a checked-in convenience raster of it. Re-run after regenerating any
# hero svg:   bash visuals/render-hero-pngs.sh
#
# Uses headless Chrome/Edge at 2x for a crisp raster. No node/puppeteer needed.
set -euo pipefail
cd "$(dirname "$0")"

# the four lab-style hero visuals (basename, no extension)
HEROES=(
  55-hero-statcard
  59-hero-capability-matrix
  60-hero-turntax-curves
  61-hero-benchmark-sweep
)

# locate a Chromium-family browser across platforms
find_chrome() {
  for c in \
    "${CHROME:-}" \
    "/c/Program Files/Google/Chrome/Application/chrome.exe" \
    "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "$(command -v google-chrome 2>/dev/null || true)" \
    "$(command -v chromium 2>/dev/null || true)" \
    "$(command -v chromium-browser 2>/dev/null || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  echo "ERROR: no Chrome/Edge found; set CHROME=/path/to/chrome" >&2
  return 1
}
CHROME_BIN="$(find_chrome)"
echo "using browser: $CHROME_BIN"

for h in "${HEROES[@]}"; do
  svg="$h.svg"
  png="$h.png"
  [ -f "$svg" ] || { echo "skip (no svg): $svg"; continue; }
  # intrinsic height from the svg root, so the viewport matches the art exactly
  hgt="$(grep -oE 'height="[0-9]+"' "$svg" | head -1 | grep -oE '[0-9]+')"
  wid="$(grep -oE 'width="[0-9]+"' "$svg" | head -1 | grep -oE '[0-9]+')"
  "$CHROME_BIN" --headless --disable-gpu --hide-scrollbars \
    --force-device-scale-factor=2 --default-background-color=00000000 \
    --window-size="${wid},${hgt}" \
    --screenshot="$PWD/$png" "$PWD/$svg" >/dev/null 2>&1 || true
  if [ -f "$png" ]; then
    echo "wrote $png  (${wid}x${hgt} @2x)"
  else
    echo "FAILED to render $png" >&2
  fi
done
