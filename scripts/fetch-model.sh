#!/usr/bin/env bash
# fetch-model.sh — one command to produce the real model weights the in-kernel
# engine (`fak serve --engine inkernel`, or `--engine inkernel` on any verb) loads
# from FAK_MODEL_DIR. See ../GETTING-STARTED.md §4b.
#
# It wraps internal/model/export_oracle.py: creates a throwaway Python venv, installs
# torch/transformers/numpy (CPU is enough), downloads the HF checkpoint, and exports
# it to a flat f32 blob + manifest + config (+ the per-layer oracle the model tests
# verify against) under internal/model/.cache/<name>/.
#
# Usage:
#   scripts/fetch-model.sh            # export the default SmolLM2-135M-Instruct
#   scripts/fetch-model.sh --check    # preflight only: report python + the plan
#
# Knobs (env):
#   FAK_EXPORT_MODEL   HF model id            (default HuggingFaceTB/SmolLM2-135M-Instruct)
#   FAK_MODEL_DIR      output dir             (default <fak>/internal/model/.cache/<name>)
#   FAK_EXPORT_VENV    venv dir              (default <fak>/.cache/export-venv)
#   PYTHON             python interpreter     (default python3, then python)
set -euo pipefail

MODEL="${FAK_EXPORT_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"

# Resolve the fak module root (this script lives in <fak>/scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_PY="$ROOT/internal/model/export_oracle.py"
REQ="$ROOT/scripts/requirements-export.txt"

# Default output dir derives from the model's short name so two models don't clobber
# each other; the docs' canonical path is internal/model/.cache/smollm2-135m.
default_name() {
  local n="${MODEL##*/}"          # strip the HF org prefix
  n="$(printf '%s' "$n" | tr '[:upper:]' '[:lower:]')"
  case "$n" in
    smollm2-135m-instruct) echo "smollm2-135m" ;;
    *)                     echo "$n" ;;
  esac
}
OUT="${FAK_MODEL_DIR:-$ROOT/internal/model/.cache/$(default_name)}"
# Default the venv under the already-git-ignored model cache so it is never committed.
VENV="${FAK_EXPORT_VENV:-$ROOT/internal/model/.cache/export-venv}"

# Pick a python.
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "fetch-model: need python3 (set PYTHON=/path/to/python)" >&2
  exit 1
fi

if [[ ! -f "$EXPORT_PY" ]]; then
  echo "fetch-model: cannot find $EXPORT_PY — run this from inside the fak/ tree" >&2
  exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "fetch-model preflight"
  echo "  python : $("$PY" --version 2>&1)  ($(command -v "$PY"))"
  echo "  model  : $MODEL"
  echo "  out    : $OUT"
  echo "  venv   : $VENV"
  echo "  export : $EXPORT_PY"
  echo "(run without --check to create the venv, download, and export)"
  exit 0
fi

# venv + deps (idempotent: reused on re-run).
if [[ ! -d "$VENV" ]]; then
  echo "fetch-model: creating venv at $VENV" >&2
  "$PY" -m venv "$VENV"
fi
# venv layout differs by platform: POSIX uses bin/, Windows uses Scripts/.
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif [[ -f "$VENV/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/Scripts/activate"
else
  echo "fetch-model: venv at $VENV has no activate script" >&2
  exit 1
fi

echo "fetch-model: installing export deps (CPU torch + transformers + numpy)..." >&2
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "$REQ"

mkdir -p "$OUT"

# export_oracle.py pins HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 via os.environ.setdefault,
# which would refuse to download on a fresh machine. Pre-set them to 0 so the first
# from_pretrained() is allowed to fetch; the HF cache makes later runs offline-safe.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

echo "fetch-model: exporting $MODEL -> $OUT" >&2
# The HF download happens inside export_oracle.py (transformers from_pretrained,
# which itself verifies the LFS sha256). Catch its failure here so a down / renamed /
# rate-limited source ends in one clear, actionable message instead of a bare die.
if ! python "$EXPORT_PY" --model "$MODEL" --out "$OUT"; then
  echo "fetch-model: export failed for '$MODEL'." >&2
  echo "  The HuggingFace download may be down, rate-limited, or the model id is wrong." >&2
  echo "  Re-run when online (the HF cache makes repeats offline-safe), or point" >&2
  echo "  FAK_EXPORT_MODEL at a local checkpoint dir or an alternate HF id." >&2
  exit 1
fi

# Post-export integrity: manifest.json records nbytes per tensor written into
# weights.f32, so a truncated / partial export shows up as a size mismatch. Verify
# before declaring done — delete the bad blob and exit non-zero rather than leave a
# corrupt artifact the engine would only trip over later.
echo "fetch-model: verifying exported weights against manifest..." >&2
python - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
man_path = os.path.join(out, "manifest.json")
w_path = os.path.join(out, "weights.f32")
try:
    with open(man_path) as f:
        manifest = json.load(f)
except OSError as e:
    sys.stderr.write(f"fetch-model: INTEGRITY FAIL - cannot read {man_path}: {e}\n")
    sys.exit(1)
expected = sum(int(v["nbytes"]) for v in manifest.values())
try:
    actual = os.path.getsize(w_path)
except OSError as e:
    sys.stderr.write(f"fetch-model: INTEGRITY FAIL - cannot stat {w_path}: {e}\n")
    sys.exit(1)
if actual != expected:
    try:
        os.remove(w_path)
    except OSError:
        pass
    sys.stderr.write(
        f"fetch-model: INTEGRITY FAIL - weights.f32 is {actual} bytes but manifest.json "
        f"sums to {expected} ({len(manifest)} tensors). Export was truncated/corrupt; "
        f"deleted the bad blob. Delete {out} and re-run.\n")
    sys.exit(1)
sys.stderr.write(
    f"fetch-model: verified weights.f32 ({actual} bytes) matches manifest "
    f"({len(manifest)} tensors).\n")
PY

echo
echo "fetch-model: done. To serve the real weights:"
echo "  export FAK_MODEL_DIR=\"$OUT\""
echo "  ./fak serve --engine inkernel --model $(default_name)"
