#!/usr/bin/env bash
# fak bench — subsystem-latency walkthrough runner (issue #319).
#
# Runs the in-process-vs-spawned-hook fusion check against the PREBUILT fak binary
# (no Go toolchain needed at run time), then prints the witness gate and where the
# headline number lives. See README.md for the full reading.
#
# Usage:
#   FAK_BIN=/path/to/fak ./examples/bench-latency/run.sh
#   ./examples/bench-latency/run.sh                 # falls back to `fak` on PATH
#
# Must be run from the repo root so the cwd-relative `testdata/tau2` suite resolves.
set -euo pipefail

FAK_BIN="${FAK_BIN:-fak}"
SUITE="${SUITE:-tau2-smoke}"
N="${BASELINE_N:-100}"
if [ -n "${OUT_DIR:-}" ]; then
  OUT_DIR="$OUT_DIR"
else
  OUT_DIR="$(mktemp -d)"
  cleanup() { rm -rf "$OUT_DIR"; }
  trap cleanup EXIT
fi
OUT="${OUT_DIR}/report.json"

if ! command -v "${FAK_BIN}" >/dev/null 2>&1; then
  echo "error: fak binary not found (set FAK_BIN=/path/to/fak or put fak on PATH)" >&2
  exit 1
fi

if [ ! -f "testdata/tau2/${SUITE}.json" ]; then
  echo "error: run from the repo root — testdata/tau2/${SUITE}.json not found from $(pwd)" >&2
  exit 1
fi

echo "== running: ${FAK_BIN} bench --suite ${SUITE} --baseline-n ${N} --out ${OUT} =="
"${FAK_BIN}" bench --suite "${SUITE}" --baseline-n "${N}" --out "${OUT}"

echo
echo "== artifacts =="
echo "report.json   : ${OUT}"
echo "baseline.json : ${OUT_DIR}/baseline.json"

# Witness gate: report.json gate_primary == "pass" AND baseline.json p50_ns > 1ms.
gate="$(grep -o '"gate_primary": *"[^"]*"' "${OUT}" | head -1 | sed 's/.*"\([^"]*\)"$/\1/')"
base_ns="$(grep -o '"p50_ns": *[0-9]*' "${OUT_DIR}/baseline.json" | head -1 | grep -o '[0-9]*')"

echo
echo "== witness gate =="
echo "report.json   gate_primary == \"pass\"   -> ${gate}"
if [ "${base_ns}" -gt 1000000 ]; then
  echo "baseline.json p50_ns ( ${base_ns} ) > 1ms -> yes"
else
  echo "baseline.json p50_ns ( ${base_ns} ) > 1ms -> NO (spawned floor too low — measuring noise?)"
fi

if [ "${gate}" = "pass" ] && [ "${base_ns}" -gt 1000000 ]; then
  echo "WITNESS: PASS  (in-process fold beats the spawned-hook floor)"
else
  echo "WITNESS: FAIL"
  exit 1
fi
