#!/usr/bin/env bash
# glm52_ep_preflight_selfcheck.sh — exercise the EP_PREFLIGHT gate of tools/glm52_ep_witness.sh
# with NO GPU in the loop.
#
# It extracts the real '# --- pre-flight per-GPU free-VRAM gate' block from the witness script,
# stubs the four things that block reads from its host (log/OUT/DONE, plus RANKS/VIS), puts a mock
# nvidia-smi on PATH that answers per '-i <idx>', and runs the scenarios published in
# experiments/glm-gpu-witness/glm52-ep-preflight-guard-selfcheck-2026-07-15.json.
#
# This proves the gate's ARITHMETIC and MESSAGING only. It is not a witness for the sanctioned
# 8-rank GPU-node LOAD_READY run, which remains #4952's open definition-of-done.
#
# Usage: bash tools/glm52_ep_preflight_selfcheck.sh [repo-root]
set -uo pipefail
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK="$(mktemp -d)"
cleanup() { rm -f "$WORK/bin/nvidia-smi"; rmdir "$WORK/bin" 2>/dev/null; rm -f "$WORK"/*; rmdir "$WORK" 2>/dev/null; }
trap cleanup EXIT

sed -n '/^# --- pre-flight per-GPU free-VRAM gate/,/^# --- end pre-flight/p' \
  "$ROOT/tools/glm52_ep_witness.sh" > "$WORK/block.sh"
[ -s "$WORK/block.sh" ] || { echo "EXTRACT_FAIL: no pre-flight block in $ROOT/tools/glm52_ep_witness.sh"; exit 2; }

# mkbin "<idx>:<freeMiB>,<totalMiB>;..." — an empty argument means nvidia-smi is absent.
mkbin() {
  mkdir -p "$WORK/bin"; rm -f "$WORK/bin/nvidia-smi"
  [ -z "$1" ] && return 0
  {
    echo '#!/usr/bin/env bash'
    echo 'idx=""; while [ $# -gt 0 ]; do [ "$1" = "-i" ] && { idx="$2"; shift; }; shift; done'
    echo 'case "$idx" in'
    IFS=';' read -ra rows <<<"$1"
    for row in "${rows[@]}"; do
      echo "  ${row%%:*}) echo '${row#*:}' | tr ',' ' ' | awk '{print \$1\", \"\$2}' ;;"
    done
    echo '  *) exit 1 ;;'
    echo 'esac'
  } > "$WORK/bin/nvidia-smi"
  chmod +x "$WORK/bin/nvidia-smi"
}

# run <name> <RANKS> <VIS> <gpu-map> <want-exit> <want-substring>
run() {
  mkbin "$4"
  local out rc
  out=$(
    PATH="$WORK/bin:$PATH"; export PATH
    RANKS="$2" VIS="$3" OUT="$WORK/out.txt" DONE="$WORK/done.txt"
    export RANKS VIS OUT DONE
    bash -c 'set -uo pipefail; log(){ echo "$*"; }; source "'"$WORK"'/block.sh"' 2>&1
  )
  rc=$?
  if [ "$rc" != "$5" ]; then
    echo "FAIL [$1] exit=$rc want=$5"; echo "$out" | sed 's/^/    /'; return 1
  fi
  if ! printf '%s' "$out" | grep -q -- "$6"; then
    echo "FAIL [$1] output missing '$6'"; echo "$out" | sed 's/^/    /'; return 1
  fi
  echo "ok   [$1] exit=$rc"; echo "$out" | sed 's/^/       /'
  return 0
}

# 80 GiB-class sm_80 card = 81920 MiB total. 77005 MiB = 75.2 GiB, 80486 = 78.6, 71987 = 70.3.
PRISTINE_8="1:80486,81920;2:80486,81920;3:80486,81920;4:80486,81920;5:80486,81920;6:80486,81920;7:80486,81920;8:80486,81920"
STALE_8="1:77005,81920;2:77005,81920;3:77005,81920;4:77005,81920;5:77005,81920;6:77005,81920;7:77005,81920;8:77005,81920"
DIRTY_ONE_8="1:80486,81920;2:80486,81920;3:71987,81920;4:80486,81920;5:80486,81920;6:80486,81920;7:80486,81920;8:80486,81920"
PRISTINE_7="1:80486,81920;2:80486,81920;3:80486,81920;4:80486,81920;5:80486,81920;6:80486,81920;7:80486,81920"

fails=0
run "A stale-residency (reproduces #4952)" 8 "1,2,3,4,5,6,7,8" "$STALE_8"      93 "EP_PREFLIGHT_REFUSE (8/8 short)" || fails=$((fails + 1))
run "B pristine"                           8 "1,2,3,4,5,6,7,8" "$PRISTINE_8"   0  "EP_PREFLIGHT_OK"                 || fails=$((fails + 1))
run "C config-too-tight"                   7 "1,2,3,4,5,6,7"   "$PRISTINE_7"   93 "CARD TOO SMALL"                  || fails=$((fails + 1))
run "D no nvidia-smi"                      8 "1,2,3,4,5,6,7,8" ""              0  "EP_PREFLIGHT_SKIP"               || fails=$((fails + 1))
run "E one dirty GPU"                      8 "1,2,3,4,5,6,7,8" "$DIRTY_ONE_8"  93 "EP_PREFLIGHT_REFUSE (1/8 short)" || fails=$((fails + 1))
run "F unarmed rank count"                 5 "1,2,3,4,5"       "1:1024,81920"  0  "EP_PREFLIGHT_SKIP"               || fails=$((fails + 1))

echo "---"
if [ "$fails" = 0 ]; then echo "EP_PREFLIGHT_SELFCHECK_PASS"; else echo "EP_PREFLIGHT_SELFCHECK_FAIL=$fails"; fi
exit "$fails"
