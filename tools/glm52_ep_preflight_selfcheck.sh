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

# pathWithoutNvidiaSmi echoes PATH with every directory that holds a real nvidia-smi removed.
# The "no nvidia-smi" scenario has to be true on a host that HAS one — this script is meant to run
# on the GPU node too, where merely omitting the mock would let the block read the real GPUs and
# turn a fail-open assertion into a spurious refusal. Dropping the mock is not enough; the real
# binary has to be off PATH. awk/tr/sed/grep stay reachable because only nvidia-smi's own
# directories are dropped.
pathWithoutNvidiaSmi() {
  local out="" d
  local IFS=:
  for d in $PATH; do
    [ -z "$d" ] && continue
    [ -x "$d/nvidia-smi" ] && continue
    [ -x "$d/nvidia-smi.exe" ] && continue
    out="${out:+$out:}$d"
  done
  printf '%s' "$out"
}

# run <name> <RANKS> <VIS> <gpu-map> <want-exit> <want-substring>...
run() {
  mkbin "$4"
  local name="$1" ranks="$2" vis="$3" map="$4" wantrc="$5"
  shift 5
  local base="$PATH"
  [ -z "$map" ] && base="$(pathWithoutNvidiaSmi)"
  local out rc
  out=$(
    PATH="$WORK/bin:$base"; export PATH
    RANKS="$ranks" VIS="$vis" OUT="$WORK/out.txt" DONE="$WORK/done.txt"
    export RANKS VIS OUT DONE
    bash -c 'set -uo pipefail; log(){ echo "$*"; }; source "'"$WORK"'/block.sh"' 2>&1
  )
  rc=$?
  if [ "$rc" != "$wantrc" ]; then
    echo "FAIL [$name] exit=$rc want=$wantrc"; echo "$out" | sed 's/^/    /'; return 1
  fi
  local want
  for want in "$@"; do
    if ! printf '%s' "$out" | grep -q -F -- "$want"; then
      echo "FAIL [$name] output missing '$want'"; echo "$out" | sed 's/^/    /'; return 1
    fi
  done
  echo "ok   [$name] exit=$rc"; echo "$out" | sed 's/^/       /'
  return 0
}

# 80 GiB-class sm_80 card = 81920 MiB total. 77005 MiB = 75.2 GiB, 80486 = 78.6, 71987 = 70.3.
PRISTINE_8="1:80486,81920;2:80486,81920;3:80486,81920;4:80486,81920;5:80486,81920;6:80486,81920;7:80486,81920;8:80486,81920"
STALE_8="1:77005,81920;2:77005,81920;3:77005,81920;4:77005,81920;5:77005,81920;6:77005,81920;7:77005,81920;8:77005,81920"
DIRTY_ONE_8="1:80486,81920;2:80486,81920;3:71987,81920;4:80486,81920;5:80486,81920;6:80486,81920;7:80486,81920;8:80486,81920"
PRISTINE_7="1:80486,81920;2:80486,81920;3:80486,81920;4:80486,81920;5:80486,81920;6:80486,81920;7:80486,81920"

# The require_free= assertions below are the load-bearing ones: they pin the threshold the SHELL
# actually computed, not a restatement of it. A wrong awk formula that still landed between the
# pristine and stale free values would keep every verdict correct and slip through otherwise.
fails=0
run "A stale-residency (reproduces #4952)" 8 "1,2,3,4,5,6,7,8" "$STALE_8"      93 \
  "require_free=77.1GiB/gpu" "EP_PREFLIGHT_REFUSE (8/8 short)" "short=1.9GiB" || fails=$((fails + 1))
run "B pristine"                           8 "1,2,3,4,5,6,7,8" "$PRISTINE_8"   0  \
  "require_free=77.1GiB/gpu" "EP_PREFLIGHT_OK 8/8"                            || fails=$((fails + 1))
run "C config-too-tight"                   7 "1,2,3,4,5,6,7"   "$PRISTINE_7"   93 \
  "require_free=85.6GiB/gpu" "CARD TOO SMALL"                                 || fails=$((fails + 1))
run "D no nvidia-smi"                      8 "1,2,3,4,5,6,7,8" ""              0  \
  "EP_PREFLIGHT_SKIP no nvidia-smi"                                           || fails=$((fails + 1))
run "E one dirty GPU"                      8 "1,2,3,4,5,6,7,8" "$DIRTY_ONE_8"  93 \
  "require_free=77.1GiB/gpu" "EP_PREFLIGHT_REFUSE (1/8 short)" "short=6.8GiB"  || fails=$((fails + 1))
run "F unarmed rank count"                 5 "1,2,3,4,5"       "1:1024,81920"  0  \
  "EP_PREFLIGHT_SKIP no published per-rank plan total for RANKS=5"            || fails=$((fails + 1))

echo "---"
if [ "$fails" = 0 ]; then echo "EP_PREFLIGHT_SELFCHECK_PASS"; else echo "EP_PREFLIGHT_SELFCHECK_FAIL=$fails"; fi
exit "$fails"
