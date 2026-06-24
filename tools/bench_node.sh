#!/usr/bin/env bash
# bench_node.sh -- repeatable bench-node runner. ONE command to run tests/benchmarks on a
# dedicated bench node over Tailscale, folding in every step this had to be done by hand the
# first time (offline-wait, host-key trust, ssh user/key, go toolchain, zsh-vs-bash, repo
# selection, placement law). See tools/bench_nodes.example.json for the registry schema.
#
#   tools/bench_node.sh <node> ping|wait|info
#   tools/bench_node.sh <node> tests          # go test ggufload+model (correctness)
#   tools/bench_node.sh <node> bench           # kernel-latency microbenches (ns/op)
#   tools/bench_node.sh <node> kernels         # arm64 NEON quant hot-path: correctness gate +
#                                              #   percell-vs-row4 GEMM A/B + prove/refute verdict
#   tools/bench_node.sh <node> cmd <shell...>  # arbitrary command on the node
#   WAIT=1 tools/bench_node.sh <node> bench    # auto-wait if the node is asleep
#   KCOUNT=8 tools/bench_node.sh <node> kernels  # benchmark -count (default 6)
#
# Every run writes a lineage.json (utc datetime, fak version+commit+branch, node, go) next to
# the result -- observability/lineage so any number is traceable (HARDWARE-CATALOG rigor rule).
#
# SAFETY (per the adversarial design review): this driver is READ-ONLY w.r.t. the committed
# tree. Results land ONLY in tools/_registry/bench-runs/ (gitignored). It never writes a
# committable artifact, because the repo's scrub audit is shape-only and would NOT redact a
# CPU brand / hostname. Promoting a result to a committed path needs a real redaction pass.
set -uo pipefail

RUNNER_VERSION="0.3.0"   # 0.3.0: kernels subcommand (arm64 NEON hot-path A/B + prove/refute); 0.2.0: lineage.json
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REG="${BENCH_NODES:-$SELF_DIR/bench_nodes.json}"
SCRATCH="${BENCH_RUNS_DIR:-$SELF_DIR/_registry/bench-runs}"   # gitignored

# resolve a python 3 interpreter (py3 -> python -> py -3), like the repo's other tools
PY=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[ -z "$PY" ] && command -v py >/dev/null 2>&1 && PY="py -3"
[ -z "$PY" ] && { echo "bench_node: no python interpreter found (py3/python/py)" >&2; exit 70; }

usage(){ cat >&2 <<'U'
usage: bench_node.sh <node> <subcommand> [args]
  ping            SSH-handshake reachability check (exit 0 = reachable)
  wait            poll (SSH handshake, backoff) until reachable or BENCH_WAIT_MAX_S
  info            print resolved node facts (sanitized name only)
  cmd <shell...>  run an arbitrary command on the node (-> gitignored scratch)
  tests           go test ./internal/ggufload ./internal/model -count=1
  bench           kernel-latency microbenches (adjudicator+ctxmmu, count=8, ns/op)
  kernels         arm64 NEON quant hot-path: bit-exact correctness gate + per-cell-vs-row4
                  GEMM A/B (MAC/ns) + a prove/refute verdict on the row4 speedup claim
env: WAIT=1 auto-wait when offline; BENCH_ALLOW_COLOCATED=1 override placement law;
     KCOUNT=N kernels benchmark -count (default 6)
U
exit 64; }

[ $# -ge 1 ] || usage
NODE="$1"; shift; SUB="${1:-info}"; [ $# -gt 0 ] && shift || true

# --- registry reader: hard-fail (exit 4) on missing/empty field unless allow_empty=1 ---
field(){
  $PY - "$REG" "$NODE" "$1" "${2:-0}" <<'PY'
import json,sys
reg,node,key,allow=sys.argv[1:5]
try: d=json.load(open(reg))
except Exception as e: sys.stderr.write("registry %s: %s\n"%(reg,e)); sys.exit(3)
ns=[x for x in d.get("nodes",[]) if x.get("name")==node]
if not ns:
    sys.stderr.write("node '%s' not in %s (have: %s)\n"%(node,reg,[x.get('name') for x in d.get('nodes',[])])); sys.exit(3)
v=ns[0].get(key)
if isinstance(v,(list,dict)): v=json.dumps(v)
if v in (None,"") and allow!="1":
    sys.stderr.write("node '%s' field '%s' is empty in %s\n"%(node,key,reg)); sys.exit(4)
print("" if v is None else v)
PY
}

SAN="$(field sanitized_name)"   || exit $?
IP="$(field tailnet_ip)"        || exit $?
SSH_USER="$(field ssh_user)"    || exit $?   # NB: namespaced, never the reserved $USER
SSH_KEY="$(field ssh_key)"      || exit $?; SSH_KEY="${SSH_KEY/#\~/$HOME}"
SSH_PORT="$(field ssh_port)"    || exit $?
REPO="$(field repo_path)"       || exit $?
TC="$(field toolchain_env)"     || exit $?
GO_BIN="$(field go_bin 1)"; [ -z "$GO_BIN" ] && GO_BIN="/usr/local/go/bin"
HOSTKEY="$(field host_key 1)"
AGENT_HOST="$(field agent_host 1)"
CAPS="$(field capabilities 1)"; [ -z "$CAPS" ] && CAPS='[]'

# remote preamble shared by every run path: put go on PATH (login PATH isn't inherited by
# `bash -s`), set the toolchain, cd to the canonical repo. \$PATH stays literal for the node.
PRE="export PATH=\"$GO_BIN:\$PATH\"; export $TC; cd $REPO"

# --- pinned host key (review: verify against registry, don't blind accept-new) ---
KH="$(mktemp)"; trap 'rm -f "$KH"' EXIT
# ServerAlive* aborts a session whose node went to sleep MID-COMMAND: without it a dead
# connection hangs on the TCP keepalive default (minutes), freezing a long bench. 8s x 3
# misses => the ssh gives up in ~24s and the runner can retry the node's next wake window.
SSH_OPTS=(-i "$SSH_KEY" -p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=8 -o ServerAliveCountMax=3 -o UserKnownHostsFile="$KH")
if [ -n "$HOSTKEY" ]; then
  printf '[%s]:%s %s\n%s %s\n' "$IP" "$SSH_PORT" "$HOSTKEY" "$IP" "$HOSTKEY" > "$KH"
  SSH_OPTS+=(-o StrictHostKeyChecking=yes)
else
  echo "WARN: no pinned host_key for '$NODE'; using accept-new (add host_key to the registry to pin)." >&2
  SSH_OPTS+=(-o StrictHostKeyChecking=accept-new)
fi
ssh_node(){ ssh "${SSH_OPTS[@]}" "$SSH_USER@$IP" "$@"; }

# --- placement law: refuse if the target IS this host (real check: tailscale IP match) ---
SELF_IP="$(tailscale ip -4 2>/dev/null | head -1)"
if [ "$AGENT_HOST" = "true" ] || { [ -n "$SELF_IP" ] && [ "$SELF_IP" = "$IP" ]; }; then
  echo "REFUSE: '$NODE' ($IP) is THIS agent's own host -- a colocated heavy bench skews the number ~4x and starves the agent." >&2
  [ "${BENCH_ALLOW_COLOCATED:-0}" = "1" ] || exit 8
  echo "WARN: BENCH_ALLOW_COLOCATED=1 -- proceeding under contention." >&2
fi

reachable(){ ssh_node true >/dev/null 2>&1; }   # SSH handshake -- a node can answer ping with no sshd

do_wait(){
  local max="${BENCH_WAIT_MAX_S:-86400}" d=5 t=0
  echo "[wait] $NODE: SSH-handshake poll, backoff to 60s (cap ${max}s)..." >&2
  until reachable; do
    sleep "$d"; t=$((t+d)); d=$((d*2)); [ "$d" -gt 60 ] && d=60
    [ "$t" -ge "$max" ] && { echo "[wait] gave up after ${t}s (no SSH on $NODE)" >&2; return 1; }
  done
  echo "[wait] $NODE reachable after ${t}s" >&2
}

ensure_online(){
  reachable && return 0
  if [ "${WAIT:-0}" = "1" ] || [ "$SUB" = "wait" ]; then do_wait; return $?; fi
  echo "OFFLINE: $NODE not SSH-reachable. Re-run with WAIT=1 (or the 'wait' subcommand)." >&2; return 3
}

newrun(){ # echo a fresh gitignored run dir for $SUB
  local ts; ts="$(date -u +%Y%m%dT%H%M%SZ)"
  local d="$SCRATCH/$SAN/$ts-$SUB"; mkdir -p "$d"; echo "$d"
}

# lineage.json: the observability/provenance stamp written next to every run. Combines
# driver-side facts (UTC datetime, runner version) with node-side facts (fak version, git
# commit/branch, go) so ANY number traces to a version + date/time + machine. Only the
# SANITIZED node name is recorded.
lineage(){ # lineage <outdir> <subcommand>
  local out="$1" sub="$2" utc nf go commit branch ver
  utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nf="$(ssh_node "bash -s" <<EOF 2>/dev/null
$PRE
echo "go=\$(go version 2>/dev/null)"
echo "commit=\$(git rev-parse --short HEAD 2>/dev/null)"
echo "branch=\$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "ver=\$(cat VERSION 2>/dev/null || cat fak/VERSION 2>/dev/null)"
EOF
)"
  go="$(printf '%s\n' "$nf" | sed -n 's/^go=//p')"
  commit="$(printf '%s\n' "$nf" | sed -n 's/^commit=//p')"
  branch="$(printf '%s\n' "$nf" | sed -n 's/^branch=//p')"
  ver="$(printf '%s\n' "$nf" | sed -n 's/^ver=//p')"
  cat > "$out/lineage.json" <<JSON
{
  "lineage_schema": "fleet-bench-lineage/1",
  "utc": "$utc",
  "runner": "bench_node.sh",
  "runner_version": "$RUNNER_VERSION",
  "subcommand": "$sub",
  "node": "$SAN",
  "node_capabilities": $CAPS,
  "fak_version": "${ver:-unknown}",
  "git_commit": "${commit:-unknown}",
  "git_branch": "${branch:-unknown}",
  "go_version": "${go:-unknown}"
}
JSON
  echo "lineage: $SAN fak=${ver:-?}@${commit:-?} $utc -> $out/lineage.json"
}

case "$SUB" in
  info)
    echo "node=$NODE sanitized=$SAN repo=$REPO toolchain=$TC pinned_hostkey=$([ -n "$HOSTKEY" ] && echo yes || echo no)"
    ;;
  ping)
    if reachable; then echo "REACHABLE $NODE (as $SAN)"; else echo "UNREACHABLE $NODE" >&2; exit 3; fi
    ;;
  wait)
    ensure_online || exit $?
    ;;
  cmd)
    ensure_online || exit $?
    [ $# -ge 1 ] || { echo "cmd needs a command" >&2; exit 64; }
    OUT="$(newrun)"; lineage "$OUT" "$SUB"
    printf '%s\n%s\n' "$PRE" "$*" | ssh_node "bash -s" 2>&1 | tee "$OUT/run.log"
    echo "-> $OUT (gitignored)"
    ;;
  tests)
    ensure_online || exit $?
    OUT="$(newrun)"; lineage "$OUT" "$SUB"
    ssh_node "bash -s" > "$OUT/run.log" 2>&1 <<EOF
set -uo pipefail; $PRE
go -C fak test ./internal/ggufload/ ./internal/model/ -count=1
EOF
    rc=$?
    tail -8 "$OUT/run.log"; echo "tests rc=$rc -> $OUT (gitignored)"; exit $rc
    ;;
  bench)
    ensure_online || exit $?
    OUT="$(newrun)"; lineage "$OUT" "$SUB"
    ssh_node "bash -s" > "$OUT/run.log" 2>&1 <<EOF
set -uo pipefail; $PRE
echo "### adjudicator ###"
go -C fak test -run='^\$' -bench=. -benchmem -count=8 ./internal/adjudicator/
echo "### ctxmmu ###"
go -C fak test -run='^\$' -bench=. -benchmem -count=8 ./internal/ctxmmu/
EOF
    rc=$?
    # presence guard (review): `go test -bench` exits 0 even with ZERO benchmarks. Check
    # EACH "### name ###" section so a per-package absence (e.g. ctxmmu bench_test.go missing
    # at this commit) warns loudly instead of hiding behind another package's numbers.
    $PY - "$OUT/run.log" >&2 <<'PY'
import sys,re
log=open(sys.argv[1],encoding='utf-8',errors='replace').read()
secs=re.split(r'^### (.+?) ###$', log, flags=re.M)
for i in range(1,len(secs)-1,2):
    name,body=secs[i].strip(),secs[i+1]
    if not re.search(r'Benchmark\S+\s+\d+\s+[\d.]+ ns/op', body):
        sys.stderr.write("WARN: package '%s' produced 0 benchmarks -- bench_test.go likely absent at this commit.\n"%name)
PY
    grep -E 'Benchmark(Decide|AdjudicateArgScaling|ScreenBytes|Admit)' "$OUT/run.log" | head -40
    echo "bench rc=$rc -> $OUT (gitignored)"; exit $rc
    ;;
  kernels)
    # arm64 NEON quant HOT PATH (the real decode/prefill kernel, not an idle microbench). Runs
    # the bit-exact correctness GATE first, then the per-cell-vs-row4 GEMM A/B, then folds the
    # MAC/ns into a prove/refute VERDICT on the row4 load-reuse speedup claim. These kernels are
    # //go:build arm64 — on a non-arm64 node they vanish (0 benches), so the parser WARNs loudly.
    ensure_online || exit $?
    KCOUNT="${KCOUNT:-6}"
    OUT="$(newrun)"; lineage "$OUT" "$SUB"
    ssh_node "bash -s" > "$OUT/run.log" 2>&1 <<EOF
set -uo pipefail; $PRE
echo "### arch ### \$(uname -m)"
echo "### correctness ###"
go -C fak test -run 'TestQGemm8Row4Sane|TestQGemm8IntoMatchesScalarNEON|TestQMatRows4NEONMatchesCell|TestQGemm8TileNEONMatchesCell' -count=1 -v ./internal/model/
echo "### CORRECTNESS_RC=\$? ###"
echo "### bench ###"
go -C fak test -run '^\$' -bench 'BenchmarkDotKernelSingleCore|BenchmarkGemmKernelSingleCore' -benchmem -count=$KCOUNT ./internal/model/
EOF
    rc=$?
    # fold: arch guard + correctness gate + median MAC/ns A/B + prove/refute verdict
    $PY - "$OUT/run.log" >&2 <<'PY'
import sys,re
log=open(sys.argv[1],encoding='utf-8',errors='replace').read()
def median(xs):
    s=sorted(xs); n=len(s)
    return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2.0
arch=(re.search(r'### arch ### (\S+)',log) or [None,'?'])[1]
if not re.search(r'arm64|aarch64',arch or ''):
    sys.stderr.write("WARN: node arch '%s' is not arm64 -- the NEON quant kernels are //go:build arm64 and DID NOT run.\n"%arch)
m=re.search(r'### CORRECTNESS_RC=(\d+) ###',log)
crc=int(m.group(1)) if m else 1
gate="PASS" if crc==0 else "FAIL"
# collect MAC/ns by (bench, size, variant)
data={}
for name,path,macns in re.findall(r'Benchmark(\w+)/(\S+?)-\d+\s+\d+\s+[\d.]+ ns/op\s+([\d.]+) MAC/ns',log):
    parts=path.split('/'); variant=parts[-1]; size='/'.join(parts[:-1])
    data.setdefault((name,size),{}).setdefault(variant,[]).append(float(macns))
# baseline->candidate per bench family (the A/B we care about)
AB={'GemmKernelSingleCore':('percell','row4'),'DotKernelSingleCore':('asm','unroll4')}
print("CORRECTNESS GATE: %s (rc=%d)"%(gate,crc))
verdict_lines=[]; ratios=[]
for (name,size),v in sorted(data.items()):
    base,cand=AB.get(name,(None,None))
    if base in v and cand in v:
        b=median(v[base]); c=median(v[cand]); r=c/b if b else 0
        print("  %-22s %-16s %-8s %6.2f -> %-7s %6.2f  MAC/ns  (%s %+.1f%%)"%(name,size,base,b,cand,c,cand,(r-1)*100))
        if name=='GemmKernelSingleCore': ratios.append(r); verdict_lines.append((size,r))
if ratios:
    med=median(ratios)
    claim="CONFIRMED" if med>=1.5 else "REFUTED"
    print("\nVERDICT (row4 load-reuse GEMM vs per-cell):")
    for size,r in verdict_lines: print("  %-16s row4 = %.3fx per-cell  (%+.1f%%)"%(size,r,(r-1)*100))
    print("  median speedup = %.3fx"%med)
    print("  CODE CLAIM ~1.8x  -> %s%s"%(claim," (row4 is real but single-digit-%, not 1.8x)" if claim=='REFUTED' else ""))
sys.exit(0 if crc==0 else 1)
PY
    pyrc=$?
    echo "kernels: correctness $([ $pyrc -eq 0 ] && echo PASS || echo FAIL) | bench rc=$rc -> $OUT (gitignored)"
    exit $pyrc
    ;;
  *) usage ;;
esac
