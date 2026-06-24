#!/usr/bin/env bash
# run.sh — operator observability walkthrough for `fak serve`.
#
# Builds the fak kernel, serves it OFFLINE (no --base-url => deterministic mock engine,
# so this needs no model / API key / GPU), drives a few calls through the kernel
# (one allow, one deny, one quarantined result, plus a repeated read-only call that the
# vDSO serves from cache), then:
#   1. scrapes GET /metrics and prints the lines behind the four operator questions,
#   2. fetches GET /debug/vars and prints the break-glass JSON blocks,
#   3. demonstrates the auth path (/metrics under --require-key-env: 401 -> 200).
# Everything it starts is torn down on exit.
#
#   ./run.sh                         # build, serve, drive, scrape, teardown
#
# Requires: Go (to build fak) + curl. Optional: jq (falls back to python3, then raw JSON).
# Env knobs:
#   FAK_OBS_ADDR   gateway bind addr      (default 127.0.0.1:8080)
#   FAK_BIN        prebuilt fak binary    (default: build ./cmd/fak)
#   FAK_OBS_SKIP_AUTH=1   skip the auth-path demonstration at the end
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"               # examples/observability -> fak/
ADDR="${FAK_OBS_ADDR:-127.0.0.1:8080}"
POLICY="$FAK_DIR/examples/customer-support-readonly-policy.json"
BASE="http://$ADDR"
log(){ printf '\033[36m[obs]\033[0m %s\n' "$*" >&2; }

command -v curl >/dev/null || { log "curl not found — install it (the walkthrough scrapes HTTP)"; exit 1; }

# A poisoned tool result: a leaked credential the result-side floor must contain.
POISON='{"page":"config loaded. api_key=sk-abcdef0123456789abcdef0123 was found in env"}'

# Pretty-print a JSON filter: jq if present, else python3, else the raw object.
# usage: json_view '<jq-filter>' '<python-expr over d>' <<<"$json"
json_view(){
  local jqf="$1" pyf="$2" body; body="$(cat)"
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$body" | jq "$jqf"
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s' "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps($pyf, indent=2))"
  else
    printf '%s\n' "$body"
  fi
}

BIN_DIR=""; KPID=""
cleanup(){
  [ -n "$KPID" ] && kill "$KPID" 2>/dev/null || true
  [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1) the kernel binary (honor a prebuilt FAK_BIN, else build one)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak kernel -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

# start_server <extra serve args...> — boots `fak serve` offline and waits for /healthz.
start_server(){
  if curl -sf "$BASE/healthz" >/dev/null 2>&1; then
    log "$ADDR already has a kernel — stop it or set FAK_OBS_ADDR"; exit 1
  fi
  "$BIN" serve --addr "$ADDR" --policy "$POLICY" "$@" >"${BIN_DIR:-/tmp}/fak-obs.log" 2>&1 & KPID=$!
  local tries=0
  until curl -sf "$BASE/healthz" >/dev/null 2>&1; do
    kill -0 "$KPID" 2>/dev/null || { log "kernel died on startup — last log lines:"; tail -20 "${BIN_DIR:-/tmp}/fak-obs.log" >&2 || true; exit 1; }
    tries=$((tries + 1)); [ "$tries" -ge 100 ] && { log "kernel did not become healthy within ~30s"; exit 1; }
    sleep 0.3
  done
}
stop_server(){ [ -n "$KPID" ] && kill "$KPID" 2>/dev/null || true; KPID=""; sleep 0.5; }

# 2) serve offline (mock engine) and confirm health
log "starting kernel: fak serve $ADDR  (offline mock engine, floor = customer-support-readonly-policy.json)"
start_server
log "kernel healthy: $(curl -s "$BASE/healthz")"
echo

# 3) drive a few adjudicated calls so the counters have something to show
log "driving calls through the kernel: allow, deny, quarantine, + a repeated read-only call (vDSO)"
curl -s -X POST "$BASE/v1/fak/syscall" -d '{"tool":"search_kb","arguments":{"q":"refund window"}}' >/dev/null   # ALLOW
curl -s -X POST "$BASE/v1/fak/syscall" -d '{"tool":"refund_payment","arguments":{"amount":500}}'   >/dev/null   # DENY (POLICY_BLOCK)
curl -s -X POST "$BASE/v1/fak/admit"   -d "{\"tool\":\"fetch_url\",\"result\":$POISON}"             >/dev/null   # QUARANTINE (SECRET_EXFIL)
# same read-only call twice with the same witness: the second is a vDSO cache hit
curl -s -X POST "$BASE/v1/fak/syscall" -d '{"tool":"search_kb","arguments":{"q":"sla"},"read_only":true,"witness":"commit-abc"}' >/dev/null
curl -s -X POST "$BASE/v1/fak/syscall" -d '{"tool":"search_kb","arguments":{"q":"sla"},"read_only":true,"witness":"commit-abc"}' >/dev/null
echo

# 4) scrape /metrics — the lines behind the four operator questions
log "GET /metrics — the lines an operator queries:"
echo "# Q1  Is the floor firing?  (verdict counters + kernel deny/quarantine totals)"
curl -s "$BASE/metrics" | grep -E '^fak_(gateway_operations_total|kernel_(denies|quarantines)_total)' || true
echo
echo "# Q2  Is the cache hitting?  (vDSO hit ratio + backing counters)"
curl -s "$BASE/metrics" | grep -E '^fak_(gateway_vdso_hit_ratio|kernel_(vdso_hits|submits)_total)' || true
echo
echo "# Q3  Is anything stuck inflight?  (aggregate gauge + oldest-request age)"
curl -s "$BASE/metrics" | grep -E '^fak_gateway_(inflight_requests|inflight_max_age_seconds)' || true
echo
echo "# build labels (version / engine / model / vdso)"
curl -s "$BASE/metrics" | grep -E '^fak_gateway_build_info' || true
echo

# 5) /debug/vars — the break-glass JSON snapshot (distinct from Prometheus: one-shot, interactive)
log "GET /debug/vars — break-glass live process view:"
echo "# kernel counters block"
curl -s "$BASE/debug/vars" | json_view '.kernel' "d['kernel']"
echo "# gateway: inflight + uptime + auth_required"
curl -s "$BASE/debug/vars" | json_view '{inflight: .gateway.inflight_requests, uptime_s: .gateway.uptime_seconds, auth_required: .gateway.auth_required}' \
  "{'inflight': d['gateway']['inflight_requests'], 'uptime_s': round(d['gateway']['uptime_seconds'],3), 'auth_required': d['gateway']['auth_required']}"
echo "# runtime: goroutines + heap (is it leaking / wedged?)"
curl -s "$BASE/debug/vars" | json_view '{goroutines: .runtime.num_goroutine, alloc_bytes: .runtime.memory.alloc_bytes, num_gc: .runtime.memory.num_gc}' \
  "{'goroutines': d['runtime']['num_goroutine'], 'alloc_bytes': d['runtime']['memory']['alloc_bytes'], 'num_gc': d['runtime']['memory']['num_gc']}"
echo

# 6) auth path: both surfaces honor --require-key-env (Bearer)
if [ "${FAK_OBS_SKIP_AUTH:-0}" != "1" ]; then
  log "restarting WITH --require-key-env to demonstrate the auth path"
  stop_server
  export FAK_OBS_DEMO_KEY="s3cret-operator-token"
  start_server --require-key-env FAK_OBS_DEMO_KEY
  printf '  %-46s -> HTTP %s\n' "GET /metrics  (no token)"      "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/metrics")"
  printf '  %-46s -> HTTP %s\n' "GET /metrics  (Bearer token)"  "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $FAK_OBS_DEMO_KEY" "$BASE/metrics")"
  printf '  %-46s -> HTTP %s\n' "GET /healthz  (always open)"    "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/healthz")"
  echo
fi

log "done — see README.md for the PromQL queries and EXAMPLE-OUTPUT.md for a captured run."
