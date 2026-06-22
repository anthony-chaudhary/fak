#!/usr/bin/env bash
# run.sh — the fak auth-hardening walkthrough: --require-key-env, end to end.
#
# Starts `fak serve --require-key-env FAK_TOKEN` (the network-facing hardening flag)
# with the secret in the env, then runs FOUR witnesses against the gated surface:
#   1. Authorization: Bearer <correct>  -> 200   (OpenAI / fak-native clients)
#   2. Authorization: Bearer <wrong>    -> 401
#   3. no auth header                   -> 401
#   4. x-api-key: <correct>             -> 200   (Claude Code / Anthropic SDK shape)
# plus two OPERATOR-ROUTE witnesses proving /metrics inherits the SAME gate:
#   5. GET /metrics  (no header)        -> 401
#   6. GET /metrics  (Bearer correct)   -> 200
#
# Why these endpoints: the auth gate is a pure function of (token, header), checked
# in front of EVERY route except /healthz. So /v1/models and /metrics answer
# 200-when-authed with NO model, key, GPU, or --base-url configured — the witnesses
# exercise the gate itself, not an upstream. (/healthz is the one exempt route, so
# it is the unauthenticated readiness probe below.)
#
#   ./run.sh                       # build, serve, run the six witnesses, teardown
#
# Env knobs:
#   FAK_DEMO_PORT  gateway port                       (default 8080)
#   FAK_BIN        prebuilt fak binary to use         (default: build ./cmd/fak)
#   FAK_TOKEN      the secret to require              (default: a demo secret)
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/auth-hardening -> fak/
PORT="${FAK_DEMO_PORT:-8080}"
BASE="http://127.0.0.1:$PORT"
# The secret the gateway will REQUIRE. --require-key-env names the ENV VAR, never the
# literal — the token stays out of argv (where `ps` would leak it). A real deployment
# injects it from a secret manager; here we default one so the demo is self-contained.
export FAK_TOKEN="${FAK_TOKEN:-s3cret-demo-token}"
WRONG_TOKEN="not-the-token"
TMP="${TMPDIR:-/tmp}"; TMP="${TMP%/}"
log(){ printf '\033[36m[auth]\033[0m %s\n' "$*" >&2; }

KPID=""; BIN_DIR=""; FAILS=0
cleanup(){
  [ -n "$KPID" ] && kill "$KPID" 2>/dev/null || true
  [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1) the fak binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak )
fi

# 2) start the gateway REQUIRING the secret (no --base-url → no model needed; the
#    witnesses hit the auth gate, which sits in front of every route but /healthz).
if curl -sf "$BASE/healthz" >/dev/null 2>&1; then
  log "port $PORT already has a server — stop it or set FAK_DEMO_PORT"; exit 1
fi
FAKLOG="$TMP/fak-auth-hardening.log"
log "starting gateway: fak serve $BASE  --require-key-env FAK_TOKEN"
"$BIN" serve --addr "127.0.0.1:$PORT" --require-key-env FAK_TOKEN >"$FAKLOG" 2>&1 & KPID=$!
# /healthz is the ONE route exempt from the gate — so readiness needs no token.
tries=0
until curl -sf "$BASE/healthz" >/dev/null 2>&1; do
  if ! kill -0 "$KPID" 2>/dev/null; then
    log "gateway died on startup (addr=127.0.0.1:$PORT) — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1
  fi
  tries=$((tries + 1)); if [ "$tries" -ge 200 ]; then log "gateway did not become healthy within ~60s — last log lines:"; tail -20 "$FAKLOG" >&2 || true; exit 1; fi
  sleep 0.3
done
log "gateway healthy (auth=on): $(curl -s "$BASE/healthz")"
echo

# 3) witness: assert the HTTP status of one request against what we expect.
#    `curl -o /dev/null -w '%{http_code}'` prints ONLY the status line — the gate's
#    verdict, with no body noise.
witness(){
  local want="$1" desc="$2"; shift 2
  local got
  got="$(curl -s -o /dev/null -w '%{http_code}' "$@")"
  if [ "$got" = "$want" ]; then
    printf '  \033[32m✓\033[0m %-46s -> %s\n' "$desc" "$got"
  else
    printf '  \033[31m✗\033[0m %-46s -> %s (wanted %s)\n' "$desc" "$got" "$want"
    FAILS=$((FAILS + 1))
  fi
}

log "FOUR auth witnesses against the gated surface (GET /v1/models):"
witness 200 "1. Authorization: Bearer <correct>" -H "Authorization: Bearer $FAK_TOKEN"   "$BASE/v1/models"
witness 401 "2. Authorization: Bearer <wrong>"   -H "Authorization: Bearer $WRONG_TOKEN" "$BASE/v1/models"
witness 401 "3. no auth header"                                                          "$BASE/v1/models"
witness 200 "4. x-api-key: <correct>"            -H "x-api-key: $FAK_TOKEN"              "$BASE/v1/models"
echo
log "operator-route witnesses — /metrics inherits the SAME gate:"
witness 401 "5. GET /metrics (no header)"                                                "$BASE/metrics"
witness 200 "6. GET /metrics (Bearer correct)"   -H "Authorization: Bearer $FAK_TOKEN"   "$BASE/metrics"
echo

if [ "$FAILS" -ne 0 ]; then
  log "$FAILS witness(es) FAILED"; exit 1
fi
log "all six witnesses passed — both header shapes authenticate; both failure modes 401; operator route gated."
