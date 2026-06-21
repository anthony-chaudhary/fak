#!/usr/bin/env bash
# fleet_endpoints.sh -- resolve / check OPTIONAL private benchmark endpoints
# from the bash (WSL / git-bash) side of the parity workflow.
#
# The reachability gate that actually matters for benchmarking is "can we open the
# model serve port", so that is the hard signal used here (a TCP connect), not a
# self-report. An endpoint is usable only when  enabled == true  AND its serve port
# accepts a connection; otherwise --resolve exits non-zero and the caller skips it.
# This makes the endpoints OPTIONAL by construction: a batch driver treats a
# non-zero resolve as "skip", a single explicit run treats it as "not available
# yet, bring the server up (see the runbook)".
#
# Usage:
#   tools/fleet_endpoints.sh --list                 # names + enabled + host
#   tools/fleet_endpoints.sh --check <name>         # probe one, human-readable (stderr)
#   tools/fleet_endpoints.sh --all                  # probe every enabled endpoint
#   tools/fleet_endpoints.sh --resolve <name>       # print base-url if usable, else exit 1
#
# Reads tools/fleet_endpoints.local.json by default, falling back to the public
# example template. Override with FAK_FLEET_ENDPOINTS=/path/to/private.json.
# Read-only: it connects, it changes nothing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${FAK_FLEET_ENDPOINTS:-}" ]]; then
  REG="$FAK_FLEET_ENDPOINTS"
elif [[ -f "$ROOT/tools/fleet_endpoints.local.json" ]]; then
  REG="$ROOT/tools/fleet_endpoints.local.json"
elif [[ -f "$ROOT/tools/fleet_endpoints.json" ]]; then
  REG="$ROOT/tools/fleet_endpoints.json"
else
  REG="$ROOT/tools/fleet_endpoints.example.json"
fi
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

[[ -f "$REG" ]] || { echo "fleet_endpoints: registry not found: $REG" >&2; exit 2; }

# All registry logic + the TCP probe live in one python helper (robust JSON parse,
# portable socket connect with a timeout). Mode + optional name come in as argv.
run_py() {
  "$PY" - "$REG" "$@" <<'PYEOF'
import json, socket, sys

reg_path, mode = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else None
reg = json.load(open(reg_path, encoding="utf-8"))
eps = reg.get("endpoints", [])

def find(n):
    for e in eps:
        if e.get("name") == n:
            return e
    return None

def port_open(host, port, timeout=2.5):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False

def reach(e):
    # Prefer the stable tailscale IP; fall back to MagicDNS host.
    for h in (e.get("tailscale_ip"), e.get("magicdns"), e.get("tailnet_host")):
        if h and port_open(h, e.get("serve_port")):
            return True, h
    return False, e.get("tailscale_ip") or e.get("tailnet_host")

if mode == "--list":
    for e in eps:
        print(f"{e['name']:16} enabled={str(bool(e.get('enabled'))).lower():5} "
              f"{e.get('tailnet_host')} ({e.get('tailscale_ip')}) serve:{e.get('serve_port')}")
    sys.exit(0)

if mode == "--resolve":
    e = find(name)
    if e is None:
        print(f"fleet_endpoints: no such endpoint: {name}", file=sys.stderr); sys.exit(2)
    if not e.get("enabled"):
        print(f"fleet_endpoints: {name} is disabled (opt-in-pending); skipping", file=sys.stderr); sys.exit(1)
    ok, host = reach(e)
    if not ok:
        print(f"fleet_endpoints: {name} serve port {e.get('serve_port')} not reachable "
              f"(server not up?); skipping", file=sys.stderr); sys.exit(1)
    print(f"http://{host}:{e.get('serve_port')}/v1")  # the only thing on stdout
    sys.exit(0)

# --check <name> / --all : human-readable status to stderr
targets = [find(name)] if (mode == "--check" and name) else [e for e in eps if e.get("enabled")]
rc = 0
for e in targets:
    if e is None:
        print(f"fleet_endpoints: no such endpoint: {name}", file=sys.stderr); sys.exit(2)
    if not e.get("enabled"):
        print(f"  {e['name']:16} DISABLED (opt-in-pending)", file=sys.stderr); continue
    ok, host = reach(e)
    state = f"READY  serve {host}:{e.get('serve_port')} live" if ok else \
            f"WAITING serve :{e.get('serve_port')} not up (reachable={host})"
    print(f"  {e['name']:16} {state}", file=sys.stderr)
    rc = rc or (0 if ok else 1)
sys.exit(rc)
PYEOF
}

MODE="${1:-}"
case "$MODE" in
  --list)               run_py --list ;;
  --resolve)            run_py --resolve "${2:?need an endpoint name}" ;;
  --check)              run_py --check "${2:?need an endpoint name}" ;;
  --all|"")             echo "fleet endpoints (enabled):" >&2; run_py --all ;;
  -h|--help)            sed -n '2,30p' "$0" ;;
  *) echo "fleet_endpoints: unknown mode '$MODE' (try --list|--check|--all|--resolve)" >&2; exit 2 ;;
esac
