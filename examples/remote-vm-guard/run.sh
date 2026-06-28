#!/usr/bin/env bash
# run.sh — the fak "guard on a random VM" walkthrough: the network-egress floor.
#
# THE SCENARIO. You move your coding agent (Claude Code, Codex, …) onto an ephemeral
# cloud VM and step away — there is nobody to click "approve" on a tool call. The one
# attack that turns that convenience into a breach is an SSRF to the cloud-instance
# METADATA endpoint (169.254.169.254 & peers): a single GET there returns the VM's IAM
# role credentials, and a prompt-injected agent walks out of the box with them.
#
# THE FLOOR. `fak guard` carries a structural egress rung INTO the VM: a tool call that
# reaches the metadata / link-local family is refused BY SHAPE — EGRESS_BLOCK, no model
# and no human in the loop. This demo proves it the deterministic way, with `fak egress
# check`, which runs the SAME kernel floor a guarded session enforces.
#
#   ./run.sh                       # build fak, run the egress witnesses, report PASS/FAIL
#
# Needs only Go (to build fak) — NO model, key, GPU, server, or network. Every witness
# is a pure function of the destination, so the result is identical on every run.
#
# Env knobs:
#   FAK_BIN   prebuilt fak binary to use   (default: build ./cmd/fak)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/remote-vm-guard -> fak/
log(){ printf '\033[36m[vm-guard]\033[0m %s\n' "$*" >&2; }

BIN_DIR=""; FAILS=0
cleanup(){ [ -n "$BIN_DIR" ] && rm -rf "$BIN_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# 1) the fak binary (build it, or honor a prebuilt FAK_BIN)
BIN="${FAK_BIN:-}"
if [ -z "$BIN" ]; then
  command -v go >/dev/null || { log "Go not found — install from https://go.dev/dl (or set FAK_BIN to a prebuilt fak)"; exit 1; }
  BIN_DIR="$(mktemp -d)"; BIN="$BIN_DIR/fak"
  log "building fak -> $BIN"
  ( cd "$FAK_DIR" && go build -o "$BIN" ./cmd/fak ) || { log "build failed"; exit 1; }
fi

# 2) witness: run `fak egress check` and assert its exit code (0 allowed, 1 blocked).
#    The destination is passed exactly as an agent's tool arg would carry it.
witness(){
  local want="$1" desc="$2"; shift 2
  "$BIN" egress check "$@" >/dev/null 2>&1
  local got=$?
  local label="ALLOW"; [ "$want" = 1 ] && label="BLOCK"
  if [ "$got" = "$want" ]; then
    printf '  \033[32m✓\033[0m %-52s -> %s\n' "$desc" "$label"
  else
    local gotlabel="ALLOW"; [ "$got" = 1 ] && gotlabel="BLOCK"; [ "$got" = 2 ] && gotlabel="USAGE-ERR"
    printf '  \033[31m✗\033[0m %-52s -> %s (wanted %s)\n' "$desc" "$gotlabel" "$label"
    FAILS=$((FAILS + 1))
  fi
}

echo
log "BLOCKED — the cloud-credential-theft SSRF class (every reachable IMDS address/name):"
witness 1 "AWS/GCP/Azure IMDS via WebFetch"        --url "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
witness 1 "GCP metadata by NAME (DNS, not IP)"     --url "http://metadata.google.internal/computeMetadata/v1/"
witness 1 "AWS ECS task-metadata address"          --url "http://169.254.170.2/v2/credentials/"
witness 1 "Alibaba Cloud metadata address"         --command "curl -s http://100.100.100.100/latest/meta-data/"
witness 1 "AWS IMDSv6 (fd00:ec2::254)"             --url "http://[fd00:ec2::254]/latest/meta-data/"
witness 1 "bare metadata IP straight to curl"      --command "curl 169.254.169.254"
witness 1 "metadata fetch buried in a pipeline"    --command "echo go && curl http://169.254.169.254/ | tee creds.json"
witness 1 "host classifier (link-local /16)"       --host "169.254.0.1"
echo
log "ALLOWED — ordinary destinations a real session needs (NOT egress-blocked):"
witness 0 "the Anthropic API (the provider)"       --url "https://api.anthropic.com/v1/messages"
witness 0 "a public git clone over https"          --command "git clone https://github.com/anthony-chaudhary/fak.git"
witness 0 "a public host"                          --host "api.openai.com"
witness 0 "a private RFC1918 host (not link-local)" --host "10.0.0.5"
echo

if [ "$FAILS" -ne 0 ]; then
  log "$FAILS witness(es) FAILED"; exit 1
fi
log "all witnesses passed — the metadata/link-local SSRF class is refused by structure,"
log "every reachable IMDS address AND its DNS name, while real provider/git traffic flows."
log "wrap a live agent the same way: fak guard -- claude   (the floor rides into the VM)."
