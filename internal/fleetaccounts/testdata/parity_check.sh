#!/usr/bin/env bash
# One-shot parity proof: build a fixture account tree + sessions.json, run BOTH the Python
# tools/fleet_accounts.py and the native `fak fleet-accounts`, and diff their `json`
# envelopes (ignoring the time-dependent registry_age_min). Not run by `go test`; a manual
# witness for the byte-compatibility claim. Run from the repo root.
set -euo pipefail
FX="$(mktemp -d)"
trap 'rm -rf "$FX"' EXIT
mkdir -p "$FX/home/.claude/projects" "$FX/home/.claude-gem8-acct/projects" \
  "$FX/home/.claude-backup-acct/projects" "$FX/cfg/opencode-glm" \
  "$FX/home/.claude-noproj" "$FX/reg"
printf '%s' '{"oauthAccount":{"accountUuid":"uuid-default","emailAddress":"default@example.com","organizationUuid":"org-1","organizationType":"team","seatTier":"pro"}}' > "$FX/home/.claude/.claude.json"
printf '%s' '{"oauthAccount":{"accountUuid":"uuid-gem8","emailAddress":"gem8user@example.com","organizationUuid":"org-2","organizationType":"team"}}' > "$FX/home/.claude-gem8-acct/.claude.json"
printf '%s' '{"model":"zai-coding-plan/glm-5.2","small_model":"zai-coding-plan/glm-5.2-air"}' > "$FX/cfg/opencode-glm/opencode.json"
printf '%s' '{"generated_utc":"2026-06-29T12:00:00Z","throttle":{".claude-gem8-acct":{"reset":"Dec 31, 1pm"}},"auth":{},"sessions":[{"account":"opencode-glm","project":"work","disp":"LIVE","age_min":3}]}' > "$FX/reg/sessions.json"
export FLEET_USER_HOME="$FX/home" FLEET_CONFIG_HOME="$FX/cfg" FLEET_REG_DIR="$FX/reg" FLEET_POLICY_PATH="$FX/none.json"
go build -o "$FX/fak" ./cmd/fak
(cd tools && python3 fleet_accounts.py json) > "$FX/py.json"
"$FX/fak" fleet-accounts json > "$FX/go.json"
python3 - "$FX/py.json" "$FX/go.json" <<'PY'
import json, sys
py = json.load(open(sys.argv[1])); go = json.load(open(sys.argv[2]))
def strip(d):
    for r in d.get("accounts", []) + d.get("available_accounts", []):
        r.pop("registry_age_min", None)
strip(py); strip(go)
assert list(py) == list(go), "envelope key order differs"
ok = (py == go)
print("envelope value-equal:", ok)
print("n accounts:", len(py["accounts"]), len(go["accounts"]))
for p, g in zip(py["accounts"], go["accounts"]):
    pk = [k for k in p if k != "registry_age_min"]; gk = [k for k in g if k != "registry_age_min"]
    assert pk == gk, f"key order differs for {p.get('account')}: {pk} vs {gk}"
    if p != g:
        print("VALUE DIFF", p.get("account")); print(" py", p); print(" go", g)
print("PARITY OK" if ok else "PARITY FAIL")
sys.exit(0 if ok else 1)
PY
