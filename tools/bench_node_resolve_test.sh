#!/usr/bin/env bash
# bench_node_resolve_test.sh -- hermetic witness for tools/bench_node.sh's dynamic-IP
# resolver (#12). Proves the runner asks the provisioner for an EPHEMERAL host's CURRENT
# IP just-in-time (resolve_cmd) instead of relying on a static tailnet_ip that rotates per
# launch. No network: every resolve_cmd is a local stub. Run: bash tools/bench_node_resolve_test.sh
set -uo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SELF_DIR/bench_node.sh"
fail=0
t(){ printf '  %-46s ' "$1"; }
ok(){ echo "ok"; }
no(){ echo "FAIL: $1"; fail=1; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
REGF="$TMP/reg.json"
reg(){ cat > "$REGF"; }
run(){ BENCH_NODES="$REGF" BENCH_RUNS_DIR="$TMP/runs" bash "$RUNNER" "$@"; }   # echoes exit via $?

echo "bench_node resolve_cmd tests:"

t "bash -n syntax"
if bash -n "$RUNNER"; then ok; else no "syntax error"; fi

# 1) a connecting subcommand resolves the IP just-in-time (resolve_cmd is actually invoked)
SENT="$TMP/invoked.sentinel"; rm -f "$SENT"
reg <<JSON
{ "schema":"fleet-bench-nodes/1","nodes":[
  { "name":"eph","sanitized_name":"node-eph",
    "resolve_cmd":"printf 127.0.0.1; : > '$SENT'",
    "ssh_user":"u","ssh_key":"~/.ssh/id_ed25519","ssh_port":22,"host_key":"",
    "repo_path":"~/r","toolchain_env":"GOTOOLCHAIN=auto" } ]}
JSON
t "resolve_cmd invoked on a connecting subcommand"
run eph ping >/dev/null 2>&1
if [ -f "$SENT" ]; then ok; else no "resolve_cmd was never run (no sentinel)"; fi

# 2) an empty resolver result is REFUSED (exit 5) -- never silently ssh to ""
reg <<JSON
{ "schema":"fleet-bench-nodes/1","nodes":[
  { "name":"eph","sanitized_name":"node-eph","resolve_cmd":"true",
    "ssh_user":"u","ssh_key":"~/.ssh/id_ed25519","ssh_port":22,"host_key":"",
    "repo_path":"~/r","toolchain_env":"GOTOOLCHAIN=auto" } ]}
JSON
t "empty resolver result refused (exit 5)"
run eph ping >/dev/null 2>&1; rc=$?
if [ "$rc" -eq 5 ]; then ok; else no "expected exit 5, got $rc"; fi

# 3) a multi-line resolver result is REFUSED (ambiguous -- must print exactly one host/IP)
reg <<JSON
{ "schema":"fleet-bench-nodes/1","nodes":[
  { "name":"eph","sanitized_name":"node-eph","resolve_cmd":"printf '1.2.3.4\\n5.6.7.8\\n'",
    "ssh_user":"u","ssh_key":"~/.ssh/id_ed25519","ssh_port":22,"host_key":"",
    "repo_path":"~/r","toolchain_env":"GOTOOLCHAIN=auto" } ]}
JSON
t "multi-line resolver result refused (exit 5)"
run eph ping >/dev/null 2>&1; rc=$?
if [ "$rc" -eq 5 ]; then ok; else no "expected exit 5, got $rc"; fi

# 4) info is OFFLINE-SAFE: it reports ip_source=resolve_cmd WITHOUT calling the provisioner
SENT2="$TMP/info_should_not_invoke.sentinel"; rm -f "$SENT2"
reg <<JSON
{ "schema":"fleet-bench-nodes/1","nodes":[
  { "name":"eph","sanitized_name":"node-eph",
    "resolve_cmd":"printf 127.0.0.1; : > '$SENT2'",
    "ssh_user":"u","ssh_key":"~/.ssh/id_ed25519","ssh_port":22,"host_key":"",
    "repo_path":"~/r","toolchain_env":"GOTOOLCHAIN=auto" } ]}
JSON
t "info is offline-safe + reports ip_source=resolve_cmd"
out="$(run eph info 2>/dev/null)"
if echo "$out" | grep -q "ip_source=resolve_cmd" && [ ! -f "$SENT2" ]; then ok
else no "info invoked the resolver or omitted ip_source (out: $out)"; fi

# 5) a static tailnet_ip node still works with no resolve_cmd (back-compat) + ip_source=static
reg <<JSON
{ "schema":"fleet-bench-nodes/1","nodes":[
  { "name":"stat","sanitized_name":"node-stat","tailnet_ip":"203.0.113.7",
    "ssh_user":"u","ssh_key":"~/.ssh/id_ed25519","ssh_port":22,"host_key":"",
    "repo_path":"~/r","toolchain_env":"GOTOOLCHAIN=auto" } ]}
JSON
t "static tailnet_ip back-compat (ip_source=static)"
out="$(run stat info 2>/dev/null)"
if echo "$out" | grep -q "sanitized=node-stat" && echo "$out" | grep -q "ip_source=static"; then ok
else no "static node info failed (out: $out)"; fi

# 6) Windows OpenSSH can launch cmd.exe as the login shell. The runner must still
# handshake with a cmd-safe no-op, then pipe real run scripts into WSL bash.
FAKEBIN="$TMP/bin"; mkdir -p "$FAKEBIN"
cat > "$FAKEBIN/ssh" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${SSH_LOG:?}"
if [ -n "${SSH_STDIN_LOG:-}" ]; then
  cat >> "$SSH_STDIN_LOG"
else
  cat >/dev/null
fi
exit 0
SH
chmod +x "$FAKEBIN/ssh"
SSH_LOG="$TMP/ssh.log"; : > "$SSH_LOG"
reg <<JSON
{ "schema":"fleet-bench-nodes/1","nodes":[
  { "name":"win","sanitized_name":"node-win","tailnet_ip":"203.0.113.9",
    "ssh_user":"u","ssh_key":"~/.ssh/id_ed25519","ssh_port":22,"host_key":"",
    "remote_shell":"windows-wsl","wsl_distro":"Ubuntu",
    "repo_path":"~/r","toolchain_env":"GOTOOLCHAIN=auto" } ]}
JSON
t "windows-wsl ping uses a cmd-safe probe"
PATH="$FAKEBIN:$PATH" SSH_LOG="$SSH_LOG" run win ping >/dev/null 2>&1
if grep -q "cmd /c exit 0" "$SSH_LOG"; then ok
else no "windows-wsl ping did not use cmd probe (log: $(cat "$SSH_LOG"))"; fi

: > "$SSH_LOG"
t "windows-wsl run scripts enter WSL bash"
PATH="$FAKEBIN:$PATH" SSH_LOG="$SSH_LOG" run win cmd 'echo OK' >/dev/null 2>&1
if grep -q "wsl.exe -d Ubuntu -e bash -s" "$SSH_LOG"; then ok
else no "windows-wsl cmd did not use WSL bash (log: $(cat "$SSH_LOG"))"; fi

SSH_STDIN_LOG="$TMP/ssh.stdin"; : > "$SSH_STDIN_LOG"
reg <<JSON
{ "schema":"fleet-bench-nodes/1","nodes":[
  { "name":"mac","sanitized_name":"node-macos-a","tailnet_ip":"203.0.113.10",
    "ssh_user":"u","ssh_key":"~/.ssh/id_ed25519","ssh_port":22,"host_key":"",
    "repo_path":"~/r","toolchain_env":"GOTOOLCHAIN=auto" } ]}
JSON
t "keepawake sends caffeinate in remote script"
PATH="$FAKEBIN:$PATH" SSH_LOG="$SSH_LOG" SSH_STDIN_LOG="$SSH_STDIN_LOG" BENCH_KEEPAWAKE_S=123 run mac keepawake >/dev/null 2>&1
if grep -q "nohup caffeinate -dimsu -t \"123\"" "$SSH_STDIN_LOG" && grep -q "KEEP_AWAKE armed duration_s=123" "$SSH_STDIN_LOG"; then ok
else no "keepawake did not arm caffeinate (stdin: $(cat "$SSH_STDIN_LOG"))"; fi

echo
if [ "$fail" -eq 0 ]; then echo "PASS (bench_node resolve_cmd)"; exit 0
else echo "SOME TESTS FAILED"; exit 1; fi
