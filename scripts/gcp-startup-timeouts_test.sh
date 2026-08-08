#!/usr/bin/env bash
# gcp-startup-timeouts_test.sh - offline regression for #3479.
#
# The defect: the unattended GCE startup scripts fetched the tailscale installer
# (`curl ... | sh`) and the NVIDIA cuda-keyring .deb with NO `--max-time` and no
# `--connect-timeout`, while sibling curls in the SAME script carried `--max-time
# 30/5`. Those startup scripts run sequentially under `set -euxo pipefail` on a
# freshly-created (expensive, GPU) VM, so one stalled TCP connect wedges the whole
# provisioning run — including the idle-GPU and budget reapers appended *after* it,
# which are exactly the things that would have reaped the wedged VM.
#
# The rule this pins, over the three unattended startup scripts:
#   (1) EVERY curl invocation carries `--max-time` (a wall-clock ceiling), and
#   (2) every curl that fetches a LITERAL external http(s):// url additionally
#       carries `--connect-timeout` (so a black-holed SYN fails fast, not at the
#       full --max-time), and
#   (3) the two named download sites are still present and bounded.
#
# It is a static scan of the checked-in text, so it needs no network, no gcloud,
# and no GCP project. Run: bash scripts/gcp-startup-timeouts_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The three unattended startup scripts named by #3479. Their curls run with no
# operator watching and no `--max-run-duration` cap on the instance, so an
# unbounded one has no outer reaper.
SCRIPTS=(gcp-glm-serve.sh gcp-qwen-serve.sh gcp-dogfood-control-vm.sh)

fails=0
pass() { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; fails=$((fails + 1)); }

# curl_lines prints "<lineno>:<line>" for every real curl INVOCATION in a file.
# Requiring a `-` flag right after the word is what keeps `apt-get install -y
# ... curl build-essential` (curl as a PACKAGE NAME) out of the scan.
curl_lines() {
  grep -nE '(^|[|;&(]|[[:space:]])curl[[:space:]]+-' "$1"
}

for s in "${SCRIPTS[@]}"; do
  src="$HERE/$s"
  if [[ ! -f "$src" ]]; then
    fail "$s: script exists"
    continue
  fi

  found=0
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    found=$((found + 1))
    lineno="${entry%%:*}"
    line="${entry#*:}"

    # (1) every curl invocation needs a wall-clock ceiling.
    if [[ "$line" != *"--max-time"* ]]; then
      fail "$s:$lineno: curl has no --max-time -> can hang the startup script forever: $line"
      continue
    fi

    # (2) a literal external url is an unattended DOWNLOAD: bound the connect too.
    if [[ "$line" == *http://* || "$line" == *https://* ]]; then
      if [[ "$line" != *"--connect-timeout"* ]]; then
        fail "$s:$lineno: external download has no --connect-timeout: $line"
        continue
      fi
      pass "$s:$lineno: external download bounded (--connect-timeout + --max-time)"
    else
      pass "$s:$lineno: curl bounded (--max-time)"
    fi
  done < <(curl_lines "$src")

  if [[ "$found" -eq 0 ]]; then
    fail "$s: scan found no curl invocations at all (regex drifted?)"
  fi
done

# (3) The two download sites #3479 named must still be there AND bounded — so a
# future edit cannot "pass" rule (1)/(2) by simply deleting the fetch.
assert_bounded_site() {
  local script="$1" marker="$2" what="$3"
  local line
  line="$(grep -F -- "$marker" "$HERE/$script" | grep -E '(^|[|;&(]|[[:space:]])curl[[:space:]]+-' | head -n 1)"
  if [[ -z "$line" ]]; then
    fail "$script: $what download site not found (marker: $marker)"
    return
  fi
  if [[ "$line" == *"--max-time"* && "$line" == *"--connect-timeout"* ]]; then
    pass "$script: $what download carries --connect-timeout + --max-time"
  else
    fail "$script: $what download is UNBOUNDED: $line"
  fi
}

assert_bounded_site gcp-glm-serve.sh          'tailscale.com/install.sh'   'tailscale-install'
assert_bounded_site gcp-qwen-serve.sh         'tailscale.com/install.sh'   'tailscale-install'
assert_bounded_site gcp-dogfood-control-vm.sh 'tailscale.com/install.sh'   'tailscale-install'
assert_bounded_site gcp-glm-serve.sh          'cuda-keyring_1.1-1_all.deb' 'cuda-keyring'

if [[ "$fails" -eq 0 ]]; then
  echo "PASS - gcp startup-script downloads are all bounded (#3479)"
  exit 0
fi
echo "FAILURES: $fails"
exit 1
