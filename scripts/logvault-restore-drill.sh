#!/usr/bin/env bash
# logvault-restore-drill.sh — the restore-path cadence drill for `fak logvault`
# (#2453, part of epic #2447). A backup nobody has restored from is a hypothesis:
# this drill RESTORES one captured source into a temp dir and VERIFIES it (re-hash
# every restored byte against the manifest chain + re-run the chained-journal
# verifiers), on a cadence, so the restore path cannot rot unnoticed.
#
# Hermetic by default: it seeds a throwaway source tree + vault under a temp dir
# and points HOME at an empty temp home, so it runs anywhere (CI, a fresh box)
# without ever reading the operator's live vault or live state. That makes it safe
# to wire into a green gate or a nightly cadence.
#
# For an operator cadence run against the REAL vault instead, build fak and run the
# drill verb directly, appending a durable row to the repo ledger:
#   fak logvault -repo <repo> capture
#   fak logvault drill -ledger docs/nightrun/logvault-drill.jsonl
#
# Exit 0 only on a DRILL PASS (0 hash mismatches, every restored chained journal
# clean); any restore refusal or mismatch fails the drill, red, on purpose.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d 2>/dev/null || mktemp -d -t fak-logvault-drill)"
trap 'rm -rf "$work"' EXIT

# A built fak: use $FAK if provided (an operator/CI may pass a prebuilt binary),
# else build one into the temp dir so the drill is fully self-contained.
FAK="${FAK:-}"
if [ -z "$FAK" ]; then
	FAK="$work/fak"
	( cd "$root" && go build -o "$FAK" ./cmd/fak )
fi

repo="$work/repo"
vault="$work/vault"
# An empty HOME / config dir so the home- and config-rooted DefaultSources (harness
# store, user ledgers, per-user CLI state) resolve as absent — the drill captures
# ONLY the seeded source, never live state. Cover every var Go's os.UserHomeDir /
# os.UserConfigDir read across hosts: HOME + USERPROFILE (home), XDG_CONFIG_HOME +
# AppData (config, Linux + Windows).
export HOME="$work/home"
export USERPROFILE="$work/home"
export XDG_CONFIG_HOME="$work/home/.config"
export APPDATA="$work/home/AppData"
export LOCALAPPDATA="$work/home/LocalAppData"
mkdir -p "$repo/.dispatch-runs/drill" "$HOME" "$XDG_CONFIG_HOME" "$APPDATA" "$LOCALAPPDATA"

# Seed one captured source: a plain append-only log that round-trips by re-hash.
printf 'drill-row-1\ndrill-row-2\ndrill-row-3\n' > "$repo/.dispatch-runs/drill/loops.jsonl"

echo "logvault-restore-drill: capture seeded source -> temp vault ($vault)"
"$FAK" logvault capture -repo "$repo" -vault "$vault"

echo "logvault-restore-drill: drill (restore one source into a temp dir + verify)"
out="$("$FAK" logvault drill -vault "$vault")"
echo "$out"

if ! printf '%s\n' "$out" | grep -q "DRILL PASS"; then
	echo "logvault-restore-drill: FAILED — restore drill did not report DRILL PASS" >&2
	exit 1
fi
echo "logvault-restore-drill: OK — the restore path round-trips clean"
