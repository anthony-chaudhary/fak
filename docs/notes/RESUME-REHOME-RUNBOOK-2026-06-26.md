# Resume & rehome runbook — restarting rate-limit-crashed Claude sessions (2026-06-26)

The durable procedure for restarting Claude Code sessions that crashed on a rate
limit, rehoming them onto a healthy account, and tracking them to completion. It
indexes the existing fleet tooling and records the non-obvious seams that cost a
session to rediscover.

> **Owes a Go port.** Every tool below is legacy `tools/*.py`, frozen by the
> `internal/pythongate` ratchet. The durable home for this is a `cmd/fak` verb
> (`fak resume sweep|plan|run|audit`) backed by `internal/resume`; `fak accounts`
> already models the rehome/identity layer in Go. Do not extend the Python — port
> the touched slice. Blocked while `cmd/fak` does not compile (peer WIP).

## The pipeline (four stages, read-only until the launcher)

| Stage | Tool | What it does |
|---|---|---|
| Discover | `tools/resume_sweep.py --json` | Manifest-free walk of every `~/.claude*` transcript; buckets each crashed session: `LIMIT_RESET_PASSED` (resumable now), `LIMIT_RESET_FUTURE` (wait for reset), `API_ERR` (transient, resumable), `AUTH` (needs login), `LIVE` (leave alone). |
| Plan | `tools/fleet_sessions.py registry --window H --probe blocked` | Authors `_registry/resume_plan.json`: decides the resume account per session, **rehoming** throttled/disabled sessions onto a healthy admissible target. Writes `sessions.json` too. |
| Launch | `tools/fleet_resume_watchdog.py --live` | Gated, once-ever launcher. Refreshes the plan itself, copies each rehomed transcript into the target account dir, sets `CLAUDE_CONFIG_DIR`, spawns `claude --resume <sid> -p "<resume prompt>"`. Dry-run by default; `FAK_LIVE=1`/`--live` to act. |
| Audit | `tools/resume_relaunch_audit.py --json` | Verifies OUTCOME from the transcript, not the ledger's word: `RELAUNCHED_OK` (last real turn newer than last error) vs `STRANDED` (still on the error, classified `AUTH`/`LIMIT`/`API_ERR`). |

The always-on `FleetResumeWatchdog` scheduled task runs the Launch stage on a
~10-minute cadence, so the loop continues unattended. `FleetStrandedRecovery` is a
sibling task for stranded sessions.

## Making an account an admissible rehome target (the day26 lesson)

A fresh healthy account (call it `<acct>`) is **invisible to the planner** until two
gates pass. A real worked example on 2026-06-26: a brand-new `max`-subscription seat
had valid creds but the planner would not route onto it:

1. **Account marker.** `fleet_accounts._discover_claude` classifies a `.claude*`
   dir as `non-account` if it has **no `projects/` subdir**. Fix:
   `mkdir ~/.claude-<acct>/projects/`. Now it classifies as `worker`.
2. **Positive-evidence verdict.** `_admissible_targets` admits load only onto an
   account whose `verdict_source` is `probe` or `passive` (`_has_positive_evidence`).
   A brand-new account has `none` → rejected. Fix: probe it —
   `python tools/account_probe.py --account <acct>` (or the planner's `--probe all`).
   A clean `OK` probe stamps a fresh `probe` row (observed: `OK` in 3.4s, `pong`).

After both: `ADMISSIBLE rehome targets: ['.claude-<acct>']`. A note + a
`route_weights` lift in the gitignored host `_registry/accounts_policy.json` make the
preference durable across sessions; the watchdog's per-tick `--probe` keeps the
verdict fresh so it stays in the pool. (The live host policy and the specific account
identity are gitignored / host-local — never commit a real account name to this repo.)

## The "organization has disabled Claude subscription access" message is MISLEADING

It is **not** real org policy — it is the OAuth token-twin smear (a login under a
different identity overwrites a dir's `.oauth-token`). Proof: a clean seat on the
same org (day26) probes `OK`. So `AUTH`-bucket crashes are curable by rehoming to a
clean seat, not write-offs. Cross-check with `fak accounts check-twins` (a benign
same-account `.claude`↔`.claude-gem8-netra` pair is expected and not a smear).

## Inter-launch spacing — do not fire resumes back-to-back

The launcher used to spawn all `MAX_PER_TICK` resumes within the same second. A
burst that big trips the server-side `API Error: Server is temporarily limiting
requests (not your usage limit)` 529 — every freshly-resumed session gets one turn
out, then strands on that transient wall (observed 2026-06-26: a cap-4 then cap-8
batch stranded their whole sets on identical same-second errors). Fixed by
`FAK_LAUNCH_SPACING_SEC` (default 8s) — a sleep between spawns so the shared rate
budget refills. **Operationally: one paced tick, then wait. Synchronous ticks
re-strand on the same 529.**

## Failure-class disposition

- **API_ERR** (transient 529 / upstream model error) — self-heals; `resume_blocked`
  keeps it eligible (outcome `recoverable`) up to `MAX_ATTEMPTS` (3). Retries on the
  next watchdog tick. The spacing fix prevents most of these.
- **LIMIT** (real per-account usage cap, carries a reset window) — `DEFER`; resumes
  automatically when its window opens (the watchdog re-probes blocked accounts each
  tick). Not stranded permanently.
- **AUTH** (login/access wall) — `unrecoverable` for re-resume; needs a human
  `/login` OR a rehome onto a clean seat (see the smear note above).

## fak context-vs-context

The rehome physically carries each transcript **plus** its sidecar subagents and the
slug-scoped agent-memory store (`memory_cotravel`, gated by `FAK_MEMORY_COTRAVEL`) to
the target account, and the resume prompt ("Resume where you left off; re-establish
any /goal or /loop and continue toward it") tells the session to reload its working
set and continue its goal rather than start cold.

## One-shot operator recipe

```sh
# 1. see the crashed set
python tools/resume_sweep.py --window 1440 --json | less

# 2. (one-time per new target) make a fresh account admissible
mkdir -p ~/.claude-<name>-netra/projects
python tools/account_probe.py --account <name>

# 3. plan + launch one paced tick (cap small; spacing on by default)
FLEET_REG_DIR=tools/_registry FAK_LIVE=1 FAK_MAX_PER_TICK=3 FAK_WINDOW_H=24 \
  python tools/fleet_resume_watchdog.py

# 4. verify outcomes (scope to your sids; the full-fleet audit walks ~1700 files and is slow)
python tools/resume_relaunch_audit.py --json

# then WAIT — the FleetResumeWatchdog task continues on its cadence; do not re-fire ticks.
```
