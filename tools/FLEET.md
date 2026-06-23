# `fleet` — the operator console for the agent fleet

One command to see the agent fleet running on a host: which Claude Code sessions are
live, which stopped and why, which accounts you can actually resume on right now, and
what needs you. It is the session-health companion to `dos top` (the lane-occupancy
view) — run them side by side and you have both halves of "what is going on."

```
python tools/fleet.py            # the live view (Ctrl-C to quit)
python tools/fleet.py install    # drop a `fleet` wrapper on PATH, then just type `fleet`
```

## The two planes

An operator watches two different things, and they are easy to confuse:

| Question | Command | Plane |
| --- | --- | --- |
| Which dos.toml **lane** is held right now? | `dos top` | lane occupancy (locks / leases) |
| Which **session/account** is healthy, stopped, or resumable? | `fleet top` | session + account health |

`fleet top` is built to the same contract as `dos top` on purpose — a boxed,
live-refreshing header, chip glyphs, and `--once` / `--json` / `--interval` — so the
two read like one tool.

## Verbs

Everything after the verb is passed straight through to the underlying tool.

| Verb | Shows | Under the hood |
| --- | --- | --- |
| *(none)* / `top` | live session/account watchdog | `fleet_top.py` |
| `status` | one-shot snapshot of the same view | `fleet_top.py --once` |
| `json` | machine snapshot | `fleet_top.py --json` |
| `sessions` | per-session disposition table | `fleet_sessions.py summary` |
| `resume` | account-correct resume commands | `fleet_sessions.py resume` |
| `accounts` | worker-account availability | `fleet_accounts.py` |
| `pane` | the full control pane | `fleet_control_pane.py status` |
| `install` / `uninstall` | manage the `fleet` command on PATH | `install_agent_command.py` |
| `help` | the verb list | — |

## What the live view tells you

```
┌─ fleet top · C:\work\fak · 2026-06-23T18:33:44Z ─────────────
SESSIONS  706 in 10h window
  🟢 LIVE      8   live 8
  🔴 INFRA   154   rate_limit 109, auth 35, api_error 10
  🟡 HANGING   3   ambiguous_quiet 2, parked_on_task 1
  ⚪ AGENT   540   completed 431, killed_mid_turn 85, crash_mid_tool 24
ACCOUNTS  3/8 usable
  🟢 available  host-a, host-b, host-c
  🔴 throttled  host-d  resets Jun 26, 6pm
ATTENTION  2
  🔴 5 session(s) resumable on an available account
       $ $env:CLAUDE_CONFIG_DIR='...'; claude --resume <id> ...
  🟡 3 session(s) parked/quiet >30m
└─ 10h window · refresh 5s · Ctrl-C to quit ─
```

- **SESSIONS** — every session in the lookback window, bucketed by category
  (`LIVE` / `INFRA` / `HANGING` / `AGENT` / `USER`) with a cause breakdown.
- **ACCOUNTS** — worker accounts: usable now, throttled (with the reset), or blocked.
- **ATTENTION** — the ranked "what needs me now": a stopped autonomous session whose
  account is actually free (with the exact account-correct resume command), an account
  that needs an interactive `/login`, a throttle that means a resume would re-die.

## Defaults

- Bare `fleet` is the **live** view; `fleet status` is a single frame for a pipe or CI.
- Lookback window is **10h** (`--window`), refresh cadence **5s** (`--interval`).
- Color is auto: on for a terminal, off when piped or under `NO_COLOR` (so `--json`
  and `status` stay diffable). Force it off with `--no-color`.
- A long-parked `HANGING` session is flagged after **30m** (`FLEET_TOP_HANGING_MIN`).

## Install

`fleet install` writes thin `fleet` wrappers (`.cmd` + `.ps1` on Windows, a shell
script on POSIX) into `~/.local/bin` (or a system dir with `--system`). The wrappers
pin `FLEET_ROOT` to this clone and call `tools/fleet.py`, so `fleet` works from any
directory. It also puts that bin dir on `PATH` for you — appending to the persistent
user PATH on Windows (the registry value your shells read, not just the current
process) or to `~/.profile` on POSIX — so a bare `fleet` resolves in the next terminal
you open. Pass `--no-path` to skip that and manage PATH yourself. `--dry-run` shows the
plan; `fleet uninstall` removes the wrappers (and leaves PATH untouched, since other
tools share that bin dir). The installer refuses to clobber a non-`fleet` command of
the same name unless you pass `--force`.
