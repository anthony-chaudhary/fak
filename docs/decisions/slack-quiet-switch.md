---
title: "The global Slack quiet switch — one variable mutes every outbound surface"
description: "The decision to add FAK_SLACK_QUIET, a single opt-in brake that silences every outbound Slack write in the fleet without editing a surface, holds queued outbox rows instead of destroying them, and re-grades the health verdict to QUIET so the brake cannot manufacture its own alarm."
---

# The global Slack quiet switch

## The problem

fak's outbound Slack surface grew one feeder at a time. There are now **19
workflow env blocks** that arm a post, a dozen `cmd/fak` verbs that post directly,
a durable outbox that drains in the background, a chatrelay bridge, and a handful
of non-Go scripts. Each was added with its own gate, and the gates are all
**positive** — they answer "may I post?" and default to yes.

That is the right default for a fleet that wants to be heard, and the wrong one
for an operator who needs it to be **silent** for a while: to work through a
backlog, to stop a noisy loop from paging, or to debug a feeder without a
hundred cards landing in the channel. Before this switch the only ways to go
quiet were to revoke the bot token (which breaks every surface at once, including
the ones you want to keep reading), to disable 19 workflows by hand (and remember
all 19 to re-enable), or to let it run and ignore it.

## Decision

Add **one** switch — `FAK_SLACK_QUIET` — that every outbound Slack write consults.
Set it and the fleet goes quiet; unset it and nothing changes.

```bash
# Quiet the fleet (process environment) …
export FAK_SLACK_QUIET=1

# … or, for a whole workspace at once, in the gitignored file the operator
# already uses for the token and channel:
echo 'FAK_SLACK_QUIET=1' >> .env.slack.local

# Disengage: unset it, or set it to any non-on value.
unset FAK_SLACK_QUIET
```

For CI, set the **repository variable** `FAK_SLACK_QUIET` (Settings → Secrets and
variables → Actions → Variables). Every Slack feeder workflow binds it into its
job environment, so one variable mutes all of them. It is a variable, not a
secret: the value is a single word, and a variable is what an operator can flip
from the UI without re-encrypting.

### Polarity: truthy means quiet

`1` / `on` / `true` / `yes` / `enable` / `enabled` engage the brake. **Unset,
blank, or anything unrecognized leaves notifications flowing.**

This is deliberately the *opposite* of the fleet's other default-on guards
(`internal/gardenbundle`'s `offValues`, `cmd/fak`'s `fleetEnvSwitchOn`), and the
reason is that this switch's ON state is silence. Those guards are named for a
feature that is on by default, so an off-ish word disables it. Applied here, that
grammar would make an **unset** `FAK_SLACK_QUIET` mean "muted" — the fleet would
go silent on a fresh clone, which is the opposite of the fail-loud default every
other surface has. So the vocabulary is the opt-in one, and the value an operator
will actually type for a variable named `QUIET` is the one that works. A typo
fails toward the pre-existing behavior rather than silently muting the fleet.

## How it is wired

The brake is one package, `internal/slackquiet`, and one injected option.

- **`internal/slackquiet`** (tier 2) owns the switch: the vocabulary, the
  resolution through `internal/slackenv` (so it is settable from the environment
  *or* `.env.slack.local`), and the diagnostic `Source()`.
- **`internal/slackwire`** (tier 1, the ONE Slack transport) takes the brake as an
  injected `Option` — `slackwire.WithQuiet(slackquiet.Engaged)` — and refuses every
  write with `ErrQuiet` before any HTTP. It cannot import `slackquiet` itself: a
  tier-1 primitive may import only `abi`, and the transport should not know policy
  anyway. **Reads stay live** (`History`, `Replies`, `AuthTest` are untouched), so
  a muted fleet can still be inspected.
- **The composition layer** passes `Engaged` in at every `slackwire.New` call
  site: `internal/scoreboard`, `internal/chatrelay`, `cmd/fak/slack.go`,
  `cmd/fak/slack_outbox.go`, `cmd/fak/loop.go`. Because scoreboard is the
  publisher most feeders ride, wiring it there mutes the whole family without
  touching a single feeder.

### The outbox HOLDS, it does not drain

`internal/slackoutbox` is the durable spool. Under the brake it **holds** every
pending row: no attempt, no state transition, no `sending` intent marker, and
`DrainReport.Held` / `ReapReport.Held` count them. Holding rather than draining is
load-bearing:

- Draining would **burn the attempt budget**. Five quiet drains — twice
  `MaxAttempts` — would dead-letter the entire queue, so unmuting would deliver
  nothing.
- Draining would write a `sending` marker that the next drain treats as an
  ambiguous post and probes for.

Held rows are byte-identical to how they were enqueued and deliver on the first
drain after the operator disengages. `Reap` (chat.delete) is held too: a quiet
fleet must not keep mutating channels by deleting its own history.

### The health verdict re-grades to QUIET

`fak slack health` classifies each surface `OK | INCOMPLETE | AUTH_FAIL | STALE |
DEFERRED | DEAD_ROWS | STALLED`. Under the brake, `STALE`, `STALLED`, and
`DEAD_ROWS` re-grade to **`QUIET`** — expected and non-actionable, so it never
trips the gate.

Without this rung the brake would manufacture its own alarm: one budget window of
silence reddens every cadence surface and the outbox, and `slack-watchdog.yml`
would file a GitHub issue *and* post an in-channel beat about the very silence the
operator asked for. The original verdict is preserved in the row's `Detail`, so
the diagnosis is not thrown away.

**`AUTH_FAIL` and `INCOMPLETE` deliberately survive the brake.** A rotated token
and a config drift are real problems whether or not we are posting — the operator
must learn about them *before* unmuting, or the first post after the quiet window
silently goes nowhere.

## Known bypasses — surfaces this switch does NOT mute

The brake covers every write that goes through `internal/slackwire`. It is **not**
a chokepoint for the whole fleet, and an operator who sets `FAK_SLACK_QUIET=1`
expecting total silence should know exactly what still speaks. An adversarial
audit of all 174 notification surfaces across both repos found these:

| surface | repo | why it bypasses |
| --- | --- | --- |
| `tools/slack_post.py` | public | its own stdlib-only Python poster; builds the URL and calls `urllib.request.urlopen` directly |
| `tools/node_report.py` | private | its own `_slack_call` over `urllib` |
| `tools/dgxbridge/internal/slack.go` | private | the GPU server bridge's hand-rolled client (`chat.postMessage` / `chat.delete`) |
| `ops/glm52_local_ud_post.sh` | private | a raw `curl` to `chat.postMessage` |
| `platform/alertnotify` | private | an ntfy-compatible webhook, not Slack at all — a different transport with its own topic URL |

These are **not** covered by design, not by oversight: each is a separate
transport in a separate language or process, and folding them onto `slackwire`
would be a much larger change than a mute switch. They are listed here so the
switch's coverage is honest. If total silence is required, these five must be
handled separately (unset their token, or stop their caller).

Two notes on the audit that produced this table, because a bypass list is only
useful if it is exact:

- **`tools/fleet_slack_status.py` is a caller, not a bypass.** It posts the fleet
  roll-up, but it does so *through* `slack_post.py`'s resolver and invents no
  transport of its own — so it is already covered by the first row. Muting
  `slack_post.py` mutes it too.
- **`experiments/fakrpc/fakrpcd.sh` was cited and does not exist.** It is recorded
  here as a stale citation rather than a bypass, so a future reader does not go
  looking for a file that was never there.

## What this is not

- **Not a kill switch for the token.** The token still resolves and `auth.test`
  still runs; the brake refuses at the write, so a muted fleet can still be
  diagnosed.
- **Not a per-surface switch.** That is the status quo this replaces — N switches
  to flip and N to forget.
- **Not a queue destroyer.** Held rows survive; see above.
- **Not retroactive.** It does not delete anything already posted. Use
  `fak slack outbox reap` for ephemeral cleanup.

## Witness

- `internal/slackquiet` — vocabulary, polarity, and both resolver halves.
- `internal/slackwire/slackwire_quiet_test.go` — every write refuses with
  `ErrQuiet` and the test server sees **zero** requests; reads still reach it.
- `internal/slackoutbox/quiet_test.go` — rows are held with `Attempts == 0` after
  ten quiet drains, then deliver on the first unbraked drain.
- `cmd/fak/slack_health_quiet_test.go` — faults re-grade `QUIET`, the gate stays
  open, and `AUTH_FAIL` survives.
- `cmd/fak/scoreboardpost_quiet_test.go` — a braked post is a clean skip that
  needs neither channel nor token, and is never spooled.
- `.github/workflows/slack_quiet_switch_test.go` — every workflow that binds
  `FAK_SCOREBOARD_TOKEN` also binds `FAK_SLACK_QUIET`, so a new feeder cannot
  silently miss the switch.
