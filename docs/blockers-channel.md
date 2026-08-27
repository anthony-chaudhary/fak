---
title: "The #blockers channel"
description: "How the fak fleet uses the #blockers channel: the one place any agent or CI run records what is currently stopping progress, so blockers stay visible."
---

# The #blockers channel

`#blockers` is the one place the fleet records what is stopping progress. Any agent, CI
job, or background loop that hits a wall posts it here, so an operator watching one
channel sees both the ongoing impediments and the few that actually need a human — and
isn't paged for the ones that don't.

It is a **Slack status surface**, the twin of `#scoreboard` / the bench channel — an
OUTBOUND post, never an inbound listener. fak does not take orders from a Slack message
here; it posts a blocker it already detected locally. It lives in the public scoreboard
workspace (team `T0BDEJF1HGB`) and posts with the same `FAK_SCOREBOARD_TOKEN` bot —
never the private lab `SLACK_BOT_TOKEN`. By default it lands in the shared **CI/CD
reporting sink** (`C0BGQ411TCJ`, `scoreboard.CICDReportChannel`) alongside the rest of
the status feeders (see [decisions/cicd-reporting-slack-sink.md](decisions/cicd-reporting-slack-sink.md));
set `FAK_BLOCKERS_CHANNEL` to split blockers back into their own room.

## Two tiers: background vs surfaced

The whole reason this channel is distinct from the status feeders is **severity-as-
surfacing**. A blocker renders at one of three states, and the state decides how loud the
post is:

| Severity | Glyph | Pages anyone? | Use it for |
|---|---|---|---|
| `status` | :hourglass_flowing_sand: | no | an ongoing, tracked impediment — recorded, scrolls by. "GPU-gated, waiting on GPU server hours", "peer merge in flight". |
| `operator` | :rotating_light: | **yes** — `<!here>` or a named owner | a blocker that needs a **human** to act, with a "do this next". "FAK_SCOREBOARD_TOKEN missing", "CPU server host unreachable — needs a manual restart". |
| `clear` | :white_check_mark: | no | an all-clear heartbeat — only after a successful query returns no open blockers. |
| `unknown` | :warning: | no | the blocker source could not be evaluated; the feeder records the diagnostic and fails without posting or claiming all-clear. |

Only `operator` is surfaced; `status`, `clear`, and `unknown` are the background tiers. An operator
blocker carries the broadcast mention in **both** the notification fallback and the lead
block, which is what makes Slack actually page — see `internal/blockerpost/render.go`.

## Posting a blocker (`fak blockers post`)

```bash
# background — an ongoing impediment, recorded quietly, pages no one
fak blockers post --title "GPU-gated, waiting on GPU-server hours" \
  --detail "Rungs 1/2/3/5 need the private GPU server." --ref "#921"

# surfaced — needs a human; pages the channel's active members (<!here>)
fak blockers post --severity operator --title "CPU host unreachable" \
  --detail "CPU GLM-5.2 node not responding." \
  --action "restart the CPU-host serve" --action-url "https://…/runbook"

# surfaced to ONE person instead of <!here>
fak blockers post --severity operator --owner "<@U0OPERATOR>" \
  --title "FAK_SCOREBOARD_TOKEN missing" --detail "the feeders are running dry-run only"
```

`--dry-run` renders the card to stdout and posts nothing — safe to preview anywhere. To
page a specific person, pass `--owner` a real Slack mention token (`<@U…>`); a bare name
will show but won't notify.

## The daily heartbeat (`fak blockers feed`)

`blockers-feed.yml` runs daily and folds the open GitHub backlog filtered to the blocker
label (`FAK_BLOCKERS_LABEL`, default `operator-agentic-blocked`) into one card. The surfacing follows
ownership — the honest dual of "status in background, operator surfaced":

- **Successful query, 0 blocked issues** → a quiet green all-clear card (no page).
- **Missing configured label or failed/invalid query** → an `unknown` diagnostic and
  failed workflow; no all-clear and no Slack post.
- **≥1 with an UNOWNED issue** → an `operator` card: `<!here>` + a triage link (paged) —
  a blocker with no assignee needs a human to pick it up.
- **≥1 but all assigned** → a muted background-`status` card (ownership recorded, no
  inference that work is active or that no action is needed).

`fak blockers feed --issues <gh-json>` is a pure fold of a
`gh issue list --json number,title,url,assignees,labels` payload, so it is unit-tested
without `gh` or the network.

## Configuration

Resolution mirrors the other feeders' `.env.slack.local` idiom (one gitignored file
configures every workspace):

| Key | Default | Meaning |
|---|---|---|
| `FAK_BLOCKERS_TOKEN` | falls back to `FAK_SCOREBOARD_TOKEN` | the bot token (never the lab `SLACK_BOT_TOKEN`). |
| `FAK_BLOCKERS_CHANNEL` | `C0BGQ411TCJ` (CI/CD reporting sink) | target channel; falls back to the family sink, then `FAK_CICD_REPORT_CHANNEL`; never inherits `FAK_SCOREBOARD_CHANNEL`. |
| `FAK_BLOCKERS_LABEL` | `operator-agentic-blocked` | the feeder's issue-label filter (repo variable). |

**Operator one-time step:** add the repo secret `FAK_SCOREBOARD_TOKEN` (the
scoreboard-workspace bot token) under *Settings → Secrets and variables → Actions*.
Without it the feeder runs `--dry-run` and writes the card to the step summary, so a fork
or secret-less run never hard-fails.
