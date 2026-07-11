# Catch-up scorecard — one 0..1 "how caught up is the dev system" number (2026-07-11)

`fak score catchup` folds the dev system's **backlog/pace** into one control-pane
card: not "how much happened over a window" (that is `fak cadence`), but **how far
BEHIND the dev system is right now, at each level** — a number a human glances at,
a gate ratchets on, and `fak scoreboard post` ships to Slack.

## The levels

The dev system is caught up (or behind) at several independent LEVELS, each with a
0..1 caught-up fraction (1.0 == fully caught up) **and** a raw, unbounded behind
count in its own unit:

| level | caught-up fraction | behind unit | source wired today |
|-------|--------------------|-------------|--------------------|
| `intake` | triaged / open | untriaged items | `--intake-behind/--intake-total` flags |
| `measurement` | measured / all cards | unmeasured cards | `--scores-from <control-pane.json>` |
| `index` | fresh entities / declared | stale index entities | **auto** — `devindex.CheckFreshness` over the tree (no network) |
| `trunk` | green / all checks | blocking build checks | `--trunk-behind/--trunk-total` flags |
| `loops` | on-cadence / all loops | overdue loops | `--loops-behind/--loops-total` flags |

Every level is **nil-able**: a level with no evidence this run is EXCLUDED from the
fold (never scored 0), exactly like the nil-able cache families in
`ComposeCacheHealth`. The `index` level is the deterministic star — it auto-collects
from the dev self-index and needs no flags; the others are thin, caller-supplied
follow-ups until each is wired to its own live ledger (the fold already supports
them).

## Why an UNBOUNDED backlog headline

The 0..1 fraction feeds the grade and the pass-line pressure, but a bounded bar
saturates: a queue 3× as far behind can't read heavier than one 1× behind once both
are near 0.0. So the fold also exposes `catchup_backlog` — the **sum of raw behind
counts across every level**, unbounded — so a system three times as far behind reads
three times as heavy. `catchup_debt` (the ratchet integer) is the count of levels
below the pass line; `ok == (debt == 0)`.

A level below the `PassLine` (0.8) is debt, retired by discharging its **real**
backlog (triage the queue, measure the cards, refresh the index, green the trunk, run
the loops) — **never** by weakening the floor.

## Shape

- `internal/catchupscore` — the pure fold over `pkg/scorecard` (deterministic,
  fixture-tested, imports no cmd/report package). `Facts` → `Compose` → `Payload`.
- `cmd/fak/catchupscore.go` — the leased `fak score catchup` shell: auto-collects
  `index`, reads `measurement` from `--scores-from`, takes the rest from flags, emits
  through the shared `emitScorecard` control-pane surface (`--json/--markdown/--compare`).
- Routed under `fak score catchup` (score.go), pinned by `score_test.go`.

## The Slack reporting sink

The `--json` payload is exactly what the scoreboard sink consumes, so a caught-up read
reaches Slack in one pipe:

```
fak score catchup --json | fak scoreboard post --from - --debt-key catchup_debt
```

The card carries the standard control-pane envelope (`value`/`grade`/`pressure`/
`slack` plus `catchup`, `catchup_backlog`, `catchup_worklist`), so every existing
control-pane consumer reads it without special-casing.
