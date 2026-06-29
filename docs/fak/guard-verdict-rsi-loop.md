---
title: "fak guard verdict-pattern RSI loop"
description: "The RSI loop for fak guard that closes on our OWN usage — the real decision
journal — with no hardware gate. The hardware-free sibling of the latency loop (#733/#734)."
---

# fak guard verdict-pattern RSI loop

> **Audience.** Anyone wiring or auditing the `fak guard` self-improvement loop — by the end you'll know which of the two guard RSI loops closes on a normal box and how its keep-bit stays non-forgeable.

`fak guard -- <agent>` fronts a coding agent with the kernel adjudicating every tool call,
and writes a **default-on, hash-chained decision journal** (`guard-audit.jsonl`) of every
verdict it reaches. That journal is the self-improvement signal our own workflow produces.

There are two RSI loops for `fak guard`. They optimise different signals, and only one of
them can close on a normal machine:

| loop | tool | signal | closes on a normal box? |
|---|---|---|---|
| **latency** | `tools/guard_hop_rsi.py` (#733) | guard-hop overhead (TTFB delta) | **No** — its keep/revert rung needs a live `fak serve` + a direct mock on one box (#734). It runs in plan mode (every candidate `PENDING_MEASUREMENT`) and honestly fences that gate rather than fabricating a wall-clock. The journal holds *verdicts*, not latency, so latency is the wrong thing to read from it. |
| **verdict** | `fak guard-verdict-rsi` | verdict-quality of the real journal | **Yes** — it reads the real `guard-audit.jsonl`, scores the verdict distribution's honesty, and keeps a refinement only on a strict gain + an external witness. No hardware. |

This doc is about the verdict loop — the one that learns from our usage today.

## What it optimises

A *good* guard journal is one where every block is **explained** (a closed-vocabulary
`reason`, never blank) and every verdict is **classified** (in the kernel's known set:
`ALLOW / DENY / TRANSFORM / QUARANTINE / WITNESS / DEFER / INDETERMINATE`). An unexplained
block or an `UNCLASSIFIED` verdict is exactly the prose-drift the kernel exists to kill.

`verdict_quality` is a deterministic 0–100 score over the folded distribution:

```
quality = max(0, 1 - (blank_reason_on_deny + unknown_verdict) / total_rows) * 100
```

Both penalties are **rates** over total rows, so the score is scale-free (a clean 10-row
journal and a clean 10k-row journal both score 100) and a **pure function of the journal
bytes** — same rows in, same score out. That determinism is what lets a KEEP be trusted: it
cannot be a one-run fluke.

## How it closes (keep / revert)

```
guard-audit.jsonl (our real sessions, discovered via the dogfood_coverage reader)
   -> fold verdicts: ALLOW/DENY/QUARANTINE + by_reason + honesty holes
   -> score baseline verdict_quality
   -> propose ONE refinement targeting the worst bucket (worst-first)
   -> replay -> re-score
   ┌── KEEP iff: rows > 0 AND replayed quality STRICTLY higher
   │            AND an external witness (go test ./... / fak policy check) is green
   └── else REVERT
```

The keep-bit is **non-forgeable**: `check_iteration` rejects any `kept=true` that lacks real
rows, a strictly-positive delta, or a green witness — the same honesty contract
`guard_hop_rsi.check_plan` enforces for the latency loop, and the same discipline as the DOS
enforcement-tuning loop. A kept gain rests on a re-measured number + a witness the loop did
not author, never on the loop's say-so.

### Empty-journal honesty

A loop that learns from real usage must not fabricate a gain from no data. When the journal
holds **zero adjudicated rows**, the row count *is* the gate: the loop refuses any keep and
`diagnose_audit_gap` says *which* blank the zero is (no journal directory / journal but no
files / files but all blank) so the operator gets the unblock action, not an undifferentiated
`0`.

## Producing real rows

Any `fak guard -- <agent>` session appends rows. To seed without a live agent, drive real
tool-call proposals through a journal-enabled gateway — the kernel decides and journals each:

```bash
FAK_AUDIT_JOURNAL=.dispatch-runs/guard-audit/seed.jsonl \
  fak serve --addr 127.0.0.1:8231 --policy cmd/fak/guard-default-policy.json &
curl -s -XPOST 127.0.0.1:8231/v1/fak/adjudicate -d '{"tool":"Bash","arguments":{"command":"rm -rf /"}}'
# ... a mix of allow + deny proposals ...
fak audit verify .dispatch-runs/guard-audit/seed.jsonl   # the tamper-evident chain
```

## Usage

```bash
go run ./cmd/fak guard-verdict-rsi fold                 # the verdict distribution + quality
go run ./cmd/fak guard-verdict-rsi run                  # one iteration: propose -> replay -> keep/revert
go run ./cmd/fak guard-verdict-rsi run --witness '{"ok":true,"suite":"go test ./... PASS"}'
go run ./cmd/fak guard-verdict-rsi route                # CLOSE the loop: route the worst bucket to a finding
go run ./cmd/fak guard-verdict-rsi --check iter.json    # honesty gate over an emitted iteration
```

## Route — closing the loop

`fold`/`run` *find* the worst bucket; without a route, that finding is rendered to a human
and dropped — so a recurring wall is re-discovered every session and never tracked. `route`
is the closure rung: it reviews the journal and, when the worst bucket is a real finding,
materializes it through two existing idempotent sinks so the next dispatch can *pick it up*.

```
guard-audit.jsonl -> fold -> worst bucket -> guardroute.Decide (pure)
   ├─ blank_reason_on_deny / unknown_verdict  -> P1 honesty-hole -> queue row + gh issue
   ├─ a denial reason recurring >= --threshold -> P2 advisory    -> queue row only
   └─ clean fold / empty journal               -> no route (self-diagnosing why-not)
```

- **The pickable queue row** (always, for any routed finding) is appended by
  `tools/findings_route.py` — append-only, **idempotent per key**, damping-folds concurrent
  sessions onto one row by a **content-stable `cause_key`** (`guard-journal:<bucket>`), and
  **escalates severity** (P2→P1→P0) when a cause re-fires after a "fixed" close.
- **The deduped GitHub issue** (only for an honesty-hole, P1/P0) is created/updated through
  `internal/dogfoodissues` — the same marker-keyed create-vs-update path the dogfood
  scorecard uses, so a re-run edits the issue in place instead of opening a duplicate.

Issue filing is **on by default** when you run `route` by hand; `--no-issues` is queue-only
(for a host without `gh` auth) and `--dry-run` plans the issue without touching `gh`. Every
step is **fail-open**: a `findings_route` / `gh` failure is reported, never fatal — the loop
must not die on its own closure layer.

The route also fires automatically on the scheduled `fak garden` tick (the non-gating
`guard_route` member, run `--no-issues` so an unattended tick never needs `gh`), so **every
guarded session leaves a pickable trace** without anyone invoking the command.

```bash
go run ./cmd/fak guard-verdict-rsi route --json          # review + route (issues on); JSON envelope
go run ./cmd/fak guard-verdict-rsi route --no-issues     # queue row only (no gh)
go run ./cmd/fak guard-verdict-rsi route --dry-run       # plan the issue without touching gh
python tools/findings_route.py status                    # show the routed findings queue
```

The loop's maturity + realized value is scored by `fak guard-rsi-scorecard` (the
`guard_rsi_debt` member of the control-pane ratchet) and driven on a `/loop` cadence by the
`guard-rsi-score` skill.

## Read next

- [The five-rung RSI loop](../rsi-loop.md) — the keep/revert discipline this loop instantiates.
- [The dojo RSI loop](dojo-rsi-loop.md) — the same keep-bit machine pointed at the calibration ledger.
- [Engineering is building loops](../explainers/engineering-is-building-loops.md) — the loops doctrine behind it.
