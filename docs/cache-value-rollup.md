# Cache-Value Roll-Up

> The cache-value roll-up is the front door for reading whether fak's cache work is
> paying off. It keeps the kernel-reuse proof and the provider-dollar economics in
> separate tracks so the report can show a trend without blending unlike evidence.

## The Problem

Before the roll-up, cache-effectiveness evidence was scattered across five places:

- `docs/nightrun/cache-value.jsonl`, the durable session ledger.
- `fak nightrun score`, the all-time regression gate over that ledger.
- `internal/cachevaluereport`, the weekly Track-1 trend fold.
- Benchmark packets such as `docs/benchmarks/GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md`.
- Slack or scoreboard posts, where operators expect one card rather than several raw files.

That made single-session evidence easy to inspect but hard to trend. The roll-up is the
reader-facing layer over those sinks: one place to ask what moved, what evidence supports
it, and what must not be inferred from it.

## The Two Tracks

| Track | What it answers | Evidence | Current status |
|---|---|---|---|
| Track 1: WITNESSED kernel value | Did fak's own kernel reuse KV-prefix work on multi-turn sessions? | `cachevalueledger.Row` fields: `prompt_tokens`, `reused_tokens`, turn regimes, and weekly buckets from `internal/cachevaluereport`. | Shipped for realized reuse trend. |
| Track 2: OBSERVED net-dollar savings | Did the deployed gateway reduce provider spend after its own costs and provider-cache behavior? | Billing/provider observations joined to the session timeline. | Not yet in this docs rung; tracked by epic #1301 rungs B/C. |

The tracks stay unblended because they answer different questions. Track 1 is a mechanism
proof: fak authored reuse inside the kernel and can witness the token counters. Track 2 is
an economic outcome: the provider bill, prompt-cache discount, and gateway overhead decide
whether the mechanism saved money. A combined number would hide the failure mode where
reuse is real but not net-positive, or where dollars improve for a reason unrelated to
kernel reuse.

## Honesty Fences

- **#1066 marginal-over-warm-KV fence.** The published Track-1 number is realized
  KV-prefix reuse over multi-turn sessions. It is not the vs-naive re-prefill multiple
  `1/(1-reuse)`. The honest single-session cache value is marginal over a tuned warm-KV
  server, approximately `1.0x`; the larger value can only come from cross-worker shared
  prefix reuse.
- **WITNESSED vs OBSERVED.** WITNESSED means fak can read back the kernel ledger it wrote.
  OBSERVED means an external bill, provider metric, or operator surface reported the
  outcome. A card must label which one it is showing.
- **Net, not gross.** Provider-dollar savings must be net of fak's own cost and any
  upstream cache behavior. A gross token drop is useful diagnostic evidence, not a
  publishable dollar-savings headline.
- **Thin corpus falls open.** Single-turn cold runs have no reuse opportunity. A thin
  multi-turn corpus reports `INSUFFICIENT` instead of fabricating a regression or a win.

## Reading The Card

A cache-value card should be read top-down:

- **Verdict** says whether the current window is measured or still insufficient.
- **Latest reuse** is the most recent Track-1 weekly realized reuse ratio, over
  multi-turn sessions only.
- **Trend** compares the latest weekly bucket with the prior bucket using the report
  dead-band; flat means the movement is inside noise.
- **Thin** means the bucket has fewer than `cachevalueledger.MinGateTurns` multi-turn
  turns, so it is visible but not trend-significant.
- **Regime `f/p/c`** is frozen, partial, and cold turns; it explains where reuse came
  from before anyone turns it into a headline.
- **Next action** names the missing evidence, usually more multi-turn sessions or the
  Track-2 provider-dollar join.

## Reproduce

The shipped Track-1 witness on current `main` is:

```bash
fak nightrun score --json
```

That command reads `docs/nightrun/cache-value.jsonl`, excludes single-turn cold runs,
prints the realized reuse ratio, and carries the #1066 self-labels. The weekly fold behind
the roll-up is pinned by:

```bash
go test ./internal/cachevaluereport
```

The intended cachevalue front-door spelling for a dated operator report is:

```bash
fak cachevalue report --since 2026-06-22
```

Do not treat that spelling as the witness on a build that does not expose the
`cachevalue report` subcommand yet; use `fak nightrun score --json` for the shipped
Track-1 gate on current `main`.

## See Also

- [CLAIMS.md](../CLAIMS.md) for the shipped/stub honesty ledger.
- [Net-true value standard](standards/net-true-value.md) for the net-not-gross rule.
- [GLM-5.2 fak-kernel cache value packet](benchmarks/GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md)
  for the benchmark packet shape.
- [Recent fak logs audit](notes/AUDIT-recent-fak-logs-effectiveness-fidelity-2026-06-28.md)
  for an example of the thin-corpus fence in action.
