# agentbenchdemo — the self-tax: what the kernel costs per tool call

The **performance** micro-benchmark of fak's agentic spine. It answers one question a
skeptic always asks about a safety kernel: *what does it cost?* It folds a fixed plan
of tool calls through the **real fak kernel** — the same `internal/agentdemo` path
(`adjudicator.Default.SetPolicy` + a live `kernel.Fold` per call) that
[`cmd/timewolfdemo`](../timewolfdemo) and `fak preflight` use — times the loop, and
reports the **per-call adjudication cost**: the "self-tax" the safety floor adds to an
agent's critical path.

The headline is a net-value number, not a micro-optimisation: a single LLM
tool-calling round-trip is ~1–3 s, while the kernel adjudicates a tool call in a
handful of **microseconds** — so the floor is, in practice, free. **No model, no GPU,
no key, no network.** The benchmark completes in well under a second.

## Run it (headless — no browser)

```bash
go run ./cmd/agentbenchdemo                  # the self-tax table in the terminal
go run ./cmd/agentbenchdemo -selfcheck       # assert the self-tax invariants (CI gate)
go run ./cmd/agentbenchdemo -json            # the measured result as JSON
go run ./cmd/agentbenchdemo -n 20000         # more iterations for a tighter mean
```

## What you'll see

A small table: the iteration count, the deterministic allow/deny split (1 allow + 2
deny per iteration), the **mean wall-clock per adjudicated call** measured on your box,
the throughput in calls/sec, and the net-value line — roughly how many times cheaper
the per-call adjudication is than one ~1.5 s LLM round-trip. `-selfcheck` re-runs a
small batch and asserts the structural invariants (the deterministic counts, plus a
generous anti-hang ceiling on the per-call cost), printing one `... invariants hold`
line and exiting non-zero on drift.

## Determinism and stability

The **workload is deterministic**: the same plan is folded every run and the allow/deny
outcome is byte-identical (1 allow + 2 deny per iteration), so a re-run is repeatable
and `-selfcheck` gates cross-platform in CI. Only the measured *wall-clock* varies —
that is the whole point of a benchmark.

## What this does not claim

This measures the per-call adjudication cost **on the box you run it on**; it is not a
portable latency guarantee, not a comparison against another kernel, and not a model
benchmark (there is no model — the planner is a fixed plan). The ~1.5 s LLM round-trip
reference is a deliberately conservative low end of the ~1–3 s range used only to frame
the ratio, so the "N× cheaper" line is an under-statement, never an over-statement. For
the broader honesty rubric these demos hold to, see the
[net-true-value standard](../../docs/standards/net-true-value.md).

Part of the hosted live-demo gallery refresh — see epic
[#1167](https://github.com/anthony-chaudhary/fak/issues/1167).
