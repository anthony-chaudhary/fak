# fanoutdemo — the spine-first fan-out planner, end to end

The **productization** demo for the issue fan-out planner (`internal/issuefanout`, the
`fak issue fanout` verb). It shows a stranger, in one deterministic command, the
spine-first default working end to end: the moment a working spine ships, its whole
follow-on backlog — QA, dogfooding, productization, observability, integration, docs,
release — is filed at creation time, each item already carrying a complete,
**dispatchable** issue contract.

The demo drives the **real planner** twice to tell the whole story:

1. **The spine-first guard.** Asked to fan out with *no* spine witness, the planner
   **refuses** — a stranger sees the default is enforced below the caller, not merely
   suggested.
2. **The fan-out.** Given a shipped spine, it expands the fixed taxonomy into
   contract-ready follow-ons and proves **every one is dispatchable** (scope · witness ·
   acceptance gate · lane · closure binding).

The plan is a pure function of its input (stdlib + `issuecontract` only — no `gh`, no
disk, no clock), so the output is byte-identical every run. **No model, no GPU, no key,
no network.** It completes instantly.

## Run it (headless — no browser)

```bash
go run ./cmd/fanoutdemo              # the spine-first fan-out story in the terminal
go run ./cmd/fanoutdemo -selfcheck   # assert the fan-out invariants (CI gate)
go run ./cmd/fanoutdemo -json        # the refusal + the full plan as JSON
```

## What you'll see

The refusal message the planner returns when handed a spine-less input, then the full
fan-out from the planner's **own** shipped spine (`5b8f0bd1`, `internal/issuefanout` +
`fak issue fanout`): one line per contract-ready follow-on with its area, marker key,
step budget, and generation, followed by the per-area counts and a line confirming every
follow-on carries a complete, dispatchable issue contract. `-json` emits the same as a
machine-readable envelope (`spine_first_refusal` + the `fak.issue-fanout-plan.v1` plan);
`-selfcheck` re-drives both paths and prints one `... invariants hold` line, exiting
non-zero on drift. A captured run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## Determinism and stability

The planner is **pure**: the same `Input` yields a **byte-identical** `Plan` on any box,
which `-selfcheck` asserts directly (two builds, deep-equal). It also checks the
spine-first refusal is enforced, that every candidate carries the `fanout-issuefanout-`
marker-key prefix and passes the issue contract as **dispatchable**, and that the
per-area counts sum to the candidate count. Because these are **structural** invariants
rather than a hardcoded template count, the gate stays green as the taxonomy grows and
still exits non-zero the moment the planner's contract regresses. It needs no model, key,
or network, so it gates cross-platform in CI.

## What this does not claim

This demo does **not** file any issue — it exercises the pure *planner* (what the
follow-ons are and that each one is dispatchable), not the `gh` filing path, which the
`fak issue fanout` / `fak issue cohort` verbs own. It is not a benchmark and not a
security proof. The self-referential spine (fanning out the fan-out planner's own leaf)
is a deliberate dogfood, not a claim about any other subsystem. For the honesty rubric
these demos hold to, see the
[net-true-value standard](../../docs/standards/net-true-value.md).

Part of epic [#2510](https://github.com/anthony-chaudhary/fak/issues/2510) (spine-first +
fan-out defaults); this demo closes [#2517](https://github.com/anthony-chaudhary/fak/issues/2517).
