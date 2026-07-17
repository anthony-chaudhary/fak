# Example output — fanoutdemo

Captured from `go run ./cmd/fanoutdemo` and `-selfcheck`. The plan is a pure function of
its input, so this output is **byte-identical every run** on any box.

## `go run ./cmd/fanoutdemo` (the spine-first fan-out story)

```text
fanoutdemo · the spine-first fan-out planner, end to end

1. spine-first guard — asked to fan out with NO spine witness, the planner refuses:
   issuefanout: spine_ref is required — ship the minimal working spine first (or file the spine issue itself), then fan out from its witness

2. fan-out — given a shipped spine, it files the whole dispatchable backlog:
fanout: 15 contract-ready follow-ons for issue fanout planner (spine: 5b8f0bd1 (internal/issuefanout + fak issue fanout))
  [qa           ] qa: adversarial + edge-case sweep for issue fanout planner  (key fanout-issuefanout-qa-edge-sweep, ~3 steps, gen/now)
  [qa           ] qa: failure-path + refusal coverage for issue fanout planner  (key fanout-issuefanout-qa-failure-paths, ~3 steps, gen/now)
  [qa           ] qa: determinism + race witness for issue fanout planner  (key fanout-issuefanout-qa-determinism, ~2 steps, gen/now)
  [dogfood      ] dogfood: run issue fanout planner on this repo's own live work  (key fanout-issuefanout-dogfood-self-run, ~3 steps, gen/now)
  [dogfood      ] dogfood: usage ledger so issue fanout planner adoption is measured, not claimed  (key fanout-issuefanout-dogfood-usage-ledger, ~4 steps, gen/next)
  [product      ] product: docs/cli-reference.md entry + usage parity for issue fanout planner  (key fanout-issuefanout-product-cli-reference, ~2 steps, gen/next)
  [product      ] product: LCD demo/example for issue fanout planner meeting the run-the-demos bar  (key fanout-issuefanout-product-lcd-demo, ~5 steps, gen/next)
  [product      ] product: refusal/error-message quality pass for issue fanout planner  (key fanout-issuefanout-product-error-ux, ~3 steps, gen/next)
  [observability] observability: outcome counters for issue fanout planner  (key fanout-issuefanout-obs-outcome-counters, ~4 steps, gen/next)
  [observability] observability: scorecard fold for issue fanout planner adoption/health  (key fanout-issuefanout-obs-scorecard, ~4 steps, gen/next)
  [integration  ] integration: advisory commit-gate nudge for issue fanout planner  (key fanout-issuefanout-int-guard-gate, ~5 steps, gen/next)
  [integration  ] integration: dos.toml lane + reason wiring for issue fanout planner  (key fanout-issuefanout-int-dos-wiring, ~3 steps, gen/next)
  [integration  ] integration: super-loop/dispatch default hookup for issue fanout planner  (key fanout-issuefanout-int-superloop, ~5 steps, gen/next)
  [docs         ] docs: doctrine + INDEX/llms.txt linkage for issue fanout planner  (key fanout-issuefanout-docs-doctrine-linkage, ~2 steps, gen/next)
  [release      ] release: CLAIMS.md tag + release-note line for issue fanout planner  (key fanout-issuefanout-release-claims, ~2 steps, gen/next)
next: file with gh (milestone + labels at creation), or wave-plan via `fak issue cohort --from-plan`

area counts: qa=3 dogfood=2 product=3 observability=2 integration=3 docs=1 release=1
every one of the 15 follow-ons carries a complete, dispatchable issue contract (scope · witness · acceptance gate · lane · closure binding).
```

One shipped spine fans out into **15 contract-ready follow-ons** across 7 areas — and
note follow-on #7 is *this very demo* (`product-lcd-demo`), so the planner is dogfooding
the leaf that ships it.

## `go run ./cmd/fanoutdemo -selfcheck` (the CI gate)

```text
fanoutdemo -selfcheck: the fan-out invariants hold (15 dispatchable follow-ons across 7 areas · spine-first refusal enforced)
```

`-selfcheck` re-drives both paths and asserts the structural invariants — the spine-first
refusal is enforced, every follow-on is dispatchable and marker-key-prefixed, the area
counts are internally consistent, and the plan is byte-identical across two builds — then
exits **0**, or non-zero on any drift.

## `go run ./cmd/fanoutdemo -json` (the machine-readable envelope, head)

```json
{
  "spine_first_refusal": "issuefanout: spine_ref is required — ship the minimal working spine first (or file the spine issue itself), then fan out from its witness",
  "plan": {
    "schema": "fak.issue-fanout-plan.v1",
    "input": {
      "title": "issue fanout planner",
      "leaf": "issuefanout",
      "spine_ref": "5b8f0bd1 (internal/issuefanout + fak issue fanout)",
      "parent_ref": "#2510"
    },
    "area_counts": { "docs": 1, "dogfood": 2, "integration": 3, "observability": 2, "product": 3, "qa": 3, "release": 1 },
    "candidates": [ /* 15 fully-scoped issuecontract candidates */ ]
  }
}
```
