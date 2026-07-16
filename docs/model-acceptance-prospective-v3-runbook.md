# Prospective exact-model v3: post-reset campaign runbook

Issue: [#4845](https://github.com/anthony-chaudhary/fak/issues/4845)
Production-readiness parent: [#4633](https://github.com/anthony-chaudhary/fak/issues/4633)
Supersedes-context: v2 HOLD readout `docs/model-acceptance-prospective-v2-readout.md`

## Why this exists

The v2 prospective campaign was preregistered in commit `8c5edc9c3249` and then
produced 18/18 typed provider/infrastructure weekly-limit rejections (HOLD) — zero
eligible capability observations. Under the declared no-replacement rule those
attempts stand and cannot be silently replaced. The selected seat reported a weekly
reset at **2026-07-16 06:00 America/Los_Angeles**. Per the issue owner, a fresh
campaign after the reset "requires a new committed declaration and may not replace
these attempts."

This is that new committed declaration. It is a **pre-observation** artifact only:
it fixes prompts, exact model IDs, tiers, repetitions, the structured sentinel-line
grammar, thresholds, and the replacement/stopping rules before any provider output.
It carries zero runs and folds to HOLD/0-samples for every model until an
authenticated campaign attaches real streams. The in-lane gate
`TestProspectiveV3DeclarationPrecedesObservations`
(`internal/modelaccept/modelaccept_test.go`) enforces that discipline.

## Committed declaration

- File: `examples/model-acceptance-prospective-v3.json`
- Corpus ID: `top3-prospective-sentinel-v3`
- Declaration SHA-256: `0e6404b25db0610f2d309092d04bc2635a7b2f9c840551e45a0f8262c70dd967`
- Exact IDs: `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`
- Two production-shaped task classes per model (read-only multi-record synthesis;
  typed transient retry recovery), three repetitions each — 18 fixed attempts.

## What remains — the operator-run authenticated campaign

The run harness `fak model acceptance-run` shells to an **authenticated Claude
subscription CLI seat** (`--claude-config-dir` holding `.credentials.json`); it
deliberately strips `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`, so the ambient fak
gateway cannot serve it and no API-key path substitutes. Launching consumes that
seat's scarce weekly quota, so the seat selection and the decision to spend the
weekly budget are operator calls, not an autonomous worker's. Run it post-reset:

```bash
go build -o /tmp/fak-accept ./cmd/fak      # do NOT drop fak.exe into the repo root
/tmp/fak-accept model acceptance-run \
  --input      examples/model-acceptance-prospective-v3.json \
  --output     <operator-scratch>/model-acceptance-prospective-v3-report.json \
  --raw-dir    <operator-scratch>/raw-v3 \
  --claude-config-dir <authenticated seat dir with .credentials.json> \
  --fixture-command   /tmp/fak-accept \
  --timeout    2m
```

Exit codes: `0` PASS (all eligible runs satisfied the declared contract), `4` HOLD
(any provider/infrastructure, harness, refusal, or capability failure — like v2),
`2` setup/usage error.

## After the run

- Raw provider JSONL under `--raw-dir` contains provider/session metadata — it stays
  in operator scratch and is **not** committed (same as v2).
- Publish a scrubbed readout mirroring `docs/model-acceptance-prospective-v2-readout.md`:
  declaration SHA-256, canonical raw-manifest SHA-256, completed-report SHA-256, the
  per-model eligible/provider-failure table, and the runner exit code.
- Link results to #4845 and #4633 without treating the retrospective refold or the v2
  attempts as this prospective campaign.
