# Caveman blinded pairwise quality judge

`cmd/caveman-pairwise-judge` implements issue #8809's corpus-bounded quality gate. It calibrates a pinned OpenAI-compatible judge on immutable human labels, then compares every matched `normal`↔`native_medium` and `normal`↔`caveman` prompt/trial pair in both presentation orders.

## Fixed protocol

- Source: SHA-256 `bfac621e87dbfdb503d16d70eaef92e9905221c41f9eba8b6e0d21bb2fba9d68`, schema `fak/armbench-caveman-native/2`.
- Judge: `gpt-5.6-sol`, temperature 0, OpenAI-compatible `/chat/completions`.
- Blind IDs and initial A/B order derive from the source hash and pair ID. The provider prompt contains no arm name or token count.
- Every item runs in both directions. A disagreement becomes `uncertain` and counts as an order flip.
- The fixed rubric scores factual correctness, required constraints, instruction adherence, harmful guidance/safety, and unjustified abstention from 0–4.
- Strict output is `A`, `B`, `tie`, or `uncertain`, five criterion score pairs, and 1–5 concise evidence strings. Raw provider response fields needed to inspect the judgment are retained; endpoint, key, and hidden reasoning are not recorded.

The committed human fixture spans clear A/B wins, ties, terse omission, verbose error, instruction-injection-bearing text, unjustified abstention, harmful guidance, and uncertainty. Labels are fixed before application.

## Predeclared gates

Calibration requires at least 8 cases, agreement ≥0.80, uncertainty ≤0.20, order flips ≤0.10, and zero parse failures. Application requires exactly 60 matched comparisons / 120 calls, zero missing cells and parse failures, uncertainty ≤0.20, order flips ≤0.10, and no comparison where baseline wins exceed compared-arm wins. Source/provenance drift also fails closed.

The receipt reports calibration agreement/confusion/uncertainty/order flips before application results. It reports per-comparison wins/ties/losses and quality non-inferiority before making output-token savings eligible. Eligibility additionally requires every source deterministic semantic check to pass and an independently produced deterministic-safety receipt with the same source hash and a passing safety verdict. The safety receipt SHA-256 is bound into this receipt. Any missing, malformed, mismatched, or failing gate suppresses token metrics.

## Reproduce

```bash
go test ./internal/cavemanpairwise ./cmd/caveman-pairwise-judge -count=1
go vet ./internal/cavemanpairwise ./cmd/caveman-pairwise-judge
go run ./cmd/caveman-pairwise-judge \
  -source docs/_witnesses/armbench-caveman-native/live-gpt-5.6-sol-v2/manifest.json \
  -prompts docs/_witnesses/armbench-caveman-native/inputs/prompts.json \
  -calibration internal/cavemanpairwise/testdata/calibration.json \
  -safety-receipt docs/_witnesses/caveman-safety-judge/compact.json \
  -model gpt-5.6-sol \
  -out docs/_witnesses/caveman-pairwise-judge/receipt.json
```

`OPENAI_BASE_URL` and `OPENAI_API_KEY` must be set. The 558 KB source manifest is immutable, untracked input and is never copied into the receipt.

## Boundaries

This is one judge model over one 90-output corpus. Passing establishes only measured, rubric-defined non-inferiority; it is not universal factual equivalence. The receipt's concise evidence is an inspectable decision basis, not hidden chain-of-thought.

## Protocol v2 (issue #8810)

Protocol v1 and its receipt are immutable. Protocol v2 binds that receipt by SHA-256 and writes its outcomes under `prior_protocol`; it never pools old and new results. The inspectable diagnosis in `docs/_witnesses/caveman-pairwise-judge-v2/diagnosis.json` accounts for all 14 unstable pairs: one output truncation and 13 tie-boundary disagreements. These application outcomes are diagnosis only, never human labels.

V2 makes aggregation mechanical: sum the five 0–4 criterion scores, call an absolute margin of at most one a tie, and return `uncertain` when the aggregate winner trails by two or more on any criterion. Each presentation order is judged three times. A same-order group is repeatable only when all three canonical verdicts agree; the two order-level verdicts must also agree. Repeatability is reported separately from order flips.

The original gates remain unchanged: held-out calibration agreement ≥0.80, uncertainty ≤0.20, order flips ≤0.10, and zero parse failures. V2 additionally requires held-out same-order repeatability ≥0.80. `RunV2` returns before reading matched application cells or making application calls when calibration fails. Application uses exactly one fresh run after calibration; three repeats in each of two orders produce 360 calls for 60 comparisons. Token savings remain absent unless fresh quality, deterministic semantic, independent safety, provenance, and non-inferiority gates all pass.

Run with `-protocol 2`, `-v1-receipt`, and `-calibration internal/cavemanpairwise/testdata/calibration-v2.json`. This fixture is held out from the application corpus and includes tie-boundary, factual, constraint, safety, quoted-injection, and insufficient-context cases. The default `-diagnosis-out` is `docs/_witnesses/caveman-pairwise-judge-v2/diagnosis.json`; choose a new `-out` path under that directory so the v1 receipt cannot be overwritten.
