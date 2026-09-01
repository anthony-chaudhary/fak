---
title: "fak-native TOON scorecard"
description: "The SCORE child of the TOON epic (#3064), issue #3068: a fak-native scorecard that answers one question about TOON with a single measured number."
---

# fak-native TOON scorecard

The [SCORE] child of the TOON epic (#3064), issue **#3068**. It answers one question with
evidence from **fak's own payloads**, not a borrowed benchmark: *for which real fak data does
encoding a tool result as TOON instead of JSON actually save tokens, and for which does it
cost more?* TOON's public numbers are contested (official ~68.7% vs one independent ~47.5% on
the same task class, and a collapse to last place on nested data), so fak measures on its own
data, in **both directions** — where it helps and where it hurts.

This scorecard ships the **OBSERVED token-delta half**: deterministic encoders, a fixed
bytes/4 token yardstick, no model call. The **WITNESSED accuracy half** is honestly **not yet**
run — see "Accuracy — not yet" below.

## Reproduce

```
go test -run Scorecard ./internal/toon/ -v
```

The measurement lives in `internal/toon/score.go` (`Scorecard(families []Family) Report`) and
the families + assertions in `internal/toon/score_test.go`. The numbers below are **copied from
that test's output** — none are hand-entered. The `nightrun_jsonl` row reads real rows straight
off `docs/nightrun/harness-resources.jsonl`, so its exact count tracks whatever that committed
telemetry file holds; the other two families are pinned in the test.

## Results (OBSERVED, `go test` output on 2026-07-06)

| family | json tok | toon tok | delta % | elig | verdict | accuracy |
|---|---:|---:|---:|---:|---|---|
| `index_leaves` (uniform flat) | 281 | 202 | **−28.1%** | 1.00 | **FIRE** | not yet |
| `nightrun_jsonl` (real telemetry) | 1503 | 1516 | +0.9% | 0.00 | SKIP(TABULAR_ELIGIBILITY_LOW) | not yet |
| `nested_config` (mixed object) | 99 | 102 | +3.0% | 0.00 | SKIP(TABULAR_ELIGIBILITY_LOW) | not yet |

both-directions: **1 win** (TOON fewer tokens), **2 loss/tie** — the honesty gate the test asserts.

Raw test output (verbatim):

```
family                       json_tok toon_tok    delta%   elig  verdict
------------------------------------------------------------------------------------
index_leaves (uniform flat)       281      202    -28.1%   1.00  FIRE
nightrun_jsonl (real telemetry)     1503     1516     +0.9%   0.00  SKIP(TABULAR_ELIGIBILITY_LOW)
nested_config (mixed object)       99      102     +3.0%   0.00  SKIP(TABULAR_ELIGIBILITY_LOW)
------------------------------------------------------------------------------------
both-directions: 1 win(s) (TOON fewer tokens), 2 loss/tie(s)
accuracy delta:  not yet — requires live paired model eval (follow-on)
```

- **json tok / toon tok** — bytes/4 token estimate of the compact `json.Marshal` encoding vs the
  TOON `Encode` encoding. Same yardstick as `internal/memview` (`tokenEstimate`) and the codec's
  own `Decide` tokenizer fallback, so the numbers compose across the three surfaces.
- **delta %** — `(toon − json) / json`. Negative = TOON wins (fewer tokens).
- **elig** — `TabularEligibility(payload)`: the fraction of scalar leaves that sit inside a
  uniform, flat, tabular-eligible array. 1.00 = every leaf is a tabular cell; 0.00 = none.
- **verdict** — the real gate's decision, `Decide(payload, …)`: `FIRE`, or `SKIP(<reason>)` from
  the closed skip-reason vocabulary.

## Where it helps / where it hurts

**Helps — a uniform array of flat objects (`index_leaves`, −28.1%, FIRE).** This is TOON's happy
path. Each of the 8 rows repeats the same five keys (`name`, `tree`, `dir`, `exists`, `desc`) in
JSON; TOON declares them **once** in the tabular header and emits bare, delimiter-joined value
rows. Eligibility is a perfect 1.00, and the gate fires because every rung passes — uniform
shape, enough rows/bytes to amortize the header, a lossless round-trip, and a real token saving
past the margin. This is the case #3067 would eventually wire.

**Hurts — real nightrun telemetry (`nightrun_jsonl`, +0.9%, SKIP).** The interesting middle. The
rows *look* tabular, but each `harness-resources` record nests sub-objects (`kernel`, `agent`),
so the array is **not** uniform-flat. TOON cannot build a tabular header; it falls back to a safe
per-item JSON list (`- {…}` per row), which is compact JSON **plus** a `- ` prefix and a header
line — hence a slight *loss* (+0.9%). Eligibility is 0.00 and the gate correctly skips with
`TABULAR_ELIGIBILITY_LOW`. The lesson: telemetry that reads as a table on the page is not
tabular to the codec once any cell holds a nested object.

**Hurts — a nested/mixed config (`nested_config`, +3.0%, SKIP).** A deliberately nested tool
result (nested `cache`/`guard`/`routing` policy objects, a mixed array). No repeated field-name
structure to amortize, so TOON's per-line `key:` overhead makes it *larger* than compact JSON.
Eligibility 0.00, gate skips with `TABULAR_ELIGIBILITY_LOW`. The scorecard shows the loss
plainly rather than hiding it — that transparency is the point.

**Verdict, per family:**

- `index_leaves` → **fire here.** Uniform, flat, enough rows: a real, measured ~28% saving.
- `nightrun_jsonl` → **don't fire here.** Nested cells deny the tabular win; TOON is a slight
  net loss. The gate already refuses it.
- `nested_config` → **don't fire here.** Nested/mixed shape; TOON costs more. The gate refuses.

These OBSERVED deltas and the eligibility values are the empirical calibration the `GATE` child
(#3066) uses for its thresholds — τ (tabular-eligibility floor), R_min (min rows), and the
net-token margin. The two SKIPs land on `TABULAR_ELIGIBILITY_LOW` at elig 0.00, well under the
default τ=0.8, and the one FIRE sits at elig 1.00 — consistent with the shipped defaults.

## Accuracy — not yet

Issue #3068's second deliverable is a **WITNESSED accuracy delta**: the same retrieval/extraction
task posed over the JSON form and the TOON form of each family, graded by the dos referee or a
held answer key (type-aware compare, **no LLM judge**), reported as **accuracy-per-1K-tokens** so
efficiency and correctness sit on one axis.

That half is **not run here**, and this scorecard does not invent it. Every row's accuracy column
reads `not yet`, and the code carries the same string as a constant (`toon.AccuracyNotYet =
"not yet — requires live paired model eval (follow-on)"`) so no caller can silently render an
accuracy number the repo never measured. Running it requires live model access and cost across
≥2 models spanning the fitness range (a frontier model and a smaller/local one, since the
research shows model capability dominates the format effect) — not feasible in this background
session.

**Methodology a follow-on would run** (so the gap is a checkable next step, not a vague marker):

1. For each family, author a small held answer key: N retrieval questions with exact expected
   answers (e.g. "the `dir` of the `toon` leaf", "the count of `guard.reasons`").
2. For each (family × model), pose each question twice — once with the payload rendered as JSON,
   once as TOON (input-only; TOON output support is weak) — at temperature 0.
3. Grade each answer by type-aware exact/normalized compare against the key (no model judge).
4. Report JSON accuracy, TOON accuracy, and **accuracy-per-1K-tokens** using the token counts
   above, plus a fire/don't-fire line that folds BOTH token delta and accuracy delta.
5. Feed the measured accuracy floor back to #3066's φ (model-fitness) and τ thresholds.

## Provenance fence

Every token-delta number here is **OBSERVED** — deterministic encoders, a fixed bytes/4
estimator — never modeled and never blended with dollars. Every accuracy claim would be
**WITNESSED** (paired on/off eval, held key) — and until it is run, it is reported as `not yet`,
never self-reported. No TOON path flips default-on without the passing correctness witness this
scorecard is one input to (#2830's absolute rule, inherited via #3064 / #2828).
