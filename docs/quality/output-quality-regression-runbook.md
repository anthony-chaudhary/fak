---
title: "Output-quality regression runbook"
description: "How to catch, localize, and recover from an engine-caused output-quality regression: detect, freeze, replay, classify, reference, bisect, mitigate, fix, recover — with a hermetic captured proof."
slug: output-quality-regression-runbook
keywords:
  - output quality regression
  - decode parity
  - accuracy regression
  - incident runbook
  - deterministic replay
  - engine correctness
date: 2026-07-13
---

# Output-quality regression runbook

This runbook covers the missing middle: a defect where the engine still returns
tokens and every primitive-correctness test is green, but the *content* of the
output has regressed. Primitive tests do not catch it and end benchmarks are too
coarse and too late to localize it. When executive-report dogfood or a nightly
accuracy suite flags drift, this is the procedure that turns "the output looks
worse" into a first actionable divergence, a scrubbed replay artifact, and a
witnessed recovery.

Parent epic: [#4509](https://github.com/anthony-chaudhary/fak/issues/4509)
(the missing-middle validation ladder). This page fills the incident-response
layer of that ladder.

## The load-bearing rule

Missing or inconclusive evidence is never a pass. A case with no frozen golden,
an empty baseline, or an oracle that could not run is `INCONCLUSIVE`, and an
inconclusive case blocks promotion exactly like a failing one. A green result
has to come from a comparison that actually happened.

## The captured proof (run it yourself)

The runbook ships with a hermetic tabletop that exercises the whole method in a
clean, independently replayable environment: stdlib-only Python, no network, no
hardware, no host state. It stands in a planted representative defect, shows the
detector catching it, then shows the fix passing.

```bash
# one command proves the contract: planted defect FAILS, the fix PASSES
python docs/quality/regression_runbook_witness.py --selftest

# or drive each arm by hand
python docs/quality/regression_runbook_witness.py --defect bos-drop   # exit 1 (FAIL)
python docs/quality/regression_runbook_witness.py --defect tie-break  # exit 1 (FAIL)
python docs/quality/regression_runbook_witness.py                     # exit 0 (PASS)
```

The frozen capture of that run lives beside it in
[`regression_runbook_witness.capture.txt`](regression_runbook_witness.capture.txt),
and a scrubbed replay artifact from the failing arm is in
[`regression_runbook_replay.sample.json`](regression_runbook_replay.sample.json).
Exit codes are the contract: `0` = PASS, `1` = FAIL (an actionable divergence),
`2` = INCONCLUSIVE (never a pass).

The tabletop uses a toy decoder so it can run anywhere in under a second. The
*method* it demonstrates — freeze a golden, replay deterministically, report the
first divergence with full provenance, classify it, never pass on missing
evidence — is the same one you apply to a real engine regression below.

## The procedure

### 1. Detect

The signal arrives from one of the ladder's upper rungs: an executive-report
dogfood diff, a nightly accuracy or sampling-stability suite, or an engine-mode
parity check. Record what fired and the observed symptom (wrong answer, degraded
rubric score, changed token stream) before touching anything.

### 2. Freeze

Pin the exact inputs so the failure is reproducible. Capture, for the failing
case, the full provenance record:

| Field | Why it is load-bearing |
|---|---|
| `model` | a different checkpoint is a different oracle |
| `tokenizer` | most silent output drift is a tokenizer or special-token change |
| `engine` / `backend` | GPU / Metal / CPU / vLLM / SGLang can each diverge |
| `seed` or deterministic oracle | without it, "different output" is not yet a bug |
| `code_revision` | the bisect target range starts here |
| `tolerance` / `baseline` provenance | names *what* green means and where it came from |

If any field is unknown, the case is inconclusive, not a pass.

### 3. Replay

Re-run the frozen case deterministically. Identical inputs must give identical
output; if they do not, the nondeterminism itself is the first bug to fix before
anything downstream can be trusted. The witness models this: same file, same
seed, same result on every host.

### 4. Classify

Label the first divergence so it can be routed. The witness classifier draws the
line most incidents fall along:

- A stream misaligned from position 0 points at tokenization or normalization
  (a dropped `<bos>`, a changed chat template, a special-token shift).
- An in-stream substitution at a later position points at the decode or sampler
  path (a tie-break flip, a logits-processor change, a cache-reuse bug).
- A length divergence points at stopping criteria or truncation.

### 5. Reference

Compare against the frozen golden, not against a fresh run of the suspect build.
The baseline provenance recorded in step 2 is the reference. A regression is a
divergence from a *trusted* prior, so the reference must predate the suspected
change.

### 6. Bisect

Report the first actionable divergence — earliest case, then earliest token
position — and bisect the `code_revision` range between the last known-good
baseline and the failing build. The first divergence is the one to chase; later
ones are usually downstream of it.

### 7. Mitigate

If the regression is live, mitigate before you fix: pin to the last known-good
revision, or route the affected tier away from the suspect path. Mitigation buys
time; it is not the recovery.

### 8. Fix

Land the correction at the source the bisect identified. In the tabletop the fix
is dropping the planted defect; in a real incident it is the code change the
first divergence pointed at.

### 9. Recover

Recovery is witnessed, not asserted. Re-run the frozen case in a clean
environment and confirm it now matches the golden — the same `PASS` the witness
emits after the defect is removed. An incident is closed when the captured proof
that failed now passes on an independent replay.

## Tiering: assign every case a cost

Each case belongs to exactly one tier, and the tier sets where it runs and what
it may cost:

| Tier | Runs on | Budget |
|---|---|---|
| `pr` | every pull request | seconds, no hardware — cheap deterministic oracles only |
| `nightly` | scheduled | minutes to hours — sampling stability, engine-mode parity, larger sets |
| `release` | pre-promotion | full accuracy suites and hardware-specific qualification |

The witness tags each case with its tier and cost, and `--tier pr` restricts a
run to the cheap gate. Keep the expensive checks off the PR path so the front
gate stays fast, and keep the cheap deterministic checks off the "nightly only"
list so a regression is caught at the earliest tier that can see it.

## See also

- [`regression_runbook_witness.py`](regression_runbook_witness.py) — the hermetic captured-proof harness.
- Epic [#4509](https://github.com/anthony-chaudhary/fak/issues/4509) — the missing-middle validation ladder this runbook serves.

## Preflight a live evaluation endpoint

Run the bounded capability probe before downloading a corpus or starting a quality campaign:

```bash
fak quality probe --endpoint http://127.0.0.1:8080 --model exact-model-id
```

The command queries `/v1/models`, then sends separate one-token generation, completion echo/prompt-logprob, and `reasoning_effort` requests. Each arm is reported independently as `supported`, `unsupported`, or `infrastructure_error`.

Interpret the receipt narrowly:

- Generation may be `supported` while `prompt_logprobs` is `unsupported`; absent logprobs do not erase successful generation evidence.
- Rejection of `reasoning_effort` changes only the reasoning arm.
- An unreachable endpoint, invalid endpoint, or model absent from `/v1/models` is an infrastructure/configuration error, not unsupported model behavior and not a quality regression.
- `native: true` requires explicit `engine: "fak-native"` evidence and `fallbacks: 0`. A generic compatible endpoint remains `openai-compatible`; any fallback prevents a native claim.
- `accuracy_evaluated` is always false. This probe establishes readiness and capability only; it never reports accuracy or a quality pass.

Exit `0` means all requests completed without infrastructure/configuration errors; individual capabilities may still be unsupported. Exit `2` means usage, endpoint, model, transport, or server infrastructure prevented a trustworthy preflight.
