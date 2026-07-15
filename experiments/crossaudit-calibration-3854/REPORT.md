# Cross-audit calibration report — issue #3854

Date: 2026-07-15  
Schema: `fak-crossaudit-calibration/v1`  
Corpus: `fak-crossaudit-accidental-corpus/v1` (24 frozen samples; 12 corrupt / 12 clean)  
Run manifest: [`run-manifest.json`](run-manifest.json)  
Derived report: [`report.json`](report.json)

## Verdict

**Not yet safe to enable a production closure gate.** Both live hosted arms found
all 12 corrupt fixtures (observed recall 1.00), but the sample is small and the
95% Wilson lower bound is only 0.758. The Claude-high arm's observed false-positive
rate is 0.429 (3 false positives among 7 non-abstained clean decisions), above the
predeclared 0.10 floor. GPT-xhigh observed one clean false positive and two clean
abstentions; its non-abstained FPR is 0.10, but the 95% upper bound is 0.404.
The local/open-weight arm is explicitly **not-yet**, so the required three-way
route has not been calibrated.

These are witnessed observations on this frozen synthetic corpus, not a general
accuracy claim. They do not justify lowering independence, structural tests, or
policy gates.

## Predeclared design and budget

- Maximum 24 samples per arm; maximum total provider spend USD 5.00.
- High-severity classes: `reverted_safety_check`, `build_poison`, and
  `race_lost_update`.
- Candidate production floors: high-severity recall >= 0.90 and FPR <= 0.10.
- Held-out pair prefixes were declared before report derivation:
  `docs_cli_drift` and `clean_hard_refactor`.
- Prompt contract: `issue-resolution-audit/v2`; every observation binds its exact
  bundle digest and request-envelope digest.
- Identity and weights/revision labels are the exact configured/read-back facts
  available to this run. Hosted services did not expose immutable weight hashes,
  so their service revision labels are provenance limitations, not hash claims.

## Results

| Arm | Samples | TP / FP / TN / FN | Abstain / unavailable | Precision | Recall | FPR* | Recall 95% low | FPR 95% high | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude Opus high | 24 | 12 / 3 / 4 / 0 | 5 / 0 | 0.800 | 1.000 | 0.429 | 0.758 | 0.750 | 12.52 s |
| GPT 5.6 sol xhigh | 24 | 12 / 1 / 9 / 0 | 2 / 0 | 0.923 | 1.000 | 0.100 | 0.758 | 0.404 | 24.01 s |
| Local/open-weight | 0 | not-yet | not-yet | not-yet | not-yet | not-yet | not-yet | not-yet | not-yet |

`*` FPR is computed over non-abstained clean classifications (`FP/(FP+TN)`).
Pairwise Claude/GPT disagreement was 6/24 (0.25).

Per-class rows, p95 latency, truncation counts, and Wilson intervals are in the
machine-derived `report.json`. No evidence bundle was truncated in this run.

## Cost and token accounting

- Claude CLI read-back reported 528,580 input and 5,467 output tokens. The
  calibration observation format retained token counts but the current CLI
  adapter did not bind its dollar amount into `AuditTokenCost`, so USD cost is
  **not-yet** in the derived table rather than modeled.
- Codex CLI printed aggregate token usage to stderr, but the current structured
  reviewer result did not carry usage into `AuditTokenCost`; GPT token and dollar
  cost cells are **not-yet**.
- The run remained below the predeclared external USD 5.00 cap by operator account
  observation, but this is not a receipt-bound cost witness and is not used as a
  headline claim.

## Local/open-weight arm

No authenticated sanctioned-node bridge was active for this session. The arm is
therefore present in the manifest and report as `not-yet`, not silently omitted or
replaced by a fake local result. To complete it, use the sanctioned compute route
from `docs/fleet-compute-nodes.md`, expose the configured model through the same
OpenAI-compatible audit contract, record its exact weights content digest, and
re-run the frozen 24 bundles without changing labels or thresholds.

## Reproduction

The report is a pure fold of the committed manifest plus the two scrubbed
observation arrays:

```powershell
go run ./cmd/crossauditcalibrate `
  --manifest experiments/crossaudit-calibration-3854/run-manifest.json `
  --observations gpt-xhigh=experiments/crossaudit-calibration-3854/gpt-xhigh-observations.json `
  --observations claude-high=experiments/crossaudit-calibration-3854/claude-high-observations.json `
  --out experiments/crossaudit-calibration-3854/report.json
```

The fold refuses unknown truth IDs, duplicate arm/sample rows, bundle-digest
mismatches, incomplete auditor provenance, prompt-version drift, arm identity
mismatch, sample-cap overflow, and cost-cap overflow. Tests deliberately mutate a
bundle digest and prompt version and require refusal.

## Threshold decision

- **Claude-high:** uncalibrated for gating; observed FPR exceeds the declared floor.
- **GPT-xhigh:** promising but uncalibrated for gating; confidence bounds are too
  wide, cost telemetry is incomplete, and no local fallback arm exists yet.
- **Local/open-weight:** not-yet; no run was fabricated.
- **Overall:** keep cross-audit advisory/fail-open. Do not enable #3860 until a
  larger held-out corpus, receipt-bound cost telemetry, and the local arm satisfy
  the predeclared thresholds with acceptable confidence bounds.

## Limitations

The corpus is synthetic and small, clean samples are intentionally terse, hosted
model revisions are service labels rather than immutable hashes, wrapper-attack
robustness is covered by the separate adversarial corpus rather than recomputed
in this accidental-corpus run, and no claim is made that frozen scores generalize
to future model revisions. Every future model/revision/prompt/policy change must
produce a new report rather than inherit this one.
