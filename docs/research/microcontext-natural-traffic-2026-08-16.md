# Natural-traffic tool-admission calibration — 2026-08-16

## Verdict

On 120 naturally distributed public GitHub-issue questions (93 tune / 27 frozen held-out), an independently run NVIDIA Llama 3.1 8B selector reached held-out exact-set accuracy **55.6%** (Wilson 95% CI 37.3%–72.4%) while opening 2.00 evidence classes per record. The tuned fixed cascade opened 4.00 classes and reached 0.0% exact-set accuracy because unnecessary authority counts as an error.

The deterministic construction labels tie the semantic selector on this held-out set, so this result does **not** establish a quality gain. It establishes a reproducible natural-traffic measurement seam and a cost/authority envelope. The sample is repository-local and the interval is wide; this is research evidence, not a production-router claim.

## Protocol and leakage control

- Public issue titles plus bounded bodies were SHA-256 ranked without class balancing.
- The frozen split rule is source-hash byte modulo five: 93 tune and 27 test.
- Labels are multi-valued: `repo_search`, `docs_read`, `issue_state`, `commit_state`; clarification remains typed.
- Gold labels use only the independently authored construction/read-back contract and the first `gpt-5.6-sol` judgment. They must agree per label; disagreement remains frozen rather than becoming a negative.
- The model-distinct NVIDIA `meta/llama-3.1-8b-instruct` judgment is held out of gold construction and evaluated as the semantic policy, preventing self-grading.
- The fold retains 92 records with construction/reference disagreement; this limits the witnessed envelope.

## Held-out rows

| Policy | Exact set | Wilson 95% CI | Mean evidence classes | Mean observed work ms | Mean authority classes |
|---|---:|---:|---:|---:|---:|
| deterministic | 55.6% | 37.3%–72.4% | 2.00 | 440.7 | 2.00 |
| tuned_fixed_cascade | 0.0% | 0.0%–12.5% | 4.00 | 1111.5 | 4.00 |
| semantic_admission | 55.6% | 37.3%–72.4% | 2.00 | 440.7 | 2.00 |
| selective_parallelism | 55.6% | 37.3%–72.4% | 2.00 | 440.7 | 2.00 |

The committed report retains 480 typed receipts: all four bounded `git`/`gh` evidence seams for all 120 records. `no_match` remains an observed result, not rewritten as success. Policy cost folds only the evidence classes that policy opened.

## Break-even and falsification

The rule is `avoided fixed-cascade work × (1-cache-hit-rate) − selector cost − error cost`. Sensitivity rows use both a modeled 5 ms selector and observed remote selector wall time per record, at cache-hit rates 0%, 50%, and 90% and error costs 0, 25, and 100 ms. The grid contains 8 winning and 10 losing regimes. Adaptive admission is falsified whenever selector plus error cost exceeds avoided work; observed remote inference and high cache-hit rates are explicit losing regimes.

## Reproduce

```powershell
go run ./cmd/microcontextdemo -natural-traffic-corpus experiments/microcontext/s8u-natural-traffic-corpus-2026-08-10.json -natural-traffic-fold-a experiments/microcontext/s8u-natural-openai-2026-08-10.json -natural-traffic-fold-b experiments/microcontext/s8u-natural-nvidia-2026-08-16.json -natural-traffic-fold-output experiments/microcontext/s8u-natural-multilabel-fold-2026-08-16.json
go run ./cmd/microcontextdemo -natural-traffic-corpus experiments/microcontext/s8u-natural-traffic-corpus-2026-08-10.json -natural-traffic-fold-a experiments/microcontext/s8u-natural-multilabel-fold-2026-08-16.json -natural-traffic-report-output experiments/microcontext/s8u-natural-report-2026-08-16.json
go run ./cmd/microcontextdemo -natural-traffic-corpus experiments/microcontext/s8u-natural-traffic-corpus-2026-08-10.json -verify-natural-traffic experiments/microcontext/s8u-natural-report-2026-08-16.json
```

The second judgment was captured live from NVIDIA's OpenAI-compatible endpoint. Credentials are not stored.

## SHA-256

- `s8u-natural-traffic-corpus-2026-08-10.json`: `c62ea95020119723fd94f53056941932ca3aab449de3db76aace8c2d3b820d8c`
- `s8u-natural-openai-2026-08-10.json`: `42678b59ea7b9efc493d1220f54b0221f1ee5b9e711902f8812c00834464cb19`
- `s8u-natural-nvidia-2026-08-16.json`: `51377d72c7071a3c78323ac2332e88659aa48d69b5340fce5567aba44e953db4`
- `s8u-natural-multilabel-fold-2026-08-16.json`: `08fbda786cb61478527b20e22996bb0c2443ffb232db098a23ebe71dffb875e2`
- `s8u-natural-report-2026-08-16.json`: `650f43c4cdac2d0646cfffba43fc290d83c42a72cfd80c824ae5c83b06168da7`

## Scope boundary

This is one repository's operator traffic, not a cross-domain population. The construction/reference fold leaves disputes visible. Evidence work is observed locally; selector latency includes the remote service path. No net-true production gain is claimed.
