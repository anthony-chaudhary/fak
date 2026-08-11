# CubicQuant bounded evaluation (arXiv:2608.06763v1)

## Verdict

**Supported for bounded scalar-reconstruction research; delegate model quality and hardware performance.** The independently read fixture produces a deterministic 24-row ledger (three public synthetic distributions x W1-W8). `integrate` means CubicQuant beat both tuned baselines in this fixture; otherwise the row says `abstain`. This is not a universal format selection.

## Pinned contract and provenance

| Dimension | Pin |
|---|---|
| Research artifact | `arxiv:2608.06763v1`, <https://arxiv.org/abs/2608.06763v1> |
| Artifact digest | v1 PDF SHA-256 `245a523fd1b06203c123e6b03c39ed8e3cef107dd7cc33917b280511e71c9df0` |
| Fixture | `internal/cubicquanteval/testdata/evaluation-v1.json`; schema `fak.cubicquanteval.fixture/v1` |
| Recipe | `cubicquant-bounded-reconstruction-v1@1`; seed 42; 384 samples/distribution; groups of 128; W1-W8; shape grid step 0.25; 17 scale candidates |
| Runtime | `fak/internal/cubicquanteval@contract-v1`; Go stdlib CPU delegate |
| Model | `synthetic-reference-distributions@splitmix64-boxmuller-inversecdf-v1`; no weights (`weights_sha256=none`) |
| Fixture digest | SHA-256 `9e6cab3b9593d08157113e99349241a3b658216122ff8f550de0417e9d4d02a8` |

The paper's format is preserved rather than replaced by a fak-only artifact: W1 is the paper's separately defined symmetric binary case. For W2-W8, `M=2^(B-1)-1`, `t=i/M`, and reconstruction magnitudes are `s[a*t + b*t^2 + (1-a-b)*t^3]`; the implementation rejects shape candidates whose derivative is negative on `[0,1]`. The paper and this fixture both use group size 128 and seed 42. This bounded fixture deliberately lowers the paper's 15,360 samples per distribution to 384 (three groups) so the complete W1-W8 contract remains fast and reviewable.

## Evidence taxonomy

- **Observed here:** scalar reconstruction RMSE from the committed deterministic fixture, evaluated by the Go leaf. The source inputs are **modeled synthetic distributions**, so “observed” describes execution of the pinned recipe, not measurements from model weights.
- **Reported by the paper, not reproduced here:** at W4/G128/15,360 samples, CubicQuant reports RMSE reductions versus optimally clipped INT4 of 3.90% (Uniform), 13.49% (Gaussian), and 28.14% (Laplace), and versus its enumerated finite-float baseline of 3.90%, 9.44%, and 6.27%. Those are artifact claims, not rows in our observed ledger.
- **Not observed / delegated:** downstream model quality, task accuracy, GPU kernel throughput, latency, memory, and cross-device behavior. `model-quality` returns `delegate/CUBICQUANT_QUALITY_REROUTE`; `hardware-performance` returns `delegate/CUBICQUANT_ACCELERATOR_REROUTE` and must go through the sanctioned private-comms/lab route before any performance claim.

## Baselines and decision rule

“Tuned uniform” searches clipping scale for the cubic family's exact uniform special case `(a,b)=(1,0)`. “Tuned non-uniform” is a symmetric per-group Lloyd-Max reconstruction oracle initialized from magnitude quantiles; it is a quality reference, **not** an executable format or performance baseline. CubicQuant searches the pinned feasible cubic shape/scale grid. All three use each same group. A row is `integrate` only when cubic RMSE is strictly lower than both tuned baselines; ties and losses abstain. Percentage is `(baseline-cubic)/baseline*100`; negative means cubic lost.

## Independent result ledger

| Distribution | Width | Cubic RMSE | Tuned uniform RMSE | Tuned non-uniform RMSE | vs uniform | vs non-uniform | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| uniform | W1 | 0.286632 | 0.286632 | 0.286632 | 0.00% | 0.00% | abstain |
| uniform | W2 | 0.190458 | 0.190458 | 0.190126 | 0.00% | -0.17% | abstain |
| uniform | W3 | 0.078525 | 0.080111 | 0.078138 | 1.98% | -0.49% | abstain |
| uniform | W4 | 0.035324 | 0.037172 | 0.034541 | 4.97% | -2.27% | abstain |
| uniform | W5 | 0.017559 | 0.018147 | 0.016731 | 3.24% | -4.95% | abstain |
| uniform | W6 | 0.008603 | 0.009110 | 0.008120 | 5.57% | -5.94% | abstain |
| uniform | W7 | 0.004177 | 0.004561 | 0.003234 | 8.42% | -29.17% | abstain |
| uniform | W8 | 0.002139 | 0.002216 | 0.000139 | 3.44% | -1,442.72% | abstain |
| gaussian | W1 | 0.591627 | 0.591627 | 0.591627 | 0.00% | 0.00% | abstain |
| gaussian | W2 | 0.481985 | 0.481985 | 0.436324 | 0.00% | -10.47% | abstain |
| gaussian | W3 | 0.200548 | 0.213962 | 0.199902 | 6.27% | -0.32% | abstain |
| gaussian | W4 | 0.089293 | 0.105564 | 0.102086 | 15.41% | 12.53% | integrate |
| gaussian | W5 | 0.044038 | 0.054152 | 0.058876 | 18.68% | 25.20% | integrate |
| gaussian | W6 | 0.020757 | 0.026265 | 0.033748 | 20.97% | 38.50% | integrate |
| gaussian | W7 | 0.010377 | 0.012873 | 0.013135 | 19.39% | 21.00% | integrate |
| gaussian | W8 | 0.005021 | 0.006460 | 0.000357 | 22.28% | -1,307.53% | abstain |
| laplace | W1 | 0.708023 | 0.708023 | 0.708023 | 0.00% | 0.00% | abstain |
| laplace | W2 | 0.544137 | 0.544137 | 0.503585 | 0.00% | -8.05% | abstain |
| laplace | W3 | 0.225873 | 0.257572 | 0.228876 | 12.31% | 1.31% | integrate |
| laplace | W4 | 0.107411 | 0.130767 | 0.134655 | 17.86% | 20.23% | integrate |
| laplace | W5 | 0.047483 | 0.066849 | 0.087658 | 28.97% | 45.83% | integrate |
| laplace | W6 | 0.023619 | 0.032619 | 0.064932 | 27.59% | 63.63% | integrate |
| laplace | W7 | 0.011637 | 0.016069 | 0.030279 | 27.58% | 61.57% | integrate |
| laplace | W8 | 0.005897 | 0.008083 | 0.000099 | 27.05% | -5,871.86% | abstain |

The ledger's narrow envelope is intentional: 384 synthetic values per distribution on the runtime named by the witness, with no model weights and no accelerator kernel. In particular, high-width percentage deltas can look large when the oracle denominator is near zero; the absolute RMSE columns are authoritative and no throughput or model-quality inference is permitted.

## Typed boundary behavior

| Input | Outcome / reason |
|---|---|
| Pinned reconstruction fixture | `supported / CUBICQUANT_EVALUATED` |
| Unknown schema | `unsupported / UNKNOWN_SCHEMA_VERSION` |
| Tampered artifact, recipe, runtime, or model provenance | `unsupported / PROVENANCE_MISMATCH` |
| Unknown scope or unsupported recipe combination | `unsupported / CUBICQUANT_COMBINATION_REJECTED` |
| Model-quality request | `delegate / CUBICQUANT_QUALITY_REROUTE` |
| Hardware-performance request | `delegate / CUBICQUANT_ACCELERATOR_REROUTE` |

There is no silent fallback. The result repeats artifact, recipe, runtime, and model provenance, labels evidence as `observed` over `modeled-synthetic` inputs, reports its exact OS/architecture/Go envelope, and carries the fixture digest.

## Reproduction and witness

From the repository root:

```text
go test ./internal/cubicquanteval -run TestPinnedFixtureProducesCompleteObservedLedger -v
# Windows host policy: run the package/full validation under WSL as required by AGENTS.md.
fak validate --mine internal/cubicquanteval --mine docs/research/quantization/cubicquant.md
fak buildcheck --vet --mine internal/cubicquanteval/contract.go --mine internal/cubicquanteval/doc.go
```

`TestPinnedFixtureProducesCompleteObservedLedger` independently reads the committed JSON fixture, requires all 24 rows and pinned provenance, and rejects any result not labeled observed-over-modeled. Boundary tests prove unknown schema, tampered provenance, unsupported scope, and both delegation paths. The deterministic read-back test evaluates the fixture twice and compares the serialized ledgers.


