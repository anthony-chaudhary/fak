# Recurrent Residual Quantization evaluation

**Named work:** *Recurrent Residual Quantization: A Progressive Multi-Precision Representation for LLMs*  
**Disposition: ABSTAIN** from integration until a checkpoint and executable runtime are independently pinned. The metadata contract is suitable to **MONITOR** and can later **DELEGATE** execution to a pinned external runtime; fak does not claim an RRQ kernel.

## Pinned source and reproduction

| Item | Pin |
|---|---|
| Paper | `arXiv:2608.04048v1` — <https://arxiv.org/abs/2608.04048v1> |
| Public PDF | <https://arxiv.org/pdf/2608.04048v1> |
| PDF SHA-256 | `08d39e39738289bcd2784364a713ed337dc4bca6467df8e81046dfb2cf6dcd45` |
| Recipe | `rrq-paper-algorithm-1/v1`: 2-bit PTQ-or-RTN base, successive 2-bit RTN residuals, effective tiers `[2,4,6,8]`, calibration-free |
| Artifact | Paper only. No released weight URI or weight digest was available in the reviewed v1 materials. |
| Runtime | `external-rrq-runtime` is a delegation boundary, not an implementation. No public package/version/ABI was available in v1; the abstract says code will be released upon publication. |
| Review date | 2026-08-10 |

Bounded public reproduction (PowerShell):

```powershell
Invoke-WebRequest https://arxiv.org/pdf/2608.04048v1 -OutFile 2608.04048v1.pdf
(Get-FileHash 2608.04048v1.pdf -Algorithm SHA256).Hash.ToLower()
go test ./internal/residualquant -run 'TestNamedResearchWitness|TestResearchEvaluationSeparatesEvidence|TestResearchDocumentPinsContract'
```

The expected hash is the pin above. The Go witness independently reads this document and exercises at least three outcomes, including unknown contract, invalid tier, absent executable artifact, and a fully pinned delegate fixture.

## What the paper actually evaluates

RRQ stores a low-bit base plus recurrent quantized residual corrections. A prefix reconstructs an effective precision: base only is 2-bit, then one, two, or three 2-bit residuals produce 4-, 6-, or 8-bit weights. The authors evaluate six model families, use WikiText-2 perplexity and eight LM Evaluation Harness zero-shot tasks, and report quantization experiments on **8 NVIDIA A100 80GB GPUs**. Algorithm 1 is the recipe pinned above.

The following boundaries are mandatory:

- **Source-reported, not locally observed:** Table 5 reports an all-RTN Qwen3-8B 2/4/6/8-bit package construction time of **1,293 seconds**, **3.3x** versus the authors' measured MatGPTQ run. This repository did not reproduce that timing or its quality tables.
- **Source-reported quality:** the paper reports model-dependent 4-bit behavior and competitive 6-/8-bit results. No numeric quality claim is adopted here because neither weights nor runnable code were released and pinned.
- **Observed here:** the public v1 PDF bytes match the SHA-256 pin; its text provides a recipe and reported hardware envelope but no checkpoint digest or runtime ABI.
- **Modeled here:** RRQ prefixes can be represented as artifact tiers and included in cache identity. This is a contract mapping, not evidence that concatenated tensors are directly consumable by an existing runtime or cache transport.

No lab GPU run is warranted for the present acceptance gate: there is no independently identifiable RRQ artifact or runtime to execute. Hardware absence is not the blocker; missing public executable inputs are. Once those inputs exist, the run must use the sanctioned lab/private-comms route and record artifact digest, recipe ID, runtime/version, device, commands, raw outputs, and observed quality/performance separately.

## Typed capability contract

`internal/residualquant` exposes `fak.residualquant/v1` and never silently chooses another precision or kernel.

| Request | Outcome | Reason |
|---|---|---|
| Inspect the exact paper/recipe pin and a tier in `[2,4,6,8]` | `supported` | `RRQ_METADATA_OK` |
| Unknown contract/method, malformed tier chain, unpinned paper/recipe, or unavailable operation | `unsupported` | Specific `RESIDUALQUANT_*` reason code |
| Execute without a weight URI + digest | `unsupported` | `RESIDUALQUANT_ARTIFACT_UNPINNED` |
| Execute without runtime ID + version + device | `unsupported` | `RESIDUALQUANT_RUNTIME_UNPINNED` |
| Execute with all artifact, recipe, and runtime pins | `delegate` | `RRQ_RUNTIME_HANDOFF` and the exact runtime ID |

The delegate result says only that the request is sufficiently identified to hand to an external runtime. It does not say the runtime supports the tensors, that quality is preserved, or that performance improves.

## Artifact tiers and cache transport decision

**Integrate metadata, abstain on executable transport.** A cache key may model a tier as:

```text
(method=rrq, recipe=rrq-paper-algorithm-1/v1,
 base-artifact-digest, ordered-residual-digests, prefix-count,
 runtime-id, runtime-version, device)
```

The ordered residual digest list and prefix count are semantic: omitting either could alias 2-, 4-, 6-, and 8-bit reconstructions. The paper's additive representation makes this mapping plausible, but v1 does not define a stable interchange container or runtime ABI. Consequently fak must not create a fak-only weight format or treat cache transport as executable support.

## Promotion gate

Change the disposition from **ABSTAIN** to integration only when all are independently readable:

1. an official artifact URI and cryptographic digest;
2. an exact conversion recipe/version and ordered residual layout;
3. a runtime/package version and supported device envelope;
4. a clean-room command that loads every claimed tier without hidden fallback;
5. observed quality results against named baselines and datasets; and
6. observed construction/load/serve measurements from sanctioned compute, with raw evidence.

Until then, monitor the source release and return typed unsupported/delegate outcomes rather than inferring support.


