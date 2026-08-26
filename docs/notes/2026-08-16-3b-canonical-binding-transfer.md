# Local 3B canonical bound-routing transfer — 2026-08-16

**Verdict: neither 3B model reaches the frozen 23/24 unique-exact admission bar.**
Qwen2.5-3B and SmolLM3-3B each finish at **80/96 exact calls and 20/24
unique requests** after deterministic binding. Both are perfectly invariant across the
four source-catalog orders. The result rejects a simple parameter-count ladder: these
3B checkpoints do not improve on the frozen-route 1.5B/1.7B evidence.

## Question

Does moving the final canonical named-tool plus deterministic-binding route to two
already-resident 3B local model families identify a smaller admitted routing tier?

This is the captured spine for [#6977](https://github.com/anthony-chaudhary/fak/issues/6977)
under cross-model sensitivity umbrella
[#6692](https://github.com/anthony-chaudhary/fak/issues/6692).

## Frozen protocol

- **Models:** Qwen2.5-3B-Instruct and SmolLM3-3B, loaded from local immutable
  weight sets. Provenance digests are captured below and in the raw artifact.
- **Hardware:** one sanctioned NVIDIA L4-class accelerator; models ran sequentially.
- **Workload:** the same 24 held-out requests and 24 confusable tools used by the
  canonical named-catalog, binding, and sub-500M transfer witnesses.
- **Order control:** canonical input, reversed input, and two seeded input shuffles.
  Every arm canonicalizes to the same lexicographically sorted named-tool catalog.
- **Decode:** unchanged prompt, JSON Schema prefix constraint, greedy generation,
  128-token cap.
- **Postprocessor:** unchanged fail-closed `send_email` literal binder; selected-tool
  changes are forbidden.
- **Calls:** 192 total (24 requests × 4 source orders × 2 models).
- **Admission rule, declared before the run:** at least 23/24 unique requests exact
  after binding.

## Results

| Model | Valid | Correct tool | Exact before | Exact after | Unique exact | Output tokens | Generation sum | Median/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B | 96/96 | 92/96 | 80/96 | **80/96** | **20/24** | 2,316 | 142.662 s | 1296.972 ms |
| SmolLM3-3B | 96/96 | 96/96 | 76/96 | **80/96** | **20/24** | 1,776 | 92.558 s | 940.815 ms |

Each source order produced exactly the same per-model result:

| Model | Canonical | Reversed | Shuffle 20260816 | Shuffle 6692 |
|---|---:|---:|---:|---:|
| Qwen2.5-3B after binding | 20/24 | 20/24 | 20/24 | 20/24 |
| SmolLM3-3B after binding | 20/24 | 20/24 | 20/24 | 20/24 |

Frozen-route reference points:

| Model | Exact calls | Unique exact |
|---|---:|---:|
| Qwen2.5-0.5B | 32/96 | 8/24 |
| Qwen2.5-1.5B | **84/96** | **21/24** |
| Qwen2.5-3B | 80/96 | 20/24 |
| SmolLM2-360M | 0/96 | 0/24 |
| SmolLM2-1.7B | **80/96** | **20/24** |
| SmolLM3-3B | **80/96** | **20/24** |

The binder repaired four SmolLM3 email calls and no Qwen calls. Qwen's four
always-failing requests were all three knowledge-base searches plus one email request;
it selected the correct search tool but produced wrong search arguments, and selected
a notification tool for the email miss. SmolLM3 selected the correct tool on every
call, but all three search arguments and one high-priority ticket argument remained
wrong. This makes argument fidelity—not catalog order or structural validity—the
observed floor at 3B.

## Interpretation and boundary

Neither model meets admission. Qwen2.5-3B is one unique request worse than the frozen
Qwen2.5-1.5B route despite twice the nominal parameters; SmolLM3-3B ties SmolLM2-1.7B.
Parameter count alone is therefore not a valid route selector for these checkpoints.
The family/checkpoint and exact argument behavior matter more than a monotonic size
assumption.

This is a frozen transfer, not a prompt-tuning or quantization study. It does not claim
that every 3B checkpoint fails, nor that the 1.5B model is globally stronger. It proves
only that these two local 3B checkpoints do not clear this route's predeclared bar under
the same prompt, tools, constraints, and requests.

## Reproduction and validation

- Qwen2.5-3B provenance digest: `4a2db554b4dcfb1a854f97b1f0402b8cd631a0ddd539096367674be1ec12a8c3`
- SmolLM3-3B provenance digest: `9789f3c20ddaa9adeaba2c60e41d59a5dfecc4a75966203ece56478c47775c8f`
- Runner SHA-256: `05175d28e2e91e0a68044a5804360ba8706ea37e5c29d16b8cc9855c56c59afe`
- Raw artifact SHA-256: `92934ff65bfc7962a1553ce3cf86ba4d8b380469ffe8f7ab715ec569b8c8b470`
- Captured log SHA-256: `b250dbddfb2cc4038c90ead7e97a982097d1e8a633015a530ef746dd12bf0893`
- Runtime: 331.239 seconds
- Environment: Python 3.10.12, PyTorch 2.13.0+cu130,
  Transformers 5.14.1, `lm-format-enforcer`
  0.11.3, CUDA device `NVIDIA L4`.

An independent validator recomputed the canonical catalog digest and every aggregate,
checked 96 rows per model and 24 per source order, matched every prompt digest to the
captured fixture, proved raw outputs were invariant across source orders, and verified
that binding never changed the selected tool. Remote and local raw hashes matched. The
accelerator returned to 1,022 MiB and 0% utilization after the run.

Captured artifact:
[`2026-08-16-3b-canonical-binding-transfer.json`](../benchmarks/model-sensitivity/2026-08-16-3b-canonical-binding-transfer.json).
