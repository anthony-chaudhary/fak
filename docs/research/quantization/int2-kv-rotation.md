# Output-aware INT2 KV-cache rotation evaluation

This leaf pins and adjudicates one **bounded, observed** evaluation of the output-aware
rotation idea in [OptR, arXiv:2608.02691v2](https://arxiv.org/abs/2608.02691v2). It is not a
claim that OptR is universally best and does not reproduce the paper's model-weight tables.
The record covers candidate rotations, calibration and decode latency, packed memory,
post-`W_O` drift, a declared quality check, repeated-query variance, and a pinned
hardware/software envelope.

## Contract and honest outcomes

`internal/kvint2eval` accepts `kvint2eval/v1` records and returns one typed outcome:

- `supported / KVINT2_SUPPORTED` only for an observed CUDA record with complete artifact,
  recipe, runtime, model, hardware, measurements, candidate search, variance, and digest;
- `unsupported` for unknown contract versions, methods/layouts, incomplete provenance,
  invalid measurements, or a changed witness digest;
- `delegate / KVINT2_MODELED_NOT_OBSERVED` for modeled numbers and
  `delegate / KVINT2_RUNTIME_DELEGATION_REQUIRED` when a valid recipe still needs CUDA.

The supported recipe is deliberately narrow: output-aware orthogonal rotation, INT2,
group size 128, per-token-group K/V quantization, and a post-`W_O` output-NMSE objective.
There is no implicit fallback. Artifact, recipe, runtime delegation, and observed hardware
are separate fields rather than interchangeable claims.

## Named observed witness

```text
TestOutputAwareINT2KVRotationWitness
```

The independent read-back is `internal/kvint2eval/testdata/l4-observed.json`; its producer is
`internal/kvint2eval/testdata/l4_producer.cu`. The latter is a reproducible CUDA-host
reference evaluation, not a production kernel. It was compiled and run through the
sanctioned GCP GPU route on Ubuntu 22.04.5, CUDA 12.9 / driver 580.159.03, and one NVIDIA
L4 (23,034 MiB reported). Producer SHA-256 is
`653f025b233038a199476e1ed8d0b86f688c52a54b92df6abcdd2263470bae44`; binary SHA-256 is
`2a2afb63578807ac7ad798a864a5084818a2039c1c8a4caa83bf2dbe54716f2c`.

Observed envelope: 4,096 cached tokens, 4 KV heads, head dimension 128, output dimension
128, seed 6260. The identity/no-rotation baseline and eight deterministic signed-Hadamard
candidates each search clip ratios `{1.00, 0.95, 0.90, 0.85, 0.80}` on one calibration
query. This makes the no-rotation baseline tuned by the same search budget rather than a
naive strawman. Candidate selection minimizes calibration post-`W_O` NMSE. Ten fresh
queries then measure mean and population standard deviation for output drift, threshold
quality, and decode latency. Rotation 7 was selected.

| Observation | tuned no-rotation INT2 | selected rotation 7 |
|---|---:|---:|
| searched clip ratio | 0.80 | 1.00 |
| calibration + transform + quantize | 306.837196 ms | 319.267520 ms |
| packed K+V cache | 1,048,576 bytes | 1,048,576 bytes |
| post-`W_O` NMSE, mean � sd (10 queries) | 52.353732 � 51.571284 | 1.959262 � 0.431511 |
| bounded threshold quality (`NMSE <= 10`), mean � sd | 0.10 � 0.30 | 1.00 � 0.00 |
| decode/query, mean � sd | 19,907.198 � 1,938.676 us | 19,187.186 � 740.247 us |

All nine candidate rows, not just the selected winner, are in the JSON fixture. The memory
number is the exact packed 2-bit K+V payload; it excludes allocator/workspace overhead.
Calibration timing includes host rotation, five clip candidates' quantize-and-objective
search, and selection. Decode timing is host reference math while a CUDA device-allocation
and synchronization probe witnesses the declared CUDA envelope; therefore it is **not a
CUDA-kernel throughput claim**.

## Scope and reproduction

These are observed synthetic-slice results, not paper results, model quality, or production
throughput. The one-bit threshold task only makes the quality field executable; it does not
stand in for AIME, GPQA, coding, retrieval, or perplexity. Ten queries expose variance but
do not make a statistically powered benchmark. No result may be extrapolated to another
model, context length, runtime, GPU, search space, or benchmark.

Run the contract witness on Linux/WSL:

```bash
go test ./internal/kvint2eval -run TestOutputAwareINT2KVRotationWitness -v
```

To regenerate on a sanctioned CUDA node:

```bash
nvcc -O3 -std=c++17 internal/kvint2eval/testdata/l4_producer.cu -o /tmp/kvint2eval
/tmp/kvint2eval
```

A changed producer, binary, runtime, model envelope, candidate table, or metric requires a
new pin and witness digest. A CPU-only consumer may adjudicate the checked-in observation,
but must return the typed delegate outcome for a request to create fresh CUDA observations.

