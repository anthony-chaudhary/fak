# GLM-5.2 pure-fak 8-GPU result — 2026-07-15

**Verdict:** `[SHIPPED]` The exact 11-shard GLM-5.2 UD-Q4_K_M checkpoint loaded on eight 80GB datacenter GPUs through the pure `fak` CUDA+NCCL expert-parallel path, all ranks joined, and two serialized warm requests returned coherent content. This is an end-to-end functional and baseline-throughput witness, **not** a performant result.

This artifact is the scrubbed public read-back for issue #4777. Private control-channel, hostname, session, and absolute storage details are intentionally omitted.

## Provenance

| Field | Witnessed value |
|---|---|
| fak revision | `57f36579e98053c984df36d2141cd14f45db3dc5` |
| module revisions | `internal/gateway@r485+g57f36579e`; `internal/model@r310+g054e43186` |
| checkpoint | `GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf` (GGUF split auto-load) |
| shards / bytes | 11 / 465,825,525,088 bytes |
| backend | pure `fak`; `cuda,nccl`; no external inference engine |
| topology | 8 ranks, one rank per 80GB CUDA GPU, device process group required |
| serve flags | `--backend cuda --expert-parallel 8 --context-budget-tokens 4096 --model glm-5.2` |
| environment | `FAK_EP_RANK=0..7`, one `CUDA_VISIBLE_DEVICES` GPU per rank, `FAK_Q4K=1`, device PG required |
| exact result SHA-256 | `c4072fd683f794f74b663b02d1f42f7b8a810a58a64efdffe96691602d505032` |
| warm JSONL SHA-256 | `5240de25b9285a40f685109e80206d3c51b03fd9919fa00d76d3d2296918d698` |

The benchmark files were independently read back through a host-bound GPU server control session on 2026-07-16. The bridge advertised the expected host and passed its known-host check before these values were admitted.

## Functional witness

The exact result recorded:

```text
LOAD_READY 2937s
rank 0/8 loads experts [0,32) of 256 resident
rank 1/8 loads experts [32,64) of 256 resident
rank 2/8 loads experts [64,96) of 256 resident
rank 3/8 loads experts [96,128) of 256 resident
rank 4/8 loads experts [128,160) of 256 resident
rank 5/8 loads experts [160,192) of 256 resident
rank 6/8 loads experts [192,224) of 256 resident
rank 7/8 loads experts [224,256) of 256 resident
SMOKE_OK
EP_WITNESS_DONE rc=0 load_s=2937
```

The smoke generation returned HTTP success and coherent `ok` content. Immediately after decode, every GPU reported 21,109 MiB resident; utilization read-back was nonzero on all eight GPUs (31–63%).

## Serialized warm baseline

Both requests used 29 prompt tokens and requested 8 completion tokens. Rates divide API-reported completion tokens by client-observed wall time.

| Arm | Prompt | HTTP | Wall | Completion | Decode rate | GPU memory | GPU utilization |
|---|---|---:|---:|---:|---:|---:|---|
| repeated prompt | `Reply with the single word: ok` | 200 | 37.378 s | 8 tokens | **0.214030 tok/s** | 21,109 MiB/GPU | 24–61% |
| distinct prompt | `Reply with the single word: yes` | 200 | 87.277 s | 8 tokens | **0.091662 tok/s** | 21,127 MiB/GPU | 4–74% |

Returned content was coherent (`ok…` and `yes…`) but included the model's end-of-message marker because generation stopped at the requested token limit.

## Claim boundary

- `[SHIPPED]` Exact full-checkpoint pure-fak load, eight-rank join, device residency, coherent generation, and warm throughput are directly witnessed.
- `[SHIPPED]` The distinct-prompt arm is the honest uncached baseline: **0.091662 tok/s**.
- `[SIMULATED]` No concurrency sweep was run; aggregate throughput at concurrency greater than one is not measured.
- `[STUB]` Cold TTFT was not separated from the 2,937-second first-load readiness time.
- `[STUB]` This revision is not performant. Issue #4843 owns the resident CUDA expert path; its first Q4_K gate/up routing change landed later and is not represented by these measurements.

No external-engine number is presented as a fak result. The artifact establishes the pre-optimization baseline that later resident-device work must beat under the same exact checkpoint and topology.

