# Issue #9044 — exact Q8 Metal residency witness

This directory preserves the scrubbed, validator-readable P=32/T=64 receipt for the
exact Qwen3.8-27B Q4_K_M Apple M3 Pro envelope. The model executed in fak's native
`metal/qwen35-hybrid-session-v1` path. No llama.cpp inference or runtime fallback
participated.

The captured artifact was exactly 17,106,775,008 bytes with SHA-256
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`, from
`unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`.
The temporary measurement overlay is bound in `profile.receipt.json` by source
revision, diff digest, and binary digest; it is evidence only and is not part of the
runtime implementation.

## Result

- Engine/forward path: `fak-native` / `metal/qwen35-hybrid-session-v1`.
- Q8 band: 272 Metal GEMMs during the one P=32 prefill, then 5,120 singleton GEMVs
  plus 3,072 four-weight GEMV groups during 64 decode forwards. That recomputes to
  `272 + 5,120 + 4×3,072 = 17,680` promised Q8 projection executions.
- Fallback: zero promised CPU executions; `FAK_METAL_Q8_UPLOAD` and every candidate
  control were unset.
- Output: all final logits were finite. The capture refuses before writing either
  artifact if a final logit is NaN or infinite.
- Raw lifecycle: 14,833 command buffers and 23,025 encoders. Every event was
  committed, waited, host-read back, and timed. The raw event SHA-256 is
  `803117cc3fcbe540dce8b852731aa0b1da616e71c7b2886280f7472baf37f02b`.
- Timing includes load/setup, prefill, first token, 63 more decode steps,
  verification, and teardown. It is an operating receipt, not a speedup claim.

## Memory and restoration

Darwin `getrusage(RUSAGE_SELF).ru_maxrss` reported an OS peak footprint of
18,132,647,936 bytes. The receipt records 24,595,492,864 logical model-resident
bytes, 38,654,705,664 physical bytes, and a 30,150,672,384-byte Metal recommended
working-set limit. No memory or Q8 admission override was used.

System swap used rose from 2,458.44 MiB to 4,232.31 MiB: an explicit +1,773.87 MiB
over the complete capture-and-supervisor-restoration window. That is material paging,
not a zero-swap claim. The peak process footprint remained below both the 36 GiB
physical envelope and Metal working-set limit, and the exact supervised port-8090
owner was restored with both `/health` and `/v1/models` returning HTTP 200. The
sidecar conservatively binds the whole observed swap delta rather than attributing it
only to modelbench.

## Readback

```console
go test ./docs/_witnesses/issue-9044-q8-metal-residency -count=1
```

The test pins the copied files by SHA-256, revalidates the native-performance graph,
receipt binding, raw Metal event digest/totals, fallback digest/total, exact Q8
operation count, strict controls, finite-output contract, memory values, and service
restoration fields.
