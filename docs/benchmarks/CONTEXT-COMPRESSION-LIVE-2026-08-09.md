# Context compression: live native vs LLMLingua-2 witness

Date: 2026-08-09  
Contract: `fak-headroom-comparison/1`  
Issue: #3204 (adapter spine); broader end-to-end comparison remains #6064

## Setup

The same frozen seven-input `headroom.BenchCorpus()` was sent through all three registered arms, five sequential times:

- `none`: tuned local pass-through control; provider prefix-cache behavior is outside this local witness.
- `native`: fak's in-process, recoverable structural compressor.
- `lingua`: fak's first-class HTTP adapter backed by the real `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` model.

The fak process ran on Windows amd64 on an AMD Ryzen 9 9950X (16 cores / 32 logical processors). The LLMLingua service ran on the sanctioned `fak-cuda-build-l4` node: Ubuntu 22.04.5, NVIDIA L4 23,034 MiB, driver 580.159.03, Torch 2.13.0+cu130 with CUDA enabled. An SSH tunnel connected the local adapter to the service, so the `lingua` local-latency measurement includes HTTP, tunnel, and model inference.

The committed CLI equivalent is:

```text
FAK_LINGUA_URL=http://127.0.0.1:18765 FAK_LINGUA_TARGET_RATIO=0.5 fak headroom compare --via none,native,lingua --json
```

Because unrelated peer WIP temporarily prevented rebuilding `cmd/fak` in the shared checkout, this capture invoked the same exported `headroom.CompareBench` function directly from ignored `_scratch/`; the durable runner is the `fak headroom compare` verb, not the scratch shim.

## Captured result

Five sequential runs produced identical byte totals. Timing below is the median per input, with the observed range:

| Arm | Original bytes | Output bytes | Saved | Median ns/input | Range ns/input |
|---|---:|---:|---:|---:|---:|
| none | 9,472 | 9,472 | 0.00% | 0 | 0–75,628 |
| native | 9,472 | 1,746 | 81.57% | 609,928 | 521,100–902,442 |
| lingua | 9,472 | 5,440 | 42.57% | 230,542,428 | 224,074,771–237,128,200 |

The corpus is intentionally dominated by structured logs, ANSI, repeated lines, and pretty JSON, so native structural compression wins total bytes and is about 378x faster by the medians above. This is an observed local result, not a claim that native is generally better.

The issue's dense-prose discriminator behaves differently:

| Arm | Original bytes | Output bytes | Saved | Result |
|---|---:|---:|---:|---|
| none | 303 | 303 | 0.00% | no effect |
| native | 303 | 303 | 0.00% | no effect |
| lingua | 303 | 170 | 43.89% | model reported 54 → 25 tokens |

A direct live service read-back also retained the deliberately planted fact:

```json
{
  "text": "customer account 12345 critical BLUE ORCHID preserve",
  "model": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
  "original_tokens": 21,
  "compressed_tokens": 11
}
```

The adapter's gate test independently resolves the pre-compression bytes from fak's shared CAS and checks byte-for-byte equality, proving that lossy view generation does not discard the recoverable original.

## Honest verdict

`arms_complete=true`, but `complete=false`. This witness proves that the real first-class adapter runs on the same corpus, that LLMLingua compresses dense prose where native does not, and that fak preserves the original. It does **not** establish a net-true winner. The comparison remains pending for:

- task success;
- retained-fact recall over a graded task corpus;
- provider input tokens;
- time to first token;
- context-regrowth tax; and
- total cost.

Those metrics must be gathered under the same model, prompts, cache state, and grader before #6064 or the native benchmark contract can be complete.
