# Cross-model prompt-contract extension — 14B completed, 27B fault isolated — 2026-08-15

**Verdict:** On the fixed 24-task slice, the 14B local model reversed the small-model ordering: the heavier reasoning scaffold reached 24/24 while the concise contract reached 22/24. The planned pure-fak 27B A100 cell did not complete; its 48th fresh-trace request exposed a CUDA GDN preflight panic, now tracked as #6906. These are observed extension results, not native Caveman/Ponytail evidence.

## Why this run

The first #6692 cell on 135M and 0.5B checkpoints found that a concise answer contract beat both direct prompting and a heavier “solve carefully” scaffold. This run held the task definitions, answer extractor, arm prompts, temperature, and output cap constant while moving to larger lab-served models.

## Completed 14B cell

- Node class: sanctioned larger local-model lab host.
- Served model ID: `qwen2.5:14b` through fak's OpenAI-compatible gateway and a local Ollama llama-server.
- Effective runner command identified `/root/.ollama/models/blobs/sha256-2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54`; the gateway/provider did not expose a readable model-file checksum from the host namespace, so this is an effective runtime blob identifier, not a separately verified content hash.
- Protocol: 24 fixed four-choice tasks; greedy decoding (`temperature=0`); `max_tokens=24`; first explicit answer-letter phrase for semantic grading; full-response one-letter regex for strict compliance.
- Raw artifact: [`2026-08-15-qwen25-14b-scaffold.json`](2026-08-15-qwen25-14b-scaffold.json), SHA-256 `6bb7cfac1785a24be9c824a97a97248d4480c59232bd5042691f038bf6bdf79f`.

| Arm | Correct | Accuracy | Strict one-letter | Prompt tokens | Completion tokens | Median request latency |
|---|---:|---:|---:|---:|---:|---:|
| direct | 10/24 | 41.7% | 0/24 | 1,495 | 575 | 10,976.338 ms |
| concise contract | 22/24 | 91.7% | 23/24 | 2,023 | 70 | 4,225.472 ms |
| reasoning scaffold | 24/24 | 100.0% | 24/24 | 2,359 | 48 | 4,229.082 ms |

Observed deltas:

- Concise contract versus direct: **+50.0 percentage points**, 505 fewer completion tokens, and 6.75 seconds lower median request latency.
- Reasoning scaffold versus concise contract: **+8.3 percentage points**, 336 more prompt tokens, 22 fewer completion tokens, and effectively unchanged median latency (+3.61 ms).
- This interaction differs from both smaller checkpoints, where the reasoning scaffold scored 8.3 points below the concise contract. Prompt-arm ordering is therefore model-sensitive on this slice.

No monetary cost is reported because the model was served locally and no provider bill existed. The table reports observed token counts and wall-clock latency; it does not price machine occupancy.

## Planned 27B pure-fak cell: typed failure witness

- Node class: sanctioned 40 GiB CUDA lab host.
- Artifact: `/opt/qwen36-q4k/Qwen3.6-27B-Q4_K_M.gguf`, SHA-256 `33625d8dc3a5dd8d88c324d47db58561b11f7072816287078bfe58b4c55782f9`.
- Server: pure-fak CUDA backend, model ID `Qwen3.6-27B-Q4_K_M`, context budget 4,096.
- Each request used a unique `X-Trace-ID` to avoid cross-request session-budget contamination.
- Raw failure transcript: [`2026-08-15-qwen36-27b-gdn-failure.txt`](2026-08-15-qwen36-27b-gdn-failure.txt), SHA-256 `587ded0dc26747b4d934c82ba87fbfad8140cd130f97fcb1440d46c7446cfebc`.

The server completed 24 direct-arm requests and 10 concise-contract requests. The 35th fresh-trace request failed at layer 0:

```text
handler_panic: backend "cuda" forward "qwen35-gdn" via
"cuda/qwen35-gdn-ssm-decode-v1" failed closed ... cuda Qwen3.5 GDN kernel
failed closed at preflight (code 10002); no CPU fallback; session closed
```

No 27B accuracy headline is reported: the cell is incomplete, and direct responses were not reliably answer-extractable. Issue #6906 carries the exact reproduction and requires 72 fresh-trace requests without handler panic before closure.

## Scope and next interpretation

- **Observed:** the concise contract was strongly positive across 135M, 0.5B, and 14B cells.
- **Observed:** the heavier scaffold was harmful relative to the concise contract on the two small checkpoints but beneficial on 14B.
- **Observed:** the larger pure-fak 27B path is not reliable enough for this ablation envelope; the failure is preserved rather than imputed or silently dropped.
- **Not yet:** strong-frontier provider cell, cost-oriented provider cell, native comparator parity, repeated trials, confidence intervals, calibrated judge adapters, monetary machine cost, or a completed 27B interaction.
- **Decision:** retain all three prompt arms in subsequent model cells. Do not promote either concise-contract or reasoning-scaffold treatment as universally best.
