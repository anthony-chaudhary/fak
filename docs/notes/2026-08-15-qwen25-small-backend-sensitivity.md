# Qwen2.5 small-model backend sensitivity — pure-fak Q8 — 2026-08-15

**Verdict:** Moving the pinned 0.5B checkpoint from pure-fak CPU to pure-fak CUDA preserved the three-arm aggregate accuracy and format ordering exactly while reducing median latency by 1.81–2.14×. This is backend parity at the aggregate decision level, not bitwise output equivalence. The matched 1.5B CUDA startup exceeded the 32 GiB host-RAM envelope and was kernel OOM-killed before serving, so no 1.5B quality comparison is claimed.

## Question

Does the small-model prompt-arm decision survive a backend change when checkpoint, task protocol, prompts, and decoding are held fixed?

## Pinned envelope

- Node class: sanctioned local-accelerator lab host.
- Runtime: pure-fak in-kernel GGUF serve, CPU and CUDA backends, through the OpenAI-compatible chat endpoint.
- Protocol: 24 fixed four-choice tasks; `temperature=0`; `max_tokens=24`; fresh unique trace ID per request.
- Artifact: Qwen2.5-0.5B-Instruct Q8_0 GGUF, SHA-256 `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- CPU result: [`2026-08-15-qwen25-05b-q8-scaffold.json`](../benchmarks/model-sensitivity/2026-08-15-qwen25-05b-q8-scaffold.json), SHA-256 `8e193fe6c6de1436fb1c8f9015665d51a31da69d44a69fc54c16423da2ec776d`.
- CUDA result: [`2026-08-15-qwen25-05b-q8-cuda-scaffold.json`](../benchmarks/model-sensitivity/2026-08-15-qwen25-05b-q8-cuda-scaffold.json), SHA-256 `1c401a8c9a029e68f701ead6e6540035efc018d550b9f1f987bcfcd32d579865`.

## Matched result

| Arm | CPU correct | CUDA correct | CPU strict | CUDA strict | CPU median | CUDA median | Median speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 3/24 | 3/24 | 0/24 | 0/24 | 2,101.800 ms | 983.130 ms | 2.14× |
| concise contract | 16/24 | 16/24 | 22/24 | 22/24 | 1,813.639 ms | 908.558 ms | 2.00× |
| reasoning scaffold | 8/24 | 8/24 | 24/24 | 24/24 | 1,836.902 ms | 1,013.245 ms | 1.81× |

The concise contract remains the measured 0.5B default candidate on both backends. It beats the scaffold by 8/24 on each and preserves the same format-compliance totals.

## Output-level comparison

Aggregate parity did not mean identical text or answers:

- Direct parsed answers matched on 23/24 tasks; correctness matched on 24/24. CUDA emitted 444 completion tokens versus CPU's 450.
- Concise-contract parsed answers and correctness matched on all 24 tasks, with identical 29-token completion totals.
- Scaffold parsed answers matched on 21/24 tasks; correctness matched on 22/24, while aggregate correctness remained 8/24. Both used exactly 24 completion tokens.

Accordingly, this witness supports aggregate routing-decision parity only. It does not claim deterministic row-level equivalence across numerical backends.

## 1.5B operating-envelope result

The same CUDA image and 1.5B Q8 artifact were attempted twice in a temporary isolated container. Both attempts exited with code 137 during model initialization before the endpoint bound. The second attempt first stopped the auxiliary model service, freeing approximately 10.8 GiB, but still failed. The kernel witness recorded:

- first attempt: fak process approximately 15.1 GiB anonymous RSS at kill;
- second attempt: fak process approximately 26.3 GiB anonymous RSS at kill;
- host envelope: 32 GiB RAM, no swap.

The temporary container was removed and the stopped service restored. Because no request ran, there is no 1.5B CUDA quality or latency claim. This is an operational envelope finding, not evidence of a model-quality difference.

## Honest boundary

- Latency is single-host observed wall time, not a portable throughput claim and not net of machine occupancy or monetary cost.
- There are no repeated trials or confidence intervals.
- The parity result covers one checkpoint, one task set, and one decoding setup.
- The 1.5B failure does not imply the checkpoint cannot run on a host with a larger RAM envelope or a lower-memory loader.

## Decision

Treat the 0.5B concise-contract choice as backend-stable for this pinned CPU/CUDA pair. Do not infer bitwise determinism. Keep 1.5B CUDA outside the supported envelope on a 32 GiB no-swap host until a startup-memory witness succeeds.
