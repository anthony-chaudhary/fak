# GLM-5.2 SGLang/vLLM serving-readiness preflight

- Generated: `2026-06-25T14:20:31.265014Z`
- Model: `zai-org/GLM-5.2` (753.0B)
- Node verdict: **`BLOCKED_ARCH`**
- GPU: `unknown` × `0` — `unknown` (cc `None`)
- Total VRAM: `0.0` GB  |  recommended quant: `None`
- DSA stock kernel floor: sm_90 (Hopper) / sm_100 (Blackwell); supported GPUs: `H100, H200, B200, B300, GB200, GB300`

| Engine | Verdict | Arch OK | Mem OK | Installed | Quant | Need (GB) | Have (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `vllm` | `BLOCKED_ARCH` | False | False | False | `fp8` | 865.9 | 0.0 |

- `vllm`: GPU compute capability is unknown; pass --gpu-name or run on the node.

> KV headroom is modeled as a flat 0.15 fraction of weights, independent of served context length. At long context (the serve script defaults to 131072, up to 1M) the real KV/activation working set is larger and not modeled here — verify on-node before committing to a high `--context-length`.

After a READY/READY_PENDING_INSTALL node serves the endpoint, capture the issue-#130 evidence with `tools/glm52_serving_witness.py --base-url <url>/v1 --engine-cache-engine <engine>`.
