# NVIDIA NIM coding seats - 2026-07-06

Public benchmark snapshot for the built-in opencode NIM seat trio. Ranking favors
coding-agentic strength (SWE, Terminal Bench, LiveCodeBench, long-context/tool use),
not price, latency, or quota.

| Rank | Seat tag | Model ID | Why it is here |
|---|---|---|---|
| 1 | `opencode-nim-deepseek-v4-pro` | `deepseek-ai/deepseek-v4-pro` | DeepSeek V4 Pro reports 1.6T total / 49B active parameters, 1M context, LiveCodeBench 93.5, Terminal Bench 2.0 67.9, SWE Verified 80.6, and SWE Pro 55.4. |
| 2 | `opencode-nim-kimi-k26` | `moonshotai/kimi-k2.6` | Kimi K2.6 reports SWE-Bench Pro 58.6, SWE-Bench Verified 80.2, SWE-Bench Multilingual 76.7, Terminal-Bench 2.0 66.7, and LiveCodeBench v6 89.6. NVIDIA also marks this API unsupported after 2026-07-07, so treat this seat as short-lived. |
| 3 | `opencode-nim-glm52` | `z-ai/glm-5.2` | GLM-5.2 reports 753B parameters, 1M input/output context, SWE-bench Pro 62.1, NL2Repo 48.9, Terminal Bench 2.1 81.0, MCP-Atlas 76.8, and Tool-Decathlon 48.2. |

The built-in default route weights encode that order: DeepSeek 30, Kimi 20,
GLM-5.2 10. If Kimi is unavailable after 2026-07-07, set
`"opencode:nim-kimi-k26": 0` or exclude the seat in the live host policy.

Sources:
- https://build.nvidia.com/deepseek-ai/deepseek-v4-pro/modelcard
- https://build.nvidia.com/moonshotai/kimi-k2.6/modelcard
- https://build.nvidia.com/z-ai/glm-5.2/modelcard
- https://build.nvidia.com/stepfun-ai/step-3.5-flash/modelcard
- https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard
