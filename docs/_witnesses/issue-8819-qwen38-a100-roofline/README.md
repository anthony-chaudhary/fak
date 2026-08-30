---
title: "Issue #8819 — Qwen3.8 Q4_K_M A100 bottleneck checkpoint"
description: "Reference documentation for Issue #8819 — Qwen3.8 Q4_K_M A100 bottleneck checkpoint, preserving the page's implementation details, evidence, and operating context."
---

# Issue #8819 — Qwen3.8 Q4_K_M A100 bottleneck checkpoint

This checkpoint ran the exact GGUF (`7e78da5d…`) through the fak-native CUDA
`cuda/qwen35-gdn-ssm-decode-v1` path on the sanctioned 40 GiB A100. The matrix
was five unique cold requests followed by five identical requests; the latter
produced four confirmed 22-token prefix hits.

## Result

`fak nativeperf profile --receipt receipt.json` ranks the cache-hit decode path
first: 4.701 s of the 4.730 s median hit. Prefix device cloning copied 161,218,560
bytes in a 29.15 ms median (0.6%). No host stage, host restore, or backend transfer
occurred, so those suspected operations are exonerated for this envelope. The one
next lever is **Qwen GDN recurrent-state decode after device-prefix restore**.

The receipt is deliberately `HOLD_QUALITY_BELOW_FLOOR`: this fresh-current-origin
run did not reproduce the earlier exact-Q38 quality pass. Performance attribution
is still usable because all four hits took the same 4.71–4.73 s decode path, but
no gain may be claimed until quality is restored.

Raw evidence is scrubbed: `prefix-profile.jsonl` and `rows.json`; the raw server log was used to derive `profile.json` but is intentionally not committed as regenerable output.
The profile events are opt-in via the explicit `model.SetPrefixProfilePath` config API; ordinary execution pays no
clock or I/O overhead.
