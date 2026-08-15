# Confusable tool-catalog scale ablation (2026-08-16)

Issue: #6692

## Verdict

The Phi-specific JSON skeleton does **not** survive catalog expansion. It produces 24/24 strict
calls with the original eight tools but only 4/24 after 16 semantically confusable distractors
are appended. The larger catalog causes Phi to resume emitting `name` instead of `tool` on 20
calls—even though the prompt still contains the exact `{"tool":"TOOL_NAME","args":{}}`
skeleton and every argument object remains correct.

Qwen2.5-3B and SmolLM3-3B are substantially more stable. Qwen falls from 24/24 to 23/24 under
both prompts because it chooses `search_tickets` instead of `list_tickets` for one closed-ticket
request. SmolLM's concise contract remains 23/24 at both catalog sizes; its skeleton remains
22/24. The evidence supports keeping concise Qwen/SmolLM routes, but it rejects deploying the
Phi skeleton without catalog identity/size in the routing key and a fallback validator.

## Question and matrix

Does the Phi schema repair and the already-working concise routes survive a larger catalog with
near-neighbor tool names and schemas?

- **Base catalog:** the original eight target tools.
- **Confusable catalog:** the same eight targets plus 16 distractors such as `search_tickets`,
  `get_ticket`, `list_orders`, `refund_order`, `cancel_payment`, and `send_notification`.
- **Requests:** the same 24 target-tool requests used in the prior two studies.
- **Prompts:** concise `{tool,args}` contract versus the canonical JSON skeleton.
- **Models:** Qwen2.5-3B-Instruct, SmolLM3-3B, and Phi-3.5-mini-instruct.
- **Matrix:** three models x two catalogs x two prompts x 24 tasks = 288 greedy generations.
- **Decode:** temperature 0, sampling off, 128-token cap; no generation hit the cap.
- **Runtime:** PyTorch 2.13.0+cu130, Transformers 5.15.0, CUDA 13.0, one L4-class accelerator.
- **Thinking:** SmolLM3 native thinking disabled with `/no_think` and
  `enable_thinking=False`; not applicable to Qwen2.5 and Phi-3.5.
- **Scoring:** first decodable JSON object, with exact tool, exact arguments, complete
  `{tool,args}` shape, and strict whole-output JSON scored separately.

## Results

| Model / prompt | Base complete | 24-tool complete | Delta | 24-tool exact tool | 24-tool exact args | 24-tool tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen contract | 24/24 | 23/24 | -1 | 23/24 | 24/24 | 436 |
| Qwen skeleton | 24/24 | 23/24 | -1 | 23/24 | 24/24 | 436 |
| SmolLM contract | 23/24 | 23/24 | 0 | 24/24 | 23/24 | 543 |
| SmolLM skeleton | 22/24 | 22/24 | 0 | 23/24 | 22/24 | 423 |
| Phi contract | 3/24 | 6/24 | +3 | 7/24 | 21/24 | 521 |
| Phi skeleton | 24/24 | 4/24 | -20 | 4/24 | 24/24 | 523 |

Strict counts equal complete counts in every cell. Median generation latency for the confusable
catalog is 771/767 ms for Qwen contract/skeleton, 973/738 ms for SmolLM, and 911/942 ms for Phi.
Sequential arm execution means these are workload-local generation measurements, not serving
throughput claims.

## Failure anatomy

- **Qwen:** both prompts make the same single error: `search_tickets` replaces `list_tickets` on
  the closed-ticket request. The argument object remains exactly correct.
- **SmolLM concise:** catalog expansion introduces no new failure. The existing email argument
  error remains. Skeleton has the same two failures at both sizes: one tool error and one email
  argument error.
- **Phi skeleton:** all 24 argument objects remain exact and no distractor is selected, but 20
  outputs use `name` rather than `tool`. The failure is schema-key adherence, triggered by
  catalog context rather than inability to identify the target or extract arguments.
- **Phi concise:** 17 calls fail schema-key compliance, two select `search_tickets`, and three
  argument objects are wrong. Its increase from 3 to 6 complete calls is not a useful scale
  improvement; it remains far below an executable routing threshold.

## Interpretation

1. **Catalog identity belongs in the routing key.** Model plus prompt plus token cap is still
   insufficient. The exact same Phi skeleton moves from perfect to 4/24 when only catalog
   context changes.
2. **A one-cell prompt repair is not a capability guarantee.** The preceding 8-tool result was
   real but narrow. This expansion falsifies its use as a general Phi tool-schema route.
3. **Validate structure before policy or execution.** Phi's 20 key substitutions would be caught
   cheaply by requiring exactly `{tool,args}`. A parser that silently accepts `name` would hide
   the regression and weaken the kernel seam.
4. **Tool and argument scores expose different mitigations.** Qwen needs disambiguation for one
   near-neighbor pair; Phi needs robust schema enforcement; SmolLM's residual failures are
   argument construction. A generic “better prompt” diagnosis would conflate them.
5. **No net efficiency claim for shorter SmolLM skeleton output.** Its 22/24 correctness is below
   the concise route's 23/24, so lower tokens and latency do not constitute a net-true gain.

## Reproduction and artifact

Raw artifact:

- [`2026-08-16-confusable-tool-catalog-ablation.json`](2026-08-16-confusable-tool-catalog-ablation.json)
- SHA-256: `ae053a8bb58df670b9b6935b88bbbce99abc2bb06ebc92c0b688a87fdf514310`
- Run window: `2026-08-15T23:33:24Z to 2026-08-15T23:38:35Z`

Captured weight-set SHA-256 values:

- Qwen2.5-3B-Instruct: `8229c6ddee3ba5c33ce55182898c0a5c6cadf98086cea8c71b3f1f01f346fcae`
- SmolLM3-3B: `a23fcc3bd5d5908c861e9f985327ddd3277edbe5c659cd2b0ff2017a1e372dcd`
- Phi-3.5-mini-instruct: `12be042c8285552d8b4617959a43f66e259bbb84ff27b7469ffe89b6cb27deda`

The artifact captures both catalogs, requests, exact prompts, expected calls, generations,
parsed objects, every score component, cap hits, tokens, latency, configs, runtime, weight files,
and digests. A conforming verifier must establish:

1. catalogs contain 8 and 24 unique tools, with the original eight targets preserved;
2. three models, four arms each, 24 unique tasks per arm, and 288 rows total;
3. first-object parsing and tool, argument, complete, strict, and cap decisions recompute;
4. summaries, output-token totals, and median latencies recompute from rows;
5. Phi skeleton arguments remain 24/24 while complete calls fall from 24/24 to 4/24;
6. artifact and weight-set digests match the recorded values.

## Limits

- Synthetic, single-turn, in-prompt catalogs; no actual execution, observations, retries, policy
  decisions, constrained decoding, or multi-turn correction.
- Distractors are designed but not an exhaustive taxonomy of production tool confusion.
- One model artifact per family, fixed execution order, one greedy run, and no independent repeat.
- Exact schema equality intentionally rejects aliases such as `name`.
- Token and latency totals are workload-local and are not provider-cost or concurrent-throughput
  claims.
