# Cross-family structured tool-routing ablation (2026-08-16)

Issue: #6692

## Verdict

A generic "plan and check" instruction is not a safe cross-model routing default for structured
tool calls. On 24 synthetic requests across eight tools, Qwen2.5-3B is perfect under both
prompts (24/24), SmolLM3-3B falls from 23/24 to 22/24, while Phi-3.5-mini rises from 3/24 to
21/24 only when the deliberate prompt receives the 128-token budget. The same intervention is
neutral, harmful, or strongly beneficial depending on model identity.

Token budget is also treatment-specific. A 32-token cap is fully sufficient for every Qwen and
SmolLM cell and for Phi's concise contract, but truncates five Phi deliberate outputs. Raising
that one cell to 128 tokens recovers three parseable, correct calls (18/24 to 21/24), while the
remaining three failures consistently emit schema key `name` instead of required key `tool`.

## Question and treatment

Can small local instruction models choose a tool and construct exact JSON arguments, and does
an instruction to plan/check improve the result under short and long decode budgets?

- **Contract:** select one tool and return exactly one JSON object with keys `tool` and `args`.
- **Deliberate:** plan the tool and arguments, check each argument against the request, do not
  reveal the plan, then return the same JSON contract.
- **Models:** Qwen2.5-3B-Instruct, SmolLM3-3B, and Phi-3.5-mini-instruct.
- **Battery:** 24 requests across `search_kb`, `get_order`, `list_tickets`, `create_ticket`,
  `update_address`, `refund_payment`, `cancel_order`, and `send_email`; one- and two-argument
  schemas include strings, enums, identifiers, and integers.
- **Matrix:** three models x two prompts x two token caps x 24 tasks = 288 greedy generations.
- **Decode:** temperature 0, sampling off, 32- or 128-generated-token cap.
- **Runtime:** PyTorch 2.13.0+cu130, Transformers 5.15.0, CUDA 13.0, one L4-class accelerator.
- **Native thinking:** disabled for SmolLM3 using `/no_think` and
  `enable_thinking=False`; not applicable to Qwen2.5 and Phi-3.5.
- **Scoring:** parse the first decodable JSON object; separately score parseability, exact tool,
  exact argument object, complete `{tool,args}` equality, and strict whole-output JSON.

## Results

| Model / treatment | Cap | Parse | Tool | Exact args | Complete | Strict | Cap hits | Output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen contract | 32 | 24 | 24 | 24 | 24 | 24 | 0 | 436 |
| Qwen contract | 128 | 24 | 24 | 24 | 24 | 24 | 0 | 436 |
| Qwen deliberate | 32 | 24 | 24 | 24 | 24 | 24 | 0 | 436 |
| Qwen deliberate | 128 | 24 | 24 | 24 | 24 | 24 | 0 | 436 |
| SmolLM contract | 32 | 24 | 24 | 23 | 23 | 23 | 0 | 543 |
| SmolLM contract | 128 | 24 | 24 | 23 | 23 | 23 | 0 | 543 |
| SmolLM deliberate | 32 | 24 | 24 | 22 | 22 | 22 | 0 | 540 |
| SmolLM deliberate | 128 | 24 | 24 | 22 | 22 | 22 | 0 | 540 |
| Phi contract | 32 | 24 | 3 | 24 | 3 | 3 | 0 | 525 |
| Phi contract | 128 | 24 | 3 | 24 | 3 | 3 | 0 | 525 |
| Phi deliberate | 32 | 21 | 18 | 21 | 18 | 18 | 5 | 592 |
| Phi deliberate | 128 | 24 | 21 | 24 | 21 | 21 | 2 | 827 |

Median generation latency ranged from 749-793 ms for Qwen, 933-943 ms for SmolLM, and
847-888 ms for Phi. Arm execution was sequential, so these are workload-local generation
measurements rather than throughput claims.

## Failure anatomy

- **Qwen:** all 96 generations are strict, exact calls. Prompt and cap changes have no observable
  effect in this battery.
- **SmolLM:** all calls parse and select the correct tool. The only failures are exact-argument
  mistakes; deliberate prompting introduces one additional argument error rather than fixing
  the existing one.
- **Phi concise contract:** arguments are exact in all 24 calls, but 21 calls use schema key
  `name` instead of required key `tool`. This is a schema-compliance failure, not a routing or
  extraction failure hidden by the aggregate score.
- **Phi deliberate prompt:** the model adopts `tool` on most tasks. At 32 tokens, five outputs hit
  the cap and three are unparseable. At 128 tokens those three complete correctly; two other
  outputs still hit the cap only after a complete strict JSON object, so cap-hit count alone is
  not equivalent to failure. The final three errors retain `name` for every address-update call.

## Interpretation

1. **Route by exact model and prompt identity.** The deliberate prompt changes complete accuracy
   by 0 for Qwen, -1 for SmolLM, and +18 for Phi at the 128-token cap.
2. **Budget follows output behavior.** A larger cap has zero value in 11 of 12 model/prompt cells
   when 32/128 arms are compared; it matters only for Phi plus deliberate prompting.
3. **Decompose tool-call quality.** A single accuracy number would hide Phi's correct argument
   extraction behind the `name`/`tool` schema mismatch and would conflate truncation with tool
   choice. Parse/tool/argument/strict scores identify different remediation seams.
4. **Schema examples may be higher leverage than generic deliberation.** Phi's residual failures
   are a stable key-name substitution. This study did not test a canonical output example, so
   that is a hypothesis rather than a result.
5. **This is routing evidence, not safety authorization.** Correct generation of destructive
   tools such as refunds and cancellation does not imply they should execute. Structural policy
   remains the independent, default-deny gate.

## Reproduction and artifact

Raw artifact:

- [`2026-08-16-tool-routing-ablation.json`](2026-08-16-tool-routing-ablation.json)
- SHA-256: `2e15b854473d5e5d64cbc5216c60cc72e892e10d6cd110237dffc46029047e37`
- Run window: `2026-08-15T22:55:12Z` to `2026-08-15T23:00:38Z`

Captured weight-set SHA-256 values:

- Qwen2.5-3B-Instruct: `8229c6ddee3ba5c33ce55182898c0a5c6cadf98086cea8c71b3f1f01f346fcae`
- SmolLM3-3B: `a23fcc3bd5d5908c861e9f985327ddd3277edbe5c659cd2b0ff2017a1e372dcd`
- Phi-3.5-mini-instruct: `12be042c8285552d8b4617959a43f66e259bbb84ff27b7469ffe89b6cb27deda`

The artifact captures the full catalog, requests, expected calls, prompts, generated text,
parsed object, each score component, cap hits, tokens, latency, runtime, configs, weight files,
and digests. A conforming verifier must establish:

1. three models, four arms each, 24 unique task rows per arm, and 288 rows total;
2. parsed objects recompute from the first decodable JSON object;
3. tool, argument, complete, strict, and cap-hit decisions recompute from each row;
4. arm counts, output tokens, and median latencies recompute from rows;
5. paired 32/128 outputs are identical except the five Phi deliberate generations recorded in
   the artifact;
6. artifact and weight-set digests match the values above.

## Limits

- Synthetic single-turn requests with an in-prompt catalog; no real tool execution, observations,
  retries, policy decisions, or multi-turn recovery.
- One model artifact per family, fixed model/arm order, one greedy run, and no independent repeat.
- Exact argument equality intentionally rejects semantically plausible schema variation.
- The catalog is small and tool names are semantically distinct; larger or confusable catalogs
  may change routing error rates.
- Token and latency results are workload-local and are not converted to provider cost or
  concurrent serving throughput.
