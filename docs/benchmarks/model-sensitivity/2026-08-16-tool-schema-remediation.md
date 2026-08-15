# Cross-family tool-schema remediation study (2026-08-16)

Issue: #6692

## Verdict

A literal output-key constraint or canonical JSON skeleton completely repairs Phi-3.5-mini's
stable `name`-instead-of-`tool` failure: complete strict calls rise from 3/24 to 24/24. The
skeleton is the cleaner Phi treatment because it reaches the same result with 523 output tokens
instead of 645 for the literal warning and 525 for the broken baseline.

That remediation must not become a generic cross-model prompt. Qwen2.5-3B remains 24/24 under
the warning and skeleton, but skeleton plus plan/check regresses it to 22/24. SmolLM3-3B falls
from 23/24 to 22/24 under every remediation treatment. Route the schema skeleton specifically
to the affected Phi model/prompt cell; preserve each already-working concise path.

## Question and treatments

Can direct schema constraints repair the Phi `name` -> `tool` substitution observed in the
preceding tool-routing study without harming other small-model families?

All treatments use the same eight-tool catalog, 24 requests, greedy decoding, and 128-token cap:

1. **Contract:** return exactly one object with keys `tool` and `args`.
2. **Literal-key warning:** add “The first key must be spelled `tool`, never `name`.”
3. **Skeleton:** require copying `{"tool":"TOOL_NAME","args":{}}`, preserving the `tool` key.
4. **Skeleton + deliberate:** privately plan/check tool and arguments, then copy the skeleton.

Models are Qwen2.5-3B-Instruct, SmolLM3-3B, and Phi-3.5-mini-instruct. The matrix contains
three models x four prompts x 24 tasks = 288 generations on one L4-class accelerator using
PyTorch 2.13.0+cu130, Transformers 5.15.0, and CUDA 13.0. SmolLM3 native thinking is disabled
with `/no_think` plus `enable_thinking=False`.

Scoring parses the first decodable JSON object and separately checks the exact tool, exact
argument object, complete `{tool,args}` shape, and strict whole-output JSON.

## Results

| Model / treatment | Parse | Tool | Exact args | Complete | Strict | Output tokens | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen contract | 24 | 24 | 24 | 24 | 24 | 436 | 788.261 ms |
| Qwen literal key | 24 | 24 | 24 | 24 | 24 | 436 | 749.762 ms |
| Qwen skeleton | 24 | 24 | 24 | 24 | 24 | 436 | 738.505 ms |
| Qwen skeleton + deliberate | 24 | 24 | 22 | 22 | 22 | 438 | 746.569 ms |
| SmolLM contract | 24 | 24 | 23 | 23 | 23 | 543 | 941.321 ms |
| SmolLM literal key | 24 | 24 | 22 | 22 | 22 | 545 | 938.287 ms |
| SmolLM skeleton | 24 | 23 | 22 | 22 | 22 | 416 | 691.861 ms |
| SmolLM skeleton + deliberate | 24 | 23 | 22 | 22 | 22 | 414 | 690.986 ms |
| Phi contract | 24 | 3 | 24 | 3 | 3 | 525 | 844.341 ms |
| Phi literal key | 24 | 24 | 24 | 24 | 24 | 645 | 1,062.154 ms |
| Phi skeleton | 24 | 24 | 24 | 24 | 24 | 523 | 851.659 ms |
| Phi skeleton + deliberate | 24 | 24 | 24 | 24 | 20 | 778 | 885.818 ms |

No generation hit the 128-token cap.

## Failure anatomy

- **Phi baseline:** reproduces the prior result exactly—21 calls use `name`, while all 24
  argument objects are correct.
- **Phi literal key and skeleton:** both eliminate every key substitution. The skeleton does so
  without increasing output length relative to baseline.
- **Phi skeleton + deliberate:** all first JSON objects are correct, but four outputs continue
  with a visible `Check:` explanation, violating the strict no-prose contract and adding 255
  tokens versus skeleton alone.
- **Qwen deliberate regression:** the combined instruction introduces two argument errors even
  though contract, literal warning, and skeleton remain perfect.
- **SmolLM remediation regression:** literal warning introduces an additional argument error;
  skeleton treatments also produce one tool error. Their lower token and latency totals do not
  offset reduced correctness under a net-true standard.

## Interpretation

1. **Repair the observed seam directly.** Phi already extracted every argument correctly; a
   canonical key skeleton fixes schema adherence without paying for generic deliberation.
2. **Do not globalize a local fix.** The same prompt family is neutral on Qwen and harmful on
   SmolLM. Model identity and exact prompt remain mandatory routing keys.
3. **Plan/check can leak into output.** Even when told not to reveal the plan, Phi emits visible
   checks in four cases. Complete-call and strict-contract scores must stay separate.
4. **Lower tokens are not automatically a gain.** SmolLM skeleton outputs are about 23% shorter
   than baseline, but lose one correct call. This study therefore does not report that cell as
   an efficiency improvement.
5. **Policy remains independent.** Better schema compliance for refund, cancellation, address,
   and email calls does not authorize execution; fak's structural policy checkpoint still
   decides capability.

## Reproduction and artifact

Raw artifact:

- [`2026-08-16-tool-schema-remediation.json`](2026-08-16-tool-schema-remediation.json)
- SHA-256: `b6d078540124e2f9dcd9ea0cdf0a02366f1c7353dea02009a223040089fd822a`
- Run window: `2026-08-15T23:10:37Z to 2026-08-15T23:15:51Z`

Captured weight-set SHA-256 values:

- Qwen2.5-3B-Instruct: `8229c6ddee3ba5c33ce55182898c0a5c6cadf98086cea8c71b3f1f01f346fcae`
- SmolLM3-3B: `a23fcc3bd5d5908c861e9f985327ddd3277edbe5c659cd2b0ff2017a1e372dcd`
- Phi-3.5-mini-instruct: `12be042c8285552d8b4617959a43f66e259bbb84ff27b7469ffe89b6cb27deda`

The artifact captures the catalog, requests, exact prompts, expected calls, generations, parsed
objects, score components, cap hits, tokens, latency, model configs, runtime, weight files, and
digests. A conforming verifier must establish:

1. three models, four arms each, 24 unique tasks per arm, and 288 rows total;
2. first-object parsing and tool, argument, complete, strict, and cap decisions recompute;
3. arm counts, output-token totals, and median latencies recompute from rows;
4. the Phi baseline reproduces 21 `name` substitutions and both direct schema treatments remove
   all of them;
5. artifact and weight-set digests match the recorded values.

## Limits

- Synthetic single-turn routing with an in-prompt catalog; no real execution, observations,
  retries, policy decisions, or multi-turn correction.
- One model artifact per family, fixed execution order, one greedy run, and no independent
  repeat or randomized prompt order.
- A small, semantically distinct tool catalog; confusable or much larger catalogs may differ.
- Exact schema equality intentionally rejects plausible aliases.
- Sequential generation latency is not concurrent serving throughput, and token totals are not
  converted to provider cost.
