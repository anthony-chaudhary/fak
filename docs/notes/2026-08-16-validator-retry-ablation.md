# Validator-driven tool-call retry study (2026-08-16)

Issue: #6692

## Verdict

A fail-closed structural validator plus one error-specific retry recovers the confusable-catalog
Phi skeleton from 4/24 to 24/24 correct calls. It does so by rejecting all 20 `name`-key outputs
and converting every retry to strict `{tool,args}` JSON. The recovery is real but expensive:
output grows from 523 to 970 tokens and median per-request generation latency rises from 944 ms
to 1,820 ms because 20/24 requests take a second pass.

The same mechanism cannot repair every error. It does not retry structurally valid but
semantically wrong calls: Phi's concise route remains 22/24 because both `search_tickets`
selections pass schema validation, and SmolLM concise remains 23/24 because punctuation in an
email subject is structurally valid. On Qwen, one malformed wrong-tool call is repaired into a
structurally valid **but still wrong** call. Validation is a recovery seam for detectable
contract violations, not an oracle for intent.

## Protocol

The first attempt uses either the concise `{tool,args}` contract or canonical skeleton against
the 24-tool confusable catalog from the preceding study. A deterministic validator checks:

- exactly one JSON object and no trailing prose;
- top-level keys exactly `tool` and `args`;
- tool membership in the 24-tool catalog;
- exact per-tool argument-key set;
- integer/string/enum primitive constraints.

Only rejected outputs receive one retry. The retry includes the original request, catalog,
previous output, and exact validation error. Structurally valid calls are never retried based on
knowledge of the expected answer. This preserves the operational distinction between what a
kernel can detect and what an offline benchmark oracle knows.

- **Models:** Qwen2.5-3B-Instruct, SmolLM3-3B, Phi-3.5-mini-instruct.
- **Matrix:** three models x two first-attempt prompts x 24 requests = 144 first attempts plus
  39 validator-triggered retries.
- **Decode:** greedy, temperature 0, sampling off, 128 tokens per attempt.
- **Runtime:** PyTorch 2.13.0+cu130, Transformers 5.15.0, CUDA 13.0, one L4-class accelerator.
- **Thinking:** SmolLM3 native thinking disabled with `/no_think` and
  `enable_thinking=False`; not applicable to Qwen2.5 and Phi-3.5.

## Results

| Model / first prompt | First valid | First correct | Retried | Retry valid | Retry correct | Final valid | Final correct | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen contract | 23 | 23 | 1 | 1 | 0 | 24 | 23 | 449 |
| Qwen skeleton | 23 | 23 | 1 | 1 | 0 | 24 | 23 | 449 |
| SmolLM contract | 24 | 23 | 0 | 0 | 0 | 24 | 23 | 543 |
| SmolLM skeleton | 23 | 22 | 1 | 0 | 0 | 23 | 22 | 445 |
| Phi contract | 8 | 6 | 16 | 16 | 16 | 24 | 22 | 893 |
| Phi skeleton | 4 | 4 | 20 | 20 | 20 | 24 | 24 | 970 |

Cost decomposition:

| Model / prompt | First-attempt tokens | Retry tokens | First median latency | Final median total latency |
|---|---:|---:|---:|---:|
| Qwen contract | 436 | 13 | 805 ms | 823 ms |
| Qwen skeleton | 436 | 13 | 765 ms | 799 ms |
| SmolLM contract | 543 | 0 | 962 ms | 962 ms |
| SmolLM skeleton | 423 | 22 | 734 ms | 734 ms |
| Phi contract | 521 | 372 | 907 ms | 1,777 ms |
| Phi skeleton | 523 | 447 | 944 ms | 1,820 ms |

Median totals hide rare retry cost for Qwen and SmolLM because fewer than half their requests
retry. The raw artifact carries per-request and summed retry latency.

## Failure anatomy

- **Phi skeleton:** all 20 first-pass structural failures are `name`/`tool` violations. Every
  error-specific retry is structurally valid and semantically correct, yielding 24/24.
- **Phi contract:** 16 invalid calls recover correctly. Two first-pass calls select
  `search_tickets` with valid `query` arguments, so the validator has no structural reason to
  reject them; final accuracy is 22/24.
- **Qwen:** the wrong `search_tickets` call initially carries invalid `status`; retry changes it
  to valid `query: closed` but preserves the wrong tool. Structure improves while task accuracy
  does not.
- **SmolLM contract:** its one email error differs only by a trailing period in the subject. That
  is valid against the string schema and receives no retry.
- **SmolLM skeleton:** its malformed `list_tickets` argument object is retried once but repeated
  unchanged, so final structural validity remains 23/24.

## Interpretation

1. **Fail closed, then repair detectable violations.** The validator converts Phi's schema drift
   into an explicit retry seam instead of silently accepting an alias.
2. **Do not equate validation with adjudication.** A catalog-valid wrong tool can pass every
   structural check. Intent-level ambiguity needs a separate adjudicator, disambiguation step,
   or higher-quality route.
3. **Retry routing is model- and error-specific.** Phi benefits dramatically; Qwen gains no
   correctness; SmolLM's one attempted retry fails. A universal retry policy adds cost without
   universal value.
4. **Net value depends on the alternative.** Phi skeleton retry reaches 24/24 but nearly doubles
   output tokens and median generation latency. Whether that beats routing directly to another
   model depends on queueing, model residency, and handoff cost, which this study does not price.
5. **Keep policy after successful repair.** Structural recovery only forms a valid proposed call;
   destructive tools still require fak's independent capability/policy decision.

## Reproduction and artifact

Raw artifact:

- [`2026-08-16-validator-retry-ablation.json`](../benchmarks/model-sensitivity/2026-08-16-validator-retry-ablation.json)
- SHA-256: `c45300caa60f66fe36864b714d07f79bbe3216cc8930e52225faedbe98626e08`
- Run window: `2026-08-15T23:47:53Z` to `2026-08-15T23:51:44Z`

Captured weight-set SHA-256 values:

- Qwen2.5-3B-Instruct: `8229c6ddee3ba5c33ce55182898c0a5c6cadf98086cea8c71b3f1f01f346fcae`
- SmolLM3-3B: `a23fcc3bd5d5908c861e9f985327ddd3277edbe5c659cd2b0ff2017a1e372dcd`
- Phi-3.5-mini-instruct: `12be042c8285552d8b4617959a43f66e259bbb84ff27b7469ffe89b6cb27deda`

The artifact captures the validator and retry rule, catalog, tasks, prompts, first and retry
outputs, parsed calls, validation errors, eligibility, correctness, per-attempt tokens/latency,
runtime, configs, weight files, and digests. A conforming verifier must establish:

1. three models, two arms, 24 unique tasks per arm, and 144 first attempts;
2. retry eligibility equals structural rejection exactly and every eligible row attempts once;
3. validation and semantic correctness decisions recompute for first/retry/final calls;
4. arm eligibility, recovery, final, token, and median-latency summaries recompute;
5. Phi skeleton has 20 eligible retries, all 20 recover, and final accuracy is 24/24;
6. artifact and weight-set digests match the recorded values.

## Limits

- Synthetic, single-turn, in-prompt catalog; no actual tool execution, observations, provider
  structured-output API, grammar-constrained decoding, policy decision, or multi-turn user turn.
- One retry maximum; repeated invalid output is not escalated further.
- The validator checks structure and primitive schema, not semantic intent or exact benchmark
  arguments. Final correctness uses the offline oracle only for reporting.
- One artifact per model family, fixed execution order, one greedy run, no independent repeat.
- Sequential generation latency and output tokens are not concurrent serving throughput or
  provider cost.
