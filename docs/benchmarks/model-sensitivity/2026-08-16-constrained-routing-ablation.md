# Grammar-constrained decoding for sub-2B tool routing

**Run date:** 2026-08-16 UTC  
**Hardware:** one NVIDIA L4-class accelerator  
**Models:** Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, SmolLM2-360M-Instruct, and SmolLM2-1.7B-Instruct, all FP16  
**Raw artifact:** [`2026-08-16-constrained-routing-ablation.json`](2026-08-16-constrained-routing-ablation.json)  
**Raw SHA-256:** `3cbde74b02a587a4eff114476d7c5fde5e7d217609b95db791fa7c8539ebc6a4`

## Verdict

Catalog-derived grammar constraints turned every tested output into structurally valid JSON, but only Qwen2.5-0.5B gained substantial exact routing correctness: **1/24 → 13/24**. SmolLM2-360M moved from 0/24 to only 2/24. The 1.5B and 1.7B controls retained their freeform scores, 20/24 and 24/24 respectively.

Constrained decoding therefore separated two failure classes. The 0.5B Qwen had meaningful latent routing knowledge hidden behind protocol failures; the 360M SmolLM largely did not. Structural validity alone remains insufficient for tool admission.

This result is bounded to the exact checkpoints, concise prompt, catalog-derived schema, decoder implementation, greedy decode, and 128-token cap tested.

## Question

The preceding small-model study found that 0.36B and 0.5B checkpoints mostly failed the one-object protocol. One validator-feedback retry did not repair them. This ablation asks whether enforcing the output language *during decoding* can recover usable semantic routing, or merely convert malformed text into schema-valid wrong calls.

## Protocol

- Catalog and tasks: the established 24-tool confusable catalog and 24 exact expected calls.
- Prompt: the concise contract used in the preceding study.
- Arms:
  - **freeform:** ordinary deterministic greedy decoding;
  - **constrained:** the same decode with LM Format Enforcer 0.11.3 prefix-token filtering against a JSON Schema derived directly from the catalog.
- Schema: one `oneOf` object branch per tool; exact `{tool,args}` top-level keys; tool-specific required argument keys; no additional properties; string, integer, and enum constraints.
- Decode: greedy, temperature 0, 128 generated-token cap.
- Repetition: two complete runs per model/arm—384 calls total.
- Timing: synchronized accelerator generation; models loaded sequentially.
- Compatibility note: LM Format Enforcer's Transformers integration imported a tokenizer base class from an internal module removed in Transformers 5.14.1. The run changed that import to the equivalent public `transformers.PreTrainedTokenizerBase`; parser and token-filtering logic were otherwise unchanged.

## Results

Every generated output, parse, validity decision, correctness decision, and token count reproduced exactly across both repetitions.

| Model | Freeform valid | Freeform correct | Constrained valid | Constrained correct | Freeform tokens | Constrained tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 1/24 | 1/24 | **24/24** | **13/24** | 1,495 | 787 |
| Qwen2.5-1.5B | 24/24 | 20/24 | 24/24 | 20/24 | 708 | 708 |
| SmolLM2-360M | 0/24 | 0/24 | **24/24** | 2/24 | 2,606 | 716 |
| SmolLM2-1.7B | 24/24 | 24/24 | 24/24 | 24/24 | 803 | 803 |

Counts and token totals are per 24-request repetition.

Median end-to-end latency by repetition:

| Model | Freeform | Constrained first pass | Constrained second pass |
|---|---:|---:|---:|
| Qwen2.5-0.5B | 1,082-1,097 ms | 2,285 ms | 982 ms |
| Qwen2.5-1.5B | 944-946 ms | 2,330 ms | 1,115 ms |
| SmolLM2-360M | 4,356-4,371 ms | 1,199 ms | 1,028 ms |
| SmolLM2-1.7B | 960-966 ms | 1,191 ms | 1,019 ms |

The constrained first pass includes substantial lazy token-enforcer/cache construction and is not a steady-state estimate. The second pass is warm within the same model/arm. These timings measure synchronous in-process generation only; they do not include network, queue, or provider costs.

## Semantic failure anatomy

### Qwen2.5-0.5B

The constrained arm selected the correct tool on 19/24 calls but only 13 exact calls. Its 11 misses were:

- five wrong tools among confusable neighbors;
- six correct tools with wrong exact argument values, including shortened queries/titles, altered capitalization, and an incorrect refund amount.

The grammar supplied shape and legal tool names; it did not supply task semantics.

### SmolLM2-360M

The constrained arm reached only 2/24 exact calls. It repeatedly collapsed onto a small subset of schema-valid tools—especially `search_kb`, `get_order`, and `get_weather`—regardless of the requested operation. This is valid JSON but not useful routing.

### Controls

- Qwen2.5-1.5B emitted exactly the same outputs in freeform and constrained arms. Its four value-level errors remained unchanged.
- SmolLM2-1.7B likewise emitted the same 24/24-correct outputs. Constraint filtering added no quality because the model already stayed inside the allowed language.

## Interpretation

1. **Constrained decoding can reveal latent capability.** Qwen2.5-0.5B improved by 12 exact calls while emitting fewer tokens.
2. **It cannot manufacture semantic discrimination.** SmolLM2-360M became 24/24 structurally valid but remained 2/24 correct.
3. **Validity metrics alone are dangerously optimistic.** Both smallest models reached 24/24 valid under constraints, despite 11 and 22 semantic failures.
4. **Constraints do not repair exact values.** The 1.5B control's four semantic errors were byte-for-byte unchanged.
5. **Strong models may pay overhead for no quality gain.** SmolLM2-1.7B stayed 24/24 while warm constrained latency was about 59 ms higher than its freeform median in this implementation.
6. **Fail-closed admission still applies.** A schema-valid call is only syntactically admissible; semantic adjudication or a stronger routing model remains necessary before execution when exact correctness matters.

## Reproduction and validation

The raw artifact captures platform and accelerator identity, library/CUDA versions, decoder/library details, the exact derived JSON Schema, model configurations, per-weight SHA-256 manifests, full catalog/tasks/prompt, every generated output, token counts, synchronized timings, parsed calls, and exact correctness.

Post-capture checks established:

- four models, two arms, two repetitions, and 24 rows per repetition;
- structural validity, correctness, and token summaries recompute from row data;
- every output, parsed object, validity result, correctness result, and token count is identical across repetitions;
- every constrained output passes the independent deterministic validator;
- local and accelerator SHA-256 digests match.

## Bounded deployment conclusion

Grammar-constrained Qwen2.5-0.5B is a materially better candidate than its freeform form, but **13/24 is not an execution-admission result** for this catalog. SmolLM2-360M remains unsuitable despite perfect structural validity. The tested SmolLM2-1.7B concise arm remains the smallest witnessed 24/24 route; constrained decoding should be treated as a protocol mechanism, not a semantic safety boundary.
