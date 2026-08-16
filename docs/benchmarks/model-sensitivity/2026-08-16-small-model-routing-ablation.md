# Sub-2B tool routing on a confusable catalog

**Run date:** 2026-08-16 UTC  
**Hardware:** one NVIDIA L4-class accelerator  
**Models:** Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, SmolLM2-360M-Instruct, and SmolLM2-1.7B-Instruct, all FP16  
**Raw artifact:** [`2026-08-16-small-model-routing-ablation.json`](2026-08-16-small-model-routing-ablation.json)  
**Raw SHA-256:** `7b6c511d50ffce5f4d2315ffb36c45bc09d6ca851acf6500b4b70bccec63d448`

## Verdict

Parameter count alone did not predict reliable tool routing below 2B on this exact 24-tool confusable catalog. SmolLM2-1.7B with the concise contract was the only tested arm to reach **24/24** exact calls. Qwen2.5-1.5B reached 20/24 with that contract. The 0.5B Qwen and 360M SmolLM arms mostly failed the one-object protocol, and one validator-feedback retry did not make either deployable.

The result is bounded to these exact local checkpoints, prompts, catalog, validator, deterministic decode, and 128-token cap. It is not a general ranking of model families or sizes.

## Why this test

Earlier local evidence started at 3B. The deployment pool already contained smaller instruct checkpoints, including the same 0.5B family served on the local-model node. This study asks the more useful admission question: which sub-2B checkpoints can satisfy a confusable tool-call contract before backend optimization matters?

## Protocol

- Catalog: the established 24-tool confusable catalog spanning knowledge search, orders, tickets, email, refunds, weather, and invoices.
- Tasks: 24 exact expected calls, one per catalog member.
- Prompt arms:
  - **contract:** concise instruction to return exactly one JSON object with top-level keys `tool` and `args` and no prose;
  - **skeleton:** the same contract plus a canonical `{"tool":"...","args":{...}}` skeleton.
- Gate: deterministic structural validation of one JSON object, exact top-level keys, catalog membership, exact per-tool argument keys, and primitive string/integer/enum constraints.
- Recovery: one retry if and only if structural validation rejects; the retry receives the prior output, exact validator error, original request, and catalog.
- Decode: greedy, temperature 0, 128 generated-token cap.
- Repetition: two complete runs per model/prompt arm: 384 first attempts in total, plus 214 eligible retries.
- Timing: synchronized accelerator generation. Models ran sequentially to avoid residency pressure.

## Results

Both repetitions produced identical text, validity, correctness, token counts, and retry decisions for every row. Latency ranges below span the two runs.

| Model / prompt | First valid | First correct | Retried | Retry correct | Final valid | Final correct | Output tokens | Median total latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B contract | 1 | 1 | 23 | 0 | 1 | 1 | 4,439 | 4,466-4,478 ms |
| Qwen2.5-0.5B skeleton | 0 | 0 | 24 | 0 | 0 | 0 | 5,864 | 6,744-6,746 ms |
| Qwen2.5-1.5B contract | 24 | 20 | 0 | 0 | 24 | 20 | 708 | 934-935 ms |
| Qwen2.5-1.5B skeleton | 21 | 19 | 3 | 0 | 21 | 19 | 564 | 571-574 ms |
| SmolLM2-360M contract | 0 | 0 | 24 | 1 | 16 | 1 | 4,085 | 5,308-5,323 ms |
| SmolLM2-360M skeleton | 0 | 0 | 24 | 1 | 10 | 1 | 3,311 | 3,874-3,899 ms |
| SmolLM2-1.7B contract | **24** | **24** | 0 | 0 | **24** | **24** | 803 | 952-954 ms |
| SmolLM2-1.7B skeleton | 19 | 18 | 5 | 2 | 21 | 20 | 878 | 630-641 ms |

Counts are per 24-request repetition. Output tokens include eligible retries.

## Failure anatomy

### 0.36B and 0.5B: protocol failure dominates

- Qwen2.5-0.5B contract produced only 1/24 structurally valid calls. Its other 23 outputs mixed prose with JSON. Under the skeleton prompt, 19 outputs had no decodable JSON object and five included extra text.
- SmolLM2-360M put extra text around every first attempt in both arms. Retry made 16 contract outputs and 10 skeleton outputs structurally valid, but only one in each arm was semantically exact.
- The retry prompt therefore changed surface compliance for the 360M model without producing useful routing correctness. Structural recovery counts alone would overstate capability.

### 1.5B and 1.7B: exact semantics and prompt fit dominate

- Qwen2.5-1.5B contract was structurally perfect but made four exact-semantic mistakes: shortened an order query, changed a ticket query, shortened a ticket title, and changed email-subject capitalization.
- Its skeleton arm added three Markdown-fence violations that repeated after validator feedback and retained two ticket-title mismatches.
- SmolLM2-1.7B contract was exact on all 24 calls.
- Its skeleton arm duplicated two otherwise-correct search calls; retry repaired both. Three refund calls encoded integer amounts as strings and repeated that error after feedback. One order query was semantically shortened while remaining structurally valid, so the validator correctly did not oracle-retry it.

## Interpretation

1. **The smallest viable model was prompt- and family-specific.** A 1.7B SmolLM2 arm beat the 1.5B Qwen arm, while neither sub-0.5B arm was close to admission.
2. **The JSON skeleton was not a universal improvement.** It reduced median latency and output tokens for some arms, but reduced correctness for both larger checkpoints and catastrophically failed Qwen2.5-0.5B.
3. **Validator retry is not capability creation.** It repaired duplicated JSON from SmolLM2-1.7B, but could not teach integer typing, exact values, or reliable routing to the smaller models in one turn.
4. **Backend speed should be optimized after capability admission.** A fast CPU/GPU comparison of the 0.5B GGUF would not rescue its observed 1/24 contract correctness. The model/prompt contract is the binding bottleneck first.
5. **Fail-closed validation remains necessary.** The 0.36B and 0.5B arms generated large volumes of superficially tool-like but non-executable output. Rejection prevents those outputs from crossing the tool boundary.

## Reproduction and validation

The raw artifact captures platform and accelerator identity, library/CUDA versions, model configurations, per-weight SHA-256 manifests, full catalog/tasks/prompts, every first and retry output, parsed objects, validation errors, correctness, output tokens, and synchronized latency.

Post-capture validation established:

- four model records, two prompt arms, two repetitions, and 24 rows per repetition;
- eligibility equals first-attempt structural rejection on every row;
- all and only eligible rows were retried exactly once;
- first/retry/final validity and correctness plus token totals recompute from row data;
- generated text, validation outcomes, correctness, and token counts are identical across repetitions;
- the local raw artifact digest matches the accelerator-produced SHA-256.

## Bounded deployment conclusion

For this catalog and contract, admit SmolLM2-1.7B with the concise prompt as the tested sub-2B path. Do not admit the tested 0.36B or 0.5B arms for direct tool execution, and do not infer that a canonical JSON skeleton or one validator retry compensates for insufficient instruction-following capacity. Rerun the exact admission witness whenever weights, quantization, prompt, catalog, or token cap changes.
