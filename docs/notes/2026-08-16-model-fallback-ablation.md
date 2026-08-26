# Validator retry versus cross-model fallback on a confusable tool catalog

**Run date:** 2026-08-16 UTC
**Hardware:** one NVIDIA L4-class accelerator
**Models:** Phi-3.5-mini-instruct and Qwen2.5-3B-Instruct, both FP16 and simultaneously resident
**Raw artifact:** [`2026-08-16-model-fallback-ablation.json`](../benchmarks/model-sensitivity/2026-08-16-model-fallback-ablation.json)
**Raw SHA-256:** `a4926583b2316975a9fc006cc64015c318d2e000379697473c251864d1d08449`

## Verdict

On this exact 24-tool confusable catalog, keeping recovery on Phi was the only tested route that reached 24/24 correct. Sending Phi's 20 structurally rejected calls to resident Qwen reduced median recovery latency by about 164 ms and saved 79 output tokens per 24 requests, but stopped at 23/24. Changing whether Qwen saw Phi's validator error changed *which* call failed, not the aggregate result.

This does **not** establish that same-model retry generally beats model fallback. It establishes a narrow routing fact for these exact weights, prompts, catalog, deterministic decode, token cap, and dual-resident execution.

## Question and real alternatives

The preceding validator-retry witness recovered Phi's skeleton arm from 4/24 to 24/24. The next operational question was whether a router should instead hand a rejected Phi call to a concise Qwen model.

Three recovery routes received the same 20 validator-rejected first attempts:

1. **Phi retry:** Phi receives its prior output, the exact validator error, original request, and catalog.
2. **Qwen repair:** resident Qwen receives the same repair prompt.
3. **Qwen fresh:** resident Qwen receives the original skeleton prompt, without Phi's output or validator error.

Structurally valid first attempts were never routed. That preserves the production-relevant limitation: this validator does not act as a semantic oracle.

## Protocol

- Catalog: the same 24 confusable tools used by the prior catalog-stress and validator-retry studies.
- Tasks: 24 exact expected tool calls, one per catalog member.
- First attempt: Phi with the canonical JSON skeleton prompt.
- Gate: deterministic structural validation of one JSON object, exact `{tool,args}` keys, catalog membership, exact per-tool argument keys, and primitive string/integer/enum constraints.
- Recovery: exactly once if and only if the first attempt fails structural validation.
- Decode: greedy, temperature 0, 128 generated-token cap.
- Repetitions: three complete rounds (72 first attempts and 60 calls per recovery route).
- Execution: synchronous in-process generation; both models loaded on the same accelerator; candidate recovery-call order rotated by eligible row and round; no network or external queue handoff.
- Timing: CUDA synchronized around generation. `end_to_end_ms` additionally includes local chat-template/tokenization and decode overhead.

The two models reserved 14,080,278,528 accelerator bytes together. Cold sequential loads were 5,284 ms for Qwen and 4,493 ms for Phi in this run, but those one-time values are **not** included in per-call latency because the comparison assumes both models are already resident.

## Results

Every round reproduced the same correctness and token totals:

| Recovery route | Recovery correct | Final correct | Recovery output tokens | Total output tokens | Median recovery end-to-end | Median total end-to-end |
|---|---:|---:|---:|---:|---:|---:|
| Phi retry | 20/20 | **24/24** | 447 | 970 | 954 ms | 1,797-1,810 ms |
| Qwen repair | 19/20 | 23/24 | **368** | **891** | **781-800 ms** | **1,677-1,688 ms** |
| Qwen fresh | 19/20 | 23/24 | 372 | 895 | 819-824 ms | 1,680-1,695 ms |

The Phi first stage itself was stable at 4/24 correct, 523 output tokens per round, and 934-939 ms median end-to-end latency.

Relative to Phi retry, Qwen repair saved 79 output tokens over the 24-request workload (8.1% of final-path output tokens) and reduced median recovery end-to-end latency by 154-174 ms (16-18%). Its median total request latency was 119-133 ms lower. The cost was one unrecovered call per round.

These are observed accelerator-generation results, not provider-cost or concurrent-throughput measurements. They exclude queueing, network transport, model transfer, and eviction/reload costs.

## Stable failure anatomy

Deterministic decode made each failure repeat in all three rounds:

- **Qwen repair:** `ticket-02` selected the correct `create_ticket` tool but omitted required argument `priority`.
- **Qwen fresh:** `list-02` selected the correct `search_tickets` tool but emitted `{"status":"closed"}` instead of the catalog's exact `{"query":"status:closed"}` argument shape.
- **Phi retry:** no residual failure in this sample.

The repair context therefore mattered semantically, but did not improve aggregate correctness: it moved Qwen's single miss from one exact-schema ambiguity to another.

## Interpretation

1. **A fallback model is not automatically a stronger repair model.** Qwen was faster and shorter here, yet Phi used its own malformed output plus validator feedback more effectively.
2. **Model identity is not enough to define a route.** Qwen repair and Qwen fresh had the same score but different stable failures. Routing evidence must bind model, prompt, catalog, and validation contract.
3. **Residency determines whether the latency comparison is actionable.** Dual residency fit at about 14.08 GB reserved on this accelerator. If fallback requires loading or remotely queueing Qwen, this witness supplies no estimate for that added cost.
4. **The validator remains fail-closed.** Both Qwen failures remained structurally invalid and therefore rejectable. No malformed call should execute merely because a fallback was attempted.
5. **Semantic adjudication remains separate.** A structurally valid wrong tool or wrong value would bypass this retry gate. This experiment does not solve that class.

## Reproduction and validation

The raw artifact includes model configuration, per-weight SHA-256 digests, full prompts/catalog/tasks, every generated output, input/output token counts, synchronized timings, validation errors, parsed calls, rotated call order, and summaries.

Recompute checks performed locally after capture:

- exactly three rounds and 24 rows per round;
- exactly 20 structurally rejected/eligible Phi first attempts per round;
- eligibility equals structural rejection for every row;
- all and only eligible rows contain all three recovery calls;
- final validity/correctness, token totals, and latency medians recompute from rows;
- stable failures and aggregate results agree across all rounds;
- local artifact SHA-256 matches the accelerator-produced digest.

## Bounded conclusion

For this exact dual-resident deployment, choose Phi validator retry when the last 1/24 correctness matters; choose Qwen fallback only if the measured ~0.16 s recovery-latency reduction and 79-token workload saving justify retaining a 1/24 rejected call. Do not generalize either choice to another catalog or residency topology without rerunning the route-level witness.
