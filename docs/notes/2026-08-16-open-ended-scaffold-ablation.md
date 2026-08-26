# Cross-family open-ended scaffold and token-cap study (2026-08-16)

Issue: #6692

## Verdict

Removing answer options and answer letters does not reveal a general correctness benefit from
the exact solve/check scaffold. Across three small instruction-model families and 24
exact-answer tasks, the scaffold is tied on Qwen2.5-3B (21/24 versus 21/24), loses one semantic
answer on SmolLM3-3B (22/24 versus 23/24), and gains one on Phi-3.5-mini-instruct (24/24 versus
23/24). The direction is model-specific and the pooled total is exactly tied at 67/72.

An 8-token generation cap is sufficient for this answer-only workload: all 288 generations
terminated within eight tokens, and every 8-token output exactly matches its paired 64-token
output. This is a bounded token-cap result, not evidence that eight tokens suffice for
reasoning-visible or general open-ended tasks.

## Question and treatment

Does the exact solve/check scaffold improve short open-ended answers when the model cannot rely
on option letters, and does allowing 64 rather than 8 generated tokens change the result?

- **Contract:** `Answer the question with only the answer and no explanation.`
- **Scaffold:** `Solve the question carefully and check the result. Respond with only the answer and no explanation.`
- **Models:** Qwen2.5-3B-Instruct, SmolLM3-3B, and Phi-3.5-mini-instruct
- **Tasks:** 24 short exact-answer questions spanning arithmetic, logic, code, state tracking,
  and factual recall
- **Matrix:** three models x two prompts x two token caps x 24 tasks = 288 greedy generations
- **Decode:** temperature 0, sampling off, 8- or 64-generated-token cap
- **Runtime:** PyTorch 2.13.0+cu130, Transformers 5.15.0, CUDA 13.0, one L4-class accelerator
- **Native thinking:** disabled for SmolLM3 with `/no_think` and
  `enable_thinking=False`; not applicable to the other two models
- **Primary scoring:** bounded semantic answer scoring recorded in the artifact. Numeric tasks
  use the final standalone integer, text tasks require a boundary-delimited accepted answer,
  and the Go equality task requires `==`.
- **Exact scoring:** normalized full-output equality with an accepted answer, reported separately
  to distinguish task correctness from answer-only contract adherence.

## Results

The 8- and 64-token cells have identical correctness and exact-format counts for every model
and treatment. This table reports the 8-token cells; the artifact contains both caps.

| Model | Contract semantic | Scaffold semantic | Delta | Contract exact | Scaffold exact | Contract tokens | Scaffold tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | 21/24 | 21/24 | 0 | 21/24 | 21/24 | 56 | 55 |
| SmolLM3-3B | 23/24 | 22/24 | -1 | 17/24 | 18/24 | 73 | 71 |
| Phi-3.5-mini-instruct | 23/24 | 24/24 | +1 | 23/24 | 24/24 | 72 | 72 |
| **Pooled** | **67/72** | **67/72** | **0** | **61/72** | **63/72** | **201** | **198** |

Median generation latency at the 8-token cap was essentially unchanged by the scaffold:

| Model | Contract | Scaffold |
|---|---:|---:|
| Qwen2.5-3B-Instruct | 86.201 ms | 85.627 ms |
| SmolLM3-3B | 84.130 ms | 83.802 ms |
| Phi-3.5-mini-instruct | 124.677 ms | 125.016 ms |

No output reached the 8-token cap. For every model, prompt, and task, the generated text in the
8-token cell is byte-identical to its 64-token counterpart (288/288 paired generations when
both cap cells are counted). The larger budget therefore added no output, correctness, or
format value on this workload.

## What changed and what did not

1. **The scaffold is not a model-independent accuracy intervention.** Its semantic delta is
   zero, minus one, and plus one across the three families. Pooling does not rescue a gain.
2. **Option-letter artifacts do not explain the earlier tie.** The same heterogeneous, net-zero
   result appears when models must generate answers such as `Tuesday`, `==`, `queue`, or an
   integer instead of selecting A-D.
3. **Format behavior remains model-specific.** SmolLM3 often adds short wrappers such as
   `Answer: 3` or a sentence, so its semantic score exceeds exact answer-only adherence. The
   scaffold improves its exact count by one but lowers semantic correctness by one.
4. **The cap can be routed from observed termination, not assumed reasoning needs.** On this
   concise answer-only cell, eight tokens reproduce every 64-token output. A router can safely
   use the smaller cap only when the task and output contract match this bounded cell.
5. **Prompt identity remains part of the key.** These conclusions apply to the two exact strings
   above. Prior paraphrase studies already show that a semantically similar scaffold can behave
   differently.

## Reproduction and artifact

Raw artifact:

- [`2026-08-16-open-ended-scaffold-ablation.json`](../benchmarks/model-sensitivity/2026-08-16-open-ended-scaffold-ablation.json)
- SHA-256: `48f99842b1970acec73ab747a0c5c3580c578f58b0bdd0a110af46f841a422d3`
- Run window: `2026-08-15T22:44:30Z` to `2026-08-15T22:46:19Z`

Captured weight-set SHA-256 values:

- Qwen2.5-3B-Instruct: `8229c6ddee3ba5c33ce55182898c0a5c6cadf98086cea8c71b3f1f01f346fcae`
- SmolLM3-3B: `a23fcc3bd5d5908c861e9f985327ddd3277edbe5c659cd2b0ff2017a1e372dcd`
- Phi-3.5-mini-instruct: `12be042c8285552d8b4617959a43f66e259bbb84ff27b7469ffe89b6cb27deda`

The artifact captures every task and accepted answer, exact prompts, full generated text,
semantic and exact decisions, output tokens, latency, model configuration, runtime versions,
weight files, and digests. A conforming verifier must establish:

1. three models, four arms per model, 24 rows per arm, and 288 rows total;
2. each arm contains every task ID exactly once;
3. semantic and exact decisions recompute under the recorded bounded rules;
4. correctness, exact-format, output-token, and median-latency summaries recompute from rows;
5. each 8-token output equals its corresponding 64-token output;
6. artifact and weight-set digests match the values above.

## Limits

- One model artifact per family, one accelerator/runtime build, one greedy run, and fixed model
  and arm order; no randomized order or independent repeat.
- The battery contains short, closed-form answers with an answer-only contract. It is not a
  substitute for open-generation, tool-use, coding, or multi-turn evaluation.
- Bounded semantic rules are transparent but narrow; they are not an LLM judge and do not
  credit arbitrary paraphrases.
- Latency is sequential generation latency, not concurrent serving throughput. Token counts are
  workload-local and are not converted into provider cost.
- The 8-token result must not be generalized to prompts that request explanations or expose
  reasoning; those workloads have a different output contract.

