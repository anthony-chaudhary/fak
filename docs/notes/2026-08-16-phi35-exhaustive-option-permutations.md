# Phi-3.5 exhaustive option-permutation scaffold study (2026-08-16)

Issue: #6692

## Verdict

On Phi-3.5-mini-instruct, the exact solve/check scaffold and concise contract are a
correctness tie across all 24 option permutations: 491/576 versus 486/576, with scaffold
wins/ties/losses of 9/7/8. The five-observation aggregate difference is too small and too
sign-mixed to justify correctness routing. The scaffold does, however, turn an extremely
verbose default into near-perfect answer-only output: 574/576 strict responses and 1,156
output tokens versus 7/576 and 19,216. For this exact model, prompt, runtime, and answer-letter
task class, route the scaffold for bounded output/latency, not for a claimed correctness gain.

This third-family replication reinforces the cross-family boundary: exact prompt identity is
part of the treatment, and a scaffold that changes output behavior dramatically need not
change task accuracy materially.

## Question and treatment

Does the exact scaffold survive every ordering of four answer options on a third small
instruction-model family?

- **Contract:** `Choose the correct option. Return exactly one capital letter: A, B, C, or D.`
- **Scaffold:** `Solve the question carefully, check the result against the options, then return exactly one capital letter: A, B, C, or D. Do not output reasoning.`
- **Model:** `microsoft/Phi-3.5-mini-instruct` (3.8B, two safetensors shards)
- **Matrix:** 24 fixed tasks x all `4! = 24` option permutations x two treatments = 1,152 greedy generations
- **Decode:** temperature 0, sampling off, 64 generated-token cap
- **Runtime:** PyTorch 2.13.0+cu130, Transformers 5.15.0, CUDA 13.0, one L4-class accelerator
- **Thinking control:** not applicable; Phi-3.5-mini-instruct has no native thinking mode
- **Scoring:** first explicit answer-letter phrase, with the expected letter remapped after each permutation

The tasks, complete prompts, outputs, expected/predicted letters, option maps, latency, token
counts, model config, runtime versions, weight digest, and per-arm summaries are captured in
the raw artifact.

## Results

| Treatment | Correct | Mean / 24 | Median | Range | Strict answer-only | Output tokens | Median of arm median latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| concise contract | 486/576 | 20.25 | 21.0 | 17-23 | 7/576 | 19,216 | 1,353.961 ms |
| exact scaffold | 491/576 | 20.46 | 20.5 | 17-23 | 574/576 | 1,156 | 90.679 ms |

Paired by permutation, scaffold minus contract:

- wins: 9
- ties: 7
- losses: 8
- mean delta: +0.208/24
- median delta: 0
- range: -2 to +3

Paired by task and permutation (576 pairs):

- both correct: 469
- both wrong: 68
- contract only: 17
- scaffold only: 22

The scaffold reduced generated output by 94.0% (18,060 fewer tokens), while adding five
correct observations. That is a witnessed output/format-efficiency result for this exact
cell, not a general throughput claim: arm execution was sequential and latency includes only
generation, so no concurrency or end-to-end serving conclusion follows.

## Interpretation

1. **No correctness winner.** Aggregate accuracy differs by 0.87 percentage points, while
   nearly as many permutations favor the contract as the scaffold. Routing either treatment
   for correctness would overread this matrix.
2. **A large output-control effect is real.** The concise contract often elicited explanations
   despite asking for one letter; the scaffold's explicit `Do not output reasoning` produced
   574 strict responses and used about 6% as many output tokens.
3. **The cross-family pattern remains heterogeneous.** The exhaustive Qwen2.5-1.5B cell favored
   this scaffold strongly, Qwen2.5-3B and SmolLM3-3B were correctness ties, and Phi-3.5 is also
   a tie. Model identity remains part of the routing key.
4. **Scope stays narrow.** These are short four-option tasks under greedy decoding. The study
   does not establish benefits for open-ended work, tool use, alternate scaffold wording,
   sampling, other token budgets, or other Phi releases.

## Reproduction and artifact

Raw artifact:

- [`2026-08-16-phi35-option-permutations.json`](../benchmarks/model-sensitivity/2026-08-16-phi35-option-permutations.json)
- SHA-256: `2c309e062e1860e8af2bfae186e3accb342978855ba85a448f2cda5c13000e21`
- Captured weight-set SHA-256: `b346492e5d2c707a4f6e9dbac2a9967af799206d9ebaff9d8d3c89bac5616232`
- Run window: `2026-08-15T22:09:08Z` to `2026-08-15T22:21:54Z`

The captured artifact is the reproduction authority. A conforming rerun loads the recorded
model revision from local files, applies the recorded chat template to every captured prompt,
runs greedy decoding with the recorded 64-token cap, and recomputes expected letters from the
stored permutations. Before interpreting a rerun, verify:

1. 48 arms, 24 rows per arm, and 1,152 rows total;
2. each treatment covers all 24 unique permutations exactly once;
3. each row's expected letter maps to the original correct option after permutation;
4. row correctness, strict-format, output-token, and latency summaries recompute from rows;
5. the artifact and weight-set digests match the values above.

## Limits

- One model artifact, accelerator class, runtime build, decode budget, and sequential arm order.
- No randomized treatment order, repeated stochastic seeds, confidence interval, or independent
  rerun; greedy decoding removes sampling noise but not runtime/order effects.
- The answer-letter parser accepts the first explicit answer phrase; strict format is reported
  separately.
- Output-token and generation-latency reductions are workload-local and are not priced as a
  provider-cost or serving-throughput gain.
