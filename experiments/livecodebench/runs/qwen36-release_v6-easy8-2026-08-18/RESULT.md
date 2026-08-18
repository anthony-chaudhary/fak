# Qwen3.6-27B release_v6 easy8 — official LiveCodeBench run

**Verdict:** official LiveCodeBench `pass@1 = 0.25` (2/8) for one deterministic sample per problem. This is a real-model result on a pinned, official `release_v6` subset; it is not a full-release leaderboard score and not a GLM-5.2 result.

## Pinned workload

- Dataset: `livecodebench/code_generation_lite`, Hugging Face snapshot `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.
- Source file: `test6.jsonl`, SHA-256 `bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5`.
- Deterministic selection: first eight shortest `easy` rows by source-row byte length, ties by `question_id`.
- IDs: `abc398_a`, `abc389_b`, `abc391_a`, `abc400_a`, `abc392_a`, `abc387_a`, `abc400_b`, `abc393_a`.
- Snapshot rows: `../../snapshots/release_v6-easy8.rows.json`, SHA-256 `88a67c588e3322ac710fa064ac3c8c03e6eac8207b9ec543184d668675c7c4b5`.
- Normalized suite: `../../snapshots/release_v6-easy8.suite.json`, SHA-256 `117ce32eba560a54b071aced7f58cf1ec23e5710d2ed9a89f96dc1725e2f2222`.

## Generation

- Compute: sanctioned GCP GPU node, NVIDIA A100 40 GB.
- Model/backend: `Qwen3.6-27B-Q4_K_M.gguf`, fak in-kernel CUDA backend.
- Binary SHA-256: `58f98662dbf7308c0d16b6041090e418d2169537c2bf9187f3e540e1d24d9910`.
- Config: `n=1`, `temperature=0`, `seed=42`, `concurrency=1`, `max_tokens=512`, trace `lcb-v6-easy8-20260818-d`.
- Exact report: `raw-report.json`, SHA-256 `ee69ca29de08031064e42d7206b36eb7f4b0d7cd6b38bb9ee4c6f30c16d55001`.
- Four generations ended at `finish_reason=length`; they remain in the denominator and were not repaired.

## Official grading

The exact saved generations were exported with `livecodebench export`, then processed with the official harness's `extract_code(..., LMStyle.OpenAIChat)` and graded by unmodified `codegen_metrics`.

- Harness: LiveCodeBench commit `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- Python: 3.11.15; Windows compatibility shim only supplies Unix `SIGALRM`/`alarm` names. The official evaluation code is unchanged; its outer multiprocessing timeout remains active.
- Export: `custom-output.json`, SHA-256 `19c79989846b906a9180b5c8b1baccf9c0d33ef7522d7f50663327beed111c9b`.
- Extracted exact code: `extracted-generations.json`, SHA-256 `5b51de7d2c95401303cb5d618790ea3e606e7b4eee56d1405a834505198d291e`.
- Official evaluator inputs (42–44 public+private tests/problem): `grader-samples.json`, SHA-256 `7a2b5caba523f88a8a9954523ceec0ccf04da79defa80396d331abfd75ca9e6c`.
- Authoritative output: `official-grading-result.json`, SHA-256 `ffa5488b846e1921f7f377468b942c56d94e83fc08808e265661a51d13e886f1`.
- Passed: `abc400_a`, `abc387_a`. Official `pass@1 = 0.25`.

This closes the model-generation → exact-export → official-evaluator spine. A same-workload fak comparison arm and the GLM-5.2 campaign remain separate work; this result must not be relabeled as either.


