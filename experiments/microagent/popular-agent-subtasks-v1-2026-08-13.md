# fak microagent paired corpus

- Schema: `fak-micro-corpus/1`
- Corpus: `popular-agent-subtasks-v1`
- Execution: **PASS**
- Value: **NOT_YET**
- Reason: paired corpus execution is measured, but gateway dollars and retry/context/verify/mode ablations are not yet available; no quality/$ winner is claimed

| Task | Complexity | Micro | Managed baseline | Micro tokens | Baseline tokens | Micro ms | Baseline ms |
|---|---|---:|---:|---:|---:|---:|---:|
| instruction | one-step | true | true | 35 | 9228 | 1209 | 5133 |
| extract | structured | true | true | 54 | 218 | 4105 | 5040 |
| policy | reasoning | true | true | 61 | 237 | 4484 | 4924 |

## Ablation readiness

| Layer | Status | Reason |
|---|---|---|
| retry | NOT_YET | pinned tasks do not inject a witnessed transient failure |
| context | NOT_YET | pinned tasks do not cross the context compaction threshold |
| verify | NOT_YET | exact-answer scoring is external and does not exercise the microagent Verifier hook |
| mode | NOT_YET | the real gateway microagent currently exposes completion mode only; #2026 owns bash/tool mode parity |

## Provenance and interpretation

- Captured 2026-08-13 with a clean-archive binary.
- Microagent endpoint: sanctioned remote shared fak kernel, OpenAI-compatible `qwen2.5:14b`.
- Baseline: current Claude Code through `fak manage --probe`, model alias `sonnet` resolving to provider-reported `claude-sonnet-5`.
- Correctness is exact-answer scoring over the same prompt in both arms.
- Token and baseline dollar fields are provider-reported. Wall time is client-observed.
- Microagent dollars are unsupported by this endpoint, represented as `null`, not zero.
- This is a narrow three-task spine, not SWE-bench and not evidence of general coding-task parity.
