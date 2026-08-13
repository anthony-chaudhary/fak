# fak microagent paired corpus

- Schema: `fak-micro-corpus/1`
- Corpus: `popular-agent-subtasks-v1`
- Execution: **PASS**
- Value: **NOT_YET**
- Reason: paired corpus execution and grounded retry contribution are measured, but gateway dollars and context/verify/mode ablations are not yet available; no quality/$ winner is claimed

| Task | Complexity | Micro | Managed baseline | Micro tokens | Baseline tokens | Micro ms | Baseline ms |
|---|---|---:|---:|---:|---:|---:|---:|
| instruction | one-step | true | true | 35 | 18760 | 2043 | 21682 |
| extract | structured | true | true | 54 | 8380 | 4143 | 29296 |
| policy | reasoning | true | true | 61 | 8399 | 1343 | 15076 |

## Ablation readiness

| Layer | Status | Reason |
|---|---|---|
| retry | PASS | retry-off failed after one attempt; retry-on completed after exact transient evidence was fed back, bounded at two attempts |
| context | NOT_YET | pinned tasks do not cross the context compaction threshold |
| verify | NOT_YET | exact-answer scoring is external and does not exercise the microagent Verifier hook |
| mode | NOT_YET | the real gateway microagent currently exposes completion mode only; #2026 owns bash/tool mode parity |

### Retry witness

- retry off: completed=false, attempts=1
- retry on: completed=true, attempts=2
- evidence re-fed verbatim: `fixture transient: upstream reset`
