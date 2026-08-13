# fak microagent paired corpus

- Schema: `fak-micro-corpus/1`
- Corpus: `popular-agent-subtasks-v1`
- Execution: **PASS**
- Value: **NOT_YET**
- Reason: paired corpus execution plus retry, bounded history, verification, and action-mode contributions are measured, but gateway dollars remain unavailable; no quality/$ winner is claimed

| Task | Complexity | Micro | Managed baseline | Micro tokens | Baseline tokens | Micro ms | Baseline ms |
|---|---|---:|---:|---:|---:|---:|---:|
| instruction | one-step | true | true | 35 | 18760 | 2043 | 21682 |
| extract | structured | true | true | 54 | 8380 | 4143 | 29296 |
| policy | reasoning | true | true | 61 | 8399 | 1343 | 15076 |

## Ablation readiness

| Layer | Status | Reason |
|---|---|---|
| retry | PASS | retry-off failed after one attempt; retry-on completed after exact transient evidence was fed back, bounded at two attempts |
| context | PASS | naive FIFO lost an early durable pointer; managed compaction retained it while peak context stayed within the same cap |
| verify | PASS | verifier-off accepted claimed completion; verifier-on independently read back the absent artifact and refused it |
| mode | PASS | the same structured extraction task completed through portable string action and provider-native typed-tool action modes |

### Retry witness

- retry off: completed=false, attempts=1
- retry on: completed=true, attempts=2
- evidence re-fed verbatim: `fixture transient: upstream reset`

### Context witness

- same cap: 64 tokens across 24 long-history turns
- durable pointer retained: naive=false, compacted=true
- managed compactions=20, peak tokens=64, final tokens=57

### Mode witness

- same extraction task correct: string=true, typed-tool=true
- fixture-reported tokens: string=23, typed-tool=19
- token delta is fixture-only interface evidence, not a provider performance claim

### Verifier witness

- verifier off: claimed completion accepted=true
- verifier on: claimed completion accepted=false, caught=true
- independent readback: `artifact-absent`; evidence: `artifact-absent: independent readback found no claimed artifact`
