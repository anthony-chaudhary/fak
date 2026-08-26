# Indexed literal code-search alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6184](https://github.com/anthony-chaudhary/fak/issues/6184) tracks real search-system runs and independent resource/cost witnesses.

## Capability and workload

`internal/trigram.Index` builds distinct-rune trigram postings and verifies candidate lines exactly; it also falls back safely for short queries. This contract covers indexed literal search only. Regex candidate extraction and lexical similarity remain separate capability debt.

Every arm loads/indexes the same six Go files and executes four queries: a literal in three files, a short literal, a Unicode literal, and an absent literal. Correctness requires all exact ordered paths and line numbers. Build/load and query phases are measured separately.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native trigram indexed literal search | native | available |
| optimized in-memory linear scan | tuned no-index baseline | available |
| ripgrep | external | unavailable |
| git grep | external | unavailable |
| Zoekt | external | unavailable |
| livegrep | external | unavailable |
| Sourcegraph Search | external | unavailable |

No equivalent first-class fak integration was found. A future equivalent must appear as its own `fak + integration` arm. External rows remain measurement-zero until real processes/services run the identical corpus; wrappers and reimplementations are not witnesses.

## Completion evidence

Complete arms report exact queries, false positives/negatives, location errors, build/load and query latency, throughput, CPU/RSS, corpus/index/storage/network bytes, setup/operator time, and total cost. Versions, index configuration, commands, raw output, and independent read-back must be pinned.

`TestCompareLocalKeepsIndexedSearchAlternativesExplicit` locks inventory, native/baseline correctness, and unavailable zeros. `BenchmarkBuildAndQueryLiteralCorpus` rebuilds the real index and runs all four queries per iteration. Local timing is not a cross-system claim; no system is ranked until #6184 has complete real runs.
