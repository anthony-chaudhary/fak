# Issue near-duplicate detection alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6228](https://github.com/anthony-chaudhary/fak/issues/6228) tracks real hosted and embedding-system witnesses.

## Same workload

`internal/issuededup.Index.Check` classifies two paraphrased duplicates and one unrelated candidate against the same three-issue backlog, returning the exact existing issue pointer. Correctness requires precision, recall, and pointer accuracy; backlog census clustering remains separate debt.

| Arm | Class | Status |
|---|---|---:|
| fak native near-duplicate gate | native | available |
| normalized exact-title equality | tuned baseline | available; misses paraphrases |
| fak + GitHub issue search | integration | unavailable |
| GitHub duplicate detection | external | unavailable |
| Linear duplicate detection | external | unavailable |
| Jira similar requests | external | unavailable |
| sentence-transformer cosine retrieval | external | unavailable |

Complete witnesses pin versions/data/configuration and report precision, recall, false decisions, pointer accuracy, latency/throughput, CPU/RSS, bytes/network/storage, setup/operator time, and total cost. `TestCompareLocalKeepsIssueDedupAlternativesExplicit` locks inventory and unavailable zeros; `BenchmarkIssueNearDuplicateGate` executes the real index. Three local Windows/amd64 samples were 55,198, 110,657, and 75,229 ns/op (median 75,229 ns/op; about 47,441 B/op; 987 allocs/op). This is not a cross-system ranking; no external ranking exists yet.
