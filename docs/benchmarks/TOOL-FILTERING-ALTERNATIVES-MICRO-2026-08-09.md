# Tool-filtering native-vs-alternative microbenchmark witness

Date: 2026-08-09  
Host class: Windows amd64, AMD Ryzen 9 9950X 16-Core Processor  
Command:

```text
go test ./internal/gateway -run '^$' -bench '^BenchmarkToolFilteringAlternatives/tools=64' -benchtime=10x -count=1 -benchmem
```

Correctness gate:

```text
go test ./internal/gateway -run TestToolFilteringAlternativesRetainRequiredTool -count=1
```

The fixture holds one required customer-record search tool in a 64-tool catalog and requires both retrieval arms to retain it. This is a **microbenchmark only**: it measures local selector cost and allocation, not end-to-end task quality, provider token count, TTFT, or total cost. It therefore does not complete the ToolRAG-class comparison contract in `internal/nativebench` and must not be cited as a net-true product win.

| Arm | ns/op | B/op | allocs/op | Interpretation |
|---|---:|---:|---:|---|
| fak native hybrid ranker | 65,553,250 | 19,719,196 | 40,448 | current native selector rebuilds/queries the selfquery catalog |
| tuned all-schemas baseline | 30 | 0 | 0 | local pass-through only; provider-side token/TTFT tax is outside this microbenchmark |
| incumbent substring retrieval | 3,620 | 16 | 1 | local lexical baseline; weaker than the required next-best ToolRAG-class arm |

## Honest verdict

The native selector is dramatically more expensive locally than the incumbent lexical filter on this fixture. Whether it is net-true depends on end-to-end recall, task success, input-token savings, provider latency, and total cost. The next benchmark must run the same real tool-use corpus through native hybrid ranking, all schemas with provider caching, and the strongest practical ToolRAG-class implementation; until then the contract remains `not yet`.
