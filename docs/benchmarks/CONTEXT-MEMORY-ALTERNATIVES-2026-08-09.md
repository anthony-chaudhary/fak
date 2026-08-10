# Context-memory native-vs-alternative comparison witness

Date: 2026-08-09  
Capability: `context_memory_management`  
Status: **INCOMPLETE**

## Arms

- `fak native`: ctxmmu context paging, quarantine, tool-page residency, and durable-memory write admission.
- tuned no-feature baseline: retain full history without a memory-management layer.
- strongest practical external alternative: standalone Letta.
- equivalent first-class integration contracts, each measured separately: `fak + mem0`, `fak + Letta`, `fak + Zep/Graphiti`, and `fak + LangMem/LangGraph memory`.

The integration list comes from `docs/integrations/agent-memory.md`. Documentation of a safe write/read seam is not performance evidence; each system remains a separate unavailable arm until executed.

## Same-workload local runner

`ctxmmu.CompareLocal` emits `fak-context-memory-comparison/1`. Its bounded local corpus sends identical candidate writes through the native memory-write adjudicator and full-history baseline. It covers an explicit durable fact, an oversized entry, and a secret-shaped entry. Native structural write classification is 3/3 on this fixture; the retain-everything baseline admits only the legitimate fact correctly and therefore scores 1/3. These values are **write-admission fixture accuracy**, not long-horizon task success or retained-fact recall.

Letta and all four `fak + integration` arms stay `available=false`; mocks and adapter registration cannot satisfy them.

## Local overhead witness

Host: Windows amd64, AMD Ryzen 9 9950X. Command:

```text
go test ./internal/ctxmmu -run '^$' -bench '^BenchmarkContextMemoryComparison$' -benchmem -benchtime=10000x -count=5
```

Median across five runs:

| Arm | ns / three writes | Approx. ns / write | B/op | allocs/op |
|---|---:|---:|---:|---:|
| fak-native memory-write gate | 1,267,479 | 422,493 | 1,339 | 22 |

The oversized body intentionally exercises the configured 16 KiB bound, so this is not a small-message fast-path benchmark. It excludes model calls, store I/O, retrieval, consolidation, and read-back.

## Missing live witness

Completion requires one frozen long-horizon workload through all seven arms using the same model, context/token budget, planted facts, distractors, read-back questions, concurrency, and independent grader. Capture task success, memory-write precision, retained-fact recall, input tokens, p50/p95 latency, peak RSS or equivalent resources, and total cost. Version every service and configuration; preserve temporal semantics for Zep/Graphiti rather than reducing it to timeless key/value recall.

## Honest verdict

No net-true memory-system winner is established. The local runner proves bounded structural write behavior only. It neither prices the native read/page path nor demonstrates external retrieval quality.
