# Memory Query Demo

`memqdemo` is a no-model, deterministic walkthrough of the memory-operation algebra. It shows an agent composing retrieve, clean, compact, and a novel query over one store without the kernel hardcoding a single memory strategy.

## Prerequisites

Requires Go only. It runs offline over an in-memory corpus and does not need a model, database, network, or API key.

## Quick Start

From the repo root:

```bash
go run ./cmd/memqdemo
go run ./cmd/memqdemo -report memqdemo-report.json
```

The normal run completes in a few seconds, writes nothing, and returns exit code 0 only if the sealed poison never renders into context. The optional `-report` path writes a deterministic JSON summary when you explicitly ask for one.

## What You See

```text
memq: an agent authoring its own memory strategy (build SQL, not a specific query)
== 3. EXPLAIN compact (the plan, before anything runs) ==
== 7. a NOVEL query the agent authored (no built-in driver expresses it) ==
== 8. the safety floor (render EVERYTHING - the sealed poison is refused) ==
witness: an agent composed retrieve/clean/compact AND a novel query over one algebra
```

## What This Does Not Claim

This demo does not claim a production memory database, a learned compaction policy, or integration with a live agent session. It proves the query algebra, fail-closed effects, and sealed-span refusal over a fixed in-memory store.

## Related Docs

- [Agent memory integration](../../docs/integrations/agent-memory.md)
- [Context is not memory](../../docs/CONTEXT-IS-NOT-MEMORY.md)
