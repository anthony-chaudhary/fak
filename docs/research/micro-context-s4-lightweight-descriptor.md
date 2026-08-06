# Micro-context S4a: harness assumptions and lightweight descriptor

**Status:** observed local contract/bring-up witness, 2026-08-06. This is the harness-assumption child #5789, not the tool/effect-safety S4 child #5791.

`fak-micro-context-descriptor/1` captures only the semantics needed to schedule one logical context:

```json
{
  "schema": "fak-micro-context-descriptor/1",
  "id": "d-0001",
  "base_id": "immutable-agent-base-v1",
  "task_delta": "task-0001",
  "capability_set": ["read_record"],
  "budget": {"max_turns": 1, "max_output_tokens": 8},
  "continuation": [],
  "output_contract": {"kind": "exact", "expected": "DONE"}
}
```

The adapter uses the existing `microagent.Host` and `Gateway`: it expands a registered immutable base plus continuation and task delta into one model turn, advertises only the named tool capabilities, applies the output-token budget, and refuses an output outside the contract. It does not create a second runtime.

## Assumption inventory

| Assumption | Classification | Carrier / disposition |
|---|---|---|
| model endpoint and shared instructions | semantic | `base_id`, resolved once by adapter |
| task input | semantic | `task_delta` |
| tool authority | semantic | explicit `capability_set` |
| bounded work | semantic | `budget` (v1 is exactly one turn) |
| prior state | semantic | `continuation` |
| accepted result | semantic | `output_contract` |
| one OS process per context | session convenience | omitted |
| terminal/TUI per context | session convenience | omitted |
| cwd | adapter concern | capability/resource-lease adapter |
| ambient credentials | unsafe convenience | omitted; explicit capability adapter |
| interactive approval UI | adapter concern | policy/effect adapter |

This is where Ultracode-style decomposition helps: independently specified deltas and output contracts are useful. Its concurrent orchestration is prior art only; it is not inference/cache evidence and is not embedded as another scheduler.

## Reproduce

```powershell
go run ./cmd/microcontextdemo `
  -descriptor-bench experiments/microcontext/s4-local-descriptor-benchmark-2026-08-06.json `
  -contexts 1000
go run ./cmd/microcontextdemo `
  -verify-descriptor-bench experiments/microcontext/s4-local-descriptor-benchmark-2026-08-06.json
```

## Observed envelope

The checked-in run executed 1,000 descriptors through one existing 16-worker `microagent.Host` and one deterministic gateway fixture:

- one immutable base installation;
- 1,000/1,000 gateway calls;
- 210 serialized descriptor bytes/context;
- about 2.2 MiB sampled Go allocation delta after drain;
- local version-command process startup p50: fak 210.9 ms, Claude Code 145.1 ms, Codex 115.4 ms.

The CLI numbers are deliberately narrow process/version probes. They do **not** measure full harness task completion, child-process RSS, prompt tokens, model quality, or Ultracode orchestration. The in-process fixture is likewise not model TTFT or throughput. A fair full-task corpus and per-process RSS still require harness-specific noninteractive adapters; the current witness names that limitation rather than manufacturing a ratio.
