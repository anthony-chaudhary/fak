# fak + CrewAI manager-worker crew (hierarchical pattern)

This example covers the [CrewAI](https://docs.crewai.com) **manager-worker** pattern —
CrewAI's *hierarchical* process, where a manager agent decomposes a goal and delegates
subtasks to worker agents — and what `fak` adds to it:

1. **Governance.** Every tool call a worker proposes passes the capability floor in
   [`policy.json`](policy.json) before it runs. A refusal is a verdict, not a crash
   (deny-as-value).
2. **Coordination-overhead reduction.** The manager re-sends a shared crew brief on
   every delegation; fak prefills that brief once and reuses it across delegations
   instead of re-prefilling it each time.

It is the manager-worker companion to the generic CrewAI recipe in
[`docs/fak/agent-framework-integration.md` → CrewAI](../../docs/fak/agent-framework-integration.md#crewai),
which covers the one-line base-URL repoint and the `BaseTool` wrapper. This directory
adds the **hierarchical** specialization: the example crew, the manager-role reduction,
and the performance model.

## Run it

The proof is **dependency-free** — it does not import `crewai`, does not call a model,
and needs no key, network, or GPU. From the repository root:

```bash
python examples/crewai-crew/crew_demo.py        # -> ends with "summary: PASS"
go run ./cmd/fak policy --check examples/crewai-crew/policy.json   # the crew's floor is valid
```

Captured output is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md). The verdicts the demo
prints are exactly what the kernel returns — cross-check any line:

```bash
go run ./cmd/fak preflight --policy examples/crewai-crew/policy.json --tool send_email --args '{}'
# verdict=DENY reason=POLICY_BLOCK by=monitor   (the Writer's blocked call)
go run ./cmd/fak preflight --policy examples/crewai-crew/policy.json --tool read_dataset --args '{}'
# verdict=ALLOW reason=NONE by=monitor          (the Analyst's read_ tool)
```

To run the same adjudication through a live gateway instead of the local evaluator,
start a `fak serve` and set `FAK_BASE_URL`:

```bash
FAK_BASE_URL=http://127.0.0.1:8080 python examples/crewai-crew/crew_demo.py
```

The printed verdicts are identical either way — the local evaluator mirrors the kernel's
name-level allow-list decision (explicit deny → exact allow → prefix allow → fail-closed
`DEFAULT_DENY`).

## Manager-role coordination reduction

In CrewAI's hierarchical process the manager agent is the coordination hot spot. Every
time it delegates a subtask it sends the worker a prompt that carries the **shared crew
brief** — the overall goal, the worker roster, the tool schemas, and the task context
accumulated so far — plus a small worker-specific instruction. The shared brief is the
same bytes on every delegation; only the worker instruction changes.

Without a cross-agent shared prefix, that brief is re-prefilled on **every** delegation:
the coordination cost grows with the number of delegations, and the manager dominates the
crew's token bill. fak removes the repeated work by treating the shared brief as a
kernel KV-cache object — prefilled once, spliced on the original bytes (a memcpy, never a
re-marshal), and reused across every delegation so the provider's prompt-cache prefix
stays byte-identical. Only the per-delegation unique instruction is new prefill.

This is the same lever fak applies to any agent fleet (do shared prefill once; later
calls read it for free), specialized to the one agent — the manager — that re-sends the
most shared context. It does **not** change CrewAI's topology: the manager still
delegates, workers still run their own tools, and hand-offs are unchanged. fak only
governs *which* worker tool calls run and prefills the shared brief once.

## Performance comparison

`crew_demo.py` Part 2 computes the coordination-overhead model deterministically. For a
crew with a 2000-token shared brief, a 300-token per-delegation instruction, and 12
manager→worker delegations (3 workers × ~4 rounds):

| | prefill tokens | how |
|---|---|---|
| Naive (brief re-sent each delegation) | `12 × (2000 + 300)` = **27,600** | the manager re-prefills the shared brief 12 times |
| fak (brief prefilled once, reused) | `2000 + 12 × 300` = **5,600** | the shared brief is a reused KV-cache object |
| **Modeled prefill reduction** | **4.93×** | on the manager's shared coordination prefix |

This is **deterministic token accounting** — the manager-worker specialization of fak's
fan-out geometry — **not** a measured wall-clock, and it is honest about that. The two
authoritative paths:

- **Host-runnable deterministic ladder:** `go run ./cmd/fanbench --grid canonical` emits
  the D-001 coordination-overhead-vs-N ladder (N = 1, 100, 500, 1000) as in-process
  kernel arithmetic. Every published number is governed by
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
- **Live-model wall-clock (deferred):** the CrewAI-vs-native end-to-end wall-clock
  comparison is the bench-node run tracked by **D-001 (#255)** — it needs CrewAI
  installed and a live model on a bench node, which is out of scope for this
  dependency-free example. No wall-clock figure is asserted here.

## Wiring a real CrewAI crew

You do not reimplement any of the demo's plumbing in a real crew. CrewAI drives models
through LiteLLM, whose `LLM` wrapper takes a custom `base_url`; point it at `fak serve`
and run the hierarchical process as usual.

```python
from crewai import LLM, Agent, Crew, Task, Process

# 1. Repoint every model the crew uses at the fak gateway (Mode A — one line).
#    Prefix the model id with its provider (the LiteLLM convention); you can also set
#    OPENAI_API_BASE=http://127.0.0.1:8080/v1 in the environment instead of the kwarg.
fak_llm = LLM(model="openai/gpt-4o", base_url="http://127.0.0.1:8080/v1", api_key="fak-local")

researcher = Agent(role="Researcher", goal="Gather sources", backstory="...", llm=fak_llm)
analyst    = Agent(role="Analyst",    goal="Analyze findings", backstory="...", llm=fak_llm)
writer     = Agent(role="Writer",     goal="Draft the report", backstory="...", llm=fak_llm)

research = Task(description="Research the topic", expected_output="notes", agent=researcher)
report   = Task(description="Write the report",  expected_output="report", agent=writer)

# 2. The hierarchical process gives the manager-worker pattern. The manager_llm is
#    ALSO routed through fak, so the coordination prompts hit the shared-brief cache.
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research, report],
    process=Process.hierarchical,
    manager_llm=fak_llm,
)
result = crew.kickoff()
```

For tool-site governance (Mode B), subclass `BaseTool` and route `_run` through the
shared `guarded(...)` helper from the integration guide so a poisoned result is
quarantined before it enters the crew's shared context:

```python
from crewai.tools import BaseTool

class FetchTool(BaseTool):
    name: str = "fetch_url"
    description: str = "Fetch the text at a URL."

    def _run(self, url: str) -> str:
        return guarded("fetch_url", _http_get)(url=url)   # guarded() from the integration guide
```

The tool **names** in [`policy.json`](policy.json) must match the names your crew
registers (CrewAI's delegation tools and your `BaseTool.name` values) — verify against
your installed CrewAI version, as the integration guide notes. Anything not on the
allow-list hits the fail-closed `DEFAULT_DENY`.

## What this does not claim

- It does **not** run a live CrewAI crew, LiteLLM, or any model — it proves the verdict
  boundary and the coordination-overhead model with a dependency-free adapter. The final
  binding belongs in an app pinned to the CrewAI version it ships with.
- The 4.93× figure is a **modeled** token reduction at one illustrative geometry, not a
  measured wall-clock. The live-model comparison is the deferred D-001 (#255) bench-node
  run.
- The floor bounds **which tools** run, by tool *name* — it does not filter the
  *arguments* of an allow-listed tool (argument predicates are a separate, partial
  surface; see [`POLICY.md`](../../POLICY.md)).

## Sources

- CrewAI hierarchical process and `LLM` configuration:
  <https://docs.crewai.com/concepts/processes>, <https://docs.crewai.com/concepts/llms>.
- Generic CrewAI recipe (Mode A repoint + `BaseTool` wrapper):
  [`docs/fak/agent-framework-integration.md` → CrewAI](../../docs/fak/agent-framework-integration.md#crewai).
- The coordination-overhead geometry and published numbers:
  [`cmd/fanbench`](../../cmd/fanbench/main.go) (`--grid canonical`) ·
  [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) ·
  [`docs/explainers/compounding-benefits-of-a-saved-call.md`](../../docs/explainers/compounding-benefits-of-a-saved-call.md).
- Track-D parity status (where this child sits):
  [`docs/notes/track-d-agent-framework-parity-tracking-304.md`](../../docs/notes/track-d-agent-framework-parity-tracking-304.md).
