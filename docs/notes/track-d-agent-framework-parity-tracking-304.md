---
title: "fak Track D agent-framework-parity tracker: the 8 children, honest on-disk status, and the gate that keeps the epic open"
description: "Umbrella tracker for epic #304 (Track D - Agent Framework Parity). Records the CORRECT child issue map (the epic body's links are stale internal-tracker ids), each child's honest shipped/scaffold/wire-supported/unimplemented status with deciding files, and the named gate — a deferred bench-node headline run + unbuilt adapters/features — that keeps the roll-up open. No number is invented."
---

# Track D — Agent Framework Parity (epic #304) — status tracker

> **Umbrella tracker for [#304](https://github.com/anthony-chaudhary/fak/issues/304)** —
> *"N=100+ benchmarks, LangChain/AutoGen/CrewAI integration, workflow orchestration."*
>
> **#304 is a pure roll-up: it closes only when all 8 children close.** This doc does not
> claim the epic is done — it records the **honest on-disk state** of each child so an
> operator (or the next agent) can see exactly what shipped, what is wire-supported but
> acceptance-open, what is a scaffold tracked by a sibling epic, and what is not built at
> all. **House rule: every number comes from a real run.** The one benchmark child's
> *published* table is deferred to a bench node and says so plainly; no token/throughput
> figure is invented here. Written **2026-06-25** on a win32 dev box (no live-model bench
> node, no LangGraph/AutoGen/CrewAI install reachable from this host).

---

## 0. The child map — the epic's own links are stale (read this first)

The migrated epic body (#304) lists its children as **#314, #316, #319, #321, #324, #326,
#328, #331**. Those numbers are **wrong** — they are pre-migration internal-tracker ids
that GitHub renumbered on import, and every one now points at an unrelated issue (e.g.
#314 is a *closed* `examples: auth-hardening` walkthrough, #328 is a *closed*
`examples: operator observability` walkthrough, #326 is a `serve(gateway)` error-envelope
issue). The epic's note *"cross-references are auto-maintained"* is therefore **not
holding** for this epic. The **real** children — the issues actually tagged
`track/D-agent-framework` with a `D-00x` slug — are:

| Slug | Live issue | Priority | Title | Epic body shows (WRONG) |
|---|---|---|---|---|
| **D-001** | [#255](https://github.com/anthony-chaudhary/fak/issues/255) | P0 | N=100-1000 Agent Benchmark | #314 |
| **D-002** | [#253](https://github.com/anthony-chaudhary/fak/issues/253) | P1 | LangChain/LangGraph Integration *(CLOSED)* | #316 |
| **D-003** | [#250](https://github.com/anthony-chaudhary/fak/issues/250) | P1 | CrewAI Pattern Support | #319 |
| **D-004** | [#248](https://github.com/anthony-chaudhary/fak/issues/248) | P1 | AutoGen Integration | #321 |
| **D-005** | [#245](https://github.com/anthony-chaudhary/fak/issues/245) | P2 | Workflow Orchestration Layer | #324 |
| **D-006** | [#243](https://github.com/anthony-chaudhary/fak/issues/243) | P2 | Tool Ecosystem Expansion | #326 |
| **D-007** | [#241](https://github.com/anthony-chaudhary/fak/issues/241) | P2 | Multi-Agent Coordination Protocol | #328 |
| **D-008** | [#238](https://github.com/anthony-chaudhary/fak/issues/238) | P2 | Agent Testing Framework | #331 |

> The children's own migrated bodies still cite the **internal** epic id *"Epic #264"*; the
> live GitHub umbrella is **#304**. Both refer to the same Track D.

This doc keys everything off the **live** numbers above.

> **Operator action (the one GitHub-side fix):** for #304's *"closes when all children
> close"* mechanism to actually fire, repoint the epic body's checkbox list from the stale
> ids to the live ones — `#314→#255, #316→#253, #319→#250, #321→#248, #324→#245,
> #326→#243, #328→#241, #331→#238`. Until then the epic tracks eight unrelated (and partly
> closed) issues and will never auto-close from its children. This in-repo tracker is the
> authoritative map in the meantime.

---

## 1. Honest status — each child, verified against the working tree (2026-06-25)

Status is read from the tree + `git log`, not from any worker's say-so. "Scaffold" means a
host-correct, test-witnessed implementation whose *published acceptance* is a run deferred
to a bench node. "Wire-supported" means the framework already reaches fak by repointing its
OpenAI base URL (the kernel's intended integration shape), but the issue's *bespoke
adapter / example / benchmark* acceptance is not yet built.

| Slug · issue | On-disk state | What is real today | Deciding file(s) |
|---|---|---|---|
| **D-001** #255 benchmark (P0) | 🟡 **Scaffold shipped; published run bench-node-gated** | A host-runnable N=1/100/500/1000 fan-out scale harness (**N≥1024 capable**) that computes coordination-overhead-vs-N=1 and cross-agent-reuse-uplift **deterministically** (turnbench is in-process kernel arithmetic, not model calls), wrapping the already-witnessed fan-out engine. `cmd/fanbench --grid canonical` now emits the exact D-001 ladder as a user-facing command path. **CPU smoke green** (`go test ./internal/bench -run FanScale` → `ok`). The published HEADLINE — live-model wall-clock + the LangGraph/AutoGen/CrewAI comparison + the results table — is explicitly **deferred** to a bench node (`DeferredRun`), not fabricated here. | [`internal/bench/fanscale.go`](../../internal/bench/fanscale.go) · [`internal/bench/fanscale_test.go`](../../internal/bench/fanscale_test.go) · [`internal/turnbench/fanout.go`](../../internal/turnbench/fanout.go) · [`cmd/fanbench/main.go`](../../cmd/fanbench/main.go) |
| **D-002** #253 LangChain/LangGraph (P1) | 🟢 **CLOSED — works by wire repoint** | LangChain (`ChatOpenAI`) and LangGraph (via its underlying LangChain chat model) reach the fak gateway by setting `base_url` to fak's OpenAI-compatible endpoint — documented and marked supported in the compatibility matrix (rows 58-59). fak's intended integration shape **is** "repoint the base URL," so no bespoke adapter is required. The only closed child. | [`docs/integrations/compatibility-matrix.md`](../integrations/compatibility-matrix.md) · [`docs/integrations/README.md`](../integrations/README.md) |
| **D-003** #250 CrewAI (P1) | 🟢 **Host-tractable acceptance shipped; live-model wall-clock folded into D-001** | CrewAI's `LLM(base_url=...)` (routed through LiteLLM) reaches the gateway today (matrix row 61, "Yes"). A runnable, dependency-free **example crew** (`examples/crewai-crew/`) now demonstrates the manager-worker pattern: governance over every worker tool call (verdicts match `fak preflight`) **and** the **manager-role coordination-overhead reduction**, documented with a **deterministic performance model** (modeled 4.93× prefill reduction at the illustrative geometry). The only open piece — the **live-model CrewAI-vs-native wall-clock** — is the deferred bench-node run owned by D-001 (#255), not asserted here. | [`examples/crewai-crew/`](../../examples/crewai-crew/README.md) · [`docs/integrations/compatibility-matrix.md`](../integrations/compatibility-matrix.md) (row 61) |
| **D-004** #248 AutoGen (P1) | 🟡 **Wire-supported; bespoke adapter/example acceptance open** | AutoGen / AG2 `OpenAIChatCompletionClient(base_url=...)` reaches the gateway today (matrix row 62, "Yes"). The issue's acceptance — an AutoGen **agent adapter**, **conversation-state preservation**, **multi-agent examples**, a **benchmark vs native** — is not shipped. | [`docs/integrations/compatibility-matrix.md`](../integrations/compatibility-matrix.md) (row 62) |
| **D-005** #245 Workflow Orchestration (P2) | 🔴 **Not implemented** | No workflow DSL (YAML/JSON), no DAG execution engine, no built-in map-reduce / fan-out *product* layer. The fan-out topology in `turnbench`/`fanbench` is the **D-001 benchmark harness**, not a user-facing orchestration layer. Design-only. | — (none) |
| **D-006** #243 Tool Ecosystem (P2) | 🔴 **Not implemented** | No built-in 20+ tool library (filesystem/HTTP/DB connectors). fak's architecture **gates** the agent's existing tools (deny-by-structure / repair / quarantine); it is not a tool *provider* — so this child is partly an architecture-fit question, not just unbuilt code. Design-only. | — (none) |
| **D-007** #241 Coordination Protocol (P2) | 🟡 **Partial — shared-state scaffold; full protocol is a sibling epic** | A **shared-task-record contract** (fixtures + JSON schema + validator) gives coordinated multi-agent workflows a shared record, and `internal/comm` is a low-level comm driver. The full **message-passing protocol / shared-KV API / RFC** the issue names is the active epic **#639** (MPI-shaped message-passing primitives). | [`examples/shared-task-record`](../../examples/shared-task-record) · [`internal/comm/comm.go`](../../internal/comm/comm.go) |
| **D-008** #238 Testing Framework (P2) | 🟡 **Partial — internal harness; public API not shipped** | Deterministic seeded agent-session generation + fixture-based scoring (`turnbench`), plus a transcript corpus + the trajectory-replay substrate (epic **#498**). No **public** assertion library, mock-tool-response API, or reproduce-from-transcript CLI yet. | [`internal/turnbench/`](../../internal/turnbench) · [`examples/trajectory`](../../examples/trajectory) |

Legend: 🟢 done · 🟡 partial (scaffold / wire-supported / sibling-epic) · 🔴 not implemented.

**Roll-up: 1 / 8 children closed (#253).** Of the 7 open: 1 is a scaffold whose published run
is bench-node-gated (D-001), D-003's host-tractable acceptance (example crew + manager-role
reduction doc + a deterministic performance model) is now shipped with only its live-model
wall-clock folded into the D-001 deferred run, 1 (D-004) is wire-supported with its
adapter/example acceptance still open, 2 are partial scaffolds whose public surface is tracked
by a separate epic (D-007→#639, D-008→#498), and 2 are unimplemented features (D-005, D-006).
So #304 stays **OPEN** — correctly (its published headline run is still deferred).

---

## 2. The gate — why #304 cannot honestly close here

The seven open acceptance gates fall into four classes, and **none is reachable on this
host**:

1. **Bench-node-gated published run (D-001).** The acceptance includes *"Published results"*
   and a *LangGraph/AutoGen/CrewAI comparison* — a live-model wall-clock sweep with the three
   frameworks installed. The harness is **shipped and host-runnable** (the deterministic
   geometry runs even at N=1000) and its smoke is green, but this win32 dev box has no
   live-model bench node and no framework install. Per BENCHMARK-AUTHORITY rules the published
   table stays a deferred `DeferredRun`; no throughput number is asserted here.

2. **Genuinely unimplemented features (D-005 workflow DSL, D-006 tool library).** These are
   not "scaffold + gated" — there is no implementation to measure. Each is a days-of-engineering
   leaf (a DAG execution engine; a 20+ tool library with safety annotations), and D-006 is also
   an architecture-fit question (fak gates tools rather than shipping them).

3. **Bespoke-adapter parity gap (D-003 CrewAI, D-004 AutoGen).** Both frameworks already reach
   fak by wire repoint, but the issues' acceptance asks for **native adapters + worked examples
   + a benchmark vs native** — not yet built. Each is a separate leaf, not a single small change.

4. **Scaffolds whose public surface is a sibling epic (D-007, D-008).** The shared-task-record
   contract and the turnbench/trajectory harness exist, but the protocol RFC (epic #639) and the
   public testing API (epic #498) are tracked elsewhere and unshipped.

**The single honest gate, stated plainly:** epic #304 closes only when all 8 children meet
their acceptance, and **7 are open** — spanning a deferred bench-node run, two unbuilt features,
two unbuilt adapters, and two scaffolds whose public surface is tracked by separate epics. No
code or doc change on this host can flip those bits without fabricating a benchmark number or
shipping multi-day features — which the repo's witness ledger and `make claims-lint` would
reject. So the correct deliverable is this tracker, not a closed epic.

---

## 3. Smallest next step per child (for the agent that picks one up)

| Child | Smallest honest next step | Where it runs |
|---|---|---|
| D-001 #255 | Stand up the headline run: install LangGraph/AutoGen/CrewAI on a bench node, run `fanbench` at N=100/500/1000 with a live model, record coordination-overhead + cross-agent-reuse vs the frameworks, publish the table | a live-model bench node |
| D-002 #253 | Done (closed). Optional: add a 3-workflow LangChain/LangGraph migration guide to harden the closure narrative | host-tractable (docs) |
| D-003 #250 | **Shipped** (`examples/crewai-crew/`): example crew + manager-role coordination-reduction doc + deterministic performance model. Remaining: the live-model CrewAI-vs-native wall-clock, owned by D-001 (#255) | a live-model bench node (wall-clock only) |
| D-004 #248 | Build a minimal AutoGen multi-agent example pointed at fak, show conversation-state preservation through the gateway, benchmark vs native | host-tractable (Python example) |
| D-005 #245 | Spec a workflow DSL (YAML) + a map-reduce/fan-out/DAG executor as a `new_leaf` package; CPU-correct first, then fault tolerance | host-tractable |
| D-006 #243 | Decide the architecture fit first (does fak ship tools, or only gate them?); if shipping, add filesystem read/write/glob with safety annotations as the first leaf | host-tractable |
| D-007 #241 | Graduate the shared-task-record contract into a message-passing API on `internal/comm`; write the RFC under epic #639 | host-tractable |
| D-008 #238 | Expose the turnbench fixture/scoring harness as a public assertion + mock-tool-response API; add reproduce-from-transcript over the trajectory corpus (epic #498) | host-tractable |

---

## 4. Provenance

- **Child set:** `gh issue list --label track/D-agent-framework --state all` (live, 2026-06-25);
  acceptance criteria from each child body (#255/#253/#250/#248/#245/#243/#241/#238).
- **Stale-link evidence:** `gh issue view` on the epic-body ids #314/#316/#319/#321/#324/#326/#328/#331
  (every one an unrelated `examples:`/`serve(gateway)` issue; #314 and #328 closed).
- **On-disk evidence:** `internal/bench/fanscale.go` + `fanscale_test.go`, `internal/turnbench/fanout.go`,
  `cmd/fanbench/main.go` (D-001); `docs/integrations/compatibility-matrix.md` rows 58-62 +
  `docs/integrations/README.md` (D-002/003/004); `examples/shared-task-record` + `internal/comm/comm.go`
  (D-007); `internal/turnbench/` + `examples/trajectory` (D-008).
- **Witnessed gate:** `go test ./internal/bench -run FanScale -count=1` → `ok ... 0.440s` (WSL Ubuntu-24.04;
  native Windows `go test` is OS-blocked — see AGENTS.md).
- **Honesty rails:** `make claims-lint`, BENCHMARK-AUTHORITY — the D-001 published cell is deferred,
  never asserted.

## 5. See also

- [`docs/notes/track-b-performance-parity-tracking-306.md`](track-b-performance-parity-tracking-306.md) · [`docs/notes/gpu-parity-tracking-480.md`](gpu-parity-tracking-480.md) · [`docs/notes/simd-cpu-parity-tracking-400.md`](simd-cpu-parity-tracking-400.md) — sibling parity trackers in the same house format.
- [`docs/integrations/compatibility-matrix.md`](../integrations/compatibility-matrix.md) — the wire-level support matrix that grounds D-002/003/004.
- Epics **#639** (MPI-shaped message-passing primitives, the D-007 protocol) and **#498** (trajectory-replay substrate, the D-008 testing surface).
