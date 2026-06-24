# INDEX — the full map of the fak repo

The curated map of everything in this repo that is not on the front page. If a
doc exists, it is reachable from here. New dated notes go under
[`docs/notes/`](docs/notes/) and get a line in the **Notes & research** section
below, so the front door never has to grow to stay complete.

> New here? Start with the front doors, not this map: the
> [README](README.md), [START-HERE](START-HERE.md), and the
> [agent orientation](AGENTS.md). This page is the index for when you already
> know roughly what you want and need the link.

## Start here

- [README](README.md) — what fak is and why, in one read.
- [START-HERE](START-HERE.md) — run a local model behind the gate in ten minutes.
- [Getting started](GETTING-STARTED.md) — install the single binary, point your agent at it.
- [AGENTS](AGENTS.md) — orientation for coding agents (build, test, the hard rules).
- [llms.txt](llms.txt) — the answer-engine index (and its inlined `llms-full.txt`).
- [Docs home](docs/index.md) — the published documentation landing page.

## What fak supports

The dedicated, cross-linked capability pages — what fak works with, each grounded in the
repo and the sourced [compatibility matrix](docs/integrations/compatibility-matrix.md).

- [What fak supports (hub)](docs/supported/README.md) — the index of every "supported" page.
- [Models](docs/supported/models.md) — any model you front, plus the in-kernel architectures proven bit-exact.
- [Features](docs/supported/features.md) — every capability with its shipped / simulated / stub status.
- [Clouds & hosted providers](docs/supported/clouds.md) — Anthropic, OpenAI, Gemini, xAI, Bedrock, Vertex, Azure, OpenRouter, Together, Groq, Fireworks.
- [APIs, wires & MCP](docs/supported/apis-and-protocols.md) — the wires fak speaks and the fak-native + MCP endpoints.
- [Agent harnesses & frameworks](docs/supported/agent-harnesses.md) — Claude Code, Cursor, Codex, and the framework field.
- [Serving engines](docs/supported/engines.md) — Ollama, vLLM, SGLang, llama.cpp, LM Studio, and the in-kernel engine.

## Measuring sticks (the scorecards)

Each turns a fuzzy goal into a number you can drive toward zero.

- [Industry scorecard](docs/industry-scorecard/README.md) — the one OUTWARD stick: an industry-first taxonomy of the dimensions the LLM-serving field competes on (vLLM/SGLang/TensorRT-LLM/llama.cpp), with fak's honest position on each (mostly named gaps). Two driven numbers: **coverage** (of the field) and **parity-debt** (honesty of the rows). Modular folder, regenerated from `tools/industry_scorecard.data/`.
- [Agent-readiness scorecard](docs/AGENT-READINESS-SCORECARD.md) — can an AI agent discover, adopt, and build on fak (friction-debt across the three steps an agent walks).
- [Repo-hygiene scorecard](docs/REPO-HYGIENE-SCORECARD.md) — verbosity, organization, indexing, and accessibility of the whole tree.
- [Code-quality scorecard](docs/CODE-QUALITY-SCORECARD.md) — the Go module's code-debt.
- [Observability scorecard](docs/OBSERVABILITY-SCORECARD.md) — the observability plane (metrics, dashboards, alerts, traces, proofs, ship-audit), graded so every dashboard/alert/doc points at a metric the binary emits and every claim is verifiable.
- [Demo-quality scorecard](docs/DEMO-QUALITY-SCORECARD.md) — are the demos runnable, honest, self-contained.
- [Demo-robustness scorecard](docs/DEMO-ROBUSTNESS-SCORECARD.md) — do the demos survive bad input and odd environments.
- [Learning-docs scorecard](docs/LEARNING-SCORECARD.md) — does the teaching set actually teach (learning-debt: a how-to with no runnable command, a tutorial with no worked output, an orphan lesson, an uncovered topic). Pedagogy counterpart of repo-hygiene (structure) and doc-appeal (voice).
- [Code-2x program](docs/CODE-2X-PROGRAM.md) — the plan to halve code-debt, then halve it again.

## Architecture & design

- [Architecture](ARCHITECTURE.md) — the registry seams and the frozen ABI.
- [GPU forward pass](GPU.md) — the in-kernel Llama decode on the GPU: a real on-box run witnessed against the CPU reference, with the honest gap to llama.cpp.
- [Partitioning](PARTITION.md) — how the kernel splits work across lanes and leaves.
- [Extending fak](EXTENDING.md) — plug in an optimization, prove it correct, prove it faster.
- [SOTA comparison](SOTA-COMPARISON.md) — where fak sits next to the state of the art.
- [Interoperability stance](docs/integrations/interoperability.md) — why fak adopts whatever agent/model/framework you run (the one opinion kept is the floor) + the honest per-wire grade for the flagship harnesses and every interop protocol.
- [Agent-to-agent value](docs/a2a-value-opportunities.md) — where the cross-agent cache pays off.
- [Agent–machine link protocol](docs/agent-machine-link-protocol.md) — how an agent and a host node bind.
- [DOS kernel transfer playbook](docs/dos-kernel-transfer-playbook.md) — moving the trust substrate to a new repo.
- [Prefill visuals](docs/prefill-visuals.md) — the diagrams behind prefill reuse.

## Operating the agent fleet

- [`fleet` operator console](tools/FLEET.md) — one command to watch the agent fleet on a host: live session/account health, what stopped and why, which accounts are resumable, and what needs you. The session-health companion to `dos top`; install it onto PATH with `fleet install`.

## Benchmarks & methodology

- [Benchmark authority](BENCHMARK-AUTHORITY.md) — the single source of truth for every number.
- [Benchmark template](BENCHMARK-TEMPLATE.md) — the shape a new benchmark result doc must take.
- [Hardware bench plan](docs/bench-plan.md) — auto-generated: the next highest-value test per bench-node (regenerate with `tools/bench_plan.py`, don't hand-edit).
- [Web-agent baselines](docs/webbench-baselines.md) — the measured WebVoyager numbers.
- [Web-agent blockers](docs/webbench-blockers.md) — what is not yet measured, and why.
- [Web-agent measurement summary](docs/webbench-real-measurements-summary.md) — the rolled-up real-run numbers.
- [DOS hook cost](docs/perf-dos-hook-cost.md) — what the commit-time guards cost in practice.
- [Runaway-guard cost](docs/perf-runaway-guard.md) — the cost of the process-reaper backstop.

## Status & tracking

Working docs that track a specific effort. Dated by design; they age out.

- [GPU parity tracking (#480)](docs/notes/gpu-parity-tracking-480.md) — bringing the GPU path to parity.
- [Model-arch seam status (#487)](docs/notes/model-arch-seam-status-487.md) — the model-architecture seam work.
- [Trust-floor decomposition (#492)](docs/notes/trust-floor-decomposition-492.md) — splitting the trust floor into checkable parts.

## Plans & media

- [Video content plan](docs/explainers/video-content-plan.md) — the explainer-video storyboard.

## Notes & research (`docs/notes/`)

Dated working notes: research, audits, and run logs. Kept for the record, not the
front page.

- [Agent optimization methods — field inventory, survey & index (2026-06-23)](docs/notes/RESEARCH-agent-optimization-methods-survey-2026-06-23.md) — 370 distinct methods across 16 families spanning the whole stack (reasoning, test-time compute, context/memory, retrieval, KV reuse, tools, planning, multi-agent, routing, serving, training, reliability), each with a one-line definition + what it optimizes + maturity, plus an alphabetical master index. The umbrella that cross-links the deep notes ([agentic caching](docs/notes/AGENTIC-CACHING-SOTA-2026-06-19.md), [scaling laws](docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md), [O(1) context economics](docs/explainers/o1-context-window-economics.md)).
- [Three layers of agent optimization — skill, trajectory, substrate (2026-06-24)](docs/notes/RESEARCH-three-layers-of-agent-optimization-2026-06-24.md) — the conceptual companion to the field inventory above: separates *what* an agent should do (skill), *how any run unfolds* abstract over skill (trajectory), and *what is kept hot* (substrate), then spends its length on the three differences. Uses only the refined claims that survived an adversarial pass — the "harness carries the weak agent" convergence is true on a live re-decode and false for a frozen trajectory (fak's astropy 1/1→0 counterexample + the `turnbench/ope.go` divergence frontier); trajectory↔substrate is a complement not an adjoint; the compiler-stack analogy is a leaky gradient because fak fuses the safety and reuse boundary. Closes on the LIVE-vs-DORMANT `req.Raw` passthrough gap.
- [Agentic caching SOTA (2026-06-19)](docs/notes/AGENTIC-CACHING-SOTA-2026-06-19.md)
- [Planned view measured over real transcripts (2026-06-23)](docs/notes/CTXPLAN-REAL-TRANSCRIPT-MEASUREMENT-2026-06-23.md) — the empirical counterpart to scaling.go's synthetic model: replays the heaviest real Claude Code sessions turn-by-turn through the real planner + page-fault handler. 13.3× fewer resident tokens than linear; 31.7% forecast-miss rate with 100% of misses served; exact recall 715/715 turns vs compaction losing facts on 695/715. Closes #559 (`cmd/ctxplanbench`).
- [Planner per-turn compute flatten, measured (2026-06-23)](docs/notes/CTXPLAN-PLANNING-COST-FLATTEN-2026-06-23.md) — the sibling that measures the planner's OWN work, not just its output: a persistent `ctxplan.Index` bounds the per-turn probe (Θ(N²)→Θ(c·N)) over the same real sessions. 851 turns: peak-probe ≤ cap on every turn, 1.5× less planner work (growing with N), resident-token efficiency preserved (within 1.2%), and the honest cost↔fidelity finding — the bounded probe is byte-identical to the full scan when it sees the whole candidate set (342/342) but diverges at default width on long sessions (every divergence a served page-fault, never a lost fact). #558/#559 (`cmd/ctxplanbench`).
- [Context-window baseline + self-reduce 2× (2026-06-23)](docs/notes/CTXWIN-CONTEXT-WINDOW-BASELINE-2026-06-23.md) — what a Claude Code session's context window is actually made of, and how much halves at low risk vs bounded-loss. The empirical pass over real transcripts (`tools/ctxwin.py`) the [O(1)-turn planner](docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md) names as its missing measurement.
- [Multiple-sink transmission for the cache lifecycle (2026-06-23)](docs/notes/MULTI-SINK-CACHE-LIFECYCLE-2026-06-23.md) — is one kernel op fanning Promote/Rollback across every registered `ProvisionalSink` useful for the KV/context cache lifecycle? Yes, but latent: one registrant today, the fan never exercised with >1 sink (now witnessed in `internal/spec`), and best-effort not atomic. Ranks which cache tiers should become sinks next.
- [DOS effective-usage audit (2026-06-22)](docs/notes/DOS-EFFECTIVE-USAGE-AUDIT-2026-06-22.md)
- [DOS `salience` usefulness audit (2026-06-24)](docs/notes/DOS-SALIENCE-USEFULNESS-AUDIT-2026-06-24.md) — is the "is this true thing LIVE or true-but-PARKED?" verdict useful? Sound, fail-safe-to-RETAIN, fills a real gap (the keep-but-park dual of `retire`) and matches spec (PARKED→exit 3, LIVE→0, INDETERMINATE→4) — but **latent**: nothing routes on the verdict, the CLI is single-item only (the no-loss `partition` fold is import-only), the evidence bits are host-produced. Dogfooded over 10 real true-but-parked fak findings; names the wiring fak still needs.
- [Gate down the stack (2026-06-22)](docs/notes/EXPLAINER-gate-down-the-stack-2026-06-22.md)
- [Trust floor, two lenses (2026-06-17)](docs/notes/EXPLAINER-trust-floor-two-lenses-2026-06-17.md)
- [GLM-5.2 DSA attention projections on the pure kernel (2026-06-22)](docs/notes/GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-GPU-SERVER-2026-06-22.md)
- [GLM-5.2 DSA sparse attention on the pure kernel (2026-06-23)](docs/notes/GLM52-DSA-SPARSE-ATTENTION-ON-PURE-KERNEL-2026-06-23.md)
- [GLM-5.2 DSA index selection on the pure kernel (2026-06-23)](docs/notes/GLM52-DSA-INDEX-SELECTION-ON-PURE-KERNEL-2026-06-23.md)
- [GLM-5.2 full DSA forward on the pure kernel, sm_80 GPU server (2026-06-23)](docs/notes/GLM52-DSA-FULL-FORWARD-ON-PURE-KERNEL-GPU-SERVER-2026-06-23.md)
- [GLM-5.2 five GPU server benchmarks (2026-06-22)](docs/notes/GLM52-PERFORMANT-GPU-SERVER-FIVE-BENCHMARKS-2026-06-22.md)
- [GLM-5.2 kernel + turn demos (2026-06-21)](docs/notes/GLM52-PURE-KERNEL-AND-AGENT-TURN-DEMOS-RESULTS-2026-06-21.md)
- [GLM-5.2 pure kernel on GPU server (2026-06-21)](docs/notes/GLM52-PURE-KERNEL-ON-GPU-SERVER-2026-06-21.md)
- [GLM-5.2 real-checkpoint generation matches HuggingFace on the pure engine (2026-06-23)](docs/notes/GLM52-REAL-ORACLE-GENERATION-ON-PURE-FAK-2026-06-23.md)
- [Mac bench refresh: CPU Q8 parity + uncontended radix (2026-06-23)](docs/notes/MAC-BENCH-REFRESH-2026-06-23.md)
- [The O(1) current turn: a planned view over the lossless history (2026-06-23)](docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md) — cost-based, forecast-driven context planner; the middle ground between an unbounded transcript and lossy compaction.
- [On-demand context KV reuse (2026-06-19)](docs/notes/ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md)
- [Cloud/VM/remote-control agent landscape + fak strategy (2026-06-23)](docs/notes/RESEARCH-cloud-vm-remote-agent-landscape-2026-06-23.md)
- [idea-scout triage: Doberman-Core as prior art (2026-06-23)](docs/notes/RESEARCH-doberman-prior-art-triage-2026-06-23.md)
- [idea-scout triage: defensive misdirection (CMPE) — theory-side support for detector-is-not-the-floor, deceptive mechanism not adopted (2026-06-23)](docs/notes/RESEARCH-defensive-misdirection-triage-2026-06-23.md)
- [idea-scout triage: 'When AUC 0.998 Is Not Enough' — probe-AUC is not detection; evidence for detector-is-not-the-floor (2026-06-23)](docs/notes/RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md)
- [Git operations in the kernel prefilter — what's possible (2026-06-22)](docs/notes/RESEARCH-git-in-kernel-prefilters-2026-06-22.md)
- [Grammar-constrained tool-call decoding, upstream vs downstream (2026-06-22)](docs/notes/RESEARCH-grammar-constrained-tool-call-decoding-2026-06-22.md)
- [A local model on the wire: first use case + when-it-makes-sense (2026-06-23)](docs/notes/RESEARCH-local-model-on-the-wire-2026-06-23.md) — small local model as a witnessed lossy proposer; first rung is a semantic poison screen behind ctxmmu's regex floor; outbound compression is blocked on the req.Raw passthrough.
- [Handoff prompts: extending the local-model-on-the-wire spine (2026-06-23)](docs/notes/PROMPTS-local-model-on-the-wire-next-agents-2026-06-23.md) — paste-ready prompts for the next agents to build rungs 2-5 on the shipped spine (epic #568); names each seam, witness, and the #555 outbound blocker.
- [idea-scout triage: MCPPrivacyDetector — protocol-induced MCP-server leakage; prior art + a threat fak already gates at runtime (result-admit), static analyzer not adopted (2026-06-23)](docs/notes/RESEARCH-mcp-privacy-leakage-triage-2026-06-23.md)
- [idea-scout triage: Tool-Guard / isolated planning — a threat axis fak does not yet gate (2026-06-23)](docs/notes/RESEARCH-tool-guard-isolated-planning-triage-2026-06-23.md)
- [idea-scout triage: relinking at the compression boundary — fak immune (extractive compression + call-not-prompt floor); prior art + a guarded invariant for future generative summarizers (2026-06-23)](docs/notes/RESEARCH-relinking-compression-boundary-triage-2026-06-23.md)
- [Ultra-long-context: levels, levers, naming (2026-06-22)](docs/notes/RESEARCH-ultra-long-context-levels-and-naming-2026-06-22.md)
- [Scaling laws of agents (2026-06-19)](docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md)
- [Capability floor, a security note (2026-06-18)](docs/notes/SECURITY-capability-floor-2026-06-18.md)
- [Turn-state demos audit (2026-06-21)](docs/notes/TURN-STATE-DEMOS-AUDIT-AND-DOGFOOD-2026-06-21.md)
- [Benchmarking visuals status (2026-06-18)](docs/notes/VISUALS-benchmarking-status-2026-06-18.md)
- [Permission-systems visuals (2026-06-18)](docs/notes/VISUALS-permission-systems-2026-06-18.md)
