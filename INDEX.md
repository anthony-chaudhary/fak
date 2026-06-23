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

## Measuring sticks (the scorecards)

Each turns a fuzzy goal into a number you can drive toward zero.

- [Industry scorecard](docs/industry-scorecard/README.md) — the one OUTWARD stick: an industry-first taxonomy of the dimensions the LLM-serving field competes on (vLLM/SGLang/TensorRT-LLM/llama.cpp), with fak's honest position on each (mostly named gaps). Two driven numbers: **coverage** (of the field) and **parity-debt** (honesty of the rows). Modular folder, regenerated from `tools/industry_scorecard.data/`.
- [Agent-readiness scorecard](docs/AGENT-READINESS-SCORECARD.md) — can an AI agent discover, adopt, and build on fak (friction-debt across the three steps an agent walks).
- [Repo-hygiene scorecard](docs/REPO-HYGIENE-SCORECARD.md) — verbosity, organization, indexing, and accessibility of the whole tree.
- [Code-quality scorecard](docs/CODE-QUALITY-SCORECARD.md) — the Go module's code-debt.
- [Observability scorecard](docs/OBSERVABILITY-SCORECARD.md) — the observability plane (metrics, dashboards, alerts, traces, proofs, ship-audit), graded so every dashboard/alert/doc points at a metric the binary emits and every claim is verifiable.
- [Demo-quality scorecard](docs/DEMO-QUALITY-SCORECARD.md) — are the demos runnable, honest, self-contained.
- [Demo-robustness scorecard](docs/DEMO-ROBUSTNESS-SCORECARD.md) — do the demos survive bad input and odd environments.
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

- [Agentic caching SOTA (2026-06-19)](docs/notes/AGENTIC-CACHING-SOTA-2026-06-19.md)
- [Context-window baseline + self-reduce 2× (2026-06-23)](docs/notes/CTXWIN-CONTEXT-WINDOW-BASELINE-2026-06-23.md) — what a Claude Code session's context window is actually made of, and how much halves at low risk vs bounded-loss. The empirical pass over real transcripts (`tools/ctxwin.py`) the [O(1)-turn planner](docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md) names as its missing measurement.
- [Multiple-sink transmission for the cache lifecycle (2026-06-23)](docs/notes/MULTI-SINK-CACHE-LIFECYCLE-2026-06-23.md) — is one kernel op fanning Promote/Rollback across every registered `ProvisionalSink` useful for the KV/context cache lifecycle? Yes, but latent: one registrant today, the fan never exercised with >1 sink (now witnessed in `internal/spec`), and best-effort not atomic. Ranks which cache tiers should become sinks next.
- [DOS effective-usage audit (2026-06-22)](docs/notes/DOS-EFFECTIVE-USAGE-AUDIT-2026-06-22.md)
- [Gate down the stack (2026-06-22)](docs/notes/EXPLAINER-gate-down-the-stack-2026-06-22.md)
- [Trust floor, two lenses (2026-06-17)](docs/notes/EXPLAINER-trust-floor-two-lenses-2026-06-17.md)
- [GLM-5.2 DSA attention projections on the pure kernel (2026-06-22)](docs/notes/GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-GPU-SERVER-2026-06-22.md)
- [GLM-5.2 DSA sparse attention on the pure kernel (2026-06-23)](docs/notes/GLM52-DSA-SPARSE-ATTENTION-ON-PURE-KERNEL-2026-06-23.md)
- [GLM-5.2 full DSA forward on the pure kernel, sm_80 GPU server (2026-06-23)](docs/notes/GLM52-DSA-FULL-FORWARD-ON-PURE-KERNEL-GPU-SERVER-2026-06-23.md)
- [GLM-5.2 five GPU server benchmarks (2026-06-22)](docs/notes/GLM52-PERFORMANT-GPU-SERVER-FIVE-BENCHMARKS-2026-06-22.md)
- [GLM-5.2 kernel + turn demos (2026-06-21)](docs/notes/GLM52-PURE-KERNEL-AND-AGENT-TURN-DEMOS-RESULTS-2026-06-21.md)
- [GLM-5.2 pure kernel on GPU server (2026-06-21)](docs/notes/GLM52-PURE-KERNEL-ON-GPU-SERVER-2026-06-21.md)
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
- [idea-scout triage: MCPPrivacyDetector — protocol-induced MCP-server leakage; prior art + a threat fak already gates at runtime (result-admit), static analyzer not adopted (2026-06-23)](docs/notes/RESEARCH-mcp-privacy-leakage-triage-2026-06-23.md)
- [idea-scout triage: Tool-Guard / isolated planning — a threat axis fak does not yet gate (2026-06-23)](docs/notes/RESEARCH-tool-guard-isolated-planning-triage-2026-06-23.md)
- [Ultra-long-context: levels, levers, naming (2026-06-22)](docs/notes/RESEARCH-ultra-long-context-levels-and-naming-2026-06-22.md)
- [Scaling laws of agents (2026-06-19)](docs/notes/SCALING-LAWS-OF-AGENTS-2026-06-19.md)
- [Capability floor, a security note (2026-06-18)](docs/notes/SECURITY-capability-floor-2026-06-18.md)
- [Turn-state demos audit (2026-06-21)](docs/notes/TURN-STATE-DEMOS-AUDIT-AND-DOGFOOD-2026-06-21.md)
- [Benchmarking visuals status (2026-06-18)](docs/notes/VISUALS-benchmarking-status-2026-06-18.md)
- [Permission-systems visuals (2026-06-18)](docs/notes/VISUALS-permission-systems-2026-06-18.md)
