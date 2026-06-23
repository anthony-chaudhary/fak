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

- [Industry scorecard](docs/INDUSTRY-SCORECARD.md) — the one OUTWARD stick: fak vs the state of the art (vLLM/SGLang/llama.cpp), graded for honest parity / true gain / shown losses.
- [Agent-readiness scorecard](docs/AGENT-READINESS-SCORECARD.md) — can an AI agent discover, adopt, and build on fak (friction-debt across the three steps an agent walks).
- [Repo-hygiene scorecard](docs/REPO-HYGIENE-SCORECARD.md) — verbosity, organization, indexing, and accessibility of the whole tree.
- [Code-quality scorecard](docs/CODE-QUALITY-SCORECARD.md) — the Go module's code-debt.
- [Demo-quality scorecard](docs/DEMO-QUALITY-SCORECARD.md) — are the demos runnable, honest, self-contained.
- [Demo-robustness scorecard](docs/DEMO-ROBUSTNESS-SCORECARD.md) — do the demos survive bad input and odd environments.
- [Code-2x program](docs/CODE-2X-PROGRAM.md) — the plan to halve code-debt, then halve it again.

## Architecture & design

- [Architecture](ARCHITECTURE.md) — the registry seams and the frozen ABI.
- [Partitioning](PARTITION.md) — how the kernel splits work across lanes and leaves.
- [Extending fak](EXTENDING.md) — plug in an optimization, prove it correct, prove it faster.
- [SOTA comparison](SOTA-COMPARISON.md) — where fak sits next to the state of the art.
- [Agent-to-agent value](docs/a2a-value-opportunities.md) — where the cross-agent cache pays off.
- [Agent–machine link protocol](docs/agent-machine-link-protocol.md) — how an agent and a host node bind.
- [DOS kernel transfer playbook](docs/dos-kernel-transfer-playbook.md) — moving the trust substrate to a new repo.
- [Prefill visuals](docs/prefill-visuals.md) — the diagrams behind prefill reuse.

## Benchmarks & methodology

- [Benchmark authority](BENCHMARK-AUTHORITY.md) — the single source of truth for every number.
- [Benchmark template](BENCHMARK-TEMPLATE.md) — the shape a new benchmark result doc must take.
- [Web-agent baselines](docs/webbench-baselines.md) — the measured WebVoyager numbers.
- [Web-agent blockers](docs/webbench-blockers.md) — what is not yet measured, and why.
- [Web-agent measurement summary](docs/webbench-real-measurements-summary.md) — the rolled-up real-run numbers.
- [DOS hook cost](docs/perf-dos-hook-cost.md) — what the commit-time guards cost in practice.
- [Runaway-guard cost](docs/perf-runaway-guard.md) — the cost of the process-reaper backstop.

## Status & tracking

Working docs that track a specific effort. Dated by design; they age out.

- [Scaling laws of agents (2026-06-19)](docs/SCALING-LAWS-OF-AGENTS-2026-06-19.md) — when the cache win kicks in as a fleet grows.
- [GPU parity tracking (#480)](docs/gpu-parity-tracking-480.md) — bringing the GPU path to parity.
- [Model-arch seam status (#487)](docs/model-arch-seam-status-487.md) — the model-architecture seam work.
- [Trust-floor decomposition (#492)](docs/trust-floor-decomposition-492.md) — splitting the trust floor into checkable parts.

## Plans & media

- [Video content plan](docs/explainers/video-content-plan.md) — the explainer-video storyboard.

## Notes & research (`docs/notes/`)

Dated working notes: research, audits, and run logs. Kept for the record, not the
front page.

- [Agentic caching SOTA (2026-06-19)](docs/notes/AGENTIC-CACHING-SOTA-2026-06-19.md)
- [DOS effective-usage audit (2026-06-22)](docs/notes/DOS-EFFECTIVE-USAGE-AUDIT-2026-06-22.md)
- [Gate down the stack (2026-06-22)](docs/notes/EXPLAINER-gate-down-the-stack-2026-06-22.md)
- [Trust floor, two lenses (2026-06-17)](docs/notes/EXPLAINER-trust-floor-two-lenses-2026-06-17.md)
- [GLM-5.2 DSA attention projections on the pure kernel (2026-06-22)](docs/notes/GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-DGX-2026-06-22.md)
- [GLM-5.2 DSA sparse attention on the pure kernel (2026-06-23)](docs/notes/GLM52-DSA-SPARSE-ATTENTION-ON-PURE-KERNEL-2026-06-23.md)
- [GLM-5.2 five DGX benchmarks (2026-06-22)](docs/notes/GLM52-PERFORMANT-DGX-FIVE-BENCHMARKS-2026-06-22.md)
- [GLM-5.2 kernel + turn demos (2026-06-21)](docs/notes/GLM52-PURE-KERNEL-AND-AGENT-TURN-DEMOS-RESULTS-2026-06-21.md)
- [GLM-5.2 pure kernel on DGX A100 (2026-06-21)](docs/notes/GLM52-PURE-KERNEL-ON-GPU-DGX-A100-2026-06-21.md)
- [Mac bench refresh: CPU Q8 parity + uncontended radix (2026-06-23)](docs/notes/MAC-BENCH-REFRESH-2026-06-23.md)
- [On-demand context KV reuse (2026-06-19)](docs/notes/ON-DEMAND-CONTEXT-KV-REUSE-2026-06-19.md)
- [Cloud/VM/remote-control agent landscape + fak strategy (2026-06-23)](docs/notes/RESEARCH-cloud-vm-remote-agent-landscape-2026-06-23.md)
- [idea-scout triage: Doberman-Core as prior art (2026-06-23)](docs/notes/RESEARCH-doberman-prior-art-triage-2026-06-23.md)
- [idea-scout triage: 'When AUC 0.998 Is Not Enough' — probe-AUC is not detection; evidence for detector-is-not-the-floor (2026-06-23)](docs/notes/RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md)
- [Git operations in the kernel prefilter — what's possible (2026-06-22)](docs/notes/RESEARCH-git-in-kernel-prefilters-2026-06-22.md)
- [Grammar-constrained tool-call decoding, upstream vs downstream (2026-06-22)](docs/notes/RESEARCH-grammar-constrained-tool-call-decoding-2026-06-22.md)
- [idea-scout triage: Tool-Guard / isolated planning — a threat axis fak does not yet gate (2026-06-23)](docs/notes/RESEARCH-tool-guard-isolated-planning-triage-2026-06-23.md)
- [Ultra-long-context: levels, levers, naming (2026-06-22)](docs/notes/RESEARCH-ultra-long-context-levels-and-naming-2026-06-22.md)
- [Capability floor, a security note (2026-06-18)](docs/notes/SECURITY-capability-floor-2026-06-18.md)
- [Turn-state demos audit (2026-06-21)](docs/notes/TURN-STATE-DEMOS-AUDIT-AND-DOGFOOD-2026-06-21.md)
- [Benchmarking visuals status (2026-06-18)](docs/notes/VISUALS-benchmarking-status-2026-06-18.md)
- [Permission-systems visuals (2026-06-18)](docs/notes/VISUALS-permission-systems-2026-06-18.md)
