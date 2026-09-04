# Start here: human navigator and task router

*`README.md` is the single canonical front door for product overview, value proposition, and quick proof. This page owns one job: human task routing. It takes a goal you have right now and routes you to the single authoritative document without circular loops.*

> **TL;DR:** Newcomers should focus on the 16 core pages in the Newcomer Tier below. For immediate tasks, use the Task Router table to jump directly to the right guide.

The documentation repository contains over 1,200 pages spanning specifications, historical research notes, and benchmarks. To prevent navigation confusion, documentation is split into two distinct tiers: the **Newcomer Tier** (the essential 16 core pages) and the **Operator/Maintainer Tier**.

---

## The Newcomer Tier (Essential 16 pages)

If you are new to fak, focus exclusively on these core documents. They cover everything required to understand, install, configure, and operate the runtime:

| Document | Purpose |
|---|---|
| [`README.md`](README.md) | Canonical front door, product overview, and 60-second proof. |
| [`GETTING-STARTED.md`](GETTING-STARTED.md) | Concrete setup sequence: offline verification, gateway setup, and local serve. |
| [`docs/fak/tutorial.md`](docs/fak/tutorial.md) | Step-by-step guided first session with captured command outputs. |
| [`docs/repro-packet.md`](docs/repro-packet.md) | Deterministic offline reproducibility packet and policy checks. |
| [`docs/fak/governed-agent-quickstart.md`](docs/fak/governed-agent-quickstart.md) | 10-minute quickstart to launch a governed agent offline. |
| [`docs/fak/server-quickstart.md`](docs/fak/server-quickstart.md) | Stand up a shared OpenAI, Anthropic, or MCP gateway (`fak serve`). |
| [`docs/integrations/README.md`](docs/integrations/README.md) | Integration chooser for Claude Code, Codex, Cursor, and other harnesses. |
| [`docs/integrations/claude.md`](docs/integrations/claude.md) | Default one-command proxy recipe (`fak guard -- claude`). |
| [`POLICY.md`](POLICY.md) | Schema and reference for authoring default-deny capability floors. |
| [`docs/showcase.html`](docs/showcase.html) | Interactive browser tour of tool call adjudication and caching. |
| [`docs/adoption/troubleshooting-first-run.md`](docs/adoption/troubleshooting-first-run.md) | Symptoms, causes, and one-line fixes for common first-run issues. |
| [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) | Performance map for token savings, turn elimination, and context reuse. |
| [`docs/architecture.md`](docs/architecture.md) | Architecture boundary and whole-path agent coordination target. |
| [`docs/glossary.md`](docs/glossary.md) | Precise definitions for core runtime terms. |
| [`docs/FAQ.md`](docs/FAQ.md) | Frequently asked questions on setup, protocols, and security. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Guidelines for contributing code or documentation. |

### Quick verification: try it (30 seconds)

Test tool-call adjudication locally without a model or GPU:

```bash
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"
```

**Expected output:**
```
verdict=DENY reason=POLICY_BLOCK by=monitor
```

---

## Task Router: what do you want to do today?

Select the task you need to accomplish to find its primary guide and immediate next action:

| You want to… | Authoritative route | Next action |
|---|---|---|
| Understand what fak is and read the pitch | [`README.md`](README.md) | Read the front door and inspect the hardware benchmark table. |
| Verify the tool-call boundary offline | [`docs/repro-packet.md`](docs/repro-packet.md) | Run the deterministic offline proof with zero downloads (no key, model, or GPU). |
| Install fak on your local machine | [`GETTING-STARTED.md`](GETTING-STARTED.md) | Download a prebuilt binary or run `go install`, then verify your install. |
| Protect an agent you already use | [`docs/integrations/README.md`](docs/integrations/README.md) | Pick your agent harness and launch it behind `fak guard`. |
| Run a shared model gateway | [`docs/fak/server-quickstart.md`](docs/fak/server-quickstart.md) | Launch `fak serve` pointing to an upstream provider or Ollama. |
| Author custom tool allow/deny rules | [`POLICY.md`](POLICY.md) | Dump the built-in policy with `fak policy --dump` and customize it. |
| Fix a command that failed or misbehaved | [`docs/adoption/troubleshooting-first-run.md`](docs/adoption/troubleshooting-first-run.md) | Match your symptom to its cause and apply the one-line fix. |
| Understand how fak coordinates execution | [`docs/architecture.md`](docs/architecture.md) | Review the five-layer observation to typed-effect architecture. |
| Contribute code or documentation | [`CONTRIBUTING.md`](CONTRIBUTING.md) | Check prerequisites, sign the DCO, and follow the contribution workflow. |

---

## Operator and Maintainer Tier

Use these resources when deploying to production, running fleet infrastructure, or working on the kernel codebase:

### Production and deployment
- Cloud deployment: [`docs/deployment.md`](docs/deployment.md) covers local, fleet, and air-gapped deployment patterns.
- Supported providers: [`docs/supported/clouds.md`](docs/supported/clouds.md) details hosted model provider support.
- Reference architecture: [`docs/vendor/neo-cloud-reference-architecture.md`](docs/vendor/neo-cloud-reference-architecture.md) documents production cloud topologies.
- Rollback procedures: [`docs/ROLLBACK.md`](docs/ROLLBACK.md) details safe recovery steps for bad deployments.
- Security policy: [`SECURITY.md`](SECURITY.md) defines capability security guarantees and private vulnerability disclosure.

### Benchmarks and hardware
- Hardware matrix: [`docs/HARDWARE-MATRIX.md`](docs/HARDWARE-MATRIX.md) lists supported acceleration targets (Apple Silicon, AMD, NVIDIA).
- Benchmark authority: [`BENCHMARK-AUTHORITY.md`](BENCHMARK-AUTHORITY.md) provides canonical benchmarks and claim boundaries.
- Claims register: [`CLAIMS.md`](CLAIMS.md) records shipped, simulated, and stub status for all claims.
- Product status: [`STATUS.md`](STATUS.md) tracks subsystem implementation maturity.

### Kernel internals and repository workflows
- Extending the kernel: [`EXTENDING.md`](EXTENDING.md) explains how to register new leaves without editing core files.
- Internal architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md) details kernel subsystem contracts.
- Automated coding agents: [`AGENTS.md`](AGENTS.md) contains operational rules for autonomous workers on the shared trunk.
- Exhaustive documentation index: [`INDEX.md`](INDEX.md) catalogs the full 1,200+ page corpus for name-based lookups and archival research notes.
- Kernel boundary separation: [`docs/fak-vs-dos.md`](docs/fak-vs-dos.md) distinguishes the agent execution path (FAK) from the work decision path (DOS).
