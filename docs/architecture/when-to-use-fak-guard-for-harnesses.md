---
title: "When to use fak guard vs native harnesskit for custom harnesses"
description: "Architectural evaluation, trade-off analysis, and decision matrix for agent harness builders choosing between out-of-process proxy supervision (fak guard) and in-process native SDK construction (pkg/harnesskit)."
---

# When to Use `fak guard` vs Native `harnesskit` for Custom Agent Harnesses

> **Status:** Approved Architectural Specification  
> **Issue:** #11610  
> **Scope:** CLI (`fak guard`), Public SDK (`pkg/harnesskit`), Gateway (`internal/gateway`), Adjudication (`internal/adjudicator`)  
> **Date:** September 2026  
> **Authors:** fak Architecture Guild  

---

## 1. Executive Summary & Problem Statement

Builders designing custom AI agent harnesses frequently encounter an architectural fork in **fak**:

1. **Path A: Out-of-Process Wire Supervision (`fak guard`)**: Run an external, pre-existing agent runtime (such as Claude Code, OpenAI Codex, OpenCode, Aider, or custom Python/TypeScript frameworks) behind an intercepting HTTP/MCP proxy that transparently enforces security policies, deduplicates read-only tools via vDSO caching, and journals decisions into a tamper-evident audit log.
2. **Path B: In-Process Native Construction (`pkg/harnesskit`)**: Build a standalone, turnkey Go agent product from scratch using the exported `pkg/harnesskit` SDK, embedding the execution loop, model routing, tool definitions, and lifecycle management directly into a single binary.

Confusion arises when builders assume `fak guard` is a required wrapper for *every* custom harness, or conversely attempt to use `pkg/harnesskit` when their goal was simply to govern an existing CLI agent. 

This document defines the clear boundary, performance tradeoffs, and architectural decision criteria for choosing between `fak guard` and `pkg/harnesskit`, including considerations across the public open-source runtime and private autonomous factory environments.

### Strategic Alignment with Core Tenets (P1–P4)

| Tenet | `fak guard` (Proxy Supervision) | `pkg/harnesskit` (Native SDK) |
|---|---|---|
| **P1 Managed Context** | Stabilizes prompt-cache prefixes, compresses tool output history via Context MMU compactor, and injects dynamic instruction profiles at the wire boundary. | Harness directly manages in-memory context planes, instruction adapters, and token budgets via typed Go structs without wire serialization. |
| **P2 Net-True Efficiency** | Eliminates redundant tool re-executions via wire-level vDSO caching (40–70% token/turn savings), offset by ~0.5–2ms loopback HTTP proxy overhead. | Zero IPC/network serialization overhead. Direct Go function calls execute at CPU memory bandwidth; optimal for local inference loops. |
| **P3 Bounded Adaptation** | Default-deny capability floor enforced externally; child agent processes cannot talk past the proxy or tamper with policy manifests (`SELF_MODIFY` refusal). | Capability requirements and platform constraints are declared statically in `harness.lock.json` and validated at build/activation time. |
| **P4 Integrated Operations** | Transparent sidecar model: wraps arbitrary binaries without source changes, writing tamper-evident SHA256 decision journals and auto-healing crashes. | Compiled single-binary deployment (`fak up` philosophy); self-contained operational footprint with unified health and selfcheck endpoints. |

---

## 2. Architectural Deep-Dive: The Two Paradigms

```
                     PARADIGM 1: OUT-OF-PROCESS SUPERVISION (fak guard)
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                                                                             │
   │   ┌─────────────────────┐       HTTP / MCP        ┌─────────────────────┐   │
   │   │  External Agent CLI │ ──────────────────────> │      fak guard      │   │
   │   │ (Claude / Codex /   │                         │  (Interception      │   │
   │   │  OpenCode / Custom) │ <────────────────────── │   Gateway & Proxy)  │   │
   │   └─────────────────────┘                         └──────────┬──────────┘   │
   │                                                              │              │
   │                                                    Adjudicate / Policy      │
   │                                                    vDSO Cache / Audit Log   │
   │                                                              ▼              │
   │                                                   ┌─────────────────────┐   │
   │                                                   │  Upstream Provider  │   │
   │                                                   │ (Anthropic/OpenAI/  │   │
   │                                                   │  fak serve)         │   │
   │                                                   └─────────────────────┘   │
   └─────────────────────────────────────────────────────────────────────────────┘

                      PARADIGM 2: IN-PROCESS NATIVE HARNESS (pkg/harnesskit)
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                       CUSTOM HARNESS BINARY (Compiled Go)                   │
   │                                                                             │
   │   ┌─────────────────────────────────────────────────────────────────────┐   │
   │   │ User Customization (product/config.go, instructions, tools)          │   │
   │   └──────────────────────────────────┬──────────────────────────────────┘   │
   │                                      │ In-process API calls                 │
   │   ┌──────────────────────────────────▼──────────────────────────────────┐   │
   │   │ pkg/harnesskit (Factory, Activation, Planes, Lock v2)               │   │
   │   └──────────────────────────────────┬──────────────────────────────────┘   │
   │                                      │ In-process / HTTP stream             │
   │   ┌──────────────────────────────────▼──────────────────────────────────┐   │
   │   │ Kernel Runtime / Model Serving (fak serve or in-tree engine)        │   │
   │   └─────────────────────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. When `fak guard` IS Useful for Harness Builders

`fak guard` is the optimal solution when your objective is **governance, observation, or enhancement of an agent loop you do not fully own or do not wish to rewrite**:

### 1. Fronting Existing or Third-Party Agent Runtimes
- **Zero Source-Code Modification:** You want to run Claude Code, OpenAI Codex, OpenCode, Aider, Hermes, Cursor, or a framework like LangChain/AutoGen without modifying their internal execution loops.
- **Single Environment Configuration:** By passing `-- claude` or setting `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`, the existing agent transparently routes all model turns and tool executions through `fak guard`.

### 2. Enforcing a Default-Deny Capability Floor
- **Untrusted Agent Code Execution:** In enterprise environments where agents generate shell commands, edit arbitrary files, or query internal networks, `fak guard` acts as an unbypassable policy firewall.
- **Fail-Closed Security Postures:** Enforces path containment (`PathScope`), prevents dangerous rm/curl/eval commands, restricts unauthorized network egress, and prevents self-modification attacks (`SELF_MODIFY`) where the agent tries to weaken its own guardrails.

### 3. Wire-Level Tool Caching (vDSO)
- **Zero-Compute Redundant Reads:** Autonomous coding agents repeatedly inspect the same workspace files, run identical `git status` commands, or repeat read-only queries. `fak guard` intercepts these tool calls and returns verified-fresh cached results immediately without executing subprocesses or wasting model tokens.

### 4. Durable, Tamper-Evident Audit Trails
- **Compliance & Forensics:** For legal, enterprise, or autonomous multi-agent systems, every single tool call proposal, argument, and kernel verdict is recorded into a hash-chained JSONL decision journal (`--audit`). The resulting journal can be cryptographically verified via `fak audit verify`.

### 5. Prompt-Cache Alignment & Context MMU Optimization
- **Provider Cache Economics:** Upstream frontier providers (Anthropic, OpenAI, DeepSeek) offer significant prompt-caching discounts (e.g. 90% cheaper input tokens) provided prompt prefixes remain byte-identical. `fak guard` normalizes and pins tool schemas, injects standardized instruction profiles, and uses the Positive Compactor to prune stale tool noise while preserving cache prefixes across dozens of turns.

### 6. Long-Running Autonomous Resilience
- **Watchdog Auto-Healing & Account Rotation:** In unattended headless operations (`fak guard --rotate auto`), `fak guard` monitors process health, handles API rate-limit backoffs (HTTP 429), rotates across designated account seats, and recovers from transient network drops without terminating the agent session.

---

## 4. When `fak guard` is NOT Useful (Or Counterproductive)

Using `fak guard` is unnecessary, inefficient, or actively harmful in the following scenarios:

### 1. Building a Fak-Native, Single-Binary Agent (`pkg/harnesskit`)
- **Unnecessary Interception Layer:** If you are implementing your own agent loop in Go using `pkg/harnesskit` or `internal/agent`, placing `fak guard` in front of yourself is an anti-pattern. You already have programmatic access to the kernel's tool registry, policy evaluators, and context managers in-process. Adding an external HTTP proxy introduces pointless IPC overhead, port allocation, and JSON marshaling round-trips.

### 2. High-Frequency, Ultra-Low-Latency Local Inference Loops
- **Serialization Bottlenecks:** When serving local quantized models via `fak serve` (using native Metal, CUDA, or SIMD kernels) where token generation and tool calls execute in tens of milliseconds, interposing a loopback HTTP/SSE proxy adds 1–3ms of network and parsing overhead per turn. For tight benchmark loops or real-time robotics/voice agents, in-process execution via `pkg/harnesskit` is required.

### 3. Non-Standard Streaming Protocols & Generative UI
- **Wire Mismatch:** `fak guard` specializes in standard wire protocols: OpenAI Chat Completions, Anthropic Messages, and standard MCP JSON-RPC. If your custom harness uses a bespoke bidirectional WebSocket protocol, reactive event graphs, or streaming generative UI schemas (e.g. dynamic canvas updates), `fak guard`'s wire parser cannot inspect or validate tool calls without breaking your stream.

### 4. High-Throughput Autonomous Factory Dispatchers (`fak-private`)
- **Process Spawning Overhead:** In continuous autonomous development factories (such as `fak-private/platform/dispatch/`), worker agents are dispatched in waves of dozens or hundreds across disjoint repository lanes. Spawning a separate `fak guard` daemon process for every single micro-worker consumes substantial host memory, sockets, and CPU handles. Instead, factory workers invoke fast Go library calls (`pkg/harnesskit/lockv2`) or execute single-shot CLI verbs (`fak dev ... --json`).

### 5. Embedded Read-Only or Single-Task CLI Utilities
- **Overkill for Constrained Tools:** For small command-line utilities, read-only code search tools, or single-turn summarizers where tools cannot mutate disk or network state, running a multi-threaded supervision gateway is architectural bloat.

---

## 5. Trade-Off Analysis: Latency, Memory, Governance, and Complexity

Selecting the appropriate harness integration pattern requires balancing four fundamental engineering trade-offs:

### 5.1 Latency Overhead & Wire Serialization
- **`fak guard` (Wire Loopback Proxy):** Interposes an HTTP/MCP reverse proxy between the agent and upstream LLMs or tool engines.
  - *Per-Turn Overhead:* Adds ~0.5ms to 2.0ms per request (loopback socket I/O, HTTP parsing, capability AST checks, and in-memory vDSO table lookups).
  - *Remote Model Context:* When calling external frontier models (Claude 3.5 Sonnet, GPT-4o, DeepSeek) where network RTT is 20–150ms and generation takes 500–4000ms, an extra ~1ms represents <0.2% latency overhead—statistically imperceptible.
  - *Net-Negative Trajectory Latency:* When vDSO cache hits resolve read-only tools (e.g. repeated file inspections, `git status`), `fak guard` answers in <1ms without executing the shell command or invoking the model, cutting aggregate session wall-clock time by 15–35%.
- **`pkg/harnesskit` (In-Process Native SDK):** Executes tool dispatch, context manipulation, and model routing via direct Go function calls.
  - *Per-Turn Overhead:* <10 microseconds (CPU register and pointer dereference).
  - *Local Inference Prerequisite:* Crucial when pairing with `fak serve` over native Metal, CUDA, or SIMD kernels where local token generation and tool calls execute in 10–30ms. Interposing a 2ms wire hop in local loops would impose an unacceptable 10–20% latency penalty.

### 5.2 Memory Footprint & Resource Scaling
- **`fak guard` (Independent Supervisor Process):**
  - Runs as an autonomous daemon alongside the agent runtime (e.g., Node.js for Claude Code/OpenCode, Python for Aider/LangChain).
  - Consumes ~25MB–50MB RSS for HTTP connection pools, context MMU buffers, and vDSO in-memory tables.
  - Allocates OS file descriptors, loopback TCP ports, or named pipes for each connected child agent.
  - *High-Density Fan-Out Constraint:* Running 50–100 parallel subagent workers concurrently on a single workstation (e.g. autonomous wave dispatchers) can consume 2.5GB–5GB of RAM and exhaust ephemeral ports or file descriptors.
- **`pkg/harnesskit` (Compiled In-Process Library):**
  - Shares the host binary's single Go runtime heap and garbage collector.
  - Zero extra OS processes, zero socket descriptors allocated for IPC.
  - *High-Density Scaling Advantage:* Enables spinning up hundreds of concurrent agent goroutines inside a single process, bounded only by model KV-cache and application memory.

### 5.3 Governance Boundaries & Security Guarantees
- **`fak guard` (Hard OS Process Boundary):**
  - *Adversarial Containment:* The proxy executes in an isolated process memory space. Untrusted or prompt-injected agents running in child processes (Python, Node, Bash) cannot inspect, manipulate, or disable the policy engine.
  - *Immunity to Self-Modification (`SELF_MODIFY` Refusal):* Even if an agent acquires arbitrary local code execution, it cannot alter the active `fak guard` policy or silence the decision journal.
  - *Cryptographic Auditing:* Writes an append-only, SHA256 hash-chained JSONL decision journal (`--audit`) verifiable via `fak audit verify`.
- **`pkg/harnesskit` (Cooperative In-Process Contract):**
  - *Type-Safe Contract Enforcement:* Tools, extensions, and context planes are bounded by Go type definitions, capability descriptors, and canonical lockfiles (`harness.lock.json` validated via `pkg/harnesskit/lockv2`).
  - *Intra-Process Trust Model:* Protects against model steerability failures, unauthorized tool capabilities, and schema drift. However, because user-defined tools share the same memory space, an intentional memory-unsafe exploit (e.g. cgo or unsafe pointers) could theoretically access process memory.

### 5.4 Operational Lifecycle & Execution Complexity
- **`fak guard` (Zero-Code Drop-In Wrapper):**
  - *Zero Integration Cost:* Requires zero source modifications to the agent; configure via one CLI flag (`fak guard -- claude`) or an environment variable (`ANTHROPIC_BASE_URL`).
  - *Process Supervision Burden:* Requires managing the lifecycle of two coupled processes (the proxy daemon and the agent CLI), coordinating shutdown signals, and handling orphan processes.
  - *Built-in Autonomous Reliability:* Provides built-in process watchdog auto-healing, terminal reset recovery, rate-limit backoff handling, and multi-account seat rotation (`--rotate auto`).
- **`pkg/harnesskit` (Single-Binary Operational Simplicity):**
  - *Upfront Construction Cost:* Requires Go development, project scaffolding (`fak harness init`), module dependency management, and compilation.
  - *Operational Simplicity in Production:* Results in a single, static binary with zero runtime dependencies (`fak up` philosophy). Trivial to package into scratch Docker containers, systemd services, or Kubernetes pods.
  - *Lifecycle Autonomy:* The host application owns its own process restarts, error handling, and telemetry integration.

---

## 6. Comparative Decision Matrix

| Dimension | `fak guard` (Out-of-Process Supervision) | `pkg/harnesskit` (In-Process Native SDK) |
|---|---|---|
| **Primary Use Case** | Governed execution of existing CLI agents (Claude, Codex, OpenCode, Aider, Hermes). | Developing a new, standalone, branded agent product or single-binary harness. |
| **Language / Runtime** | Agnostic (any agent speaking HTTP Chat/Messages or MCP). | Go (imports `pkg/harnesskit`). |
| **Process Model** | Supervised child process behind an intercepting loopback proxy. | Single monolithic binary embedding the agent loop. |
| **Setup Overhead** | Zero code changes: 1 flag or environment variable repoint. | Go project initialization (`fak harness init`), compilation, and typed configuration. |
| **Tool Execution** | Intercepted at HTTP wire or MCP stdio/SSE socket boundary. | Directly registered and dispatched in-process via Go interfaces. |
| **Latency Overhead** | ~0.5–2ms per request (loopback socket + JSON parsing + policy check). | <10µs (direct in-memory function call). |
| **Memory Footprint** | ~25–50MB RSS per supervisor process + OS socket handles. | 0 MB overhead; shared within host Go runtime heap. |
| **Security Enforcement** | External default-deny floor; child process cannot bypass proxy (`SELF_MODIFY` refusal). | Internal API contracts; builder controls policy and extension registration. |
| **Audit Capabilities** | Tamper-evident, hash-chained JSONL decision journal (`--audit`). | Telemetry streams emitted via `EventStream` / `StreamObserver`. |
| **Prompt Caching** | Automated prefix stabilization and tool output compaction. | Builder explicitly manages system prompt and context planes. |
| **Resilience / Watchdog** | In-flight process auto-healing, terminal recovery, account rotation. | Application must handle its own process lifecycles and recovery. |
| **Operational Simplicity** | Manages dual processes (supervisor + agent); zero build step. | Single compiled binary (`fak up` philosophy); requires Go build. |

---

## 7. Architectural Decision Tree for Builders

```
Are you building or customizing an agent harness?
  │
  ├─► Do you already have a working agent runtime (Claude Code, Codex, OpenCode, Python framework)?
  │     │
  │     ├─► YES: Use `fak guard`.
  │     │         - Point ANTHROPIC_BASE_URL or OPENAI_BASE_URL at `fak guard`.
  │     │         - Gain capability floors, vDSO tool cache, audit logs, and prompt-cache optimization with 0 code changes.
  │     │
  │     └─► NO: Are you writing a new Go-based agent product from scratch?
  │           │
  │           ├─► YES: Use `pkg/harnesskit` and `fak harness init`.
  │           │         - Build directly on the public Go SDK.
  │           │         - Keep tool execution and context management in-process for maximum throughput.
  │           │
  │           └─► NO (Writing from scratch in Python/TypeScript/Rust):
  │                     - Build your agent to speak standard OpenAI or Anthropic wire format.
  │                     - Front your agent with `fak guard` or point directly at `fak serve`.
```

---

## 8. Public vs. Private Boundary Considerations

When deploying custom harnesses across the boundary between **public `fak`** (open-core runtime) and **private `fak-private`** (proprietary enterprise serving and autonomous development factory):

1. **Public Harnesses:**
   - Must import only exported packages under `pkg/` (such as `pkg/harnesskit`, `pkg/abi`, `pkg/fakclient`).
   - Use `fak guard` with public-safe capability policies (`examples/customer-support-readonly-policy.json`) that forbid out-of-tree writes and dangerous system calls.
2. **Private Factory Harnesses (`fak-private`):**
   - High-throughput factory workers in `platform/dispatch/` bypass `fak guard` interactive wrappers to prevent socket exhaustion and latency spikes during bulk issue resolution waves.
   - Private factory code consumes public contracts via `go.work` and `pkg/harnesskit/lockv2` without importing `internal/*`.
   - Secret-bearing environments enforce secret quarantine: `fak guard`'s external redaction and leak-scan filters are used at perimeter boundaries, while internal dispatch loops rely on verified lockfile hashes.

---

## 9. Related Artifacts & References

- **Issue #11610:** Architectural evaluation of when `fak guard` is useful vs native `harnesskit`.
- **Issue #11611:** Public-harness vs private-factory contract and isolation rules across `fak` and `fak-private`.
- **Public vs. Private Harness Boundary:** [`docs/architecture/public-private-harness-boundary.md`](public-private-harness-boundary.md)
- **Public SDK Contract:** [`docs/harness-kit-contract.md`](../harness-kit-contract.md)
- **Harness Generator Guide:** [`docs/harness-init.md`](../harness-init.md)
- **Supported Harnesses Matrix:** [`docs/supported/agent-harnesses.md`](../supported/agent-harnesses.md)
- **Dev-Process Private Boundary:** [`docs/dev-process-private-boundary.md`](../dev-process-private-boundary.md)
