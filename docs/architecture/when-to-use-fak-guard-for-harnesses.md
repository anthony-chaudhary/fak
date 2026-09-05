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

## 5. Comparative Decision Matrix

| Dimension | `fak guard` (Out-of-Process Supervision) | `pkg/harnesskit` (In-Process Native SDK) |
|---|---|---|
| **Primary Use Case** | Governed execution of existing CLI agents (Claude, Codex, OpenCode, Aider, Hermes). | Developing a new, standalone, branded agent product or single-binary harness. |
| **Language / Runtime** | Agnostic (any agent that speaks HTTP or MCP). | Go (imports `pkg/harnesskit`). |
| **Process Model** | Supervised child process behind an intercepting loopback proxy. | Single monolithic binary embedding the agent loop. |
| **Setup Overhead** | Zero code changes: 1 flag or environment variable repoint. | Go project initialization (`fak harness init`), compilation, and typed configuration. |
| **Tool Execution** | Intercepted at HTTP wire or MCP stdio/SSE socket boundary. | Directly registered and dispatched in-process via Go interfaces. |
| **Latency Overhead** | ~0.5–2ms per request (loopback socket + JSON parsing + policy check). | <10µs (direct in-memory function call). |
| **Security Enforcement** | External default-deny floor; child process cannot bypass proxy. | Internal API contracts; builder controls policy and extension registration. |
| **Audit Capabilities** | Tamper-evident, hash-chained JSONL decision journal (`--audit`). | Telemetry streams emitted via `EventStream` / `StreamObserver`. |
| **Prompt Caching** | Automated prefix stabilization and tool output compaction. | Builder explicitly manages system prompt and context planes. |
| **Resilience / Watchdog** | In-flight process auto-healing, terminal recovery, account rotation. | Application must handle its own process lifecycles and recovery. |

---

## 6. Architectural Decision Tree for Builders

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

## 7. Public vs. Private Boundary Considerations

When deploying custom harnesses across the boundary between **public `fak`** (open-core runtime) and **private `fak-private`** (proprietary enterprise serving and autonomous development factory):

1. **Public Harnesses:**
   - Must import only exported packages under `pkg/` (such as `pkg/harnesskit`, `pkg/abi`, `pkg/fakclient`).
   - Use `fak guard` with public-safe capability policies (`examples/customer-support-readonly-policy.json`) that forbid out-of-tree writes and dangerous system calls.
2. **Private Factory Harnesses (`fak-private`):**
   - High-throughput factory workers in `platform/dispatch/` bypass `fak guard` interactive wrappers to prevent socket exhaustion and latency spikes during bulk issue resolution waves.
   - Private factory code consumes public contracts via `go.work` and `pkg/harnesskit/lockv2` without importing `internal/*`.
   - Secret-bearing environments enforce secret quarantine: `fak guard`'s external redaction and leak-scan filters are used at perimeter boundaries, while internal dispatch loops rely on verified lockfile hashes.

---

## 8. Related Artifacts & References

- **Issue #11610:** Architectural evaluation of when `fak guard` is useful vs native `harnesskit`.
- **Issue #11611:** Public-harness vs private-factory contract and isolation rules across `fak` and `fak-private`.
- **Public SDK Contract:** [`docs/harness-kit-contract.md`](../harness-kit-contract.md)
- **Harness Generator Guide:** [`docs/harness-init.md`](../harness-init.md)
- **Supported Harnesses Matrix:** [`docs/supported/agent-harnesses.md`](../supported/agent-harnesses.md)
- **Dev-Process Private Boundary:** [`docs/dev-process-private-boundary.md`](../dev-process-private-boundary.md)
