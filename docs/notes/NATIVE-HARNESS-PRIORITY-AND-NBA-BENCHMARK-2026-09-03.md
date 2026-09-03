# Strategic Priority: Native fak Harness & Next-Best Alternative (NBA) Benchmarking

**Status:** ACTIVE SPINE / ARCHITECTURAL DOCTRINE  
**Date:** 2026-09-03  
**Problem Centrality:** Core (P1: Managed Context, P2: Net-True Efficiency, P3: Bounded Adaptation, P4: Integrated Operations)  
**Authority:** Issue #10720, Epics #1315, #6552, #10177  

---

## 1. Executive Summary & The Strategic Shift

During earlier development, `fak` focused heavily on managing, wrapping, and fronting external agent harnesses—principally Claude Code, OpenAI Codex, OpenCode, Cursor, and Aider—via the proxy gateway path (`fak manage`, `fak serve`, MCP, base-URL repointing). While this gateway path was essential for initial adoption and allowed zero-code-change capability floor enforcement, it created structural constraints:

1. **Proxy Ceiling:** On a proxy path, `fak` intercepts requests over the network or stdio wire, but the foreign harness owns the turn loop, prompt planning, speculative tool execution, and transcript persistence. On this path, `fak` *cannot* synthesize tool results before execution (vDSO caching), cannot place a proactive write barrier before consumption, cannot repair malformed tool arguments without round-trip model retries, and cannot pool multi-agent memory in a single process.
2. **Tool-Dialect Tax:** Significant engineering effort was expended resolving foreign harness dialect friction (e.g. Codex `snake_case` with `functions.*` prefixes, Claude `PascalCase`, OpenCode lowercase with `camelCase` args, differing auto-continue behaviors on tool denials).
3. **Fleet Bloat:** Co-hosting 20 concurrent headless agent workers using external Node/Electron/Python runtimes consumed ~12.9 GiB of host RAM (~600 MiB/seat) with duplicate MCP processes.

### The Decision

**We prioritize the native `fak` harness (`fak agent --native`, `internal/agent`, `pkg/harnesskit`) for our mainstream development and fleet execution.** Mainstream development shifts to leaning fully into our architectural kernel advantages.

Simultaneously, we maintain a **low-ego, market-realistic posture**:
- The market for external harnesses (OpenCode, OpenAI Codex, Cursor, Claude Code) is massive, well-resourced, and will remain massive. We do not pretend we replaced them or carry hubris.
- We maintain an **always-learning, field-borrowing posture**: continuously studying and adopting the best UX patterns, tool-calling idioms, prompt structures, and agent skills from the ecosystem.
- External harnesses remain **supported up to a point**: `fak serve` (OpenAI Chat Completions, Anthropic Messages, OpenAI Responses), MCP (`/mcp` and `--stdio`), and `fak manage` proxy gates remain tier-1 supported integration paths for users who prefer them.
- Features proven in the native harness are systematically evaluated for **upstream adaptation** back to external harnesses via MCP, hooks, or proxy transformations.
- We **always benchmark against the next-best alternative (NBA)** (or better) with rigorous 4-arm matched-envelope comparisons. We never claim superiority from an unmeasured baseline or strawman.

---

## 2. Low-Ego Foundation: Market Realism & Continuous Learning

| Principle | Meaning in Practice | Anti-Pattern Rejected |
|---|---|---|
| **Market Realism** | External harnesses have massive developer bases and corporate backing. They are the default ecosystem. | Believing or claiming that "fak replaces the market" or that developers must abandon existing tools. |
| **Always Learning (Field-Borrowing)** | Proactively study upstream harnesses (`/study-repo`, `/field-borrow`, `/study-pr-queue`). Borrow proven patterns with provenance. | Not-invented-here syndrome or dismissing external UX/tooling advancements. |
| **Tier-1 Support Up to the Seam** | Maintain first-class compatibility guides, dialect profiles, and proxy gates for Claude Code, Codex, OpenCode, and Cursor. | Breaking or deprecating proxy gates to force users onto the native harness. |
| **NBA Benchmarking Invariant** | Evaluate every performance or efficiency claim against the tuned next-best alternative on identical workloads under matched envelopes. | Comparing against naive baselines, untuned defaults, or strawman configurations. |
| **Pioneered Native, Adapted Outward** | Build capabilities natively first to exploit kernel depth; then export clean subsets outward via MCP, CLI, or gateway adapters. | Keeping innovations locked inside proprietary seams with no path for external harness users. |

### What We Learn From Each Harness (The Field-Borrowing Roster)

1. **OpenCode:**
   - *Strengths borrowed:* Ultra-clean CLI/terminal ergonomics, open-source transparency, straightforward provider configuration, low ceremony tool loops.
   - *Seams integrated:* Lowercase tool naming conventions, standard OpenAI Chat Completions repointing, local config discovery.
2. **OpenAI Codex:**
   - *Strengths borrowed:* Deep reasoning integration, structured plan management (`update_plan`), clean responses-wire lifecycle, ephemeral execution containment.
   - *Seams integrated:* `fak codex` dedicated launcher, responses-wire emulation in `fak serve`, MCP server bridge.
3. **Cursor:**
   - *Strengths borrowed:* Fluid diff previewing and inline file application, rich workspace contextual indexing, background semantic search, operator interruptibility.
   - *Seams integrated:* Fast MCP tool server integration (`fak serve --stdio`), trajectory disk offload conventions.
4. **Claude Code:**
   - *Strengths borrowed:* Bounded task decomposition (`Task` / `TodoWrite`), hierarchical file conventions (`CLAUDE.md`), auto-compact hooks (`PreCompact`), cost and token transparency.
   - *Seams integrated:* Anthropic Messages wire adapter, session OAuth token discovery, auto-compaction and Stop hook integration in `fak manage`.

---

## 3. Leaning into Kernel Advantages: Why Native for Mainstream Dev

When `fak` owns the agent loop (`RunArm` in `internal/agent`), it accesses capabilities that are physically impossible on a foreign-runtime proxy:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   NATIVE FAK HARNESS                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              Host Turn Loop (RunArm)                              │  │
│  │   • Context Assembly      • In-Kernel Planner / Model     • Speculative Lookahead│  │
│  └────────────────────────────────────────┬─────────────────────────────────────────┘  │
│                                           │                                            │
│                                  k.Syscall Boundary                                    │
│                                           │                                            │
│  ┌────────────────────┬───────────────────┴──────────────────┬──────────────────────┐  │
│  │  In-Kernel vDSO    │ Proactive Write   │ In-Syscall       │ Confinement &        │  │
│  │  Tool Caching      │ Barrier (barWrite)│ Grammar Repair   │ Zero-Daemon WAL      │  │
│  │  (Sub-µs repeat    │ (Squashed spec-   │ (Normalize args  │ (Append-only state,  │  │
│  │   reads served)    │  ulation halted)  │  w/o model turn) │  zero DB processes)  │  │
│  └────────────────────┴───────────────────┴──────────────────┴──────────────────────┘  │
│                                           │                                            │
│  ┌────────────────────────────────────────┴─────────────────────────────────────────┐  │
│  │             Single-Process Multi-Worker Co-Hosting (Shared Tool Catalog)         │  │
│  │             20 workers in < 500 MiB RAM vs 12.9 GiB across external runtimes      │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **In-Kernel vDSO Tool Caching (`internal/vdso`):**
   Idempotent reads (`Read`, `Grep`, `Glob`, unchanged file stat) are resolved in sub-microsecond time directly from kernel memory. In external harnesses, every read spawns a subprocess or makes an RPC hop, burning system CPU and model turns.
2. **Proactive Write Barrier Before Consumption (`speculation.barWrite`):**
   When speculative reads diverge or are squashed, the kernel strictly bars follow-on mutations from reaching the filesystem. External harnesses reactively execute writes and then attempt cleanup after damage is done.
3. **In-Syscall Grammar and Alias Repair (`abi.VerdictTransform`):**
   Model parameter hallucinations or minor schema aliases are transformed in-flight during the syscall. This eliminates wasted "model apologies and retries" turns that plague external harnesses.
4. **Speculative Lookahead & Turn Promotion:**
   The native loop predicts the next effect-free call, executes it provisionally during model generation, and promotes the result immediately if the model's call matches.
5. **Fleet Resource Pooling (Epic #6552):**
   Co-hosts 20+ native worker arms within a single Go process sharing a single loaded tool catalog and memory pool. This cuts memory from ~600 MiB/seat to <25 MiB/seat, eliminating process exhaustion on developer machines.
6. **Zero-Daemon Crash-Consistent State (`sessionjournal.jsonl`, `PendingTurn`):**
   State is maintained via append-only, fsynced write-ahead logs. No PostgreSQL, SQLite locks, or resident daemon processes that can orphan or desynchronize.
7. **Unified Memory & KV Prefix Sharing (`ctxmmu` / `radixkv`):**
   Zero-copy pass-by-reference (`abi.Ref`) keeps multi-megabyte tool outputs from thrashing heap or wire memory, preserving byte-exact provider prompt cache prefixes across turns.

---

## 4. Feature Upstream Adaptation Matrix

When a capability is pioneered in the native harness, we systematically categorize its adaptability:

| Capability | Native Implementation | Upstream Adaptability | Supported Integration Mechanism |
|---|---|---|---|
| **Session Budgets & Pacing** | `internal/agent/session_gate.go` | **Adaptable** | MCP middleware, `pkg/harnesskit`, Gateway preflight |
| **Policy Capability Floor** | `internal/adjudicator/decide.go` | **Adaptable** | `fak serve` proxy, `fak preflight`, MCP endpoint |
| **Context Shedding & Byte Splicing** | `internal/headroom/native.go` | **Adaptable** | Gateway reverse proxy (Anthropic/OpenAI wire) |
| **Trajectory & Audit Logging** | `internal/trajectory` | **Adaptable** | Gateway hash-chained decision journal |
| **Confinement & Concurrency Lib** | `internal/codetools` | **Adaptable** | Go library in `pkg/harnesskit`, standalone CLI tools |
| **In-Kernel vDSO Read Cache** | `internal/vdso` | **Native-Only** | Requires in-syscall interception before engine dispatch |
| **Proactive Pre-Consumption Write Barrier** | `internal/agent/loop_turn.go` | **Native-Only** | Requires turn speculation control and epoch barriers |
| **In-Syscall Grammar Repair** | `internal/agent/loop_turn.go` | **Native-Only** | External SDKs control raw argument deserialization |
| **Single-Process Worker Co-Hosting** | `internal/agent` + `cmd/fak` | **Native-Only** | External harnesses require distinct OS processes |
| **Zero-Copy ABI Reference MMU** | `internal/ctxmmu` + `abi.Ref` | **Native-Only** | Crosses language/runtime boundaries on external tools |

---

## 5. 4-Arm NBA Comparative Benchmarking Specification

In accordance with `docs/benchmarks/NATIVE-IMPLEMENTATION-COMPARISONS.md`, the native harness loop declares an explicit 4-arm contract:

### The 4 Arms

1. **Arm 1: `fak-native` (Candidate Arm):**
   - Native Go agent loop (`fak agent --native`, `internal/agent`).
   - In-kernel vDSO tool cache, write barrier, in-syscall grammar repair, shared memory refs.
2. **Arm 2: `Tuned Baseline` (OpenCode CLI):**
   - OpenCode terminal agent executing the identical repository task with direct tool calling over identical model weights.
   - Represents the strongest open, unmediated terminal harness.
3. **Arm 3: `Next-Best Alternative` (Codex CLI / Cursor Agent / Claude Code):**
   - Top-tier proprietary/frontier harnesses:
     - OpenAI Codex (`codex-cli` on OpenAI Responses wire).
     - Cursor Agent (Composer with local workspace indexing).
     - Claude Code (`claude-code` on Anthropic Messages wire).
4. **Arm 4: `First-Class Integration` (`fak + Harness` Proxy Gate):**
   - `fak manage -- <harness>` or external harness pointed at `fak serve`.
   - Isolates the exact contribution of the capability floor and gateway caching from the native loop mechanics.

### The 6 Evaluation Dimensions & Required Metrics

```
┌─────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Dimension               │ Concrete Metrics                                            │
├─────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Turn & Step          │ • task_success_rate (0.0–1.0 on identical task fixtures)    │
│    Efficiency           │ • turns_to_completion (fewer turns = higher efficiency)     │
│                         │ • turns_saved_by_repair (count of avoided retry turns)      │
├─────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. Tool Call Avoidance  │ • tool_calls vs engine_calls (execution density)            │
│    & Execution Density  │ • vdso_cache_hits (sub-µs read hits)                        │
│                         │ • writes_barred (prevented invalid/speculative mutations)   │
├─────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Token & Cache        │ • prompt_tokens & completion_tokens                         │
│    Economics            │ • tokens_saved (net reduction vs baseline)                  │
│                         │ • provider_cache_hit_rate (prefix preservation %)           │
├─────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4. Wall-Clock & OS      │ • wall_clock_ms (end-to-end task completion time)           │
│    Resource Overhead    │ • rss_per_session_mb (resident memory footprint)           │
│                         │ • process_spawns (total OS subprocesses executed)           │
├─────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5. Safety, Confinement  │ • injection_in_context (tainted tool results to model: 0)   │
│    & Gate Adherence     │ • destructive_executed (unadjudicated destructive actions: 0)│
│                         │ • confinement_escapes (path traversal allowed: 0)           │
├─────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 6. Crash Resilience &   │ • interrupted_turn_recovery (resumption without turn-0 reset)│
│    Witness Honesty      │ • witness_verification_rate (required proof before done: 100%)│
└─────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 6. Migration and Operational Roadmap

1. **Phase 1: Tool Surface Parity (Completed):**
   `internal/agent/codetools.go` provides standard coding primitives (`Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob`) under strict canonical root confinement and optimistic versioning.
2. **Phase 2: Comparative Tooling & Contract Registration (This Change):**
   - Register `agent_harness_loop` contract in `internal/nativebench/nativebench.go` and `docs/benchmarks/NATIVE-IMPLEMENTATION-COMPARISONS.md`.
   - Ship `fak harness compare` CLI verb for immediate, transparent comparison across architectural dimensions, adaptation status, and NBA metrics.
3. **Phase 3: Internal Dogfooding Priority:**
   - Autonomous background workers and super-loop instances default to `fak agent --native`.
   - Maintainer tasks lean into native harness workflows while validating that proxy configurations continue to pass all conformance suites.
4. **Phase 4: Paired Empirical Evidence Capture:**
   - Run counterbalanced, matched-envelope benchmark tasks comparing `fak-native` against OpenCode, Codex, Cursor, and Claude Code on representative repository problems.
   - Archive immutable receipts under `docs/witnesses/` and update `BENCHMARK-AUTHORITY.md`.
