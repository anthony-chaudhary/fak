# ADR: Composable System Prompt Algebra and Prefix-Caching Geometry

- **Status:** Approved / RFC Reference
- **Date:** 2026-09-03
- **Issue:** #10903 (Parent Epic: #10902)
- **Centrality:** Core
- **Affected Packages:** `internal/promptcomp`, `internal/harnessinit`, `internal/promptmmu`, `internal/ctxmmu`, `internal/radixkv`

---

## 1. Problem Framing & Motivation

- **For:** Developers, multi-agent framework authors, and operators self-hosting coding and reasoning models (Qwen3.8, Qwen2.5-Coder 7B/14B/32B, DeepSeek) with agent harnesses (Claude Code, OpenCode, Cline).
- **Problem:** Current agent harnesses dump 35,000 to 45,000 tokens of monolithic system prompts, formatting rules, and 70+ verbose tool schemas into turn 0. While cloud frontier providers absorb this via elastic clusters and proprietary prompt caching, local self-hosted deployments suffer severe failures:
  1. **Massive KV-Cache VRAM Footprint:** ~11.5 GB allocated on 32B models for the prompt alone before the first turn begins, preventing concurrent batching.
  2. **Prefill Latency (TTFT):** 20–50s prefill delays on workstation hardware (RTX 4090, Apple Silicon M4 Max, L4) every turn prefix cache misses.
  3. **Context Exhaustion:** Rapidly breaches 32k/64k token limits in multi-agent fan-out loops (e.g. OpenCode 1.18 429 errors documented in #10597).
  4. **Attention Dilution ("Lost in the Middle"):** 7B and 14B models experience instruction-following collapse and tool hallucinations under 40k-token reasoning scaffolds.
- **Today:** `fak` provides `defer-cold-tools` (Anthropic wire only) and `promptmmu` tool pruning. However, local OpenAI `/v1/chat/completions` and in-kernel GGUF wire paths still serialize monolithic prompts and entire tool catalogs verbatim.
- **Better Because:** Implements a composable prompt architecture where the system prompt is a compiled projection of small, content-addressed, versioned fragments:
  $$\text{SystemPrompt} = \text{Compile}(\mathcal{F}, \text{ModelProfile}, \text{AgentTier}, \text{ContextBudget})$$
  guaranteeing byte-level prefix stability for 100% KV cache hit rates on tokens $0 \dots K$, dynamic tool thinning, and scale-specialized contracts.
- **Witness:** Deterministic topological resolution tests in `internal/promptcomp/*_test.go` proving acyclic ordering, cache breakpoint immutability, and 1,000-iteration byte-exact reproducibility.

---

## 2. Formal Prompt Algebra

### 2.1 The Atomic Fragment ($\text{PromptPart}$)

Let $\mathcal{F} = \{ f_1, f_2, \dots, f_m \}$ be the closed universe of registered prompt fragments. An atomic fragment $f \in \mathcal{F}$ is a 7-tuple:

$$f = \langle \text{ID}, \text{Digest}, \text{Content}, \text{Kind}, \text{Rank}, \mathcal{D}, \mathcal{C}, \mathcal{P} \rangle$$

Where:
- **$\text{ID} \in \Sigma^*$:** Canonical kebab-case identifier (e.g., `spine.fak-core`, `policy.floor`, `contract.leaf-s1`).
- **$\text{Digest} \in \{0..9, a..f\}^{64}$:** Hex-encoded SHA-256 digest of the canonical UTF-8 bytes of $\text{Content}$.
- **$\text{Content} \in \text{UTF-8}^*$:** Raw immutable fragment text without extraneous markdown whitespace.
- **$\text{Kind} \in \{ \text{KindSpine}, \text{KindPolicy}, \text{KindContract}, \text{KindTools}, \text{KindOverlay} \}$:** Segment tier defining topological placement relative to the cache breakpoint.
- **$\text{Rank} \in \mathbb{Z}$:** Secondary stable sorting tie-breaker within the same `Kind`.
- **$\mathcal{D} \subseteq \Sigma^*$:** Dependency set; IDs of prerequisites that must be admitted before $f$.
- **$\mathcal{C} \subseteq \Sigma^*$:** Conflict set; IDs of mutually exclusive fragments that cannot co-exist with $f$.
- **$\mathcal{P}: \text{Env} \to \{\text{true}, \text{false}\}$:** Runtime predicate evaluating environment variables (target model family, agent tier, remaining context tokens, wire format).

### 2.2 Segment Kinds and Cache Geometry

The system prompt segments are partitioned into two distinct physical zones across a fixed **Cache Breakpoint** $K$:

```
┌─────────────────────────────────────────────────────────────┐
│ ZONE 1: WARM IMMUTABLE PREFIX (Tokens 0 .. K)               │
│ - KindSpine: Root identity & universal attention anchors     │
│ - KindPolicy: Invariable safety, license, and tool floor     │
├─────────────────────────────────────────────────────────────┤  <--- CACHE BREAKPOINT (K)
│ ZONE 2: DYNAMIC EPHEMERAL OVERLAY (Tokens K+1 .. N)         │
│ - KindContract: Role-specialized contract (Coordinator/Leaf)│
│ - KindTools: Thin active tool schemas & discovery grammar   │
│ - KindOverlay: Paged working-set rules and capability cards │
└─────────────────────────────────────────────────────────────┘
```

#### Theorem 1 (Prefix Cache Invariance):
Let $S_A$ and $S_B$ be two system prompts compiled for arbitrary agents $A$ and $B$ under the same deployment root $\mathcal{F}$ with identical Zone 1 admissions. Then:
$$\text{Tokens}(S_A)[0..K] \equiv \text{Tokens}(S_B)[0..K]$$
Therefore, the physical KV-cache page blocks covering $[0..K]$ in any radix-tree KV store (`radixkv`, vLLM, SGLang) are $100\%$ shared and require $0$ prefill compute across all fan-out workers.

---

## 3. Inspirations & External System Analysis

### 3.1 Aider Borrow: Model-Specific Prompt Contracts
- **Observation:** Aider demonstrates that 7B–14B coding models (such as Qwen2.5-Coder-7B or DeepSeek-Coder-6.7B) suffer severe regression when prompted with elaborate chain-of-thought scaffolds or multi-turn conversational preambles. They perform highest with concise, imperative, unified-diff or edit-block grammar. Conversely, 70B+ frontier models thrive on explicit architectural guidelines.
- **Application in fak:**
  - When `Model.IsSmallLocal()` is true, `KindContract` selects ultra-concise imperative directives ("Read target file before Edit; verify via unit test; emit zero commentary").
  - Reasoning scaffolds and conversational framing are pruned before compilation.

### 3.2 Cursor Borrow: Paged Rules & Shadow Context
- **Observation:** Cursor avoids injecting entire multi-kilobyte `.cursorrules` or documentation trees into the system prompt. Instead, it maintains a lightweight index and dynamically faults rule cards into context only when referenced files, globs, or tool calls activate them.
- **Application in fak:**
  - `KindOverlay` treats `AGENTS.md`, `.cursorrules`, and specialized skill instructions as discrete, content-addressed cards ($\le 350$ tokens each).
  - A 150-token trigger summary is placed in Zone 2; full rule cards are paged into context on-demand via `internal/ctxmmu` and evicted when context budgets tighten.

---

## 4. Tiered Agent Prompt Taxonomy

The system defines three distinct operational tiers for autonomous agents:

| Metric | Tier 1: Massive Coordinator | Tier 2: S0/S1 Leaf Worker | Tier 3: Micro Validator |
|---|---|---|---|
| **Role** | Task decomposition, wave launch, receipt folding | Single observable deliverable, 1-3 files, 1 witness | Predicate evaluation, sanity checks, lint grading |
| **Active Tools** | Full catalog / delegation / tool search | 3–4 basic tools (`Read`, `Edit`, `Bash`, `Glob`) | Zero tools (pure text/JSON output) |
| **Contract Style** | Orchestration protocol, collision pricing, rollback | Direct, concise, imperative; zero preamble | Strict JSON schema or binary single-token verdict |
| **Prompt Footprint** | 2,500 – 4,000 tokens | 600 – 800 tokens | 120 – 200 tokens |
| **Turn 0 Prefill (L4)** | ~1.8s | ~0.15s | ~0.04s |
| **KV VRAM (32B FP16)** | ~1.1 GB | ~0.2 GB | ~0.05 GB |

---

## 5. Topological Graph Resolution & Compilation Pipeline

Compilation follows a deterministic four-stage algorithm:

1. **Admit & Filter:**
   Evaluate predicate $\mathcal{P}(f)(\text{Env})$ for all $f \in \mathcal{F}$. Filter admitted subset $\mathcal{F}_{\text{active}}$.
2. **Dependency & Conflict Validation:**
   - For every $f \in \mathcal{F}_{\text{active}}$, assert that $\forall d \in f.\mathcal{D}, d \in \mathcal{F}_{\text{active}}$. If missing, fail-closed with `ErrMissingDependency`.
   - For every $f \in \mathcal{F}_{\text{active}}$, assert that $\forall c \in f.\mathcal{C}, c \notin \mathcal{F}_{\text{active}}$. If present, fail-closed with `ErrConflictingFragments`.
3. **Topological Ordering:**
   Build a directed acyclic graph $G = (V, E)$ where edges represent dependencies. Order vertices using Kahn's algorithm or DFS. In the presence of equal dependency depths, break ties deterministically by `(Kind ASC, Rank ASC, ID ASC)`. Detect cycles and fail-closed with `ErrCyclicDependency`.
4. **Serialization & Breakpoint Partitioning:**
   Concatenate fragments into canonical UTF-8 bytes separated by single newlines `\n\n`. Record the byte and token offset of the boundary between Zone 1 (`KindSpine`, `KindPolicy`) and Zone 2 (`KindContract`, `KindTools`, `KindOverlay`).

---

## 6. Implementation Deliverables & Verification

- **Deliverable 1:** ADR document committed at `docs/research/composable-system-prompt-algebra-2026-09.md` (Issue #10903).
- **Deliverable 2:** Package `internal/promptcomp` implementing registry, dependency solver, and deterministic compiler (Issue #10904).
- **Deliverable 3:** Dynamic synthesizer in `internal/harnessinit` (Issue #10905).
- **Deliverable 4:** Local wire tool-schema thinning in `internal/promptmmu` (Issue #10906).
- **Verification Witness:**
  ```bash
  go test -v ./internal/promptcomp/...
  go test -v ./internal/harnessinit/...
  go test -v ./internal/promptmmu/...
  ```
