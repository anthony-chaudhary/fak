---
title: "Goal: Less comment verbosity by default — a Ponytail-inspired comment restraint ladder"
description: "Establishing less comment verbosity as a first-class default: why model-generated comment bloat hurts context and cache economics, the Ponytail-inspired ladder of comment restraint, and the non-negotiable correctness and safety carve-outs."
date: 2026-09-03
---

# Goal: Less comment verbosity by default — a Ponytail-inspired comment restraint ladder

> **Goal statement.** Establish **less comment verbosity by default** across code generated,
> edited, or maintained in the repository. Apply Ponytail's core philosophy of simplicity,
> restraint, and explicit safety carve-outs to code comments: self-documenting code over
> syntax narration, document the non-obvious *why* rather than the obvious *what*, and eliminate
> conversational or tautological comment bloat before it enters the working tree.

---

## 1. Value frame

- **For:** autonomous coding agents, human maintainers, and harness operators working in
  and around FAK.
- **Problem:** language models have a persistent default bias toward excessive, low-information
  code comments (syntax narration, tautological docstrings, apologetic commentary, decorative
  banners). In long-running agent loops, this bloat consumes generation output tokens, pollutes
  future context windows on every subsequent tool read, degrades prompt cache token density,
  accelerates comment rot during refactors, and buries critical invariants in trivia.
- **Today:** `AGENTS.md` mandates *"Keep comments durable. Explain non-obvious invariants,
  safety, concurrency... do not narrate syntax."* `.github/copilot-instructions.md` asks for
  succinct comments, and `tools/code_slop_scorecard.py` checks `comment_slop` (tautological
  docstrings and commented-out code blocks). However, there is no explicit, unified *restraint
  ladder* or default-minimal posture codified as a named goal.
- **Better because:** adapting Ponytail's simplicity ladder (`DietrichGebert/ponytail`) to code
  comments establishes an auditable hierarchy of restraint: self-documenting code first (Rung 0),
  concise exported API contracts when non-obvious (Rung 1), durable invariants and platform quirks
  (Rung 2), with explicit carve-outs for machine directives, legal notices, and concurrency
  safety.
- **Witness:** clean diffs with minimal comment lines; green `tools/code_slop_scorecard.py`;
  zero loss of required godoc coverage or safety-critical invariant explanations.

Problem centrality is **Stewardship** and **Enabling**. It protects context window budgets (P1),
delivers net-true token savings (P2), establishes bounded adaptation (P3), and integrates with
existing repository linters and scorecards (P4).

---

## 2. The defect: why LLM comment verbosity hurts

In human software engineering, over-commenting is a mild code smell. In autonomous agent
systems, **comment verbosity is an active operational tax**:

1. **Generation token tax (turn latency and cost):**
   Generating 30 lines of comments to accompany 15 lines of code doubles output tokens,
   increasing turn latency, cost, and time-to-first-token/completion across every turn.
2. **Context window colonization (future turns):**
   Once committed to the working tree, verbose comments persist. Every subsequent agent that
   reads the file via `Read`, `Grep`, or `FileEdit` must re-ingest those tokens into its
   prompt context. Across dozens of turns and sessions, comment bloat wastes thousands of
   tokens of scarce context window.
3. **Prompt-cache and KV-cache dilution:**
   Long-context caching relies on high semantic token density. Diluting code with repetitive
   prose degrades cache efficiency and increases compaction/eviction pressure.
4. **Semantic rot and agent confusion:**
   Code evolves rapidly during autonomous development. Comments that narrate syntax or
   state ("what the code does") desynchronize from logic within a few commits. Future agents
   read stale comments as authoritative intent, leading to hallucinated bugs or flawed refactors.
5. **Signal-to-noise inversion:**
   When every standard conditional or loop has an explanatory comment, real, load-bearing
   invariants (e.g., lock ordering, memory barriers, Windows Defender AV quarantine workarounds)
   become invisible.

### Common manifestations of comment bloat

- **Syntax narration:** `// loop through items and check if valid` above `for _, item := range items`.
- **Tautological doc comments:** `// Server represents the server` above `type Server struct`.
- **Conversational / apologetic artifacts:** `// As requested, we add an error check here` or
  `// Note: this implementation handles edge cases carefully`.
- **Visual banners / decorative dividers:** `// ==================== UTILITY FUNCTIONS ====================`.
- **Speculative / orphaned task markers:** speculative future notes without an issue anchor (`we could consider caching this later`).

---

## 3. Inspiration from Ponytail: restraint as a default virtue

Ponytail ([`DietrichGebert/ponytail@2ed6c52c9d7e5e56942508591085fd45dea277d3`](https://github.com/DietrichGebert/ponytail/tree/2ed6c52c9d7e5e56942508591085fd45dea277d3),
studied in [`CONCEPT-STUDY-PONYTAIL-2026-08-13.md`](CONCEPT-STUDY-PONYTAIL-2026-08-13.md))
introduced the "laziest senior dev" and "code you never wrote" paradigm for AI coding agents.
Its native adaptation in FAK (`internal/syspromptmmu`, `WorkProfilePonytailNativeMed`) operates
as FAK's default work profile:

> *"Challenge unnecessary additions. In order, consider: no code change, deletion, configuration,
> existing project primitives, standard library, then new machinery. Stop at the first option
> that completely and correctly satisfies the task... Aggressiveness applies only to avoidable
> complexity."*

Translating this philosophy to code comments yields a direct parallel:

| Ponytail code principle | Ponytail comment restraint analog |
|---|---|
| The best code is code you didn't have to write | The best comment is the one you didn't have to write because the code is self-explanatory |
| Prefer deletion and reuse over addition | Prefer clear naming and standard idioms over explanatory commentary |
| Stop at the simplest rung that satisfies the task | Stop at Rung 0 (zero comments) unless an invariant or contract demands higher |
| Aggressiveness applies only to avoidable complexity | Omit syntax narration; never omit critical safety/concurrency invariants |
| Explicit correctness and safety carve-outs | Explicit carve-outs for directives, licenses, and non-obvious invariants |

---

## 4. The comment restraint ladder

Code comments in FAK must follow an explicit, ordered restraint ladder:

```text
┌─────────────────────────────────────────────────────────────┐
│ Rung 2: Invariants, "Why", & Platform Quirks (High-Value)   │  Explain non-obvious rationale
├─────────────────────────────────────────────────────────────┤
│ Rung 1: Exported API Contracts (Minimal & Non-Tautological) │  Contract/preconditions only
├─────────────────────────────────────────────────────────────┤
│ Rung 0: Self-Documenting Code (DEFAULT — Zero Comments)     │  Clean naming, idiomatic structure
└─────────────────────────────────────────────────────────────┘
  ▲
  │  (Below the line: PROHIBITED)
┌─────────────────────────────────────────────────────────────┐
│ Prohibited: Syntax Narration, Chatty Fillers, Banner Cruft  │  Refused by style & linters
└─────────────────────────────────────────────────────────────┘
```

### Rung 0: Self-documenting code (default — zero comments)
- **Rule:** If code can be made clear through better variable naming, function decomposition,
  or standard idioms, refactor the code; do not add a comment.
- Standard conditionals, loops, error checks (`if err != nil`), return statements, and
  straightforward assignments must have **zero comments**.
- The presence of an inline comment explaining *what* a 3-line block does is an indicator that
  the code itself lacks clarity.

### Rung 1: Exported API contracts (minimal & non-tautological)
- **Rule:** Document exported symbols (package, type, func, const) where required by godoc or
  public API consumers, but focus strictly on the *contract*, preconditions, postconditions, or
  caller responsibilities that cannot be inferred from the signature alone.
- **Anti-tautology filter:** Never repeat the symbol name without adding information. If a comment
  like `// ProcessQueue processes the queue` adds no semantic value beyond `func ProcessQueue()`,
  it must be omitted or rewritten to state what guarantees or side-effects apply.

### Rung 2: Invariants, "Why", & Platform Quirks (the high-value rung)
- **Rule:** Inline comments are reserved exclusively for explaining the non-obvious **WHY**,
  never the obvious **WHAT**.
- Acceptable Rung 2 triggers:
  1. **Concurrency and memory invariants:** lock ordering, channel ownership, atomic/barrier
     semantics, or goroutine lifecycle constraints where omission risks deadlock or data race.
  2. **Platform and OS quirks:** specific operational fences (e.g., Windows Defender ML
     quarantine of transient test binaries, WSL filesystem boundaries, host-specific syscall limits).
  3. **Non-obvious bug workarounds:** subtle regression fixes where the intuitive or naive
     approach would reintroduce the bug.
  4. **Frozen ABI / wire stability:** explicit callouts where struct layouts or fields are
     frozen for backward compatibility (`internal/abi`).
  5. **SOTA / literature citations:** references to specific RFCs, papers, or upstream commits
     explaining the mathematical formula or protocol being implemented.

### Prohibited: comment slop
- Line-by-line syntax narration ("narrating the code as you go").
- Conversational or apologetic phrases ("Per the user's instruction...", "We carefully check...").
- ASCII dividers, decorative separator lines, or section headers.
- Commented-out dead code blocks (policed by `tools/code_slop_scorecard.py`).
- Orphaned or speculative future-work comments without an accompanying issue or PR reference.

---

## 5. Correctness and tooling carve-outs

Just as Ponytail strictly forbids narrowing requested scope or weakening security,
comment restraint **never compromises required tooling or safety documentation**:

1. **Toolchain and compiler directives:**
   Directives such as `//go:build`, `//go:generate`, `//nolint`, `//export`, or `/* #cgo */`
   are structural instructions for the compiler, not commentary. They are preserved intact.
2. **Legal and licensing notices:**
   Standard SPDX identifiers and Apache-2.0 copyright notices (`// SPDX-License-Identifier: Apache-2.0`)
   are required legal provenance and must not be altered or removed.
3. **Automated verification anchors:**
   Markers consumed by repository tooling (e.g., `<!-- readme-verified: ... -->`,
   `<!-- code-quality-scorecard: ... -->`, or `// +k8s:deepcopy-gen`) are structural data.
4. **Safety-critical invariants:**
   When removing a comment would make subtle concurrent code incomprehensible to the next
   maintainer or agent, that comment is a load-bearing Rung 2 invariant and must stay.

---

## 6. P1-P4 problem checklist alignment

- **P1 — Managed context:** Prevents working-tree files from accumulating redundant prose,
  ensuring that context injected into future agent turns has high semantic density and minimal
  waste.
- **P2 — Net-true efficiency:** Reduces output token generation costs and latency on every turn,
  while lowering long-term prompt cache footprint across repeated turns and sessions.
- **P3 — Fast, bounded adaptation:** Provides an unambiguous ladder of restraint that agents
  can apply across multiple languages (Go, Python, TypeScript, Shell) without guessing.
- **P4 — Integrated operations:** Enforceable through existing linters, commit preview gates
  (`fak commit --preview`), and automated scorecards (`tools/code_slop_scorecard.py`).

---

## 7. Operational application across FAK

1. **Contributor instructions:**
   Reiterate the restraint ladder in `AGENTS.md` and harness instructions so all automated
   workers and human contributors default to minimal comments.
2. **Harness & work profiles:**
   Align agent work profiles (`internal/syspromptmmu/workprofile.go`) with comment restraint
   as a natural companion to `ponytail:native:medium` implementation simplicity.
3. **Automated gates:**
   Maintain and expand `tools/code_slop_scorecard.py` (`kpi_comment_slop`) to continuously
   catch tautological comments, dead code, and syntax narration before landing on `main`.
