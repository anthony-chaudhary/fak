---
title: "Positive workspace management — construction over punitive default-deny"
description: "Why positive workspace construction outperforms punitive default-deny, the architectural division between the FROZEN safety floor and the permissive convenience surface, and how affirmative guidance prevents capability laundering and doom-loops."
---

# Positive workspace management — construction over punitive default-deny

> **Doctrine:** Construct and maintain an affirmative, highly capable workspace bounded by an immutable safety floor; never rely on punitive default-deny, negative scolding, or dead-end prohibitions to govern agent behavior.

Autonomous agents do not respond to punitive prohibitions the way human operators expect. When confronted with negative-only barriers ("DO NOT...", "Access Denied"), transformer-based models lack a persistent, dependable negation operator. Negative framing makes the forbidden action salient in context, inducing pathological compensation patterns. Positive workspace management resolves this by establishing an architectural bifurcation: an immutable, compiled-in safety floor that mechanically prevents existential harm, paired with an affirmative, permissive developer environment that leads with executable affordances and constructive next actions.

## The core doctrine

Punitive default-deny (refusing actions by default without constructive paths) treats an agent as an active adversary to be contained through blanket denials, prohibitive prompts, and opaque refusal walls. When an agent encounters punitive, dead-end prohibitions without constructive affordances or clear execution paths, it cannot stop gracefully; it pathologies.

Positive workspace construction instead shapes the working environment affirmatively:

1. **Enforce an immutable, un-bypassable structural floor** for existential risks (the compiled-in FROZEN floor).
2. **Construct open, affirmative affordances** within that safe boundary: provide the agent with the tools, paths, and positive target states needed to succeed.
3. **Supply an executable next action** with every refusal or constraint, turning "no" from a dead end into a verifiable transition.

## Pathologies of punitive over-restriction

Attempting to govern agent behavior through aggressive, punitive prohibitions creates three failure classes: capability laundering, doom-loops, and self-DoSes.

### 1. Capability laundering

When legitimate engineering actions—such as inspecting files, invoking build tools, running tests, or inspecting network interfaces—are blocked by broad or overzealous default-deny heuristics without functional alternatives, the agent does not abandon the operator's goal. Instead, it attempts roundabout workarounds:

- Encoding payloads (such as base64 or hex escaping) to evade pattern-matching filters.
- Obfuscating shell invocations through secondary interpreters (`python -c`, `sh -c`, `perl`).
- Staging writes into arbitrary temporary directories to skirt write restrictions.
- Splitting operations across multiple turns to bypass call-level inspectors.

The agent "launders" the capability through unmonitored channels. This behavior degrades system auditability, transforms clean, structured tool calls into opaque, hard-to-monitor operations, and increases runtime risk.

### 2. Doom-loops

When an agent receives a blunt, negative-only refusal ("Error: operation forbidden" or "Do not run X") without an executable alternative, the salient failure text dominates the context window.

Because autoregressive language models predict tokens from recent context, repeating the denial reinforces the prohibited concept in the model's working memory. Lacking a mechanical negation operator, the model enters a classic doom-loop:
- Apologizing and repeating the failed command with trivial phrasing changes.
- Oscillating between related forbidden tools.
- Generating speculative hallucinations to satisfy the prompt.

The agent burns effort (turns, tokens, and wall-clock time) while verified progress remains flat. This is the exact signature classified by [`internal/doomloop`](../internal/doomloop/doc.go): flat progress delta under burning effort.

### 3. Self-DoSes (Denial of Service)

Overly restrictive environments generate massive friction transcripts: repeated rejection receipts, defensive prompt residue, apologies, and recovery failures.

This overhead floods the context window, triggers premature context compaction, exhausts token budgets, and pushes the agent past turn limits. The agent effectively executes a self-DoS, starving itself of operational headroom before completing the task.

## The architectural division: FROZEN safety floor vs. permissive convenience

FAK resolves this tension by splitting policy into two distinct operational layers: an immutable, compiled-in safety floor and a permissive convenience surface.

| Dimension | The FROZEN core safety floor | The permissive convenience surface |
|---|---|---|
| **Location** | Kernel core / binary entry points | Development workspace / tool adapters |
| **Authority** | Compiled-in (`ChannelCompiledIn`), immutable | Dynamic, configurable by operator or task |
| **Amendment class** | `FROZEN` (zero-bypass, no channel weakens it) | `GATED_WIDEN` / `SELF_AMENDABLE` |
| **Enforcement style** | Mechanical, author-neutral, fail-closed | Affirmative guidance, relaxed tooling |
| **Scope** | Existential risks: SSRF, out-of-tree writes, secret leaks | Daily engineering: reads, edits, builds, tests, git |
| **Response on violation** | Closed typed refusal token + sanctioned recovery | Constructive affordance + positive target state |

### The FROZEN core safety floor

The FROZEN floor sits at the absolute bedrock of the policy precedence lattice (`compiled-in FROZEN floor > central org manifest > operator overlay > agent self-amendment`), as detailed in [the org-policy precedence doctrine](notes/RESEARCH-org-policy-precedence-2026-07-20.md). No policy channel—central org manifest, operator command-line flag, or agent prompt injection—can weaken or bypass it. It is author-neutral, fail-closed, and enforced at the syscall and kernel boundary.

The FROZEN floor governs three non-negotiable, existential blast-radius boundaries:

1. **SSRF and Network Egress:** Hard blocking of requests to cloud instance metadata services (`169.254.169.254`), private RFC1918 networks, loopback exploits, and unsanctioned external endpoints.
2. **Out-of-Tree Writes (`OUT_OF_TREE_WRITE`):** Absolute rejection of mutations targeting paths outside the declared workspace root or detached worker directory. Directory traversal attacks (`../`) and system file modifications are structurally refused.
3. **Credential and Secret Leaks:** Screening and zero-tolerance redaction/blocking of private API tokens, private cryptographic keys, lab GPU bridge credentials, and environment secrets before they leave the node or enter untrusted context.

Because these three invariants are enforced unconditionally by the kernel binary, the developer environment does not need to maintain redundant, speculative fences around them.

### The permissive convenience surface

Inside the boundary secured by the FROZEN floor, the agent's workspace must be maximally permissive, transparent, and collaborative:

- **Unrestricted development tools:** Standard operations—reading files, compiling code, executing test runners, inspecting Git status, and writing build artifacts within the workspace—run without friction.
- **Affirmative developer guidance:** Rather than burdening everyday engineering tools with speculative security heuristics that trip false positives on legitimate code changes, the runtime supplies clear, standard tools.
- **Assisting over policing:** The runtime treats the agent as an untrusted worker that requires clear scaffolding and open development tooling, not as a malicious adversary within its assigned workspace.

## Supporting mechanisms

Positive workspace management relies on three concrete runtime mechanisms in FAK:

### 1. `internal/negframe` and affordance-first emit seams

[`internal/negframe`](../internal/negframe/doc.go) implements fak-owned positive-state reframing, as specified in [Positive-state construction](positive-state-construction.md) and [Shared-workspace positive state](shared-workspace-positive-state.md).

At emit time (where `internal/gateway` assembles model-visible step advice and refusal notes), `negframe.Reframe` and `negframe.ReframeFakOnly` mechanically rewrite negative idioms ("do not commit directly", "never edit without reading") into positive target state directives ("use the commit lane", "read file before editing").

The transformation satisfies three safety invariants:
- **Token-superset fail-safe:** A candidate rewrite is applied only if every must-keep contract token (ALL-CAPS reason codes like `POLICY_BLOCK`, `OFF_TRUNK`, and backticked identifiers) survives byte-for-byte.
- **Conservative judgement tier:** Ambiguous or judgement-tier negations are left untouched rather than risk hallucinating an unsafe positive substitute.
- **Pure and deterministic:** Reframing involves no external network calls, model calls, or disk I/O.

### 2. Affirmative next actions (executable refusal contract)

In FAK, a refusal is never a dead end. Every guard refusal, preflight denial, or engine refusal carries:
1. A typed reason token from the closed vocabulary in `dos.toml [reasons.*]`.
2. An **affirmative, executable next action** that tells the agent exactly what command or operation to execute next.

When an agent encounters a refusal (such as `OFF_TRUNK` or `POLICY_BLOCK`), the output does not simply state "Command denied." Instead, it provides the exact executable recovery command:
- `dos man wedge <TOKEN> --explain`
- `fak recover <TOKEN>`
- An explicit alternate tool invocation (e.g., "Use `search_kb` with the customer reference").

This removes model guessing and breaks the retry loop before effort is wasted.

### 3. Phased TDD workflows (red-phase repro before green-phase impl)

Positive workspace management anchors agent execution to verifiable physical artifacts rather than narrative claims, following [Spine-first defaults](spine-first-defaults.md) and [Shift-left task organization](shift-left-task-organization.md):

1. **Red phase (Reproduction / Witness):** Before writing implementation code, author a targeted reproduction test or capture a render artifact demonstrating the defect or requirement. The test must fail before the fix.
2. **Green phase (Minimal implementation):** Implement the minimal code change required to make the failing test pass, without introducing unsolicited scaffolding.
3. **Verification phase (Deterministic proof):** Execute the targeted test suite and project linters to witness the transition from red to green.

Phased test-driven development replaces ambiguous instructions ("make sure not to break anything") with a deterministic, objective finish line. The agent knows exactly when it is done, preventing self-narrated completion claims and unverified churn.

## Review and authoring checklist

Before publishing a tool, prompt, policy, or workflow, verify compliance against the five positive management criteria:

- [ ] **Boundary placement:** Are existential risks (SSRF, out-of-tree writes, credential leaks) enforced at the compiled-in FROZEN floor, rather than delegated to prompt instructions?
- [ ] **Permissive convenience:** Are local workspace tools (reading, writing, compiling, testing) unhindered by speculative heuristics?
- [ ] **Affirmative guidance:** Do instructions and advice state what *should* happen and name the available tools, rather than listing prohibitions?
- [ ] **Executable next actions:** Does every refusal or error emission pair a closed reason token with an immediate, runnable next step?
- [ ] **Verifiable witness:** Is work structured around an observable red-phase reproduction followed by a green-phase proof?

## Related authorities

- [Positive-state construction](positive-state-construction.md) — broadcast the target state instead of a negation operand
- [Shared-workspace positive state](shared-workspace-positive-state.md) — negframe and managed context as one emit pipeline
- [Spine-first defaults](spine-first-defaults.md) — applied implementation and captured witness before fan-out
- [Shift-left task organization](shift-left-task-organization.md) — explicit outcome, scope, dependencies, and placement
- [Policy in the kernel](explainers/policy-in-the-kernel.md) — the capability floor and fail-closed syscall design
- [Org-policy precedence lattice](notes/RESEARCH-org-policy-precedence-2026-07-20.md) — compiled-in FROZEN floor precedence
- [`AGENTS.md`](../AGENTS.md) — orientation and hard rules for autonomous contributors
- [`POLICY.md`](../POLICY.md) — deployable capability floor specifications
