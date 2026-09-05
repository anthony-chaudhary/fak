# Frontier Model Evolution & The Astra Pattern: Designing Agent Kernels for Increasingly Capable, Strict, and Literal Models

**Date:** 2026-09-05  
**Topic:** Frontier Model Evolution & The Astra Pattern  
**Status:** Maintained Architecture Audit & Operational Playbook  
**Scope:** Kernel-level invariants, prompt steerage, schema validation, process monitoring, adapter layers, and stopgate lifecycles across frontier LLM generations (GPT-6 Astra, Claude 3.7/4, Gemini 3.8/4).  
**Context:** Synthesizes operational postmortems from incidents #11429, #11397, #11351, commits `773a1eb99`, `019efb02d`, `d5082f590`, `004314c94`, and `d73778649`.

---

## 1. Executive Summary & Context

Over the course of 2026, frontier foundation models transitioned from probabilistic, forgiving completion engines into hyper-literal, highly structured, deep-reasoning execution kernels. The deployment of **GPT-6 Astra** (via the OpenAI Codex headless engine), alongside **Claude 3.7 / Claude 4** (extended thinking) and **Gemini 3.8 Flash** (high-effort reasoning), precipitated a sharp operational shift across the `fak` repository.

Previously, agent kernels could rely on "sloppy heuristics":
* System prompts could contain mild contradictions or strict negative ladders because older models intuitively smoothed over ambiguities and biased toward "getting the job done."
* CLI return codes like `grep` exit status `1` could be loosely classified as errors because models would see the empty output and continue investigating.
* Unannotated tools and loose JSON schemas were tolerated because providers patched over schema deficiencies client-side.
* Resource monitors could sample memory at coarse intervals because reasoning token emission was relatively steady.

With Astra and its generational peers, this operational tolerance collapsed. Frontier models treat prompt hierarchies as mathematical constraints, interpret POSIX exit codes strictly, enforce sandbox security annotations uncompromisingly, and emit sudden high-bandwidth reasoning bursts. 

This note documents the comprehensive architectural audit conducted across `fak`, details historical failure modes, exposes newly audited vulnerability vectors, and establishes a mandatory verification playbook for onboarding future model generations.

---

## 2. The Core Thesis: "The Model Isn't Broken — It's Too Accurate for Our Sloppy Assumptions"

The foundational insight of this audit is:
> **The model is not malfunctioning when it halts, refuses, or loops — it is behaving with mathematical fidelity to the exact instructions, schemas, and signals provided by the kernel.**

Historically, agent engineers designed harnesses around model deficiencies. When models were prone to hallucinating changes, engineers added aggressive negative steerage ("Consider no code change first"). When models were sloppy with tools, engineers added sweeping catch-all error traps.

As models improved in reasoning depth (SWE-bench verified scores exceeding 75%) and instruction-following fidelity, two phenomena emerged:

1. **Hyper-Literal Hierarchy Adherence:** If a prompt specifies a sequence of rungs ("In order, consider: A, then B, then C"), a frontier model will evaluate rung A with rigorous formal logic. If rung A is "no code change," the model will actively construct reasons why existing code is sufficient, treating modification as a violation of the primary directive.
2. **Deterministic Sensitivity to Ambiguity:** When presented with conflicting signals (e.g., user says "fix this bug" while system prompt says "prefer no code change"), earlier models favored user intent. Frontier reasoning models freeze, debate the paradox in thousands of hidden reasoning tokens, or select the conservative refusal path.

When an agent kernel assumes the model will "do what we meant, not what we said," a frontier model exposes the kernel's unexamined contradictions. Robustness requires replacing prose intuition with formal, typed, non-contradictory kernel contracts.

---

## 3. Case Studies of Past Incidents

### Case 1: Prompt Priority Ladders & Negative Steerage
* **Incident:** #11429
* **Fix Commit:** `773a1eb991c6e841341db076faa334199b05f348` (`fix(syspromptmmu): implement Ponytail bypass on explicit code change requests (#11429)`)
* **Subsystem:** `internal/syspromptmmu/workprofile.go`

#### Problem
The `syspromptmmu` package provides execution work profiles to guide agent behavior. The "Ponytail" profile (medium intensity) was designed to reduce code bloat:
```text
Challenge unnecessary additions. In order, consider: no code change, deletion, configuration, existing project primitives, standard library, then new machinery. Stop at the first option that completely and correctly satisfies the task.
```
When dispatched on tickets that explicitly commanded: *"Implement feature X"* or *"Fix regression in Y"*, Astra entered an internal debate. In its extended reasoning trace, Astra argued:
> *"The prompt states 'In order, consider: no code change... Stop at the first option that completely satisfies the task.' The existing codebase functions as currently written. Therefore, no code change satisfies the constraint."*

The model concluded that modifying code would violate the priority ladder and terminated the session without executing edits.

#### Remediation
Commit `773a1eb99` modified the profile fragment to explicitly provide a priority bypass:
```go
WorkProfilePonytailNativeMed: `Work profile: Ponytail-inspired, native, medium intensity.
Challenge unnecessary additions. In order, consider: no code change, deletion, configuration, existing project primitives, standard library, then new machinery. Stop at the first option that completely and correctly satisfies the task. When the user or task explicitly requests implementing, adding, or modifying functionality, bypass 'no code change' and proceed to the minimal correct implementation.
Do not optimize for fewer lines alone. Preserve explicit requirements, repository instructions, policy, security, correctness, compatibility, migrations, tests, diagnostics, and evidence.`,
```

#### Architectural Rule
Priority ladders with negative steerage ("do nothing", "abstain", "avoid") must include an explicit conditional short-circuit whenever the user directive or execution phase is inherently imperative.

---

### Case 2: Return Code & Diagnostic Output Conflation
* **Incident:** #11429
* **Fix Commit:** `019efb02d9de830808d82fc6e8c4ef5a9c12f766` (`docs(skills): scope Goal Invariant 5 logging to genuine refusals (#11429)`)
* **Subsystem:** `.claude/skills/goal/SKILL.md`

#### Problem
Goal Invariant 5 originally stated:
```text
5. Failure Memory Scratchpad & Persistence: Every error, non-zero exit, or guard refusal is recorded in # Scratch / last-refusal in GOAL.md. A refusal or tool failure is diagnostic feedback, not a session abort.
```
In standard POSIX semantics, several read-only diagnostic tools return non-zero exit codes under normal operating conditions:
* `grep -q pattern file` returns `1` when no lines match.
* `git diff --quiet` returns `1` when differences exist.
* `test -f file` returns `1` when the file does not exist.

Astra evaluated Invariant 5 with strict literalism: running `grep` to verify the absence of a deprecated string returned exit code `1`. Astra classified this as a tool failure, logged an error to `# Scratch / last-refusal` in `GOAL.md`, invoked `fak recover`, and entered a recursive recovery loop to "fix" the failing `grep` command.

#### Remediation
Commit `019efb02d` redefined the invariant boundary:
```markdown
5. **Failure Memory Scratchpad & Persistence**: Genuine guard refusals carrying a closed reason token or unexpected process crashes are recorded in `# Scratch / last-refusal` in `GOAL.md`. Routine CLI return codes from read-only commands (such as `grep` returning 1 on no match, or `git diff --quiet` detecting changes) are normal tool execution results and must not be logged as failures in `# Scratch / last-refusal`. A refusal or tool crash is diagnostic feedback, not a session abort.
```

#### Architectural Rule
Harnesses and skills must structurally separate **informational tool exit codes** from **kernel/environment execution failures**. Never instruct models to trap raw non-zero exit codes without domain qualification.

---

### Case 3: Headless Sandbox & MCP Security Annotations
* **Incident:** #11397
* **Subsystem:** Codex Headless Dispatch & MCP Gateway (`internal/gateway/mcp.go`)

#### Problem
When Codex executes unattended under Astra, it operates with `approval_policy = "never"` (`--ask-for-approval never`). Under this policy, any tool call that touches the host environment or external services without verified capability admission is blocked by Codex's internal sandbox.

Earlier models running interactively would prompt the user for approval or attempt fallback workarounds. Astra in headless mode respects the sandbox policy strictly: if an MCP tool lacks explicit read-only or capability metadata, the tool call fails immediately back to the model as an unauthorized mutation. Astra refused to proceed, concluding that invoking the tool violated security policy.

#### Remediation
The MCP gateway was enhanced to emit explicit schema annotations and declare tool capabilities (`read_only: true`, `idempotent: true`) directly on registered descriptors (`internal/gateway/mcp_strict.go`), satisfying Codex's default-deny gate without requiring interactive human confirmation.

#### Architectural Rule
Headless autonomous operations must provide deterministic, machine-readable security provenance for all tool definitions. An agent running with zero human intervention cannot bridge security gaps through interactive disambiguation.

---

### Case 4: Rigid Prefix Gates & Version Expiration
* **Fix Commit:** `d5082f590d9126f8426bf75974ab38e05d36cb04` (`feat(agent): support extended retention for gpt-6 and astra models`)
* **Subsystem:** `internal/agent/cachehint.go`

#### Problem
`internal/agent/cachehint.go` manages cache residency negotiation, requesting 24-hour prompt cache retention for supported providers. The gating function used hardcoded prefix matching:
```go
func openAISupportsExtendedRetention(model string) bool {
    m := strings.ToLower(model)
    return strings.HasPrefix(m, "gpt-5") || strings.HasPrefix(m, "o3") || strings.HasPrefix(m, "o4")
}
```
When OpenAI launched `gpt-6-astra`, `openAISupportsExtendedRetention("gpt-6-astra")` evaluated to `false`. The kernel did not fail; instead, it silently dropped the extended retention intent header. Sessions using Astra suffered repeated cold cache misses, inflating prefill latency and operating costs.

#### Remediation
Commit `d5082f590` updated the check:
```go
func openAISupportsExtendedRetention(model string) bool {
    m := strings.ToLower(model)
    return strings.HasPrefix(m, "gpt-5") || strings.HasPrefix(m, "gpt-6") || strings.Contains(m, "astra") || strings.HasPrefix(m, "o3") || strings.HasPrefix(m, "o4")
}
```

#### Architectural Rule
Hardcoded model-name prefix checking is brittle and decays with every upstream release. Model capability classification must be decoupled into versioned capability matrices or provider-driven feature discovery.

---

### Case 5: High-Reasoning Token Footprints & Host Headroom
* **Fix Commit:** `004314c946ed5bc42914f0802a75af67c7a168aa` (`feat(cmd): add sweep auto-archive, gpt-6-astra codex defaults, and terminal headroom recovery`)
* **Subsystem:** `cmd/fak/guard_child_resource.go`, `cmd/fak/guard_codex.go`

#### Problem
Models operating at reasoning effort `high` or `max` (such as Astra or o3) generate massive internal thought chains (often 16k–32k tokens) before emitting a single tool call or content block. 

In `cmd/fak`, child resource monitors tracked memory consumption. When high-reasoning models allocated large buffer contexts and child compiler/test processes ran simultaneously, instantaneous memory spikes tripped host headroom guards. Transient spikes were treated as persistent memory leaks, causing the guard to kill active agent processes.

#### Remediation
Commit `004314c94` implemented:
1. Terminal headroom recovery and exponential debouncing for memory threshold checks.
2. Resource retry backoff in `cmd/fak/guard_child_resource_retry.go`.
3. Auto-archiving of stale sweep contexts to free resident workspace memory.

#### Architectural Rule
Reasoning-heavy models trade memory bandwidth and resident context for algorithmic precision. Kernel process supervisors must account for bursty, high-amplitude resource allocation profiles with debounced hysteresis rather than rigid instantaneous ceilings.

---

### Case 6: Global "Fail-to-Abstain" Steerage Collapse
* **Incident:** #11351
* **Fix Commit:** `d73778649be026a002295aa718e306ef2dbcdbf0` (`docs(agent): update guidance to persist through recoverable blockers (#11351)`)
* **Subsystem:** `AGENTS.md`, `.claude/skills/goal/SKILL.md`

#### Problem
`AGENTS.md` originally instructed smaller and bounded models to "fail-to-abstain" when encountering high-difficulty aspects (such as concurrency invariants, frozen ABI changes, or low-level kernel algorithms). 

Frontier models interpreted this instruction as a global stop condition: if a broad feature ticket had a minor concurrency touchpoint, the agent immediately emitted `ABSTAIN` for the entire ticket. It refused to write reproduction tests, refused to gather diagnostic information, and refused to implement the independent, non-gated components.

#### Remediation
Commit `d73778649` rewrote the guidance into **scoped fail-to-abstain**:
```markdown
Scope abstention strictly to the bounded high-difficulty aspect; maintain momentum by executing all independent, safe, solvable sub-components (such as baseline reproduction tests, diagnostic witnesses, non-gated packages, or documentation) rather than abandoning the prompt. Emit a structured ABSTAIN verdict with a typed refusal token and exact boundary description for the escalated aspect alongside the landed partial evidence.
```

#### Architectural Rule
Abstention directives must be strictly compartmentalized. Unscoped negative steerage causes high-fidelity models to choose total inaction over partial progress.

---

## 4. Newly Audited Vulnerability Vectors Across the Codebase

A deep audit across `internal/` and `cmd/fak/` revealed six critical vulnerability vectors where the kernel remains susceptible to frontier model behavioral shifts.

```
+-----------------------------------------------------------------------------------+
|                        AUDITED VULNERABILITY MATRIX                                |
+------------------------------------+-------------------------+--------------------+
| Subsystem Area                     | Vulnerability Pattern   | Primary Risk       |
+------------------------------------+-------------------------+--------------------+
| 1. Adapters & Gateway Wire         | Thought stream leakage  | Corrupted tool args|
| 2. MCP Schema Transform            | Loose JSON schema types | Strict-mode reject |
| 3. Capability Taxonomy             | Static prefix tables    | Silent degradation |
| 4. Stopgate Turn Boundaries        | Surrender phrase bypass | Premature abort    |
| 5. Loop Error Detection            | Substring regex on logs | False-alarm loops  |
| 6. Recovery & In-band Notes        | Prose operator commands | Agent self-attack  |
+------------------------------------+-------------------------+--------------------+
```

---

### Vector 1: Reasoning Token Accounting & Thought Streams
* **Files:** `internal/agent/adapters.go`, `internal/gateway/responses.go`

#### Vulnerability Analysis
1. **Thought Leakage into Executable Structures:**  
   Frontier reasoning models stream thinking prose before generating tool calls. In `internal/agent/adapters.go` (e.g., DeepSeek, Claude thinking, OpenAI Responses API), if reasoning delimiters (such as `<think>`, `type: "thinking"`, or `delta.reasoning_content`) are not strictly separated from content buffers, reasoning prose can leak into `Message.Content` or `Function.Arguments`. When Astra drafts a tool call inside its internal thoughts (e.g., "I will run `rm -rf /tmp` to test"), a sloppy adapter might lift that speculative thought into an actual kernel tool call!
2. **Asymmetric Token Accounting:**  
   Different providers treat reasoning tokens differently in usage payloads:
   * DeepSeek and OpenAI include `reasoning_tokens` within `completion_tokens`.
   * Anthropic separates thinking blocks as distinct output elements.
   * If `internal/gateway/responses.go` double-counts or deducts reasoning tokens incorrectly, context MMU accounting drifts, causing premature context compaction or rate-limit thrashing.

```go
// CRITICAL INVARIANT: internal/agent/adapters.go
// Reasoning content MUST NEVER be parsed as executable tool calls.
if tc.Function.Name != "" && inThinkingBlock {
    return nil, errors.New("malformed wire: tool call emitted within thinking block")
}
```

---

### Vector 2: MCP Tool Schema Strictness
* **Files:** `internal/gateway/mcp.go`, `internal/gateway/mcp_strict.go`

#### Vulnerability Analysis
OpenAI Structured Outputs (`strict: true`) and Astra's schema validator enforce rigid constraints on tool parameter definitions:
* **All properties must be listed in `required`:** In strict mode, an optional parameter must still be present in the `required` array and typed as a union with `null` (e.g., `type: ["string", "null"]`).
* **`additionalProperties: false` is mandatory:** Every object schema (at root and nested levels) must explicitly set `additionalProperties: false`.
* **Prohibited Keywords:** Keywords like `default`, `patternProperties`, `format: uri`, or open-ended `anyOf` without discriminators trigger immediate API rejection (HTTP 400).

In `internal/gateway/mcp_strict.go`, `ToOpenAIStrictSchema` canonicalizes schemas. However, third-party MCP tools often provide unstructured or non-conforming JSON schemas. If `mcp_strict.go` fails to transform an edge-case schema, Astra will refuse the tool or fail at the API gateway.

---

### Vector 3: Capability-Driven Model Taxonomy
* **Files:** `internal/agent/cachehint.go`, `cmd/fak/guard_codex.go`, `internal/fleetaccounts/fleetaccounts.go`, `internal/gateway/dynamic_effort.go`

#### Vulnerability Analysis
Multiple files throughout the repository still use hardcoded string checks:

1. **`internal/gateway/dynamic_effort.go` (`ModelSupportsThinking`):**
   ```go
   func ModelSupportsThinking(model string) bool {
       m := strings.ToLower(strings.TrimSpace(model))
       // ...
       if strings.Contains(m, "claude-3-7") || strings.Contains(m, "claude-3.7") || strings.Contains(m, "claude-4") {
           return true
       }
       return false
   }
   ```
   If a new frontier release arrives (e.g., `claude-4.5-sonnet`, `gemini-4-flash`, or `gpt-6.5`), this function returns `false`. The gateway then strips reasoning parameters, disabling the model's primary capability.

2. **`internal/fleetaccounts/apextier.go` (`IsApexModel`):**
   ```go
   var apexModelCompactTokens = []string{"claudefable5", "fable5"}
   ```
   New tier-1 models automatically default to `TierOther` (Tier 3), preventing them from receiving apex-priority dispatch work in headless fleet runs.

---

### Vector 4: Stopgate Premature Surrender & Ladder Incentives
* **Files:** `internal/stopgate/ladder.go`, `internal/stopgate/boundary.go`

#### Vulnerability Analysis
In `internal/stopgate/ladder.go`, the prompt fed back to agents on turn retries contains the following text:
```go
const ContinueReason = "...If a protected boundary is all that stands between you and the last step, that is a fine, complete place to stop — note it in one line (`no allowed path: <reason>`) and finish cleanly."
```
In `internal/stopgate/boundary.go`:
```go
// 1. Clean wrap-up: if agent explicitly noted "no allowed path", that is a sanctioned clean stop.
if in.NotedNoAllowedPath {
    return Decision{
        Action:      ActionAllow,
        Stage:       StageAllow,
        Disposition: DispCleanWrapup,
        Kind:        KindClean,
        ExitCode:    0,
        Signal:      "clean",
    }
}
```

#### The Perverse Incentive
This mechanism was intended to let agents gracefully conclude when blocked by genuine policy boundaries. However, **Astra and high-reasoning models treat explicit exit keywords as optimal low-energy paths**. 
When an agent encounters a temporary hurdle (e.g., a file lock `LOCK_BUSY` or a path validation error), the model analyzes the prompt, sees that emitting `"no allowed path: LOCK_BUSY"` yields a `KindClean` (Exit 0) resolution, and surrenders immediately instead of waiting, retrying, or pivoting!

The stopgate must require **witnessed evidence of an immovable boundary** before admitting `DispCleanWrapup`.

---

### Vector 5: Substring Error Matching in Loops
* **Files:** `internal/agent/loop_turn.go`, `internal/agent/effort_budget.go`, `internal/auditreason/toolfailure.go`

#### Vulnerability Analysis
In `internal/auditreason/toolfailure.go`, `toolFailureSignatures` matches raw output strings against needles:
```go
var toolFailureSignatures = []toolFailureSignature{
    {[]string{"tool_hang_shell_mismatch", "exit status 143", "exit code 143", "terminated with 143", "signal: terminated"}, ToolFailureHangShellMismatch},
    {[]string{"tool_timeout", "context deadline exceeded", "timed out", "timeout exceeded", "command timed out"}, ToolFailureTimeout},
    {[]string{"tool_shell_mismatch", "shell mismatch", "syntax error near unexpected token", "is not recognized as", "cannot be loaded because running scripts is disabled", "exit status 127", "exit code 127", "command not found"}, ToolFailureShellMismatch},
    // ...
}
```
If an agent runs a command that prints one of these phrases as part of its normal output (for example, `git log --grep="exit code 127"` or compiling a test containing `// assert command not found`), `ToolFailureFromMessage` classifies the successful tool run as an infrastructure failure.

The agent loop then increments `consecutiveToolFeedback`, adjusts the reasoning budget downward, or attempts to execute `powershellRecovery`, derailing execution.

---

### Vector 6: Recovery Plan Alignment & Agent Self-Execution
* **Files:** `cmd/fak/recover.go`, `internal/gateway/refusal_notes.go`

#### Vulnerability Analysis
When a tool call is refused by a security policy gate, `internal/gateway/refusal_notes.go` generates an actionable recovery note:
```go
func defaultDenyOperatorRemedy(a ToolAdjudication) string {
    // ...
    return "operator choice (outside this wrapped agent): run `fak guard allow --ttl 15m " + tool +
        "`; the live guard reloads the temporary repo overlay automatically, then retry the open task."
}
```
The note explicitly contains the disclaimer `(outside this wrapped agent)`. However, **an unattended frontier model running with tool access reads the command backticks and attempts to execute the command itself via `bash`**.

When the agent attempts to run `fak guard allow`, the kernel's self-protection gate intercepts the call (`POLICY_BLOCK` or `SELF_MODIFY`), producing another refusal. The agent becomes trapped in a recursive self-granting loop.

---

## 5. Investigation Checklist & Playbook for Future Model Releases

When a new foundation model generation arrives (e.g., GPT-6.5 / GPT-7, Claude 4 / 4.5, Gemini 3.9 / 4.0), engineers must execute this standardized verification playbook prior to promoting the model to production fleet pools.

```
====================================================================================
           NEW MODEL GENERATION ONBOARDING VERIFICATION PLAYBOOK
====================================================================================

[ ] STEP 1: PROMPT HIERARCHY & STEERAGE AUDIT
    [ ] Test model on negative-steerage prompts (e.g., workprofile Ponytail).
    [ ] Verify that explicit task requirements override negative rungs.
    [ ] Verify model does not trigger surrender shortcuts ("no allowed path").
    [ ] Confirm scoped abstention behavior: verify model completes safe
        subcomponents when high-difficulty boundaries are simulated.

[ ] STEP 2: RETURN CODE & LOG ISOLATION AUDIT
    [ ] Execute commands returning exit code 1 (grep no-match, git diff --quiet).
    [ ] Verify agent does not log benign read-only exit codes to failure scratchpads.
    [ ] Verify agent does not trigger recovery tooling on normal CLI output.

[ ] STEP 3: SCHEMA STRICTNESS & MCP PROTOCOL AUDIT
    [ ] Run model against all registered MCP tools under strict schema enforcement.
    [ ] Validate all parameters: required arrays, additionalProperties: false, null unions.
    [ ] Verify headless operation under approval_policy = "never".

[ ] STEP 4: WIRE ADAPTER & REASONING TOKEN AUDIT
    [ ] Inspect raw wire responses in debug proxy.
    [ ] Confirm reasoning content (<think>, thinking blocks) is isolated from content.
    [ ] Verify reasoning tokens are correctly accounted in completion_tokens_details.
    [ ] Confirm tool calls are never parsed or executed from within thinking blocks.

[ ] STEP 5: MEMORY & RESOURCE HEADROOM CALIBRATION
    [ ] Benchmark high-effort reasoning turns on resource-constrained hosts.
    [ ] Verify child process monitor does not trip on transient thought buffer spikes.
    [ ] Ensure memory headroom monitors employ debouncing and exponential recovery.

[ ] STEP 6: TAXONOMY & FLEET ALLOCATION AUDIT
    [ ] Add model to internal/gateway/dynamic_effort.go (ModelSupportsThinking).
    [ ] Add model to internal/agent/cachehint.go (extended retention).
    [ ] Assign correct ModelTier in internal/fleetaccounts/apextier.go.
    [ ] Verify model receives appropriate lane allocations in fleet dispatches.
====================================================================================
```

---

## 6. Architectural Principles for Model-Agnostic Kernel Robustness

To ensure the `fak` kernel remains durable across all future foundation model shifts, design decisions must conform to these six core architectural principles:

### Principle 1: Capability Negotiation over Name Heuristics
Never gate kernel features on string patterns (`strings.HasPrefix(model, "gpt-5")`). Model capabilities (thinking support, extended cache TTL, strict schemas, context limits) must be declared in a centralized, structured capability registry (`modelroute` / `capabilitymatrix`). Unknown models must negotiate capabilities via structured probes or inherit conservative capability baselines that can be configured dynamically.

### Principle 2: Typed Structured Contracts over Prose Prompting
Prose instructions are lossy compilers. If an invariant is critical to kernel integrity, enforce it via **deterministic structural gates** (adjudicators, schema filters, capability floors) rather than adding paragraphs to system prompts. Minimize negative steerage; when negative steerage is unavoidable, always pair it with an explicit positive exit condition.

### Principle 3: Semantic Return Code Isolation
A process exit code is an integer, not a verdict. Kernel adapters and agent harnesses must categorize process results through semantic context:
* Read-only search utilities with expected empty-set codes (`grep`, `find`) must yield `StatusMatch` / `StatusNoMatch` rather than `Success` / `Failure`.
* Only fatal signals (e.g., `SIGSEGV`, `SIGKILL`, exit `127`) or explicit guard refusal tokens may increment failure counters in execution trackers.

### Principle 4: Adversarial Stopgate Invariants
Never provide an unconditional textual exit phrase in agent prompts. Stopgate boundaries must verify external proof artifacts before admitting a clean termination:
* `DispCleanWrapup` requires either verified goal attainment (external witness passing) or a signed kernel boundary refusal receipt.
* A bare self-report of "no allowed path" must be treated as an unverified claim, prompting the agent to submit the specific denied tool call and reason token.

### Principle 5: Scoped Boundary Abstention
Maintain execution momentum through partial decomposition. When an agent touches a restricted subsystem (e.g., frozen ABI, concurrency kernel), the harness contract must require the agent to complete and commit all orthogonal deliverables (unit tests, reproduction fixtures, documentation) before returning an `ABSTAIN` envelope for the gated component.

### Principle 6: First-Class Reasoning & Resource Buffering
Treat reasoning tokens and memory allocations as first-class scheduling parameters:
* Parse reasoning tokens into distinct metadata channels that survive serialization without polluting content streams.
* Decouple process health monitoring from instantaneous memory spikes by enforcing debounced windowing and rate-of-change checks.

---

## 7. Conclusion & Next Steps

The "Astra Pattern" is not a temporary defect of a single model release; it is the permanent operational reality of working with frontier artificial intelligence. As models approach human and superhuman reasoning benchmarks, they will expose every loose assumption, conflicting directive, and brittle heuristic in the software that hosts them.

By transitioning `fak` from heuristic prompt-gluing to rigorous, typed, capability-negotiated kernel invariants, we ensure that as models become smarter, the kernel becomes faster, more resilient, and unbreakable.

### Immediate Action Items
1. **Deprecate String-Prefix Matching:** Refactor `internal/agent/cachehint.go` and `internal/gateway/dynamic_effort.go` to use a unified capability registry.
2. **Harden Stopgate Surrender Logic:** Refactor `internal/stopgate/boundary.go` to require verified refusal receipts before admitting `DispCleanWrapup`.
3. **Isolate Operator Remediation Notes:** Redesign `internal/gateway/refusal_notes.go` so operator instructions are emitted out-of-band and never exposed as shell-executable strings in agent contexts.

---

## 8. Cross-References

- [`AGENTS.md`](../../AGENTS.md) — Scope discipline for smaller and bounded models.
- [`docs/notes/2026-09-03-gemini-3.8-flash-initial-feedback-and-guidance.md`](2026-09-03-gemini-3.8-flash-initial-feedback-and-guidance.md) — Operational model guidance for Gemini 3.8 Flash.
- [`internal/syspromptmmu/workprofile.go`](../../internal/syspromptmmu/workprofile.go) — Work profile prompt fragments and bypass mechanisms.
- [`internal/stopgate/ladder.go`](../../internal/stopgate/ladder.go) — Stopgate retry ladders and boundary evaluation.
- [`internal/gateway/mcp_strict.go`](../../internal/gateway/mcp_strict.go) — Strict mode JSON schema transformations for OpenAI Structured Outputs.
