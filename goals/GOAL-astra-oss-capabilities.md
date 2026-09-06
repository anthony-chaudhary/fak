---
loop: goal
goal_slug: astra-oss-capabilities
issue: 11768
witness: "go test -v ./internal/headlesslint/... ./internal/stopgate/... ./internal/agent/... ./internal/orchestration/... ./internal/gateway/..."
budget: { max_iters: 25 }
lane: multi
---
# Objective
Parent Epic: #11768 (`epic(astra): adopt latest OSS agent harness patterns to work better with GPT-6 Astra`).

Adopt state-of-the-art OSS agent harness patterns (Aider, OpenHands, SWE-agent, OpenCode, Agentless) to maximize GPT-6 Astra reasoning productivity, eradicate literal return-code panic loops, isolate operator remediation strings, enforce prefix-preserving subagent delegation (Gemini 3.8 Flash leaf workers), and guarantee wire-level Responses API compliance.

# Non-Goals
- Do not edit frozen ABI (`internal/abi`).
- Do not alter public exported SDK contracts (`pkg/*`).
- Do not leak private platform implementations into public `fak`.
- Do not commit peer WIP or use `git add -A`.
- Do not introduce external third-party dependencies or unpackaged C-bindings into core packages.
- Do not weaken existing security boundaries, capability floors, or test suites.

# Plan
- [ ] 1. **Milestone 1: Tool Semantic Exit Qualification & JSON Host Boundary (#11769)**
  - *Pattern Source:* OpenHands deterministic JSON `$PS1` protocol (`openhands-tools/openhands/tools/terminal/metadata.py:45@07307cb`) and SWE-agent command return-code handling.
  - *Target Seams:* `internal/auditreason/toolfailure.go`, `internal/assumecheck/assumecheck.go`, and tool execution boundary runners.
  - *Problem:* In POSIX environments, diagnostic commands like `grep`, `git diff --quiet`, and `test -e` return exit code 1 to indicate "not found" or "differences exist" (an expected evaluation result during verification). Astra's hyper-literal logic treats return code 1 as a fatal process crash, incrementing failure counters, triggering expensive recovery tools, and burning thousands of reasoning tokens in unnecessary recovery loops.
  - *Implementation:*
    - Qualify exit codes in `internal/auditreason/toolfailure.go`: map exit code 1 on query/predicate tools (`grep`, `git diff --quiet`, file existence checks) to `StatusPatternNotFound` / `StatusMatchNotFound` instead of an infrastructure failure.
    - Intercept shell host boundaries using a structured JSON envelope protocol separating process exit code, current working directory, and subprocess metadata from stdout, eliminating fragile regex parsing.
    - Update `internal/assumecheck/assumecheck.go` to evaluate assumption predicates without tripping false-positive failure alerts on benign exit code 1 results.
  - *Verification:* Deterministic unit tests in `internal/auditreason` and `internal/assumecheck` proving that read-only negative searches do not trigger failure escalation.

- [ ] 2. **Milestone 2: Out-of-Band Operator Remediation Quarantine (#11770)**
  - *Pattern Source:* Operational study failure mode analysis (Failure Mode 2: Tool Error Self-Attack Loops & Out-of-Band Instruction Bleed).
  - *Target Seams:* `internal/gateway/refusal_notes.go`, `internal/gateway/anchor_refusal.go`.
  - *Problem:* Security and policy guardrails often return remediation advice aimed at human operators, formatted with shell backticks (e.g., ``Operator remedy: run `fak guard allow --ttl 15m <tool>` ``). Operating autonomously in headless mode, Astra parses the backticked command literally and executes it via `bash`, tripping `SELF_MODIFY_BLOCKED` and entering an aggressive self-attack retry loop until exhaustion.
  - *Implementation:*
    - Sanitize all model-visible refusal notes emitted by `internal/gateway/refusal_notes.go` and associated refusal formatters.
    - Strip executable shell backticks, operator remediation strings, and privilege escalation suggestions from agent-visible error payloads.
    - Quarantine operator instructions exclusively to out-of-band operator telemetry, audit logs, and CLI output.
    - Provide models with clear, immutable boundary descriptions (e.g., `"PERMISSION_DENIED: tool is disallowed by policy; select an alternative approach within current capability envelope"`).
  - *Verification:* Unit tests in `internal/gateway/refusal_notes_test.go` proving model-visible error strings contain zero executable command syntax or self-escalation instructions.

- [ ] 3. **Milestone 3: Adversarial Stopgate Invariants & Signed Witness Gate (#11771)**
  - *Pattern Source:* SWE-agent trajectory verification + study Subsystem 6 (Adversarial Stopgate Invariants & Doom-Loop Adjudication).
  - *Target Seams:* `internal/stopgate/boundary.go`, `internal/stopgate/stopgate.go`, `internal/headlesslint/headlesslint.go`.
  - *Problem:* High-reasoning models under negative steerage ladders exhibit an optimization bias toward low-energy state transitions. Astra emits textual surrender phrases such as `"no allowed path: <reason>"` to justify zero-code exits, bypassing necessary retries, alternative approaches, and goal completion.
  - *Implementation:*
    - Eliminate unverified textual surrender bypasses in `internal/stopgate/boundary.go`.
    - Treat `"no allowed path"` as an unproven assertion: stopgate refuses `DispCleanWrapup` unless accompanied by a cryptographically signed kernel refusal receipt or deterministic witness token.
    - Enforce that active goals reject premature surrender notes, forcing `ActionContinue` (exit 2) and providing actionable persistence steerage.
    - Integrate MD5 tool signature tracking to distinguish genuine doom-loops from intentional exploratory retries.
  - *Verification:* Tests in `internal/stopgate/...` and `internal/headlesslint/...` proving unverified surrender is rejected with `ActionContinue` while valid signed kernel refusal tokens allow clean completion.

- [ ] 4. **Milestone 4: Wire-Level Responses API Adapter & Encrypted Reasoning Item Preservation (#11772)**
  - *Pattern Source:* OpenAI Responses API Specification (2026), guided decoding standards, and stream demuxing patterns.
  - *Target Seams:* `internal/gateway/responses.go`, `internal/gateway/mcp_strict.go`.
  - *Problem:* Provider endpoints enforce strict JSON schema compliance (`additionalProperties: false`, all properties declared in `required`, nullable type unions). Furthermore, reasoning models emit thinking tokens (`delta.reasoning_content` or `<think>` tags) and encrypted reasoning items; if reasoning tokens bleed into tool parameter buffers, thoughts can be misparsed as tool calls.
  - *Implementation:*
    - Implement strict schema normalization in `internal/gateway/mcp_strict.go`: automatically inject `additionalProperties: false`, move all declared properties into `required`, convert optional parameters to explicit `["<type>", "null"]` unions, and strip unsupported keywords (`default`, `patternProperties`).
    - Implement clean stream demuxing in `internal/gateway/responses.go`: isolate reasoning tokens into an out-of-band `ReasoningStream` channel, ensuring only finalized content and verified tool calls enter the execution buffer.
    - Preserve encrypted reasoning items across conversation turns to maintain chain-of-thought coherence without exposing raw reasoning thoughts to tool parameter parsers.
  - *Verification:* Unit tests in `internal/gateway/...` verifying strict schema conversion, zero reasoning-token parameter leakage, and correct round-tripping of Responses API message envelopes.

- [ ] 5. **Milestone 5: Architect/Editor Dual-Loop Subagent Delegation (#11773)**
  - *Pattern Source:* Aider dual-model separation (`aider/coders/architect_coder.py:35-120@5dc9490b`) and Agentless hierarchical localization (`agentless/fl/FL.py:28-75@5ce5888b`).
  - *Target Seams:* `internal/orchestration/solroute.go`, `internal/orchestration/`.
  - *Problem:* Astra is a high-cost tier-1 reasoning model ($10/M in, $30/M out, 128k-256k context window). Naive subagent dispatch returns full raw execution transcripts (massive compiler dumps, full AST searches, test traces) to the primary orchestrator, flooding Astra's attention, triggering the "context cliff", and creating prohibitive token billing.
  - *Implementation:*
    - Partition agent roles in `internal/orchestration/solroute.go`: GPT-6 Astra serves as the Lead Architect generating structured task specs, while leaf workers (Gemini 3.8 Flash / Editor) execute exploratory sweeps and mechanical edits.
    - Implement a strict **500-token summary egress filter**: leaf workers must distill raw outputs into structured JSON summaries containing only modified file paths, symbol locations, and verified test assertions.
    - Strictly bar raw execution transcripts from entering Astra's 128k context window.
    - Cryptographically tag incoming worker summaries (`taint: UNTRUSTED_EXTERNAL`) to guard against prompt injection.
  - *Verification:* Unit tests in `internal/orchestration/...` verifying task routing, summary egress budget bounding (<= 500 tokens), and rejection of raw worker transcripts.

- [ ] 6. **Milestone 6: Prefix-Preserving Byte-Splice Compaction with 85% Safety Headroom (#11774)**
  - *Pattern Source:* OpenCode two-tier compaction (`compaction.ts:280-316@03cb63243`), Aider prefix stability, and OpenHands invariant-checked view manipulation.
  - *Target Seams:* `internal/agent/anthropic_compact.go`, `internal/agent/anthropic_compact_view.go`, `internal/ctxmmu/compactor.go`.
  - *Problem:* Extended prompt caching provides up to 80% cost reduction and 5-10x latency improvements on frontier models, but requires bit-for-bit prefix stability from token 0. Naive compaction reorders history or mutates early turns, destroying the KV-cache prefix. Furthermore, compaction triggered too late crashes into the hard context limit.
  - *Implementation:*
    - Anchor prompt prefixes at token 0: system instructions, environment policies, and immutable tool catalogs remain locked across the entire session.
    - Implement two-tier prefix-preserving byte-splice compaction:
      - Tier 1: Replace older tool outputs with compact tombstones (`[Old tool result content cleared]`) at the first threshold, preserving turn boundaries and message structure.
      - Tier 2: Perform structured semantic summarization while enforcing an **85% safety headroom** (triggering before reaching 85% of effective context capacity) to prevent attention dilution and context cliff aborts.
    - Enforce invariant-checked view slicing (`BatchAtomicity`, `ToolCallMatching`, and `ToolLoopAtomicity`) so context compaction never severs a tool call from its result or separates reasoning from action.
  - *Verification:* Deterministic unit tests in `internal/agent/...` and `internal/ctxmmu/...` confirming prefix cache stability, 85% headroom trigger adherence, and atomicity preservation across compactions.

# Reference Study
- Private Platform Architecture Study: `fak-private/docs/notes/STUDY-LATEST-OSS-ASTRA-OPERATIONAL-PATTERNS-2026-09-05.md`
- Master Reference Commits & Upstream Studies:
  - Aider: `paul-gauthier/aider@5dc9490b` (Architect/Editor separation, PageRank RepoMap)
  - OpenHands: `All-Hands-AI/OpenHands@4524a91` / SDK `@07307cb` (JSON $PS1 metadata, invariant view manipulation)
  - SWE-agent: `princeton-nlp/SWE-agent@3ea751c` (Windowed ACI line viewer, delta linting, trajectory separation)
  - OpenCode: `anomalyco/opencode@03cb63243` (Doom-loop breaker, two-tier compaction)
  - Agentless: `OpenAutoCoder/Agentless@5ce5888b` (Hierarchical fault localization, consensus reranking)
  - LangGraph: `langchain-ai/langgraph@f09cfe8f` (Pregel DAG checkpoint isolation)

# Scratch / Status
- **Parent Epic:** #11768 (`epic(astra): adopt latest OSS agent harness patterns to work better with GPT-6 Astra`)
- **Milestone Tracking:**
  - Milestone 1: #11769 (semantic CLI exit qualification)
  - Milestone 2: #11770 (quarantine out-of-band operator remediation text)
  - Milestone 3: #11771 (adversarial anti-surrender invariants)
  - Milestone 4: #11772 (wire-level Responses API adapter)
  - Milestone 5: #11773 (architect-editor dual-loop subagent delegation)
  - Milestone 6: #11774 (prefix-preserving byte-splice compaction)
- **Baseline Established:** Comprehensive study completed in `fak-private/docs/notes/STUDY-LATEST-OSS-ASTRA-OPERATIONAL-PATTERNS-2026-09-05.md` identifying the 6 primary failure modes and mapping 16 candidate borrows.
- **Current Status:** All 6 implementation milestones planned and pending (#11769–#11774 under epic #11768).
- **Next Step:** Begin Milestone 1 (#11769) implementation by adding semantic exit qualification for diagnostic tools in `internal/auditreason/toolfailure.go` and `internal/assumecheck/assumecheck.go`.
