---
title: "Harness tool dialects: why fak manage worked for Claude but looped under Codex/fakc"
description: "Postmortem note for the 2026-07-03 fakc Codex repeat loop: Claude Code's host-tool names were on the embedded guard floor, Codex's snake_case planning/tool names were not. Captures integration implications, architectural limits, anti-patterns, and ticket-ready follow-ups for future harnesses."
---

# Harness tool dialects: guard floor compatibility postmortem

Date: 2026-07-03 local / 2026-07-04 UTC.

## TL;DR

`fak manage` did what it was designed to do: it fail-closed on a tool name that the embedded
floor did not affirmatively allow. The bug was not the provider wire, OAuth, or Codex model
routing. The bug was **harness dialect coverage**:

- Claude Code worked because the default guard floor already allowed Claude-style host tool
  names: `Bash`, `PowerShell`, `Read`, `Edit`, `Write`, `TodoWrite`, `Task`, `ToolSearch`,
  etc.
- Codex/fakc looped because Codex used snake_case host tools: especially `update_plan`
  and, for real shell work, `shell_command`. Some Codex surfaces expose the same tools
  with namespace-qualified spellings such as `functions.update_plan`,
  `functions.shell_command`, `tool_search.tool_search_tool`, and
  `multi_tool_use.parallel`; those spellings need the same floor treatment.
- `update_plan` was not on the embedded floor, so the guarded Codex session received:
  `update_plan: DENY (DEFAULT_DENY/TERMINAL)`.
- The active `/goal` machinery auto-continued because the objective was still active.
  Each continuation tried the same planning seam first, was denied, and made no progress.

Observed impact from the audited session:

- Codex thread: `019f2af4-fe54-7451-ba23-3b59a1f8c840`.
- Launch chain was structurally correct: `fakc -> fak codex -> fak manage -> codex`.
- Guard journal: repeated `DENY tool=update_plan reason=DEFAULT_DENY`.
- Codex session ended paused/interrupted after **211,527 tokens** and **257 seconds**.
- No DOS stop failure was responsible; the root refusal came from the `fak manage`
  capability floor.

The narrow fix is to admit Codex's host-orchestration tool names — bare and
namespace-qualified — and put `shell_command.command` / `functions.shell_command.command`
under the same dangerous-command argument rules as `Bash.command` and
`PowerShell.command`. The broader lesson is that **"supports an agent harness" means
supporting its tool dialect, not only its model wire.**

## What happened

`fak codex` correctly wrapped Codex with `fak manage` and injected the Responses provider
override:

```text
codex -c model_provider=fak
      -c model_providers.fak.base_url="http://127.0.0.1:<port>/v1"
      -c model_providers.fak.wire_api="responses"
```

The child reached the guard gateway. The gateway saw a host tool call named `update_plan`.
The embedded policy in `cmd/fak/guard-default-policy.json` admitted the existing Claude /
Ultracode / OpenCode shapes but did not admit that Codex snake_case name. Because the
posture is fail-closed, `update_plan` resolved to `DEFAULT_DENY`.

That is correct kernel behavior for an unknown capability. It is bad product behavior for
an advertised `fakc` path: planning is harness plumbing, not a dangerous side effect.

## Relationship to other harnesses

The failure belongs to the same family as previous harness-coherence findings:

| Harness / surface | Tool dialect shape | Integration risk |
|---|---|---|
| Claude Code | PascalCase host tools (`Bash`, `TodoWrite`, `Task`, `ToolSearch`) plus hooks | Works only if those orchestration tools are admitted and effectful tools carry arg rules. |
| Codex CLI / `fakc` | snake_case host tools (`update_plan`, `shell_command`, MCP resource helpers, goal helpers) | A missing planning/tooling name can hard-loop under `/goal` auto-continue before any useful work starts. |
| OpenCode | lowercase tools (`bash`, `read`, `write`, `edit`) | A lower-case shell name needs the same danger floor as `Bash`; otherwise either safe work is denied or dangerous commands bypass the typed rule. |
| MCP clients | `mcp__server__tool` names and resource APIs | Read-only MCP verbs can be safe plumbing; mutating MCP verbs must remain off-floor or carry explicit arg/runtime rules. |
| Future IDE/cloud harnesses | Tool names, approval semantics, and lifecycle hooks are vendor-specific | A provider-compatible `/v1` wire does not imply compatible tool-call semantics or stop/continue semantics. |

Future integrations need a **harness profile contract**:

1. The model/provider wire (`responses`, `chat/completions`, Anthropic messages, MCP).
2. The host tool namespace and exact names.
3. Which tools are pure orchestration/read-only vs effectful.
4. The argument fields that carry shell commands, paths, URLs, or secrets.
5. The stop/continue semantics: when a denied tool ends a turn, pauses, retries, or loops.
6. A replay fixture proving the advertised launcher can complete one inspect-and-edit loop
   without a tool-name `DEFAULT_DENY`.

Without that contract, each new harness will rediscover the same class of bug.

## Architectural limitation exposed

The proxy guard can adjudicate only the calls the external harness exposes to it. That is
powerful, but it is not ownership of the loop:

- The external harness owns planning, auto-continue, stop hooks, transcript persistence, and
  sometimes tool execution.
- `fak manage` owns the capability floor at the tool-call boundary.
- If the harness proposes an unknown host-tool name, the floor must deny it.
- If the harness treats that denial as a retryable planning failure while the goal remains
  active, the session can spin.

This is a proxy-path limitation, not a reason to weaken fail-closed. The kernel should not
guess that an unknown `update_plan`-like name is safe. The integration layer must make the
host dialect explicit and test it.

This also reinforces the native-harness note (`docs/notes/native-harness-progress-tracking-1315.md`):
until fak owns the full loop, it cannot fully control auto-continue policy, summarization,
or how a harness reacts to refused tools. It can make refusals explicit and bounded, but
the external harness still decides whether to re-prompt.

## Anti-patterns to avoid

1. **"The provider wire works, therefore the harness works."** False. Codex reached the
   Responses gateway; the failure was the host tool dialect.
2. **Static global allow-list as the only integration mechanism.** A single embedded floor
   will drift behind new harnesses. The floor needs profile-derived coverage tests, not
   only manual additions.
3. **Admit effectful shell aliases without mirroring danger rules.** If `shell_command` or
   `functions.shell_command` is allowed but only `Bash.command` has `rm -rf` / `sudo` /
   `curl|sh` rules, the policy gets weaker while fixing compatibility.
4. **Treat planning tools as harmless only by vibes.** `update_plan` is safe because its
   schema is plan-state only; that should be encoded in the harness profile or regression
   fixture, not inferred from the English name.
5. **Let deny-all turns auto-continue unbounded.** A repeated `DEFAULT_DENY` on the same
   tool name is not useful work. The loop should surface a bounded diagnostic and stop or
   degrade to a text answer.
6. **Use self-report as the witness.** The proof is the guard journal + session JSONL +
   regression test, not "Codex said it was checking."
7. **MCP/proxy conflation.** MCP integration can give Codex explicit fak tools while leaving
   Codex's model wire untouched. The proxy path gates model-wire tool calls. They have
   different evidence and failure modes.

## Ticket-ready follow-ups

### 1. Guard profile coverage: assert default floor admits every first-class harness dialect

**Why:** Prevent `fakc`/Codex, OpenCode, Claude Code, Cursor, or future IDE harnesses from
shipping with an advertised launcher whose first planning/tool call is `DEFAULT_DENY`.

**Scope:**

- Add a table of first-class harness profiles: Claude Code, Codex, OpenCode, MCP client.
- For each profile, list required host tool names and effectful command/path fields.
- Test the embedded guard floor admits required orchestration/read-only tools.
- Test every effectful shell alias carries the same dangerous-command denies.

**Witness:** `go test ./cmd/fak -run 'TestGuardDefaultPolicy|TestHarnessProfile'` plus
`fak policy --check cmd/fak/guard-default-policy.json`.

### 2. Codex deny-all loop breaker: stop repeated guarded `/goal` spins on same refused tool

**Why:** The audited run burned 211k tokens because an active goal kept re-entering the same
denied planning seam.

**Scope:**

- Detect consecutive guarded turns whose only proposed tool is denied with the same
  `DEFAULT_DENY` reason.
- After a small threshold, stop auto-continuing and surface: tool name, reason, policy path,
  and suggested fix (`allow-list host plumbing` vs `use text-only answer`).
- Keep the failure closed; do not silently bypass the floor.

**Witness:** A replay fixture of `update_plan DEFAULT_DENY` produces a bounded stop instead
of >N continuations.

### 3. Harness integration contract: separate model-wire compatibility from host-tool compatibility

**Why:** Future integrations should not be accepted because `/v1/responses` or MCP connects
once. They need tool dialect and lifecycle evidence.

**Scope:**

- Extend `docs/integrations/` with an "integration acceptance checklist" section:
  model wire, tool dialect, arg-rule coverage, stop semantics, replay fixture.
- Link it from the Codex guide and guard docs.
- Add a minimal fixture for each advertised launcher.

**Witness:** `tools/agent_readiness_scorecard.py` or a Go equivalent flags an integration
doc/launcher with no dialect fixture.

### 4. Policy generation from harness profiles, not hand-edited sprawl

**Why:** The embedded floor is accumulating hand-maintained variants (`Bash`, `bash`,
`PowerShell`, `shell_command`). Manual edits are easy to miss and easy to weaken.

**Scope:**

- Define a small data source for host-tool aliases and effectful fields.
- Generate or lint the embedded floor from that source.
- Require shell-like aliases to inherit the common dangerous-command rule set.

**Witness:** A test that adds a new shell alias without inherited rules fails.

## Immediate note for operators

If `fakc` repeats "All proposed tool calls were refused" or repeatedly says "I will
inspect..." without tool progress:

1. Check the guard journal under `.dispatch-runs/guard-audit/interactive-*.jsonl`.
2. Look for repeated `DENY` rows with the same `tool` and `reason`.
3. If the tool is host plumbing (`update_plan`, `tool_search_tool`, MCP list/read), it is
   a harness-dialect coverage bug.
4. If the tool is effectful (`shell_command`, `write`, MCP mutation), verify the dangerous
   arg rules before allow-listing anything.
5. Do not solve it by disabling `fak manage`; solve it by making the harness profile explicit.
