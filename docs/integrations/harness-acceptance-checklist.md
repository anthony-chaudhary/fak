---
title: "Harness integration acceptance checklist"
description: "The contract a harness must meet to be advertised as a first-class fak launcher: not only a compatible model/provider wire, but a compatible host-tool dialect, effect classification, argument-field coverage, stop/continue semantics, and a replay fixture. Model-wire compatibility is necessary but not sufficient."
---

# Harness integration acceptance checklist

Putting `fak` in front of an agent has **two** compatibility surfaces, not one:

1. **The model/provider wire** — Responses, Chat Completions, Anthropic Messages, MCP.
   This is what "repoint one base URL" delivers, and it is the easy half.
2. **The host-tool dialect** — the exact tool *names* the harness proposes, their
   argument shapes, which are pure orchestration vs effectful, and what the harness does
   when a tool is denied. This is the half that a working wire silently hides.

A harness can connect its wire perfectly and still fail at the tool boundary. The
`fak guard` capability floor is **fail-closed**: a host-tool name it does not
affirmatively admit resolves to `DEFAULT_DENY`. If the harness treats that denial as a
retryable planning failure while its objective is still active, the session spins —
denied, re-proposed, denied — making no progress.

That is exactly the 2026-07-03 Codex/`fakc` repeat loop: Claude Code's PascalCase tools
(`Bash`, `Read`, `TodoWrite`, …) were on the floor, but Codex's snake_case `update_plan`
was not, so a guarded `/goal` turn burned **211,527 tokens** re-proposing the same
refused planning call. Full postmortem:
[`../notes/HARNESS-TOOL-DIALECT-GUARD-FLOOR-2026-07-03.md`](../notes/HARNESS-TOOL-DIALECT-GUARD-FLOOR-2026-07-03.md).

> **The rule this checklist encodes: "supports a harness" means supporting its tool
> dialect, not only its model wire.** A `/v1/responses` or MCP connection that succeeds
> once is not acceptance evidence.

---

## The acceptance checklist

Before an agent path is advertised as a first-class fak launcher, document all six of
these for it. The first is the wire; the rest are the dialect.

1. **Model/provider wire** — which wire the harness speaks to fak: Responses
   (`/v1/responses`), Chat Completions (`/v1/chat/completions`), Anthropic Messages
   (`/v1/messages`), or MCP (`--stdio` / `/mcp`).
2. **Host tool namespace and exact names** — the literal tool names the harness proposes,
   including any namespace-qualified spellings (e.g. `update_plan` *and*
   `functions.update_plan`; `Bash` vs `bash` vs `shell_command`).
3. **Safe orchestration/read-only vs effectful classification** — which of those tools are
   pure plumbing (planning, list/read, goal state) that the floor should admit, and which
   carry side effects (shell execution, file writes, mutating MCP verbs) that must stay
   off-floor or carry explicit argument rules.
4. **Argument fields that carry shell commands, paths, URLs, or secrets** — the exact arg
   keys the danger rules must bind (e.g. `Bash.command`, `shell_command.command`,
   OpenCode's camelCase `filePath`). An admitted shell alias whose `command` field carries
   no danger rules is a *weaker* floor, not a compatible one.
5. **Stop/continue behavior after a denied tool** — whether a denial ends the turn,
   pauses, retries, or auto-continues under an active goal. A harness that auto-continues
   on `DEFAULT_DENY` needs a bounded deny-all breaker (see
   [`claude.md`](claude.md) → "Deny-all auto-continue").
6. **A replay or paste-and-run fixture** proving the advertised launcher completes one
   inspect-and-edit loop without a tool-name `DEFAULT_DENY`.

---

## The machine-checkable gate (checklist items 2–4 and 6)

Items 2–4 are enforced today by a Go gate over the embedded guard floor, and item 6 has a
replay fixture — no new tooling required to check a first-class harness:

```bash
# Item 2–4: the harness-profile floor-coverage gate. Asserts the embedded floor ADMITS
# every first-class harness's required orchestration/read-only tools and carries the same
# dangerous-command denials for every effectful shell alias — and fails if a newly
# admitted shell-like tool is left unclassified. (cmd/fak/guard_harness_profiles_test.go)
go test ./cmd/fak -run 'TestHarnessProfile'

# Item 6: replay the shipped floor end to end over a recorded trace on either wire and
# assert every call lands on its expected ALLOW/DENY disposition — the launcher completing
# an inspect-and-edit loop without a false DEFAULT_DENY. (cmd/fak/guard_replay_test.go)
go test ./cmd/fak -run 'TestGuardReplay'
fak manage --replay-trace internal/gateway/testdata/guard-trace-e2e.json --replay-wire openai
```

The gate is **data, not prose**: each first-class harness is a
`harnessFloorProfile` row (required tools + shell aliases with their danger commands) in
`cmd/fak/guard_harness_profiles_test.go`. Adding a harness to the checklist means adding a
row; the gate then forces the floor to admit its tools and carry its danger rules. Two
teeth-proving tests confirm the gate is not vacuous: dropping `update_plan` from the floor
surfaces a coverage gap, and admitting an unclassified shell alias (`zsh`) surfaces one.

**Manual check until a harness has a profile row.** For a not-yet-gated harness, adjudicate
its required tool names by hand against the embedded floor — no model, no key, no network:

```bash
# ALLOW: an orchestration/read-only tool that must not DEFAULT_DENY-loop
fak preflight --tool update_plan --args '{}' --policy cmd/fak/guard-default-policy.json

# DENY (POLICY_BLOCK): an effectful shell alias must refuse the danger dialect
fak preflight --tool shell_command --args '{"command":"rm -rf /tmp/x"}' \
  --policy cmd/fak/guard-default-policy.json
```

A required plumbing tool that comes back `DENY` is a dialect-coverage gap to fix in the
floor before advertising the launcher. `fak preflight --explain` shows *why* a verdict
landed.

---

## Record a new harness's dialect and stop semantics here

When you add a harness guide under `docs/integrations/`, fill in this table for it. It is
the durable place to record the tool dialect and stop semantics the postmortem asks every
future harness doc to carry, so the next integration does not rediscover the loop.

| Field | What to record |
|---|---|
| **Wire** | Responses / Chat Completions / Anthropic Messages / MCP |
| **Tool namespace** | bare, `functions.*`, `mcp__server__tool`, lowercase, PascalCase |
| **Required orchestration/read-only tools** | exact names the floor must admit (planning, list/read, goal state) |
| **Effectful tools** | shell executors, file writes, mutating MCP verbs |
| **Dangerous-arg fields** | the arg keys danger rules bind (`command`, `filePath`, URL/secret fields) |
| **Stop/continue after deny** | end turn / pause / retry / auto-continue — and the breaker if it auto-continues |
| **Fixture** | the profile row + replay trace, or the manual `fak preflight` check until one is wired |

### First-class harness coverage today

The advertised launchers and whether the gate/fixture covers their dialect. "Profile row"
= a row in `firstClassHarnessFloorProfiles` checked by `TestHarnessProfileFloorCoverage`.

| Harness / launcher | Wire | Dialect shape | Dialect fixture |
|---|---|---|---|
| Claude Code (`fak guard -- claude`) | Anthropic Messages | PascalCase (`Bash`, `Read`, `TodoWrite`, `Task`) | Profile row + replay (anthropic wire) |
| OpenAI Codex / `fakc` (`fak codex`) | Responses | snake_case (`update_plan`, `shell_command`, `functions.*`) | Profile row + replay (openai wire) |
| OpenCode (`fak guard --provider openai -- opencode`) | Chat Completions | lowercase (`bash`, `read`, `write`, `edit`) | Profile row |
| MCP client (`fak serve --stdio`) | MCP | `mcp__server__tool`, resource APIs | Profile row |

Live gateway-transited end-to-end proof is recorded per launcher in its own guide (the
Anthropic-wire seat is witnessed in [`claude.md`](claude.md) → "Prove it"; the OpenAI-wire
seat's live proof is the open task named there). This checklist gates the **dialect**; the
per-launcher live proof is tracked in each guide.

---

## Cross-references

- [Integration index](README.md) — the front door for every advertised launcher.
- [OpenAI Codex guide](openai-codex.md) — MCP vs guarded model-wire proxy, the two paths this checklist separates.
- [Claude Code guide](claude.md) — the flagship guarded launcher, the deny-all auto-continue breaker, and the live subscription-seat proof.
- [OpenCode](claude.md) — the lowercase-dialect seat (in the Claude guide's OpenCode section).
- [Harness tool-dialect postmortem](../notes/HARNESS-TOOL-DIALECT-GUARD-FLOOR-2026-07-03.md) — why this checklist exists.
- [Compatibility matrix](compatibility-matrix.md) — the surveyed wire each harness speaks (the first checklist item, at scale).
