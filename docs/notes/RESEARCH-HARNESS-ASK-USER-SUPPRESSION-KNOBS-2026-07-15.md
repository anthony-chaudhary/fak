---
title: "Can you turn off 'ask the user'? A survey of the native ask-user knobs in popular harnesses, and where fak's evidence-first gate sits above them (2026-07-15)"
description: "Every autonomous-coding harness has some way to suppress or redirect its 'ask the human' end state — Claude Code's disallowedTools/bypassPermissions/dontAsk/canUseTool/PreToolUse, Codex's approval_policy=never / --yolo / sandbox_mode, Aider's --yes-always, opencode's --auto, Cursor's --force, and the interrupt/resume of the agent frameworks. This note (1) surveys those native knobs with primary-source citations and the tradeoff of each, (2) shows they collapse into exactly two shapes — blind all-or-nothing suppression, or per-harness hand-wiring you must hand-code — neither of which DECIDES the question from the task's own evidence, (3) audits what fak's guard already does at this seam (two Stop-hook rungs: a linguistic operator-directed sensor and an evidence-first operator-question adjudicator over a harness-agnostic normal form) and where it is not yet wired (the PLAN_APPROVAL resolver is nil in production; the CLARIFY resolver runs a single oracle; the gate has no first-class dial and rides --operator-directed), and (4) scopes the follow-on harness-design tickets, including the new inline-interception opportunity the survey surfaced: Claude Code guarantees AskUserQuestion always falls through to PreToolUse, so fak can answer it from evidence BEFORE the turn stalls rather than catching the stall at the Stop hook. Composes with CONCEPT-AGENTIC-OPERATOR-QUESTIONS-2026-07-10 (the forward decision procedure) and the closed epic #4701; ships no code, names the tickets."
---

# The native ask-user knobs, and fak's evidence-first layer above them

> Date: 2026-07-15. Status: research + design note. Ships **no code**. It answers the
> operator question "can the harness's *ask a human* end state be disabled or modified?" from
> primary sources, then maps the answer onto this repo's guard. Composes with —
> and does not duplicate —
> [`CONCEPT-AGENTIC-OPERATOR-QUESTIONS-2026-07-10.md`](CONCEPT-AGENTIC-OPERATOR-QUESTIONS-2026-07-10.md)
> (the *forward* decision procedure: inventory the operator-questions, decide the decidable
> part from a witness, escalate only the residue) and the now-closed epic **#4701** and its
> children (#4702 adapter, #4703 CLARIFY pre-adjudicator, #4704 PLAN-APPROVAL pre-adjudicator,
> #4706 Codex adapter, #4722 native-tool-input exposure). This note is the *sideways* view:
> what the harnesses natively let you do at this seam, and how fak's layer relates to it.

## 0. The ask

Three parts:

1. **Work on how the fak manage handles "ask user question"–type end states.** A turn that ends
   by asking a human — Claude Code's `AskUserQuestion` / `ExitPlanMode`, Codex's
   `request_user_input` / `update_plan`, or just prose ("Do you want me to push?") — is a *stop*
   the agent did not have to take. On an unattended run there is no one to answer, so the work
   silently stalls.
2. **Research whether that can be disabled or modified in popular harness configs.** §2.
3. **Think about, and file, the tickets for our own harness design.** §4.

## 1. What fak already does at this seam (audit)

The guard already senses and acts on this end state in **two Stop-hook rungs**, both gated by
`fak manage --operator-directed off|shadow|warn|enforce` (default **warn**), and both capped so
an *attended* interactive session is never blocked (`guardOperatorDirectedEffectiveMode`: an
`enforce` ever reaching the hook implies the child was headless — the operator is absent by
construction, so continuing the false stop cannot silence a real human question).

- **Rung 1 — the linguistic sensor** (`cmd/fak/guard_operator_directed.go`,
  `internal/headlesslint`). Folds the *final assistant turn's prose*; if it ended by addressing
  a person, enforce feeds back the `choicetriage` remediation ("take the obvious action, restate
  the assumption, file a ticket") and blocks the stop so the agent acts instead of asking. A
  `HUMAN_RESIDUAL` finding is routed as a typed escalation, not re-prompted.
- **Rung 2 — the evidence-first operator-question adjudicator** (`cmd/fak/guard_operator_question.go`,
  `internal/operatorquestion`, `internal/operatorresolve`, `internal/planresolve`). Consumes the
  *native structured tool call* from the transcript (#4722 exposed those inputs), normalizes it
  through the harness-profile registry into one harness-independent `OperatorQuestion{Kind,…}`
  (`CLARIFY` / `CHOOSE_APPROACH` / `PLAN_APPROVAL` / `PERMISSION` / `CONFIRM_IRREVERSIBLE`), and
  adjudicates from read-only oracles into the closed `choicetriage` vocabulary
  (`TAKE_OBVIOUS` / `FRESH_CONTEXT` / `FILE_TICKET` / `HUMAN_RESIDUAL`, default `FRESH_CONTEXT`).

This is already the design thesis of §2's synthesis, realized: **decide the decidable part from
a witness; hand the human only the residue.** The adapters, the CLARIFY resolver, and the
PLAN-APPROVAL resolver all shipped and their epic (#4701) is closed.

**But three gaps are load-bearing for the follow-on work, and this note verified each against
the tree at HEAD:**

1. **The PLAN_APPROVAL evidence path is dead in production.** `guardOperatorQuestionPlanResolver`
   (`cmd/fak/guard_operator_question.go:34`) is a package var that is **only ever assigned in
   tests** — no concrete `planresolve.OracleSet` is built in `cmd/fak`. So a real guard session
   hitting an `ExitPlanMode` / `update_plan` gate always returns `PLAN_ORACLES_UNAVAILABLE` and
   escalates. The pure resolver (`internal/planresolve`, #4704) and its oracle interface are
   fully implemented and green — only the install-time wiring is missing. A closure-lag gap.
2. **The CLARIFY / CHOOSE_APPROACH resolver runs a single oracle.**
   `guardOperatorQuestionClarifyResolver` carries only `GitIsolationOracle` (the shared-tree
   commit-isolation fold from the #4699 example). The §3 "H3 ladder" of the composing note —
   repo-fact resolution (grep/`git`/`dos verify`) and scorecard-axis dominance — is not yet wired,
   so most clarify questions fall through to `FRESH_CONTEXT` for want of an oracle that speaks to
   them.
3. **The gate has no first-class dial.** Rung 2 rides `--operator-directed`, the *linguistic*
   rung's flag. There is no `fak manage --operator-question off|shadow|warn|enforce` an operator
   can reach to disable or tune *just* the structured adjudicator — which is exactly the shape of
   knob every surveyed harness exposes for its own ask-user tool (§2).

## 2. The survey — how each harness lets you disable / modify "ask the user"

Sourced from primary docs (URLs inline). Two claims are flagged **uncertain** and want a
version-pinned re-check before anyone relies on them in code.

### 2a. Claude Code (Anthropic CLI + Agent SDK)

Claude asks for human input in two situations that share one code path: a tool that isn't
pre-approved, and the `AskUserQuestion` clarifying tool; `ExitPlanMode` behaves like a
permission-requiring tool. Evaluation order (first step to resolve wins):
**Hooks → Deny rules → Ask rules → Permission mode → Allow rules → `canUseTool`**
([permissions](https://code.claude.com/docs/en/agent-sdk/permissions),
[user-input](https://code.claude.com/docs/en/agent-sdk/user-input)).

**The subtlety that matters most for fak:** `AskUserQuestion` and MCP tools carrying
`_meta["anthropic/requiresUserInteraction"]` **always fall through to `canUseTool` / a
`PreToolUse` hook, even when an allow rule matches** — they cannot be silently auto-approved by
an allow rule, and in `dontAsk` mode they are **denied**, never answered. This is the inline
interception opportunity §3 names.

| Mechanism | Exact name | Effect | Tradeoff |
|---|---|---|---|
| Remove the ask tool | `disallowedTools:["AskUserQuestion"]` / `--disallowedTools` | Tool never appears; Claude cannot ask at all | Full suppression — it guesses instead |
| Pre-approve tools | `allowedTools` / `--allowedTools` | Auto-approves listed tools | Does NOT silence `AskUserQuestion` (always falls through) |
| Mode: default | `--permission-mode default` | Unmatched tools + `AskUserQuestion` hit `canUseTool` | Needs a callback or nothing answers |
| Mode: acceptEdits | `acceptEdits` | Auto-approves in-workspace edits | Edits only; asks still prompt |
| Mode: plan | `plan` | Read-only; `AskUserQuestion` common to gather requirements | Designed to prompt |
| Mode: bypassPermissions | `--dangerously-skip-permissions` | Approves everything reaching the mode step | Blind approve-all; deny rules, ask rules, hooks, and `requiresUserInteraction` tools still intercept; subagents inherit |
| Mode: dontAsk | `dontAsk` | Any prompt → hard **denial**; `canUseTool` never called | Suppresses AND denies — the ask is denied, not answered |
| Mode: auto | `auto` | A model classifier approves/denies (availability-gated) | Decides by classifier, not evidence/human |
| Runtime callback | `canUseTool` (SDK) | Fires for un-approved tools AND `AskUserQuestion`; return `{behavior:"allow", updatedInput}` — for an ask, `updatedInput.answers` maps question→label, i.e. **answer it programmatically** | You must implement the answer source |
| Headless permission answering | `--permission-prompt-tool <mcp_tool>` | In `--print`/headless, forwards each *permission* request to a named MCP tool that returns allow/deny | Permission prompts only, not clarifying questions; you build the tool |
| Hook interception | `PreToolUse` hook → `hookSpecificOutput.permissionDecision` = allow/deny/ask (+ `updatedInput`) | Runs BEFORE every other step; can approve, deny, or **rewrite the tool input**. `allow`+`updatedInput` on an `AskUserQuestion`/`ExitPlanMode` satisfies the interaction requirement so it runs without a live prompt — you inject the answer | The only place to gate every call (deny holds even under bypass); the answer is whatever you hand-code |
| Notification | `Notification` / `PermissionRequest` hook | Fires an external notification when Claude waits | Redirects *where* a human sees it; the decision still needs making |
| Headless | `-p` / `--print`, `--output-format stream-json` | No TTY; an unanswered `AskUserQuestion` stalls/denies unless a programmatic answerer is wired | Docs pattern: pre-approve with `--allowedTools`+`--permission-mode` so unattended runs don't block |

Sources: [permissions](https://code.claude.com/docs/en/agent-sdk/permissions),
[user-input](https://code.claude.com/docs/en/agent-sdk/user-input),
[hooks](https://code.claude.com/docs/en/hooks). **Uncertain:** the exact `--permission-prompt-tool`
MCP return shape is described via the headless/permissions pages rather than a dedicated
flag-reference page — confirm before coding against it.

### 2b. OpenAI Codex CLI

Two orthogonal axes: `approval_policy` (when it asks) and `sandbox_mode` (how far it acts
without asking). Approval/clarification surfaces through `request_user_input`; plans through
`update_plan`. Sources:
[approvals & security](https://developers.openai.com/codex/agent-approvals-security),
[config](https://developers.openai.com/codex/config-reference),
[sandboxing](https://developers.openai.com/codex/concepts/sandboxing).

| Mechanism | Exact name | Effect | Tradeoff |
|---|---|---|---|
| Approval | `approval_policy="untrusted"` (`--ask-for-approval untrusted`) | Auto-runs known-safe reads; prompts for mutation | Most human-in-loop |
| Approval | `approval_policy="on-request"` | Model asks when it hits a sandbox block (Auto preset default) | Balanced; still asks |
| Approval | `approval_policy="on-failure"` | Run in sandbox, ask only after a failure | **Uncertain — reported deprecated but still in CLI help** |
| Approval | `approval_policy="never"` (`--ask-for-approval never`) | No approval prompts; sandbox-blocked ops **fail back to the model** instead of prompting | Suppresses the human prompt; "fail rather than ask" (safer than blind-approve, but no human answer) |
| Preset | `--full-auto` | approvals off + `workspace-write` sandbox | **Uncertain — reported deprecated** for explicit flags |
| Bypass all | `--dangerously-bypass-approvals-and-sandbox` (`--yolo`) | Disables approvals AND sandbox; full user permissions | Blind approve-all, no sandbox; docs restrict to isolated VMs |
| Sandbox | `sandbox_mode=read-only\|workspace-write\|danger-full-access` (`--sandbox`) | Bounds what runs without asking | Prefer `never`+`workspace-write` over full bypass |

Known reliability gap: the VS Code Codex extension does not perfectly honor
`approval_policy="never"`/`"untrusted"` ([codex #5038](https://github.com/openai/codex/issues/5038),
[#5443](https://github.com/openai/codex/issues/5443)).

### 2c. Aider, opencode, Cursor (brief)

- **Aider** — `--yes-always` (env `AIDER_YES_ALWAYS`) auto-answers "yes" to every confirmation;
  pair with `--message`/`--message-file` for CI. Tradeoff: blind approve-all, no "answer the
  question" path, no first-class `--no`. It notably does *not* auto-run suggested shell commands
  ([#3903](https://github.com/Aider-AI/aider/issues/3903)).
  [options](https://aider.chat/docs/config/options.html),
  [scripting](https://aider.chat/docs/scripting.html).
- **opencode** — `opencode.json` `permission` block, per-tool `allow|ask|deny` with pattern
  objects; CLI `--auto` auto-approves anything not denied. All-or-per-pattern suppression, no
  programmatic answer. [permissions](https://opencode.ai/docs/permissions/).
- **Cursor CLI** — headless `agent -p`; `permissions.allow`/`.deny`, `--force` to run
  write/shell without prompts, `--approve-mcps` to auto-approve all MCP tools. Blanket
  auto-approve / pattern suppression. [permissions](https://cursor.com/docs/cli/reference/permissions).

### 2d. Agent frameworks (brief): first-class interrupt/resume

- **OpenAI Agents SDK** — tool `needsApproval` (bool/fn); run pauses → `result.interruptions`;
  resolve with `state.approve/reject` (sticky `alwaysApprove`) then resume. `needsApproval:false`
  → never asks; approve/reject can be code, not a human.
  [human-in-the-loop](https://openai.github.io/openai-agents-js/guides/human-in-the-loop/).
- **LangGraph** — `interrupt(payload)` pauses; resume with `Command(resume=value)`. Not calling
  `interrupt()` removes the pause; a caller can auto-resume. [docs](https://docs.langchain.com/oss/python/langchain/human-in-the-loop).
- **AutoGen / AG2** — `UserProxyAgent(human_input_mode="ALWAYS"|"TERMINATE"|"NEVER")`; `"NEVER"`
  fully suppresses (relies on auto-reply cap / termination). [docs](https://microsoft.github.io/autogen/0.2/docs/reference/agentchat/user_proxy_agent/).

## 3. The synthesis, and where fak sits

Every native knob is one of **two shapes, and neither decides the question from evidence:**

1. **All-or-nothing suppression** — remove or blanket-approve the prompt globally
   (`bypassPermissions` / `--dangerously-skip-permissions` / `dontAsk` / `disallowedTools`,
   Codex `never` / `--yolo`, Aider `--yes-always`, opencode `--auto`, Cursor `--force`, AutoGen
   `NEVER`). The agent then either blindly approves or hard-denies the question, and in the
   deny/suppress cases **can no longer ask even when it genuinely should** — it guesses or fails.
2. **Per-harness hand-wiring** — a callback/hook/MCP tool/interrupt (`canUseTool`, `PreToolUse`,
   `--permission-prompt-tool`; Codex granular per-tool policy; Agents SDK `needsApproval`;
   LangGraph `interrupt`+`resume`) that lets *your code* answer — but only with whatever logic
   you hand-write into it: a policy list, a fixed answer, a relayed Slack click. None read the
   *task's own evidence* (the diff, the repo norms, a witness/oracle) to decide whether the
   question is even necessary and what the answer should be.

**fak is the missing third shape: evidence-first and harness-agnostic.** The guard's Rung 2
already normalizes each harness's native gate into one `OperatorQuestion` and decides the
decidable part from read-only oracles — the same fold whether the child is Claude Code or Codex.
That is what neither native shape offers. So fak does not *replace* the native knobs; it
**composes above** them, and the survey sharpens three concrete design moves:

- **Inline interception beats after-the-fact catching.** Today Rung 2 reads the transcript at
  the *Stop* hook — after the turn already ended on the ask. But Claude Code guarantees
  `AskUserQuestion` / `requiresUserInteraction` tools always fall through to `PreToolUse`, and a
  `PreToolUse` hook returning `permissionDecision:"allow"` with `updatedInput.answers` **answers
  the question inline, before the turn stalls.** fak already installs a `PreToolUse` commit-gate
  hook — the same seam can carry an evidence-first `AskUserQuestion` answerer.
- **A first-class dial, because operators expect one.** Every harness exposes a named knob to
  turn its ask-user handling off or tune it. fak should too: `fak manage --operator-question
  off|shadow|warn|enforce`, split from the linguistic `--operator-directed` rung, so the
  structured adjudicator is independently disableable. (Feeds the mode-debt census, #4397.)
- **Native-suppression awareness.** If the operator already set `disallowedTools:AskUserQuestion`
  or `approval_policy=never`, the question **never surfaces** — fak's interception is moot and the
  agent will guess. fak should detect that config and either say so or adjust, and a compatibility
  matrix (each harness's native knob × fak's layer) should be documented so the two do not fight.

## 4. Tickets (filed under epic #4881; builds on the closed epic #4701)

A distinct **harness-integration / operability** theme, separate from #4701's pre-adjudication
*build*. Filed 2026-07-15 as epic **#4881** with children **#4883–#4887**:

- **T1 (#4883) — wire the PLAN_APPROVAL resolver into the guard (fix the closure-lag).** Build a concrete
  `planresolve.OracleSet` in `cmd/fak` (`TreeDisjoint` via `dos arbitrate`; `DirectionAllowed`
  via architest / gitgate / core-lock; `DoneVerifiable` via `dos verify`) and assign
  `guardOperatorQuestionPlanResolver`, so a real `ExitPlanMode` / `update_plan` gate is decided
  from evidence instead of always returning `PLAN_ORACLES_UNAVAILABLE`. DoD: an
  install-time-wired resolver; a collision plan auto-refuses, a reversible+witnessable plan
  auto-approves, an irreversible/unwitnessable step escalates — end-to-end through the Stop hook,
  not just the pure package. (References #4704, epic #4701; adjacent to the plan-regime epic
  #2390.)
- **T2 (#4884) — first-class `fak manage --operator-question off|shadow|warn|enforce` dial.** Split Rung 2
  from the linguistic `--operator-directed` flag so the structured adjudicator is independently
  disableable/tunable, mirroring the native ask-user knobs (§2). Keep the operator-absent cap.
  DoD: the flag exists, is documented in `guard_help`, defaults to today's behavior, and `off`
  fully disables Rung 2 without touching Rung 1. (Feeds mode-debt #4397.)
- **T3 (#4885) — PreToolUse inline answerer for `AskUserQuestion` / `ExitPlanMode`.** Add a `PreToolUse`
  rung that runs the evidence-first resolver and, on `TAKE_OBVIOUS`/`AUTO_APPROVE`, returns
  `permissionDecision:"allow"` with `updatedInput.answers` so the question is answered *before*
  the turn ends — catching it at the source instead of at the Stop hook. Escalate the residue
  unchanged. DoD: a Claude Code `AskUserQuestion` with a git-decidable option is answered inline
  in a headless run; the residue still surfaces. (New; enabled by the §2a fall-through guarantee.)
- **T4 (#4886) — native-suppression awareness + harness-config compatibility matrix.** Detect when the
  child's config already suppresses the native gate (`disallowedTools:AskUserQuestion`,
  `approval_policy=never`) and surface that fak's interception is moot; document the matrix of
  each harness's native knob × fak's layer so they compose rather than fight. DoD: a detector +
  a doc table; no false "fak is handling this" when the harness will never emit the gate.
- **T5 (#4887) — expand the CLARIFY / CHOOSE_APPROACH oracle set beyond git-isolation.** Add the §3 "H3
  ladder" oracles to `guardOperatorQuestionClarifyResolver`: repo-fact resolution
  (grep / `git` / `dos verify`) and scorecard-axis dominance, so a knowable clarify resolves
  instead of defaulting to `FRESH_CONTEXT`. DoD: a repo-fact question resolves without asking; a
  reversible dominated choice decides on a recorded axis. (References #4703.)

## 5. Honest fences

- **Composing above the native knobs, not replacing them.** If an operator wants blunt
  `bypassPermissions`, that is their call; fak's layer is for the runs that want the *smaller,
  sharper* human question, not zero questions.
- **Reversibility and authority stay human, permanently** (inherited from
  [`CONCEPT-AGENTIC-OPERATOR-QUESTIONS-2026-07-10.md`](CONCEPT-AGENTIC-OPERATOR-QUESTIONS-2026-07-10.md)
  law 2 and `operatorresolve.authorityFork`). An `AUTHORITY_FORK` / irreversible plan step
  escalates; it is never auto-answered to avoid asking.
- **T3 answers only what an oracle settles.** Injecting `updatedInput.answers` from a *guess*
  would be worse than the native suppression it improves on — the whole point is that the answer
  is witness-backed, and the residue still escalates.
- **No new brain.** T1/T3/T5 are folds over oracles that already exist (`planresolve`,
  `operatorresolve`, `dos verify`/`arbitrate`); if a ticket starts proposing a new
  model-judgment authority instead of an evidence fold, it has drifted off this note.
- **Two survey claims are uncertain** (Codex `on-failure`/`--full-auto` deprecation; the
  `--permission-prompt-tool` return shape) — re-verify against installed versions before coding.
