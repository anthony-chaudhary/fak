---
title: "In-flight repair of a sanctioned compiled sidestep (auto-substitute `git push` → `fak sync push`) — design + config"
description: "The reversibility gate NAMES the sanctioned sidestep (`fak sync push`) but only as advisory text; the agent must re-issue by hand. The in-flight rewrite-and-execute mechanism already exists (the TRANSFORM verdict) but is scoped to same-tool argument edits. This note designs a conservative, opt-in auto-substitution for the safe subset (bare `git push` with no force/delete/refspec) that reuses the existing apply seam, why it is sound (the pause is theater for the safe subset, load-bearing for the dangerous variants which stay paused), the config surface, and the implementation constraints (internal/adjudicator is core-locked)."
---

# In-flight repair of a sanctioned compiled sidestep

**Filed:** 2026-07-15, from a live session where a bare `git push` was refused
(`REQUIRE_WITNESS/ESCALATE` preview-confirm gate) and the operator asked: *if
`fak sync push` is such an obvious sanctioned equivalent that the guard already
names it, why doesn't the guard repair the call in flight instead of making the
agent re-issue it?*

This note answers that, and designs the conservative form of the capability.

## The honest "why not today"

It is **not** that in-flight repair is impossible — the mechanism exists and runs
live. Three stacked facts explain the current behavior:

1. **In-flight rewrite-and-execute already exists, but only for _arguments_.**
   The `abi.VerdictTransform{TransformPayload{NewArgs}}` verdict is emitted by the
   adjudicator and applied at `internal/gateway/adjudicate_proposed.go:446-454`
   (`tc.Function.Arguments = repaired`), then dispatched — on the execute paths
   (`internal/gateway/mcp.go:357`, `internal/gateway/http.go:1161`,
   `internal/kernel/kernel.go` Submit) the gateway runs the rewritten call itself.
   Three producers today: secret-field redaction, `attenuateCLIGrammar` (gh/git
   read-only grammar rewrite, `internal/adjudicator/cli_grammar.go`), and
   `stripConfirmationTransform` (drops `_fak_confirm` after a valid confirm). All
   rewrite the **same tool's args**.

2. **The `git push → fak sync push` substitution is deliberately left as _text_.**
   The reversibility rung returns `VerdictRequireWitness` (a HOLD); the gateway
   **drops** the call (`adjudicate_proposed.go` `default: dropped++`); the
   sanctioned verb rides along only as advisory prose:
   `reversibilityFamily.hint` (`internal/adjudicator/reversibility.go:245`) →
   `Meta["dry_run_hint"]` → `Detail["remedy"]` → `remedyNote`
   (`internal/gateway/reversibility_note.go:87-93`). The agent must re-issue.

3. **`#4428` ("error affordance rewrites") is _not_ this** — despite "rewrites" in
   the subject it is a closed 5-entry `reason-code → sentence` map
   (`internal/gateway/error_affordance.go`) used purely as fallback note text; it
   never touches the git-push case and executes nothing.

Accurate statement: **the guard can rewrite-and-run a call in flight, but that
power is scoped to argument edits and was never wired to the sanctioned-alternative
path.**

## What already exists (the ~90%)

| Piece | Location | Role for a repair feature |
|---|---|---|
| Apply-and-dispatch seam | `internal/gateway/adjudicate_proposed.go:446-454` + kernel/execute paths | Rewrite `command` arg → dispatch. Reusable unchanged. |
| `{family → runnable command}` table | `internal/adjudicator/reversibility_remedy_test.go:19` (`"git-push": {"fak sync push", "git push --dry-run"}`) | The concrete rewrite target — **test-only** today, with invariants that each target is itself gate-clean (`TestFamilyRedirectCommandsAreReversible`) and complete (`TestFamilyRemedyTableIsComplete`). |
| `compiledSidestepRemedy` | `internal/gateway/refusal_notes.go:64-66` | Machine-distinguishes a `fak `-verb sidestep (git-push, issue-create-tool, slack, messaging-tool) from prose-only remedies (mail, webhook, sql-drop, …). The predicate to gate on. |
| Safe-subset flag parser | `gitDryRunPreview` / `hasWhatIfFlag` (`internal/adjudicator/reversibility.go:401-449`) | Already parses git-push flags structurally, dash-preserving. The boundary detector is here. |

## The two real gaps

1. **The rewrite target is stranded in a `_test` file.** `familyRemedyCommands`
   must be promoted into the production `reversibilityFamily` struct as a
   structured `rewrite` field distinct from the human `hint`.
2. **The ABI has no `NewTool`.** `abi.TransformPayload` carries only `NewArgs`
   (`internal/abi/types.go:236-239`), so substituting a *different tool* (e.g. an
   MCP `git_push` tool → shell `fak sync push`) is unrepresentable. **This does
   not block the actual trigger:** bare `git push` and `fak sync push` are both
   **Bash** calls — substitution is a pure `NewArgs` rewrite of the Bash `command`
   string. The common case needs **zero ABI change**.

## Why auto-substitution is sound here (not a capability-wall removal)

`SafePush` (`internal/safesync/safepush.go`) pushes the **current branch**
(`PushOptions.Branch` defaults to the current branch, :62/:162-168), non-force,
fast-forward-only, with a transient-race retry; it never force/merge/reset/stash.
So **bare `git push` ≈ `fak sync push`** for the common case. Where they differ is
exactly the flag-bearing variants — `--force`/`-f`/`--force-with-lease`,
`--delete`/`-d`, `--tags`, `--all`, `--mirror`, `--prune`, or an explicit
`src:dst` refspec — and those are **structurally decidable from the command
string** (the `gitDryRunPreview` parser already reads git-push flags without
shelling out).

For the bare safe subset, the pause is theater: the agent either re-runs
`fak sync push` or adds `_fak_confirm` to the identical `git push` — both proceed
to an outward push anyway. The pause carries real safety value only for the
**dangerous** variants, and those stay paused under the predicate below.
Auto-repair therefore removes friction where the gate is theater and preserves it
where it is load-bearing. Auto-laundering a `--force`/`--delete`/refspec push into
a non-force verb would **silently change intent** — worse than refusing — so it is
explicitly excluded.

## Design (conservative, opt-in)

**Fire condition** — in the reversibility rung (`internal/adjudicator/decide.go:517-537`),
substitute instead of holding **iff all** hold:

- the matched family declares a compiled `fak` sidestep (`compiledSidestepRemedy`), **and**
- the refused call is a **Bash** command (same-tool rewrite; MCP-tool variants stay a HOLD until a `NewTool` ABI field lands), **and**
- the command is in the family's **safe-equivalent subset** — for `git-push`: bare
  `git push [remote]` with none of `--force`/`-f`/`--force-with-lease`,
  `--delete`/`-d`, `--tags`, `--all`, `--mirror`, `--prune`, or an explicit
  `src:dst` refspec (reuse the `gitDryRunPreview` flag parser).

Then emit `VerdictTransform` whose `NewArgs` rewrites `command` → `fak sync push`,
instead of `reversibilityGateVerdict`. Everything downstream (apply, dispatch, the
"repaired arguments for: Bash" note) is reused. Anything outside the safe subset
stays a `RequireWitness` pause.

**Config surface** (matching existing knob idioms):

- `reversibilityFamily.rewrite` — a structured `{safeSubset predicate,
  targetCommand}` in `reversibility.go`, promoting `familyRemedyCommands` out of
  `_test`. Existing invariant tests already guarantee targets are gate-clean.
- `FAK_GUARD_AUTOREPAIR` env — `off` (default) / `sidestep` (auto-substitute
  compiled sidesteps). **Opt-in**, because it changes outward-facing behavior;
  mirrors `FAK_GUARD_STOPS_MODE`.
- Keep the annotation even when auto-applied:
  `repaired arguments for: Bash — substituted sanctioned verb 'fak sync push' for
  'git push'`, so the transcript shows what happened (auditability, and the model
  learns to reach for the verb directly next time).

**Guardrails carried over:**

- Do not reintroduce the confirm-token loop (`docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md` / #3306).
- Respect the quote-aware payload-scan view — a `git push` inside a commit message
  or grep pattern must not trip the rewrite.
- Stay consistent with `adjudicateProposedTurn`
  (`internal/gateway/adjudicate_proposed.go:469-497`), the loop-body-witness second
  pass that can re-demote admitted calls, or a substitute could be silently reverted.

## Implementation constraints

- `internal/adjudicator` is a **hard-self core-locked** surface
  (`internal/corelockaudit/corelockaudit.go:64`): editing `decide.go` /
  `reversibility.go` requires `fak commit --core-lock-maintenance-witness` and a
  diff-witnessed commit audit. Scope the family-table + rung change as a single
  witnessed commit.
- Adding a `reversibilityFamily.rewrite` field forces the completeness invariant
  (`TestFamilyRemedyTableIsComplete`) to cover it — a new family must declare its
  rewrite (or an explicit "no auto-rewrite") or the test reds.
- The gateway apply seam and the `FAK_GUARD_AUTOREPAIR` env read are **not**
  core-locked (they live in `internal/gateway` / `internal/guardvars`), so the
  wiring can land separately from the core-locked classifier change.

## Suggested increments

1. Promote `familyRemedyCommands` → a production `reversibilityFamily.rewrite`
   (safe-subset predicate + target), git-push only. (core-locked)
2. In the reversibility rung, emit the substitute `VerdictTransform` when
   `FAK_GUARD_AUTOREPAIR=sidestep` and the safe-subset predicate passes. (core-locked)
3. Gateway/annotation wiring + the `FAK_GUARD_AUTOREPAIR` env knob. (not core-locked)
4. Extend to the other compiled-sidestep families (issue-create-tool, slack,
   messaging-tool) once the Bash-command shape is proven; the MCP-tool variants
   wait on a `NewTool` ABI field.
