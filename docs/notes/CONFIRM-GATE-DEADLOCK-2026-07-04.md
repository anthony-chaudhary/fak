---
title: "The preview-confirm gate on raw gh is an unsatisfiable token loop (but fak issue create is the sanctioned sidestep)"
description: "Raw model-proposed `gh issue create` via Bash is escalated by design (REQUIRE_WITNESS/ESCALATE); the preview-confirm recovery it advertises ('re-propose byte-identical + add _fak_confirm') never converges — a fresh token issues every attempt. The correct path is the compiled `fak issue create` verb, which sidesteps the classifier the same way `fak commit` sidesteps `git push`. Real, narrower defects in the gate remain."
---

# The preview-confirm loop on raw `gh` — and the sanctioned way around it

**Filed:** 2026-07-04, from a live session weaving Slack into userland value.

## Correction up front (what I got wrong, then right)

The escalation of raw `gh issue create` is **working as designed**, not a bug:
`internal/adjudicator/reversibility.go` deliberately escalates model-proposed raw
`gh` (REQUIRE_WITNESS/ESCALATE). The sanctioned path is the compiled
**`fak issue create`** verb (`cmd/fak/issue_create.go`), which shells to `gh` from
the trusted binary — the same way `fak commit` / `fak sync push` sidestep the
`git push` family. I hit the wall only because my local `fak.exe` was **stale**
(built before `fak issue create` landed); rebuilding and filing through the verb
worked first try → issue **#2650**.

The **real** defect is narrower: the raw-`gh` refusal advertises a
preview-confirm recovery that is itself unsatisfiable, and points nowhere useful.
A refusal that escalates should say "use `fak issue create`", not hand out a
confirm token that can never converge.

**Severity of the remaining defects:** medium — the sanctioned path exists and
works; the gate's advertised recovery is misleading and burns agent turns.

## Symptom

An outward-facing Bash call (`gh issue create …`) is refused with:

> `Bash (REQUIRE_WITNESS/ESCALATE) preview-confirm gate … re-propose it
> byte-identical … with "_fak_confirm":"fak-XXXX" added … A preview-confirm
> refusal is a pause, not a denial.`

Following that instruction exactly — re-proposing the same command with the
just-issued `_fak_confirm` key added and nothing else changed — is refused
**again**, with a **new** token. Observed four times in one session:

| attempt | command shape | token issued |
|---|---|---|
| 1 | `SC=$(cygpath …); gh issue create …` | `fak-2a7d84a4a332fef9` |
| 2 | same | `fak-660c21ab4cc16ac5` |
| 3 | `gh issue create --repo … --body-file …` (no prefix) | `fak-7e7d475475723f9d` |
| 4 | same, abs `--body-file` path | `fak-a7e52dee8f7078d5` |

The loop never converges: to use token N you must add the confirm key, and
adding the key yields token N+1.

## Root-cause hypotheses (either is sufficient to cause the deadlock)

1. **The token is hashed over a payload that includes (or is perturbed by) the
   `_fak_confirm` key.** The gate says it "verifies and strips the key before
   dispatch, so the tool never sees it" — but if it hashes the tool input
   *before* stripping, then adding the key changes the hash and invalidates the
   very token it just handed back. The hash MUST be computed over the command
   text with the `_fak_confirm` key excluded, and MUST be stable across
   byte-identical command text.

2. **A nonce/timestamp is folded into the token.** If so, no re-proposal can ever
   match, and "re-propose byte-identical" is not a reachable recovery at all.

A stability test the fix must pass: propose identical command text twice; the
issued token must be identical. In this session, four proposals (two pairs of
identical text) issued four distinct tokens — so the token is NOT a pure
function of the command text today.

## Secondary defects surfaced

- **`REQUIRE_WITNESS` is not in the DOS closed refusal vocabulary.**
  `dos_check_reason REQUIRE_WITNESS` → `known:false, category:UNCLASSIFIED,
  "Do NOT emit it."` The gate refuses with an UNCLASSIFIED reason token — the
  exact prose-drift the DOS kernel exists to kill. The refusal should carry a
  declared reason (add it to `dos.toml [reasons]`, or reuse an existing
  OPERATOR_GATE-category token).
- **No escape hatch after N failed confirms.** The gate should escalate to a
  human approve/deny prompt (or state the specific reachable next step) rather
  than re-emit another doomed token. An unsatisfiable "pause" is worse than a
  hard deny: it burns turns and reads as agent flailing while the agent follows
  the documented protocol exactly.
- **`gh issue create` is over-gated.** Filing a GitHub issue is routine and
  reversible (issues edit/close). It belongs on an allow-list with a light
  confirmation, not a cryptographic preview-confirm handshake that can deadlock.

## Repro

1. In an agent session with the fak guard's outward-facing preview-confirm gate
   active, propose `gh issue create --repo <r> --title <t> --body-file <f>`.
2. Observe the `REQUIRE_WITNESS/ESCALATE` refusal + a `_fak_confirm` token.
3. Re-propose byte-identical with `"_fak_confirm":"<that token>"` added.
4. Observe a second refusal with a **different** token. Loop never converges.

## Fixes (in priority order)

1. **Point the escalation at the sanctioned verb.** When raw `gh issue create`
   (or any `gh` write) is escalated, the refusal should say "file it with
   `fak issue create` (compiled sidestep)" — the way a `git push` escalation
   should name `fak commit`. That single message change turns a dead end into a
   one-step recovery and would have saved this entire loop.
2. **Fix or retire the preview-confirm token on this path.** ✅ **LANDED**
   (`fix(adjudicator): bind reversibility confirm token to the command, not the
   description`). Root cause pinned: `ReversibilityConfirmToken` hashed over ALL
   args minus the confirm keys, so a Bash call's mandatory free-text
   `description` — which Claude Code regenerates every turn — rotated the token
   even for a byte-identical command. Fix: `argsForToken` now also excludes
   incidental annotation keys (`description`/`explanation`) from the hash, so the
   token binds only to the effect-bearing args. Regression
   `TestReversibilityConfirmTokenIgnoresDescriptionDrift` replays the shape (same
   command, reworded description → confirm now lands) and enforces the stability
   test named below. Re-witnessed by session f0e7ac0f (deleting
   `tools/new_leaf.py` looped on this exact rotation).
3. **Declare the refusal reason** in the closed vocabulary so `dos_check_reason`
   recognizes it (today `REQUIRE_WITNESS` → `known:false, UNCLASSIFIED`).

**Filed:** fix 1 is tracked as **#2651** (name the `fak issue create` sidestep in
the refusal). Fixes 2–3 are noted there as separate follow-ons.

## Work this surfaced (now filed via the sanctioned path)

- **Keystone spine issue #2650** — "control the fak fleet from Slack: the
  read-only inbound door spine" (`internal/chatops` + `fak chatops`, v0 read verbs
  status/health/verify). Contract-clean (`fak issue contract --file` →
  `ok:true, dispatchable:1`), filed through `fak issue create`. It is the live
  umbrella for the open door children #2264/#2265/#2266 after epic #2259 was
  closed. This is the proof the sidestep works — the raw-`gh` loop was the only
  thing that ever blocked, and it was never the right path.

## Non-blocking note

`gh` itself is fine — authenticated, read-only calls succeed. The deadlock is
specific to the outward-facing WRITE path's confirm handshake.
