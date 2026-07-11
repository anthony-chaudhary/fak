---
title: "Auto-complain, done right: witness-by-default now, objective auto-DRAFT next (2026-07-10)"
description: "The operator asked for `fak complain` to fire 'by default when needed'. Auto-LIVE-filing on every refusal is the wrong default — a false-positive DENY is byte-identical to a correct one in the decision journal, so 'when needed' cannot be read off the journal, and auto-filing an outward GitHub issue per refusal is spam the guard-accuracy loop would then have to un-learn. This note splits the ask into the correct two: (A1, SHIPPED with this note) make the appeal the agent already copy-pastes WITNESS-BOUND by construction — complaintHint emits the refused call's args_digest as an exact `--args-digest` selector, so `--from-journal` attaches the witnessed verdict instead of hitting SelectDenial's ambiguous→no-witness path on a busy journal; and (A2..A5) an OBJECTIVE 'when needed' detector — refused-then-admitted-same-digest and remedy-mismatch, both computable from evidence not self-report — that STAGES a dry-run draft (fak complain is already dry-run by default) for one-call agent confirmation, feeding the existing `guard-accuracy --complaints` advisory intake. Auto-DRAFT, never auto-LIVE."
---

# Auto-complain, done right (witnessed intake)

> Date: 2026-07-10. Status: design note + shipped foundation slice (A1: witness-by-default
> in `complaintHint`). Composes with the appeal channel (`internal/guardcomplaint`,
> `cmd/fak/complain.go`), the accuracy RSI fold (`internal/guardaccuracy`,
> `fak guard-accuracy --complaints`), and the per-session livelock observer
> (`internal/gateway/adjudicate_proposed.go` `annotateToolLivelock`,
> `internal/guardrsi`). Does **not** duplicate or replace any of them.

## 0. The trigger

The operator's ask, verbatim: make `fak complain` fire **"by default when needed."** The
seed for it was concrete: a governed agent hit a `POLICY_BLOCK` / `TERMINAL` refusal it
judged wrong, went to appeal it, and the default `fak complain … --from-journal` it was
handed **filed witness-less** — because on a busy decision journal a bare `--reason`/`--tool`
match is ambiguous, and `guardcomplaint.SelectDenial` refuses to guess which refused call
the agent meant, dropping the witness rather than attaching the wrong one. So the appeal
channel exists, is surfaced in-band on every refusal (`complaintHint`), and is dry-run-safe —
yet its *default* path silently loses the one thing that makes an appeal actionable: the
witnessed verdict.

"By default when needed" is really two asks, and they must not be conflated:

1. **The default appeal must be right** — when an agent *does* appeal, the witness must
   attach by construction, not by the agent hand-selecting a journal seq it cannot see.
2. **"When needed" must be objective** — the loop should proactively surface an appeal only
   on an evidence-backed false-positive signal, never on every refusal.

## 1. What exists today — and the one gap

| Piece | What it is | Evidence | State vs the ask |
|---|---|---|---|
| `complaintHint` | in-band invitation to `fak complain` appended to every DENY note; substitutes concrete `--reason`/`--tool` on a single-denial turn | `internal/gateway/refusal_notes.go` | surfaced everywhere, but **handed a witness-less default** (the A1 gap) |
| `fak complain` | files ONE deduplicating gh issue for a guard false-positive; **dry-run by default** (prints the planned appeal, touches nothing) | `cmd/fak/complain.go:19`, `:137` `mode := "dry-run"` | safe channel already exists; live filing is an explicit opt-in |
| `guardcomplaint.SelectDenial` | binds an appeal to the exact refused journal row; `--from-journal` + `--args-digest`/`--trace-id` selectors | `internal/guardcomplaint` (`SelectDenial`) | refuses an **ambiguous** reason/tool match → no witness; needs a call-identifying selector |
| decision journal | append-only `DECIDE` / `RESULT_DENY` rows carrying `tool`, `args_digest`, `verdict`, `reason` | `internal/guardrsi/recovery_test.go:95-96` | the witness substrate — a DENY row and the later ALLOW/TRANSFORM on the same digest both live here |
| `annotateToolLivelock` | per-session observer that already folds `(trace, tool, args_digest, verdict)` for admitted **and** failed calls | `internal/gateway/adjudicate_proposed.go:247-296` | the natural host for a refused-then-admitted detector — the observations already flow through it |
| `guard-accuracy --complaints` | folds agent-authored complaints as an **advisory** (Soft, never debt) field false-positive signal into the accuracy scorecard → garden / control-pane ratchet | `internal/guardaccuracy/complaints.go:46-60`, `FieldComplaint{Kind,Summary,Occurrences}`, `BuildScorecardWithComplaints` | the RSI intake already exists; it just needs a higher-signal feed |

The channel, the witness substrate, the observer, and the RSI intake **all already exist.**
The gap is (a) the default appeal drops its witness, and (b) nothing turns the objective
false-positive signals the kernel already observes into a staged appeal.

## 2. Why auto-LIVE-filing is the wrong default (the anti-goal)

The temptation is to make every refusal auto-file a `fak complain` issue. That is wrong on
two independent grounds, and the note states them so the epic never drifts back to it:

- **Epistemic: "false positive" is not journal-observable.** A false-positive DENY is
  **byte-identical** to a correct DENY in the decision journal — same tool, same args_digest,
  same reason. The kernel's own honesty/accuracy RSI folds cannot self-detect it; the *only*
  authority on "this refusal was wrong" is the agent that made the call
  (`complaintHint`'s own docstring makes this claim). So "auto-file every refusal" would file
  a false positive **and** a correct refusal with equal confidence — noise, not signal.
- **Outward-facing: it is issue spam.** `fak complain --file` opens a real GitHub issue.
  Auto-filing one per refusal publishes an unbounded stream of unvetted issues that the
  `guard-accuracy` loop would then have to *un-weight* — the loop's whole value is
  triaging real field false-positives worst-first, and drowning it in auto-noise inverts that.
  Filing to an external service is exactly the "confirm first, hard to reverse" class.

The correct default is therefore **auto-DRAFT, never auto-LIVE**: stage the appeal, let the
agent (the sole competent witness) confirm it with one call. Because `fak complain` is
already dry-run by default (§1), a staged draft is *inert* until the agent adds the live flag
— the automation carries zero new outward risk.

## 3. The two objective "when needed" signals

"When needed" must rest on evidence the kernel already holds, not on the model's say-so. Two
signals qualify — both are computable from the decision journal / the live observer, and both
are **self-witnessing** (the kernel's own later behavior contradicts the refusal):

1. **Refused-then-admitted-same-digest.** Within one session, tool `T` with `args_digest D`
   was **DENIED**, and the *same* `(T, D)` was later **ADMITTED** (`ALLOW`) with no intervening
   policy/state change that would explain the flip. The kernel both refused and admitted the
   identical call shape → the refusal is a false-positive candidate by the kernel's own hand.
   The `guardrsi` recovery fold already models the adjacent `RESULT_DENY → DECIDE` transition
   (`recovery_test.go:95-96`), and `annotateToolLivelock` already observes both verdicts keyed
   by `(trace, tool, args_digest)` — so the detector is a fold over signal that already flows.
   - **Critical exclusion:** a `TRANSFORM`-then-admit is the guard *repairing* the call
     (correct behavior) — NOT a false positive. Only a pure `DENY(D) … ALLOW(D)` on the
     byte-identical digest counts. This exclusion is the whole reason the signal is trustworthy.
2. **Remedy-mismatch.** A refusal whose interpolated sanctioned-alternative / remedy asserts a
   danger predicate the refused call's *actual arguments* do not satisfy — e.g. a note that
   says "use `--dry-run` for a recursive delete" on a call whose args carry no recursive/force
   flag. Re-checking the refused args against the remedy's asserted predicate is a pure,
   evidence-only test; a mismatch means the refusal misclassified the call.

Both signals produce a specific `(tool, args_digest, reason)` triple — exactly the
call-identifying selector A1 emits — so a draft built from either is witness-bound for free.

## 4. The shape

```
refusal ──> complaintHint  ─(A1, shipped)─>  appeal is WITNESS-BOUND by default
                                              (`--from-journal --args-digest <D>`)

session journal / annotateToolLivelock
        │
        ├─ refused-then-admitted-same-digest  ┐
        └─ remedy-mismatch                    ├─(A2)─> objective "needed" verdict
                                              ┘            │
                                                           ▼
                                   (A3) STAGE a dry-run draft in-band:
                                   `fak complain --summary "<auto>" --reason R
                                        --tool T --from-journal --args-digest D`
                                   agent confirms with one call (never auto-live)
                                                           │
                                                           ▼
                                   (A4) confirmed drafts ──> FieldComplaint{Kind,Summary,
                                   Occurrences} ──> `guard-accuracy --complaints` advisory fold
                                   ──> garden / control-pane ratchet (worst-first, Soft)
```

**A1 is shipped with this note.** `complaintHint` now emits the refused call's `args_digest`
as an exact `--args-digest` selector on a single-denial turn, so the default copy-paste appeal
binds its witness by construction instead of filing witness-less. A mixed turn keeps the
`<REASON>`/`<TOOL>` placeholders and emits no selector (it cannot attribute one call's digest
to another). Pinned by `TestComplaintHintBindsWitnessSelectorOnSingleDenial` /
`TestComplaintHintKeepsPlaceholdersOnMixedTurn` in `internal/gateway/refusal_notes_test.go`.

## 5. Epic + DoD child tickets

**Epic — Auto-complain, witnessed: make the appeal right by default, and surface it only on
an objective signal.** Composes with `guardcomplaint` (channel), `guardaccuracy` (RSI intake),
and `guardrsi`/`annotateToolLivelock` (the observer). Never auto-files live.

- **A1 — witness-by-default in `complaintHint`.** *(this note ships it.)*
  DoD: a single-denial appeal emits `--from-journal --args-digest <D>` so `SelectDenial` binds
  the witnessed verdict; a mixed turn emits no selector and keeps placeholders; digest-less
  denials fall back to the bare form (no dangling flag). Unit-pinned; `go test ./internal/gateway/`
  green for the refusal-note tests. No hot-path behavior change beyond the emitted hint text.
- **A2 — objective "needed" detector (pure).** DoD: a pure fold over the session's decision
  rows / `annotateToolLivelock` observations that emits a `(tool, args_digest, reason, signal)`
  candidate for **refused-then-admitted-same-digest** (excluding `TRANSFORM`-repair) and
  **remedy-mismatch**; unit tests cover the ALLOW-flip positive, the TRANSFORM-repair negative,
  and the remedy-predicate mismatch. No I/O, no filing.
- **A3 — stage a dry-run draft in-band (agent-confirm).** DoD: when A2 fires, `complaintHint`
  (or an adjacent note renderer) surfaces a **pre-filled, witness-bound, dry-run** `fak complain`
  command the agent can run/confirm in one call; the note states it is dry-run and that filing
  is the agent's explicit opt-in. Never emits a live-file command. Flag-gated first.
- **A4 — feed the accuracy intake.** DoD: a confirmed appeal serializes to
  `FieldComplaint{Kind,Summary,Occurrences}` (`internal/guardaccuracy/complaints.go`) so it
  flows through `guard-accuracy --complaints` into the scorecard/garden ratchet as an advisory
  (Soft, never debt) worst-first signal; `Occurrences` carries the recurrence weight for a
  re-fired candidate. Witnessed end-to-end against a fixture journal.
- **A5 — de-dup + rate-fence.** DoD: the same `(tool, args_digest, reason)` never stages twice
  per session; a per-session cap bounds draft volume; a `log()`-equivalent surfaces what was
  suppressed so a silenced draft is never mistaken for "nothing to appeal." Grade the fold cost
  (must be O(1) amortized per adjudication).

## 6. Honest fences

- **A1 ships correctness, not automation.** It makes the appeal the agent *already* chooses
  witness-bound; it does not proactively appeal anything. The proactive surface is A2–A3.
- **Auto-DRAFT is the ceiling; auto-LIVE is out of scope, permanently.** The epic never files a
  GitHub issue without the agent's explicit confirmation — §2 is a standing constraint, not a
  first-cut simplification to be relaxed later.
- **The detector can only be a *candidate* generator.** Even refused-then-admitted-same-digest
  can have a legitimate cause (a genuine state change between the two calls that the digest
  doesn't capture — e.g. a lease acquired in between). A2 therefore emits *candidates* for the
  agent to confirm, and the agent's confirmation — not the detector — is the witness that
  reaches the RSI intake. The detector narrows attention; it never adjudicates truth.
- **The intake stays advisory.** Confirmed complaints fold into `guard-accuracy` as a Soft,
  never-debt signal (its existing contract). This note does not propose promoting a complaint
  to a build-gating debt; over-refusal is triaged worst-first, not enforced.
- **`args_digest` is a hash, not the args.** The selector A1 emits carries a `sha256:` digest,
  never the call's arguments, so surfacing it in-band leaks nothing sensitive.
