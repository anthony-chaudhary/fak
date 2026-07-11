---
title: "Operating the operator-questions: inventory + a safe way for an agentic process to answer the harness's human-gates (2026-07-10)"
description: "A harness operator-question — plan approval, a clarifying AskUserQuestion, a tool-permission prompt, a confirm-before-irreversible pause — is a decision request with the evidence stripped out: the harness hands a human the choice because IT has neither the authority nor the witness to decide. This note (1) inventories every operator-style question the Claude Code harness and this repo's own guard/kernel layer raise, (2) shows the fak/DOS inversion — re-attach the evidence, decide from it, and hand the human only the residue no witness can settle, as a closed-vocabulary reason with an appeal channel, never as prose — and (3) scopes the work. Finding: the harness already delegates several of these to the agent (EnterPlanMode, PushNotification, ScheduleWakeup, mid-task continue), and the repo already evidence-operates two more (tool-permission via repo_guard; confirm-before-irreversible via REQUIRE_WITNESS). The genuine frontier is exactly two questions — PLAN APPROVAL and CLARIFY/CHOOSE-APPROACH — and both decompose into an evidence-decided majority plus a small, precisely-named human residue. Composes with the auto-complain recovery note (the appeal half); ships no code, names the first slice."
---

# Operating the operator-questions (inventory + safe-operation design)

> Date: 2026-07-10. Status: design + scoping note. Ships **no code**; it inventories the
> surface, gives the per-question decision procedure grounded in primitives that already
> exist, and scopes the build. Composes with — and does not duplicate —
> [`CONCEPT-AUTO-COMPLAIN-WITNESSED-INTAKE-2026-07-10.md`](CONCEPT-AUTO-COMPLAIN-WITNESSED-INTAKE-2026-07-10.md)
> (the *recovery* half: how a wrong autonomous decision is appealed), the refusal
> vocabulary (`dos.toml [reasons]`, 53 tokens today), the DOS evidence oracles
> (`dos verify` / `arbitrate` / `commit-audit` / `status` / `review`, plus the
> `dos_recall` memory-re-verification oracle), the
> `repo_guard` PreToolUse hook (`internal/repoguard`, `cmd/repoguard`), and the reversibility
> confirm gate (`internal/adjudicator/reversibility.go`).

## 0. The ask

Verbatim intent, three parts:

1. **Inventory** every operator-style question a harness proposes — "approve this plan?" and
   its siblings.
2. **Design a safe way for an agentic process to *operate* those** instead of blocking on a
   human.
3. **Think deeply and scope the work** — produce shippable increments, not a manifesto.

## 1. What an operator-question actually is

Every one of these prompts has the same shape: **a decision request with the evidence
stripped out.** "Approve this plan?" "Allow this `Bash` call?" "Which approach did you mean?"
"This is hard to reverse — confirm?" In each, the harness is saying *I have neither the
authority nor a witness to settle this, so I am handing it to a human.* The human is being
asked to supply, from their head, a judgment the harness could not derive from evidence.

The fak/DOS thesis is the exact inversion, and the repo already applies it to the *kernel's*
own gates: **re-attach the evidence, decide the decidable part from a witness, and hand the
human only the residue no witness can settle — as a closed-vocabulary reason carrying a
machine-checkable `fix`, never as free-text prose.** That is precisely what the 53-token
refusal vocabulary is: every reason is simultaneously *emittable* (a producer stamps it),
*verifiable* (an oracle can re-check the condition it names), and *refusable* (the loop routes
it to a replan or an operator). A "no" is a first-class, auditable value instead of a dead end.

The gap this note closes: that inversion has been applied to the **guard/kernel** operator-
questions (lease collisions, ship claims, git-trunk hazards, TOON auto-fire) but **not** to
the **generic harness** operator-questions (plan approval, clarifying questions, permission,
confirm). Those still block a human by default. This note maps them onto the same machinery.

## 2. The inventory

### 2a. Generic harness operator-interrupts (Claude Code)

| # | Question the harness raises | Surface | The decision underneath | Current default | Reversible? / blast |
|---|---|---|---|---|---|
| H1 | **"Approve this implementation plan?"** | `ExitPlanMode` | Is the plan's scope, direction, and success-criterion sound before code is written? | **Blocks** for human sign-off | Plan is inert; the code it authorizes may not be |
| H2 | "Is this task big/ambiguous enough to plan first?" | `EnterPlanMode` | Should I stop and plan? | **Agent decides** (already agentic) | Reversible |
| H3 | **"Which approach / what did you mean?"** | `AskUserQuestion` | Resolve an ambiguity or pick among options | **Blocks** for human answer | Asking is reversible; the choice may commit |
| H4 | "Allow this `Bash`/`Edit`/`Write`/MCP call?" | permission modes / PreToolUse | Is this specific tool call safe to run? | Prompts unless mode allows | Per-call; ranges reversible→destructive |
| H5 | "This is hard to reverse / outward-facing — confirm?" | proceed/confirm guidance | May I take an irreversible or published action? | **Confirm first** | Irreversible / outward by definition |
| H6 | "Should I keep going?" | mid-task proceed check | Continue to the next step, or stop? | Usually proceeds | Reversible per step |
| H7 | "Is this worth interrupting the human?" | `PushNotification` | Pull attention now, or stay silent? | **Agent decides** (already agentic) | Reversible |
| H8 | "When should I resume?" | `ScheduleWakeup` / loop pacing | Self-pace the next tick | **Agent decides** (already agentic) | Reversible |

**Read the "current default" column.** Four of the eight (H2, H7, H8, and de-facto H6) the
harness *already* delegates to the agent — it trusts the model to decide from guidance, no
human in the loop. So the agentic-operation pattern is not novel; the harness already bets on
it where the blast radius is low and reversible. The design question is only: *how far up the
blast-radius ladder can we push it, and with what witness?*

### 2b. Kernel / guard gates already structured (this repo)

The repo has already climbed part of that ladder. Two of the human-blocking harness questions
are **already evidence-operated** here:

- **H4 (tool permission) → `repo_guard`.** `internal/repoguard` / `cmd/repoguard` is a
  PreToolUse hook that decides *allow / deny / transform* from evidence for whole classes of
  call: `OUT_OF_TREE_WRITE` (target resolves outside the repo root), `INTERACTIVE_HANG` (a
  command that waits on a TTY the session lacks), `FOREGROUND_SLEEP`, `LIVE_MONITOR_OUTPUT_READ`,
  `WORKSPACE_PATH_UNMAPPED`. Each is a permission prompt the human never sees because a witness
  settled it and pre-filled the non-interactive equivalent.
- **H5 (confirm-before-irreversible) → `REQUIRE_WITNESS`.** `internal/adjudicator/reversibility.go`
  pauses an irreversible/outward call pending an echoed confirm token *or* a switch to a
  sanctioned compiled sidestep (`fak sync push` for a gated `git push`). The confirm is
  structured (a token bound to the exact command bytes), not a free-text "yes".

And the genuinely-reserved-for-a-human decisions are already named as **`OPERATOR_GATE`**
tokens — the subset of the vocabulary whose `fix` is a human judgment, not a wait:
`BLOCKED_BY_HUMAN` (a dispatch unit an open decision no automation clears; the static-router
label at `cmd/fak/dispatch_skipped.go`), `INDETERMINATE`
(the verification ladder exhausted → fail-closed escalate), `DOOM_LOOP` (graduated NUDGE →
operator ESCALATE), `CORE_SELF_MODIFY` (touching the machinery that would catch its own bad
edit). The rest of the `OPERATOR_GATE` set (and all of `STALE_CLAIM` / `TRUE_DRAIN` /
`MISROUTE`) are *not* human questions at all — their `fix` is "wait", "replan", or "reroute",
which an agent executes without a human.

**Net of §2:** subtract what the harness already delegates (H2, H6, H7, H8) and what the repo
already operates (H4, H5). The residue — the actual frontier of this note — is exactly two
questions: **H1 plan approval** and **H3 clarify/choose-approach**.

## 3. The decision procedure — operate from evidence, escalate the residue

Four laws govern any autonomous answer to an operator-question. They are the fences that keep
"operate it" from degrading into "confidently guess."

1. **Decide the decidable part from a witness; escalate the residue — never auto-answer to
   avoid asking.** The failure mode is a confident wrong "yes". A part of the question a
   witness cannot settle must escalate (fail-closed, the `INDETERMINATE` discipline), never
   be guessed to keep moving.
2. **Reversibility gates autonomy.** Reversible **and** witness-backed → operate it. Irreversible
   or outward-facing → the human confirm stays (`REQUIRE_WITNESS`), and — per the recovery
   note's standing law — any outward *filing* is **auto-DRAFT, never auto-LIVE**.
3. **Escalate as a closed-vocabulary reason with an appeal channel, not prose.** The residue
   goes to the human as a structured `OPERATOR_GATE`-class reason + options + the evidence for
   each, routed through the in-product appeal seam (`fak complain`). `operator-heaviness-score`
   makes that appeal channel a *hard* gate: no autonomous refusal may exist without a human
   recovery path.
4. **The answer is witnessed, not self-reported.** "The plan is fine" / "approach A is better"
   is a *claim*; it must be backed by an oracle verdict the same as any ship claim (`dos verify`
   / `commit-audit` discipline). The agent that decides is not the authority on whether it
   decided correctly.

Applying those laws to the two frontier questions:

### H1 — plan approval, decomposed

"Is this plan right?" is not one question; it is four checkable predicates plus a residue:

| Predicate of the plan | Evidence oracle that settles it | Verdict → action |
|---|---|---|
| Does its declared file-tree stay disjoint from live leases? | `dos arbitrate` | collision → **auto-refuse** (`COLLISION_RISK`), not a human question |
| Does it violate a declared invariant (layering, direction, self-modify)? | architest / `internal/gitgate` / core-lock audit | violation → **auto-refuse** (`ARCH_LAYER_VIOLATION` / `OUT_OF_DIRECTION` / `SELF_MODIFY` / `CORE_SELF_MODIFY`) |
| Is every step reversible and non-outward? | `internal/adjudicator/reversibility.go` | all reversible → **auto-approve + execute**; any irreversible/outward step → that step keeps the H5 confirm |
| Does it have a witnessable done-criterion? | `dos verify` / registered witness | unwitnessable → **escalate** (`LOOP_DONE_UNWITNESSED`-shaped) |
| Genuine design ambiguity none of the above resolves | — | **escalate** as a structured `OPERATOR_GATE` question |

So plan-approval is *mostly* evidence-decidable: auto-refuse the collision / direction /
self-modify plans (they were never a human's to approve), auto-approve the fully-reversible-
and-witnessable plan, and escalate **only** the residue — a plan carrying an irreversible or
outward step, an unwitnessable goal, or a real design fork. The human sees a smaller, sharper
question, with the machine-settled parts already marked.

### H3 — clarify / choose-approach, decomposed

| Case | Evidence oracle | Verdict → action |
|---|---|---|
| Options all reversible; one dominates on a measurable axis | the scorecards (`steerability-score`, `quality-score`, `operator-heaviness-score`) | **decide + record the basis** (which axis, which oracle) |
| The ambiguity is a fact about the repo, not a preference | grep / `git` / `dos verify` (ship-facts) | **resolve from the repo**, don't ask |
| A choice commits an irreversible/outward decision | reversibility gate | **escalate** — the human owns the irreversible fork |
| Axes genuinely conflict with no evidence tiebreak | — | **escalate** as a structured choice: options + the evidence for each |

The `question-loop` skill supplies the anti-pattern to avoid here: an agent that *also holds
the pen* slides into solution mode and downgrades a hard question into a to-do it knows how to
close. So when H3 escalates, it must escalate as a **structured question carrying the options
and the evidence for each** (the `AskUserQuestion` shape), not as the agent quietly picking and
moving on.

## 4. The classification — where each question sits today

- **A — operable now** (primitive exists; wire it into the gate): H4 (`repo_guard`), H5
  (`REQUIRE_WITNESS`), and the *auto-refuse* + *auto-approve-reversible* branches of H1 (the
  oracles in §3 all exist). H2/H6/H7/H8 already operate.
- **B — operable with new wiring** (the frontier this note scopes): the H1 residue router (the
  "escalate only the irreversible/unwitnessable/ambiguous residue" adapter) and the H3
  evidence-first resolver.
- **C — irreducibly human** (never auto-answered, by law 2): approving a plan with an
  irreversible or outward step; a design fork with no evidence tiebreak; anything touching the
  core-lock machinery. These *escalate better* (structured reason + appeal) but stay human.

The honest headline: the frontier is narrow. Most of an operator-question is either already
delegated, already operated, or evidence-refusable. The build is a **residue router**, not a
new brain.

## 5. Compose with the recovery seam

This note is the *forward* half — decide the operator-question. The
[auto-complain note](CONCEPT-AUTO-COMPLAIN-WITNESSED-INTAKE-2026-07-10.md) is the *recovery*
half — when a forward decision was wrong, the appeal is witness-bound by construction and
surfaced only on an objective signal (refused-then-admitted-same-digest; remedy-mismatch),
auto-DRAFT never auto-LIVE. Together they close the loop: **operate the question from evidence,
and make being wrong cheaply and honestly appealable.** The recovery note's standing laws
(auto-DRAFT ceiling; candidate-generator-not-adjudicator; advisory-never-debt intake) apply
unchanged to any autonomous plan/approach decision this note authorizes.

## 6. Epic + child tickets (scoped)

**Epic — Agentic operation of harness operator-questions.** Map the two frontier operator-
questions (plan approval, clarify/choose) onto the evidence-decide-then-escalate-the-residue
machinery the kernel already uses for its own gates. Composes with `repo_guard` (H4),
`reversibility` (H5), the refusal vocabulary (escalation tokens), and the auto-complain seam
(recovery). Never auto-answers an irreducibly-human residue (§4 C).

- **Q1 — plan-scope pre-adjudicator (pure, read-only).** A fold that takes a plan's declared
  file-tree + steps and runs the §3 H1 oracles (`dos arbitrate` disjointness; architest /
  gitgate / core-lock direction checks; reversibility scan of each step; `dos verify`-able
  done-criterion). Emits one of `{auto-refuse <reason>, auto-approve, escalate <residue>}`.
  DoD: unit tests for each branch — a colliding plan → `COLLISION_RISK`; a layering-violating
  plan → `ARCH_LAYER_VIOLATION`; a fully-reversible witnessable plan → auto-approve; a plan
  with one outward step → escalate that step only. No I/O beyond the read-only oracles.
- **Q2 — H1 residue router (flag-gated).** When Q1 returns `escalate`, render the *residue* as
  a structured `OPERATOR_GATE` reason + the machine-settled parts already marked, through the
  appeal-wired channel. DoD: an escalation carries the closed-vocab token, the specific
  residual predicate, and a `fak complain`-appealable handle; the auto-approve branch executes
  without a human; flag-gated off by default. Never escalates a part Q1 already settled.
- **Q3 — H3 evidence-first resolver (pure).** Before any `AskUserQuestion`, run the §3 H3
  ladder: repo-fact ambiguities resolve from grep/git/`dos verify`; reversible dominated
  choices decide on a scorecard axis and record the basis; only a genuine irreversible fork or
  an unbroken tie escalates. DoD: a repo-fact question resolves without asking (test: the fact
  is in the tree); a reversible dominated choice decides with a recorded axis; an irreversible
  fork escalates as a structured `AskUserQuestion` with options + per-option evidence.
- **Q4 — decision witness + audit trail.** Every autonomous answer (auto-approve, auto-resolve)
  writes a witness row (which oracle, which verdict) so the decision is `commit-audit`-style
  auditable and the auto-complain detector can later flag a wrong one. DoD: the row is
  non-forgeable (oracle verdict, not the model's say-so); a wrong auto-approve is recoverable
  through the existing appeal path with no new outward surface.
- **Q5 — heaviness + de-dup fence.** The new tokens/flags must not blow the
  `operator-heaviness-score` budget (fold escalation reasons into existing `OPERATOR_GATE`
  tokens where they name the same recovery; no new front-door flag without graduation). DoD:
  `fak operator heaviness` pressure held flat across the epic; no reason added that the kernel
  never emits.

## 7. Honest fences (anti-goals)

- **Never auto-approve to avoid asking.** The whole value is the *smaller, sharper* human
  question, not zero human questions. A residue that can't be witnessed escalates; it is never
  guessed. (Law 1 / `INDETERMINATE`.)
- **Irreversible and outward stay human, permanently.** Reversibility is the hard ceiling on
  autonomy, not a first-cut simplification to relax later. Auto-DRAFT is the outward ceiling
  (inherited from the recovery note).
- **The pre-adjudicator is a router, not an adjudicator of design taste.** It settles the
  *checkable* predicates of a plan (collision, direction, reversibility, witnessability); it
  never rules on whether an architecture is *good*. That residue is C-class, human.
- **No new brain.** Q1–Q3 are folds over oracles that already exist. If a ticket starts
  proposing a new model-judgment authority instead of an evidence fold, it has drifted off this
  note.
- **Every autonomous "no" keeps its appeal.** By the `operator-heaviness-score` hard gate, no
  escalation token ships without the `fak complain` recovery path wired.
