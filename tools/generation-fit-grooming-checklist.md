# Generation-Fit Grooming Score

A per-issue **intake / grooming** score that grades whether an issue's generation
labeling matches its scope, its proof bar, and its time horizon — and flags the
mismatches for operator review. Closes
[#1648](https://github.com/anthony-chaudhary/fak/issues/1648) under epic
[#1625](https://github.com/anthony-chaudhary/fak/issues/1625).

This is the intake sibling of two existing artifacts; keep them distinct:

| Artifact | Question it answers | Grain |
|---|---|---|
| This grooming score | *Is the generation label even right for this issue?* | one issue, at grooming/intake |
| [Generation Readiness Gates](../docs/notes/GENERATION-READINESS-GATES-2026-06-30.md) (#1644) | *Is the evidence strong enough to promote `next` → `now`?* | one item, at promotion |
| `debt_score` in [`docs/generation.md`](../docs/generation.md) | *How much intake drift does a whole lane carry?* | one lane, at milestone report |

The canonical stream/milestone/evidence definitions this score checks against live
in [`docs/generation.md`](../docs/generation.md); this file is the checkable rubric,
not a second source of truth.

## Orthogonality (unchanged by this score)

This score grades label *fit* only. It does not touch the three axes generation is
orthogonal to, and must never be read as if it did:

- **Priority.** A low fit score never lowers (or raises) urgency. A `gen/future`
  issue can be high-value and a `gen/now` issue can be trivial cleanup — the score
  flags a *labeling* mismatch, not a value judgment. Treating `gen/future` as
  lower-priority-by-default is itself the anti-pattern this rubric exists to catch.
- **Shared trunk.** The score changes no git rule. Every stream still lands through
  `main`, by explicit path, DCO-signed, with the `(fak <leaf>)` stamp. A label never
  authorizes a branch or worktree, and a flag never authorizes one either.
- **Runtime feature gates.** The score reads planning metadata (labels, milestone,
  issue body), not runtime exposure. A `gen/next` item may ship inert behind a
  default-off gate; that is correct posture, not a mismatch. Using `gen/next` *as* a
  substitute for a runtime gate is a mismatch (check 5).

## The score

Seven boolean checks. `fit_score = count(green)`, out of 7. Any RED check raises
the **mismatch flag** — the issue is routed to operator review with the specific
failed checks and a suggested [promotion verb](../docs/generation.md#promotion-verbs)
(`promote` / `demote` / `retire` / `park`, or a plain label/milestone/witness fix).

| # | Check | Green condition | RED = mismatch flag → operator action |
|---|---|---|---|
| 1 | Label completeness | Carries `generation` **and exactly one** `gen/*` stream label. | 0 or ≥2 stream labels, or missing `generation`. → Fix labels, or `needs-triage` if unclear (never guess `gen/future`). |
| 2 | Label ↔ milestone agreement | Stream label matches its paired `Generation G#` milestone per the [streams table](../docs/generation.md#streams). | Mismatch or no milestone. → Fix milestone or label so they agree. |
| 3 | Scope-width fit | Issue scope matches the stream rule: `now`=trunk-safe current-loop improvement; `next`=one near-term seam/foundation; `second-next`=architecture needing simulation/compat; `future`=research/narrative/standards. | Scope wider/narrower than the stream (architecture bet on `now`; cleanup parked in `future`). → `promote`/`demote` to the matching horizon. Catches priority-laundering & current-work-laundering. |
| 4 | Proof-bar fit | Promised witness matches the stream's proof bar: `now`=trunk witness; `next`=contract test **plus** promotion evidence naming what moves it toward `now`; `second-next`=simulation/compat policy; `future`=memo naming the decision it could influence. | Promised proof weaker than the stream demands. → Strengthen the witness or demote the horizon. |
| 5 | Risk ↔ exposure fit | Allowed risk matches gate posture: `next`/`second-next` moderate risk only behind a gate, dogfood path, or handoff contract — never as a stand-in for a default-off runtime gate. | Unguarded moderate/high risk, or `gen/next` used in place of a feature gate. → Add the gate/dogfood/handoff, or demote. Catches feature-gate conflation. |
| 6 | Evidence completeness | Body names **promotion evidence**, **demotion/retirement evidence**, and **≥1 invalidating assumption**. | Any of the three missing. → Ask the reporter/groomer to add them before dispatch. |
| 7 | Continuation | A future agent can act from the issue alone — scope, horizon, and witness are named without rereading the parent epic. | Requires an epic reread to know what to do. → Add a self-contained scope + witness line. Catches permanent-parking & hidden-demotion drift. |

A full `7/7` is **groomed** (dispatchable). Anything `< 7` is **flagged** — still
a valid issue, just not clean intake; the operator decides the fix before it enters
a dispatch wave.

## Agent-runnable schema

An agent grooming an issue fills this frame (one object per issue). It is the
machine-readable form of the table above; a later `fak` verb or `issue_triage`
scope can emit and consume it without prose parsing.

```json
{
  "schema": "fak-generation-fit/1",
  "issue": 1648,
  "checks": {
    "label_completeness":   {"green": true, "note": ""},
    "label_milestone_agree":{"green": true, "note": ""},
    "scope_width_fit":      {"green": true, "note": ""},
    "proof_bar_fit":        {"green": true, "note": ""},
    "risk_exposure_fit":    {"green": true, "note": ""},
    "evidence_completeness":{"green": true, "note": ""},
    "continuation":         {"green": true, "note": ""}
  },
  "fit_score": 7,
  "flagged": false,
  "suggested_verb": null
}
```

`flagged := fit_score < 7`. `suggested_verb` is one of
`fix-label | fix-milestone | promote | demote | retire | park | add-witness`,
chosen from the first RED check.

## Worked before/after readout

Applying the rubric to two issues shows the flag firing and clearing.

**A — a drifted issue (flag fires).** Hypothetical intake: labeled `gen/now` +
milestone `Generation G3 - Future`, scope = "research a standards analogue", body
names no invalidating assumption.

```text
issue A  fit 4/7  FLAGGED
  1 label_completeness    GREEN
  2 label_milestone_agree RED   gen/now label vs G3-Future milestone
  3 scope_width_fit       RED   research scope belongs to gen/future, not gen/now (current-work laundering)
  4 proof_bar_fit         GREEN
  5 risk_exposure_fit     GREEN
  6 evidence_completeness RED   no invalidating assumption named
  7 continuation          GREEN
  suggested_verb: demote  (to gen/future; fix milestone; add assumption)
```

**B — issue #1648 itself (flag clears).** Labels `generation`, `gen/next`;
milestone `Generation G1 - Next Gen`; scope = one grooming-score seam; acceptance
requires an artifact that names promotion/demotion evidence and an invalidating
assumption; acceptance requires continuation without an epic reread.

```text
issue #1648  fit 7/7  GROOMED
  1..7 all GREEN
  suggested_verb: none
```

Before this rubric, "does this label fit?" was a judgment call with no shared
checklist; after it, the same seven booleans and the same flag rule apply to every
generation issue, and the disagreement surfaces as a named RED check instead of an
operator hunch.

## Promotion / demotion / assumption (for this artifact)

- **Promotion evidence** (what moves this doc-rubric toward `gen/now`): wire the
  seven booleans into a runnable surface — an `issue_triage.py --scope generation`
  row, a `fak issue fit`/`fak groom` verb, or a milestone-report column — and
  capture one **dogfood readout** grading the live open `generation`-labeled issues.
  A green contract test over the `fak-generation-fit/1` schema plus that readout is
  the promotion witness. Until then this stays a `gen/next` grooming aid, applied by
  hand or by a grooming agent.
- **Demotion / retirement evidence**: retire this rubric if the four stream
  definitions in [`docs/generation.md`](../docs/generation.md) change such that the
  checks no longer map; demote (park) it if grooming stops reading it — an unread
  checklist is decorative and should be removed, not defended. Duplication against a
  future automated `fak` fit verb is also retirement grounds: fold this into the
  verb's help text and delete the standalone file.
- **Invalidating assumption**: the score assumes labels, milestone, and issue body
  are cheap and truthful to read at grooming time, and that scope/proof/risk fit can
  be judged from the issue text without running the work. If generation intent
  migrates to a project field, or if bodies stop stating scope/proof/risk, checks
  3–6 lose their inputs and the rubric must be rebound to whatever surface then
  carries the horizon signal.

## Continue here

A future agent needs no epic reread. To advance #1648's follow-on: implement the
`fak-generation-fit/1` schema as a `fak issue fit` verb (pure logic in
`internal/generationfit/`, thin shell in `cmd/fak/`, per the Go-not-Python rule),
have it read `gh issue view --json labels,milestone,body`, emit one object per
issue, and gate it with a contract test over the worked examples above. That verb —
plus a captured dogfood readout over the open generation lane — is the promotion
evidence named above.
