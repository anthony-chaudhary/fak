# Front-door clarity scorecard

**Primary audience:** maintainers reviewing a public documentation route before it becomes or remains a recommended front door.
**Lifecycle:** current process contract.
**Generation:** `gen/now`; score the behavior and support status presented by the route today.
**Authority:** the dimensions come from the [documentation audience architecture](../project/DOCUMENTATION-AUDIENCE-ARCHITECTURE-2026-07-15.md); claim evidence follows the [proof-proximity audit](claim-proof-proximity-audit.md).
**Support:** use the [contributor route](../../CONTRIBUTING.md) to report a route that cannot meet the rubric.

**Next action:** choose one public route, capture its first screen and destination links, then complete every row in the worksheet before recommending it.

## Scoring contract

Score the rendered route that a reader actually receives, including the first linked page needed to complete its promised action. Record one evidence citation per row. Use only `0`, `1`, or `2`:

- **2 — clear:** the criterion is explicit, adjacent, and usable without inference.
- **1 — recoverable:** the information exists in the scored route but requires avoidable interpretation or an extra navigation step.
- **0 — missing or conflicting:** the criterion is absent, stale, contradictory, or depends on undocumented knowledge.

Do not average away a critical failure. Regardless of total score, each condition below forces **Revise**:

- no checkable next action;
- a broken or mismatched proof for a material claim;
- current guidance whose lifecycle, generation, release, backend, or mode is ambiguous.

A route that points to a superseded authority and cannot be corrected locally is **Replace route**.

## Worksheet

| Dimension | 2 — clear | 1 — recoverable | 0 — missing or conflicting | Evidence to capture |
|---|---|---|---|---|
| **First-screen job** | Names one primary audience, states what the page does, and puts the reader's immediate outcome in the first screen. | Audience or outcome can be inferred from nearby text. | Audience and page job compete, or the first screen leads with internal history. | Quote the first-screen audience and job statement. |
| **Affirmative framing** | Leads with what fak or the route is, does, supports, and requires. | The affirmative answer is present after caveats or defensive explanation. | Negation, debate, or comparison carries the explanation. | Quote the shortest affirmative description. |
| **Choice clarity** | Every valid mode or path has a selection criterion, default when one exists, and distinct proof. | Choices exist but one criterion, default, or proof requires inference. | Modes overlap, a false single path is implied, or no selection rule exists. | List choices, default, and selection rule. When exactly one valid path is proved, record `single path` as the evidence and still score the row; `not applicable` never removes a row or changes the denominator. |
| **Proof proximity** | Each material claim has adjacent status, scope, baseline when relevant, provenance, and a working scoped proof link. | Proof is reachable within the route but context is separated or incomplete. | Proof is absent, broken, stale, or establishes a different claim. | Record the claim and artifact using the [proof verdicts](claim-proof-proximity-audit.md#evaluate-one-claim). |
| **Lifecycle and generation** | Current, versioned, research, or archived status is explicit, with release, backend, mode, or generation where behavior differs. | Status is discoverable, but the applicable generation or mode takes interpretation. | Historical, planned, simulated, or unsupported behavior can be mistaken for current guidance. | Quote lifecycle and generation; identify any superseding authority. |
| **Navigation and duplication** | The route names one current authority per decision and deeper links add detail without repeating a competing first screen. | A duplicate exists but clearly points back to the authority. | Two pages make conflicting authority claims or the route is an exhaustive inventory without a decision path. | List inbound route, authority, and first destination. |
| **Checkable next action** | Exactly one immediate action names a command, artifact, or observable result the reader can verify. | One action exists but its result or prerequisite requires inference. | No action exists, several actions compete, or success cannot be observed. | Quote the action and record its expected result. |
| **Support boundary** | Supported, experimental, research, archived, and unavailable states are explicit, with a public escalation route. | Boundary exists but is distant or uses unexplained vocabulary. | The route implies support from existence, labels, or historical evidence alone. | Quote the boundary and support/escalation destination. |

**Total:** add the eight rows for a score from `0` to `16`.

| Verdict | Rule |
|---|---|
| **Ready** | `14–16`, every row has evidence, and no critical failure is present. |
| **Revise** | `9–13`, or any critical failure despite a higher total. |
| **Replace route** | `0–8`, or the page points at a superseded authority and cannot be corrected locally. |

The number ranks clarity debt; it does not prove product behavior. Preserve each row's evidence and critical-failure decision alongside the total.

## Repeatable read-back

Use this template in an issue comment, review artifact, or structured witness:

```text
Route and revision:
Rendered first-screen capture:
Primary audience and page job:
Scores (job, affirmative, choices, proof, lifecycle, navigation, action, support):
Total and verdict:
Critical failure (none or exact row):
One immediate next action:
Evidence links by row:
Independent reader result:
```

After the maintainer scores the route, give the same rendered route to a reader who did not author it. Ask the reader to state the audience, page job, applicable choice, current authority, and immediate next action. A **Ready** verdict requires the reader to reach the same choice and action without hidden context; record every ambiguity rather than coaching the answer.

## Mode and maintenance

This scorecard evaluates documentation mode, not runtime correctness. Run commands and inspect artifacts named by the route when scoring proof or next-action rows. Re-score after a first-screen rewrite, authority change, generation change, broken-link repair, or support-boundary change. Keep the prior worksheet when trend evidence matters; otherwise the current revision and witness are authoritative.
