# External-reader dogfood

**Primary audience:** documentation page workers validating a public route with a reader who did not author the change.

**Lifecycle:** current testing process. **Generation:** `gen/now`. **Support:** maintained documentation-quality procedure; it witnesses reader navigation and comprehension, while product behavior keeps its own runtime authority.

**Next action:** copy the [task card](#task-card), select an eligible reader, and run Task 1 without adding spoken or written coaching.

This route defines a comparable external-reader session. The worker supplies a route and a goal; the reader reports what the page says, makes any applicable choice, and follows the stated next action. The resulting read-back is evidence for clarity, not a substitute for exercising links, commands, or runtime behavior.

## Choose a session mode

| Changed route | Session mode | Required capture |
|---|---|---|
| Prose or navigation where source order carries the claim | Structured read-back | Completed task card plus exact reader answers and confusion events |
| Placement, wrapping, hierarchy, or other visual presentation carries the claim | Rendered read-back | Structured read-back plus the image profile and artifact required by the [first-screen witness](frontdoor-render-witness.md) |
| Live terminal presentation carries the claim | Terminal read-back | Structured read-back plus emitted terminal bytes and a screenshot |

**Default:** use a structured read-back for prose and navigation. Upgrade to the rendered or terminal mode whenever presentation is part of the claim.

## Eligible reader

Use a person or model session that:

- did not author or review the candidate wording;
- starts without the issue discussion, intended answer, or worker commentary;
- receives only the frozen task card, route content, and dependencies a normal reader can open;
- returns its own wording before seeing the worker's expected-answer key.

Record the reader type and session identifier. A new context from the authoring model is eligible only when it has no author transcript, hidden answer key, or retained route discussion.

## Task card

Freeze these fields before the session. Use the same card for a baseline and candidate comparison.

```text
schema: fak-external-reader-dogfood/1
route: <repository-relative path or public URL>
revision: <full commit or candidate content SHA-256>
entry_state: <direct route or inbound link followed>
reader_type: <human or model and version>
session_id: <durable identifier>
mode: <structured | rendered | terminal>
primary_goal: <one realistic reason to open this route>
applicable_choice: <choice the reader must make, or none>
expected_next_action: <one observable link, command, or destination>
allowed_dependencies: <routes the reader may open>
worker_prompt: Perform Tasks 1-3 from only the supplied route and allowed dependencies. Report confusion where it occurs; do not optimize for the expected answer.
```

Do not put the intended audience, page job, choice answer, or explanatory hints in `primary_goal`. The route must carry that information.

## Reader tasks

Run the tasks in order. Preserve the reader's first answer before asking the next task.

### Task 1 — orient

Ask the reader to state:

1. who the route is for;
2. what job the route helps that audience complete;
3. the route's current lifecycle, generation, and support context;
4. the one next action it asks the reader to take.

### Task 2 — choose

When `applicable_choice` is present, ask the reader to list the valid choices, select the choice matching `primary_goal`, and state the default and upgrade condition. Record `not_applicable` when the route has no real choice; do not invent one for the test.

### Task 3 — act

Ask the reader to identify and follow `expected_next_action`. Record the destination or command selected and the observable result. Use the [front-door link witness](../quality/frontdoor-link-witness.md) for link resolution and the applicable runtime authority for behavior; the reader session alone proves neither.

After every task, ask: `What, if anything, was ambiguous or required a guess?` Record the answer verbatim.

## Read-back record

Store the completed card with this result block in an issue comment or durable repository artifact:

```text
result:
  task_1:
    audience: <reader words>
    page_job: <reader words>
    lifecycle_generation_support: <reader words>
    next_action: <reader words>
  task_2:
    status: <pass | fail | not_applicable>
    choices_default_upgrade: <reader words>
  task_3:
    selected_action: <reader words>
    observed_result: <reader words>
    separate_action_witness: <path, URL, or command result>
  confusion_events:
    - task: <1 | 2 | 3>
      quote: <reader words>
      route_cue_used: <heading, text, or link>
      disposition: <resolved in candidate | issue URL>
  verdict: <pass | revise>
```

A blank confusion list is valid only when the reader explicitly reports no ambiguity after all three tasks.

## Pass contract

The session passes only when:

- Task 1 independently matches the route's intended audience, job, current context, and next action;
- Task 2 passes for every applicable choice and correctly carries its default and upgrade condition;
- Task 3 reaches the expected action and cites a separate link or runtime witness;
- the reader received no hidden coaching or expected-answer key;
- baseline and candidate sessions use comparable cards, or each intentional profile difference is named;
- every confusion event is resolved in the candidate or linked to an open issue.

One failed applicable task makes the verdict `revise`; do not average tasks into a passing score. Re-run the full card after a clarity repair because wording changes can move confusion between tasks.

## Authority and handoff

Use the [front-door clarity scorecard](../quality/frontdoor-clarity-scorecard.md) to define the intended answer before the reader session, the [first-screen witness](frontdoor-render-witness.md) when presentation matters, and the [front-door link witness](../quality/frontdoor-link-witness.md) for local navigation. The reader's first unaided wording remains the dogfood evidence.

Publish the card, read-back, artifact hashes when applicable, changed-link results, and publication commit on the issue. A `revise` verdict blocks the clarity claim until the route is repaired and a new eligible reader passes.
