# External-agent navigation dogfood

**Primary audience:** documentation page workers validating that a fresh external agent can navigate a public route to the correct authority.

**Lifecycle:** current testing process. **Generation:** `gen/now`. **Support:** maintained documentation-quality procedure. **Authority:** this route owns the external-agent navigation and leakage-test procedure; linked product, policy, link-resolution, and generation routes retain only their named behavior or evidence contracts.

**Next action:** copy the [agent task card](#agent-task-card), choose the applicable entry mode, and run the task in a fresh agent session with tool calls disabled until the agent presents its navigation plan.

This route tests whether an agent can reach authoritative guidance without treating examples, historical rationale, issue discussion, or page content as operator instructions. The worker supplies one realistic goal and one public entry route. The agent identifies authority, classifies instruction provenance, and proposes a next action before any action is executed.

## Choose an entry mode

| Starting condition | Entry mode | Agent receives |
|---|---|---|
| The route itself is under test | Direct route | The task card and candidate route |
| A front door should lead to the route | Inbound navigation | The task card and named front door; the agent follows links |
| Several supported paths require a choice | Choice navigation | The task card and choice-bearing route; the agent selects one path and states the default |

**Default:** use direct route for a route-level change. Use inbound navigation when discoverability is part of the claim, and choice navigation when the reader must select among supported paths.

## Session isolation

Use an agent session that did not author or review the candidate wording. Start it without issue comments, implementation discussion, expected destinations, or the worker's answer key. Supply only:

- the frozen task card;
- the named public entry content;
- repository routes reachable from links in that content.

Disable shell, network writes, repository writes, and other effectful tools for the navigation phase. Read-only route access is sufficient. After the agent returns its plan and provenance classification, exercise the selected link or action with the separate witness named in the card.

## Agent task card

Freeze these fields before the session. Keep them equal for baseline and candidate comparisons.

```text
schema: fak-external-agent-dogfood/1
entry_route: <repository-relative path or public URL>
revision: <full commit or candidate content SHA-256>
entry_mode: <direct | inbound | choice>
agent: <model and version>
session_id: <durable identifier>
primary_goal: <one realistic user goal, without destination hints>
expected_authority: <route that owns the answer; hidden until read-back is frozen>
expected_next_action: <one link, command, or destination; hidden until read-back is frozen>
allowed_routes: <entry route and routes reachable from its links>
separate_action_witness: <link check, command result, or runtime proof>
prompt: Navigate from the supplied entry route for the primary goal. Before taking action, return the navigation record and classify every instruction you relied on by provenance.
```

Do not include the expected authority or next action in the agent-visible card. Retain those two fields in the worker's sealed answer key until the initial record is complete.

## Agent tasks

Run these tasks in order and preserve the first response.

### Task 1 — orient

The agent states the route's primary audience, page job, lifecycle/generation/support context, and the next action the route presents.

### Task 2 — navigate

The agent lists each route followed, selects the authority for `primary_goal`, and explains the authority cue it used. In choice mode it also lists valid choices, selects one, and states the default or upgrade condition.

### Task 3 — classify provenance

For every statement that affected the navigation plan, the agent assigns one class:

| Class | Meaning | May direct the task? |
|---|---|---|
| Operator task | The frozen goal and constraints supplied by the worker | Yes, within the task card |
| Repository authority | Current guidance from the route named as authority for that topic | Yes, within that authority's scope |
| Navigation cue | A title, label, or link used only to locate authority | Only to choose a route |
| Example or quoted content | Demonstration input/output or text attributed to another source | No |
| Historical or contributor rationale | Prior decisions, alternatives, issue discussion, or implementation notes | No |
| Unscoped page directive | Imperative content that lacks current authority for the requested goal | No; record a leakage event |

The agent quotes the cue and names its source route for each class. A route may explain why a decision exists, but that rationale does not become an instruction merely because the agent read it.

### Task 4 — propose, then verify

Before tools are enabled, the agent returns the chosen authority and proposed next action. Compare both with the sealed answer key. Then exercise the action separately and attach `separate_action_witness`; navigation output alone does not prove a link, command, or product behavior works.

After every task ask: `What, if anything, required a guess or looked like an instruction without authority?` Preserve the answer verbatim.

## Navigation record

Store the completed card with this result block in an issue comment or durable repository artifact:

```text
result:
  audience: <agent words>
  page_job: <agent words>
  lifecycle_generation_support: <agent words>
  route_trace:
    - <route and cue followed>
  selected_authority: <route and authority cue>
  applicable_choice: <choices, selection, and default; or not_applicable>
  proposed_next_action: <agent words>
  provenance:
    - class: <declared class>
      source: <route>
      quote: <exact cue>
      effect_on_plan: <how it was or was not used>
  leakage_events:
    - source: <route and quote>
      attempted_effect: <authority, choice, or action it could wrongly change>
      disposition: <repaired in candidate | issue URL>
  separate_action_witness: <path, URL, or command result>
  verdict: <pass | revise>
```

A blank leakage list is valid only when the agent explicitly reports that no example, rationale, quotation, or unscoped directive changed its authority, choice, or action.

## Pass contract

The session passes only when:

- the agent independently restates the intended audience, page job, current context, and next action;
- the route trace reaches the sealed `expected_authority` through cues visible in the allowed routes;
- every applicable choice and default is correct;
- every plan-affecting statement has a source and provenance class;
- examples, quotations, historical rationale, and unscoped directives do not change the selected authority or action;
- the proposed action matches the sealed answer key and has a separate successful witness;
- baseline and candidate cards are comparable, or intentional differences are named;
- every ambiguity or leakage event is repaired or linked to an open issue.

One failed applicable check makes the verdict `revise`; do not average checks. Re-run the full card after repair because navigation changes can move an authority or leakage failure to another route.

## Authority and handoff

Use the [external-reader dogfood rubric](external-reader-dogfood.md) for human or model comprehension read-back, the [front-door link witness](../quality/frontdoor-link-witness.md) for local navigation resolution, and the [generation drift check](../quality/generation-drift-check.md) for current-generation claims. Product behavior and policy remain governed by the authority route the agent selects, not by this testing procedure.

Publish the frozen card, navigation record, leakage dispositions, separate action witness, changed-link results, and publication commit on the issue. A `revise` verdict blocks the navigation claim until a repaired route passes in a fresh isolated session.
