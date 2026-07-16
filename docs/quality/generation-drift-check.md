# Detect generation-context drift in current documentation

**Primary audience:** maintainers reviewing current (`gen/now`) documentation routes.
**Lifecycle:** current process authority. **Generation:** `gen/now`.
**Authority:** this page defines the documentation drift check; [`docs/generation.md`](../generation.md) defines product-horizon labels, the [audience inventory](../project/documentation-audience-inventory.json) defines each audited page's declared generation and lifecycle, and code, tests, and support contracts remain authoritative for runtime behavior.
**Next action:** select one inventory page marked `gen/now`, trace every material behavior or support statement to current authority, and record each dependency on later-horizon or historical detail in the witness table.

This check detects a current page whose explanation depends on detail from a different product horizon. It does not prohibit links to research or future work: provenance may remain reachable when the current contract stands on its own.

## What counts as drift

A `gen/now` page has **generation-context drift** when a reader needs later-horizon, historical, experimental, simulated, stubbed, or superseded material to understand or trust current behavior. Common forms are:

- a future design is written in the present tense or presented as current support;
- a dated note supplies the only definition or proof for a current claim;
- a historical command, flag, schema, backend, or default is copied into current instructions;
- a linked page's explicit generation or lifecycle conflicts with the importing sentence;
- an unmarked proposal or roadmap item is required to complete the current next action;
- a page declares `gen/now` while its issue, maintained inventory, or authority route identifies another horizon.

The following are **not drift** when their role is explicit:

- a provenance link introduced as history or rationale after current behavior is defined;
- a future option labeled with its horizon and separated from current instructions;
- an archived snapshot with a visible replacement route;
- a simulation or stub used as bounded evidence without being presented as shipped support.

Generation is a product horizon, not runtime exposure, priority, branch strategy, or completion percentage. Apply those distinctions from the [generation contract](../generation.md) before classifying a mismatch.

## Audit set and trigger

The default set is every page in the [documentation audience inventory](../project/documentation-audience-inventory.json) whose `generation` is `gen/now` and whose lifecycle is `current`. At commit `5fa5bd26b33f2bd1cf719fc018fdc811e957a3ed`, that is **17 pages**:

`README.md`, `START-HERE.md`, `llms.txt`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `docs/project/DOCUMENTATION-AUDIENCE-ARCHITECTURE-2026-07-15.md`, `docs/project/documentation-audience-inventory.json`, `docs/integrations/README.md`, `docs/supported/clouds.md`, `docs/run-the-demos.md`, `docs/FAQ.md`, `docs/PRODUCT-STATUS.md`, `docs/ROLLBACK.md`, `docs/HARDWARE-MATRIX.md`, `SECURITY.md`, and `docs/vendor/neo-cloud-reference-architecture.md`.

The two inventory entries marked `historical` are reference destinations, not current pages under test.

Run the check:

- when a current page or one of its authority links changes;
- when an inventory generation or lifecycle value changes;
- when current behavior promotes, demotes, retires, or replaces an option;
- in the audience architecture's weekly cross-page consistency pass.

Run `git rev-parse HEAD` before the audit. Record that full repository commit, which contains both the inventory and all audited page versions, as the inventory revision. No separate stamp is required. If the inventory or an audited page changes, the old result remains provenance but no longer witnesses the new tip.

## Repeatable check

For each selected `gen/now` page:

1. **Read the route contract.** Record its audience, lifecycle, generation, role, and current next action from the page and inventory.
2. **Extract material statements.** Capture commands, defaults, support claims, schemas, modes, guarantees, and required navigation—not every incidental noun.
3. **Resolve dependencies.** Follow the direct authority or proof link for each statement. Record the destination, anchor when present, and its explicit or inferred generation and lifecycle.
4. **Compare context.** Decide whether the current statement stands on current authority or depends on a different horizon. A missing marker is `unknown`, not automatically `gen/now`.
5. **Exercise the next action.** Confirm it can be understood and started using current routes alone. A future prerequisite is a finding even when every link resolves.
6. **Classify and disposition.** Apply one class below. File or link a deduplicated issue for every unresolved finding before completing the run.

A successful HTTP or local-link check proves reachability only. It cannot prove that the destination describes the right generation.

## Classification contract

| Class | Decision rule | Disposition |
|---|---|---|
| **Current-aligned** | The statement and its authority both describe current behavior, or code/tests directly witness it. | Pass. |
| **Scoped provenance** | Historical or later-horizon material is visibly labeled and optional after the current contract stands alone. | Pass; retain the boundary label. |
| **Future leakage** | Later-horizon behavior carries a current explanation, choice, guarantee, or next action. | Finding: separate and label it, or promote only with the required evidence. |
| **Historical dependency** | Dated, archived, or superseded material is the only authority for a current statement. | Finding: replace it with current authority and retain history as provenance. |
| **Contract mismatch** | Page, inventory, issue, support route, or linked authority disagree about generation or lifecycle. | Finding: reconcile the owning authority; do not pick a label by prose preference. |
| **Unknown context** | A material dependency has no reliable generation/lifecycle marker and no current code, test, or support witness. | Finding: mark and route the authority before treating it as current. |

Do not rewrite product behavior to make documentation agree. When the current authority is unclear, record `unknown context` and route the decision to the owning implementation or support contract.

## Decision checks

A page passes only when all answers are affirmative:

- **Current contract:** can a reader state what works now without opening historical or future material?
- **Choice boundary:** are current defaults and alternatives separated from options that need promotion evidence?
- **Mode and support:** do claims identify the runtime mode and support status they govern?
- **Proof:** does each material current claim reach current code, tests, a maintained contract, or a witnessed command?
- **Next action:** can the reader start it using only current prerequisites?
- **Navigation:** are deeper research and history links labeled as provenance rather than authority?

One failed answer is a finding; do not average failures into a score.

## Record the witness

Use one row per material dependency, grouping only identical authority and disposition:

| Field | Required value |
|---|---|
| Audit revision | One full repository commit containing the inventory and all audited page versions (the inventory revision) |
| Current page | Path, audience, declared generation/lifecycle, and next action |
| Statement | Quoted current behavior, choice, support claim, or prerequisite |
| Dependency | Authority/proof path and anchor |
| Dependency context | `gen/now`, later horizon, historical, or `unknown`; lifecycle and support status |
| Classification | One class from this page |
| Evidence | A successful link check for linked dependencies plus the strongest applicable current witness: code/test, maintained contract, witnessed command, or independent read-back |
| Disposition | Pass, fixed commit, open issue, or superseding route |

A finding fixed in the same run records its commit in the row disposition and is excluded from `unresolved_findings`; that field contains only still-open issue links or `[]` when none remain.

A complete run must include this summary. The independent-reader block is mandatory; the run passes only when its `ambiguities` value is an empty list after any revisions:

```text
inventory_revision: <full commit>
audited_pages: <ordered paths>
material_dependencies: <count>
class_counts:
  current_aligned: <count>
  scoped_provenance: <count>
  future_leakage: <count>
  historical_dependency: <count>
  contract_mismatch: <count>
  unknown_context: <count>
unresolved_findings: <issue links or []>
next_actions_exercised: <page -> result>
independent_reader:
  audience: <restated audience>
  page_job: <restated check>
  applicable_choice: <pass versus finding rule>
  next_action: <restated first action>
  ambiguities: []
```

## Repair and rerun

Repair at the narrowest owning seam:

- update the current route when wording imports the wrong horizon;
- update the inventory when its declaration is stale;
- add a current authority when a dated note is the only proof;
- label and move optional rationale deeper rather than deleting provenance;
- route an unsupported behavior question to code, tests, or [`docs/supported/`](../supported/README.md).

Then rerun every affected page and its direct inbound current routes. Validate links, exercise each next action again, and use the [front-door clarity scorecard](frontdoor-clarity-scorecard.md) to confirm lifecycle/generation, proof proximity, and navigation remain clear.
