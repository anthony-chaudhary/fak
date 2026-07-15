# Documentation audience architecture

**Status:** active program seed  
**Owner:** documentation  
**Started:** 2026-07-15

## Outcome

Make each public page immediately useful to its intended reader. Lead with what fak is, the result it delivers, and the shortest proof. Move implementation history, internal rationale, caveats, and version-specific detail to clearly labeled developer or archival layers.

This plan protects the program's seed while individual pages change in parallel. Page edits must preserve factual accuracy and proof links; they should reduce reader effort rather than merely shorten text.

## Audience layers

1. **Public front door** — `README.md`, `START-HERE.md`, top-level landing pages, integration landing pages, and the first screen of linked explainers. State the product, primary outcomes, proof, and next action.
2. **External builder** — install, quickstarts, API and integration guides, deployment choices, examples, and troubleshooting. Organize around tasks and supported choices.
3. **Operator** — production operation, policy, observability, upgrades, recovery, performance evidence, and support boundaries.
4. **Contributor** — architecture, development workflow, tests, invariants, generated artifacts, and current implementation contracts.
5. **Research and history** — design rationale, experiments, dated notes, superseded generations, and decision records. Keep this evidence discoverable without making it prerequisite reading.

A document has one primary audience and may link to deeper layers. It should not serve every audience in one uninterrupted narrative.

## Editorial contract

- Start with affirmative descriptions: what the component is, does, supports, and requires.
- Use contrast only when it resolves a likely concrete ambiguity; name the applicable mode or subsystem.
- Put limitations beside the option they constrain, with scope and a link to evidence.
- Prefer a choice table when several modes are valid. Identify the default, selection criteria, and proof for each.
- Remove conversational self-defense, internal debate, and chain-of-thought-like rationale from public guidance. Preserve useful rationale in an ADR, developer note, or dated research page.
- Keep claims balanced and reproducible. Concision never permits dropping scope, provenance, baseline, or support status.
- Give each page a generation context when behavior differs by release, backend, mode, or maturity level.
- End each front-facing path with one checkable next action.

## Information architecture

Every maintained document receives lightweight metadata in its opening section or index entry:

- primary audience;
- lifecycle: current, versioned, research, or archived;
- product generation or release range when relevant;
- authoritative owner or source;
- last verified proof or command when the page makes operational claims.

Navigation pages expose curated routes rather than exhaustive inventories. `llms.txt` remains the machine-oriented map; public landing pages select a small path per audience. Dated implementation notes stay out of front-door sequences unless they are the current authority.

## Delivery loop

1. **Inventory:** classify pages by audience, lifecycle, generation, authority, and inbound front-door links.
2. **Score:** measure first-screen clarity, affirmative framing, choice clarity, proof proximity, duplication, and stale-generation risk.
3. **Contract:** file one bounded issue per page or coherent route, naming the rendered/read-back witness.
4. **Edit:** make the smallest complete route improvement, including navigation consumers.
5. **Witness:** capture links, commands, renders, or text checks that prove the acceptance criteria.
6. **Dogfood:** send a fresh worker through the changed route with a concrete task; record confusion and time to next action.
7. **Reconcile:** land one issue per commit, update the inventory, file residual findings, and select the next highest-impact route.

The loop runs in cohorts with disjoint file ownership. A weekly architecture pass checks cross-page consistency and generation drift; page workers do not redesign the taxonomy independently.

## Initial sequence

1. Rewrite the first screen and choice model of `README.md`.
2. Align `START-HERE.md` and `llms.txt` with distinct human and machine roles.
3. Classify the highest-traffic integration, deployment, benchmark, and security routes.
4. Move dated rationale and superseded explanations out of public paths.
5. Add automated checks for audience metadata, stale front-door links, unexplained negation, and generation ambiguity.
6. Dogfood each route with external-agent and new-contributor tasks.

## Program gates

- The epic and this seed plan exist before broad edits begin.
- At least 50 contract-ready issues cover the obvious routes and supporting process.
- Cohorts are priced for file collisions before live launch.
- Front-door changes include a captured render or structured first-screen witness.
- No claim is accepted from worker narration alone; git history and independent read-back are the completion evidence.
- Remaining findings become issues before a run closes.

## Success measures

- A new reader can state what fak is and run the canonical proof from the README path without opening an internal note.
- Each major option is presented with a default and scoped selection criteria.
- Public routes contain no unexplained internal labels, dated implementation debates, or mode-ambiguous disclaimers.
- Contributor and historical detail remains reachable through explicit deeper links.
- Dogfood sessions show lower time-to-next-action and fewer clarification branches across repeated cohorts.

## Tracking

- Program epic: [#4882](https://github.com/anthony-chaudhary/fak/issues/4882)

