# Public documentation style

**Primary audience:** writers and reviewers changing a public documentation route.

**Lifecycle:** current writing standard. **Generation:** `gen/now`. **Support:** maintained for public documentation. **Authority:** this route owns public prose shape; product behavior, support status, and performance claims remain governed by their named authority and evidence routes.

**Next action:** rewrite one reader-facing block with the [five-step edit](#five-step-edit), then run the [mechanical review](#mechanical-review) before publication.

Public prose leads with the current supported behavior, gives the reader the choice and default they need, and names scope beside the claim. It sends implementation rationale, alternatives, and history to deeper contributor routes.

## Five-step edit

Apply these steps in order to each reader-facing block.

1. **Name the audience and job.** State who should use the route and the outcome it helps them reach.
2. **Lead affirmatively.** Say what the current path does before limits, alternatives, or prior behavior.
3. **Expose the choice.** List valid modes or paths, select a default, and state the condition that changes it.
4. **Bind scope and proof.** Put lifecycle, generation, support, measurement scope, and evidence links next to the claim they qualify.
5. **End with an action.** Give one observable link, command, decision, or destination the reader can take now.

Use the shortest wording that preserves all five facts. Concision removes repetition and internal narrative; it does not remove qualifiers, defaults, failure conditions, or evidence provenance.

## Choose the explanation depth

| Reader need | Placement | Content |
|---|---|---|
| Complete the current task | Public route | Current behavior, applicable choices, default, scope, proof cue, and next action |
| Evaluate a limitation or tradeoff | Scoped deeper section or linked standard | Named boundary, consequence, alternative, and evidence |
| Understand implementation or history | Contributor or historical route | Internals, rejected alternatives, chronology, and issue context |

**Default:** keep task-completion facts on the public route. Link deeper material when the reader asks why, needs to evaluate a boundary, or contributes to the implementation.

## Affirmative wording

Start with the supported subject and verb:

- `Use structured capture for prose-only front doors.`
- `The gateway supports the modes listed in the current matrix.`
- `This benchmark measures the named workload on the recorded environment.`

Then state a limit as a scoped boundary with its consequence and route forward:

- `Structured capture witnesses information order; use an image when placement or wrapping affects the claim.`
- `Preview support covers evaluation; use a stable path for production.`

A bare negative can have several meanings. Replace `does not support X` with the applicable fact: `X is outside this route`, `X is planned`, `X is historical`, `X requires mode Y`, or `X has no current witness`. Preserve a required prohibition when safety or correctness depends on it; name the actor, prohibited action, condition, consequence, and supported alternative.

Use the [public negation audit](../quality/public-negation-audit.md) when reviewing negation across front-door routes.

## Scoped claims

A public claim is complete when a reader can answer:

- **Subject:** which component, route, mode, or workflow?
- **Predicate:** what observable behavior or status?
- **Context:** which lifecycle, generation, support tier, workload, or environment?
- **Evidence:** which authority or witness proves it?
- **Action:** what should this audience do with the claim?

Keep a qualifier in the same sentence or adjacent block as its claim. Prefer `In the recorded workload, mode A reduced median latency by N% against baseline B` over `Mode A is faster`. Apply the [net-true value standard](net-true-value.md) to efficiency claims and the [generation drift check](../quality/generation-drift-check.md) to current-generation routes.

## Choice tables

Use a table when a reader must choose among two or more valid paths. Each row names:

1. the reader condition;
2. the supported choice;
3. the outcome or evidence supplied.

State the default immediately below the table and name every upgrade or exception condition. If the route has one supported path, state it directly rather than manufacturing a choice.

## Mechanical review

Review the candidate from top to bottom. Mark `pass` or `repair` for each check; one `repair` result blocks publication.

| Check | Pass condition |
|---|---|
| Audience and job | One primary audience and one task outcome are explicit |
| Current lead | The first explanation states supported current behavior affirmatively |
| Choice | Every real choice, default, and change condition is visible; nonexistent choices are not invented |
| Scope | Lifecycle, generation, support, and claim qualifiers sit beside the affected statement |
| Authority | Prose shape points here; behavior and evidence point to their owning routes |
| Negation | Each negative is unambiguous, scoped, and paired with a consequence or supported path |
| Depth | Public task facts stay on the route; internal debate and history move deeper |
| Action | One checkable next action is explicit |
| Links and proof | Changed links resolve and named evidence supports only the claim made |

After the mechanical pass, capture the first screen with the [front-door render witness](../testing/frontdoor-render-witness.md) when presentation matters and run an [external-reader read-back](../testing/external-reader-dogfood.md). Publish only when the reader independently restates the audience, job, applicable choice, scope, and next action without an unresolved ambiguity.

## Publication record

Record the changed route, revision or content hash, mechanical results, changed-link witness, applicable render artifact, independent read-back, and dispositions for every ambiguity. A same-run repair may close its finding; remaining work links to an open issue before publication.
