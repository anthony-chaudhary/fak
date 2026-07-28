---
title: "Documentation metadata template"
description: "The smallest metadata block that tells a reader a page's audience, lifecycle, generation, support state, authority, and next action, plus its completion check."
---

# Documentation metadata

> **Audience:** public documentation writers and reviewers labeling who a page serves and when its guidance applies.  
> **Lifecycle:** current writing template. **Generation:** `gen/now`. **Support:** maintained as the current public-documentation metadata template.  
> **Authority:** this template owns metadata presentation; the linked product, support, and evidence routes own the facts placed in it.

Use the smallest metadata block that lets a reader identify the page's audience, applicability, authority, and next action without reading a process preamble. Choose the applicable pattern below, fill every required field from an owning source, and then run the [completion check](#completion-check).

## Field contract

| Field | State the checkable fact | Do not substitute |
|---|---|---|
| **Audience** | One primary reader group and the task that brings it to the page. | A list of everyone who might read it. |
| **Lifecycle** | `current`, `preview`, `historical`, or `archived`, using the meanings below. | Vague recency such as "new" or "legacy." |
| **Generation** | The exact documentation stream or named generation, such as `gen/now`. | "Latest," "modern," or an inferred product version. |
| **Support** | The support state supplied by the owning authority, with its scope. | A promise inferred from this template or from lifecycle. |
| **Authority** | What this page owns and the route that owns each delegated fact. | Internal debate, implementation history, or an unsupported claim. |
| **Applicability / mode** | A global condition when all choices share it; otherwise a condition on each affected choice. | A mode-ambiguous negation such as "not supported" without naming where or when. |
| **Next action** | One observable action the primary audience can take now. | "Learn more" or several competing actions. |

Lifecycle describes the page, not product support:

- **current** — the page is the active authority for its stated job;
- **preview** — the page is usable only under its named preview condition and points to the current stable route;
- **historical** — the page preserves evidence or behavior from a named generation and points to its current successor;
- **archived** — the page is retained for recordkeeping, is not maintained as current guidance, and points to a successor when one exists.

Do not invent a support vocabulary here. For product behavior, copy the exact state and scope from the product or support authority. This page itself is the documentation authority for the maintenance state of this metadata template. If no support authority defines a state, write `Support: not specified by the owning authority` and link that authority rather than upgrading silence into a promise.

## Choose a metadata pattern

### Compact current route — default

Use this block when audience, lifecycle, generation, support, authority, and applicability are shared by the whole page. This is the default for a current single-mode route.

```markdown
> **Audience:** [one reader group] performing [one task].  
> **Lifecycle:** current. **Generation:** [`gen/now` or exact generation]. **Support:** [authority-defined state and scope].  
> **Authority:** this page owns [page job]; [linked owning route] owns [named delegated fact].  
> **Next action:** [one observable action].
```

Example:

> **Audience:** public documentation writers labeling a current reader route.  
> **Lifecycle:** current. **Generation:** `gen/now`. **Support:** maintained as the current metadata template by this documentation authority.  
> **Authority:** this page owns metadata presentation; the linked product and evidence routes own behavior and proof.  
> **Next action:** copy the compact block and replace every bracketed value.

### Choice-specific applicability

Use this pattern when modes differ in support, lifecycle, generation, or applicability. Put facts shared by every choice above the table and put each differing fact in its choice row. Use the [multi-mode choice-table template](documentation-choice-table.md) so each condition selects one supported choice.

```markdown
> **Audience:** [one reader group] choosing [one task path].  
> **Authority:** this page owns [choice job]; [linked owning route] owns [named delegated fact].  
> **Next action:** select the row whose condition is true, then [one observable action].

| When this is true | Choose | Lifecycle / generation / support | Outcome | Change condition | Proof |
|---|---|---|---|---|---|
| [observable condition] | **[one choice]** | [only the facts that differ] | [result] | [when to switch] | [authority or evidence] |
```

Do not place one global `Support:` or `Lifecycle:` label above the table when it would erase a row-level difference.

### Preview, historical, or archived route

Use this block when the page is not the unqualified current route. A preview names its entry condition and labels its successor consistently as `Current route`. A historical or archived page names its exact generation and current successor; preserved content cannot act as current authority.

```markdown
> **Audience:** [one reader group] investigating [preview or past-generation task].  
> **Lifecycle:** [preview | historical | archived]. **Generation:** [exact stream, release, date, or commit range].  
> **Applicability:** [entry condition for preview, or preserved scope for historical/archive use].  
> **Support:** [affirmative authority-defined scoped state, or "not specified by the owning authority"].  
> **Current route:** [stable alternative for preview, or current successor for historical/archive use].  
> **Authority:** this page owns [bounded preview or historical record]; [current route] owns current guidance.  
> **Next action:** [enter the preview under its condition, or continue to the current route].
```

Preview example:

> **Audience:** integrators evaluating a preview documentation path.  
> **Lifecycle:** preview. **Generation:** `gen/now`.  
> **Applicability:** use only when the preview flag named by the owning product route is enabled.  
> **Support:** preview within the scope defined by the owning support route.  
> **Current route:** the stable integration guide.  
> **Authority:** this page owns the preview procedure; the stable guide owns the default procedure.  
> **Next action:** verify the preview condition, or continue to the stable guide.

Historical example:

> **Audience:** maintainers tracing behavior from release `v1`.  
> **Lifecycle:** historical. **Generation:** release `v1`. **Support:** preserved evidence for release `v1`.  
> **Current route:** the `gen/now` behavior guide.  
> **Authority:** this page owns the preserved `v1` record; the `gen/now` guide owns current guidance.  
> **Next action:** cite the preserved evidence for a `v1` investigation, or continue to the current guide.

## Authority and navigation

Keep the metadata block near the title so direct inbound readers encounter it before explanatory prose. Link delegated facts at the point they matter:

- use the [public documentation style contract](../standards/public-documentation-style.md) for prose shape;
- use the [generation drift check](../quality/generation-drift-check.md) to verify generation markers and current successors;
- use the [external-reader dogfood rubric](https://github.com/anthony-chaudhary/fak/blob/main/docs/testing/external-reader-dogfood.md) to test whether a reader can orient and act without coaching.

This template does not decide product behavior, product support, evidence strength, or which route is current. Their owning authorities do. Summarize only the fact required for the reader's decision and link the authority; move implementation history and internal debate to a contributor or historical route.

## Completion check

Before publication, verify all of the following:

- [ ] One primary audience and its page task are explicit.
- [ ] One next action is observable and applicable now.
- [ ] Lifecycle and exact generation are stated; preview, historical, and archived routes name a stable alternative or current successor.
- [ ] Support wording and scope match a linked owning authority, or explicitly say that authority does not specify support.
- [ ] Authority distinguishes what this page owns from delegated product, support, and evidence facts.
- [ ] Shared applicability is global; differing mode facts are attached to the applicable choice rows.
- [ ] No internal debate or mode-ambiguous negation carries the public explanation.
- [ ] Every local link and anchor resolves.
- [ ] An independent reader can restate the page job, choose the applicable pattern, and name the next action without coaching.

**Next action:** choose the applicable pattern, replace every placeholder with authority-backed facts, and run this completion check.

