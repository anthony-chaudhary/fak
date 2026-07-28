---
title: "Audit mode-specific boundaries in public routes"
description: "A bounded wording audit that keeps a public front door affirmative and makes every boundary name its subsystem, mode, generation, and supporting evidence."
---

# Audit mode-specific boundaries in public routes

**Primary audience:** maintainers reviewing current public and contributor front doors.
**Lifecycle:** current process authority. **Generation:** `gen/now`.
**Authority:** this page defines the wording-audit procedure; runtime behavior remains authoritative in code and tests, and product support remains authoritative in [`docs/supported/`](../supported/README.md).
**Next action:** run the candidate search against the six inventory-marked routes, inspect every hit in context, and record each ambiguous finding in the witness table below.

This audit keeps a route's current behavior affirmative and makes every boundary name its subsystem, mode, generation, and evidence. A negative word is a search lead, not a defect by itself.

## Audit set and revision

The bounded audit set comes from entries with `front_door_linked: true` in the [documentation audience inventory](../project/documentation-audience-inventory.json):

- `README.md`
- `START-HERE.md`
- `llms.txt`
- `AGENTS.md`
- `CONTRIBUTING.md`
- `SECURITY.md`

The result below audits those six files at commit `bad486577331c70d6d4cd93c5625a76e9c4c23df`. It covers prose outside fenced code and HTML comments. Adding a route to the inventory, changing one of these routes, or superseding the [audience architecture](../notes/DOCUMENTATION-AUDIENCE-ARCHITECTURE-2026-07-15.md) makes the result stale and requires a new revision entry.

## Candidate search

Search case-insensitively for whole-word forms `no`, `not`, `never`, `without`, `cannot`, `can't`, `doesn't`, `does not`, `isn't`, `is not`, and `won't`. Record the file, line, matched sentence, and enough adjacent prose to identify what the statement governs. Exclude fenced commands and examples from the lexical count, then review visible headings, tables, and prose manually.

The search is deliberately broad. Each hit becomes a finding only after contextual classification; counts alone do not measure route quality.

## Classification contract

Classify each candidate once:

| Class | Decision rule | Disposition |
|---|---|---|
| **Affirmative boundary** | The sentence first names current behavior and scopes the boundary to a subject and mode, such as an offline proof that uses a deterministic planner. | Keep; link its proof or operating route. |
| **Scoped contrast** | A contrast sharpens an already-defined concept and identifies the affected subsystem or artifact. | Keep when the contrast shortens the route; otherwise rewrite affirmatively. |
| **Mode-ambiguous** | A reader cannot tell whether the statement governs preflight, offline agent, live agent, gateway, serving backend, or every mode. | Finding: rewrite with the mode and current authority. |
| **Debate-led** | What the project rejects, used to reject, or argues against carries the explanation before current behavior appears. | Finding: lead with current behavior and route rationale deeper. |
| **Unsupported absolute** | `never`, `cannot`, or an equivalent universal claim lacks a named scope, generation, or proof. | Finding: scope and prove it, or remove it. |
| **Historical or research-only** | The statement reports dated investigation or a superseded design rather than the current contract. | Move to a dated provenance route; do not use it as current front-door authority. |

A contributor prohibition can be an affirmative boundary: for example, a named repository operation with an enforced guard and recovery action is scoped behavior, not unexplained product negation. Conversely, “no model is used” is a finding when the surrounding route does not say whether it means deterministic preflight, `agent --offline`, a policy decision, or every live request.

For every candidate, answer these checks before assigning a disposition:

1. **Subject:** which component or actor does the sentence describe?
2. **Mode:** which preflight, offline, live-agent, gateway, or serving path does it govern?
3. **Generation and lifecycle:** is this the current `gen/now` contract, a dated observation, or a superseded design?
4. **Support:** does the sentence describe documented support, an evaluation proof, or merely an example?
5. **Evidence:** which command, test, maintained contract, or support route lets a reader verify it?

A missing answer to subject or mode makes the candidate mode-ambiguous. Missing generation, support, or evidence makes it an unsupported or historical finding even when the sentence reads clearly.

## Current audit result

The revision audit found **170 candidate lines across six routes**:

| Route | Candidate lines | Contextual result |
|---|---:|---|
| `README.md` | 9 | No finding. Offline, policy, cache, and standalone-server boundaries name their mode or subject. |
| `START-HERE.md` | 2 | No finding. Both statements are explicitly scoped to the offline proof. |
| `llms.txt` | 2 | No finding. The map states its reader action and the current-versus-historical authority boundary. |
| `AGENTS.md` | 120 | No finding. Candidates are repository-operation rules with named scopes, gates, recovery actions, or mode-specific proof text. |
| `CONTRIBUTING.md` | 24 | No finding. Candidates are contributor workflow and licensing boundaries with explicit subjects and routes. |
| `SECURITY.md` | 13 | No finding. Preflight, offline containment, supported-version, and disclosure boundaries identify their scope and evidence. |
| **Total** | **170** | **0 mode-ambiguous, debate-led, unsupported-absolute, or historical-authority findings.** |

Representative read-backs anchor the decision:

- `README.md` identifies the deterministic mock planner before limiting what the offline proof establishes; it does not generalize that boundary to live-model quality or latency.
- `SECURITY.md` names deterministic preflight and `agent --offline`, then routes production coverage to the selected managed path.
- `AGENTS.md` uses strong prohibitions as enforceable contributor contracts, each attached to the affected operation and usually a recovery command; those statements do not describe runtime model behavior.

The zero-finding result applies only to the stated files and revision. It is evidence that this bounded set passed the procedure, not a claim that every repository document or future revision is free of ambiguity.

## Record a finding

When a later audit finds ambiguity, create or link a deduplicated issue before declaring the audit complete, then add one row:

| Field | Required value |
|---|---|
| Revision and route | Full audited commit and inventory-listed path |
| Candidate | Line reference plus the surrounding affirmative definition |
| Classification | One class from the contract above |
| Missing context | Subject, mode, generation/lifecycle, support status, or evidence |
| Replacement | Proposed affirmative wording and its authority link |
| Disposition | Open issue, fixed commit, or superseding route |

Do not silently convert wording into a product claim. If authority is missing, record that absence and route the behavior question to its owning code, test, or [support contract](../supported/README.md).

## Structured witness

A repeat audit is complete only when its witness records:

```text
inventory_revision: <full commit>
routes: <ordered inventory paths>
search_terms: <exact whole-word list>
excluded_surfaces: fenced code, HTML comments
candidate_counts: <per route and total>
findings: <classification, context, authority, disposition>
independent_reader:
  audience: <who uses the page>
  page_job: <what the audit decides>
  candidate_vs_finding: <restated distinction>
  next_action: <the executable review action>
  ambiguities: []
```

Use the [front-door clarity scorecard](frontdoor-clarity-scorecard.md) after resolving findings. Its affirmative-framing, choice, lifecycle, proof, and next-action rows test the repaired route as a whole; this audit supplies the narrower negation evidence.
