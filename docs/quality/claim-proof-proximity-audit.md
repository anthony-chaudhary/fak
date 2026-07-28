---
title: "Verify a public fak claim from its scoped proof"
description: "The claim contracts that decide whether one public fak claim is supported: the status, scope, and baseline that must sit adjacent, and the proof route to open."
---

# Verify a public claim from its scoped proof

**Audience:** evaluators deciding whether one public fak claim is supported.

A current public claim is self-contained at the point of evaluation: its status, scope,
baseline when relevant, and link to the proof that supports it appear together. You should
not need to search internal notes or infer behavior from nearby prose.

## Choose the claim contract

Use the contract that matches the claim. The default is the narrowest applicable row.

| Claim kind | Required adjacent context | Scoped authority or proof |
|---|---|---|
| Repository capability or status | Exactly one status: `[SHIPPED]`, `[SIMULATED]`, or `[STUB]`; the claimed scope | The linked entry in [`CLAIMS.md`](../../CLAIMS.md) and its cited test, demo, or implementation |
| Performance, cost, or efficiency | Tuned real baseline, added cost, measurement scope, and provenance (`witnessed`, `observed`, or `modeled`) | The six-question [`net-true value standard`](../standards/net-true-value.md), a reproducible witness, and the canonical number in [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) |
| Shipped or done | The exact shipped unit and version | A witnessed trunk commit and the affected `module@rev`; verify the diff rather than its subject alone |
| Visual or terminal behavior | The rendered surface and the visible condition being claimed | A captured render assertion; when a live UI is involved, the corresponding before/after capture |

A claim may satisfy more than one row. In that case, apply every relevant contract rather
than replacing a stronger proof requirement with a weaker status tag.

## Evaluate one claim

1. Read the claim and only its adjacent status, scope, baseline, provenance, and proof link.
2. Follow that link to the named artifact; check that it measures or renders the same behavior,
   mode, and scope as the claim.
3. Return exactly one verdict:
   - **verified** — the scoped artifact supports the claim as written;
   - **scoped-but-unverified** — the contract is explicit, but the artifact is unavailable,
     stale, or does not independently establish the result;
   - **missing-proof** — required status, scope, baseline, provenance, or proof is absent.

**Next action:** choose one public claim now and complete those three steps. If its verdict is
not **verified**, report the missing field or mismatched artifact; do not broaden the search or
infer the result.

## Context and lifecycle

- **Mode:** `[SHIPPED]` means implemented repository behavior, `[SIMULATED]` means a bounded
  simulation, and `[STUB]` means an intentionally incomplete surface. A mode label does not
  upgrade the strength of its proof.
- **Generation:** this page is a `gen/now` evaluator route. Generated indexes may point here,
  but generated navigation is not evidence for a claim.
- **Lifecycle:** proof is current only while its artifact, baseline, and claimed scope still
  match. A superseding authority or changed implementation requires the adjacent link or claim
  to be updated; otherwise use **scoped-but-unverified**.
- **Support boundary:** missing or inaccessible proof means `not yet`, not an invitation to
  search unrelated private notes. Ask the claim owner for a scoped public artifact.

## Contributor rationale

The repository-wide authoring rules remain in [`AGENTS.md`](../../AGENTS.md), while benchmark
methodology remains in the authorities linked above. This route intentionally contains only the
evaluator's decision contract and next action.
