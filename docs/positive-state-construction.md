# Positive-state construction: broadcast the target state

> **Doctrine:** Construct and broadcast the state the model should make true; never ask it to subtract, suppress, or invert a salient unwanted state.

## Why this works

A model turn behaves like a shared workspace: salient instructions and observations are broadcast into the circuits that choose the next token and action. Text such as “do not delete X” first makes *delete X* present in that workspace, then asks the model to maintain an inversion over it. Transformers have no dependable, persistent `NOT` operator that removes the represented action from every downstream use. Repeating the warning strengthens the unwanted representation while repeatedly charging the model for the inversion.

Positive-state construction performs that subtraction outside the model. It names what is true now, the allowed action, and the desired end-state. The broadcast therefore contains the state downstream reasoning should reuse rather than an operand it must continuously negate.

This is a construction rule, not a claim that every English word such as “not” is forbidden. A literal reason token, quoted user content, or externally defined contract may need to survive byte-for-byte. The boundary preserves those facts while making fak-authored guidance affordance-first.

## The positive-construction pattern

1. **Name the invariant:** state the task, capability floor, or fact that remains true.
2. **Name the available action:** give the concrete operation the model can perform now.
3. **Name the checkable target:** describe the resulting state or witness.
4. **Remove stale operands:** omit superseded errors, abandoned plans, and “ignore the above” transcript residue.
5. **Preserve required literals:** retain reason codes, identifiers, and user-authored bytes when the contract requires them.

For example, replace “Do not retry `refund_payment`” with “Keep `POLICY_BLOCK`; use `search_kb` with the customer reference.” The reason and blocked tool remain explicit, but the directive broadcasts the available path.

## Surfaces governed by this doctrine

- **Emit-time runtime prose:** [`internal/negframe`](../internal/negframe/) classifies and reframes fak-authored text before it becomes shared model state.
- **Managed context:** compaction, task-pin restore, query-not-chat swaps, and invariant reseeds construct the current working set rather than append a ledger of stale failures.
- **Prompts and skills:** author instructions as allowed actions, invariants, and end-states; use negative examples as quoted evidence rather than the controlling directive.
- **Guard refusals and recovery:** preserve the refusal token and explain the sanctioned next operation instead of repeatedly broadcasting the denied action.
- **Complaints and operator notes:** report the observed fact, current containment, and next checkable remedy; avoid turning the defect signature into a repeated instruction.

## Boundary and fail-safe

Fak may rewrite only text it owns. User content and externally defined tokens are opaque. A rewrite must preserve required literals and remain idempotent; when the positive construction cannot retain the contract, the boundary keeps the original text or refuses the transformation rather than fabricating a safer-sounding instruction.

The implementation contract and tests live beside [`internal/negframe/reframe.go`](../internal/negframe/reframe.go). The managed-session application is described in [Query, not chat](query-not-chat.md).

## Review checklist

A governed emission is ready when a reviewer can answer yes to all five questions:

- Does the first directive state what should happen?
- Is the originating task or other invariant explicit?
- Is the next action available under the current capability floor?
- Are stale failures absent unless needed as evidence?
- Do all required reason tokens, identifiers, and opaque bytes survive?

If the answer to any question is no, construct the desired state before broadcasting the text.
