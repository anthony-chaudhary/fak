---
title: "Tool result-budget policy v1"
description: "Status: proposed spine specification (2026-08-18)"
---
# Tool result-budget policy v1

Status: proposed spine specification (2026-08-18)

## Decision

FAK should treat oversized tool-result requests as a first-class managed-context problem. Add a versioned native policy plugin that can reduce a proven result-count argument before execution, without another model turn, and can route an irreducibly large response through a bounded micro-context branch. The two mechanisms are complementary:

1. **Request shaping** prevents avoidable results from being fetched.
2. **Response branching** isolates results that cannot safely be reduced at request time.

The first production spine is deliberately narrow: an allowlisted, schema-backed integer argument such as `limit`, `first`, `per_page`, `page_size`, `top_k`, or `max_results` is clamped to a configured ceiling; the execution receipt records the original and effective arguments plus the exact policy version. FAK never guesses from an argument name alone.

Working name: **Thimble**. Like Ponytail and Caveman, this is a named, selectable adaptation with a stable version, not prompt folklore. Unlike those model-output styles, Thimble runs at the hard tool-call seam and its decisions are independently auditable.

## Value frame

- **For:** agents and operators paying to read tool output into a limited context window.
- **Problem:** models routinely request the first 100 or 500 records when 5 or 10 would answer the immediate question.
- **Today:** the provider executes the request literally; excess bytes consume latency, tool quota, context, and attention, or require another model call to repair the request.
- **Better because:** FAK can make a deterministic, policy-bounded correction in flight and preserve a continuation path when more evidence is genuinely needed.
- **Witness:** a real tool call requests `limit: 500`, executes with `limit: 10` without a model round trip, and emits a receipt proving the rewrite and policy identity.

The real next-best alternative is prompt guidance such as “request only a few results.” That is cheap but advisory, model-dependent, and not measurable at execution. Thimble keeps prompt guidance as a soft first layer while making the common, provable case enforceable.

## Centrality and problem checklist

**Centrality: Core.** The change directly advances managed context and net-true efficiency at the kernel seam.

| Check | Effect |
|---|---|
| P1 managed context | Prevents low-value tool payload from entering the main context; large unavoidable payload can stay in a child branch. |
| P2 net-true efficiency | Saves work only when avoided transfer, parsing, and prompt cost exceed policy/branch overhead; receipts expose both sides. |
| P3 bounded adaptation | A declared ceiling, tool/argument registry, exemptions, and version pin bound the behavior. |
| P4 integrated operations | The same decision is available to gateway, agent, replay, observability, and policy tooling rather than living in one harness prompt. |

## Non-goals for v1

- Inferring how many records are semantically sufficient from arbitrary natural language.
- Truncating opaque bytes and pretending the tool returned a complete result.
- Treating byte, token, time, pagination cursor, and item-count budgets as interchangeable.
- Silently changing write tools, exports, compliance scans, exhaustive searches, or user-declared cardinality requirements.
- Calling a model to approve the common clamp path.
- Making micro-context branching mandatory for every large response.

## Terms

- **Requested count:** the count explicitly encoded in the model's tool arguments.
- **Effective count:** the count sent to the tool after policy evaluation.
- **Result budget:** a typed ceiling for one dimension (`items`, later `bytes`, `tokens`, or `duration`).
- **Argument contract:** registry evidence that a JSON path controls result cardinality and that lowering it is side-effect-safe.
- **Clamp:** replace a requested integer above the ceiling with the ceiling. A clamp never raises a value.
- **Micro-context branch:** a bounded child context that receives the large result, derives a compact receipt-backed answer, and returns only that answer plus continuation metadata to the parent.

## Policy artifact

The plugin is a data-first artifact, loaded by the native policy/plugin registry and pinned by identity:

```json
{
  "kind": "fak/tool-result-budget",
  "version": "1.0.0",
  "name": "thimble/default",
  "mode": "enforce",
  "default_budget": {"items": 10},
  "contracts": [
    {
      "tool": "github.search_issues",
      "argument": "/per_page",
      "dimension": "items",
      "maximum": 10,
      "minimum": 1,
      "safe_to_reduce": true
    }
  ],
  "on_unshapable_response": "micro_context"
}
```

Identity is `kind + version + canonical-content digest`. Semantic versioning describes contract compatibility; the digest proves the exact policy bytes. Session/replay metadata pins both. Changing defaults without changing identity is invalid. Unknown major versions fail closed in `enforce` mode and report-only in `observe` mode.

## Decision pipeline

The hard seam processes a call in this order:

1. Parse arguments as lossless JSON and preserve the original bytes.
2. Resolve the exact tool plus JSON-pointer argument contract from the versioned registry.
3. Classify intent signals and exemptions from structured session/task metadata, not free-form substring matching.
4. Propose a monotonic reduction (`effective <= requested`); never add an uncontracted argument.
5. Run capability/security policy against both original and proposed calls. A rewrite cannot turn a denied call into an allowed call or widen authority.
6. Emit one decision: `pass`, `clamp`, `branch`, `deny`, or `error`.
7. Execute only the effective call, attach the receipt, and preserve a continuation cursor or rerun recipe when the tool supports one.
8. Measure actual item/byte/token counts and feed calibration; do not claim savings from requested count alone.

No-model fast path: steps 1-7 are deterministic local work. A model may run only inside the separately budgeted response branch.

## Rewrite rules

A v1 clamp is legal only when all are true:

- The tool identity and argument JSON pointer exactly match a loaded contract.
- The argument is an integer in the contract range.
- The contract states that reducing it does not increase side effects or authority.
- No structured exhaustive/export/compliance intent or explicit operator exemption applies.
- The configured mode is `enforce`; `observe` computes but does not mutate.
- The proposed value is lower than the requested value.

Missing count arguments pass in v1. Injection requires a separate contract because omitted values may have tool-specific semantics. Aliases are separate registry entries, not a global name heuristic. Nested paths are supported by JSON pointer. Invalid JSON, floats, strings, negative values, overflow, unknown tools, and unknown arguments do not get “repaired”; they pass to existing validation/policy or fail with its typed error.

## When not to clamp

Correctness outranks token thrift. Pass or exempt calls that declare a bounded need for completeness, including release inventories, security/compliance scans, migrations, exports, exact counts, test matrices, and user requests such as “list all 37.” Tools where page size changes ranking semantics or server-side aggregation need explicit contracts and tests. Write/mutation tools are excluded by default.

An exemption is visible in the receipt and may itself be capped by an operator hard maximum. “The model asked for 500” is not an exemption; structured task intent or operator policy is.

## Micro-context branch

Branching handles a different failure class: the request cannot be safely shaped, or the actual response still exceeds its item/byte/token envelope.

The branch receives:

- the original tool response by reference or bounded stream,
- the parent question and a narrow extraction objective,
- a strict input/output token budget and deadline,
- no ambient tool authority unless separately granted,
- the policy identity and response digest.

It returns a compact answer, citations/item identifiers, completeness status, actual usage, and continuation metadata. The parent never receives an unlabeled truncation. Branch failure returns a typed `branch_failed` outcome and a resumable reference; it does not fabricate a summary. Sensitive data keeps the original policy boundary.

Prefer deterministic projection/filtering before a model branch when the tool schema supports it. Branching must prove net savings after child-model tokens, latency, serialization, and cache effects.

## Decision and receipt schema

```json
{
  "decision": "clamp",
  "reason": "requested_items_above_policy_maximum",
  "tool": "github.search_issues",
  "original_args_sha256": "...",
  "effective_args_sha256": "...",
  "changes": [{"path": "/per_page", "from": 500, "to": 10, "dimension": "items"}],
  "policy": {"name": "thimble/default", "version": "1.0.0", "sha256": "..."},
  "model_round_trips": 0,
  "actual": {"items": 10, "bytes": 18422, "tokens_estimated": 4620},
  "continuation": {"kind": "cursor", "available": true}
}
```

Receipts must redact argument values according to the existing data policy while retaining hashes and numeric budget deltas. Replay applies the pinned policy to the preserved original call and detects drift rather than silently using today's defaults.

## Operator modes

- `off`: no evaluation.
- `observe`: emit proposed decisions and counterfactual estimates; execute original arguments.
- `enforce`: apply legal clamps and configured branch behavior.

Rollout is observe-first per tool contract, then enforce after false-clamp review. Per-tool, per-session, and one-call exemptions are explicit and observable. The operator can set a hard ceiling that no model/task exemption exceeds.

## Metrics and truth conditions

Count decisions by tool, contract, reason, mode, and policy identity. Measure requested/effective/actual items, response bytes, estimated input tokens, tool latency, branch latency/tokens, continuation use, override rate, and downstream re-fetch rate.

A **net-true gain** requires measured avoided cost minus policy and branch overhead. Do not multiply `(requested - effective)` by an assumed row size and report it as fact. Quality guardrails are task success, missing-evidence corrections, immediate re-fetches, and operator overrides. A high clamp rate alone is not success.

## Minimal working spine

1. One versioned built-in policy artifact with one read-only tool contract.
2. One pre-execution adapter that changes `500` to `10` without a model call.
3. One execution receipt containing original/effective hashes, exact change, and policy identity.
4. One captured end-to-end test proving the tool observed `10`, while observe mode proves it would still observe `500`.
5. One negative witness for exhaustive intent and one for an unknown argument/tool.

The spine does not require the micro-context executor, generalized schema inference, or adaptive budgets. Those remain separately dispatchable follow-ons.

## Acceptance invariants

- **Monotonicity:** the adapter never increases result cardinality.
- **Authority preservation:** adaptation never turns policy deny into allow.
- **No hidden incompleteness:** every clamp/branch is disclosed with continuation state.
- **Determinism:** same original call, structured intent, and policy identity yields the same decision.
- **Replay fidelity:** replay uses or explicitly rejects the pinned version.
- **Fail-safe:** malformed/unknown contracts cause no mutation.
- **No extra model call for clamp:** deterministic rewrite reports zero model round trips.
- **Net-truth:** savings claims use observed execution data and include branch overhead.

## Alternatives rejected

- **Prompt/skill only:** useful guidance, but advisory and neither universal nor execution-witnessed.
- **Global `limit` heuristic:** compact but unsafe because names and omission defaults vary by tool.
- **Blind response truncation:** saves parent context but destroys completeness and provenance.
- **Always branch:** controls parent context but still pays retrieval and model overhead.
- **Adaptive policy first:** cannot be bounded or calibrated before the deterministic baseline exists.

## Follow-on decomposition

The implementation backlog should remain issue-sized: contract/schema and version loader; hard-seam adapter; policy ordering; receipts/replay; observe mode; exemptions; nested arguments; provider/tool contract packs; response envelope detector; deterministic projection; micro-context executor; continuation UX; metrics and net-gain scorecard; adversarial/property tests; privacy/redaction; gateway/agent integrations; docs/demo; dogfood; release/claims; and adaptive calibration only after production evidence.
