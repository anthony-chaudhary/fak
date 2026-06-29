# Negative fixtures for the context-contract schema

These are the **must-reject** witnesses for
[`../../context-contract-schema.json`](../../context-contract-schema.json) (the portable
context-contract schema, `#1212` / `G4`). The positive round-trip — the four declared
views ([`../context-view-resident.json`](../context-view-resident.json),
[`../context-view-fault.json`](../context-view-fault.json),
[`../context-view-stale.json`](../context-view-stale.json),
[`../context-view-fabrication.json`](../context-view-fabrication.json)) and the three
review decisions ([`../context-decision-resident.json`](../context-decision-resident.json),
[`../context-decision-fault.json`](../context-decision-fault.json),
[`../context-decision-stale.json`](../context-decision-stale.json),
[`../context-decision-fabrication.json`](../context-decision-fabrication.json)) — shows the
schema *accepts* a well-formed view and its decision. These five files make the acceptance
criterion — "the view-kind set, the taint lattice, the invalidation rule, and the
deny-reason set are a **closed, validatable** vocabulary, and the check is
**fail-closed**" — checkable rather than asserted. Each isolates exactly one defect so
the rejection reason is unambiguous.

A *ContextView* (the authored INPUT) validates against the schema root; a *Decision*
(the review OUTPUT) validates against `$defs/Decision`:

| Fixture | Validated against | Defect | Why the schema rejects it |
|---|---|---|---|
| [`view-kind-out-of-set.json`](view-kind-out-of-set.json) | root (`ContextView`) | `view_kind: "paragraph"` | not in the closed `ViewKind` set — an out-of-set kind can't even be expressed; a runtime that *receives* one treats it as `UNKNOWN_VIEW_KIND`, denied fail-closed |
| [`taint-out-of-set.json`](taint-out-of-set.json) | root (`ContextView`) | `source_taint: "unknown"` | not in the closed `Taint` lattice (`trusted` < `tainted` < `quarantined`) — an unrecognized label is treated as tainted (fail-closed), never admitted |
| [`empty-span.json`](empty-span.json) | root (`ContextView`) | `source.length: 0` | a view must name a non-empty source window (`minimum: 1`) — a zero-length span is `EMPTY_SPAN`, refused (mirrors `memview.ErrEmptySpan`) |
| [`unknown-field.json`](unknown-field.json) | root (`ContextView`) | extra `summary` key | the view is closed (`additionalProperties: false`); an unknown field — here a smuggled model-authored `summary` — is not silently ignored |
| [`deny-missing-reason.json`](deny-missing-reason.json) | `$defs/Decision` | `decision: "deny"`, no `reason` | a non-`allow` verdict MUST carry a reason from the closed `DenyReason` set — a bare refusal with no closed reason is rejected (no free-text refusals) |

## The round-trip, witnessed with no fak engine

Any off-the-shelf Draft 2020-12 validator runs this. The recipe below uses the
`jsonschema` Python package (`pip install jsonschema`) — no fak engine, no network, no
model. Run it from the repo root; it exits non-zero if any expectation is unmet:

```python
import json, sys
from jsonschema import Draft202012Validator

schema = json.load(open("docs/standards/context-contract-schema.json"))
F = "docs/standards/fixtures"

# The root validates a ContextView (the authored INPUT, $ref -> #/$defs/ContextView).
root = Draft202012Validator(schema)
# A Decision is the review OUTPUT — validate it against $defs/Decision, NOT the root
# (the root's $ref is ContextView; a Decision is a different shape).
decision = Draft202012Validator(
    {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": "#/$defs/Decision"})

ok = True
def expect(name, valid, want_valid):
    global ok
    good = valid == want_valid
    ok = ok and good
    print(("PASS" if good else "FAIL"), name,
          "->", "accepted" if valid else "rejected")

# author + review round-trip; every disposition's positive case must validate
for v in ("resident", "fault", "stale", "fabrication"):
    expect(f"view {v} (author)", root.is_valid(json.load(open(f"{F}/context-view-{v}.json"))), True)
for d in ("resident", "fault", "stale", "fabrication"):
    expect(f"decision {d} (review)", decision.is_valid(json.load(open(f"{F}/context-decision-{d}.json"))), True)

# closed vocabulary + fail-closed: every negative must be rejected
for neg in ("view-kind-out-of-set", "taint-out-of-set", "empty-span", "unknown-field"):
    expect(neg, root.is_valid(json.load(open(f"{F}/context-contract-invalid/{neg}.json"))), False)
expect("deny-missing-reason", decision.is_valid(json.load(open(f"{F}/context-contract-invalid/deny-missing-reason.json"))), False)

sys.exit(0 if ok else 1)
```

Expected output — the seven positives accepted, all five negatives rejected:

```
PASS view resident (author) -> accepted
PASS view fault (author) -> accepted
PASS view stale (author) -> accepted
PASS view fabrication (author) -> accepted
PASS decision resident (review) -> accepted
PASS decision fault (review) -> accepted
PASS decision stale (review) -> accepted
PASS decision fabrication (review) -> accepted
PASS view-kind-out-of-set -> rejected
PASS taint-out-of-set -> rejected
PASS empty-span -> rejected
PASS unknown-field -> rejected
PASS deny-missing-reason -> rejected
```
