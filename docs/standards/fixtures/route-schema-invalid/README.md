# Negative fixtures for the agent-routing schema

These are the **must-reject** witnesses for
[`../../agent-routing-schema.json`](../../agent-routing-schema.json) (the portable
agent-routing schema, `#1215` / `G8`). The positive round-trip
([`../route-schema-manifest.json`](../route-schema-manifest.json) →
[`../route-schema-plan.json`](../route-schema-plan.json)) shows the schema *accepts*
a valid policy; these four files make acceptance criterion #2 — "the reduction set is
a closed, **validatable** vocabulary" — checkable rather than asserted. Each isolates
exactly one defect so the rejection reason is unambiguous, and each is a *Manifest*
(validated against the schema root):

| Fixture | Defect | Why the schema rejects it |
|---|---|---|
| [`reduce-out-of-set.json`](reduce-out-of-set.json) | `reduce: "average"` | not in the closed `Reduction` enum — an out-of-set token is `UNCLASSIFIED`, refused at the boundary |
| [`plan-zero-members.json`](plan-zero-members.json) | `default.members: []` | a plan must carry ≥ 1 member; an empty default is not fail-closed |
| [`ensemble-missing-reduction.json`](ensemble-missing-reduction.json) | 2 members, no `reduce` | an ensemble (`members > 1`) must name a reduction |
| [`unknown-field.json`](unknown-field.json) | extra `fallback` key | the manifest is closed (`additionalProperties: false`); an unknown field is not silently ignored |

## The round-trip, witnessed with no fak engine

Any off-the-shelf Draft 2020-12 validator runs this. The recipe below uses the
`jsonschema` Python package (`pip install jsonschema`) — no fak engine, no network,
no model. Run it from the repo root; it exits non-zero if any expectation is unmet:

```python
import json, sys
from jsonschema import Draft202012Validator

S = "docs/standards/agent-routing-schema"
schema = json.load(open(f"{S}.json"))
F = "docs/standards/fixtures"

root = Draft202012Validator(schema)
# A Decision is an OUTPUT shape — validate the plan fixture against $defs/Decision.
decision = Draft202012Validator(
    {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": "#/$defs/Decision"})

ok = True
def expect(name, valid, want_valid):
    global ok
    good = valid == want_valid
    ok = ok and good
    print(("PASS" if good else "FAIL"), name,
          "->", "accepted" if valid else "rejected")

# author + review round-trip: both must validate
expect("manifest (author)", root.is_valid(json.load(open(f"{F}/route-schema-manifest.json"))), True)
expect("plan (review)", decision.is_valid(json.load(open(f"{F}/route-schema-plan.json"))), True)

# closed vocabulary + fail-closed: every negative must be rejected
for neg in ("reduce-out-of-set", "plan-zero-members",
            "ensemble-missing-reduction", "unknown-field"):
    expect(neg, root.is_valid(json.load(open(f"{F}/route-schema-invalid/{neg}.json"))), False)

sys.exit(0 if ok else 1)
```

Expected output — the manifest and plan accepted, all four negatives rejected:

```
PASS manifest (author) -> accepted
PASS plan (review) -> accepted
PASS reduce-out-of-set -> rejected
PASS plan-zero-members -> rejected
PASS ensemble-missing-reduction -> rejected
PASS unknown-field -> rejected
```
