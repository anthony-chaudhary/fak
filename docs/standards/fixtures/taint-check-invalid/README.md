# Negative fixtures for the taint-check schema

These are the **must-reject** witnesses for
[`../../taint-check-schema.json`](../../taint-check-schema.json) (the portable
taint-check schema, `#1214` / `G7`). The positive round-trip
([`../taint-check-crossing-deny.json`](../taint-check-crossing-deny.json) →
[`../taint-check-decision-deny.json`](../taint-check-decision-deny.json), and the
clean [`../taint-check-crossing-allow.json`](../taint-check-crossing-allow.json) →
[`../taint-check-decision-allow.json`](../taint-check-decision-allow.json)) shows the
schema *accepts* a well-formed crossing; these five files make the acceptance
criterion — "the taint lattice and sink set are a **closed, validatable** vocabulary,
and the check is **fail-closed**" — checkable rather than asserted. Each isolates
exactly one defect so the rejection reason is unambiguous, and each is a *Crossing*
(validated against the schema root):

| Fixture | Defect | Why the schema rejects it |
|---|---|---|
| [`taint-out-of-set.json`](taint-out-of-set.json) | `value_taint: "secret"` | not in the closed `Taint` lattice — an out-of-lattice label can't even be expressed; a runtime that *receives* one treats it as tainted (`UNKNOWN_TAINT`, fail-closed) |
| [`sink-out-of-set.json`](sink-out-of-set.json) | `sink_class: "network"` | not in the closed `SinkClass` set — an unclassified sink is `UNCLASSIFIED_SINK`, refused at the boundary |
| [`boundary-out-of-set.json`](boundary-out-of-set.json) | `boundary: "memory"` | not one of the two lenses (`sink` / `share`) — an out-of-set boundary is `UNKNOWN_BOUNDARY` |
| [`sink-without-class.json`](sink-without-class.json) | `boundary: "sink"`, no `sink_class` | a sink crossing MUST classify its sink (fail-closed) — an unclassified sink is not silently treated as `none` |
| [`unknown-field.json`](unknown-field.json) | extra `domain` key | the crossing is closed (`additionalProperties: false`); an unknown field is not silently ignored |

## The round-trip, witnessed with no fak engine

Any off-the-shelf Draft 2020-12 validator runs this. The recipe below uses the
`jsonschema` Python package (`pip install jsonschema`) — no fak engine, no network,
no model. Run it from the repo root; it exits non-zero if any expectation is unmet:

```python
import json, sys
from jsonschema import Draft202012Validator

S = "docs/standards/taint-check-schema"
schema = json.load(open(f"{S}.json"))
F = "docs/standards/fixtures"

# The root validates a Crossing (the authored INPUT).
root = Draft202012Validator(schema)
# A Decision is an OUTPUT shape — validate the decision fixtures against $defs/Decision.
decision = Draft202012Validator(
    {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": "#/$defs/Decision"})

ok = True
def expect(name, valid, want_valid):
    global ok
    good = valid == want_valid
    ok = ok and good
    print(("PASS" if good else "FAIL"), name,
          "->", "accepted" if valid else "rejected")

# author + review round-trip, both lenses' positive cases: every one must validate
expect("crossing deny (author)", root.is_valid(json.load(open(f"{F}/taint-check-crossing-deny.json"))), True)
expect("decision deny (review)", decision.is_valid(json.load(open(f"{F}/taint-check-decision-deny.json"))), True)
expect("crossing allow (author)", root.is_valid(json.load(open(f"{F}/taint-check-crossing-allow.json"))), True)
expect("decision allow (review)", decision.is_valid(json.load(open(f"{F}/taint-check-decision-allow.json"))), True)

# closed vocabulary + fail-closed: every negative must be rejected
for neg in ("taint-out-of-set", "sink-out-of-set", "boundary-out-of-set",
            "sink-without-class", "unknown-field"):
    expect(neg, root.is_valid(json.load(open(f"{F}/taint-check-invalid/{neg}.json"))), False)

sys.exit(0 if ok else 1)
```

Expected output — the four positives accepted, all five negatives rejected:

```
PASS crossing deny (author) -> accepted
PASS decision deny (review) -> accepted
PASS crossing allow (author) -> accepted
PASS decision allow (review) -> accepted
PASS taint-out-of-set -> rejected
PASS sink-out-of-set -> rejected
PASS boundary-out-of-set -> rejected
PASS sink-without-class -> rejected
PASS unknown-field -> rejected
```
