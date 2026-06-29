# Negative fixtures for the verification-ladder spec

These are the **must-reject** witnesses for
[`../../verification-ladder-spec.json`](../../verification-ladder-spec.json) (the
declarable verification-ladder schema, `#1210` / `G2`). The positive round-trip — the
ladder [`../verification-ladder.json`](../verification-ladder.json) plus the two
decisions it produces,
[`../verification-ladder-decision-read.json`](../verification-ladder-decision-read.json)
(a low-risk read stops at rung 1) and
[`../verification-ladder-decision-write.json`](../verification-ladder-decision-write.json)
(a write climbs to rung 3) — shows the schema *accepts* a well-formed ladder and the
decisions a walk over it yields. These five files make the acceptance criteria — "the
verdict set is a **closed, validatable** vocabulary" and "the ladder is **fail-closed by
construction** (an `INDETERMINATE` rung escalates, never silently allows)" — checkable
rather than asserted. Each isolates exactly one defect so the rejection reason is
unambiguous:

| Fixture | Validates against | Defect | Why the schema rejects it |
|---|---|---|---|
| [`ladder-fails-open.json`](ladder-fails-open.json) | `Ladder` (root) | `on_exhaustion: "allow"` | the fail-closed tail is pinned to `const "deny"` — a ladder cannot declare a fail-OPEN exhaustion; the recipe rule is structural |
| [`ladder-no-escalate.json`](ladder-no-escalate.json) | `Ladder` (root) | `escalate_on` omits `indeterminate` | `escalate_on` MUST `contains` `indeterminate` — else an `INDETERMINATE` rung has nothing to trigger a climb and could be silently dropped |
| [`unknown-field.json`](unknown-field.json) | `Ladder` (root) | extra `domain` key | the ladder is closed (`additionalProperties: false`) and domain-free; a host-specific knob cannot smuggle in |
| [`verdict-out-of-set.json`](verdict-out-of-set.json) | `$defs/Decision` | a path step's `verdict: "maybe"` | not in the closed `Verdict` set `{allow, deny, indeterminate, defer}` — out-of-set drift is refused at the boundary |
| [`indeterminate-final-verdict.json`](indeterminate-final-verdict.json) | `$defs/Decision` | final `verdict: "indeterminate"` | the committed verdict is closed to `{allow, deny}` — an `INDETERMINATE` can never be the last word (a residual one folds to `deny` via `on_exhaustion`) |

## The round-trip, witnessed with no fak engine

Any off-the-shelf Draft 2020-12 validator runs this. The recipe below uses the
`jsonschema` Python package (`pip install jsonschema`) — no fak engine, no network, no
model. It does two things: (1) validates the four positive fixtures against their schema
parts and rejects all five negatives, and (2) **re-derives the smallest-sufficient rung
from the declared ladder data** and asserts each decision reached exactly that rung — so
"a low-risk call stops at rung 1; a write climbs" is *computed from the data*, not just
read off the fixture. Run it from the repo root; it exits non-zero if any expectation is
unmet:

```python
import json, sys
from jsonschema import Draft202012Validator

S = "docs/standards/verification-ladder-spec"
schema = json.load(open(f"{S}.json"))
F = "docs/standards/fixtures"

# The root validates a Ladder (the authored policy).
ladder_v = Draft202012Validator(schema)
# A Decision is an OUTPUT shape — validate decision fixtures against $defs/Decision.
decision_v = Draft202012Validator(
    {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": "#/$defs/Decision"})

ok = True
def expect(name, valid, want_valid):
    global ok
    good = valid == want_valid
    ok = ok and good
    print(("PASS" if good else "FAIL"), name, "->", "accepted" if valid else "rejected")

def check(name, cond):
    global ok
    ok = ok and cond
    print(("PASS" if cond else "FAIL"), name)

# (1) schema validity: every positive validates, every negative is rejected
ladder   = json.load(open(f"{F}/verification-ladder.json"))
dec_read  = json.load(open(f"{F}/verification-ladder-decision-read.json"))
dec_write = json.load(open(f"{F}/verification-ladder-decision-write.json"))
expect("ladder (author)",          ladder_v.is_valid(ladder),     True)
expect("decision read (review)",   decision_v.is_valid(dec_read),  True)
expect("decision write (review)",  decision_v.is_valid(dec_write), True)
expect("ladder-fails-open",        ladder_v.is_valid(json.load(open(f"{F}/verification-ladder-invalid/ladder-fails-open.json"))),  False)
expect("ladder-no-escalate",       ladder_v.is_valid(json.load(open(f"{F}/verification-ladder-invalid/ladder-no-escalate.json"))), False)
expect("unknown-field",            ladder_v.is_valid(json.load(open(f"{F}/verification-ladder-invalid/unknown-field.json"))),       False)
expect("verdict-out-of-set",       decision_v.is_valid(json.load(open(f"{F}/verification-ladder-invalid/verdict-out-of-set.json"))),        False)
expect("indeterminate-final",      decision_v.is_valid(json.load(open(f"{F}/verification-ladder-invalid/indeterminate-final-verdict.json"))), False)

# (2) smallest-sufficient-rung selection, DERIVED from the declared ladder (AC#3).
# Risk is ordered; a rung "covers" a claim iff its max_risk is >= the claim's risk.
RISK = {"read": 0, "write": 1, "self_modify": 2}
covers = lambda max_risk, rc: RISK[max_risk] >= RISK[rc]
# The reuse/vDSO rung only decides a REPEATED call (a cache hit), so the smallest rung
# that can conclusively decide a NOVEL claim is the cheapest non-reuse rung that covers it.
conclusive = [r for r in ladder["rungs"] if r["cost"] != "reuse"]
first = min(r["id"] for r in conclusive)                                   # cheapest conclusive rung
select = lambda rc: min(r["id"] for r in conclusive if covers(r["max_risk"], rc))

# a low-risk READ stops at the cheapest conclusive rung; it never climbs
check("read selects rung 1",  select("read") == 1 == dec_read["rung_reached"])
check("read does not climb",   (select("read") != first) == dec_read["climbed"] == False)
# a WRITE cannot be allowed by a read-only rung -> it climbs to the require-witness rung
check("write selects rung 3", select("write") == 3 == dec_write["rung_reached"])
check("write climbs",          (select("write") != first) == dec_write["climbed"] == True)
check("riskier claim reaches a costlier rung",
      dec_write["rung_reached"] > dec_read["rung_reached"])

sys.exit(0 if ok else 1)
```

Expected output — the four positives accepted, all five negatives rejected, and the
smallest-sufficient-rung selection re-derived and confirmed:

```
PASS ladder (author) -> accepted
PASS decision read (review) -> accepted
PASS decision write (review) -> accepted
PASS ladder-fails-open -> rejected
PASS ladder-no-escalate -> rejected
PASS unknown-field -> rejected
PASS verdict-out-of-set -> rejected
PASS indeterminate-final -> rejected
PASS read selects rung 1
PASS read does not climb
PASS write selects rung 3
PASS write climbs
PASS riskier claim reaches a costlier rung
```
