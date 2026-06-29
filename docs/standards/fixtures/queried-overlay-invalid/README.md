# Negative fixtures for the queried-harness-overlay schema

These are the **must-reject** witnesses for
[`../../queried-overlay-schema.json`](../../queried-overlay-schema.json) (the
queried-harness-overlay schema, `#1261` / Rung 3 / `C3` of the
[system-prompt-MMU epic](../../../notes/SYSTEM-PROMPT-MMU-2026-06-29.md)). The positive
round-trips ([`../overlay-query-fault.json`](../overlay-query-fault.json) →
[`../overlay-decision-fault.json`](../overlay-decision-fault.json), and the `hit` and
`grown` pairs) show the schema *accepts* a well-formed query + decision; these five files
make the acceptance criteria — "fill by **query, not menu**; the base holds **no bodies**;
the kind/disposition/reason vocabularies are a **closed, validatable** set; an unknown
field is **rejected**, not silently ignored" — checkable rather than asserted. Each isolates
exactly one defect so the rejection reason is unambiguous, and each is an *OverlayQuery*
(validated against the schema root):

| Fixture | Defect | Why the schema rejects it |
|---|---|---|
| [`query-no-intent.json`](query-no-intent.json) | `intent: ""` | the intent is required and non-empty (`minLength: 1`) — an empty intent is a full-menu dump, not a query; a runtime that receives one returns `NO_INTENT` (fill by **query, not menu**) |
| [`card-body-at-rest.json`](card-body-at-rest.json) | a catalog card carries a `body` field | a `CapCard` is closed (`additionalProperties: false`) and has **no** body field — the base is pointers, not bodies (invariant 4); a card holding its body at rest is `BODY_AT_REST` |
| [`unknown-kind.json`](unknown-kind.json) | `ref.kind: "plugin"` | not in the closed `CapKind` set (`skill`/`mcp-tool`/`a2a-agent`, the shipped `capindex.CapKind`) — an out-of-set kind is `UNKNOWN_KIND` |
| [`unknown-field.json`](unknown-field.json) | extra top-level `freeform` key | the query is closed (`additionalProperties: false`); an unconstrained body the agent might slip past the closed vocabulary is rejected, never silently ignored |
| [`missing-budget.json`](missing-budget.json) | no `token_budget` | the budget is required — there is no "fault in everything" default; without a budget the selector cannot bound the overlay |

## The round-trip, witnessed with no fak engine

Any off-the-shelf Draft 2020-12 validator runs this. The recipe below uses the
`jsonschema` Python package (`pip install jsonschema`) — no fak engine, no network,
no model. It validates the shapes **and** asserts the four Rung-3 acceptance clauses as
cross-field checks over the decision fixtures. Run it from the repo root; it exits
non-zero if any expectation is unmet:

```python
import json, sys
from jsonschema import Draft202012Validator

S = "docs/standards/queried-overlay-schema"
schema = json.load(open(f"{S}.json"))
F = "docs/standards/fixtures"
load = lambda p: json.load(open(f"{F}/{p}.json"))

# The root validates an OverlayQuery (the authored INPUT).
root = Draft202012Validator(schema)
# An OverlayDecision is an OUTPUT shape — validate decision fixtures against $defs/OverlayDecision.
decision = Draft202012Validator(
    {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": "#/$defs/OverlayDecision"})

ok = True
def expect(name, got, want):
    global ok
    good = got == want
    ok = ok and good
    print(("PASS" if good else "FAIL"), name, "->", got)

# 1) shape round-trip: every OverlayQuery + OverlayDecision below must validate
for q in ("overlay-query-fault", "overlay-query-hit", "overlay-query-grown"):
    expect(q, root.is_valid(load(q)), True)
for d in ("overlay-decision-fault", "overlay-decision-hit", "overlay-decision-grown"):
    expect(d, decision.is_valid(load(d)), True)

# 2) closed vocabulary + fail-closed: every negative must be rejected
for neg in ("query-no-intent", "card-body-at-rest", "unknown-kind",
            "unknown-field", "missing-budget"):
    expect(neg, root.is_valid(load(f"queried-overlay-invalid/{neg}")), False)

# 3) the four Rung-3 acceptance clauses, as cross-field assertions
qf, df = load("overlay-query-fault"), load("overlay-decision-fault")
qh, dh = load("overlay-query-hit"), load("overlay-decision-hit")
qg, dg = load("overlay-query-grown"), load("overlay-decision-grown")

# AC1 — fault only the cards the intent needs; base holds no bodies, flat as the catalog grows.
expect("AC1 selected-subset-of-catalog",
       all(s in [c["ref"] for c in qf["catalog"]] for s in df["selected"]), True)
expect("AC1 base-flat-at-breakpoint (catalog 3)", df["base_tokens"] == qf["breakpoint_at"], True)
expect("AC1 base-flat-at-breakpoint (catalog 6)", dg["base_tokens"] == qg["breakpoint_at"], True)
expect("AC1 zero-for-infinity (base unchanged as catalog doubles)",
       df["base_tokens"] == dg["base_tokens"] and len(qg["catalog"]) > len(qf["catalog"]), True)
expect("AC1 within-budget", df["selected_tokens"] <= qf["token_budget"], True)

# AC2 — re-invocation with an identical digest is a HIT that re-faults nothing.
expect("AC2 hit-on-matching-prior-digest", qh["prior_digest"] == dh["overlay_digest"], True)
expect("AC2 hit-re-faults-nothing", dh["disposition"] == "hit" and dh["faulted"] == 0, True)

# AC3 — one canonical CapRef (the #1144 fold): the schema defines CapRef exactly once.
expect("AC3 single-canonical-CapRef-def",
       "CapRef" in schema["$defs"] and json.dumps(schema).count('"#/$defs/CapRef"') >= 3, True)

# AC4 — the breakpoint does not move when the overlay changes (Rung-2 assertion stays green).
expect("AC4 breakpoint-stable-across-overlays",
       df["base_tokens"] == dh["base_tokens"] == dg["base_tokens"], True)

sys.exit(0 if ok else 1)
```

Expected output — the three queries + three decisions accepted, all five negatives
rejected, and the four acceptance clauses asserted:

```
PASS overlay-query-fault -> True
PASS overlay-query-hit -> True
PASS overlay-query-grown -> True
PASS overlay-decision-fault -> True
PASS overlay-decision-hit -> True
PASS overlay-decision-grown -> True
PASS query-no-intent -> False
PASS card-body-at-rest -> False
PASS unknown-kind -> False
PASS unknown-field -> False
PASS missing-budget -> False
PASS AC1 selected-subset-of-catalog -> True
PASS AC1 base-flat-at-breakpoint (catalog 3) -> True
PASS AC1 base-flat-at-breakpoint (catalog 6) -> True
PASS AC1 zero-for-infinity (base unchanged as catalog doubles) -> True
PASS AC1 within-budget -> True
PASS AC2 hit-on-matching-prior-digest -> True
PASS AC2 hit-re-faults-nothing -> True
PASS AC3 single-canonical-CapRef-def -> True
PASS AC4 breakpoint-stable-across-overlays -> True
```
