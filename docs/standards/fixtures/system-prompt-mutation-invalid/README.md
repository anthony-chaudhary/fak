# Negative fixtures for the system-prompt-mutation schema

These are the **must-reject** witnesses for
[`../../system-prompt-mutation-schema.json`](../../system-prompt-mutation-schema.json)
(the witness-gated runtime-modification schema, `#1263` / Rung 5 / `C5` of the
[system-prompt-MMU epic](../../../notes/SYSTEM-PROMPT-MMU-2026-06-29.md)). The positive
round-trips ([`../prompt-edit-admit.json`](../prompt-edit-admit.json) →
[`../prompt-decision-admit.json`](../prompt-decision-admit.json), and the three
refuse/demote pairs) show the schema *accepts* a well-formed edit; these five files make
the acceptance criterion — "the verb / tier / delta / witness vocabularies are a
**closed, validatable** set, and the gate is **fail-closed** and **append-mostly**" —
checkable rather than asserted. Each isolates exactly one defect so the rejection reason
is unambiguous, and each is a *PromptEdit* (validated against the schema root):

| Fixture | Defect | Why the schema rejects it |
|---|---|---|
| [`verb-out-of-set.json`](verb-out-of-set.json) | `verb: "rewrite"` | not in the closed `Verb` set — a full rewrite is not a verb (ACE: full rewrites cause context collapse + brevity bias); a runtime that *receives* one returns `UNKNOWN_VERB` |
| [`tier-out-of-set.json`](tier-out-of-set.json) | `target_tier: "tail"` | not one of the three layers (`spine`/`policy`/`overlay`) — an out-of-set tier is `UNKNOWN_TIER` |
| [`delta-full-rewrite.json`](delta-full-rewrite.json) | `delta_kind: "full_rewrite"` | not in the closed `DeltaKind` set — a full rewrite of a resident block is inexpressible (`FULL_REWRITE_FORBIDDEN`); only `append`/`version_swap`/`mask` are edits |
| [`agent-edit-no-witness.json`](agent-edit-no-witness.json) | `author: "agent"`, `verb: "add"`, no `witness` | an activating agent edit MUST carry a witness (the conditional `required`) — a self-authored edit cannot become resident without one (`WITNESS_REQUIRED`, fail-closed) |
| [`unknown-field.json`](unknown-field.json) | extra `freeform` key | the edit is closed (`additionalProperties: false`); an unconstrained body the agent might slip past the closed vocabulary is rejected, never silently ignored |

## The round-trip, witnessed with no fak engine

Any off-the-shelf Draft 2020-12 validator runs this. The recipe below uses the
`jsonschema` Python package (`pip install jsonschema`) — no fak engine, no network,
no model. Run it from the repo root; it exits non-zero if any expectation is unmet:

```python
import json, sys
from jsonschema import Draft202012Validator

S = "docs/standards/system-prompt-mutation-schema"
schema = json.load(open(f"{S}.json"))
F = "docs/standards/fixtures"

# The root validates a PromptEdit (the authored INPUT).
root = Draft202012Validator(schema)
# An EditDecision is an OUTPUT shape — validate decision fixtures against $defs/EditDecision.
decision = Draft202012Validator(
    {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": "#/$defs/EditDecision"})

ok = True
def expect(name, valid, want_valid):
    global ok
    good = valid == want_valid
    ok = ok and good
    print(("PASS" if good else "FAIL"), name,
          "->", "accepted" if valid else "rejected")

# author + review round-trip: every PromptEdit + EditDecision below must validate
for edit in ("prompt-edit-admit", "prompt-edit-spine-refuse",
             "prompt-edit-self-graded", "prompt-edit-auto-demote"):
    expect(edit, root.is_valid(json.load(open(f"{F}/{edit}.json"))), True)
for dec in ("prompt-decision-admit", "prompt-decision-spine-refuse",
            "prompt-decision-self-graded", "prompt-decision-auto-demote"):
    expect(dec, decision.is_valid(json.load(open(f"{F}/{dec}.json"))), True)

# closed vocabulary + fail-closed: every negative must be rejected
for neg in ("verb-out-of-set", "tier-out-of-set", "delta-full-rewrite",
            "agent-edit-no-witness", "unknown-field"):
    expect(neg, root.is_valid(
        json.load(open(f"{F}/system-prompt-mutation-invalid/{neg}.json"))), False)

sys.exit(0 if ok else 1)
```

Expected output — the four edits + four decisions accepted, all five negatives rejected:

```
PASS prompt-edit-admit -> accepted
PASS prompt-edit-spine-refuse -> accepted
PASS prompt-edit-self-graded -> accepted
PASS prompt-edit-auto-demote -> accepted
PASS prompt-decision-admit -> accepted
PASS prompt-decision-spine-refuse -> accepted
PASS prompt-decision-self-graded -> accepted
PASS prompt-decision-auto-demote -> accepted
PASS verb-out-of-set -> rejected
PASS tier-out-of-set -> rejected
PASS delta-full-rewrite -> rejected
PASS agent-edit-no-witness -> rejected
PASS unknown-field -> rejected
```
