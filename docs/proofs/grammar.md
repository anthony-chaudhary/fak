# D7 · grammar

The grammar rung is the first (cheapest) adjudicator in the fak tool-call pipeline (`abi.RegisterAdjudicator(5, Default)`, grammar.go:261). It enforces **argument-shape well-formedness** — the "`--name` is a flag, not a positional" class of error the model makes when it guesses a tool's argument shape wrong — *before* the call spawns, and where the fix is mechanical it auto-repairs the call in-syscall (a `Transform`) instead of burning a model turn. Its four behaviours: a well-formed call defers (nothing to prove); a malformed-but-repairable call is transformed (positional args zipped into named params, or a known synonym key renamed to its canonical param); a malformed-and-unrepairable call is denied with a model-fixable `MISROUTE` disposition; and a call to a tool with **no** registered grammar defers (FAIL-OPEN). "Correct" here is **decision-procedure soundness (regime D)**: the rung's verdict must be sound (a `Transform` must yield args that preserve the *intended* invocation, and an absent grammar must never cause a refusal), and it must fail closed only where it has authority and fail open where it does not.

---

THEOREM (1) — positional-repair arity zip
For a tool with a loaded grammar of N required params, a malformed call carrying `_positional` with exactly N values (the required named params otherwise missing) is adjudicated `VerdictTransform`/`ReasonMisroute`, and the transformed args are the map `{param_i → positional_i}` — the positional values zipped 1:1 into the grammar's named params, preserving the intended invocation — with the repair counter incremented.
REGIME  D — decision-procedure soundness (transform-correctness).
PROOF   `Adjudicate` (grammar.go:119) resolves the tool's grammar; when the call is not already well-formed (`countMissing > 0`, grammar.go:138) it inspects `m["_positional"]` (grammar.go:160). The repair branch fires **only** on exact arity, `len(pos) == len(g.Params)` (grammar.go:161), then builds `repaired[p.Name] = pos[i]` over the ordered param list (grammar.go:163-166) — an order-preserving zip. Param order is deterministic: `LoadFromJSONSchema` sorts the JSON-schema property names (grammar.go:108) before constructing the grammar, so the i-th positional binds to a stable i-th param. The repaired map is marshalled and stored via `putJSON` (grammar.go:167, 241-255) and returned as `VerdictTransform`/`ReasonMisroute` carrying `TransformPayload.NewArgs` (grammar.go:171-172); `r.repairs++` (grammar.go:169). The arity guard is corroborated negatively — an arity *mismatch* (3 values vs 1 param) cannot be zipped and falls through to `VerdictDeny` (grammar.go:178-181), so the zip is refused rather than silently wrong.
WITNESS `go test ./internal/grammar/ -count=1 -timeout 120s -run 'TestAdjudicatePositionalRepairable'`
        `TestAdjudicatePositionalRepairable` (grammar_test.go:76) asserts `Kind==VerdictTransform`, `Reason==ReasonMisroute`, `By=="grammar"`, the resolved `NewArgs` `reflect.DeepEqual` to `{"name":"alice"}` (grammar_test.go:101-108), and `repairs==1` (grammar_test.go:111) — so the green specifically witnesses the arity-matched zip, not merely a Transform verdict. `TestAdjudicatePositionalUnrepairable` (grammar_test.go:119) witnesses the arity-mismatch `Deny` companion.
VERDICT PROVEN — 2026-06-20. Ran green on this macOS native-go node: `--- PASS: TestAdjudicatePositionalRepairable (0.00s)`; full package `ok ... 0.200s`.
DOS     bound at ship.

---

THEOREM (2) — no grammar ⇒ fail-open Defer (never over-refuse)
For a tool with **no** grammar registered, `Adjudicate` returns `VerdictDefer` (`By="grammar"`) regardless of how malformed the args are (including a positional payload), and neither the repair nor the deny counter is incremented — the rung never over-refuses an un-adjudicable tool.
REGIME  D — decision-procedure soundness (fail-open completeness).
PROOF   `Adjudicate`'s first action is the grammar lookup `d, ok := r.byTool[c.Tool]` (grammar.go:121). When `ok` is false it returns `abi.Verdict{Kind: abi.VerdictDefer, By: "grammar"}` (grammar.go:124-126) **before** any arg parsing, positional inspection, or counter mutation. The two counter increments (`r.repairs++` at grammar.go:169, `r.denies++` at grammar.go:179) live exclusively in the later Transform/Deny branches that the early return never reaches, so the fail-open path is *structurally* incapable of counting a refusal. This is the unit-55 FAIL-OPEN contract documented at grammar.go:12.
WITNESS `go test ./internal/grammar/ -count=1 -timeout 120s -run 'TestAdjudicateNoGrammarDefers'`
        `TestAdjudicateNoGrammarDefers` (grammar_test.go:145) feeds an unregistered tool a positional payload `{"_positional":["a","b"]}` and asserts `Kind==VerdictDefer`, `By=="grammar"`, **and** `Stats()==(0,0)` (grammar_test.go:149-158) — the `(0,0)` assertion is precisely the "never over-refuse" half (no deny counted), so the test witnesses both halves of the theorem, not just the Defer.
VERDICT PROVEN — 2026-06-20. Ran green: `--- PASS: TestAdjudicateNoGrammarDefers (0.00s)`; full package `ok ... 0.200s`.
DOS     bound at ship.

---

## Notes on honesty / scope

- Both theorems are discharged by **existing** tests whose bodies were read and confirmed to assert the *specific* property (the repaired-args `DeepEqual` for (1); the `Stats()==(0,0)` no-refusal assertion for (2)) — not merely a passing package run.
- Adjacent guarantees observed in the same package but outside these two obligations: the **alias-rename** transform (`TestAdjudicateAliasRepair`, grammar_test.go:186) and its no-false-repair guard (`TestAdjudicateAliasNoFalseRepair`, grammar_test.go:225), the **arity-mismatch Deny** (`TestAdjudicatePositionalUnrepairable`), the **well-formed Defer** (`TestAdjudicateWellFormedDefers`), and **content-address dedup** (`TestAddDedup`, unit 57). These corroborate the soundness story but are not the two assigned theorems.
