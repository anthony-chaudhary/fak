# D1 · adjudicator

The `internal/adjudicator` package is fak's in-process **reference monitor** — the v0.1
realization of the Adjudicator seam, the fused zero-spawn dual of the `dos-preflake` hook.
Given a decoded `abi.ToolCall` and a `Policy` (a decision table of allow-lists, deny rules,
self-modify globs, redact fields, and per-argument predicates), `Adjudicate` returns exactly
one `abi.Verdict`: a provable refusal `Deny` with a structured `ReasonCode` and bounded-disclosure
witness, a `Transform` that rewrites args before dispatch, or an affirmative `Allow`. It does
**not** itself fold a chain — the kernel does. For a decision procedure, "math correct" (regime
**D**, decision-procedure soundness) means two things: the composition that resolves a chain of
such verdicts orders them by the **restrictiveness lattice** (`abi.FoldRank`) so the most-restrictive
verdict always wins, and the procedure is **fail-closed** — the zero/absent value (empty policy,
unmatched tool, empty or all-`Defer` chain, unknown verdict kind) resolves to `Deny`, never `Allow`;
and the monitor **mediates every decision** — it is wired into the single kernel fold path and no
internal code path emits `Allow` without first clearing every provable-refusal check.

Honesty note carried through both blocks below: the FoldRank *ordering* and the chain dispatch
live in `internal/kernel` (`Fold`) and `internal/abi` (`FoldRank`), not inside the adjudicator,
which emits one verdict per call. Those neighbour witnesses are real and ran green here; the
fail-closed half and the in-monitor completeness are witnessed in-scope.

---

## THEOREM 1 — FoldRank-ordered, fail-closed composition

**THEOREM.** Verdicts compose in `abi.FoldRank` order and the composition is fail-closed: the
zero/absent value (empty `Policy`, unmatched tool, empty or all-`Defer` chain) resolves to `Deny`
(`DEFAULT_DENY`), never `Allow`; `Deny` is the most-restrictive lattice element (rank 100) so it
cannot be outranked by any `Allow`/`Defer`/`Transform`/`Quarantine`/`RequireWitness`, and an
unknown verdict kind falls back to `Deny`.

**REGIME.** D — decision-procedure soundness (fail-closed + monotone-fold order).

**PROOF.**
*Fail-closed (in-scope).* The zero `Policy` is the empty decision table; `Adjudicate` falls through
every check to `defaultDeny` (`fak/internal/adjudicator/decide.go:275,278`), which returns
`Verdict{Kind: VerdictDeny, Reason: ReasonDefaultDeny}` (`decide.go:289`) unless the admit-and-log
posture downgrades a *low-risk read* — write-shaped, explicit-deny, self-modify and arg-violation
calls still fail closed. Arg predicates are **RESTRICT-ONLY** (`decide.go:248`; design comment
`decide.go:56`): a satisfied predicate never grants an `Allow`, so a tool nothing else allowed still
falls to `DEFAULT_DENY`. The chain-level fail-closed is `kernel.Fold`
(`fak/internal/kernel/kernel.go:129`): an empty chain → `Deny/DEFAULT_DENY` and an all-`Defer` chain
→ `Deny/DEFAULT_DENY` (`kernel.go:142` takes the argmax over `FoldRank`).
*FoldRank order.* `abi.FoldRank` (`fak/internal/abi/registry.go:744`) pins `VerdictDeny = 100` (most
restrictive of the core set), `VerdictAllow = 0`, and any unknown/registered kind defaults to 100
(fail-closed). The fold is a `max` over the lattice, so the result is order-independent.
`architest.TestFoldSitesOrderByFoldRank` (`fak/internal/architest/architest_test.go:770`)
machine-checks that *every* fold site keeps consulting `abi.FoldRank` rather than a hand-rolled
`Kind` comparison.

**WITNESS.**
```
go test ./internal/adjudicator/ -count=1 -run 'TestEmptyPolicyDefaultDeny|TestDefaultPolicyUnknownToolDefaultDeny|TestArgPredicatesAreRestrictOnly' -v
go test ./internal/abi/         -count=1 -run 'TestFoldRankOrdering' -v
go test ./internal/kernel/      -count=1 -run 'TestFoldDefaultDenyEmptyPolicy|TestFoldMostRestrictiveWins' -v
go test ./internal/architest/   -count=1 -run 'TestFoldSitesOrderByFoldRank' -v
```

**VERDICT.** PROVEN (2026-06-20, native go1.26 darwin/arm64). All seven witnesses ran green:
`TestEmptyPolicyDefaultDeny`, `TestDefaultPolicyUnknownToolDefaultDeny`,
`TestArgPredicatesAreRestrictOnly` (ok 0.186s); `TestFoldRankOrdering` (ok 0.168s, asserts
`FoldRank(Deny) > FoldRank(Quarantine)`, `FoldRank(Allow) == 0`, `Fallback(9999) == FallbackDeny`);
`TestFoldDefaultDenyEmptyPolicy` + `TestFoldMostRestrictiveWins` (ok 0.256s); `TestFoldSitesOrderByFoldRank`
(ok 0.235s). Honest split recorded: the fail-closed half is in-scope (`decide.go` `defaultDeny`); the
FoldRank *ordering* half is `kernel.Fold` / `abi.FoldRank` — the adjudicator emits one verdict, it
does not fold.

**DOS.** bound at ship (mechanism shipped in `f23f7cb`/`ff2dda6`/`811beea` adjudicator + `c59bb28`
architest fold-order gate; `dos commit-audit` / `dos verify` to bind at release).

---

## THEOREM 2 — the reference monitor mediates every decision

**THEOREM.** The adjudicator is a reference monitor that mediates every tool-call decision: it
self-registers into the defconfig adjudicator chain (rank 100) so it is present in the single
`kernel.Fold` path, no request-path leaf that self-registers can silently fail to load, and within
`Adjudicate` no return path yields `Allow` without first clearing the explicit-deny, self-modify
(path and shell), and arg-predicate checks — the terminal fall-through is `defaultDeny`.

**REGIME.** D — decision-procedure soundness (complete mediation / no-bypass).

**PROOF.**
`init()` calls `abi.RegisterAdjudicator(100, Default)` (`fak/internal/adjudicator/decide.go:590`,
registration at `:593`); the kernel decides *every* tool call via `Decide → Fold` over
`abi.AdjudicatorsFor(c)` (`fak/internal/kernel/kernel.go:120,129`) — the one mediated path.
`architest.TestRequestPathLeavesRegistered` (`fak/internal/architest/architest_test.go:272`)
machine-checks that a self-registering leaf is blank-imported into `internal/registrations` (the
defconfig) or on `regOffList`, so the monitor's `init()` is not dead code — i.e. the monitor is
actually *in* the chain the kernel folds, not a registration that silently never loads.
In-`Adjudicate` completeness: `Adjudicate` (`decide.go:197`) is total and every early return is a
`Deny`/`Transform`; the only `Allow` returns are gated *after* the explicit-deny (`:204`), path
self-modify (`:214`), shell self-modify (`:234`), and arg-predicate (`:248`) checks, and a
non-allowed tool reaches `defaultDeny` (`:275`). `matchGlob` / `commandSelfModify` (`decide.go:351,388`)
close the shell-launder hole (#172 Hole 1) so a `Bash` call carrying its write target in the command
string cannot bypass the file-write guard. Thus no decision escapes mediation: every call either
matches an affirmative allow (after passing all refusal checks) or fails closed.

**WITNESS.**
```
go test ./internal/architest/   -count=1 -run 'TestRequestPathLeavesRegistered' -v
go test ./internal/adjudicator/ -count=1 -run 'TestDefaultAllowsAllowedTool|TestSelfModifyDeniedWithBoundedWitness|TestSelfModifyGuardsWitnessMachinery|TestSelfModifyGuardsShellWritePath|TestAdmitAndLogPostureAllowsOnlyReadShapedDefaultDeny|TestReasonsAreInClosedVocab' -v
```

**VERDICT.** PROVEN (2026-06-20, native go1.26 darwin/arm64). `TestRequestPathLeavesRegistered`
PASS (ok 0.235s) gates that the monitor is wired into the defconfig; the adjudicator branch-coverage
witnesses all PASS (ok 0.186s): allowed tool → `Allow`; path self-modify → `Deny/SELF_MODIFY`
(`TestSelfModifyDeniedWithBoundedWitness`, `TestSelfModifyGuardsWitnessMachinery`); shell self-modify
→ `Deny/SELF_MODIFY` with reads / outside-tree writes NOT denied (`TestSelfModifyGuardsShellWritePath`);
admit-and-log downgrades only low-risk reads and still Denies write/explicit/self-modify
(`TestAdmitAndLogPostureAllowsOnlyReadShapedDefaultDeny`); every emitted reason is in the closed vocab
(`TestReasonsAreInClosedVocab`). Honest caveat: the FoldRank *order* and chain dispatch happen in
`kernel`/`abi`, not inside the adjudicator; the universal "no `Adjudicate` path bypasses" quantifier is
carried by branch coverage plus the registration architecture invariant, not a single exhaustive
in-scope fuzz — a `testing/quick` generator over arbitrary `(Policy, ToolCall)` asserting
`Allow ⇒ (¬Deny ∧ ¬SelfModify ∧ ¬argViolation)` would tighten it from PROVEN-by-coverage to
PROVEN-by-property.

**DOS.** bound at ship (`c59bb28` architest fold-order gate + adjudicator #172 self-modify hardening;
`dos commit-audit` / `dos verify` to bind at release).
