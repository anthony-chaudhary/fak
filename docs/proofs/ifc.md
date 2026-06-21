# C4 · ifc

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/ifc/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

The `ifc` package is fak's information-flow control layer — the CaMeL / FIDES complement to the lexical detectors (`canon`, `normgate`, `ctxmmu`). Every content detector is sound-but-evadable: it keys on the *text* of a tool result, and text can always be paraphrased. `ifc` instead keys on **provenance**, which a paraphrase cannot launder. It has two pure-consumer seams over the frozen `abi.Ref.Taint` lattice: a **source-stamp** data-plane gate (`StampGate`) that labels each result by its source and raises a per-trace control-flow taint **high-water mark** in a `Ledger`, and a **sink-gate** control-plane adjudicator (`SinkGate`) that refuses a sensitive sink (egress / exec / destructive) once tainted data is in flight, unless an explicit authorization declassifies the flow. "Correct" here is a **regime-C / crypto-integrity** property: the gate must resist an *adversary* (an injected prompt), not just average input — specifically (1) the taint label join must be a well-formed join-semilattice so the most-restrictive fold is well-defined, and (2) **non-interference** must hold: untrusted-derived data cannot reach a sink unless declassified, keyed on provenance not content.

---

## THEOREM 1 — the taint join is a join-semilattice (monotone · associative · commutative)

**THEOREM.** The `abi.Ref.Taint` join — implemented as a `taintRank`-max via `Ledger.Raise` — is a join-semilattice over the closed 3-element order `Trusted < Tainted < Quarantined`: monotone (a `Raise` never lowers a trace's mark; an unseen trace is the bottom/identity `Trusted`), commutative, associative, and idempotent.

**REGIME.** C (the restrictiveness lattice the most-restrictive fold and the high-water mark both depend on).

**PROOF.** The join is total-order `max`. `taintRank` (`fak/internal/ifc/ifc.go:65`) maps the deliberately-unordered enum to a real chain `Trusted=0 < Tainted=1 < Quarantined=2`; `Ledger.Raise` (`fak/internal/ifc/ifc.go:122`) reads `cur` (`Trusted` when the key is unseen — `ifc.go:126-129`, **not** the enum zero `Tainted`) and overwrites it with `t` only when `taintRank(t) > taintRank(cur)`. A `max` over a total order is, by construction, commutative, associative, idempotent, and monotone, with `Trusted` as bottom/identity; `fak/internal/abi/types.go:82` declares the lattice closed and additive. So the property is **true by construction**. But per `00-METHOD.md` a green package run does not *witness* a specific theorem — it must be *asserted* by a test. The only test exercising `Raise` semantics, `TestLedgerIsBoundedByLRUTraceMarks` (`ifc_test.go:143`), witnesses a single corner of **monotonicity** — re-raising an already-`Tainted` trace keeps it `Tainted` (`ifc_test.go:149-160`) and an unseen/reset trace reads `Trusted` (`ifc_test.go:147,172`) — but asserts **nothing** about commutativity, associativity, or the full join table over all pairs/triples. The algebra is therefore unwitnessed.

**WITNESS.** `(go test ./internal/ifc/ -count=1 -timeout 120s -run TestLedgerIsBoundedByLRUTraceMarks -v)`

**VERDICT.** **OPEN** — 2026-06-20. The property holds by construction (max over a total order) and monotone non-lowering is partially witnessed, but no deterministic test asserts commutativity / associativity / the complete join table. *Closed by:* a `testing`/`testing/quick` property test (`TestTaintJoinIsSemilattice`) iterating all ordered pairs and triples of `{Trusted, Tainted, Quarantined}`, driving `Raise` in both orders on a fresh `Ledger` and asserting `join(a,b)==join(b,a)`, `join(join(a,b),c)==join(a,join(b,c))`, `join(a,a)==a`, and `join(Trusted,a)==a` (identity).

**DOS.** bound at ship.

---

## THEOREM 2 — non-interference: provenance-keyed taint cannot be laundered into a sink

**THEOREM.** Tainted (untrusted-derived) data is barred from a sensitive sink (egress / exec / destructive) unless explicitly declassified. Taint is keyed on **provenance** (the kernel-derived source class), not on content — so neither a semantic paraphrase, nor a model-authored self-trust tag, nor a vDSO cache hit can launder it; a clean or authorized flow is *not* blocked (the precision floor).

**REGIME.** C (an adversary — an injected prompt — must not be able to exfiltrate; the guarantee is unforgeability of trust, not average-case behavior).

**PROOF.** Two seams enforce it. **Source-stamp:** `StampGate.Admit` (`fak/internal/ifc/ifc.go:487`) calls `SourceTaint` (`ifc.go:233`), which delegates to `internal/provenance.Taint` — deriving the label from the kernel-stamped result state and the host-registered tool source class **only**, ignoring the model-forgeable `ToolCall.Meta` entirely — and raises the trace high-water mark (`ifc.go:500`). **Sink-gate:** `SinkGate.Adjudicate` (`ifc.go:528`) classifies the call's sink sensitivity and, when the ledger flow (`ifc.go:544`) is `Dangerous` (rank ≥ Tainted — `ifc.go:79`), returns `VerdictDeny`/`ReasonTrustViolation` unless `Policy.Authorize` releases it (`ifc.go:554` — the explicit-declassification escape CaMeL requires). The `vdsoTaintEmitter` (`ifc.go:618`) closes the cache-bypass hole by raising the ledger from provenance on the `EvVDSOHit` event, which skips the `ResultAdmitter` chain. Three deterministic witnesses pin non-interference at exactly the laundering attempts: `TestParaphrasedExfilBlockedByProvenance` (`ifc_test.go:49`) reads an external page whose body is the marker-free paraphrase every lexical gate Defers on, asserts the session taints (`Level==Tainted`), and asserts the follow-up `send_email` is `Deny`/`ReasonTrustViolation` — content-blind, provenance-true. `TestForgedSelfTrustCannotEvadeTaint` (`ifc_test.go:83`) sets `Meta["provenance"]="trusted_local"` on the read and asserts the session **still** taints and the exfil is **still** denied — proving the key is provenance, not the model's self-tag. `TestVDSOHitDoesNotLaunderTaint` (`ifc_vdso_test.go:39`) drives the real `kernel`+vDSO path and asserts a cache-served untrusted read still taints and its egress is denied. `TestCleanSessionEgressAllowed` (`ifc_test.go:108`) and `TestAuthorizeEscape` (`ifc_test.go:244`) pin the precision floor and the declassification escape, so the `Deny` is a *gate*, not a blanket ban.

**WITNESS.** `(go test ./internal/ifc/ -count=1 -timeout 120s -run 'TestParaphrasedExfilBlockedByProvenance|TestForgedSelfTrustCannotEvadeTaint|TestVDSOHitDoesNotLaunderTaint|TestCleanSessionEgressAllowed|TestAuthorizeEscape' -v)`

**VERDICT.** **PROVEN** — 2026-06-20. All five witnesses ran green here (full package 17/17 PASS, `ok github.com/anthony-chaudhary/fak/internal/ifc 0.239s`). Reading the bodies confirms each asserts the exact non-interference claim (untrusted source → tainted session → sink DENIED; trusted/authorized → Defer), not merely that the code compiles.

**DOS.** bound at ship.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **taint-join-semilattice** → ✅ PROVEN by `TestTaintJoinLedgerRealizesSpec,TestTaintJoinIdentity,TestTaintJoinMonotone,TestTaintJoinIdempotent,TestTaintJoinCommutative,TestTaintJoinAssociative`. PROVEN by EXHAUSTIVE sweep over the closed 3-element carrier {Trusted,Tainted,Quarantined} — a complete proof, not a sample. (1) Bridge lemma TestTaintJoinLedgerRealizesSpec: the live Ledger.Raise realizes the reference join (max by taintRank, ifc.go:65) exactly over all 9 ordered pairs, so the algebraic laws bind to the real mechanism. (2) Identity: an unseen trace reads Trusted (ifc.go:147), guarded non-vacuous (Trusted is NOT the enum zero TaintTainted, abi/types.go:86), and a∨Trusted==a both orders. (3) Monotone: over all 9 (prior,b) pairs Raise never lowers the mark and the result is an upper bound of both operands. (4) Idempotent a∨a==a. (5) Commutative over all 9 pairs. (6) Associative (a∨b)∨c==a∨(b∨c) over all 27 triples — left assoc via sequential Raise on one trace, right assoc via a side-ledger precompute of (b∨c). Every test asserts a real metamorphic/algebraic equality with exact integer/enum comparison; no tolerance needed. Whole package go test green with the new file present.
