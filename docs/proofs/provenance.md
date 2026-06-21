# C3 · provenance

`internal/provenance` is the single, kernel-authored answer to "where did this byte come from, and may the kernel trust it?". It maps a `(ToolCall, Result)` pair to a label on the frozen `abi.TaintLabel` lattice (`trusted < tainted < quarantined`) using **exactly two kernel-controlled facts**: the post-call, kernel-stamped `Result` envelope (`Payload.Taint`, and the detector-stamped `quarantine_id`), and the **host-registered** source class of the tool's channel (`SourceOf`, seeded by `init`/`RegisterSource`). It deliberately does **not** read `ToolCall.Meta` on any verdict path. "Correct" here is a **regime-C integrity** property, not a numerical one: an *adversary* — an injected/derailed model that fully controls its own `ToolCall` (Op, Tool, Args, and the open `Meta` map) — must be **unable to mint trust for itself**. Two obligations capture that: (1) a forged `Meta["provenance"]="trusted_local"` self-tag is ignored for the verdict (surfaced only forensically via `AttemptedSelfTrust`); and (2) this one kernel-authored definition is the *single source* the trust gates (`ifc`, `normgate`) consult, so there is no drifting second copy a forgery could slip past.

All witnesses below are pure, sub-second `stdlib testing` unit tests — no oracle cache, no weights, no RNG — and were run **green natively** on this macOS node (go1.26 darwin/arm64) on 2026-06-20.

---

## Theorem 1 — the model cannot self-mint trust

**THEOREM.** For every `ToolCall c` and `Result r`, a model-supplied `Meta["provenance"]="trusted_local"` tag does **not** raise the provenance verdict. `Taint(c,r)` and `Trusted(c,r)` are computed from the kernel-stamped `Result` envelope (`Payload.Taint` / `quarantine_id`) and the host-registered tool source class **only**, never from `c.Meta`. On an untrusted tool the forged tag yields `Tainted` (not `Trusted`); on a tool already trusted by source class the tag changes nothing; and the attempt is recoverable **only** as a forensic signal via `AttemptedSelfTrust`.

**REGIME.** C — crypto / integrity (unforgeability of a trust label against an adversary that controls the call envelope).

**PROOF.** `Taint` (`fak/internal/provenance/provenance.go:127-140`) consults, in order: `r.Payload.Taint` and `sealed(r)` (the `quarantine_id` Meta stamp written by the detector chain *after* the call, `provenance.go:162-164`); then, on a clean result, `SourceOf(c.Tool)` (`provenance.go:99-103`) — the host registry populated by `init` + `RegisterSource` (`provenance.go:75-95`), a Go API unreachable from a model-emitted call. `c.Meta` is **never referenced on a verdict path**. The historical self-tag key is read **only** by `AttemptedSelfTrust` (`provenance.go:155-157`), which returns a `bool` and never feeds `Taint`/`Trusted`. `Trusted` (`provenance.go:146-148`) is exactly `Taint == TaintTrusted`, so it inherits the same Meta-blindness. The witness exhibits the precise attack — a `read_webpage` call carrying `Meta["provenance"]="trusted_local"` must classify `Tainted` and must not be `Trusted`, while `AttemptedSelfTrust` surfaces it (`provenance_test.go:41-67`) — and also checks that the same forged tag on a genuinely `TrustedLocal` tool (`Read`) leaves the verdict driven by the source class.

**WITNESS.** `go test ./internal/provenance/ -run 'TestModelCannotAuthorTrust' -count=1 -timeout 120s -v` (corroborated by `TestTaintBySource`, `TestNilSafe`).

**VERDICT.** **PROVEN** — 2026-06-20. `TestModelCannotAuthorTrust` PASS (0.00s); `TestTaintBySource` PASS; `TestKernelStampedResultState` PASS; `TestNilSafe` PASS; package `ok 0.277s`. Witness body read-confirmed to assert the forged-tag-ignored + forensic-surfaced behaviour.

**DOS.** bound at ship.

---

## Theorem 2 — the kernel-authored label is the single source consulted by the gates

**THEOREM.** The kernel-authored provenance label is the **single** definition the trust gates consult: the `ifc` information-flow gate and the `normgate` quarantine gate both derive their trust label by calling into package `provenance`, and neither gate independently re-reads the model-forgeable `Meta["provenance"]` tag to make a verdict — so there is one provenance classifier, not drifting copies.

**REGIME.** C — crypto / integrity (no second, divergent trust authority for an adversary to target).

**PROOF.** The classifier is a single pure function `Taint` (`fak/internal/provenance/provenance.go:127`) with its boolean form `Trusted` (`provenance.go:146`); the package doc (`provenance.go:35-37`) states it is the ONE definition `ifc`/`normgate` consume "so the whole kernel agrees on provenance from one definition instead of three drifting copies." Code evidence of the delegation: `ifc`'s `SourceTaint` returns `provenance.Taint(c,r)` (`fak/internal/ifc/ifc.go:234`) and the ledger raise folds `provenance.Taint(ev.Call, ev.Result)` (`fak/internal/ifc/ifc.go:622`); `normgate`'s trust predicate returns `provenance.Trusted(c,r)` (`fak/internal/normgate/normgate.go:59`). A grep over both gate files for an *independent* verdict-path read of `Meta["provenance"]` / `"trusted_local"` finds **only comments** (`ifc.go:230`, `normgate.go:56`) documenting that the legacy self-tag is no longer honored — no gate reads the tag for a verdict. The provenance witnesses `TestTaintBySource` (8 tool→label cases), `TestKernelStampedResultState` (sealed/stamped precedence), and `TestRegisterSourceIsHostAuthored` (host-only authorship channel) pin the behaviour of the single definition that the gates inherit.

**WITNESS.** `go test ./internal/provenance/ -run 'TestTaintBySource|TestKernelStampedResultState|TestRegisterSourceIsHostAuthored' -count=1 -timeout 120s -v`.

**VERDICT.** **PROVEN** — 2026-06-20, with one honest caveat. The single classifier exists and both gates demonstrably call it (import-site evidence at `ifc.go:234`/`ifc.go:622`/`normgate.go:59`); the provenance-definition witnesses run green (`0.277s ok`). The **no-drift** half — "no gate ever re-derives a trust label from `Meta` on its own" — currently rests on **grep evidence**, not an executable in-scope assertion. An `architest`-level gate ("no verdict-path read of `Meta["provenance"]` outside package `provenance`") would lift that half from code-evidence to a re-run-on-every-build witness; until then it is grounded but not machine-asserted.

**DOS.** bound at ship.
