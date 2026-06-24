// Package advmodel is the advisory adjudication model — the consumer of the
// harvest LabelRow corpus and the closing edge of the kernel's self-improvement
// loop (issue #580).
//
// THE MISSING EDGE. internal/harvest folds every adjudication into a frozen
// abi.LabelRow — the kernel harvests its own verdicts — but nothing CONSUMED
// that corpus. This package is the consumer: a small classifier trained over
// the harvested (call, verdict) stream that emits a learned adjudication signal.
// It is NOT a fine-tune of the fused SmolLM2 forward pass (that needs GPU +
// weights + hours, out of scope here); it is a small linear classifier over
// call-structure features — the "small syscall/adjudication model" the issue's
// acceptance explicitly permits ("Train (or fine-tune) a small … model").
//
// FAIL-CLOSED BY CONSTRUCTION. The model is an opt-in abi.Adjudicator that may
// return ONLY VerdictDeny (corroborate) or VerdictDefer (no opinion). It NEVER
// returns VerdictAllow, so under the kernel's restrictiveness fold
// (kernel.Fold: the most-restrictive non-Defer verdict wins) it can only ever
// TIGHTEN the decision — add a deny — never weaken the deterministic capability
// floor, whose own denies always stand. An empty/inert artifact scores below
// threshold on every call and defers on everything, so a mis-loaded or
// untrained model is a no-op, never an authority-widening hole.
//
// DEFAULT-OFF. The package never self-registers (no init). An operator wires it
// by loading a trained artifact (Load) and handing the resulting Adjudicator to
// the kernel's explicit per-kernel chain (kernel.WithAdjudicators) or to
// abi.RegisterAdjudicator. The frozen ABI is untouched (additive-only); the
// in-kernel bit-exactness / golden freeze is unaffected — this leaf reads
// abi.ToolCall and returns abi.Verdict, exactly the seam every other adjudicator
// uses.
//
// THE TRAINING RUN (reproducible). train.py (in this package) reads the frozen
// content-bearing harvest corpus (testdata/corpus.jsonl — every label
// re-witnessed against the REAL adjudicator floor by corpus_test.go, and
// harvest-compatible by construction), featurizes each call (tool + args token
// bag), trains a logistic-regression classifier on a deterministic train split,
// writes the model artifact (testdata/adjudicator.json), and prints held-out
// precision/recall vs the stock reference (the untrained artifact = always
// defers = predicts negative). The committed artifact is what Load reads.
//
// Tier: mechanism (2) — see internal/architest. May import only packages whose
// tier is <= 2 (abi, adjudicator, harvest are all tier 2 or below).
package advmodel
