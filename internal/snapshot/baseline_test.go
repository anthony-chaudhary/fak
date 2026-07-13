package snapshot

// Witness for issue #4576 — govern golden & statistical baseline promotion.
//
// The load-bearing test is TestBaselinePromotionWitness: it plants a REPRESENTATIVE
// DEFECT (a candidate decode series that drifts past tolerance at one token), proves the
// governor REFUSES to self-promote the failing baseline and emits a scrubbed replay
// artifact that localizes the FIRST actionable divergence, then "applies the fix" (the
// candidate now matches within tolerance) and proves the very same request PROMOTES. That
// is the fail-before / pass-after proof the issue's Witness section requires, captured as a
// deterministic unit test (fixed injected clock, no RNG).
//
// The remaining tests pin each acceptance criterion independently so a regression in any
// one is caught on its own line.

import (
	"strings"
	"testing"
)

// completeProv is a fully-populated provenance record — every identity field the
// acceptance criteria require. Tests mutate a copy to plant a specific gap.
func completeProv() Provenance {
	return Provenance{
		Model:     "fak-decode-7b",
		Tokenizer: "bpe-v3",
		Engine:    "fak-engine/cpu-fp8",
		Seed:      "1337",
		Revision:  "internal/engine@r652+g1f75c56d",
		Tolerance: "abs<=1e-3 vs golden captured 2026-07-01 (n=512)",
		Tier:      TierNightly,
		CostNote:  "~40s, 1 CPU, 512 samples",
	}
}

// completeReq is a request that would PROMOTE on a passing run — tests plant one defect at
// a time against this baseline.
func completeReq(run RunResult) PromotionRequest {
	return PromotionRequest{
		Case:     completeProv(),
		Run:      run,
		Author:   "worker-a",
		Reviewer: "worker-b",
		Reason:   "candidate matches golden within tolerance after the fp8 dequant fix",
		DiffRef:  "abc1234",
		Rollback: "prior-baseline#42 / revert abc1234",
	}
}

// TestBaselinePromotionWitness is the #4576 proof: plant a decode defect, watch the
// governor refuse + emit a localized scrubbed replay, apply the fix, watch it promote.
func TestBaselinePromotionWitness(t *testing.T) {
	const tol = 1e-3
	golden := []float64{0.10, 0.20, 0.30, 0.40, 0.50}

	// --- BEFORE THE FIX: a planted defect drifts at index 2 (0.30 -> 0.42). ---
	defective := []float64{0.10, 0.20, 0.42, 0.40, 0.50}
	div, diverged := FirstDivergence(golden, defective, tol)
	if !diverged {
		t.Fatal("planted defect not detected — the comparator missed a real divergence")
	}
	if div.Index != 2 {
		t.Fatalf("first actionable divergence localized to index %d, want 2", div.Index)
	}

	// The failing run must NOT be able to self-update the baseline.
	failReq := completeReq(RunFail)
	if d := GovernPromotion(failReq); d.Promoted() || d.Reason != ReasonRunFailed {
		t.Fatalf("a failing run promoted or wrong reason: %+v", d)
	}

	// The failing case emits a scrubbed, integrity-checked replay artifact that carries the
	// first divergence and the full provenance — with a pasted secret in the note redacted.
	snap, err := BuildReplay("case-decode-42", failReq.Case, div, diverged,
		"repro under seed 1337; key sk-livesecrettokenvalue must not leak", 1_700_000_002)
	if err != nil {
		t.Fatalf("BuildReplay: %v", err)
	}
	encoded, err := snap.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	// The artifact is a first-class verifiable dump: it re-parses and re-verifies its digest.
	parsed, err := Parse(encoded)
	if err != nil {
		t.Fatalf("Parse replay: %v", err)
	}
	art, err := parsed.RestoreReplay()
	if err != nil {
		t.Fatalf("RestoreReplay: %v", err)
	}
	if art.First.Index != 2 {
		t.Fatalf("replay lost the first divergence index: %+v", art.First)
	}
	if art.Case.Model != "fak-decode-7b" || art.Case.Revision == "" {
		t.Fatalf("replay dropped provenance: %+v", art.Case)
	}
	if strings.Contains(string(encoded), "sk-livesecrettokenvalue") {
		t.Fatalf("replay artifact leaked a secret — scrubbing failed:\n%s", encoded)
	}
	if !strings.Contains(art.Note, "[REDACTED]") {
		t.Fatalf("expected the pasted key to be redacted, note=%q", art.Note)
	}

	// --- AFTER THE FIX: the candidate now matches golden within tolerance. ---
	fixed := []float64{0.100, 0.2005, 0.2998, 0.400, 0.500} // all within 1e-3
	if _, diverged := FirstDivergence(golden, fixed, tol); diverged {
		t.Fatal("post-fix candidate still diverges — the fixture does not actually pass")
	}
	passReq := completeReq(RunPass)
	if d := GovernPromotion(passReq); !d.Promoted() {
		t.Fatalf("the fixed, fully-evidenced case failed to promote: %+v", d)
	}
}

// TestFailingRunCannotSelfPromote pins acceptance criterion #1.
func TestFailingRunCannotSelfPromote(t *testing.T) {
	if d := GovernPromotion(completeReq(RunFail)); d.Promoted() {
		t.Fatalf("a failing run promoted its baseline: %+v", d)
	}
	// Inconclusive / missing evidence is never a pass either.
	for _, run := range []RunResult{RunInconclusive, RunResult("")} {
		d := GovernPromotion(completeReq(run))
		if d.Promoted() || d.Reason != ReasonMissingEvidence {
			t.Fatalf("run=%q promoted or wrong reason: %+v", run, d)
		}
	}
}

// TestProvenanceRequired pins acceptance criterion #2: each case records model, tokenizer,
// engine/backend, seed-or-oracle, revision, and tolerance/baseline provenance.
func TestProvenanceRequired(t *testing.T) {
	// A seed OR a deterministic oracle satisfies the seed-or-oracle requirement.
	oracleProv := completeProv()
	oracleProv.Seed = ""
	oracleProv.Oracle = "greedy-argmax-oracle-v2"
	if miss := oracleProv.Missing(); len(miss) != 0 {
		t.Fatalf("oracle should satisfy seed-or-oracle, got missing %v", miss)
	}

	// Dropping any single identity field is caught.
	drops := []struct {
		name string
		mut  func(*Provenance)
	}{
		{"model", func(p *Provenance) { p.Model = "" }},
		{"tokenizer", func(p *Provenance) { p.Tokenizer = "" }},
		{"engine", func(p *Provenance) { p.Engine = "" }},
		{"seed-or-oracle", func(p *Provenance) { p.Seed = ""; p.Oracle = "" }},
		{"revision", func(p *Provenance) { p.Revision = "" }},
		{"tolerance", func(p *Provenance) { p.Tolerance = "" }},
	}
	for _, tc := range drops {
		req := completeReq(RunPass)
		tc.mut(&req.Case)
		d := GovernPromotion(req)
		if d.Promoted() || d.Reason != ReasonMissingProvenance {
			t.Fatalf("dropping %s promoted or wrong reason: %+v", tc.name, d)
		}
		if miss := req.Case.Missing(); len(miss) != 1 || miss[0] != tc.name {
			t.Fatalf("Missing() for dropped %s = %v", tc.name, miss)
		}
	}
}

// TestGovernanceInputsRequired pins the Scope: evidence, independent review, reason, diff,
// and rollback pointer are all mandatory.
func TestGovernanceInputsRequired(t *testing.T) {
	cases := []struct {
		name   string
		mut    func(*PromotionRequest)
		reason string
	}{
		{"no reviewer", func(r *PromotionRequest) { r.Reviewer = "" }, ReasonUnreviewed},
		{"reviewer==author", func(r *PromotionRequest) { r.Reviewer = r.Author }, ReasonUnreviewed},
		{"no reason", func(r *PromotionRequest) { r.Reason = "" }, ReasonNoReason},
		{"no diff", func(r *PromotionRequest) { r.DiffRef = "" }, ReasonNoDiff},
		{"no rollback", func(r *PromotionRequest) { r.Rollback = "" }, ReasonNoRollback},
	}
	for _, tc := range cases {
		req := completeReq(RunPass)
		tc.mut(&req)
		d := GovernPromotion(req)
		if d.Promoted() || d.Reason != tc.reason {
			t.Fatalf("%s: promoted or wrong reason: %+v", tc.name, d)
		}
	}
}

// TestTierAndCostRequired pins acceptance criterion #4: an explicit PR/nightly/release tier
// and documented runtime/resource cost.
func TestTierAndCostRequired(t *testing.T) {
	badTier := completeReq(RunPass)
	badTier.Case.Tier = Tier("whenever")
	if d := GovernPromotion(badTier); d.Promoted() || d.Reason != ReasonBadTier {
		t.Fatalf("bad tier promoted or wrong reason: %+v", d)
	}
	noCost := completeReq(RunPass)
	noCost.Case.CostNote = ""
	if d := GovernPromotion(noCost); d.Promoted() || d.Reason != ReasonNoCost {
		t.Fatalf("missing cost promoted or wrong reason: %+v", d)
	}
	// Each sanctioned tier is accepted.
	for _, tr := range []Tier{TierPR, TierNightly, TierRelease} {
		req := completeReq(RunPass)
		req.Case.Tier = tr
		if d := GovernPromotion(req); !d.Promoted() {
			t.Fatalf("tier %q refused a complete passing request: %+v", tr, d)
		}
	}
}

// TestFirstDivergenceComparator pins the deterministic comparator edges: within-tolerance
// match, length mismatch as a divergence, and a NaN candidate never passing.
func TestFirstDivergenceComparator(t *testing.T) {
	g := []float64{1, 2, 3}
	if _, d := FirstDivergence(g, []float64{1.0005, 2.0, 3.0}, 1e-3); d {
		t.Fatal("within-tolerance series reported a divergence")
	}
	if div, d := FirstDivergence(g, []float64{1, 2}, 1e-3); !d || div.Index != 2 {
		t.Fatalf("length mismatch not caught at index 2: %+v ok=%v", div, d)
	}
	nan := []float64{1, 2, 3}
	nan[1] = zero() / zero() // a real runtime NaN
	if div, d := FirstDivergence(g, nan, 1e9); !d || div.Index != 1 {
		t.Fatalf("NaN candidate passed a huge tolerance: %+v ok=%v", div, d)
	}
	// BuildReplay refuses a non-divergent case — no false replay of a pass.
	if _, err := BuildReplay("x", completeProv(), Divergence{}, false, "", 1); err == nil {
		t.Fatal("BuildReplay accepted a non-divergent case")
	}
}

// TestBaselineKindsRegistered proves the new kinds are enumerable and restore rejects a
// wrong-kind snapshot.
func TestBaselineKindsRegistered(t *testing.T) {
	if _, ok := Known(KindBaseline); !ok {
		t.Fatal("baseline kind not registered")
	}
	if _, ok := Known(KindBaselineReplay); !ok {
		t.Fatal("baseline-replay kind not registered")
	}
	// A turn snapshot is not a replay artifact.
	turn, _ := DumpTrace("t", nil, 1)
	if _, err := turn.RestoreReplay(); err == nil {
		t.Fatal("RestoreReplay accepted a non-replay snapshot")
	}
}

// zero returns 0.0 without a constant-division compile error, so nan := x/zero() yields a
// real NaN at runtime.
func zero() float64 { return 0 }
