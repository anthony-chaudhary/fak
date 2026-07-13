package quality

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// TestLogitToleranceInBandNoisePasses is the happy path of the declared-band
// contract: an engine whose step-0 logits carry real, nonzero cross-backend
// noise WITHIN the declared tolerance passes — exact equality is explicitly not
// the gate.
func TestLogitToleranceInBandNoisePasses(t *testing.T) {
	const tol = 1e-3
	c := ltolCase(tol)
	res, err := RunCase(c, ReferenceRunner{}, ltolEngine("", tol), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("in-band noise must pass the declared tolerance; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean run must not carry a failure bundle: %+v", res.FailureBundle)
	}
	// Non-tautological: the engine logits genuinely differ from the reference —
	// the pass is the tolerance doing work, not two identical vectors.
	ref, eng := res.Provenance.Reference.Logits[0], res.Provenance.Engine.Logits[0]
	maxDiff := 0.0
	for i := range ref {
		if d := math.Abs(ref[i] - eng[i]); d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff == 0 {
		t.Fatal("clean engine logits must differ from the reference (in-band noise), not equal it exactly")
	}
	if maxDiff > tol {
		t.Fatalf("clean engine noise %.6g exceeds the declared tolerance %.6g; test fixture is broken", maxDiff, tol)
	}
}

// TestLogitToleranceDriftFailsAtIndex is the localized-defect witness: a
// backend whose logits drift beyond the declared tolerance at one index fails,
// the first divergence pins exactly that logit index with both values, and the
// detail names the index and the measured difference.
func TestLogitToleranceDriftFailsAtIndex(t *testing.T) {
	const tol = 1e-3
	c := ltolCase(tol)
	res, err := RunCase(c, ReferenceRunner{}, ltolEngine("drift", tol), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("out-of-band drift must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != ltolOracleName {
		t.Errorf("first failing oracle = %q, want %s", fb.FailingOracle, ltolOracleName)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != ltolDriftIndex {
		t.Fatalf("expected first divergence at logit %d, got %+v", ltolDriftIndex, d)
	}
	refVal := fb.Reference.Logits[0][ltolDriftIndex]
	engVal := fb.Engine.Logits[0][ltolDriftIndex]
	if d.Reference != ltolFmt(refVal) || d.Engine != ltolFmt(engVal) {
		t.Errorf("divergence values = ref %q eng %q, want ref %q eng %q",
			d.Reference, d.Engine, ltolFmt(refVal), ltolFmt(engVal))
	}
	// The detail names the offending index and the measured difference.
	if want := fmt.Sprintf("logit %d", ltolDriftIndex); !strings.Contains(fb.Detail, want) {
		t.Errorf("detail %q does not name the drifted index %q", fb.Detail, want)
	}
	measured := math.Abs(refVal - engVal)
	if measured <= tol {
		t.Fatalf("drift fixture is broken: measured diff %.6g is within tol %.6g", measured, tol)
	}
	if want := fmt.Sprintf("%.6g", measured); !strings.Contains(fb.Detail, want) {
		t.Errorf("detail %q does not carry the measured diff %q", fb.Detail, want)
	}
}

// TestLogitToleranceMinScoreDeclaration covers the second declaration channel:
// a case with no "tol=" in its prompt reuses Rubric.MinScore as the band. The
// same engine pair flips verdicts purely on the declared band's authority.
func TestLogitToleranceMinScoreDeclaration(t *testing.T) {
	const tol = 1e-3
	c := ltolCase(tol)
	c.Prompt = "Compare backend step-0 logits under the rubric-declared band."
	c.Rubric.MinScore = tol

	v := ltolOracle{}.Judge(Trace{Logits: [][]float64{ltolReferenceLogits()}}, ltolEngine("", tol).Trace, c)
	if !v.Pass {
		t.Fatalf("in-band noise must pass under the rubric-declared band; got %s", v.Detail)
	}
	if !strings.Contains(v.Detail, "min_score") {
		t.Errorf("pass detail %q should name the min_score declaration source", v.Detail)
	}
	v = ltolOracle{}.Judge(Trace{Logits: [][]float64{ltolReferenceLogits()}}, ltolEngine("drift", tol).Trace, c)
	if v.Pass {
		t.Fatal("out-of-band drift must fail under the rubric-declared band")
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != ltolDriftIndex {
		t.Fatalf("expected divergence at logit %d, got %+v", ltolDriftIndex, v.FirstDivergence)
	}
	if v.Oracle != ltolOracleName || v.Kind != "differential" {
		t.Errorf("verdict identity = %q/%q, want %s/differential", v.Oracle, v.Kind, ltolOracleName)
	}
}

// TestLogitToleranceFailsClosed covers the degenerate traces: a side without
// step-0 logits cannot pass (the comparison never ran), and a length-mismatched
// vector fails at the first missing index.
func TestLogitToleranceFailsClosed(t *testing.T) {
	const tol = 1e-3
	c := ltolCase(tol)
	ref := Trace{Logits: [][]float64{ltolReferenceLogits()}}

	v := ltolOracle{}.Judge(ref, Trace{Text: "Throughput"}, c)
	if v.Pass {
		t.Fatal("an engine trace without logits must not pass logit parity")
	}
	if !strings.Contains(v.Detail, "logits") {
		t.Errorf("fail-closed detail %q should explain the missing logits", v.Detail)
	}

	short := ltolReferenceLogits()[:4]
	v = ltolOracle{}.Judge(ref, Trace{Logits: [][]float64{short}}, c)
	if v.Pass {
		t.Fatal("a length-mismatched logit vector must not pass")
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != len(short) {
		t.Fatalf("expected length divergence at index %d, got %+v", len(short), v.FirstDivergence)
	}
}
