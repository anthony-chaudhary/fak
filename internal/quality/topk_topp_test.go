package quality

import (
	"math"
	"strings"
	"testing"
)

// topkpWeightedCase is the hermetic boundary fixture (#4526): a four-token
// vocabulary whose logits are ln(4), ln(3), ln(2), ln(1), so the softmax
// probabilities are exactly the clean ladder 0.4, 0.3, 0.2, 0.1 (up to float
// rounding well inside the oracle's boundary epsilon). Cumulative masses land
// on 0.4 / 0.7 / 0.9 / 1.0 — with 0.7 the exact p boundary after "beta".
func topkpWeightedCase(id string, topK int, topP float64) QualityCase {
	return TopKTopPCase(id,
		[]string{"alpha", "beta", "gamma", "delta"},
		[]float64{math.Log(4), math.Log(3), math.Log(2), math.Log(1)},
		topK, topP)
}

// topkpTieCase puts an exact probability TIE straddling the k-th slot: "beta"
// and "gamma" carry the identical logit ln(2), so with k=2 the second slot is
// contested and the documented tie-break (ascending vocabulary index) must
// award it to "beta".
func topkpTieCase(id string, topK int) QualityCase {
	return TopKTopPCase(id,
		[]string{"alpha", "beta", "gamma", "delta"},
		[]float64{math.Log(4), math.Log(2), math.Log(2), math.Log(1)},
		topK, 0)
}

// topkpEngine replays a fixed kept candidate set as the engine trace — the
// deterministic mutant source these tests use: hand it the correct set for a
// faithful engine, or a wrongly truncated one for each injected defect.
func topkpEngine(label string, kept []string) ScriptedRunner {
	return ScriptedRunner{
		Label: label,
		Trace: Trace{Tokens: append([]string(nil), kept...), Text: strings.Join(kept, " ")},
	}
}

// TestTopKTopPFaithfulPasses proves an engine that computes the truncation
// correctly passes at every declared boundary: exact k, k=0 (filter unset),
// k >= vocab, p=0 (filter unset), p=1 (whole set), a tie straddling the k-th
// slot, and a token landing exactly on the p boundary.
func TestTopKTopPFaithfulPasses(t *testing.T) {
	all := []string{"alpha", "beta", "gamma", "delta"}
	tests := []struct {
		name string
		c    QualityCase
		kept []string
	}{
		{"exact-k", topkpWeightedCase("topkp-exact-k", 2, 0), []string{"alpha", "beta"}},
		{"k-zero-unset", topkpWeightedCase("topkp-k-zero", 0, 0), all},
		{"k-ge-vocab", topkpWeightedCase("topkp-k-ge-vocab", 99, 0), all},
		{"p-zero-unset", topkpWeightedCase("topkp-p-zero", 0, 0), all},
		{"p-one-whole-set", topkpWeightedCase("topkp-p-one", 0, 1), all},
		{"p-exact-boundary", topkpWeightedCase("topkp-p-boundary", 0, 0.7), []string{"alpha", "beta"}},
		{"tie-at-kth-slot", topkpTieCase("topkp-tie", 2), []string{"alpha", "beta"}},
		{"k-then-p-renormalized", topkpWeightedCase("topkp-k-then-p", 3, 0.7), []string{"alpha", "beta"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := RunCase(tc.c, ReferenceRunner{}, topkpEngine("engine-faithful", tc.kept), oraclesFor(t, tc.c))
			if err != nil {
				t.Fatalf("RunCase: %v", err)
			}
			if !res.Pass {
				t.Fatalf("faithful truncation must pass; got %s", Explain(res))
			}
			if res.FailureBundle != nil {
				t.Fatalf("clean pass must not carry a failure bundle: %+v", res.FailureBundle)
			}
		})
	}
}

// topkpMustFail runs one defect-injected engine and returns the failure bundle,
// asserting the run failed under this oracle with localizing evidence attached.
func topkpMustFail(t *testing.T, c QualityCase, eng ScriptedRunner) *FailureBundle {
	t.Helper()
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("defect-injected truncation must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "topk-topp-boundary" || fb.FailingKind != "differential" {
		t.Fatalf("failing oracle = %q (%s), want topk-topp-boundary (differential)", fb.FailingOracle, fb.FailingKind)
	}
	return fb
}

// TestTopKKeepsExtraFails is the k+1 off-by-one gate: with k=2 an engine that
// kept a third candidate fails with the divergence pinned at slot 2 — the first
// slot that should already have been empty.
func TestTopKKeepsExtraFails(t *testing.T) {
	c := topkpWeightedCase("topkp-k-plus-one", 2, 0)
	fb := topkpMustFail(t, c, topkpEngine("engine-k-plus-one", []string{"alpha", "beta", "gamma"}))
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 2 {
		t.Fatalf("expected divergence at slot 2 (first over-kept candidate), got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Engine != "gamma" || fb.FirstDivergence.Reference != "<end>" {
		t.Errorf("divergence = %+v, want reference <end>, engine gamma", fb.FirstDivergence)
	}
}

// TestTopKKeepsFewerFails is the k-1 off-by-one gate: with k=2 an engine that
// kept only one candidate fails at slot 1, naming the reference token it dropped.
func TestTopKKeepsFewerFails(t *testing.T) {
	c := topkpWeightedCase("topkp-k-minus-one", 2, 0)
	fb := topkpMustFail(t, c, topkpEngine("engine-k-minus-one", []string{"alpha"}))
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 1 {
		t.Fatalf("expected divergence at slot 1 (first dropped candidate), got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Reference != "beta" || fb.FirstDivergence.Engine != "<end>" {
		t.Errorf("divergence = %+v, want reference beta, engine <end>", fb.FirstDivergence)
	}
}

// TestTopKTieBreakWrongFails is the tie-break gate: "beta" and "gamma" tie for
// the k-th slot and the documented rule (ascending vocabulary index) awards it
// to "beta"; an engine that resolved the tie the other way fails at that slot
// with both contenders named.
func TestTopKTieBreakWrongFails(t *testing.T) {
	c := topkpTieCase("topkp-tie-wrong", 2)
	fb := topkpMustFail(t, c, topkpEngine("engine-tie-wrong", []string{"alpha", "gamma"}))
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 1 {
		t.Fatalf("expected divergence at the contested slot 1, got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Reference != "beta" || fb.FirstDivergence.Engine != "gamma" {
		t.Errorf("divergence = %+v, want reference beta, engine gamma", fb.FirstDivergence)
	}
	if !strings.Contains(fb.Detail, "beta") || !strings.Contains(fb.Detail, "gamma") {
		t.Errorf("detail must name both tie contenders; got %q", fb.Detail)
	}
}

// TestTopPBoundaryDropFails is the inclusive-boundary gate, exclusion side:
// with p=0.7 the cumulative mass lands exactly on p at "beta", so "beta" is IN
// the set; an engine that dropped the exactly-on-boundary token fails at slot 1.
func TestTopPBoundaryDropFails(t *testing.T) {
	c := topkpWeightedCase("topkp-p-drop", 0, 0.7)
	fb := topkpMustFail(t, c, topkpEngine("engine-p-drop", []string{"alpha"}))
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 1 {
		t.Fatalf("expected divergence at slot 1 (dropped boundary token), got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Reference != "beta" {
		t.Errorf("divergence reference = %q, want beta (the exactly-on-boundary token)", fb.FirstDivergence.Reference)
	}
}

// TestTopPBoundaryOvershootFails is the inclusive-boundary gate, overshoot
// side: an engine that treated the boundary comparison as strictly-greater
// keeps one candidate past the nucleus; it fails at slot 2, the first token
// beyond the exact p boundary.
func TestTopPBoundaryOvershootFails(t *testing.T) {
	c := topkpWeightedCase("topkp-p-overshoot", 0, 0.7)
	fb := topkpMustFail(t, c, topkpEngine("engine-p-overshoot", []string{"alpha", "beta", "gamma"}))
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 2 {
		t.Fatalf("expected divergence at slot 2 (first token past the nucleus), got %+v", fb.FirstDivergence)
	}
	if fb.FirstDivergence.Engine != "gamma" || fb.FirstDivergence.Reference != "<end>" {
		t.Errorf("divergence = %+v, want reference <end>, engine gamma", fb.FirstDivergence)
	}
}

// TestTopKTopPMalformedReferenceFailsClosed proves the admission posture: a
// case whose reference carries no aligned logits row cannot be judged, and the
// oracle fails closed with the malformation named rather than passing green.
func TestTopKTopPMalformedReferenceFailsClosed(t *testing.T) {
	c := topkpWeightedCase("topkp-malformed", 2, 0)
	c.Reference.Logits = nil
	res, err := RunCase(c, ReferenceRunner{}, topkpEngine("engine-any", []string{"alpha", "beta"}), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("malformed reference must fail closed; got %s", Explain(res))
	}
	if fb := res.FailureBundle; fb == nil || !strings.Contains(fb.Detail, "malformed") {
		t.Fatalf("failure bundle must name the malformation; got %+v", res.FailureBundle)
	}
}
