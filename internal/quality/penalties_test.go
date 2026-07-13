package quality

import (
	"strings"
	"testing"
)

// penaltyTestCase is the hermetic penalty-ordering fixture (#4527): a 5-token
// vocab with mixed-sign raw logits and a history where "the" was emitted twice,
// "cat" and "sat" once each, and "mat"/"dog" never (so they must ride through
// unpenalized). Strengths: repetition 1.5, presence 0.25, frequency 0.1.
func penaltyTestCase() QualityCase {
	return PenaltyCase(
		"penalty-ordering-rep-presence-freq",
		[]string{"the", "cat", "sat", "mat", "dog"},
		[]float64{2.0, 1.2, -0.8, 0.5, 3.0},
		map[string]int{"the": 2, "cat": 1, "sat": 1},
		1.5, 0.25, 0.1,
	)
}

// penaltyTestSpec re-parses the fixture's own prompt so the defect engines
// below penalize exactly the history/strengths the oracle will judge against.
func penaltyTestSpec(t *testing.T, c QualityCase) penSpec {
	t.Helper()
	s, err := penParseSpec(c.Prompt)
	if err != nil {
		t.Fatalf("penParseSpec(%q): %v", c.Prompt, err)
	}
	return s
}

// penaltyEngine wraps a penalized logits row as the engine trace for the
// fixture case.
func penaltyEngine(label string, logits []float64) ScriptedRunner {
	c := penaltyTestCase()
	return ScriptedRunner{
		Label: label,
		Trace: Trace{Tokens: c.Reference.Tokens, Logits: [][]float64{logits}},
	}
}

// TestPenaltyOrderingFaithfulPasses proves an engine that applies the
// documented pipeline — sign-aware repetition on the raw logit first, then the
// presence and frequency subtractions — passes with no failure bundle, and that
// tokens absent from the history are left untouched.
func TestPenaltyOrderingFaithfulPasses(t *testing.T) {
	c := penaltyTestCase()
	spec := penaltyTestSpec(t, c)
	faithful := penApply(c.Reference.Logits[0], c.Reference.Tokens, spec)

	// Unseen tokens ("mat", "dog") must ride through unpenalized.
	if faithful[3] != 0.5 || faithful[4] != 3.0 {
		t.Fatalf("unseen tokens must be unpenalized: got mat=%v dog=%v", faithful[3], faithful[4])
	}

	res, err := RunCase(c, ReferenceRunner{}, penaltyEngine("engine-clean", faithful), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful penalty pipeline should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean pass must not carry a failure bundle: %+v", res.FailureBundle)
	}
}

// TestPenaltyOrderingWrongOrderFails injects the ordering defect: frequency is
// subtracted BEFORE the sign-aware repetition scaling, so the scaling sees an
// already-shifted logit ((x - f*c)/r instead of x/r - f*c). The first penalized
// token, "the" at index 0, must be named as the first divergence.
func TestPenaltyOrderingWrongOrderFails(t *testing.T) {
	c := penaltyTestCase()
	spec := penaltyTestSpec(t, c)
	raw := c.Reference.Logits[0]
	vocab := c.Reference.Tokens

	wrong := make([]float64, len(raw))
	for i, x := range raw {
		count := spec.Counts[vocab[i]]
		if count <= 0 {
			wrong[i] = x
			continue
		}
		x -= spec.Frequency * float64(count) // DEFECT: frequency applied first
		if x > 0 {
			x /= spec.Repetition
		} else {
			x *= spec.Repetition
		}
		x -= spec.Presence
		wrong[i] = x
	}

	res, err := RunCase(c, ReferenceRunner{}, penaltyEngine("engine-wrong-order", wrong), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("frequency-before-repetition ordering must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "penalty-ordering" || fb.FailingKind != "differential" {
		t.Errorf("failing oracle = %q (%s), want penalty-ordering (differential)", fb.FailingOracle, fb.FailingKind)
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 0 {
		t.Fatalf("expected first divergence at token 0 (first penalized token), got %+v", fb.FirstDivergence)
	}
	if !strings.Contains(fb.Detail, `"the"`) {
		t.Errorf("detail must name the token whose penalized logit is wrong; got %q", fb.Detail)
	}
}

// TestPenaltyOrderingWrongSignFails injects the wrong-sign defect: the engine
// divides every penalized logit by r regardless of sign, instead of multiplying
// negative logits. Positive-logit tokens still match, so the first divergence
// must localize to "sat" at index 2 — the first penalized NEGATIVE logit.
func TestPenaltyOrderingWrongSignFails(t *testing.T) {
	c := penaltyTestCase()
	spec := penaltyTestSpec(t, c)
	raw := c.Reference.Logits[0]
	vocab := c.Reference.Tokens

	wrong := make([]float64, len(raw))
	for i, x := range raw {
		count := spec.Counts[vocab[i]]
		if count <= 0 {
			wrong[i] = x
			continue
		}
		x /= spec.Repetition // DEFECT: divides regardless of the logit's sign
		x -= spec.Presence
		x -= spec.Frequency * float64(count)
		wrong[i] = x
	}

	res, err := RunCase(c, ReferenceRunner{}, penaltyEngine("engine-wrong-sign", wrong), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("sign-blind repetition scaling must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 2 {
		t.Fatalf("expected first divergence at token 2 (first penalized negative logit), got %+v", fb.FirstDivergence)
	}
	if !strings.Contains(fb.Detail, `"sat"`) {
		t.Errorf("detail must name the token whose penalized logit is wrong; got %q", fb.Detail)
	}
}

// TestPenaltyOrderingDoubleApplyFails injects the double-apply defect: the
// presence penalty is subtracted twice for every penalized token. The first
// penalized token, "the" at index 0, must be the first divergence.
func TestPenaltyOrderingDoubleApplyFails(t *testing.T) {
	c := penaltyTestCase()
	spec := penaltyTestSpec(t, c)
	vocab := c.Reference.Tokens

	wrong := penApply(c.Reference.Logits[0], vocab, spec)
	for i := range wrong {
		if spec.Counts[vocab[i]] > 0 {
			wrong[i] -= spec.Presence // DEFECT: presence applied a second time
		}
	}

	res, err := RunCase(c, ReferenceRunner{}, penaltyEngine("engine-double-apply", wrong), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("double-applied presence penalty must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FirstDivergence == nil || fb.FirstDivergence.Index != 0 {
		t.Fatalf("expected first divergence at token 0, got %+v", fb.FirstDivergence)
	}
	if !strings.Contains(fb.Detail, `"the"`) {
		t.Errorf("detail must name the token whose penalized logit is wrong; got %q", fb.Detail)
	}
}

// TestPenaltyOrderingMalformedSpecFailsClosed proves the fail-closed property:
// a case whose prompt does not carry a parseable penalty spec cannot pass —
// a run that cannot be recomputed is not green.
func TestPenaltyOrderingMalformedSpecFailsClosed(t *testing.T) {
	c := penaltyTestCase()
	c.Prompt = "not a penalty spec"
	v := PenaltyOrdering{}.Judge(c.Reference, c.Reference, c)
	if v.Pass {
		t.Fatalf("malformed penalty spec must fail closed; got %+v", v)
	}
	if !strings.Contains(v.Detail, "penalty spec malformed") {
		t.Errorf("detail should say the spec is malformed; got %q", v.Detail)
	}
}
