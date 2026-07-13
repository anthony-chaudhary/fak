package quality

import (
	"strings"
	"testing"
)

// TestTpParityFaithfulShardingPasses is the parity contract on the happy path:
// a faithful sharded engine — stable within-shard order, stable all-reduce
// order — reproduces the TP=1 reference token for token at every tested
// degree, and its reassociation-level logit differences stay inside the fp
// tolerance, so the run is a clean pass with no failure bundle.
func TestTpParityFaithfulShardingPasses(t *testing.T) {
	c := TpParityCase()
	for _, degree := range []int{2, 4} {
		res, err := RunCase(c, ReferenceRunner{}, TpEngine(degree, ""), oraclesFor(t, c))
		if err != nil {
			t.Fatalf("RunCase(degree=%d): %v", degree, err)
		}
		if !res.Pass {
			t.Fatalf("faithful TP=%d engine should match TP=1; got %s", degree, Explain(res))
		}
		if res.FailureBundle != nil {
			t.Fatalf("clean TP=%d parity must not carry a failure bundle: %+v", degree, res.FailureBundle)
		}
		// Token-exact parity: the argmax survives reassociation ulps.
		for i, tok := range res.Provenance.Engine.Tokens {
			if tok != c.Reference.Tokens[i] {
				t.Fatalf("TP=%d token %d = %q, want reference %q", degree, i, tok, c.Reference.Tokens[i])
			}
		}
	}
}

// TestTpParityReorderDefectFailsAtBugStep is the localized-defect witness: a
// reduction-order/associativity bug (shard 0's fp sum reversed at step 2)
// separates the cancellation pair, the huge term absorbs the shard's small
// mass — winner bonus included — and the decoded token flips from the winner
// to the runner-up. The oracle pins the first divergence to exactly that step
// with both tokens reported, and the prefix before it is intact.
func TestTpParityReorderDefectFailsAtBugStep(t *testing.T) {
	c := TpParityCase()
	res, err := RunCase(c, ReferenceRunner{}, TpEngine(2, tpDefectReorder), oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("reduce-reorder engine must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing parity run must carry a failure bundle")
	}
	if fb.FailingOracle != "tp-parity" {
		t.Errorf("first failing oracle = %q, want tp-parity", fb.FailingOracle)
	}
	d := fb.FirstDivergence
	if d == nil || d.Index != tpBugStep {
		t.Fatalf("expected first divergence at step %d, got %+v", tpBugStep, d)
	}
	// The prefix before the bug step matches — localization is doing work.
	for i := 0; i < tpBugStep; i++ {
		if fb.Engine.Tokens[i] != fb.Reference.Tokens[i] {
			t.Fatalf("token %d before the bug step should match: reference %q, engine %q",
				i, fb.Reference.Tokens[i], fb.Engine.Tokens[i])
		}
	}
	// The construction pins WHICH token flipped to which: the stable reduction
	// decodes the winner, the reordered reduction the runner-up.
	if want := tpVocab[tpWinner(tpBugStep)]; d.Reference != want {
		t.Errorf("divergence reference token = %q, want winner %q", d.Reference, want)
	}
	if want := tpVocab[tpRunner(tpBugStep)]; d.Engine != want {
		t.Errorf("divergence engine token = %q, want runner-up %q", d.Engine, want)
	}
	if d.Reference == d.Engine {
		t.Errorf("divergence tokens must differ; both %q", d.Reference)
	}
	if !strings.Contains(fb.Detail, "TP=1") {
		t.Errorf("detail should name the TP=1 baseline; got %q", fb.Detail)
	}
}

// TestTpParityLogitToleranceRung proves the fp-tolerance contract both ways:
// a logit drift beyond tolerance fails at its step even when every token still
// matches, while the faithful sharded trace — whose logits are allowed to
// differ from TP=1 by reassociation ulps — passes the same rung.
func TestTpParityLogitToleranceRung(t *testing.T) {
	c := TpParityCase()
	ref := c.Reference

	// Beyond-tolerance drift with identical tokens: deep-copy the reference
	// logits and perturb one entry above the gate.
	eng := ref
	eng.Logits = make([][]float64, len(ref.Logits))
	for i := range ref.Logits {
		eng.Logits[i] = append([]float64(nil), ref.Logits[i]...)
	}
	eng.Logits[1][0] += 1e-6
	v := TpParity{}.Judge(ref, eng, c)
	if v.Pass {
		t.Fatalf("logit drift beyond tolerance must fail; got %+v", v)
	}
	if v.FirstDivergence == nil || v.FirstDivergence.Index != 1 {
		t.Fatalf("expected drift localized to step 1, got %+v", v.FirstDivergence)
	}
	if !strings.Contains(v.Detail, "logit drift") {
		t.Errorf("detail should name the logit drift; got %q", v.Detail)
	}

	// Within-tolerance reassociation noise passes: the faithful TP=2 trace is
	// judged clean against TP=1 even where its logits are not bit-identical.
	engTrace, err := TpEngine(2, "").Run(c)
	if err != nil {
		t.Fatalf("engine run: %v", err)
	}
	if v := (TpParity{}).Judge(ref, engTrace, c); !v.Pass {
		t.Fatalf("faithful sharded trace must pass the tolerance rung; got %+v", v)
	}
}
