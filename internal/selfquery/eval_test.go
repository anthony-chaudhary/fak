package selfquery

import "testing"

// Retrieval-quality gate for the card ranker (#3162).
//
// TestRetrievalQualityFloor is the regression fence the issue asks for: it
// pins absolute recall@K and MRR floors for the production ranker so a score()
// or ranker.go change that degrades what a guarded agent actually sees fails
// here instead of landing unnoticed. The floors are RECORDED MEASUREMENTS —
// each was read off this harness against the committed ranker, not chosen to
// look good. Raise them when a change genuinely improves retrieval; lowering
// one is the explicit admission that retrieval got worse.

const (
	// evalK is the window graded. The self-query surface hands an agent a
	// short list, so "did the right card make the top 3" is the question that
	// matters operationally.
	evalK = 3

	// recallFloorAtK / mrrFloor are the witnessed numbers for rankCards over
	// DefaultIntentFixture on the rankerFixture catalog: recall@3=1.000,
	// MRR=1.000 (the intended card ranks FIRST for all 9 intents). The retired
	// flat baseline scores 0.667/0.611 on the same fixture, which is what
	// TestRetrievalEvalDetectsRegression uses to prove these floors can fail.
	//
	// Read the 1.000 honestly: it is a SATURATED fixture, not a "retrieval is
	// 100% correct" claim. 9 hand-authored intents over a 10-card synthetic
	// catalog (7 of which carry a label) is a regression fence with no headroom
	// to show improvement — it can catch a ranker getting worse and cannot
	// grade one getting better. Widening the fixture, and grading against the
	// live catalog rather than rankerFixture, is what would make a number
	// publishable; per the issue's fence none is quotable until then.
	recallFloorAtK = 1.0
	mrrFloor       = 1.0
)

func TestRetrievalQualityFloor(t *testing.T) {
	cards := rankerFixture()
	res := EvalRetrieval(rankCards, cards, DefaultIntentFixture, evalK)
	t.Logf("production ranker: %s", res)
	t.Logf("%s", FormatEvalReport(res))

	if res.N != len(DefaultIntentFixture) {
		t.Fatalf("graded %d intents, fixture has %d", res.N, len(DefaultIntentFixture))
	}
	if res.RecallAtK < recallFloorAtK {
		t.Fatalf("recall@%d regressed below the recorded floor: got %.3f, floor %.3f (misses: %v)",
			evalK, res.RecallAtK, recallFloorAtK, res.Misses)
	}
	if res.MRR < mrrFloor {
		t.Fatalf("MRR regressed below the recorded floor: got %.3f, floor %.3f (misses: %v)",
			res.MRR, mrrFloor, res.Misses)
	}
}

// TestRetrievalEvalDetectsRegression proves the gate can actually fail. A floor
// that has never been shown to catch anything is decoration: grading the
// retired flat lexical scorer (the pre-#3235 baseline, kept off the production
// path) must score strictly worse than the production ranker on the same
// fixture. If this ever passes with equal numbers, the harness has stopped
// discriminating and the floor above is no longer meaningful.
func TestRetrievalEvalDetectsRegression(t *testing.T) {
	cards := rankerFixture()
	prod := EvalRetrieval(rankCards, cards, DefaultIntentFixture, evalK)
	base := EvalRetrieval(rankLexicalBaseline, cards, DefaultIntentFixture, evalK)
	t.Logf("production: %s", prod)
	t.Logf("retired flat baseline: %s", base)

	if !(prod.RecallAtK > base.RecallAtK || prod.MRR > base.MRR) {
		t.Fatalf("harness no longer discriminates: production (recall@%d=%.3f mrr=%.3f) does not beat the retired baseline (recall@%d=%.3f mrr=%.3f)",
			evalK, prod.RecallAtK, prod.MRR, evalK, base.RecallAtK, base.MRR)
	}
	if base.RecallAtK >= recallFloorAtK && base.MRR >= mrrFloor {
		t.Fatalf("the recorded floors are too slack to be a gate: the retired baseline clears them (recall@%d=%.3f mrr=%.3f)",
			evalK, base.RecallAtK, base.MRR)
	}
}

// TestRetrievalEvalCoverageIsReported keeps the issue's honesty fence: the
// fixture is small, so the harness must state how much of the catalog carries
// no label rather than let a high score read as full coverage.
func TestRetrievalEvalCoverageIsReported(t *testing.T) {
	cards := rankerFixture()
	res := EvalRetrieval(rankCards, cards, DefaultIntentFixture, evalK)
	if res.CardsTotal != len(cards) {
		t.Fatalf("CardsTotal=%d, catalog has %d", res.CardsTotal, len(cards))
	}
	if res.CardsCovered <= 0 || res.CardsCovered > res.CardsTotal {
		t.Fatalf("CardsCovered=%d is not a sane share of %d", res.CardsCovered, res.CardsTotal)
	}
	if res.CardsCovered >= res.CardsTotal {
		t.Fatalf("fixture claims full catalog coverage (%d/%d) — the unmeasured fence would be vacuous",
			res.CardsCovered, res.CardsTotal)
	}
	t.Logf("coverage: %d/%d labeled; %s", res.CardsCovered, res.CardsTotal, res.Unmeasured())
}

// TestEvalRetrievalMetrics pins the metric arithmetic itself against a stub
// ranker with a known ordering, so a future refactor cannot quietly redefine
// what recall@K or MRR mean.
func TestEvalRetrievalMetrics(t *testing.T) {
	cards := []FeatureCard{{Name: "a"}, {Name: "b"}, {Name: "c"}}
	fixture := []IntentLabel{
		{Intent: "first", Want: "a"},   // rank 1 -> rr 1.0, in top 2
		{Intent: "third", Want: "c"},   // rank 3 -> rr 1/3, outside top 2
		{Intent: "absent", Want: "zz"}, // never returned -> rr 0
	}
	// Identity ranker: always returns a, b, c in order.
	rank := func(cs []FeatureCard, _ string) []FeatureCard { return cs }
	res := EvalRetrieval(rank, cards, fixture, 2)

	if want := 1.0 / 3.0; !approx(res.RecallAtK, want) {
		t.Fatalf("recall@2 = %.4f, want %.4f (only 'a' lands in the top 2)", res.RecallAtK, want)
	}
	if want := (1.0 + 1.0/3.0 + 0.0) / 3.0; !approx(res.MRR, want) {
		t.Fatalf("MRR = %.4f, want %.4f", res.MRR, want)
	}
	// 'c' is returned but below K, and 'zz' is never returned: both are misses.
	if len(res.Misses) != 2 {
		t.Fatalf("expected 2 misses (below-K and absent), got %v", res.Misses)
	}
	// 'zz' is not in the catalog, so it must not inflate coverage.
	if res.CardsCovered != 2 {
		t.Fatalf("CardsCovered = %d, want 2 (a and c; zz is not in the catalog)", res.CardsCovered)
	}
}

func approx(a, b float64) bool {
	d := a - b
	return d < 1e-9 && d > -1e-9
}
