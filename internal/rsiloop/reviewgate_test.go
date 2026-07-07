package rsiloop

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// TestReviewCostReusesCacheReadMarginal proves the cost model is priced on the SAME
// compaction-economics basis (the canonical cacheprice.ReadMultiplier), not a
// mirrored literal: a re-read prefix at 0.1× plus output at 1.0×.
func TestReviewCostReusesCacheReadMarginal(t *testing.T) {
	c := ReviewCost{PrefixTokens: 10_000, OutputTokens: 500}
	want := 10_000*cacheprice.ReadMultiplier + 500
	if got := c.TokenEquiv(); got != want {
		t.Fatalf("TokenEquiv = %v, want %v", got, want)
	}
	// Guard the basis itself: if someone changed the marginal to 1.0×, the prefix would
	// cost 10× more and this asserts the gate tracks the canonical anchor.
	if cacheprice.ReadMultiplier != 0.1 {
		t.Fatalf("cache read marginal drifted to %v; the gate's cost basis moved", cacheprice.ReadMultiplier)
	}
}

func TestReviewTraceSignals(t *testing.T) {
	seen := map[string]bool{}
	FoldTrace(seen, ReviewTrace{ToolCalls: []string{"Read", "Read", "Grep"}})

	// A trace of already-seen tools/sequences is 0-novel; a brand-new tool + a new
	// sequence lifts both signals.
	old := ReviewTrace{ToolCalls: []string{"Read", "Grep"}}
	if nov := old.Novelty(seen); nov != 0 {
		t.Fatalf("all-seen novelty = %v, want 0", nov)
	}
	if seq := old.UnseenSequenceRatio(seen); seq != 0 {
		t.Fatalf("all-seen unseen-sequence = %v, want 0", seq)
	}

	fresh := ReviewTrace{ToolCalls: []string{"Bash", "Edit"}, Errors: 1}
	if nov := fresh.Novelty(seen); nov != 1 {
		t.Fatalf("all-new novelty = %v, want 1", nov)
	}
	if seq := fresh.UnseenSequenceRatio(seen); seq != 1 {
		t.Fatalf("all-new unseen-sequence = %v, want 1", seq)
	}
	if d := fresh.ErrorDensity(); d != 0.5 {
		t.Fatalf("error density = %v, want 0.5", d)
	}
	// Empty trace has no shape and no density.
	empty := ReviewTrace{}
	if empty.Novelty(seen) != 0 || empty.UnseenSequenceRatio(seen) != 0 || empty.ErrorDensity() != 0 {
		t.Fatalf("empty trace signals not all zero")
	}
}

// TestGateSpawnsOnlyWhenValueExceedsCost is the core done-condition witness: a
// high-value (novel, error-dense) trace whose expected value beats the fork cost
// spawns; the SAME trace priced against a fork cost above its expected value skips.
func TestGateSpawnsOnlyWhenValueExceedsCost(t *testing.T) {
	l, err := OpenReviewLedger("", DefaultReviewGateConfig())
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{} // empty corpus → everything is novel
	trace := ReviewTrace{ToolCalls: []string{"Bash", "Edit", "Grep"}, Errors: 3}

	// Prior keep-rate 0.2, max signal → pKeep = clamp(0.2*(0.5+1)) = 0.3, exp value =
	// 0.3 * 50_000 = 15_000 token-equiv. A cheap fork (well under 15k) spawns.
	cheap := ReviewCost{PrefixTokens: 20_000, OutputTokens: 200} // 2_000 + 200 = 2_200 teq
	est := l.Estimate(trace, seen, cheap)
	if !est.Spawn {
		t.Fatalf("expected spawn: expValue=%.0f cost=%.0f", est.ExpValueTeq, est.CostTeq)
	}
	if est.ExpValueTeq <= est.CostTeq {
		t.Fatalf("spawn but value %.0f !> cost %.0f", est.ExpValueTeq, est.CostTeq)
	}

	// A costly fork above the expected value must skip — this is "net-positive only".
	costly := ReviewCost{PrefixTokens: 100_000, OutputTokens: 20_000} // 10_000 + 20_000 = 30_000 teq
	skip := l.Estimate(trace, seen, costly)
	if skip.Spawn {
		t.Fatalf("expected skip: expValue=%.0f cost=%.0f", skip.ExpValueTeq, skip.CostTeq)
	}
}

// TestLedgerRoundTripAndDecisionOutcome exercises the durable ledger: a decision row
// and its one-to-one realized outcome row survive a reopen, and the outcome
// bookkeeping refuses a skip, an unknown seq, and a double-resolve.
func TestLedgerRoundTripAndDecisionOutcome(t *testing.T) {
	path := filepath.Join(t.TempDir(), "review-fire.jsonl")
	l, err := OpenReviewLedger(path, DefaultReviewGateConfig())
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	trace := ReviewTrace{ToolCalls: []string{"Bash", "Edit", "Grep"}, Errors: 3}

	est, seq, err := l.Decide(trace, seen, ReviewCost{PrefixTokens: 10_000, OutputTokens: 200})
	if err != nil {
		t.Fatal(err)
	}
	if !est.Spawn {
		t.Fatalf("expected a spawning decision to exercise the outcome path")
	}
	if pend := l.PendingOutcomes(); len(pend) != 1 || pend[0] != seq {
		t.Fatalf("pending outcomes = %v, want [%d]", pend, seq)
	}

	if _, err := l.RecordOutcome(seq+999, true); err == nil {
		t.Fatalf("expected error resolving an unknown decision seq")
	}
	if _, err := l.RecordOutcome(seq, true); err != nil {
		t.Fatalf("RecordOutcome: %v", err)
	}
	if _, err := l.RecordOutcome(seq, true); err == nil {
		t.Fatalf("expected error double-resolving decision seq %d", seq)
	}
	if pend := l.PendingOutcomes(); len(pend) != 0 {
		t.Fatalf("pending outcomes after resolve = %v, want none", pend)
	}

	// A SKIP decision has no fire, so it cannot be resolved.
	_, skipSeq, err := l.Decide(trace, seen, ReviewCost{PrefixTokens: 10_000_000, OutputTokens: 0})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := l.RecordOutcome(skipSeq, true); err == nil {
		t.Fatalf("expected error resolving a SKIP decision seq %d", skipSeq)
	}

	// Reopen: the decision + outcome rows survive, and the realized keep-rate folds.
	l2, err := OpenReviewLedger(path, DefaultReviewGateConfig())
	if err != nil {
		t.Fatal(err)
	}
	if rate, n := l2.RealizedKeepRate(); n != 1 || rate != 1 {
		t.Fatalf("reopened realized keep-rate = %v over %d, want 1 over 1", rate, n)
	}
}

// TestSelfTuneUsesRealizedOutcomeNotEstimate is the LOAD-BEARING anti-reward-hack
// witness (#2816). It proves the self-tune reads the REALIZED kept/not outcome, never
// the estimator's own prediction: a run of fires that all came back NOT-kept drops the
// effective keep-rate BELOW the prior and tightens the gate, so a borderline trace that
// spawned under the prior now SKIPS. A gate that could feed its own estimate back would
// never tighten — this asserts it cannot.
func TestSelfTuneUsesRealizedOutcomeNotEstimate(t *testing.T) {
	cfg := DefaultReviewGateConfig()
	cfg.MinOutcomes = 4 // enough realized outcomes to switch off the prior
	l, err := OpenReviewLedger("", cfg)
	if err != nil {
		t.Fatal(err)
	}

	// Before any realized outcome the gate uses the prior keep-rate.
	if got := l.EffectiveKeepRate(); got != cfg.PriorKeepRate {
		t.Fatalf("cold keep-rate = %v, want prior %v", got, cfg.PriorKeepRate)
	}

	seen := map[string]bool{}
	trace := ReviewTrace{ToolCalls: []string{"Bash", "Edit", "Grep"}, Errors: 3}
	// Pick a cost that spawns under the prior (exp value 15_000 teq) but NOT under a
	// collapsed realized keep-rate: pick cost between the two expected values.
	// Under prior 0.2: pKeep=0.3, exp=15_000. Under realized 0.0: pKeep=0, exp=0.
	cost := ReviewCost{PrefixTokens: 50_000, OutputTokens: 5_000} // 5_000 + 5_000 = 10_000 teq

	// Under the prior it spawns.
	if est := l.Estimate(trace, seen, cost); !est.Spawn {
		t.Fatalf("under prior: expected spawn, exp=%.0f cost=%.0f", est.ExpValueTeq, est.CostTeq)
	}

	// Fire several reviews and record every realized outcome as NOT-kept — the honest
	// realized signal that the estimator's optimism was wrong.
	for i := 0; i < cfg.MinOutcomes; i++ {
		_, seq, err := l.Decide(trace, seen, cost)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := l.RecordOutcome(seq, false); err != nil {
			t.Fatal(err)
		}
	}

	// The realized keep-rate is now 0, below the 0.2 prior — the self-tune tightened.
	if rate, n := l.RealizedKeepRate(); rate != 0 || n < cfg.MinOutcomes {
		t.Fatalf("realized keep-rate = %v over %d, want 0 over >=%d", rate, n, cfg.MinOutcomes)
	}
	if got := l.EffectiveKeepRate(); got != 0 {
		t.Fatalf("effective keep-rate after not-kept fires = %v, want 0 (realized replaces prior)", got)
	}

	// The SAME borderline trace now SKIPS: expected value collapsed to 0 < cost. The
	// gate learned from realized outcomes, not from its own inflated estimates.
	if est := l.Estimate(trace, seen, cost); est.Spawn {
		t.Fatalf("after not-kept fires the gate must tighten and SKIP, got spawn exp=%.0f cost=%.0f", est.ExpValueTeq, est.CostTeq)
	}
}

// TestRealizedKeepRateIgnoresDecisionEstimates hardens the fence directly: the
// realized keep-rate must be computed only from outcome rows, so a ledger full of
// high-estimate SPAWN decisions with no recorded outcomes reads as zero realized
// history, not a rosy self-reported rate.
func TestRealizedKeepRateIgnoresDecisionEstimates(t *testing.T) {
	l, err := OpenReviewLedger("", DefaultReviewGateConfig())
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	trace := ReviewTrace{ToolCalls: []string{"Bash", "Edit", "Grep"}, Errors: 3}
	for i := 0; i < 5; i++ {
		if _, _, err := l.Decide(trace, seen, ReviewCost{PrefixTokens: 1_000, OutputTokens: 10}); err != nil {
			t.Fatal(err)
		}
	}
	if rate, n := l.RealizedKeepRate(); n != 0 || rate != 0 {
		t.Fatalf("realized keep-rate from decisions alone = %v over %d, want 0 over 0", rate, n)
	}
}
