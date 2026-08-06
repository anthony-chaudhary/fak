package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_moe_residency_test.go — witnesses for the serve half of R6 (#5617).
//
// The ring's physics are witnessed in internal/model. What only this package can get wrong is the
// carry: that a finished request's residency survives the session teardown that destroys the ring,
// that the aggregate is a sum of counters rather than an average of rates, that a broken ring raises
// an alarm instead of a plausible dashboard, and that a serve which declared no expert budget pays
// nothing and reports nothing.

// moeLedgerReport builds a self-consistent report the way a real ring would produce one, so a
// witness can state what it is exercising (a ten-lookup request that hit six times) instead of
// hand-tuning fields until the checks pass.
func moeLedgerReport(hits, pageIns, refusals, evictions int, pageInBytes, budget, peak int64) model.MoEResidencyReport {
	return model.MoEResidencyReport{
		Ring: model.ExpertRingStats{
			Enabled:     true,
			BudgetBytes: budget,
			PeakBytes:   peak,
			Hits:        hits,
			PageIns:     pageIns,
			Refusals:    refusals,
			Evictions:   evictions,
			Lookups:     hits + pageIns + refusals,
			PageInBytes: pageInBytes,
		},
		Reconciliation: model.MoEResidencyReconciliation{OK: true},
	}
}

// TestMoEResidencyLedgerSumsCountersRatherThanAveragingRates pins the aggregation rule. Two requests
// of wildly different size are folded; the serve's hit rate must be the one computed from the summed
// counters, NOT the mean of the two per-request rates, because the mean would let a two-lookup
// request outvote a two-hundred-lookup one on the number an operator sizes a budget against.
func TestMoEResidencyLedgerSumsCountersRatherThanAveragingRates(t *testing.T) {
	p := &InKernelPlanner{}

	// A long request that mostly hit, and a tiny one that mostly missed.
	p.foldMoEResidency(moeLedgerReport(180, 20, 0, 15, 20*4096, 1<<20, 900<<10), 100)
	p.foldMoEResidency(moeLedgerReport(1, 3, 0, 0, 3*4096, 1<<20, 400<<10), 2)

	l := p.MoEResidencyStats()
	if l.Requests != 2 || l.Tokens != 102 {
		t.Fatalf("ledger booked %d requests / %d tokens, want 2 / 102", l.Requests, l.Tokens)
	}
	if l.Hits != 181 || l.PageIns != 23 || l.Lookups != 204 || l.Evictions != 15 {
		t.Fatalf("counters did not sum: %+v", l)
	}
	if l.PageInBytes != 23*4096 {
		t.Fatalf("page-in bytes = %d, want %d", l.PageInBytes, 23*4096)
	}

	want := 181.0 / 204.0
	if got := l.HitRate(); got != want {
		t.Fatalf("hit rate %.6f, want %.6f (computed from the sums)", got, want)
	}
	// The trap this test exists for: averaging the two per-request rates gives ~0.575, which would
	// report a serve that hit 89% of the time as barely better than a coin flip.
	if mean := (180.0/200.0 + 1.0/4.0) / 2; l.HitRate() == mean {
		t.Fatalf("hit rate %.6f is the MEAN of the per-request rates; a 4-lookup request must not "+
			"weigh as much as a 200-lookup one", l.HitRate())
	}
	if got, want := l.ExpertBytesPerToken(), float64(23*4096)/102; got != want {
		t.Fatalf("expert bytes/token %.3f, want %.3f", got, want)
	}

	// Peak is the high-water mark across requests, not the last one's; the budget is a declaration,
	// so the latest wins.
	if l.PeakBytes != 900<<10 {
		t.Fatalf("peak %d, want the max across requests %d", l.PeakBytes, 900<<10)
	}
	if got := l.PeakBudgetUsed(); got != float64(900<<10)/float64(1<<20) {
		t.Fatalf("peak budget used %.4f", got)
	}
	if l.Last.Ring.Lookups != 4 {
		t.Fatalf("Last is not the most recent request: lookups=%d, want 4", l.Last.Ring.Lookups)
	}
}

// TestMoEResidencyLedgerRaisesAnAlarmOnAnUnreconciledRequest is why the per-request report bothers to
// compute checks that can fail. A ring whose accounting disagrees with itself makes every aggregate
// above it wrong, and the honest response is a counter an operator can alert on rather than a
// dashboard that keeps rendering.
func TestMoEResidencyLedgerRaisesAnAlarmOnAnUnreconciledRequest(t *testing.T) {
	p := &InKernelPlanner{}
	p.foldMoEResidency(moeLedgerReport(10, 5, 0, 1, 5*4096, 1<<20, 512<<10), 8)
	if got := p.MoEResidencyStats().ReconciliationFailures; got != 0 {
		t.Fatalf("healthy request booked %d reconciliation failures", got)
	}

	broken := moeLedgerReport(10, 5, 0, 1, 5*4096, 1<<20, 512<<10)
	broken.Ring.Lookups = 99 // the identity no longer holds
	broken.Reconciliation = model.MoEResidencyReconciliation{
		OK:     false,
		Checks: []model.MoEResidencyCheck{{Name: "lookups-identity", OK: false, Detail: "15 != 99"}},
	}
	p.foldMoEResidency(broken, 8)

	l := p.MoEResidencyStats()
	if l.ReconciliationFailures != 1 {
		t.Fatalf("reconciliation failures = %d after one broken request, want 1", l.ReconciliationFailures)
	}
	if l.Requests != 2 {
		t.Fatalf("a broken request must still be counted (%d requests): dropping it would hide the "+
			"failure from the rate it corrupts", l.Requests)
	}
}

// TestMoEResidencyLedgerIsInertWithoutARing is the default every serve runs today: no operator
// declared an expert budget, so no session builds a ring, so the ledger must stay untouched rather
// than accumulate a row of zeros that reads like a ring doing nothing.
func TestMoEResidencyLedgerIsInertWithoutARing(t *testing.T) {
	p := &InKernelPlanner{}
	for i := 0; i < 3; i++ {
		p.foldMoEResidency(model.MoEResidencyReport{}, 64) // Ring.Enabled false
	}
	l := p.MoEResidencyStats()
	if !moeLedgerIsZero(l) {
		t.Fatalf("ringless requests moved the ledger: %+v", l)
	}
	if l.HitRate() != 0 || l.ExpertBytesPerToken() != 0 || l.RefusalRate() != 0 || l.PeakBudgetUsed() != 0 {
		t.Fatal("a zero ledger produced a non-zero rate; every denominator must be guarded so " +
			"telemetry JSON never carries NaN")
	}
	// A nil planner is the accessor's other honest answer, since a caller polling telemetry should
	// not have to know whether a planner was ever constructed.
	var nilp *InKernelPlanner
	if !moeLedgerIsZero(nilp.MoEResidencyStats()) {
		t.Fatal("nil planner returned a non-zero ledger")
	}
	nilp.noteMoEResidency(nil, 0) // must not panic
}

// moeLedgerIsZero is spelled out rather than compared against the zero value because the ledger
// carries a whole report, which holds slices and is therefore not comparable.
func moeLedgerIsZero(l MoEResidencyLedger) bool {
	return l.Requests == 0 && l.Tokens == 0 && l.Lookups == 0 && l.Hits == 0 && l.PageIns == 0 &&
		l.Evictions == 0 && l.Refusals == 0 && l.PageInBytes == 0 && l.BudgetBytes == 0 &&
		l.PeakBytes == 0 && l.ReconciliationFailures == 0 && !l.Last.Ring.Enabled
}

// TestMoEResidencyLedgerCallSiteRunsOnEveryRealRequest pins the seam itself: the fold is reached by
// the actual decode path, on a real MoE model, for every completed request — and on a serve that
// declared no expert budget it costs a nil check and leaves the ledger untouched.
//
// It stops short of asserting a live ring's numbers, and deliberately does not pretend otherwise.
// The device HAL forward takes its routed-expert branch only for a DeepSeek-V4 config (hal.go's
// cfg.IsDeepSeekV4 split), and this package has no V4 synthetic to build one from, so a request that
// really pages experts in cannot be driven from here. That link is witnessed one layer down, over a
// real ring, in internal/model's expert_residency_report_test.go; what is witnessed HERE is
// everything between that report and the operator: the call site, the token count it derives, the
// summing, and the alarm.
func TestMoEResidencyLedgerCallSiteRunsOnEveryRealRequest(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())
	p := NewInKernelPlanner(m, nil, "tiny-moe", false, nil, false)
	p.quant = false // the f32 CPU forward; the synthetic fixture carries no Q8_0 build

	ids := []int{1, 2, 3, 4, 5, 6}
	for i := 0; i < 3; i++ {
		gen, _, _, _, _, _ := p.generateReused(ids, 4, 0, 0, 0, nil, func(int) bool { return true })
		if gen == 0 {
			t.Fatalf("request %d generated no tokens; the decode path never ran and the call site was "+
				"never reached", i)
		}
	}
	if l := p.MoEResidencyStats(); !moeLedgerIsZero(l) {
		t.Fatalf("a serve with no declared expert budget accumulated residency rows: %+v", l)
	}

	// And the same path with the ledger primed by hand still reports the primed numbers afterwards —
	// so a ringless request cannot silently zero a serve that HAS been measuring one.
	p.foldMoEResidency(moeLedgerReport(9, 3, 0, 1, 3*4096, 1<<20, 64<<10), 12)
	if _, _, _, _, _, _ = p.generateReused(ids, 4, 0, 0, 0, nil, func(int) bool { return true }); true {
		l := p.MoEResidencyStats()
		if l.Requests != 1 || l.Lookups != 12 {
			t.Fatalf("a ringless request disturbed a primed ledger: %+v", l)
		}
	}
}
