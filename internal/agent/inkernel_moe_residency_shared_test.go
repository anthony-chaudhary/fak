package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_moe_residency_shared_test.go — a model-scoped ring reports its LIFETIME counters on
// every request, so summing them would count request 1's page-ins again on request 2. The fold
// books only each request's increment and keeps the resident bytes the ring still holds.

func sharedLedgerReport(hits, pageIns, evictions int, pageInBytes, resident int64) model.MoEResidencyReport {
	rep := moeLedgerReport(hits, pageIns, 0, evictions, pageInBytes, 1<<30, resident)
	rep.Ring.ResidentBytes = resident
	rep.Shared = &model.SharedExpertRingStats{Enabled: true, Ring: rep.Ring}
	return rep
}

// fak-test:runtime fast est=1ms lane=default
func TestMoEResidencyLedgerFoldsModelRingIncrements(t *testing.T) {
	p := &InKernelPlanner{}
	// Request 1: cold, 30 page-ins. Request 2: the same experts, all hits.
	p.foldMoEResidency(sharedLedgerReport(0, 30, 0, 30*4096, 30*4096), 10)
	p.foldMoEResidency(sharedLedgerReport(30, 30, 0, 30*4096, 30*4096), 10)
	l := p.MoEResidencyStats()
	if l.Requests != 2 || l.Lookups != 60 || l.Hits != 30 || l.PageIns != 30 || l.PageInBytes != 30*4096 {
		t.Fatalf("model-ring fold double-counted lifetime counters: %+v", l)
	}
	if l.RingScope != "model" || l.ResidentBytes != 30*4096 {
		t.Fatalf("ring scope=%q resident=%d, want model and %d", l.RingScope, l.ResidentBytes, 30*4096)
	}
	if got := l.HitRate(); got != 0.5 {
		t.Fatalf("hit rate=%v, want 0.5", got)
	}
	// A new ring (counters restart) is booked from zero, never as a negative increment.
	p.foldMoEResidency(sharedLedgerReport(2, 5, 0, 5*4096, 5*4096), 1)
	if l := p.MoEResidencyStats(); l.Hits != 32 || l.PageIns != 35 {
		t.Fatalf("restarted ring fold hits=%d page_ins=%d, want 32 and 35", l.Hits, l.PageIns)
	}
	// A request-scoped ring frees its residency at Close.
	p.foldMoEResidency(moeLedgerReport(1, 1, 0, 0, 4096, 1<<30, 4096), 1)
	if l := p.MoEResidencyStats(); l.RingScope != "request" || l.ResidentBytes != 0 {
		t.Fatalf("request ring scope=%q resident=%d, want request and 0", l.RingScope, l.ResidentBytes)
	}
}
