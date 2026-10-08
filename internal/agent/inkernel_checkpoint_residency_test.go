package agent

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type checkpointResidencyWire struct {
	Scope            string `json:"scope"`
	Reads            int    `json:"reads"`
	Hits             int    `json:"hits"`
	BytesRead        int64  `json:"bytes_read"`
	Evictions        int    `json:"evictions"`
	Failures         int    `json:"failures"`
	BudgetBytes      int64  `json:"budget_bytes"`
	ResidentBytes    int64  `json:"resident_bytes"`
	PeakBytes        int64  `json:"peak_bytes"`
	ResidentCount    int    `json:"resident_count"`
	OverlayRows      int    `json:"overlay_rows"`
	OverlayBytesRead int64  `json:"overlay_bytes_read"`
}

func checkpointFromLedger(t *testing.T, l MoEResidencyLedger) *checkpointResidencyWire {
	t.Helper()
	raw, err := json.Marshal(l)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Checkpoint *checkpointResidencyWire `json:"checkpoint"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatal(err)
	}
	return wire.Checkpoint
}

func checkpointResidencyReport(reads, hits int, bytesRead int64) model.MoEResidencyReport {
	return model.MoEResidencyReport{Checkpoint: model.ExpertCheckpointStats{
		Enabled: true, Reads: reads, Hits: hits, BytesRead: bytesRead,
		Evictions: 3, Failures: 2, BudgetBytes: 1 << 20, ResidentBytes: 320 << 10,
		PeakBytes: 512 << 10, ResidentCount: 5, OverlayRows: 7, OverlayBytesRead: 28 << 10,
	}}
}

// Checkpoint counters are model-lifetime snapshots, so repeated request folds must not sum them.
// fak-test:runtime fast est=1ms lane=default
func TestCheckpointResidencyLatestModelLifetimeSnapshotSurvivesRinglessRequests(t *testing.T) {
	p := &InKernelPlanner{}
	p.foldMoEResidency(checkpointResidencyReport(10, 4, 40<<10), 3)
	p.foldMoEResidency(checkpointResidencyReport(13, 6, 52<<10), 5)
	p.foldMoEResidency(checkpointResidencyReport(13, 6, 52<<10), 5)

	l := p.MoEResidencyStats()
	if checkpointFromLedger(t, l) == nil {
		t.Fatal("model-lifetime checkpoint snapshot was dropped because no device ring was enabled")
	}
	c := checkpointFromLedger(t, l)
	if c.Scope != "model_lifetime" {
		t.Fatalf("checkpoint scope=%q want model_lifetime", c.Scope)
	}
	if c.Reads != 13 || c.Hits != 6 || c.BytesRead != 52<<10 {
		t.Fatalf("checkpoint reads/hits/bytes=%d/%d/%d want 13/6/%d", c.Reads, c.Hits, c.BytesRead, 52<<10)
	}
	if c.Evictions != 3 || c.Failures != 2 || c.BudgetBytes != 1<<20 ||
		c.ResidentBytes != 320<<10 || c.PeakBytes != 512<<10 || c.ResidentCount != 5 ||
		c.OverlayRows != 7 || c.OverlayBytesRead != 28<<10 {
		t.Fatalf("checkpoint bounded fields evictions/failures/budget/resident/peak/count/overlay_rows/overlay_bytes=%d/%d/%d/%d/%d/%d/%d/%d",
			c.Evictions, c.Failures, c.BudgetBytes, c.ResidentBytes, c.PeakBytes, c.ResidentCount, c.OverlayRows, c.OverlayBytesRead)
	}
	if l.Requests != 0 || l.Tokens != 0 {
		t.Fatalf("ring requests/tokens=%d/%d want 0/0 for checkpoint-only observations", l.Requests, l.Tokens)
	}
}

// The mixed path preserves request-scoped ring sums beside the model-scoped snapshot.
// fak-test:runtime fast est=1ms lane=default
func TestCheckpointResidencyAddsASecondTierWithoutChangingRingSums(t *testing.T) {
	p := &InKernelPlanner{}
	rep := moeLedgerReport(9, 3, 0, 1, 12<<10, 1<<20, 64<<10)
	rep.Checkpoint = checkpointResidencyReport(13, 6, 52<<10).Checkpoint
	p.foldMoEResidency(rep, 12)

	l := p.MoEResidencyStats()
	if l.Requests != 1 || l.Hits != 9 || l.PageIns != 3 || l.PageInBytes != 12<<10 {
		t.Fatalf("ring requests/hits/page_ins/bytes=%d/%d/%d/%d want 1/9/3/%d",
			l.Requests, l.Hits, l.PageIns, l.PageInBytes, 12<<10)
	}
	if c := checkpointFromLedger(t, l); c == nil || c.Reads != 13 {
		if c == nil {
			t.Fatal("mixed ring+checkpoint path omitted checkpoint tier")
		}
		t.Fatalf("mixed checkpoint reads=%d want 13", c.Reads)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCheckpointResidencyAbsentTierIsOmitted(t *testing.T) {
	p := &InKernelPlanner{}
	p.foldMoEResidency(moeLedgerReport(1, 1, 0, 0, 4096, 1<<20, 4096), 1)
	if got := checkpointFromLedger(t, p.MoEResidencyStats()); got != nil {
		t.Fatalf("fully resident model emitted checkpoint scope=%q reads=%d", got.Scope, got.Reads)
	}
}

// Cross the production method with a real model-owned tier and Session.
// fak-test:runtime fast est=10ms lane=default
func TestCheckpointResidencyRealSessionSourceReachesPlannerLedger(t *testing.T) {
	m := model.NewSyntheticMoE(tinyMoECfg())
	m.SetExpertCheckpoint(model.NewExpertCheckpointTier(1 << 20))
	s := m.NewSession()
	defer s.Close()
	p := &InKernelPlanner{}
	p.noteMoEResidency(s, 4)
	c := checkpointFromLedger(t, p.MoEResidencyStats())
	if c == nil {
		t.Fatal("real model checkpoint source did not reach planner ledger")
	}
	if c.Scope != "model_lifetime" || c.BudgetBytes != 1<<20 {
		t.Fatalf("real checkpoint scope/budget=%q/%d want model_lifetime/%d", c.Scope, c.BudgetBytes, 1<<20)
	}
}
