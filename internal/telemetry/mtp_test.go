package telemetry

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"
)

func TestMTPCollector_RecordStepAndSnapshot(t *testing.T) {
	c := NewCollector(8)

	// Step 1: 4 proposed, 3 accepted, 1 rollback, 3 committed pages, 1 freed page
	c.RecordStep(4, 3, 3, 1, true)
	// Step 2: 4 proposed, 4 accepted, 0 rollback, 4 committed pages, 0 freed pages
	c.RecordStep(4, 4, 4, 0, false)

	snap := c.Snapshot()

	if snap.TotalProposed != 8 {
		t.Fatalf("expected total proposed 8, got %d", snap.TotalProposed)
	}
	if snap.TotalAccepted != 7 {
		t.Fatalf("expected total accepted 7, got %d", snap.TotalAccepted)
	}
	if snap.TotalRollbacks != 1 {
		t.Fatalf("expected total rollbacks 1, got %d", snap.TotalRollbacks)
	}
	if snap.RollbackCount != 1 {
		t.Fatalf("expected rollback count 1, got %d", snap.RollbackCount)
	}
	if snap.CommittedPages != 7 {
		t.Fatalf("expected committed pages 7, got %d", snap.CommittedPages)
	}
	if snap.FreedPages != 1 {
		t.Fatalf("expected freed pages 1, got %d", snap.FreedPages)
	}
	// Acceptance rate = 7/8 = 0.875
	if snap.AcceptanceRate != 0.875 {
		t.Fatalf("expected acceptance rate 0.875, got %g", snap.AcceptanceRate)
	}
	if snap.RollingRate != 0.875 {
		t.Fatalf("expected rolling rate 0.875, got %g", snap.RollingRate)
	}
	if snap.LifetimeRate != 0.875 {
		t.Fatalf("expected lifetime rate 0.875, got %g", snap.LifetimeRate)
	}
	if snap.DecodeSpeedup <= 1.0 {
		t.Fatalf("expected speedup > 1.0 for high acceptance, got %g", snap.DecodeSpeedup)
	}
}

func TestMTPCollector_SlidingWindowRollingRate(t *testing.T) {
	c := NewCollector(3) // Window capacity = 3 steps

	// Steps 1-3: 10 proposed, 10 accepted in each -> window 30/30 = 100%
	c.RecordStep(10, 10, 10, 0, false)
	c.RecordStep(10, 10, 10, 0, false)
	c.RecordStep(10, 10, 10, 0, false)

	snap := c.Snapshot()
	if snap.RollingRate != 1.0 {
		t.Fatalf("expected rolling rate 1.0, got %g", snap.RollingRate)
	}

	// Step 4 pushes out step 1. Step 4 has 10 proposed, 0 accepted.
	c.RecordStep(10, 0, 0, 10, true)

	snap = c.Snapshot()
	// Window now has: 10/10, 10/10, 0/10 -> 20/30 = 0.6666...
	wantRolling := 20.0 / 30.0
	if diff := snap.RollingRate - wantRolling; diff > 0.001 || diff < -0.001 {
		t.Fatalf("expected rolling rate ~%g, got %g", wantRolling, snap.RollingRate)
	}
	// Lifetime rate: 30 accepted / 40 proposed = 0.75
	if snap.LifetimeRate != 0.75 {
		t.Fatalf("expected lifetime rate 0.75, got %g", snap.LifetimeRate)
	}
}

func TestMTPCollector_DecodeSpeedup(t *testing.T) {
	c := NewCollector(16)

	// Perfect acceptance: 4 proposed, 4 accepted per round for 5 rounds
	for i := 0; i < 5; i++ {
		c.RecordStep(4, 4, 4, 0, false)
	}

	snap := c.Snapshot()
	// Total accepted = 20, Total proposed = 20, steps = 5
	// tokensProduced = 20 + 5 = 25
	// cost = 5 + 0.25 * 20 = 10
	// speedup = 25 / 10 = 2.5
	if snap.DecodeSpeedup != 2.5 {
		t.Fatalf("expected speedup 2.5, got %g", snap.DecodeSpeedup)
	}

	// Fallback should clamp speedup to 1.0
	c.SetFallback(true, "low_acceptance")
	snap = c.Snapshot()
	if snap.DecodeSpeedup != 1.0 {
		t.Fatalf("expected fallback speedup to be 1.0, got %g", snap.DecodeSpeedup)
	}
}

func TestMTPCollector_FallbackAndTripwire(t *testing.T) {
	c := NewCollector(16)
	c.SetFallback(true, "acceptance_below_50pct")
	c.SetTripwire(true, "greedy_arg_mismatch")

	snap := c.Snapshot()
	if !snap.InFallback || snap.FallbackReason != "acceptance_below_50pct" {
		t.Fatalf("expected in_fallback with reason, got %+v", snap)
	}
	if !snap.TripwireTripped || snap.TripwireReason != "greedy_arg_mismatch" {
		t.Fatalf("expected tripwire_tripped with reason, got %+v", snap)
	}

	summary := snap.TUISummary()
	if !strings.Contains(summary, "fallback (acceptance_below_50pct)") {
		t.Fatalf("expected summary to reflect fallback, got: %s", summary)
	}
}

func TestMTPCollector_MetalStatsIngestion(t *testing.T) {
	c := NewCollector(16)

	stats := MetalMTPAcceptanceStats{
		TotalProposed:   100,
		TotalAccepted:   85,
		TotalRollbacks:  3,
		WindowProposed:  32,
		WindowAccepted:  28,
		RollingRate:     0.875,
		LifetimeRate:    0.85,
		CommittedPages:  85,
		FreedPages:      15,
		TotalGenerated:  120,
		InFallback:      false,
		TripwireTripped: false,
	}

	c.RecordMetalStats(stats)
	snap := c.Snapshot()

	if snap.TotalProposed != 100 {
		t.Errorf("got TotalProposed %d, want 100", snap.TotalProposed)
	}
	if snap.TotalAccepted != 85 {
		t.Errorf("got TotalAccepted %d, want 85", snap.TotalAccepted)
	}
	if snap.RollbackCount != 3 {
		t.Errorf("got RollbackCount %d, want 3", snap.RollbackCount)
	}
	if snap.CommittedPages != 85 {
		t.Errorf("got CommittedPages %d, want 85", snap.CommittedPages)
	}
	if snap.FreedPages != 15 {
		t.Errorf("got FreedPages %d, want 15", snap.FreedPages)
	}
	if snap.RollingRate != 0.875 {
		t.Errorf("got RollingRate %g, want 0.875", snap.RollingRate)
	}
	if snap.AcceptanceRate != 0.875 {
		t.Errorf("got AcceptanceRate %g, want 0.875", snap.AcceptanceRate)
	}
	if snap.LifetimeRate != 0.85 {
		t.Errorf("got LifetimeRate %g, want 0.85", snap.LifetimeRate)
	}
}

func TestMTPCollector_PrometheusOutput(t *testing.T) {
	c := NewCollector(16)
	c.RecordStep(4, 3, 3, 1, true)
	c.RecordStep(4, 4, 4, 0, false)

	prom := c.Prometheus()

	expectedMetrics := []string{
		"fak_mtp_rolling_acceptance_rate",
		"fak_mtp_lifetime_acceptance_rate",
		"fak_mtp_acceptance_rate",
		"fak_mtp_decode_speedup",
		"fak_mtp_rollback_count",
		"fak_mtp_tokens_proposed_total",
		"fak_mtp_tokens_accepted_total",
		"fak_mtp_tokens_generated_total",
		"fak_mtp_pages_committed_total",
		"fak_mtp_pages_freed_total",
		"fak_mtp_fallback_active",
		"fak_mtp_tripwire_tripped",
		"fak_mtp_tokens_saved_per_minute",
	}

	for _, metric := range expectedMetrics {
		if !strings.Contains(prom, "# HELP "+metric) {
			t.Errorf("missing # HELP for %s", metric)
		}
		if !strings.Contains(prom, "# TYPE "+metric) {
			t.Errorf("missing # TYPE for %s", metric)
		}
		if !strings.Contains(prom, metric+" ") {
			t.Errorf("missing metric value line for %s", metric)
		}
	}

	var buf bytes.Buffer
	if err := c.Snapshot().WritePrometheus(&buf); err != nil {
		t.Fatalf("WritePrometheus failed: %v", err)
	}
	if buf.String() != prom {
		t.Fatalf("WritePrometheus output did not match Prometheus()")
	}
}

func TestMTPCollector_TUISummaryAndBlock(t *testing.T) {
	c := NewCollector(16)
	c.RecordStep(4, 3, 3, 1, true)

	snap := c.Snapshot()
	summary := snap.TUISummary()
	if !strings.Contains(summary, "75.0% acc") || !strings.Contains(summary, "1 rollbacks") {
		t.Errorf("unexpected TUISummary: %s", summary)
	}

	block := snap.TUIBlock(10)
	if len(block) != 2 {
		t.Fatalf("expected 2 block lines, got %d", len(block))
	}
	if !strings.Contains(block[0], "75.0% acc") {
		t.Errorf("expected block line 0 to have acc, got %s", block[0])
	}
	if !strings.Contains(block[1], "1 rollbacks") {
		t.Errorf("expected block line 1 to have rollbacks, got %s", block[1])
	}
}

func TestMTPCollector_ConcurrentAccess(t *testing.T) {
	c := NewCollector(32)
	var wg sync.WaitGroup

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				c.RecordStep(4, 3, 3, 1, j%2 == 0)
				_ = c.Snapshot()
				_ = c.Prometheus()
			}
		}(i)
	}

	wg.Wait()
	snap := c.Snapshot()
	if snap.TotalProposed != 20*50*4 {
		t.Fatalf("expected %d total proposed, got %d", 20*50*4, snap.TotalProposed)
	}
}

func TestMTPCollector_JSONRoundTrip(t *testing.T) {
	metrics := MTPMetrics{
		TotalProposed:  100,
		TotalAccepted:  80,
		TotalRollbacks: 5,
		AcceptanceRate: 0.80,
		RollingRate:    0.80,
		LifetimeRate:   0.80,
		DecodeSpeedup:  1.65,
		RollbackCount:  5,
		CommittedPages: 80,
		FreedPages:     20,
		TotalGenerated: 125,
		InFallback:     false,
	}

	data, err := json.Marshal(metrics)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}

	// Verify required witness keys are present in JSON
	for _, key := range []string{"mtp_acceptance_rate", "mtp_decode_speedup", "mtp_rollback_count"} {
		if _, ok := parsed[key]; !ok {
			t.Errorf("JSON missing required witness key %q", key)
		}
	}

	var decoded MTPMetrics
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal to struct failed: %v", err)
	}

	if decoded.AcceptanceRate != 0.80 || decoded.DecodeSpeedup != 1.65 || decoded.RollbackCount != 5 {
		t.Errorf("decoded fields mismatch: %+v", decoded)
	}
}
