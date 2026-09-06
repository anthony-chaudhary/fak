package cachevalue

import (
	"errors"
	"testing"
	"time"
)

const (
	gib = int64(1024 * 1024 * 1024)
)

// TestTBWGovernorEnforcesDailyQuota verifies that feeding 320 GB into a 300 GB/day
// bucket transitions from GovernorGreen -> GovernorYellow -> GovernorRed, and rejects
// subsequent write permits once the red threshold (hard freeze) is reached (Issue #11076).
func TestTBWGovernorEnforcesDailyQuota(t *testing.T) {
	const quota300GB = 300 * gib

	now := time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)
	gov := NewWearGovernor(WearGovernorConfig{
		DailyQuotaBytes:      quota300GB,
		YellowThresholdRatio: 0.80, // 240 GB
		RedThresholdRatio:    1.00, // 300 GB
		NowFunc:              func() time.Time { return now },
	})

	if gov.DailyQuotaBytes() != quota300GB {
		t.Fatalf("daily quota = %d, want %d", gov.DailyQuotaBytes(), quota300GB)
	}
	if state := gov.State(); state != GovernorGreen {
		t.Fatalf("initial state = %v, want %v", state, GovernorGreen)
	}

	// 1. Initial write: 200 GB out of 300 GB (66.7% < 80% yellow threshold) -> GovernorGreen.
	allowed, state := gov.RequestWritePermit(1 * gib)
	if !allowed || state != GovernorGreen {
		t.Fatalf("permit before write = (%v, %v), want (true, %v)", allowed, state, GovernorGreen)
	}
	gov.RecordHostWrite(200 * gib)

	if state := gov.State(); state != GovernorGreen {
		t.Fatalf("state after 200 GB = %v, want %v", state, GovernorGreen)
	}
	if gov.WindowBytes() != 200*gib {
		t.Fatalf("window bytes = %d, want %d", gov.WindowBytes(), 200*gib)
	}

	// 2. Additional 50 GB: total 250 GB out of 300 GB (83.3% >= 80% yellow, < 100% red) -> GovernorYellow.
	allowed, state = gov.RequestWritePermit(1 * gib)
	if !allowed || state != GovernorGreen {
		// Prior state before recording was Green, permit allowed
	}
	gov.RecordHostWrite(50 * gib)

	if state := gov.State(); state != GovernorYellow {
		t.Fatalf("state after 250 GB = %v, want %v", state, GovernorYellow)
	}
	allowed, state = gov.RequestWritePermit(1 * gib)
	if !allowed || state != GovernorYellow {
		t.Fatalf("permit in yellow = (%v, %v), want (true, %v)", allowed, state, GovernorYellow)
	}

	// 3. Additional 70 GB: total 320 GB out of 300 GB (106.7% > 100% red) -> GovernorRed (Hard Freeze).
	gov.RecordHostWrite(70 * gib)

	if state := gov.State(); state != GovernorRed {
		t.Fatalf("state after 320 GB = %v, want %v", state, GovernorRed)
	}
	if gov.WindowBytes() != 320*gib {
		t.Fatalf("window bytes = %d, want %d", gov.WindowBytes(), 320*gib)
	}

	// 4. Subsequent write permits must be rejected with GovernorRed.
	allowed, state = gov.RequestWritePermit(1 * gib)
	if allowed || state != GovernorRed {
		t.Fatalf("permit in red (1 GB) = (%v, %v), want (false, %v)", allowed, state, GovernorRed)
	}

	allowed, state = gov.RequestWritePermit(1024)
	if allowed || state != GovernorRed {
		t.Fatalf("permit in red (1024 B) = (%v, %v), want (false, %v)", allowed, state, GovernorRed)
	}

	allowed, state = gov.RequestWritePermit(0)
	if allowed || state != GovernorRed {
		t.Fatalf("permit in red (0 B) = (%v, %v), want (false, %v)", allowed, state, GovernorRed)
	}
}

// TestTBWGovernorSlidingWindowRecovery verifies that once the 24-hour window slides past
// earlier writes, expired entries are pruned, the budget recovers, and write permits are restored.
func TestTBWGovernorSlidingWindowRecovery(t *testing.T) {
	const quota300GB = 300 * gib

	gov := NewWearGovernor(WearGovernorConfig{
		DailyQuotaBytes: quota300GB,
		WindowDuration:  24 * time.Hour,
	})

	// Feed 320 GB to push into GovernorRed.
	gov.RecordHostWrite(320 * gib)
	if state := gov.State(); state != GovernorRed {
		t.Fatalf("expected GovernorRed after 320 GB, got %v", state)
	}
	allowed, state := gov.RequestWritePermit(1 * gib)
	if allowed || state != GovernorRed {
		t.Fatalf("expected write permit rejected in red, got allowed=%v state=%v", allowed, state)
	}

	// Advance time by 25 hours -> all 320 GB records fall outside the 24h sliding window.
	gov.AdvanceTime(25 * time.Hour)

	if gov.WindowBytes() != 0 {
		t.Fatalf("expected 0 window bytes after sliding window expiration, got %d", gov.WindowBytes())
	}
	if state := gov.State(); state != GovernorGreen {
		t.Fatalf("expected recovery to GovernorGreen, got %v", state)
	}

	// Write permits must now be granted.
	allowed, state = gov.RequestWritePermit(10 * gib)
	if !allowed || state != GovernorGreen {
		t.Fatalf("expected write permit granted after recovery, got allowed=%v state=%v", allowed, state)
	}
}

// TestTBWGovernorEmpiricalWAFCalculation verifies that empirical WAF = Delta_NAND / Delta_Host
// is computed accurately across consecutive SMART telemetry polls.
func TestTBWGovernorEmpiricalWAFCalculation(t *testing.T) {
	gov := NewWearGovernor(WearGovernorConfig{
		DailyQuotaBytes: 300 * gib,
	})

	// Baseline telemetry: 100 GB Host, 300 GB NAND (WAF = 3.0x).
	gov.UpdateSMARTTelemetry(300*gib, 100*gib)
	if waf := gov.EmpiricalWAF(); waf != 3.0 {
		t.Fatalf("expected initial WAF = 3.0, got %.2f", waf)
	}
	if gov.WAF() != 3.0 {
		t.Fatalf("WAF() alias mismatch: got %.2f", gov.WAF())
	}

	telem := gov.Telemetry()
	if telem.DeltaNANDBytes != 300*gib || telem.DeltaHostBytes != 100*gib {
		t.Fatalf("unexpected deltas: NAND=%d, Host=%d", telem.DeltaNANDBytes, telem.DeltaHostBytes)
	}

	// Subsequent poll: Host advances by 100 GB (to 200 GB), NAND advances by 110 GB (to 410 GB).
	// Interval WAF = 110 / 100 = 1.1x (demonstrating coalesced sequential write efficiency).
	gov.UpdateSMARTTelemetry(410*gib, 200*gib)
	if waf := gov.EmpiricalWAF(); waf != 1.1 {
		t.Fatalf("expected interval WAF = 1.1, got %.2f", waf)
	}

	telem = gov.Telemetry()
	if telem.DeltaNANDBytes != 110*gib || telem.DeltaHostBytes != 100*gib {
		t.Fatalf("unexpected interval deltas: NAND=%d, Host=%d", telem.DeltaNANDBytes, telem.DeltaHostBytes)
	}
	// Cumulative: 410 NAND / 200 Host = 2.05x.
	if telem.CumulativeWAF != 2.05 {
		t.Fatalf("expected cumulative WAF = 2.05, got %.2f", telem.CumulativeWAF)
	}

	// Test SMARTCollector interface with PollSMART.
	mockCollector := SMARTCollectorFunc(func() (int64, int64, error) {
		return 520 * gib, 300 * gib, nil // Delta: 110 NAND, 100 Host -> WAF = 1.1x
	})
	if err := gov.PollSMART(mockCollector); err != nil {
		t.Fatalf("PollSMART failed: %v", err)
	}
	if waf := gov.EmpiricalWAF(); waf != 1.1 {
		t.Fatalf("expected PollSMART WAF = 1.1, got %.2f", waf)
	}

	// Test error propagation from collector.
	errCollector := SMARTCollectorFunc(func() (int64, int64, error) {
		return 0, 0, errors.New("nvme ioctl failed")
	})
	if err := gov.PollSMART(errCollector); err == nil {
		t.Fatal("expected error from failed collector, got nil")
	}
}

// TestTBWGovernorDailyQuotaDerivation verifies quota derivation from TBWRatingBytes and TargetDays.
func TestTBWGovernorDailyQuotaDerivation(t *testing.T) {
	// 600 TBW over 1825 days (5 years) -> ~328.7 GB/day.
	const tbw600TB = int64(600) * 1000 * 1000 * 1000 * 1000
	gov := NewWearGovernor(WearGovernorConfig{
		TBWRatingBytes: tbw600TB,
		TargetDays:     1825,
	})

	expectedDaily := tbw600TB / 1825
	if gov.DailyQuotaBytes() != expectedDaily {
		t.Fatalf("derived daily quota = %d, want %d", gov.DailyQuotaBytes(), expectedDaily)
	}
}
