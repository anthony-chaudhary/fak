package safesync

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestShadowLandingDiscrepancyLogger(t *testing.T) {
	t.Run("logger writes and reads back JSONL", func(t *testing.T) {
		tmp := t.TempDir()
		logFile := filepath.Join(tmp, "custom_discrepancies.jsonl")
		logger := NewCanaryDiscrepancyLogger(tmp, logFile)

		// Initial read on non-existent file should return empty slice and nil error
		initial, err := logger.ReadDiscrepancies()
		if err != nil {
			t.Fatalf("unexpected error on empty file: %v", err)
		}
		if len(initial) != 0 {
			t.Fatalf("expected 0 initial records, got %d", len(initial))
		}

		now := time.Now().UTC().Truncate(time.Millisecond)
		d1 := CanaryDiscrepancy{
			Schema:    CanaryDiscrepancySchema,
			Timestamp: now,
			Repo:      tmp,
			Branch:    "main",
			Kind:      DiscrepancyKindRouteMismatch,
			Expected:  RouteApply,
			Actual:    RouteNoop,
			Paths:     []string{"path/a.go"},
			HeadSHA:   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			TargetSHA: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			Detail:    "test detail 1",
		}
		d2 := CanaryDiscrepancy{
			Schema:    CanaryDiscrepancySchema,
			Timestamp: now.Add(time.Second),
			Repo:      tmp,
			Branch:    "main",
			Kind:      DiscrepancyKindStateMismatch,
			Expected:  StateBehind,
			Actual:    StateInSync,
			Paths:     []string{"path/b.go"},
			HeadSHA:   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			TargetSHA: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			Detail:    "test detail 2",
		}

		if err := logger.LogDiscrepancy(d1); err != nil {
			t.Fatalf("failed to log d1: %v", err)
		}
		if err := logger.LogDiscrepancy(d2); err != nil {
			t.Fatalf("failed to log d2: %v", err)
		}

		readBack, err := logger.ReadDiscrepancies()
		if err != nil {
			t.Fatalf("failed to read discrepancies: %v", err)
		}
		if len(readBack) != 2 {
			t.Fatalf("expected 2 discrepancies, got %d", len(readBack))
		}
		if readBack[0].Kind != DiscrepancyKindRouteMismatch || readBack[0].Expected != RouteApply || readBack[0].Actual != RouteNoop {
			t.Errorf("record 0 mismatch: %+v", readBack[0])
		}
		if readBack[1].Kind != DiscrepancyKindStateMismatch || readBack[1].Expected != StateBehind || readBack[1].Actual != StateInSync {
			t.Errorf("record 1 mismatch: %+v", readBack[1])
		}
	})

	t.Run("AuditShadowLanding produces valid receipt", func(t *testing.T) {
		_, clone := setupTestOriginAndClone(t)
		ctx := context.Background()

		opts := ShadowOptions{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
			Shadow: true,
		}

		receipt, err := AuditShadowLanding(ctx, clone, opts)
		if err != nil {
			t.Fatalf("AuditShadowLanding failed: %v", err)
		}
		if receipt == nil {
			t.Fatal("receipt is nil")
		}
		if receipt.Schema != ShadowLandingReceiptSchema {
			t.Errorf("expected schema %q, got %q", ShadowLandingReceiptSchema, receipt.Schema)
		}
		if receipt.ReceiptID == "" {
			t.Error("expected non-empty ReceiptID")
		}
		if receipt.Repo != clone {
			t.Errorf("expected repo %q, got %q", clone, receipt.Repo)
		}
		if receipt.Branch != "work" {
			t.Errorf("expected branch work, got %q", receipt.Branch)
		}
		if receipt.SyncState != StateInSync {
			t.Errorf("expected SyncState in-sync, got %q", receipt.SyncState)
		}
		if !receipt.ShadowMode {
			t.Error("expected ShadowMode to be true")
		}
		if receipt.ActualRoute != RouteNoop {
			t.Errorf("expected ActualRoute ROUTE_NOOP, got %q", receipt.ActualRoute)
		}
		if receipt.CanaryRoute != RouteNoop {
			t.Errorf("expected CanaryRoute ROUTE_NOOP, got %q", receipt.CanaryRoute)
		}
		if receipt.DiscrepancyCount != 0 {
			t.Errorf("expected DiscrepancyCount 0, got %d", receipt.DiscrepancyCount)
		}
		if receipt.Decision != DecisionCleared {
			t.Errorf("expected Decision CLEARED, got %q", receipt.Decision)
		}
		if !receipt.TelemetryLogged {
			t.Error("expected TelemetryLogged to be true")
		}
	})

	t.Run("discrepancy detection when expected != actual route or state", func(t *testing.T) {
		_, clone := setupTestOriginAndClone(t)
		ctx := context.Background()
		logPath := filepath.Join(t.TempDir(), "discrepancies.jsonl")

		// Clone is initially in-sync with remote (RouteNoop, StateInSync).
		// Expect RouteApply and StateBehind to trigger both discrepancies.
		opts := ShadowOptions{
			Repo:          clone,
			Remote:        "origin",
			Branch:        "work",
			LogPath:       logPath,
			ExpectedRoute: RouteApply,
			ExpectedState: StateBehind,
			Shadow:        true,
		}

		receipt, err := AuditShadowLanding(ctx, clone, opts)
		if err != nil {
			t.Fatalf("AuditShadowLanding failed: %v", err)
		}
		if receipt.DiscrepancyCount != 2 {
			t.Fatalf("expected 2 discrepancies, got %d", receipt.DiscrepancyCount)
		}
		if receipt.Decision != DecisionDiscrepancyDetected {
			t.Errorf("expected decision DISCREPANCY_DETECTED, got %q", receipt.Decision)
		}
		if !receipt.TelemetryLogged {
			t.Errorf("expected TelemetryLogged to be true, got false (err: %s)", receipt.TelemetryError)
		}

		logger := NewCanaryDiscrepancyLogger(clone, logPath)
		discs, err := logger.ReadDiscrepancies()
		if err != nil {
			t.Fatalf("failed to read discrepancies: %v", err)
		}
		if len(discs) != 2 {
			t.Fatalf("expected 2 logged discrepancies, got %d", len(discs))
		}

		var foundRouteMismatch, foundStateMismatch bool
		for _, d := range discs {
			if d.Kind == DiscrepancyKindRouteMismatch {
				foundRouteMismatch = true
				if d.Expected != RouteApply || d.Actual != RouteNoop {
					t.Errorf("route mismatch values: expected %s, actual %s", d.Expected, d.Actual)
				}
			}
			if d.Kind == DiscrepancyKindStateMismatch {
				foundStateMismatch = true
				if d.Expected != StateBehind || d.Actual != StateInSync {
					t.Errorf("state mismatch values: expected %s, actual %s", d.Expected, d.Actual)
				}
			}
		}
		if !foundRouteMismatch {
			t.Error("did not find route_mismatch in log")
		}
		if !foundStateMismatch {
			t.Error("did not find state_mismatch in log")
		}
	})

	t.Run("fail-open behavior when log path is unwritable", func(t *testing.T) {
		_, clone := setupTestOriginAndClone(t)
		ctx := context.Background()

		// An existing directory path causes os.OpenFile(..., O_WRONLY) to fail.
		unwritableDir := t.TempDir()

		opts := ShadowOptions{
			Repo:          clone,
			Remote:        "origin",
			Branch:        "work",
			LogPath:       unwritableDir,
			ExpectedRoute: RouteApply, // force a discrepancy to trigger logging attempt
			Shadow:        true,
		}

		receipt, err := AuditShadowLanding(ctx, clone, opts)
		if err != nil {
			t.Fatalf("AuditShadowLanding must fail open, but returned err: %v", err)
		}
		if receipt == nil {
			t.Fatal("receipt must not be nil")
		}
		if receipt.TelemetryLogged {
			t.Error("expected TelemetryLogged to be false when log path unwritable")
		}
		if receipt.TelemetryError == "" {
			t.Error("expected non-empty TelemetryError")
		}
		if receipt.DiscrepancyCount == 0 {
			t.Error("expected discrepancies to still be detected and recorded on receipt")
		}
	})

	t.Run("Apply integrates shadow audit when env set", func(t *testing.T) {
		_, clone := setupTestOriginAndClone(t)
		ctx := context.Background()

		t.Setenv(EnvSafesyncShadow, "1")

		info, err := Apply(ctx, Options{
			Repo:   clone,
			Remote: "origin",
			Branch: "work",
		})
		if err != nil {
			t.Fatalf("Apply failed: %v", err)
		}
		if !info.OK || info.State != StateInSync {
			t.Fatalf("expected in-sync, got %+v", info)
		}
	})
}
