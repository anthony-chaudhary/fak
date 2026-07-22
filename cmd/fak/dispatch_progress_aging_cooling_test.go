package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
)

// TestProgressAgingFoldPausesCoolingUnit is the caller-level witness for #3715: a ready unit
// the live dispatch path is currently cooling (a recorded worker attempt inside the
// recently-attempted screen's window) folds with a PAUSED wait clock — the ineligible span is
// subtracted, so it does not read as starved — while an identical never-attempted unit is
// untouched and starves on the same raw wait (the no-regression witness). Before the overlay
// landed, both units starved: the fold dropped the cooldown window the runs dir already knew.
func TestProgressAgingFoldPausesCoolingUnit(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchProgressRunsDir)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatalf("mkdir runs dir: %v", err)
	}
	now := time.Unix(1700000000, 0).UTC()

	// Issue #123 was attempted 30 min ago: inside the default 120-min screen, so the live
	// dispatch path holds it — that ineligible span must pause its starvation clock.
	logPath := filepath.Join(runsDir, "resolve-123-20260701-010000.log")
	if err := os.WriteFile(logPath, []byte("attempt"), 0o644); err != nil {
		t.Fatalf("write attempt log: %v", err)
	}
	attempt := now.Add(-30 * time.Minute)
	if err := os.Chtimes(logPath, attempt, attempt); err != nil {
		t.Fatalf("chtime attempt log: %v", err)
	}

	// Both units became ready 6h10m ago — past the 6h starvation deadline on raw wait alone.
	ready := now.Add(-(6*time.Hour + 10*time.Minute)).Unix()
	candsPath := filepath.Join(root, "ready.json")
	cands := fmt.Sprintf(`[{"id":"123","base_weight":60,"ready_since":%d},{"id":"456","base_weight":60,"ready_since":%d}]`, ready, ready)
	if err := os.WriteFile(candsPath, []byte(cands), 0o644); err != nil {
		t.Fatalf("write candidates: %v", err)
	}

	fold, err := dispatchProgressFoldAging(root, candsPath, now)
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	// #456 (never attempted) starves on its 6h10m raw wait; #123's clock paused over its 30-min
	// cooling overlap (wait 5h40m < 6h), so exactly one unit is starved and one is aging.
	if got := fold["starved_count"]; got != 1 {
		t.Errorf("starved_count = %v, want 1 (only the never-attempted unit)", got)
	}
	if got := fold["aging_count"]; got != 1 {
		t.Errorf("aging_count = %v, want 1 (the cooling unit, paused below the deadline)", got)
	}
	if got := fold["oldest_wait_seconds"]; got != int64(6*3600+600) {
		t.Errorf("oldest_wait_seconds = %v, want %d (the non-cooling unit's full wait)", got, 6*3600+600)
	}
}

// TestOverlayLiveCoolingWindows pins the overlay's edge rules: a candidate that already
// declares a window keeps it (the supplied input wins over the derived one), a "#N"-prefixed
// id still matches its issue row, and a non-numeric id is untouched.
func TestOverlayLiveCoolingWindows(t *testing.T) {
	rows := []dispatchCooldownRow{
		{Issue: 7, LastAttemptUnix: 1000, NextEligibleUnix: 8200, Cooling: true},
		{Issue: 9, LastAttemptUnix: 2000, NextEligibleUnix: 9200, Cooling: true},
	}
	cands := []dispatchaging.Candidate{
		{ID: "7", CoolingSince: 500, CoolingUntil: 600}, // declared window wins
		{ID: "#9"},     // prefixed id still matches
		{ID: "task-x"}, // non-numeric: untouched
	}
	overlayLiveCoolingWindows(cands, rows)
	if cands[0].CoolingSince != 500 || cands[0].CoolingUntil != 600 {
		t.Errorf("declared window overwritten: got [%d, %d], want [500, 600]", cands[0].CoolingSince, cands[0].CoolingUntil)
	}
	if cands[1].CoolingSince != 2000 || cands[1].CoolingUntil != 9200 {
		t.Errorf("#9 window = [%d, %d], want [2000, 9200]", cands[1].CoolingSince, cands[1].CoolingUntil)
	}
	if cands[2].CoolingSince != 0 || cands[2].CoolingUntil != 0 {
		t.Errorf("non-numeric id gained a window: [%d, %d]", cands[2].CoolingSince, cands[2].CoolingUntil)
	}
}
