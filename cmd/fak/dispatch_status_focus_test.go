package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// TestDispatchStatusFocusOverCap: `fak dispatch status` surfaces the focus WIP-breadth
// posture, labeled distinctly from the rate-limit/collision holds, when the trajctl ledger
// is at/over the WIP cap.
func TestDispatchStatusFocusOverCap(t *testing.T) {
	root := t.TempDir()
	runsDir := t.TempDir()
	plantFocusLedger(t, root, []trajctl.Objective{
		{ID: "alpha", Statement: "resolve #100", Status: trajctl.StatusActive},
		{ID: "beta", Statement: "resolve #200", Status: trajctl.StatusActive},
		{ID: "gamma", Statement: "resolve #300", Status: trajctl.StatusActive},
		{ID: "delta", Statement: "resolve #400", Status: trajctl.StatusActive},
	})

	out, errb, code := runDispatchAt("status", "--runs-dir", runsDir, "--workspace", root, "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var snap dispatchStatusSnapshot
	if err := json.Unmarshal([]byte(out), &snap); err != nil {
		t.Fatalf("decode json: %v\n%s", err, out)
	}
	if snap.Focus == nil {
		t.Fatalf("focus section missing from status snapshot: %s", out)
	}
	if snap.Focus.Active != 4 || snap.Focus.WIPCap != 3 || snap.Focus.ExcessWIP != 1 {
		t.Fatalf("focus active/cap/excess = %d/%d/%d, want 4/3/1", snap.Focus.Active, snap.Focus.WIPCap, snap.Focus.ExcessWIP)
	}
	if !snap.Focus.Saturated || snap.Focus.Posture != dispatchtick.FocusPostureWarn {
		t.Fatalf("focus saturated/posture = %v/%q, want true/warn", snap.Focus.Saturated, snap.Focus.Posture)
	}

	// Human card names the focus posture distinctly and cites the closed token.
	human, _, hcode := runDispatchAt("status", "--runs-dir", runsDir, "--workspace", root)
	if hcode != 0 {
		t.Fatalf("human exit=%d", hcode)
	}
	if !strings.Contains(human, "focus:") || !strings.Contains(human, dispatchtick.FocusWIPSaturated) {
		t.Fatalf("human status card missing distinct focus line:\n%s", human)
	}
}

// TestDispatchStatusFocusAbsentWithoutLedger: with no trajctl ledger the focus section is
// omitted, so the snapshot stays byte-identical to today.
func TestDispatchStatusFocusAbsentWithoutLedger(t *testing.T) {
	root := t.TempDir() // no ledger planted
	runsDir := t.TempDir()

	out, _, code := runDispatchAt("status", "--runs-dir", runsDir, "--workspace", root, "--json")
	if code != 0 {
		t.Fatalf("exit=%d", code)
	}
	var snap dispatchStatusSnapshot
	if err := json.Unmarshal([]byte(out), &snap); err != nil {
		t.Fatalf("decode json: %v\n%s", err, out)
	}
	if snap.Focus != nil {
		t.Fatalf("focus section present without a ledger, want nil: %#v", snap.Focus)
	}
}
