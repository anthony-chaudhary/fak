package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These pin `fak resume drive <uuid>`, the read-only operator lens onto the drive-CARRY channel
// (#4135): it folds the same durable resume_drivestate.jsonl the watchdog reads and prints the
// drive-state a relaunch would RESTORE for one transcript UUID — carried budget for a uuid with a
// stored carry, a "would come up fresh" line for one with none — and it NEVER writes a ledger row
// (the whole promise is that inspecting the carry channel is side-effect-free).

func TestResumeDriveRendersCarryReadOnly(t *testing.T) {
	regDir := t.TempDir()
	sid := "sid-carry-1234567890"

	// One carry row for sid, written straight to the store — the carry PRODUCER is a sibling
	// cluster; this verb only READS what the fold exposes.
	row := `{"ts":"2026-07-10T12:00:00Z","session":"` + sid + `","turns_left":42,"tokens_left":51000,"context_tokens_left":120000,"spend_micro_cents_left":250,"generation":3,"objective_pin_id":"pin-1","objective_text":"ship #1234"}`
	if err := os.WriteFile(filepath.Join(regDir, "resume_drivestate.jsonl"), []byte(row+"\n"), 0o644); err != nil {
		t.Fatalf("seed drive-state store: %v", err)
	}
	// The launch ledger must be BYTE-IDENTICAL before/after — the inspector is read-only.
	ledgerPath := filepath.Join(regDir, "resume_ledger.jsonl")
	if err := os.WriteFile(ledgerPath, []byte(`{"session":"other","phase":"launched"}`+"\n"), 0o644); err != nil {
		t.Fatalf("seed launch ledger: %v", err)
	}
	before, _ := os.ReadFile(ledgerPath)

	var out, errb bytes.Buffer
	if rc := runResumeDrive(&out, &errb, []string{"--reg-dir", regDir, sid}); rc != 0 {
		t.Fatalf("drive rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"turns_left=42", "tokens_left=51000", "generation=3"} {
		if !strings.Contains(got, want) {
			t.Fatalf("readout missing carried field %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "launched") {
		t.Fatalf("read-only inspector must not print launch text:\n%s", got)
	}

	after, _ := os.ReadFile(ledgerPath)
	if !bytes.Equal(before, after) {
		t.Fatalf("launch ledger changed — the inspector wrote a row:\nbefore=%s\nafter=%s", before, after)
	}

	// --json emits the folded record with the carry payload.
	out.Reset()
	errb.Reset()
	if rc := runResumeDrive(&out, &errb, []string{"--reg-dir", regDir, "--json", sid}); rc != 0 {
		t.Fatalf("drive --json rc=%d stderr=%s", rc, errb.String())
	}
	var rec driveReadout
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("drive --json is not valid JSON: %v\n%s", err, out.String())
	}
	if rec.UUID != sid || !rec.HasCarry || rec.Carry == nil {
		t.Fatalf("json readout missing carry: %+v", rec)
	}
	if rec.Carry.TurnsLeft != 42 || rec.Carry.TokensLeft != 51000 || rec.Carry.Generation != 3 {
		t.Fatalf("json carry fields wrong: %+v", rec.Carry)
	}
}

func TestResumeDriveFreshWhenNoRecord(t *testing.T) {
	regDir := t.TempDir() // empty — no store at all
	var out, errb bytes.Buffer
	if rc := runResumeDrive(&out, &errb, []string{"--reg-dir", regDir, "sid-unknown-000"}); rc != 0 {
		t.Fatalf("drive on an unknown uuid rc=%d, want 0 (a clean read, not a not-found): stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "come up fresh") {
		t.Fatalf("unknown uuid must report the fresh-relaunch line:\n%s", out.String())
	}
}

func TestResumeDriveNeedsUUID(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runResumeDrive(&out, &errb, nil); rc != 2 {
		t.Fatalf("missing uuid must be a usage error (2), got %d", rc)
	}
}

// TestResumeDriveHoldAndCarry pins that a session carrying BOTH an operator hold and a carry row
// renders the hold token and the carried budget together — the fold reads both axes, and a
// carry-only row (no state token) never clobbers the prior hold.
func TestResumeDriveHoldAndCarry(t *testing.T) {
	regDir := t.TempDir()
	sid := "sid-both-1234567890"

	var out, errb bytes.Buffer
	// Write an operator hold via the real writer verb...
	if rc := runResumeHold(&out, &errb, []string{"--reg-dir", regDir, "--state", "paused", sid}); rc != 0 {
		t.Fatalf("hold rc=%d stderr=%s", rc, errb.String())
	}
	// ...then append a carry row (no state token) for the same sid.
	f, err := os.OpenFile(filepath.Join(regDir, "resume_drivestate.jsonl"), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open store to append carry: %v", err)
	}
	if _, err := f.WriteString(`{"session":"` + sid + `","tokens_left":9000}` + "\n"); err != nil {
		t.Fatalf("append carry: %v", err)
	}
	f.Close()

	out.Reset()
	errb.Reset()
	if rc := runResumeDrive(&out, &errb, []string{"--reg-dir", regDir, sid}); rc != 0 {
		t.Fatalf("drive rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "paused") {
		t.Fatalf("readout missing the operator hold token:\n%s", got)
	}
	if !strings.Contains(got, "tokens_left=9000") {
		t.Fatalf("readout missing the carried budget:\n%s", got)
	}
}
