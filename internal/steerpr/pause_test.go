package steerpr

// Tests for the pause/resume ledger leaf (#5031): row validation refuses the
// unholdable, the ledger round-trips append-only, and the active fold is the
// leaf's hold state — pause sets it, resume clears it, and both verbs ship
// together so a pause can never become a silent leak.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var steerPauseT0 = time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)

// A pause is a specific person stopping the spend on a specific bound intent:
// a row missing the unit, the who, or the bound #N issue holds nothing and is
// refused. The note is optional — "-m" is the reason, not the mechanism.
func TestSteerPauseRowValidation(t *testing.T) {
	if _, err := NewPause("", "op", "", "#5", steerPauseT0); err == nil {
		t.Error("an unnamed pause was accepted")
	}
	if _, err := NewPause("gateway", "", "", "#5", steerPauseT0); err == nil {
		t.Error("an unattributable pause was accepted")
	}
	for _, badIssue := range []string{"", "5", "#", "#x", "#0", "#-3"} {
		if _, err := NewPause("gateway", "op", "", badIssue, steerPauseT0); err == nil {
			t.Errorf("a pause bound to %q was accepted — nothing for the dispatch loop to hold", badIssue)
		}
	}
	p, err := NewPause("gateway", "op-jane", "clearly the wrong shape", "#1146", steerPauseT0)
	if err != nil {
		t.Fatalf("valid pause refused: %v", err)
	}
	if p.Schema != PauseSchema || p.Action != PauseActionPause || p.IssueNumber() != 1146 || p.At == "" {
		t.Fatalf("pause row = %#v, want a bound fak.steerpr.pause.v1 pause", p)
	}
	if noteless, err := NewPause("gateway", "op", "", "#1146", steerPauseT0); err != nil || noteless.Note != "" {
		t.Fatalf("a noteless pause must be valid (the reason is optional): %v %#v", err, noteless)
	}

	if _, err := NewResume("", "op", "", steerPauseT0); err == nil {
		t.Error("an unnamed resume was accepted")
	}
	if _, err := NewResume("gateway", "", "", steerPauseT0); err == nil {
		t.Error("an unattributable resume was accepted")
	}
	r, err := NewResume("gateway", "op-jane", "#1146", steerPauseT0)
	if err != nil {
		t.Fatalf("valid resume refused: %v", err)
	}
	if r.Action != PauseActionResume || r.Schema != PauseSchema {
		t.Fatalf("resume row = %#v, want a fak.steerpr.pause.v1 resume", r)
	}
}

// AppendPause refuses the rows NewPause/NewResume would refuse, so a caller
// that skips the constructor still cannot ledger an unholdable hold.
func TestSteerPauseAppendRefusesIncompleteRows(t *testing.T) {
	path := PauseLedgerPath(t.TempDir())
	for name, row := range map[string]Pause{
		"no leaf":            {Schema: PauseSchema, Action: PauseActionPause, By: "op", Issue: "#5"},
		"no by":              {Schema: PauseSchema, Action: PauseActionPause, Leaf: "gateway", Issue: "#5"},
		"bad action":         {Schema: PauseSchema, Action: "kill", Leaf: "gateway", By: "op", Issue: "#5"},
		"pause with nothing": {Schema: PauseSchema, Action: PauseActionPause, Leaf: "gateway", By: "op"},
	} {
		if err := AppendPause(path, row); err == nil {
			t.Errorf("%s: incomplete row was ledgered", name)
		}
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("refused rows still created the ledger: %v", err)
	}
}

// The hold state is the ledger fold: pause holds, resume releases, a re-pause
// holds again — and the ledger only ever grows (append-only), so the history
// of paused time stays visible after the release.
func TestSteerPauseLedgerFoldPauseResumeRepause(t *testing.T) {
	root := t.TempDir()
	path := PauseLedgerPath(root)

	pause, err := NewPause("gateway", "op-jane", "wrong shape, hold it", "#1146", steerPauseT0)
	if err != nil {
		t.Fatalf("NewPause: %v", err)
	}
	if err := AppendPause(path, pause); err != nil {
		t.Fatalf("append pause: %v", err)
	}
	if p, ok := PausedFor(LoadPauses(path), "gateway"); !ok || p.By != "op-jane" || p.Issue != "#1146" || p.At != pause.At {
		t.Fatalf("after pause, PausedFor = %#v/%v, want the appended hold", p, ok)
	}

	resume, err := NewResume("gateway", "op-jane", pause.Issue, steerPauseT0.Add(time.Hour))
	if err != nil {
		t.Fatalf("NewResume: %v", err)
	}
	if err := AppendPause(path, resume); err != nil {
		t.Fatalf("append resume: %v", err)
	}
	if _, ok := PausedFor(LoadPauses(path), "gateway"); ok {
		t.Fatal("after resume, the leaf is still held — the release path leaked")
	}
	if active := ActivePauses(LoadPauses(path)); len(active) != 0 {
		t.Fatalf("after resume, ActivePauses = %#v, want empty", active)
	}

	repause, err := NewPause("gateway", "op-omar", "", "#1146", steerPauseT0.Add(2*time.Hour))
	if err != nil {
		t.Fatalf("NewPause (again): %v", err)
	}
	if err := AppendPause(path, repause); err != nil {
		t.Fatalf("append re-pause: %v", err)
	}
	rows := LoadPauses(path)
	if p, ok := PausedFor(rows, "gateway"); !ok || p.By != "op-omar" || p.At != repause.At {
		t.Fatalf("after re-pause, PausedFor = %#v/%v, want the NEWEST hold", p, ok)
	}
	// Append-only: all three events are still on the ledger, oldest first.
	if len(rows) != 3 || rows[0].Action != PauseActionPause || rows[1].Action != PauseActionResume || rows[2].Action != PauseActionPause {
		t.Fatalf("ledger rows = %#v, want the full pause/resume/pause history", rows)
	}
}

// A torn line, a foreign row, and an unbound pause row are skipped rather
// than poisoning the fold — failure never invents (or releases) a hold. Holds
// on different leaves stay independent.
func TestSteerPauseLoadSkipsForeignRowsAndKeepsLeavesIndependent(t *testing.T) {
	root := t.TempDir()
	path := PauseLedgerPath(root)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	lines := strings.Join([]string{
		`{"schema":"fak.steerpr.pause.v1","action":"pause","leaf":"gateway","by":"op","at":"2026-07-18T12:00:00Z","issue":"#1146"}`,
		`{"schema":"fak.steerpr.ack.v1","leaf":"gateway","by":"op","at":"x","shas":["a"]}`,                          // foreign schema
		`{"schema":"fak.steerpr.pause.v1","action":"pause","leaf":"compute","by":"op","at":"2026-07-18T12:01:00Z"}`, // unbound: holds nothing
		`{torn`, // torn line
		`{"schema":"fak.steerpr.pause.v1","action":"resume","leaf":"docs","by":"op","at":"2026-07-18T12:02:00Z"}`, // resume of an unheld leaf: no-op
	}, "\n") + "\n"
	if err := os.WriteFile(path, []byte(lines), 0o644); err != nil {
		t.Fatal(err)
	}
	active := ActivePauses(LoadPauses(path))
	if len(active) != 1 {
		t.Fatalf("active = %#v, want only the bound gateway hold", active)
	}
	if p, ok := active["gateway"]; !ok || p.IssueNumber() != 1146 {
		t.Fatalf("gateway hold = %#v/%v, want the #1146-bound pause", p, ok)
	}
}
