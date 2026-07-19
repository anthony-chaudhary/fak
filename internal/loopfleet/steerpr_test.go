// Liveness fold QA for the steerpr overlay-maintenance loop's ledger (#5023):
// docs/nightrun/steerpr-overlay.jsonl folds into the cross-ledger pane like
// any other loop — last tick from `ts`, kept = the row's no-silent-drop
// invariant holds (commits_seen == assigned + orphans), witnessed = the row
// carries the external witness binding loopgate admitted the tick on.
package loopfleet

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func steerprLedgerWrite(t *testing.T, root, body string) {
	t.Helper()
	p := filepath.Join(root, "docs", "nightrun", "steerpr-overlay.jsonl")
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func steerprFindLoop(rep Report) (LoopHealth, bool) {
	for _, l := range rep.Loops {
		if l.Kind == "steerpr-overlay" {
			return l, true
		}
	}
	return LoopHealth{}, false
}

func TestFoldSteerprOverlayLedgerLiveness(t *testing.T) {
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	th := loopmgr.HealthThresholds{DefaultCadenceSeconds: 3600, DarkMultiple: 2}
	root := t.TempDir()

	older := now.Add(-20 * time.Hour).UTC().Format(time.RFC3339)
	fresh := now.Add(-10 * time.Minute).UTC().Format(time.RFC3339)
	// Row 1: invariant BROKEN (5 != 3+1) and no witness -> not kept, not witnessed.
	// Row 2: fresh, invariant holds, witness bound -> kept + witnessed, and it
	// sets the last tick that makes the loop LIVE against the daily cadence.
	steerprLedgerWrite(t, root,
		`{"schema":"fak.steerpr-overlay.v1","ts":"`+older+`","base":"a1","head":"b2","commits_seen":5,"units_total":2,"assigned":3,"residual":1,"cleared":1,"unverifiable":0,"orphans":1}`+"\n"+
			`{"schema":"fak.steerpr-overlay.v1","ts":"`+fresh+`","base":"b2","head":"c3","commits_seen":4,"units_total":2,"assigned":3,"residual":0,"cleared":2,"unverifiable":0,"orphans":1,"witness":"dos commit-audit --json b2..c3"}`+"\n")

	rep := Fold(root, now, th)
	row, ok := steerprFindLoop(rep)
	if !ok {
		t.Fatalf("steerpr-overlay loop not folded; loops=%+v skipped=%+v", rep.Loops, rep.Skipped)
	}
	if row.Ledger != "steerpr-overlay" {
		t.Errorf("ledger = %q, want steerpr-overlay", row.Ledger)
	}
	if row.State != loopmgr.HealthLive || row.Dark {
		t.Errorf("state = %q dark=%v, want live/false (fresh tick within cadence)", row.State, row.Dark)
	}
	if row.Runs != 2 || row.Keep != 1 || row.Witness != 1 {
		t.Errorf("runs/keep/witness = %d/%d/%d, want 2/1/1", row.Runs, row.Keep, row.Witness)
	}
}

// A missing ledger is skipped-and-surfaced, never silent and never fatal —
// the same contract every other adapter honors.
func TestFoldSteerprOverlayAbsentLedgerIsSurfaced(t *testing.T) {
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	rep := Fold(t.TempDir(), now, loopmgr.HealthThresholds{})
	if _, ok := steerprFindLoop(rep); ok {
		t.Fatal("absent ledger still produced a steerpr-overlay loop row")
	}
	want := filepath.Join("docs", "nightrun", "steerpr-overlay.jsonl")
	for _, s := range rep.Skipped {
		if s.Ledger == "steerpr-overlay" {
			if s.Path != want {
				t.Errorf("skipped path = %q, want %q", s.Path, want)
			}
			if s.Reason != "absent" {
				t.Errorf("skipped reason = %q, want absent", s.Reason)
			}
			return
		}
	}
	t.Fatalf("steerpr-overlay not surfaced in skipped; skipped=%+v", rep.Skipped)
}
