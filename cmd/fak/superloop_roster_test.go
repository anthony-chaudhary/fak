package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// TestSuperloopRosterFoldsWorkspaceIntoCanonicalRoster drives the read surface
// end to end (#4955): a workspace with one live cadence ledger folds into a
// roster where the ledgered loop appears exactly once (measured), the registered
// super loops are listed, and every unfoldable ledger surfaces as a KNOWN gap.
func TestSuperloopRosterFoldsWorkspaceIntoCanonicalRoster(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "docs", "cadence")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	row := `{"generated_at":"` + time.Now().UTC().Format(time.RFC3339) + `","verdict":"OK","commit":"abc123"}` + "\n"
	if err := os.WriteFile(filepath.Join(dir, "history.jsonl"), []byte(row), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if rc := runRoster(&stdout, &stderr, []string{"--json", "--workspace", root}); rc != 0 {
		t.Fatalf("runRoster rc = %d, stderr: %s", rc, stderr.String())
	}
	var r superloop.Roster
	if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
		t.Fatalf("decode roster JSON: %v\n%s", err, stdout.String())
	}
	if r.Schema != superloop.RosterSchema {
		t.Errorf("schema = %q, want %q", r.Schema, superloop.RosterSchema)
	}

	count := 0
	var cadence superloop.RosterEntry
	supers := 0
	for _, e := range r.Entries {
		if e.ID == "cadence" {
			count++
			cadence = e
		}
		if e.Kind == superloop.KindSuperloop {
			supers++
		}
	}
	if count != 1 {
		t.Fatalf("ledgered cadence loop appears %d time(s), want exactly once", count)
	}
	if !cadence.Measured {
		t.Errorf("cadence folded from a real ledger, want Measured=true")
	}
	if !cadence.Named {
		t.Errorf("cadence is hand-named by improve-loops, want Named=true")
	}
	if supers != len(superloop.Registry()) {
		t.Errorf("roster lists %d super loops, want every registered intent (%d)", supers, len(superloop.Registry()))
	}
	// The other ledgers do not exist in this workspace: they must surface as
	// KNOWN gaps, never vanish into a healthy zero.
	if len(r.Gaps) == 0 {
		t.Errorf("absent ledgers surfaced no gaps — a missing ledger must be a known gap")
	}
	if r.Total != len(r.Entries) {
		t.Errorf("rollup Total = %d, want %d", r.Total, len(r.Entries))
	}
}
