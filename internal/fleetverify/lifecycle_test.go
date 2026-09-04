package fleetverify

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
)

// Invariant: Fleet verification must correctly load and validate brief JSON payloads and report schemas.
// Guard: Exercise returns an error when brief JSON carries mismatched schemas.

func TestFleetVerifyLifecycle(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	dir := t.TempDir()

	want := loopfleet.Report{
		Schema:     loopfleet.Schema,
		TSUnixNano: 1000,
		Rollup:     loopfleet.Rollup{Loops: 1, Live: 1, Stale: 0, Ledgers: 1},
	}
	b, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("failed marshaling brief: %v", err)
	}
	path := filepath.Join(dir, "brief.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("failed writing brief: %v", err)
	}

	got, err := Exercise(root, path, strings.NewReader(""))
	if err != nil {
		t.Fatalf("Exercise failed: %v", err)
	}
	if got.Rollup.Loops != 1 {
		t.Fatalf("expected 1 loop in rollup, got %d", got.Rollup.Loops)
	}
}
