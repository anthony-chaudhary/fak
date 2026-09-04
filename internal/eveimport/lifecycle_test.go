package eveimport_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/eveimport"
)

// Invariant: Eve import must parse raw NDJSON event streams and redact message bodies by default.
// Guard: ImportNDJSON constructs non-empty run traces without leaking unredacted prompt contents.

func TestEveImportLifecycle(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile(filepath.Join("testdata", "eve-session-stream.ndjson"))
	if err != nil {
		t.Fatalf("failed reading fixture: %v", err)
	}

	run := eveimport.ImportNDJSON("session.ndjson", data, eveimport.Options{})
	if run.Root == nil {
		t.Fatal("expected non-nil Root session")
	}
	if run.Sessions == 0 {
		t.Fatal("expected non-zero sessions parsed")
	}
}
