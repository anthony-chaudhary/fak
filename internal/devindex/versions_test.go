package devindex

import (
	"os"
	"path/filepath"
	"testing"
)

// TestJoinModuleVersions is the #2465 witness: Load folds the fak-module-versions/1
// ledger onto the leaves so a leaf carries its "r<rev>+g<sha>" version — the done
// condition that fak_index_leaves rows expose module versions. It also pins the
// append-only last-row-wins rule and graceful degradation for a leaf with no row.
func TestJoinModuleVersions(t *testing.T) {
	root := t.TempDir()
	dosToml := `[lanes.trees]
gateway = ["internal/gateway/**"]
session = ["internal/session/**"]
docs    = ["docs/**"]
`
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}
	// Two rows for internal/gateway — the LAST (r5) must win over the earlier r2
	// (the ledger is append-only). internal/session has one row; docs has none, so
	// its leaf must stay version-less. A malformed line must be skipped, never fatal.
	ledger := `{"schema":"fak-module-versions/1","module":"internal/gateway","rev":2,"version":"r2+gdeadbeef"}
{"schema":"fak-module-versions/1","module":"internal/session","rev":3,"version":"r3+gcafef00d"}
{"schema":"fak-module-versions/1","module":"internal/gateway","rev":5,"version":"r5+gabcd1234"}
not-valid-json — a bad row must not break the leaf map
`
	if err := os.MkdirAll(filepath.Join(root, "docs", "nightrun"), 0o755); err != nil {
		t.Fatal(err)
	}
	ledgerPath := filepath.Join(root, "docs", "nightrun", "module-versions.jsonl")
	if err := os.WriteFile(ledgerPath, []byte(ledger), 0o644); err != nil {
		t.Fatal(err)
	}

	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	gw, ok := c.LeafByName("gateway")
	if !ok {
		t.Fatal("gateway leaf missing")
	}
	if gw.Version != "r5+gabcd1234" {
		t.Errorf("gateway version = %q, want r5+gabcd1234 (last ledger row wins)", gw.Version)
	}

	sess, ok := c.LeafByName("session")
	if !ok {
		t.Fatal("session leaf missing")
	}
	if sess.Version != "r3+gcafef00d" {
		t.Errorf("session version = %q, want r3+gcafef00d", sess.Version)
	}

	docs, ok := c.LeafByName("docs")
	if !ok {
		t.Fatal("docs leaf missing")
	}
	if docs.Version != "" {
		t.Errorf("docs version = %q, want empty (no ledger row names its Dir)", docs.Version)
	}
}

// TestLoadWithoutLedger asserts a repo with no module-versions.jsonl loads cleanly
// with empty leaf versions — the version is a staleness hint, never load-bearing, so
// its absence is a no-op rather than an error.
func TestLoadWithoutLedger(t *testing.T) {
	c, err := Load(writeSyntheticRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	gw, ok := c.LeafByName("gateway")
	if !ok {
		t.Fatal("gateway leaf missing")
	}
	if gw.Version != "" {
		t.Errorf("gateway version = %q, want empty when no ledger is present", gw.Version)
	}
}
