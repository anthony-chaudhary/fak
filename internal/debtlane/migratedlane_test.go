package debtlane

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDiscoverLanesSuppressesMigratedAwayUnit is the regression for
// fak issue #13150: a dos.toml lane whose declared tree is a Go source prefix
// ("internal/<lane>/**") but whose unit was migrated away (the package now
// lives at platform/<other>/... and no internal/<lane> exists on disk) must
// NOT be emitted as a scored debt lane. Pre-fix, discoverLanesFromDisk
// synthesized the nonexistent internal/<lane> path and the scanner reported
// its absence as five debt dimensions (coverage/test/wiring/proof/benchmark),
// ranking it as the top maturity gap ahead of real, actionable debt.
//
// A declared Go-prefixed tree whose on-disk root DOES exist must still be
// discovered (positive control), so the suppression cannot over-reach.
func TestDiscoverLanesSuppressesMigratedAwayUnit(t *testing.T) {
	tmp := t.TempDir()

	pkgSrc := "package fixturepkg\n\n// Fixture is a minimal valid package for discovery.\ntype Fixture struct{}\n"

	// The migrated-away unit: source now lives under platform/cluster/flowcredit,
	// but the stale dos.toml lane still declares internal/flowcredit/**.
	migratedDir := filepath.Join(tmp, "platform", "cluster", "flowcredit")
	if err := os.MkdirAll(migratedDir, 0o755); err != nil {
		t.Fatalf("mkdir migrated package: %v", err)
	}
	if err := os.WriteFile(filepath.Join(migratedDir, "flowcredit.go"), []byte(pkgSrc), 0o644); err != nil {
		t.Fatalf("write migrated package: %v", err)
	}

	// Positive control: a real internal/<lane> that matches its declared tree.
	realDir := filepath.Join(tmp, "internal", "realpkg")
	if err := os.MkdirAll(realDir, 0o755); err != nil {
		t.Fatalf("mkdir real package: %v", err)
	}
	if err := os.WriteFile(filepath.Join(realDir, "realpkg.go"), []byte(pkgSrc), 0o644); err != nil {
		t.Fatalf("write real package: %v", err)
	}

	dosToml := "[lanes]\n\n[lanes.trees]\n" +
		"flowcredit = [\"internal/flowcredit/**\"]\n" +
		"realpkg = [\"internal/realpkg/**\"]\n"
	if err := os.WriteFile(filepath.Join(tmp, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatalf("write dos.toml: %v", err)
	}

	lanes, err := discoverLanesFromDisk(tmp)
	if err != nil {
		t.Fatalf("discoverLanesFromDisk returned error: %v", err)
	}

	for _, l := range lanes {
		if strings.EqualFold(l.Lane, "flowcredit") {
			t.Fatalf("phantom debt lane emitted for migrated-away unit: lane=%q unit=%q next_action=%q",
				l.Lane, l.UnitOfWork, l.NextAction)
		}
		if _, err := os.Stat(filepath.Join(tmp, l.UnitOfWork)); err != nil {
			t.Errorf("emitted lane %q has no on-disk unit %q: %v", l.Lane, l.UnitOfWork, err)
		}
	}

	foundReal := false
	for _, l := range lanes {
		if strings.EqualFold(l.Lane, "realpkg") {
			foundReal = true
			break
		}
	}
	if !foundReal {
		t.Errorf("expected realpkg lane (declared Go-prefixed tree with on-disk root); got %d lanes", len(lanes))
	}
}
