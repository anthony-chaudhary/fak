package fleetaccounts

import (
	"os"
	"path/filepath"
	"testing"
)

// TestDiscoverExcludesFaklocalDogfoodHomes pins that the fak-kernel dogfood config
// homes (.claude-faklocal / .claude-faklocal-netra) are kept OFF the switcher roster
// even when they carry a projects/ subdir (a real dogfood session ran there). They are
// synthesized on demand by Resolve(--faklocal-ok), not enrolled accounts, so surfacing
// them as credential-less needs_login rows only clutters the operator roster. The
// exclusion must not be over-broad: a normal seat alongside them stays a worker.
func TestDiscoverExcludesFaklocalDogfoodHomes(t *testing.T) {
	home := t.TempDir()
	cfg := t.TempDir()
	// Isolate from the operator's real registry so a live tombstone can't skew the row.
	t.Setenv("FAK_ACCOUNTS_REGISTRY", filepath.Join(home, "no-registry.json"))

	for _, dir := range []string{
		".claude-faklocal",       // used dogfood home (projects/ present) — still excluded
		".claude-faklocal-netra", // the -netra dogfood variant — same prefix, also excluded
		".claude-gem8-acct",      // a normal seat that MUST remain offered
	} {
		if err := os.MkdirAll(filepath.Join(home, dir, "projects"), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	rows := Discover(home, cfg, DefaultPolicy())

	for _, acct := range []string{".claude-faklocal", ".claude-faklocal-netra"} {
		r := find(rows, acct)
		if r == nil {
			t.Fatalf("%s not discovered at all", acct)
		}
		if r.Kind != KindExcluded {
			t.Errorf("%s: kind=%q want %q (reason=%q)", acct, r.Kind, KindExcluded, r.Reason)
		}
	}
	if g := find(rows, ".claude-gem8-acct"); g == nil || g.Kind != KindWorker {
		t.Errorf(".claude-gem8-acct should remain a worker, got %+v", g)
	}
}
