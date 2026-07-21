package modver

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"testing"
)

// TestParseGhosts is the #2477 witness at the pure-parse seam: the SAME history
// fixture parseLog folds into the live report (logFixture + liveFixture) yields
// the deleted module as a ghost — history-only, with its final rev and the commit
// that removed it — while the two live modules never ghost. The live report and
// the tombstone report are exact complements over one history walk.
func TestParseGhosts(t *testing.T) {
	ghosts := parseGhosts([]byte(logFixture), liveFixture())
	if len(ghosts) != 1 {
		t.Fatalf("got %d ghosts, want 1 (internal/deleted): %+v", len(ghosts), ghosts)
	}
	g := ghosts[0]
	// internal/deleted was touched once, in commit bbb22222 @2026-07-01 — its only
	// and therefore last (deletion) touch. Final rev 1.
	if g.Name != "internal/deleted" || g.Kind != "internal" || g.Rev != 1 {
		t.Errorf("ghost = %+v, want internal/deleted kind=internal rev=1", g)
	}
	if g.DeletedCommit != "bbb22222" || g.DeletedDate != "2026-07-01T09:00:00Z" {
		t.Errorf("ghost deletion = %s @%s, want bbb22222 @2026-07-01T09:00:00Z", g.DeletedCommit, g.DeletedDate)
	}
	if v := g.Version(); v != "r1+gbbb22222" {
		t.Errorf("Version() = %q, want r1+gbbb22222", v)
	}
	// A live module must never appear in the tombstone report.
	for _, gg := range ghosts {
		if gg.Name == "internal/gateway" || gg.Name == "cmd/fak" {
			t.Errorf("live module %q ghosted into the tombstone report", gg.Name)
		}
	}
}

// TestGhostsRealRepo is the #2477 done-condition witness: the ghost listing
// renders over a REAL repo. A module is created, modified, then fully deleted;
// Ghosts must report it with the final rev it reached (every commit that touched
// it, deletion included) and the deletion commit, while Snapshot — the live
// report — must NOT list it, and a still-live sibling module must NOT ghost. The
// two reports are exact complements over the same git history.
func TestGhostsRealRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	repo := modverGitRepo(t)
	commitFileMV(t, repo, "internal/foo/a.go", "package foo\n// c1\n", "c1") // internal/foo #1 (create)
	commitFileMV(t, repo, "internal/foo/a.go", "package foo\n// c2\n", "c2") // internal/foo #2 (modify)
	commitFileMV(t, repo, "internal/bar/b.go", "package bar\n", "b1")        // internal/bar: stays live
	// Delete internal/foo entirely: its only file is removed, so the module is
	// absent at HEAD — a ghost. This commit is its deletion.
	gitMV(t, repo, "rm", "-q", "internal/foo/a.go")
	gitMV(t, repo, "commit", "-q", "-m", "rm foo")
	// Derive the expected deletion SHA with the same %h abbreviation the module
	// log uses (git auto-abbreviates to the shortest unambiguous length, which is
	// not necessarily rev-parse --short=8's fixed 8).
	delSHA := strings.TrimSpace(string(mustGitMV(t, repo, "log", "-1", "--pretty=format:%h")))

	ghosts, err := Ghosts(context.Background(), repo, RealRunner)
	if err != nil {
		t.Fatal(err)
	}
	if len(ghosts) != 1 {
		t.Fatalf("got %d ghosts, want 1 (internal/foo): %+v", len(ghosts), ghosts)
	}
	g := ghosts[0]
	if g.Name != "internal/foo" || g.Kind != "internal" {
		t.Fatalf("ghost = %+v, want internal/foo kind=internal", g)
	}
	if g.Rev != 3 {
		t.Errorf("internal/foo rev = %d, want 3 (create, modify, delete all touch it)", g.Rev)
	}
	if g.DeletedCommit != delSHA {
		t.Errorf("deletion commit = %s, want %s (the commit that removed the last file)", g.DeletedCommit, delSHA)
	}
	if g.DeletedDate == "" {
		t.Errorf("deletion date empty: %+v", g)
	}
	if want := fmt.Sprintf("r3+g%s", delSHA); g.Version() != want {
		t.Errorf("Version() = %q, want %q", g.Version(), want)
	}

	// Complement invariant: the live Snapshot lists the live sibling and NOT the
	// ghost; the ghost report lists the ghost and NOT the live sibling.
	rep, err := Snapshot(context.Background(), repo, RealRunner)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range rep.Modules {
		if m.Name == "internal/foo" {
			t.Errorf("deleted module internal/foo leaked into the live Snapshot report")
		}
	}
	findModuleMV(t, rep, "internal/bar") // must be live
	for _, gg := range ghosts {
		if gg.Name == "internal/bar" {
			t.Errorf("live module internal/bar ghosted into the tombstone report")
		}
	}
}
