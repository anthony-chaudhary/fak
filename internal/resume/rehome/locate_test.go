package rehome

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// writeTranscript creates <home>/<account>/projects/<slug>/<sid>.jsonl with the given
// mtime and byte size, returning the file path.
func writeTranscript(t *testing.T, home, account, slug, sid string, mtime time.Time, size int) string {
	t.Helper()
	dir := filepath.Join(home, account, "projects", slug)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, sid+".jsonl")
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mtime, mtime); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLocateOwnerNotFound(t *testing.T) {
	home := t.TempDir()
	if got := LocateOwner("sid-x", home); got != nil {
		t.Fatalf("LocateOwner on empty home = %+v, want nil", got)
	}
}

func TestLocateOwnerNewestMtimeHostLast(t *testing.T) {
	home := t.TempDir()
	sid := "abc123"
	base := time.Now().Add(-time.Hour)
	// Host holds the newest copy, but a non-host account also holds it -> non-host wins.
	writeTranscript(t, home, ".claude", "C--work-fak", sid, base.Add(30*time.Minute), 100)
	writeTranscript(t, home, ".claude-old", "C--work-fak", sid, base, 50)
	writeTranscript(t, home, ".claude-new", "C--work-fak", sid, base.Add(20*time.Minute), 80)

	owner := LocateOwner(sid, home)
	if owner == nil {
		t.Fatal("LocateOwner = nil, want owner")
	}
	if owner.Account != ".claude-new" {
		t.Fatalf("owner account = %q, want .claude-new (newest non-host)", owner.Account)
	}
	if owner.DupCount != 3 {
		t.Fatalf("dup_count = %d, want 3", owner.DupCount)
	}
	wantAccts := []string{".claude", ".claude-new", ".claude-old"}
	if len(owner.AllAccounts) != 3 {
		t.Fatalf("all_accounts = %v, want %v", owner.AllAccounts, wantAccts)
	}
	for i, a := range wantAccts {
		if owner.AllAccounts[i] != a {
			t.Fatalf("all_accounts[%d] = %q, want %q (sorted)", i, owner.AllAccounts[i], a)
		}
	}
	if owner.Project != "C--work-fak" {
		t.Fatalf("project = %q, want C--work-fak", owner.Project)
	}
}

func TestLocateOwnerHostSoleOwner(t *testing.T) {
	home := t.TempDir()
	sid := "solo"
	writeTranscript(t, home, ".claude", "C--work-fak", sid, time.Now(), 42)
	owner := LocateOwner(sid, home)
	if owner == nil {
		t.Fatal("LocateOwner = nil, want host as sole owner")
	}
	if owner.Account != ".claude" || !owner.IsHost {
		t.Fatalf("owner = %q (host=%v), want .claude host", owner.Account, owner.IsHost)
	}
	if owner.DupCount != 1 {
		t.Fatalf("dup_count = %d, want 1", owner.DupCount)
	}
}

func TestLocateOwnerCrossDirectorySlug(t *testing.T) {
	home := t.TempDir()
	sid := "xdir"
	// Session was created under a different cwd slug; LocateOwner scans every
	// projects/* so it is still found.
	writeTranscript(t, home, ".claude-w", "C--work-slack-helpers", sid, time.Now(), 10)
	owner := LocateOwner(sid, home)
	if owner == nil {
		t.Fatal("LocateOwner = nil, want cross-dir slug found")
	}
	if owner.Project != "C--work-slack-helpers" {
		t.Fatalf("project = %q, want the on-disk slug it was found under", owner.Project)
	}
}
