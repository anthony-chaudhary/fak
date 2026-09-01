package workerworktree

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func snapshotGit(t *testing.T, root string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=fak-test", "GIT_AUTHOR_EMAIL=fak@example.invalid", "GIT_COMMITTER_NAME=fak-test", "GIT_COMMITTER_EMAIL=fak@example.invalid")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func snapshotClones(t *testing.T) (string, string, string) {
	t.Helper()
	base := t.TempDir()
	bare := filepath.Join(base, "remote.git")
	snapshotGit(t, base, "init", "--bare", bare)
	a := filepath.Join(base, "a")
	snapshotGit(t, base, "clone", bare, a)
	if err := os.WriteFile(filepath.Join(a, "README"), []byte("seed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshotGit(t, a, "add", "README")
	snapshotGit(t, a, "commit", "-m", "seed")
	snapshotGit(t, a, "push", "origin", "HEAD:main")
	b := filepath.Join(base, "b")
	snapshotGit(t, base, "clone", "--branch", "main", bare, b)
	return bare, a, b
}

func TestRemoteSnapshotTwoClonesPublishFetchList(t *testing.T) {
	_, a, b := snapshotClones(t)
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	rows := []SnapshotRow{{
		HeadSHA: snapshotGit(t, a, "rev-parse", "HEAD"), BaseSHA: snapshotGit(t, a, "rev-parse", "HEAD"),
		Association: SnapshotAssociation{State: "ASSOCIATED", Lane: "lane-alpha", LeaseID: "lease-7"},
		Liveness:    SnapshotLiveness{Owner: "LIVE", Lease: "LIVE"}, Cleanliness: SnapshotCleanliness{State: "CLEAN"}, Lifecycle: "READY",
	}}
	s, err := NewRemoteSnapshot("builder-a.example", now, rows)
	if err != nil {
		t.Fatal(err)
	}
	published := PublishRemoteSnapshot(a, "origin", s, true, nil)
	if !published.OK || !published.Applied || published.PublishedOID == "" || published.ReadBackOID != published.PublishedOID {
		t.Fatalf("publish=%+v", published)
	}
	if err := FetchRemoteSnapshots(b, "origin", nil); err != nil {
		t.Fatal(err)
	}
	groups, err := ListRemoteSnapshots(b, "origin", now.Add(time.Hour), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 1 || groups[0].Host != s.Host || groups[0].Freshness != SnapshotFresh || groups[0].Authoritative || len(groups[0].Rows) != 1 {
		t.Fatalf("groups=%+v", groups)
	}
	got := groups[0].Rows[0]
	if got.Association.Lane != "lane-alpha" || got.Association.LeaseID != "lease-7" || got.HeadSHA != rows[0].HeadSHA || got.Lifecycle != "READY" {
		t.Fatalf("row=%+v", got)
	}
}

func TestRemoteSnapshotRedactsPathsHostPIDAndDirtyDetails(t *testing.T) {
	secretPath := "/Users/alice/private/fak-worker-secret"
	rows := []SnapshotRow{{HeadSHA: "abc", BaseSHA: "def", Association: SnapshotAssociation{State: "ASSOCIATED", Lane: "lane-a", LeaseID: "lease-a"}, Liveness: SnapshotLiveness{Owner: "LIVE", Lease: "LIVE"}, Cleanliness: SnapshotCleanliness{State: "DIRTY"}, Lifecycle: "DIRTY"}}
	s, err := NewRemoteSnapshot("alice-workstation.internal", time.Now(), rows)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, forbidden := range []string{secretPath, "alice-workstation.internal", "/Users/alice", "owner_pid", "dirty_paths", "command", "token"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("snapshot leaked %q: %s", forbidden, text)
		}
	}
	if !strings.HasPrefix(s.Host, "host-") || s.Host == "alice-workstation.internal" {
		t.Fatalf("host=%q", s.Host)
	}
}

func TestRemoteSnapshotStaleAndUnknownSchemaRemainNonAuthoritative(t *testing.T) {
	_, a, b := snapshotClones(t)
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	stale, _ := NewRemoteSnapshot("stale-host", now.Add(-2*RemoteSnapshotTTL), nil)
	if got := PublishRemoteSnapshot(a, "origin", stale, true, nil); !got.OK {
		t.Fatalf("stale publish=%+v", got)
	}
	unknown := RemoteSnapshot{Schema: "fak-worker-lifecycle-snapshot/999", Host: SnapshotHostID("future-host"), ObservedAt: now, ExpiresAt: now.Add(RemoteSnapshotTTL), Rows: []SnapshotRow{}}
	if got := PublishRemoteSnapshot(a, "origin", unknown, true, nil); !got.OK {
		t.Fatalf("unknown publish=%+v", got)
	}
	if err := FetchRemoteSnapshots(b, "origin", nil); err != nil {
		t.Fatal(err)
	}
	groups, err := ListRemoteSnapshots(b, "origin", now, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 {
		t.Fatalf("groups=%+v", groups)
	}
	seen := map[SnapshotFreshness]int{}
	for _, g := range groups {
		seen[g.Freshness]++
		if g.Authoritative {
			t.Fatalf("remote group authorized action: %+v", g)
		}
		if g.Freshness != SnapshotFresh && g.Reason == "" {
			t.Fatalf("missing typed reason: %+v", g)
		}
	}
	if seen[SnapshotStale] != 1 || seen[SnapshotUnknown] != 1 {
		t.Fatalf("freshness=%v groups=%+v", seen, groups)
	}
}

func TestRemoteSnapshotCASRejectsConcurrentWriter(t *testing.T) {
	_, a, _ := snapshotClones(t)
	now := time.Now().UTC()
	s, _ := NewRemoteSnapshot("same-host", now, nil)
	first := PublishRemoteSnapshot(a, "origin", s, true, nil)
	if !first.OK {
		t.Fatalf("first=%+v", first)
	}
	calls := 0
	git := func(root string, args []string) (int, string) {
		calls++
		if len(args) > 0 && args[0] == "ls-remote" && calls == 1 {
			return 0, "deadbeef\t" + SnapshotRef(s.Host) + "\n"
		}
		return defaultGit(root, args)
	}
	second := PublishRemoteSnapshot(a, "origin", s, true, git)
	if second.OK || !strings.Contains(second.Reason, "compare-and-swap") {
		t.Fatalf("second=%+v", second)
	}
}
