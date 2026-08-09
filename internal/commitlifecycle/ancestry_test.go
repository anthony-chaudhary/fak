package commitlifecycle

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func ancestryGit(ctx context.Context, repo string, args ...string) (string, int, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = repo
	b, err := cmd.CombinedOutput()
	if err == nil {
		return string(b), 0, nil
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return string(b), ee.ExitCode(), nil
	}
	return string(b), -1, err
}

func TestInspectAncestryCommittedThenShipped(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	bare := filepath.Join(root, "remote.git")
	repo := filepath.Join(root, "repo")
	mustGit(t, root, "init", "--bare", "-q", bare)
	mustGit(t, root, "init", "-q", "-b", "main", repo)
	mustGit(t, repo, "config", "user.name", "test")
	mustGit(t, repo, "config", "user.email", "[redacted:pii:14B]")
	mustGit(t, repo, "remote", "add", "upstream", bare)
	mustGit(t, repo, "config", "branch.main.remote", "upstream")
	mustGit(t, repo, "config", "branch.main.merge", "refs/heads/main")
	mustWriteCommit(t, repo, "base.txt", "base", "base")
	mustGit(t, repo, "push", "-q", "-u", "upstream", "main")
	mustWriteCommit(t, repo, "change.txt", "change", "feat(x): local only (#1) (fak x)")

	beforeHead := mustGit(t, repo, "rev-parse", "HEAD")
	a, err := InspectAncestry(ctx, repo, ancestryGit)
	if err != nil {
		t.Fatal(err)
	}
	if a.Remote != "upstream" || a.RemoteRef != "refs/remotes/upstream/main" || a.Stale || len(a.Commits) != 1 {
		t.Fatalf("ancestry before push = %+v", a)
	}
	rows := AncestryRows(a)
	if len(rows) != 1 || rows[0].State != CommittedUnpushed || rows[0].Action.Tool != "fak" || strings.Join(rows[0].Action.Args, " ") != "sync push" {
		t.Fatalf("rows before push = %+v", rows)
	}
	if got := mustGit(t, repo, "rev-parse", "HEAD"); got != beforeHead {
		t.Fatalf("inspection mutated HEAD: %s -> %s", beforeHead, got)
	}

	mustGit(t, repo, "push", "-q", "upstream", "main")
	a, err = InspectAncestry(ctx, repo, ancestryGit)
	if err != nil {
		t.Fatal(err)
	}
	rows = AncestryRows(a)
	if len(a.Commits) != 0 || len(rows) != 1 || rows[0].State != Shipped || !a.HeadOnRemote {
		t.Fatalf("after push is not remotely witnessed SHIPPED: ancestry=%+v rows=%+v", a, rows)
	}

	// A remote branch ahead of local still contains local HEAD and therefore
	// witnesses shipment; equality with the tracking ref tip is too strict.
	peer := filepath.Join(root, "peer")
	mustGit(t, root, "clone", "-q", "-b", "main", bare, peer)
	mustGit(t, peer, "config", "user.name", "test")
	mustGit(t, peer, "config", "user.email", "[redacted:pii:14B]")
	mustWriteCommit(t, peer, "peer.txt", "peer", "feat(x): remote ahead (#2) (fak x)")
	mustGit(t, peer, "push", "-q", "origin", "main")
	mustGit(t, repo, "fetch", "-q", "upstream", "main")
	a, err = InspectAncestry(ctx, repo, ancestryGit)
	if err != nil {
		t.Fatal(err)
	}
	rows = AncestryRows(a)
	if len(a.Commits) != 0 || len(rows) != 1 || rows[0].State != Shipped || !a.HeadOnRemote {
		t.Fatalf("remote-ahead ancestry is not witnessed SHIPPED: ancestry=%+v rows=%+v", a, rows)
	}
}

func TestInspectAncestryMissingRemoteFailsTowardOperator(t *testing.T) {
	repo := t.TempDir()
	mustGit(t, repo, "init", "-q", "-b", "main")
	mustGit(t, repo, "config", "user.name", "test")
	mustGit(t, repo, "config", "user.email", "[redacted:pii:14B]")
	mustWriteCommit(t, repo, "base.txt", "base", "base")
	a, err := InspectAncestry(context.Background(), repo, ancestryGit)
	if err != nil {
		t.Fatal(err)
	}
	rows := AncestryRows(a)
	if !a.Stale || len(rows) != 1 || rows[0].State != Unknown || !rows[0].Action.NeedsOperator {
		t.Fatalf("missing remote overclaimed: ancestry=%+v rows=%+v", a, rows)
	}
}

func mustGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	out, code, err := ancestryGit(context.Background(), dir, args...)
	if err != nil || code != 0 {
		t.Fatalf("git %v: code=%d err=%v out=%s", args, code, err, out)
	}
	return strings.TrimSpace(out)
}

func mustWriteCommit(t *testing.T, repo, name, body, message string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(repo, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, repo, "add", "--", name)
	mustGit(t, repo, "commit", "-q", "-m", message)
}
