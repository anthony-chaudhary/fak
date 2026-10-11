package safesync

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// dupFixRepo builds the 2026-10-10 ctxmmu shape: a base commit with
// internal/ctxmmu/closed_windows.go, a "peer" branch that renames it to
// closed_window_history.go (already landed upstream), and local main that renames the SAME
// source to closed_file_window.go. Returns the repo dir and the base SHA.
func dupFixRepo(t *testing.T, localTarget string) (string, string) {
	t.Helper()
	dir := t.TempDir()
	git(t, dir, "init", "-q", "-b", "main")
	git(t, dir, "config", "user.email", "t@example.invalid")
	git(t, dir, "config", "user.name", "t")
	git(t, dir, "config", "commit.gpgsign", "false")
	src := filepath.Join(dir, "internal", "ctxmmu")
	if err := os.MkdirAll(src, 0o755); err != nil {
		t.Fatal(err)
	}
	body := "package ctxmmu\n\n// fileWindows tracks closed windows.\nvar fileWindows = map[string]int{}\n"
	if err := os.WriteFile(filepath.Join(src, "closed_windows.go"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "other.go"), []byte("package ctxmmu\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git(t, dir, "add", "-A")
	git(t, dir, "commit", "-q", "-m", "feat(ctxmmu): add closed windows")
	base := strings.TrimSpace(gitOutput(t, dir, "rev-parse", "HEAD"))

	git(t, dir, "checkout", "-q", "-b", "peer")
	git(t, dir, "mv", "internal/ctxmmu/closed_windows.go", "internal/ctxmmu/closed_window_history.go")
	git(t, dir, "commit", "-q", "-m", "fix(ctxmmu): rename closed_windows.go off the GOOS suffix (fak ctxmmu)")

	git(t, dir, "checkout", "-q", "main")
	git(t, dir, "mv", "internal/ctxmmu/closed_windows.go", "internal/ctxmmu/"+localTarget)
	git(t, dir, "commit", "-q", "-m", "fix(ctxmmu): local repair (fak ctxmmu)")
	return dir, base
}

func TestDuplicateFixAdvisoryNamesPeerRename(t *testing.T) {
	dir, base := dupFixRepo(t, "closed_file_window.go")
	peerSHA := strings.TrimSpace(gitOutput(t, dir, "rev-parse", "peer"))

	fixes, err := DuplicateFixAdvisories(context.Background(), nil, dir, "peer")
	if err != nil {
		t.Fatalf("DuplicateFixAdvisories: %v", err)
	}
	if len(fixes) != 1 {
		t.Fatalf("want 1 duplicate fix, got %+v", fixes)
	}
	f := fixes[0]
	if f.Source != "internal/ctxmmu/closed_windows.go" || f.PeerSHA != peerSHA || f.MergeBaseSHA != base ||
		f.LocalStatus != "R" || f.LocalTarget != "internal/ctxmmu/closed_file_window.go" ||
		f.PeerStatus != "R" || f.PeerTarget != "internal/ctxmmu/closed_window_history.go" {
		t.Fatalf("advisory fields wrong: %+v", f)
	}
	var buf bytes.Buffer
	RenderDuplicateFixes(&buf, "peer", fixes)
	for _, want := range []string{"DUPLICATE_FIX", peerSHA[:12], "closed_window_history.go", "closed_file_window.go"} {
		if !strings.Contains(buf.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, buf.String())
		}
	}
}

// The converged state (the follow-up renamed to the peer's names) must be silent.
func TestDuplicateFixAdvisorySilentWhenRenamesAgree(t *testing.T) {
	dir, _ := dupFixRepo(t, "closed_window_history.go")
	fixes, err := DuplicateFixAdvisories(context.Background(), nil, dir, "peer")
	if err != nil || len(fixes) != 0 {
		t.Fatalf("identical renames must not advise, got %+v err=%v", fixes, err)
	}
	var buf bytes.Buffer
	RenderDuplicateFixes(&buf, "peer", fixes)
	if buf.Len() != 0 {
		t.Fatalf("empty advisory must render nothing, got %q", buf.String())
	}
}

func TestDuplicateFixAdvisoryMissingUpstreamIsQuiet(t *testing.T) {
	dir, _ := dupFixRepo(t, "closed_file_window.go")
	fixes, err := DuplicateFixAdvisories(context.Background(), nil, dir, "refs/remotes/origin/nope")
	if err != nil || len(fixes) != 0 {
		t.Fatalf("missing upstream must be quiet, got %+v err=%v", fixes, err)
	}
}

// PeerCommitsTouching is the gate-side lookup: after the red base, upstream already holds a
// commit touching the failing package.
func TestPeerCommitsTouchingFindsLandedFix(t *testing.T) {
	dir, base := dupFixRepo(t, "closed_file_window.go")
	peerSHA := strings.TrimSpace(gitOutput(t, dir, "rev-parse", "peer"))
	got, err := PeerCommitsTouching(context.Background(), nil, dir, base, "peer", []string{"internal/ctxmmu"})
	if err != nil || len(got) != 1 || got[0].SHA != peerSHA || !strings.Contains(got[0].Subject, "fak ctxmmu") {
		t.Fatalf("PeerCommitsTouching = %+v err=%v, want the peer fix %s", got, err, peerSHA)
	}
	if none, err := PeerCommitsTouching(context.Background(), nil, dir, base, "peer", []string{"internal/elsewhere"}); err != nil || len(none) != 0 {
		t.Fatalf("unrelated path must find nothing, got %+v err=%v", none, err)
	}
}
