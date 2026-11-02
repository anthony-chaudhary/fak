package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// fak-test:runtime integration est=2s lane=default
func TestRunSyncPushSelectedSource(t *testing.T) {
	// Real Git validates type/ancestry and literal branch names. The transport
	// seam records the normal native refspec, without any remote publication.
	repo := t.TempDir()
	syncGit(t, repo, "init", "-b", "main")
	syncGit(t, repo, "config", "user.name", "test")
	syncGit(t, repo, "config", "user.email", "test@example.com")
	syncGit(t, repo, "config", "core.autocrlf", "false")
	file := filepath.Join(repo, "source.txt")
	syncWriteFile(t, file, "source\n")
	syncGit(t, repo, "add", "source.txt")
	syncGit(t, repo, "commit", "-m", "source")
	source := syncRev(t, repo, "HEAD")
	blob := syncRev(t, repo, "HEAD:source.txt")
	syncGit(t, repo, "tag", "-a", "source-tag", "-m", "tag")
	tag := syncRev(t, repo, "refs/tags/source-tag")
	syncWriteFile(t, file, "later peer\n")
	syncGit(t, repo, "commit", "-am", "later peer")
	head := syncRev(t, repo, "HEAD")
	tree := syncRev(t, repo, "HEAD^{tree}")
	unrelated, err := gitOut(repo, "commit-tree", tree, "-m", "unrelated root")
	if err != nil {
		t.Fatal(err)
	}
	unrelated = strings.TrimSpace(unrelated)
	later, err := gitOut(repo, "commit-tree", tree, "-p", head, "-m", "HEAD moved after selection")
	if err != nil {
		t.Fatal(err)
	}
	later = strings.TrimSpace(later)
	syncWriteFile(t, filepath.Join(repo, "staged-peer.txt"), "peer staged bytes\n")
	syncGit(t, repo, "add", "staged-peer.txt")
	indexBefore, err := os.ReadFile(filepath.Join(repo, ".git", "index"))
	if err != nil {
		t.Fatal(err)
	}
	worktreeBefore, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}

	oldCapture, oldPush, oldWorktree := syncCaptureSource, syncSafePush, syncWorktree
	t.Cleanup(func() {
		syncCaptureSource, syncSafePush, syncWorktree = oldCapture, oldPush, oldWorktree
	})
	syncWorktree = func(context.Context, string) (safesync.Worktree, bool) {
		return safesync.Worktree{}, false
	}
	var captureCalls, pushCalls int
	syncCaptureSource = func(root string) (string, error) {
		captureCalls++
		return gitOut(root, "rev-parse", "HEAD")
	}
	refusal := false
	syncSafePush = func(ctx context.Context, opts safesync.PushOptions) (safesync.PushResult, error) {
		pushCalls++
		if opts.SourceRef != source || opts.Branch != "handoff/selected" || opts.TargetRef != "refs/heads/handoff/selected" {
			t.Fatalf("selection changed before transport: %+v", opts)
		}
		if !opts.NoFetchOnFailure || opts.MaxRetries != 1 {
			t.Fatalf("caller failure/retry contract lost: %+v", opts)
		}
		// Advance only the disposable fixture's HEAD after selection. Every
		// actual SafePush call must still carry the earlier exact source.
		if pushCalls == 1 {
			syncGit(t, repo, "update-ref", "refs/heads/main", later, head)
		}
		calls := 0
		opts.Runner = func(_ context.Context, root string, args ...string) safesync.RunResult {
			calls++
			want := []string{"push", "origin", source + ":refs/heads/handoff/selected"}
			if root != repo || !reflect.DeepEqual(args, want) {
				t.Fatalf("unexpected native request: %q %q", root, args)
			}
			if refusal {
				return safesync.RunResult{Code: 1, Stderr: []byte("AUTHORIZATION (blocked): approval is required\n")}
			}
			return safesync.RunResult{}
		}
		result, pushErr := safesync.SafePush(ctx, opts)
		if calls != 1 {
			t.Fatalf("native requests=%d, want one (no retry or fetch)", calls)
		}
		return result, pushErr
	}
	for _, tc := range []struct {
		name, branch string
		refused      bool
	}{
		{"short branch", "handoff/selected", false},
		{"fully qualified branch", "refs/heads/handoff/selected", false},
		{"native refusal preserved", "handoff/selected", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			refusal = tc.refused
			var out, errb bytes.Buffer
			code := runSync(&out, &errb, []string{"push", "--repo", repo, "--branch", tc.branch,
				"--source-commit", source, "--no-fetch-on-failure", "--retries", "1", "--json"})
			want := syncExitOK
			if tc.refused {
				want = syncExitRefused
			}
			if code != want {
				t.Fatalf("exit=%d want=%d stdout=%s stderr=%s", code, want, out.String(), errb.String())
			}
			var got safesync.PushResult
			if err := json.Unmarshal(out.Bytes(), &got); err != nil {
				t.Fatal(err)
			}
			if got.SourceRef != source || got.TargetRef != "refs/heads/handoff/selected" || got.Pushed == tc.refused {
				t.Fatalf("wrong exact-source receipt: %+v", got)
			}
			if tc.refused && (got.Reason != safesync.PushReasonError || got.Velocity.Qualified) {
				t.Fatalf("native refusal upgraded: %+v", got)
			}
		})
	}
	if captureCalls != 3 || pushCalls != 3 {
		t.Fatalf("capture=%d push=%d, want once per request", captureCalls, pushCalls)
	}
	if syncRev(t, repo, "HEAD") == head {
		t.Fatal("fixture did not advance HEAD after selection")
	}
	// Moving the fixture ref without touching its index leaves staged peers
	// untouched, just as the selector and native transport must leave them.
	indexAfter, err := os.ReadFile(filepath.Join(repo, ".git", "index"))
	if err != nil || !bytes.Equal(indexBefore, indexAfter) {
		t.Fatalf("peer index changed: %v", err)
	}
	worktreeAfter, err := os.ReadFile(file)
	if err != nil || !bytes.Equal(worktreeBefore, worktreeAfter) {
		t.Fatalf("peer worktree changed: %v", err)
	}

	syncSafePush = func(context.Context, safesync.PushOptions) (safesync.PushResult, error) {
		t.Fatal("invalid selector reached native publication")
		return safesync.PushResult{}, errors.New("unreachable")
	}
	// On older Git, even a false-valued promisor setting makes the absence
	// proof fail closed. No missing-object query or transport may be attempted.
	if _, probeErr := gitOut(repo, "--no-lazy-fetch", "version"); probeErr != nil {
		var exit *exec.ExitError
		if !errors.As(probeErr, &exit) || exit.ExitCode() != 129 {
			t.Fatal(probeErr)
		}
		t.Run("old Git refuses possible lazy fetch", func(t *testing.T) {
			syncGit(t, repo, "config", "remote.origin.promisor", "false")
			defer syncGit(t, repo, "config", "--unset", "remote.origin.promisor")
			var out, errb bytes.Buffer
			code := runSync(&out, &errb, []string{"push", "--repo", repo, "--branch", "handoff/selected",
				"--source-commit", source, "--json"})
			var got safesync.PushResult
			if code != syncExitRefused || json.Unmarshal(out.Bytes(), &got) != nil ||
				got.Pushed || got.Reason != "SOURCE_COMMIT_UNAVAILABLE" || got.Attempts != 0 {
				t.Fatalf("exit=%d stdout=%s stderr=%s", code, out.String(), errb.String())
			}
		})
		t.Run("old Git refuses promisor pack marker", func(t *testing.T) {
			pack := filepath.Join(repo, ".git", "objects", "pack")
			if err := os.MkdirAll(pack, 0o755); err != nil {
				t.Fatal(err)
			}
			marker := filepath.Join(pack, "test.promisor")
			if err := os.WriteFile(marker, nil, 0o600); err != nil {
				t.Fatal(err)
			}
			defer os.Remove(marker)
			var out, errb bytes.Buffer
			code := runSync(&out, &errb, []string{"push", "--repo", repo, "--branch", "handoff/selected",
				"--source-commit", source, "--json"})
			var got safesync.PushResult
			if code != syncExitRefused || json.Unmarshal(out.Bytes(), &got) != nil ||
				got.Pushed || got.Reason != "SOURCE_COMMIT_UNAVAILABLE" || got.Attempts != 0 {
				t.Fatalf("exit=%d stdout=%s stderr=%s", code, out.String(), errb.String())
			}
		})
	}
	for _, tc := range []struct {
		name, command, source, branch, reason string
	}{
		{"empty", "push", "", "handoff/selected", "SOURCE_COMMIT_INVALID"},
		{"abbreviated", "push", source[:12], "handoff/selected", "SOURCE_COMMIT_INVALID"},
		{"ref expression", "push", "HEAD~1", "handoff/selected", "SOURCE_COMMIT_INVALID"},
		{"whitespace", "push", source + " ", "handoff/selected", "SOURCE_COMMIT_INVALID"},
		{"unreadable", "push", strings.Repeat("f", len(source)), "handoff/selected", "SOURCE_COMMIT_UNAVAILABLE"},
		{"blob", "push", blob, "handoff/selected", "SOURCE_COMMIT_UNAVAILABLE"},
		{"tag object", "push", tag, "handoff/selected", "SOURCE_COMMIT_UNAVAILABLE"},
		{"unrelated commit", "push", unrelated, "handoff/selected", "SOURCE_COMMIT_NOT_ANCESTOR"},
		{"wrong verb", "check", source, "handoff/selected", "SOURCE_COMMIT_UNSUPPORTED"},
		{"omitted branch", "push", source, "", "SOURCE_COMMIT_UNSUPPORTED"},
		{"trunk", "push", source, "main", "SOURCE_COMMIT_UNSUPPORTED"},
		{"qualified trunk", "push", source, "refs/heads/main", "SOURCE_COMMIT_UNSUPPORTED"},
		{"other branch", "push", source, "topic/selected", "SOURCE_COMMIT_UNSUPPORTED"},
		{"substring", "push", source, "not-handoff/selected", "SOURCE_COMMIT_UNSUPPORTED"},
		{"wrong namespace", "push", source, "refs/tags/handoff/selected", "SOURCE_COMMIT_UNSUPPORTED"},
		{"path traversal", "push", source, "handoff/../main", "SOURCE_COMMIT_INVALID"},
		{"double slash", "push", source, "handoff//selected", "SOURCE_COMMIT_INVALID"},
		{"refspec injection", "push", source, "handoff/x:refs/heads/main", "SOURCE_COMMIT_INVALID"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := runSync(&out, &errb, []string{tc.command, "--repo", repo, "--branch", tc.branch,
				"--source-commit", tc.source, "--json"})
			var got safesync.PushResult
			if code != syncExitRefused || json.Unmarshal(out.Bytes(), &got) != nil ||
				got.Pushed || got.Reason != tc.reason || got.Attempts != 0 || got.Velocity.Qualified {
				t.Fatalf("exit=%d stdout=%s stderr=%s", code, out.String(), errb.String())
			}
		})
	}
}
