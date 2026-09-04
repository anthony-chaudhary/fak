package workerworktree

// #3613: the dispatch land seam consumes a refused Land by retrying it (bounded)
// before the reap destroys the worktree — but ONLY for the transient readback-race
// class. These tests pin the classifier that routes that decision, and bind it to
// the reason the readback path actually emits so a wording drift can never
// silently disable the re-land.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLandRefusalRetryableOnlyForReadbackMismatch(t *testing.T) {
	retryable := []string{
		LandReadbackMismatchToken,
		LandReadbackMismatchToken + ": trunk HEAD abcdef123456 does not carry intended path(s) cmd/x.go after commit — shared-index race, land not trusted (#3547)",
	}
	for _, reason := range retryable {
		if !LandRefusalRetryable(reason) {
			t.Fatalf("readback-race refusal must be retryable, got false for %q", reason)
		}
	}
	// Deterministic refusals: replaying them cannot change the verdict, so the
	// dispatch seam must go straight to the reap instead of burning re-lands.
	notRetryable := []string{
		"",
		"worktree verify failed, refusing to land: go build ./... failed: boom",
		"git apply to trunk failed",
		"post-apply disambiguation invariant failed",
		"could not read worktree diff vs HEAD (git error) — fail open",
		"no net diff in worktree vs HEAD to land",
	}
	for _, reason := range notRetryable {
		if LandRefusalRetryable(reason) {
			t.Fatalf("deterministic refusal must NOT be retryable, got true for %q", reason)
		}
	}
}

// TestLandReadbackRefusalReasonGradesRetryable binds the producer to the
// classifier: the exact reason landReadbackVerify emits on a swept commit must
// carry the structured token AND grade retryable at the dispatch seam.
func TestLandReadbackRefusalReasonGradesRetryable(t *testing.T) {
	fake := func(_ string, args []string) (int, string) {
		switch args[0] {
		case "rev-parse":
			return 0, "deadbeef00112233\n"
		case "diff-tree":
			// Trunk HEAD carries only a peer's path — the worker's intended path
			// was swept into a concurrent commit (#3547).
			return 0, "some/other/peer_file.go\n"
		}
		return 0, ""
	}
	ok, reason := landReadbackVerify("/trunk", []string{"cmd/fak/mine.go"}, fake)
	if ok {
		t.Fatalf("a commit missing the intended path must refuse, got ok with reason %q", reason)
	}
	if !strings.Contains(reason, LandReadbackMismatchToken) {
		t.Fatalf("refusal must carry the structured token %q, got %q", LandReadbackMismatchToken, reason)
	}
	if !LandRefusalRetryable(reason) {
		t.Fatalf("the emitted readback refusal must grade retryable at the dispatch seam (#3613), got %q", reason)
	}
}

func TestStripWorktreeWIPFences(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "feature.go")
	const original = "package feature\n\nfunc Run() {}\n"
	fenced := "//go:build wip_test\n\n" + original
	if err := os.WriteFile(f, []byte(fenced), 0o644); err != nil {
		t.Fatal(err)
	}

	stripWorktreeWIPFences(dir, []string{"feature.go"})

	got, err := os.ReadFile(f)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != original {
		t.Fatalf("stripWorktreeWIPFences did not unfence: got %q, want %q", string(got), original)
	}
}
