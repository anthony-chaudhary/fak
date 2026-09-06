package safecommit

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func initTempRepo(t *testing.T, repo string) {
	t.Helper()
	tempRepoGit(t, repo, "init", "-q")
	tempRepoGit(t, repo, "config", "user.name", "fak test")
	tempRepoGit(t, repo, "config", "user.email", "fak-test@example.invalid")
}

// TestAuthorityFenceBeforeStaging witnesses issue #11849:
// A real temporary Git fixture replaces owner A's lease with owner B's generation;
// A's explicit-path commit is refused before staging with unchanged index and HEAD,
// while B's valid scoped commit succeeds.
func TestAuthorityFenceBeforeStaging(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)
	writeTempRepoFile(t, filepath.Join(repo, "seed.txt"), "seed content\n")
	tempRepoGit(t, repo, "add", "seed.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial seed")

	initialHead := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	if initialHead == "" {
		t.Fatal("initial HEAD is empty")
	}

	// Create unstaged modifications for owner A and owner B in the working tree.
	fileA := "work_a.txt"
	fileB := "work_b.txt"
	writeTempRepoFile(t, filepath.Join(repo, fileA), "change from owner A\n")
	writeTempRepoFile(t, filepath.Join(repo, fileB), "change from owner B\n")

	// Step 1: Establish owner A's authority lease at generation 1.
	recA := AuthorityRecord{
		Workspace:  repo,
		Owner:      "owner-A",
		SessionID:  "sess-A",
		Generation: 1,
		Paths:      []string{fileA},
	}
	if err := WriteAuthorityRecord(repo, recA); err != nil {
		t.Fatalf("write authority A: %v", err)
	}

	// Step 2: Replace owner A's lease with owner B's generation 2.
	recB := AuthorityRecord{
		Workspace:  repo,
		Owner:      "owner-B",
		SessionID:  "sess-B",
		Generation: 2,
		Paths:      []string{fileB},
	}
	if err := WriteAuthorityRecord(repo, recB); err != nil {
		t.Fatalf("replace authority with B: %v", err)
	}

	// Step 3: Owner A attempts explicit-path commit with stale authority token (gen 1 vs active gen 2).
	fenceA := &AuthorityFence{
		Workspace:  repo,
		Owner:      "owner-A",
		SessionID:  "sess-A",
		Generation: 1,
		Paths:      []string{fileA},
	}
	optsA := Options{
		Dir:            repo,
		Paths:          []string{fileA},
		Message:        "feat: work by A (fak safecommit)",
		AuthorityFence: fenceA,
	}

	resA, err := Commit(context.Background(), optsA)
	if err != nil {
		t.Fatalf("unexpected infra error for commit A: %v", err)
	}
	if resA.Committed {
		t.Fatalf("expected commit A to be refused, but got committed=true")
	}
	if resA.Verified {
		t.Fatalf("expected commit A to not be verified, but got verified=true")
	}
	if resA.Reason != ReasonAuthorityStale {
		t.Fatalf("expected reason %q, got %q (detail: %s)", ReasonAuthorityStale, resA.Reason, resA.Detail)
	}
	if resA.Authority == nil || resA.Authority.Outcome != OutcomeRefused {
		t.Fatalf("expected Authority.Outcome = %q, got %+v", OutcomeRefused, resA.Authority)
	}
	if resA.Outcome() != OutcomeRefused {
		t.Fatalf("expected Outcome() = %q, got %q", OutcomeRefused, resA.Outcome())
	}

	// Step 4: Verify index and HEAD are COMPLETELY unchanged after A's refusal.
	headAfterA := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	if headAfterA != initialHead {
		t.Fatalf("HEAD changed after refused commit A: got %s, want %s", headAfterA, initialHead)
	}
	stagedA := strings.TrimSpace(tempRepoGit(t, repo, "diff", "--cached", "--name-only"))
	if stagedA != "" {
		t.Fatalf("index was mutated before staging refusal: staged=%q", stagedA)
	}

	// fileA and fileB remain uncommitted in the working tree.
	statusA := strings.TrimSpace(tempRepoGit(t, repo, "status", "--porcelain", "--", fileA))
	if !strings.HasPrefix(statusA, "??") {
		t.Fatalf("work_a.txt should remain untracked in working tree, got status %q", statusA)
	}

	// Step 5: Owner B's valid scoped commit succeeds.
	fenceB := &AuthorityFence{
		Workspace:  repo,
		Owner:      "owner-B",
		SessionID:  "sess-B",
		Generation: 2,
		Paths:      []string{fileB},
	}
	optsB := Options{
		Dir:            repo,
		Paths:          []string{fileB},
		Message:        "feat: work by B (fak safecommit)",
		AuthorityFence: fenceB,
	}

	resB, err := Commit(context.Background(), optsB)
	if err != nil {
		t.Fatalf("commit B failed with error: %v", err)
	}
	if !resB.Committed || !resB.Verified {
		t.Fatalf("expected commit B to succeed: committed=%v verified=%v reason=%q detail=%q",
			resB.Committed, resB.Verified, resB.Reason, resB.Detail)
	}
	if resB.Reason != "" {
		t.Fatalf("expected empty reason for B, got %q", resB.Reason)
	}
	if resB.Authority == nil || resB.Authority.Outcome != OutcomeAdmitted {
		t.Fatalf("expected Authority.Outcome = %q, got %+v", OutcomeAdmitted, resB.Authority)
	}
	if resB.Outcome() != OutcomeAdmitted {
		t.Fatalf("expected Outcome() = %q, got %q", OutcomeAdmitted, resB.Outcome())
	}

	headAfterB := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	if headAfterB == initialHead {
		t.Fatalf("HEAD did not advance after commit B")
	}
	if headAfterB != resB.SHA {
		t.Fatalf("HEAD %s != resB.SHA %s", headAfterB, resB.SHA)
	}

	// Only fileB landed in commit B.
	landed := strings.TrimSpace(tempRepoGit(t, repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"))
	if landed != fileB {
		t.Fatalf("expected only %q to land in commit B, got %q", fileB, landed)
	}

	// fileA is still untracked and unstaged in the working tree.
	statusAAfterB := strings.TrimSpace(tempRepoGit(t, repo, "status", "--porcelain", "--", fileA))
	if !strings.HasPrefix(statusAAfterB, "??") {
		t.Fatalf("work_a.txt should remain untracked after commit B, got %q", statusAAfterB)
	}
}

func TestAuthorityFenceForeign(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)
	writeTempRepoFile(t, filepath.Join(repo, "seed.txt"), "seed content\n")
	tempRepoGit(t, repo, "add", "seed.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial seed")

	initialHead := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))

	writeTempRepoFile(t, filepath.Join(repo, "target.txt"), "target content\n")

	// Active lease: owner "leader-1", session "sess-1", gen 5, paths ["target.txt"]
	rec := AuthorityRecord{
		Workspace:  repo,
		Owner:      "leader-1",
		SessionID:  "sess-1",
		Generation: 5,
		Paths:      []string{"target.txt"},
	}
	if err := WriteAuthorityRecord(repo, rec); err != nil {
		t.Fatalf("write authority: %v", err)
	}

	cases := []struct {
		name       string
		fence      *AuthorityFence
		paths      []string
		wantReason string
	}{
		{
			name: "foreign owner",
			fence: &AuthorityFence{
				Workspace:  repo,
				Owner:      "foreign-agent",
				SessionID:  "sess-1",
				Generation: 5,
				Paths:      []string{"target.txt"},
			},
			paths:      []string{"target.txt"},
			wantReason: ReasonAuthorityForeign,
		},
		{
			name: "foreign session",
			fence: &AuthorityFence{
				Workspace:  repo,
				Owner:      "leader-1",
				SessionID:  "foreign-session",
				Generation: 5,
				Paths:      []string{"target.txt"},
			},
			paths:      []string{"target.txt"},
			wantReason: ReasonAuthorityForeign,
		},
		{
			name: "foreign workspace",
			fence: &AuthorityFence{
				Workspace:  filepath.Join(repo, "nonexistent-workspace"),
				Owner:      "leader-1",
				SessionID:  "sess-1",
				Generation: 5,
				Paths:      []string{"target.txt"},
			},
			paths:      []string{"target.txt"},
			wantReason: ReasonAuthorityForeign,
		},
		{
			name: "path outside fence scope",
			fence: &AuthorityFence{
				Workspace:  repo,
				Owner:      "leader-1",
				SessionID:  "sess-1",
				Generation: 5,
				Paths:      []string{"other.txt"},
			},
			paths:      []string{"target.txt"},
			wantReason: ReasonAuthorityForeign,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			opts := Options{
				Dir:            repo,
				Paths:          tc.paths,
				Message:        "feat: foreign test (fak safecommit)",
				AuthorityFence: tc.fence,
			}
			res, err := Commit(context.Background(), opts)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if res.Committed || res.Verified {
				t.Fatalf("expected refusal, got committed=%v verified=%v", res.Committed, res.Verified)
			}
			if res.Reason != tc.wantReason {
				t.Fatalf("got reason %q, want %q (detail: %s)", res.Reason, tc.wantReason, res.Detail)
			}
			if res.Authority == nil || res.Authority.Outcome != OutcomeRefused {
				t.Fatalf("expected OutcomeRefused, got %+v", res.Authority)
			}
			// Verify index and HEAD remain untouched.
			head := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
			if head != initialHead {
				t.Fatalf("HEAD changed: %s vs %s", head, initialHead)
			}
			staged := strings.TrimSpace(tempRepoGit(t, repo, "diff", "--cached", "--name-only"))
			if staged != "" {
				t.Fatalf("index mutated: %q", staged)
			}
		})
	}
}

func TestAuthorityFenceUnavailable(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)
	writeTempRepoFile(t, filepath.Join(repo, "seed.txt"), "seed content\n")
	tempRepoGit(t, repo, "add", "seed.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial seed")

	initialHead := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))

	writeTempRepoFile(t, filepath.Join(repo, "work.txt"), "work content\n")

	// No authority file written -> authority unavailable
	fence := &AuthorityFence{
		Workspace:  repo,
		Owner:      "owner-A",
		Generation: 1,
		Paths:      []string{"work.txt"},
	}
	opts := Options{
		Dir:            repo,
		Paths:          []string{"work.txt"},
		Message:        "feat: unavailable test (fak safecommit)",
		AuthorityFence: fence,
	}

	res, err := Commit(context.Background(), opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Committed || res.Verified {
		t.Fatalf("expected refusal, got committed=%v verified=%v", res.Committed, res.Verified)
	}
	if res.Reason != ReasonAuthorityUnavailable {
		t.Fatalf("got reason %q, want %q (detail: %s)", res.Reason, ReasonAuthorityUnavailable, res.Detail)
	}
	if res.Authority == nil || res.Authority.Outcome != OutcomeRefused {
		t.Fatalf("expected OutcomeRefused, got %+v", res.Authority)
	}

	head := strings.TrimSpace(tempRepoGit(t, repo, "rev-parse", "HEAD"))
	if head != initialHead {
		t.Fatalf("HEAD changed: %s vs %s", head, initialHead)
	}
	staged := strings.TrimSpace(tempRepoGit(t, repo, "diff", "--cached", "--name-only"))
	if staged != "" {
		t.Fatalf("index mutated: %q", staged)
	}
}

func TestAuthorityFenceLegacyUnconfigured(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)
	writeTempRepoFile(t, filepath.Join(repo, "seed.txt"), "seed content\n")
	tempRepoGit(t, repo, "add", "seed.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial seed")

	writeTempRepoFile(t, filepath.Join(repo, "legacy.txt"), "legacy content\n")

	// Unconfigured nil AuthorityFence MUST proceed normally
	opts := Options{
		Dir:            repo,
		Paths:          []string{"legacy.txt"},
		Message:        "feat: legacy unconfigured commit (fak safecommit)",
		AuthorityFence: nil,
	}

	res, err := Commit(context.Background(), opts)
	if err != nil {
		t.Fatalf("commit failed: %v", err)
	}
	if !res.Committed || !res.Verified {
		t.Fatalf("expected success: committed=%v verified=%v reason=%q", res.Committed, res.Verified, res.Reason)
	}
	if res.Authority != nil {
		t.Fatalf("expected nil Authority for unconfigured commit, got %+v", res.Authority)
	}

	// Empty struct AuthorityFence MUST also proceed normally
	writeTempRepoFile(t, filepath.Join(repo, "legacy2.txt"), "legacy 2 content\n")
	opts2 := Options{
		Dir:            repo,
		Paths:          []string{"legacy2.txt"},
		Message:        "feat: legacy empty fence commit (fak safecommit)",
		AuthorityFence: &AuthorityFence{},
	}
	res2, err := Commit(context.Background(), opts2)
	if err != nil {
		t.Fatalf("commit failed: %v", err)
	}
	if !res2.Committed || !res2.Verified {
		t.Fatalf("expected success: committed=%v verified=%v reason=%q", res2.Committed, res2.Verified, res2.Reason)
	}
	if res2.Authority != nil {
		t.Fatalf("expected nil Authority for empty fence commit, got %+v", res2.Authority)
	}
}

func TestAuthorityValidatorCustom(t *testing.T) {
	repo := t.TempDir()
	initTempRepo(t, repo)
	writeTempRepoFile(t, filepath.Join(repo, "seed.txt"), "seed content\n")
	tempRepoGit(t, repo, "add", "seed.txt")
	tempRepoGit(t, repo, "commit", "-qm", "initial seed")

	writeTempRepoFile(t, filepath.Join(repo, "custom.txt"), "custom content\n")

	called := false
	customValidator := func(ctx context.Context, fence AuthorityFence, paths []string) (AuthorityReceipt, error) {
		called = true
		if fence.Owner != "authorized-worker" {
			return AuthorityReceipt{
				Outcome: OutcomeRefused,
				Reason:  ReasonAuthorityForeign,
				Detail:  "worker identity refuted by custom validator",
			}, nil
		}
		return AuthorityReceipt{
			Outcome:    OutcomeAdmitted,
			Workspace:  fence.Workspace,
			Owner:      fence.Owner,
			Generation: fence.Generation,
			Paths:      paths,
		}, nil
	}

	// 1. Refusal case via AuthorityValidator
	optsRefused := Options{
		Dir:                repo,
		Paths:              []string{"custom.txt"},
		Message:            "feat: custom validator refusal (fak safecommit)",
		AuthorityFence:     &AuthorityFence{Owner: "unauthorized"},
		AuthorityValidator: customValidator,
	}
	resRefused, err := Commit(context.Background(), optsRefused)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("custom validator was not called")
	}
	if resRefused.Committed || resRefused.Verified {
		t.Fatal("expected refusal, got success")
	}
	if resRefused.Reason != ReasonAuthorityForeign {
		t.Fatalf("reason = %q, want %q", resRefused.Reason, ReasonAuthorityForeign)
	}

	// 2. Admission case via AuthorityValidator
	called = false
	optsAdmitted := Options{
		Dir:                repo,
		Paths:              []string{"custom.txt"},
		Message:            "feat: custom validator admission (fak safecommit)",
		AuthorityFence:     &AuthorityFence{Owner: "authorized-worker"},
		AuthorityValidator: customValidator,
	}
	resAdmitted, err := Commit(context.Background(), optsAdmitted)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("custom validator was not called")
	}
	if !resAdmitted.Committed || !resAdmitted.Verified {
		t.Fatalf("expected success, got committed=%v verified=%v reason=%q", resAdmitted.Committed, resAdmitted.Verified, resAdmitted.Reason)
	}
	if resAdmitted.Authority == nil || resAdmitted.Authority.Outcome != OutcomeAdmitted {
		t.Fatalf("expected OutcomeAdmitted, got %+v", resAdmitted.Authority)
	}

	// 3. Error case via validator
	writeTempRepoFile(t, filepath.Join(repo, "custom.txt"), "subsequent change for error case\n")
	errValidator := func(ctx context.Context, fence AuthorityFence, paths []string) (AuthorityReceipt, error) {
		return AuthorityReceipt{}, errors.New("connection failed to auth service")
	}
	optsErr := Options{
		Dir:                repo,
		Paths:              []string{"custom.txt"},
		Message:            "feat: error case (fak safecommit)",
		AuthorityFence:     &AuthorityFence{Owner: "authorized-worker"},
		AuthorityValidator: errValidator,
	}
	resErr, err := Commit(context.Background(), optsErr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resErr.Reason != ReasonAuthorityUnavailable {
		t.Fatalf("reason = %q, want %q", resErr.Reason, ReasonAuthorityUnavailable)
	}
}

func TestAuthorityExitCode(t *testing.T) {
	for _, reason := range []string{ReasonAuthorityStale, ReasonAuthorityForeign, ReasonAuthorityUnavailable} {
		code, ok := AuthorityExitCode(reason)
		if !ok || code != ExitRefused {
			t.Errorf("AuthorityExitCode(%q) = (%d, %v), want (%d, true)", reason, code, ok, ExitRefused)
		}
	}
}
