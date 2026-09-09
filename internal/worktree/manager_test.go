package worktree

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

type mockGitCall struct {
	Dir  string
	Args []string
}

func TestWorktree_AllocateAndDeallocateMock(t *testing.T) {
	var mu sync.Mutex
	var calls []mockGitCall

	mockRunner := func(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
		mu.Lock()
		calls = append(calls, mockGitCall{Dir: dir, Args: args})
		mu.Unlock()

		if len(args) > 0 && args[0] == "rev-parse" {
			return "c001cafe\n", "", nil
		}
		return "", "", nil
	}

	tmpDir := t.TempDir()
	mgr := NewManager(tmpDir, WithRunner(mockRunner))

	ctx := context.Background()
	wtCtx, err := mgr.Allocate(ctx, "issue-42", "c001cafe")
	if err != nil {
		t.Fatalf("allocate failed: %v", err)
	}

	if wtCtx.TicketID != "42" {
		t.Fatalf("expected ticket ID 42, got %s", wtCtx.TicketID)
	}
	if wtCtx.Branch != "fak/ticket-42" {
		t.Fatalf("expected branch fak/ticket-42, got %s", wtCtx.Branch)
	}
	if wtCtx.BaseCommit != "c001cafe" {
		t.Fatalf("expected commit c001cafe, got %s", wtCtx.BaseCommit)
	}

	// Deallocate
	if err := mgr.Deallocate(ctx, "issue-42", false); err != nil {
		t.Fatalf("deallocate failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	var hasWorktreeAdd, hasWorktreeRemove, hasBranchDelete bool
	for _, c := range calls {
		if len(c.Args) >= 2 && c.Args[0] == "worktree" && c.Args[1] == "add" {
			hasWorktreeAdd = true
		}
		if len(c.Args) >= 2 && c.Args[0] == "worktree" && c.Args[1] == "remove" {
			hasWorktreeRemove = true
		}
		if len(c.Args) >= 2 && c.Args[0] == "branch" && c.Args[1] == "-D" {
			hasBranchDelete = true
		}
	}

	if !hasWorktreeAdd {
		t.Errorf("expected git worktree add call")
	}
	if !hasWorktreeRemove {
		t.Errorf("expected git worktree remove call")
	}
	if !hasBranchDelete {
		t.Errorf("expected git branch -D call")
	}
}

func TestWorktree_JanitorCleansOrphans(t *testing.T) {
	tmpDir := t.TempDir()
	worktreesDir := filepath.Join(tmpDir, ".worktrees")
	if err := os.MkdirAll(worktreesDir, 0755); err != nil {
		t.Fatalf("failed to create worktrees dir: %v", err)
	}

	// Create 3 mock worktree directories
	dirs := []string{"ticket-101", "ticket-102", "ticket-103"}
	for _, d := range dirs {
		p := filepath.Join(worktreesDir, d)
		if err := os.MkdirAll(p, 0755); err != nil {
			t.Fatalf("failed to create dir %s: %v", d, err)
		}
	}

	mockRunner := func(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
		return "", "", nil
	}

	mgr := NewManager(tmpDir, WithRunner(mockRunner))

	// Only ticket-102 has an active, non-expired contract
	liveContracts := []leaseref.ContractRecord{
		{
			TicketID:   "ticket-102",
			State:      leaseref.ContractStateExecuting,
			AcquiredAt: time.Now().Unix(),
			TTLSeconds: 3600,
		},
		{
			TicketID:   "ticket-103",
			State:      leaseref.ContractStateSucceeded, // Terminal! Should be cleaned
			AcquiredAt: time.Now().Unix(),
			TTLSeconds: 3600,
		},
		// ticket-101 has no contract at all (orphan)
	}

	ctx := context.Background()
	report, err := mgr.Janitor(ctx, liveContracts)
	if err != nil {
		t.Fatalf("janitor failed: %v", err)
	}

	if report.ScannedCount != 3 {
		t.Fatalf("expected 3 scanned, got %d", report.ScannedCount)
	}
	if report.RetainedCount != 1 || len(report.Retained) != 1 || report.Retained[0] != "102" {
		t.Fatalf("expected ticket-102 retained, got %+v", report.Retained)
	}
	if report.CleanedCount != 2 {
		t.Fatalf("expected 2 cleaned, got %d (%+v)", report.CleanedCount, report.Cleaned)
	}
}

func TestWorktree_RealGit_NoHeadModification(t *testing.T) {
	// Verify git is installed
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git binary not available in PATH")
	}

	tmpDir := t.TempDir()

	// Initialize real git repo
	runGit := func(dir string, args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s failed: %v (output: %s)", strings.Join(args, " "), err, string(out))
		}
		return strings.TrimSpace(string(out))
	}

	runGit(tmpDir, "init", "-b", "main")
	runGit(tmpDir, "config", "user.name", "Test User")
	runGit(tmpDir, "config", "user.email", "test@example.com")
	runGit(tmpDir, "config", "commit.gpgsign", "false")

	// Create initial commit
	readme := filepath.Join(tmpDir, "README.md")
	if err := os.WriteFile(readme, []byte("hello world\n"), 0644); err != nil {
		t.Fatalf("write failed: %v", err)
	}
	runGit(tmpDir, "add", "README.md")
	runGit(tmpDir, "commit", "-m", "initial commit")

	initialHEAD := runGit(tmpDir, "rev-parse", "HEAD")
	initialBranch := runGit(tmpDir, "rev-parse", "--abbrev-ref", "HEAD")

	mgr := NewManager(tmpDir)
	ctx := context.Background()

	// Allocate ephemeral worktree
	wtCtx, err := mgr.Allocate(ctx, "issue-555", initialHEAD)
	if err != nil {
		t.Fatalf("allocate failed: %v", err)
	}

	// Verify worktree exists and contains README.md
	wtReadme := filepath.Join(wtCtx.Path, "README.md")
	if _, err := os.Stat(wtReadme); err != nil {
		t.Fatalf("expected README.md in worktree: %v", err)
	}

	// VERIFY: parent repo HEAD and branch are UNTOUCHED!
	postAllocHEAD := runGit(tmpDir, "rev-parse", "HEAD")
	postAllocBranch := runGit(tmpDir, "rev-parse", "--abbrev-ref", "HEAD")
	if postAllocHEAD != initialHEAD {
		t.Fatalf("parent HEAD modified during allocate: got %s, expected %s", postAllocHEAD, initialHEAD)
	}
	if postAllocBranch != initialBranch {
		t.Fatalf("parent branch modified during allocate: got %s, expected %s", postAllocBranch, initialBranch)
	}

	// Commit work inside the worktree
	wtNewFile := filepath.Join(wtCtx.Path, "ticket.txt")
	if err := os.WriteFile(wtNewFile, []byte("ticket 555 fix\n"), 0644); err != nil {
		t.Fatalf("write failed: %v", err)
	}
	runGit(wtCtx.Path, "add", "ticket.txt")
	runGit(wtCtx.Path, "commit", "-m", "fix issue 555")

	// Parent repo HEAD must STILL be untouched!
	postCommitHEAD := runGit(tmpDir, "rev-parse", "HEAD")
	if postCommitHEAD != initialHEAD {
		t.Fatalf("parent HEAD modified by worktree commit: got %s, expected %s", postCommitHEAD, initialHEAD)
	}

	// Deallocate worktree
	if err := mgr.Deallocate(ctx, "issue-555", false); err != nil {
		t.Fatalf("deallocate failed: %v", err)
	}

	// Parent repo HEAD must STILL be untouched!
	finalHEAD := runGit(tmpDir, "rev-parse", "HEAD")
	finalBranch := runGit(tmpDir, "rev-parse", "--abbrev-ref", "HEAD")
	if finalHEAD != initialHEAD || finalBranch != initialBranch {
		t.Fatalf("parent HEAD/branch modified after deallocate: got %s/%s", finalHEAD, finalBranch)
	}

	// Worktree directory must no longer exist
	if _, err := os.Stat(wtCtx.Path); !os.IsNotExist(err) {
		t.Fatalf("expected worktree directory to be removed, but still exists")
	}
}

func TestWorktreeContext_EnvList(t *testing.T) {
	emptyCtx := &WorktreeContext{}
	if emptyCtx.EnvList() != nil {
		t.Fatalf("expected nil for empty env, got %+v", emptyCtx.EnvList())
	}

	wtCtx := &WorktreeContext{
		Env: map[string]string{
			"FOO": "bar",
			"BAZ": "qux",
		},
	}
	list := wtCtx.EnvList()
	if len(list) != 2 {
		t.Fatalf("expected 2 items, got %d", len(list))
	}
	hasFoo := false
	hasBaz := false
	for _, item := range list {
		if item == "FOO=bar" {
			hasFoo = true
		}
		if item == "BAZ=qux" {
			hasBaz = true
		}
	}
	if !hasFoo || !hasBaz {
		t.Fatalf("unexpected EnvList contents: %+v", list)
	}
}

func TestWorktree_Release(t *testing.T) {
	var calls []string
	mockRunner := func(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
		calls = append(calls, strings.Join(args, " "))
		return "", "", nil
	}
	tmpDir := t.TempDir()
	mgr := NewManager(tmpDir, WithRunner(mockRunner))
	ctx := context.Background()

	if err := mgr.Release(ctx, "issue-99"); err != nil {
		t.Fatalf("release failed: %v", err)
	}

	var hasRemove, hasBranchDelete bool
	for _, call := range calls {
		if strings.Contains(call, "worktree remove") {
			hasRemove = true
		}
		if strings.Contains(call, "branch -D") {
			hasBranchDelete = true
		}
	}
	if !hasRemove || !hasBranchDelete {
		t.Fatalf("expected worktree remove and branch -D, got: %+v", calls)
	}
}

func TestWorktree_Sweep(t *testing.T) {
	tmpDir := t.TempDir()
	worktreesDir := filepath.Join(tmpDir, ".worktrees")
	if err := os.MkdirAll(filepath.Join(worktreesDir, "ticket-1"), 0755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}

	mockRunner := func(ctx context.Context, dir string, env []string, args ...string) (string, string, error) {
		return "", "", nil
	}
	mgr := NewManager(tmpDir, WithRunner(mockRunner))
	ctx := context.Background()

	contracts := []leaseref.ContractRecord{
		{
			TicketID:   "ticket-1",
			State:      leaseref.ContractStateExecuting,
			AcquiredAt: time.Now().Unix(),
			TTLSeconds: 3600,
		},
	}

	report, err := mgr.Sweep(ctx, contracts)
	if err != nil {
		t.Fatalf("sweep failed: %v", err)
	}
	if report.RetainedCount != 1 || report.CleanedCount != 0 {
		t.Fatalf("unexpected sweep report: %+v", report)
	}
}
