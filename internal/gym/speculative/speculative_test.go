package speculative

import (
	"context"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gym"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// TestSpeculativeBranchForkCommitAndRollback verifies lockstep speculative execution:
// - Branch A writes broken code, detects test failure, aborts cleanly with 0 file leaks and KV eviction.
// - Branch B writes valid code, passes tests, commits cleanly with promotion to base workspace and KV retention.
func TestSpeculativeBranchForkCommitAndRollback(t *testing.T) {
	ctx := context.Background()
	baseDir := t.TempDir()

	// 1. Setup base workspace with a sample Go file
	sampleGo := filepath.Join(baseDir, "math.go")
	initialContent := `package math

func Add(a, b int) int {
	return a + b
}
`
	if err := os.WriteFile(sampleGo, []byte(initialContent), 0644); err != nil {
		t.Fatalf("failed writing sample Go file: %v", err)
	}

	cfg := gym.Config{
		BaseDir:       baseDir,
		WorkspaceName: "speculative-root-arena",
	}
	parent, err := gym.Create(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create parent arena: %v", err)
	}
	defer parent.Destroy()

	// 2. Initialize radixkv.New(50000)
	tree := radixkv.New(50000)
	baseTokens := []int{101, 102, 103, 104}

	// 3. Fork Branch A and Branch B in lockstep with respective speculative token sequences
	specTokensA := []int{201, 202, 203}
	specTokensB := []int{301, 302, 303, 304}

	forkStartA := time.Now()
	branchA, err := Fork(ctx, parent, tree, baseTokens, specTokensA)
	if err != nil {
		t.Fatalf("failed to fork branch A: %v", err)
	}
	forkDurationA := time.Since(forkStartA)
	t.Logf("Fork A duration: %v", forkDurationA)

	forkStartB := time.Now()
	branchB, err := Fork(ctx, parent, tree, baseTokens, specTokensB)
	if err != nil {
		t.Fatalf("failed to fork branch B: %v", err)
	}
	forkDurationB := time.Since(forkStartB)
	t.Logf("Fork B duration: %v", forkDurationB)

	// Verify both branches are active
	if branchA.Status != Active {
		t.Fatalf("expected branch A to be Active, got %v", branchA.Status)
	}
	if branchB.Status != Active {
		t.Fatalf("expected branch B to be Active, got %v", branchB.Status)
	}

	// Verify tree has baseTokens + specTokensA + specTokensB
	fullTokensA := append(append([]int(nil), baseTokens...), specTokensA...)
	fullTokensB := append(append([]int(nil), baseTokens...), specTokensB...)

	if m := tree.MatchLen(fullTokensA); m != len(fullTokensA) {
		t.Fatalf("branch A tokens not matched in tree: %d/%d", m, len(fullTokensA))
	}
	if m := tree.MatchLen(fullTokensB); m != len(fullTokensB) {
		t.Fatalf("branch B tokens not matched in tree: %d/%d", m, len(fullTokensB))
	}

	expectedInitialTokens := len(baseTokens) + len(specTokensA) + len(specTokensB)
	if st := tree.Stats().Tokens; st != expectedInitialTokens {
		t.Fatalf("expected initial tree tokens %d, got %d", expectedInitialTokens, st)
	}

	// 4. Branch A writes a broken change (syntax error), detects failure, and aborts
	brokenFile := filepath.Join(branchA.Path(), "broken.go")
	brokenContent := `package math

func InvalidSyntax() {
	return ?????
}
`
	if err := os.WriteFile(brokenFile, []byte(brokenContent), 0644); err != nil {
		t.Fatalf("failed writing broken file in branch A: %v", err)
	}

	// Detect test / syntax failure
	fsetA := token.NewFileSet()
	_, parseErrA := parser.ParseDir(fsetA, branchA.Path(), nil, parser.AllErrors)
	if parseErrA == nil {
		t.Fatal("expected syntax error in branch A, but ParseDir succeeded")
	}

	// Abort Branch A
	abortStart := time.Now()
	if err := branchA.Abort(ctx); err != nil {
		t.Fatalf("branchA.Abort() failed: %v", err)
	}
	abortDuration := time.Since(abortStart)
	t.Logf("Abort A duration: %v", abortDuration)

	// Verify Abort() completed and status is Aborted
	if branchA.Status != Aborted {
		t.Fatalf("expected branch A status to be Aborted, got %v", branchA.Status)
	}

	// Verify Branch A's files do NOT exist on the host/parent workspace (0 residual files)
	if _, err := os.Stat(filepath.Join(baseDir, "broken.go")); !os.IsNotExist(err) {
		t.Fatalf("broken.go leaked to base workspace! err=%v", err)
	}
	if _, err := os.Stat(filepath.Join(parent.Path(), "broken.go")); !os.IsNotExist(err) {
		t.Fatalf("broken.go leaked to parent arena! err=%v", err)
	}

	// Verify Branch A's speculative tokens are evicted from KVTree (tokens returned to pre-Branch A level)
	expectedTokensAfterAAbort := len(baseTokens) + len(specTokensB)
	if st := tree.Stats().Tokens; st != expectedTokensAfterAAbort {
		t.Fatalf("expected tokens %d after branch A abort, got %d", expectedTokensAfterAAbort, st)
	}
	if m := tree.MatchLen(fullTokensA); m != len(baseTokens) {
		t.Fatalf("expected branch A speculative tokens to be evicted; matched %d, want base %d", m, len(baseTokens))
	}
	// Branch B must remain intact
	if m := tree.MatchLen(fullTokensB); m != len(fullTokensB) {
		t.Fatalf("branch B tokens damaged by branch A abort; matched %d/%d", m, len(fullTokensB))
	}

	// 5. Branch B writes a valid change (working code), passes test, and commits
	validFile := filepath.Join(branchB.Path(), "multiply.go")
	validContent := `package math

func Multiply(a, b int) int {
	return a * b
}
`
	if err := os.WriteFile(validFile, []byte(validContent), 0644); err != nil {
		t.Fatalf("failed writing valid file in branch B: %v", err)
	}

	// Verify code passes validation
	fsetB := token.NewFileSet()
	_, parseErrB := parser.ParseDir(fsetB, branchB.Path(), nil, parser.AllErrors)
	if parseErrB != nil {
		t.Fatalf("unexpected syntax error in branch B: %v", parseErrB)
	}

	tokensBeforeBCommit := tree.Stats().Tokens

	// Commit Branch B
	commitStart := time.Now()
	if err := branchB.Commit(ctx, "operator-lease-1"); err != nil {
		t.Fatalf("branchB.Commit() failed: %v", err)
	}
	commitDuration := time.Since(commitStart)
	t.Logf("Commit B duration: %v", commitDuration)

	if branchB.Status != Committed {
		t.Fatalf("expected branch B status to be Committed, got %v", branchB.Status)
	}

	// Verify Branch B's files are promoted to the base workspace
	promotedFile := filepath.Join(baseDir, "multiply.go")
	data, err := os.ReadFile(promotedFile)
	if err != nil {
		t.Fatalf("failed reading promoted file in base directory: %v", err)
	}
	if string(data) != validContent {
		t.Fatalf("promoted content mismatch: got %q, want %q", string(data), validContent)
	}

	// Verify parent arena also reflects promoted file
	parentPromotedFile := filepath.Join(parent.Path(), "multiply.go")
	parentData, err := os.ReadFile(parentPromotedFile)
	if err != nil {
		t.Fatalf("failed reading promoted file in parent arena: %v", err)
	}
	if string(parentData) != validContent {
		t.Fatalf("parent promoted content mismatch: got %q, want %q", string(parentData), validContent)
	}

	// Verify Branch B's tokens are retained in KVTree
	tokensAfterBCommit := tree.Stats().Tokens
	if tokensAfterBCommit != tokensBeforeBCommit {
		t.Fatalf("expected tokens %d after commit, got %d", tokensBeforeBCommit, tokensAfterBCommit)
	}
	if m := tree.MatchLen(fullTokensB); m != len(fullTokensB) {
		t.Fatalf("branch B tokens lost after commit: matched %d/%d", m, len(fullTokensB))
	}
}

// TestConcurrentMultiBranchForkAndIsolatedAborts tests concurrent multi-branch forking,
// isolated execution, and mixed abort/commit workflows under contention.
func TestConcurrentMultiBranchForkAndIsolatedAborts(t *testing.T) {
	ctx := context.Background()
	baseDir := t.TempDir()

	anchor := filepath.Join(baseDir, "anchor.go")
	if err := os.WriteFile(anchor, []byte("package anchor\n\nconst Version = 1\n"), 0644); err != nil {
		t.Fatalf("failed writing anchor file: %v", err)
	}

	cfg := gym.Config{
		BaseDir:       baseDir,
		WorkspaceName: "concurrent-spec-arena",
	}
	parent, err := gym.Create(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create parent arena: %v", err)
	}
	defer parent.Destroy()

	tree := radixkv.New(50000)
	baseTokens := []int{10, 20, 30}

	const numBranches = 12
	var wg sync.WaitGroup
	errCh := make(chan error, numBranches*2)

	branches := make([]*Branch, numBranches)
	branchTokens := make([][]int, numBranches)

	for i := 0; i < numBranches; i++ {
		branchTokens[i] = []int{1000 + i*10, 1001 + i*10, 1002 + i*10}
	}

	// Concurrently fork all branches
	for i := 0; i < numBranches; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			b, err := Fork(ctx, parent, tree, baseTokens, branchTokens[idx])
			if err != nil {
				errCh <- fmt.Errorf("fork branch %d failed: %w", idx, err)
				return
			}
			branches[idx] = b

			// Write unique file in each branch
			f := filepath.Join(b.Path(), fmt.Sprintf("file_%d.txt", idx))
			if err := os.WriteFile(f, []byte(fmt.Sprintf("content_%d", idx)), 0644); err != nil {
				errCh <- fmt.Errorf("write file in branch %d failed: %w", idx, err)
			}
		}(i)
	}
	wg.Wait()

	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}

	// Concurrently abort all odd branches (1, 3, 5, 7, 9, 11) in isolated goroutines
	var wg2 sync.WaitGroup
	errCh2 := make(chan error, numBranches*2)

	for i := 1; i < numBranches; i++ {
		wg2.Add(1)
		go func(idx int) {
			defer wg2.Done()
			b := branches[idx]
			if err := b.Abort(ctx); err != nil {
				errCh2 <- fmt.Errorf("abort branch %d failed: %w", idx, err)
			}
		}(i)
	}
	wg2.Wait()

	close(errCh2)
	for err := range errCh2 {
		t.Fatal(err)
	}

	// Verify isolated aborts:
	// For all branches 1..numBranches-1:
	// - Status is Aborted.
	// - 0 file leaks to base workspace.
	// - Speculative tokens evicted from tree.
	for i := 1; i < numBranches; i++ {
		targetFile := filepath.Join(baseDir, fmt.Sprintf("file_%d.txt", i))
		fullTokens := append(append([]int(nil), baseTokens...), branchTokens[i]...)

		if _, err := os.Stat(targetFile); !os.IsNotExist(err) {
			t.Errorf("aborted branch %d file leaked to base workspace!", i)
		}
		if matched := tree.MatchLen(fullTokens); matched > len(baseTokens) {
			t.Errorf("aborted branch %d tokens not evicted from tree: matched=%d, want=%d", i, matched, len(baseTokens))
		}
		if branches[i].Status != Aborted {
			t.Errorf("branch %d status = %v, want Aborted", i, branches[i].Status)
		}
	}

	// Surviving branch 0 tokens must remain completely intact in KVTree
	fullTokens0 := append(append([]int(nil), baseTokens...), branchTokens[0]...)
	if matched := tree.MatchLen(fullTokens0); matched != len(fullTokens0) {
		t.Fatalf("surviving branch 0 tokens damaged by other aborts: matched=%d, want=%d", matched, len(fullTokens0))
	}

	// Winning branch 0 commits
	if err := branches[0].Commit(ctx, "worker-0"); err != nil {
		t.Fatalf("branch 0 commit failed: %v", err)
	}
	if branches[0].Status != Committed {
		t.Fatalf("branch 0 status = %v, want Committed", branches[0].Status)
	}

	// Verify branch 0 file promoted to base workspace
	targetFile0 := filepath.Join(baseDir, "file_0.txt")
	if _, err := os.Stat(targetFile0); err != nil {
		t.Fatalf("committed branch 0 file missing from base workspace: %v", err)
	}

	// Verify branch 0 tokens retained in tree
	if matched := tree.MatchLen(fullTokens0); matched != len(fullTokens0) {
		t.Fatalf("committed branch 0 tokens missing from tree: matched=%d, want=%d", matched, len(fullTokens0))
	}
}

// TestSpeculativeManager verifies branch lifecycle management through SpeculativeManager.
func TestSpeculativeManager(t *testing.T) {
	ctx := context.Background()
	baseDir := t.TempDir()

	cfg := gym.Config{
		BaseDir:       baseDir,
		WorkspaceName: "manager-arena",
	}
	parent, err := gym.Create(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create parent arena: %v", err)
	}
	defer parent.Destroy()

	tree := radixkv.New(50000)
	baseTokens := []int{1, 2, 3}

	mgr := NewManager(parent, tree, baseTokens)

	branch1, err := mgr.Fork(ctx, []int{10, 11})
	if err != nil {
		t.Fatalf("mgr.Fork 1 failed: %v", err)
	}
	branch2, err := mgr.Fork(ctx, []int{20, 21})
	if err != nil {
		t.Fatalf("mgr.Fork 2 failed: %v", err)
	}

	active := mgr.ActiveBranches()
	if len(active) != 2 {
		t.Fatalf("expected 2 active branches, got %d", len(active))
	}

	// Abort branch1
	if err := branch1.Abort(ctx); err != nil {
		t.Fatalf("branch1.Abort failed: %v", err)
	}

	activeAfterAbort := mgr.ActiveBranches()
	if len(activeAfterAbort) != 1 || activeAfterAbort[0].ID != branch2.ID {
		t.Fatalf("expected only branch 2 active after abort, got %d", len(activeAfterAbort))
	}

	// Commit branch2
	if err := branch2.Commit(ctx, "operator-manager"); err != nil {
		t.Fatalf("branch2.Commit failed: %v", err)
	}

	activeAfterCommit := mgr.ActiveBranches()
	if len(activeAfterCommit) != 0 {
		t.Fatalf("expected 0 active branches after commit, got %d", len(activeAfterCommit))
	}
}

// TestEdgeCases covers idempotency and invalid state transitions.
func TestEdgeCases(t *testing.T) {
	ctx := context.Background()
	baseDir := t.TempDir()

	cfg := gym.Config{
		BaseDir:       baseDir,
		WorkspaceName: "edgecase-arena",
	}
	parent, err := gym.Create(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create parent arena: %v", err)
	}
	defer parent.Destroy()

	tree := radixkv.New(50000)

	// Branch with empty speculative tokens
	branch, err := Fork(ctx, parent, tree, []int{1, 2}, nil)
	if err != nil {
		t.Fatalf("Fork with nil specTokens failed: %v", err)
	}

	// Abort is idempotent
	if err := branch.Abort(ctx); err != nil {
		t.Fatalf("first Abort failed: %v", err)
	}
	if err := branch.Abort(ctx); err != nil {
		t.Fatalf("second Abort (idempotent) failed: %v", err)
	}

	// Cannot commit aborted branch
	if err := branch.Commit(ctx, "op"); err == nil {
		t.Fatal("expected error committing aborted branch, got nil")
	}

	// Fork another branch to test commit then abort
	branch2, err := Fork(ctx, parent, tree, []int{1, 2}, []int{99})
	if err != nil {
		t.Fatalf("Fork branch2 failed: %v", err)
	}
	if err := branch2.Commit(ctx, "op"); err != nil {
		t.Fatalf("Commit branch2 failed: %v", err)
	}
	if err := branch2.Abort(ctx); err == nil {
		t.Fatal("expected error aborting committed branch, got nil")
	}
	if err := branch2.Commit(ctx, "op"); err == nil {
		t.Fatal("expected error re-committing committed branch, got nil")
	}
}
