package kvmmu

import (
	"reflect"
	"testing"
	"time"
)

// TestKVMMU_BranchForkZeroCopy creates root with 64 tokens (4 pages), forks 4 speculative branches
// (Medusa/Eagle candidates), and verifies all 4 pages have RefCount == 5 and page IDs are identical.
func TestKVMMU_BranchForkZeroCopy(t *testing.T) {
	pool := NewPagePool()
	root := NewV2PTable(pool, "root")

	// 64 tokens spanning exactly 4 physical pages (16 tokens per page)
	tokens := make([]int, 64)
	for i := 0; i < 64; i++ {
		tokens[i] = i + 1
	}

	if err := root.AppendTokens(tokens); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	if root.TotalTokens != 64 {
		t.Fatalf("expected 64 tokens in root, got %d", root.TotalTokens)
	}
	if len(root.PageIDs) != 4 {
		t.Fatalf("expected 4 pages in root, got %d", len(root.PageIDs))
	}
	if pool.ActivePageCount() != 4 {
		t.Fatalf("expected 4 active pages in pool, got %d", pool.ActivePageCount())
	}

	// Verify root initially has RefCount == 1 on all pages
	for idx, id := range root.PageIDs {
		p := pool.Page(id)
		if p == nil {
			t.Fatalf("page at index %d (id %d) is nil", idx, id)
		}
		if p.RefCount.Load() != 1 {
			t.Fatalf("expected initial RefCount 1 on page %d, got %d", id, p.RefCount.Load())
		}
	}

	// Fork 4 speculative branches (Medusa/Eagle candidates)
	branchNames := []string{"medusa_cand_0", "medusa_cand_1", "eagle_cand_0", "eagle_cand_1"}
	branches := make([]*V2PTable, 4)

	for i, name := range branchNames {
		start := time.Now()
		b, err := root.ForkBranch(name)
		duration := time.Since(start)
		if err != nil {
			t.Fatalf("ForkBranch %s failed: %v", name, err)
		}
		if duration > 50*time.Millisecond {
			t.Errorf("fork took unusually long: %v (expected < 50µs)", duration)
		}
		branches[i] = b
	}

	// Verify all 4 pages have RefCount == 5 (root + 4 branches)
	for idx, id := range root.PageIDs {
		p := pool.Page(id)
		if p == nil {
			t.Fatalf("page %d not found in pool", id)
		}
		refCount := p.RefCount.Load()
		if refCount != 5 {
			t.Fatalf("page index %d (id %d) expected RefCount 5, got %d", idx, id, refCount)
		}
	}

	// Verify all 4 branches have identical page IDs to root
	for i, b := range branches {
		if b.ParentID != "root" {
			t.Errorf("branch %s expected ParentID 'root', got '%s'", branchNames[i], b.ParentID)
		}
		if b.BranchID != branchNames[i] {
			t.Errorf("branch %s expected BranchID '%s', got '%s'", branchNames[i], branchNames[i], b.BranchID)
		}
		if b.TotalTokens != 64 {
			t.Errorf("branch %s expected TotalTokens 64, got %d", branchNames[i], b.TotalTokens)
		}
		if len(b.PageIDs) != 4 {
			t.Fatalf("branch %s expected 4 page IDs, got %d", branchNames[i], len(b.PageIDs))
		}
		for pageIdx := 0; pageIdx < 4; pageIdx++ {
			if b.PageIDs[pageIdx] != root.PageIDs[pageIdx] {
				t.Fatalf("branch %s page %d ID mismatch: got %d, expected %d",
					branchNames[i], pageIdx, b.PageIDs[pageIdx], root.PageIDs[pageIdx])
			}
		}

		readTokens := b.ReadTokens()
		if !reflect.DeepEqual(readTokens, tokens) {
			t.Fatalf("branch %s tokens do not match root", branchNames[i])
		}
	}

	// Verify pool active page count is still 4 (pure zero-copy, no new allocations)
	if pool.ActivePageCount() != 4 {
		t.Fatalf("expected 4 active pages in pool, got %d", pool.ActivePageCount())
	}
}

// TestKVMMU_CopyOnWriteOnBranchMutation modifies branch 1 with distinct draft tokens and
// verifies branch 1 gets private pages while branch 2, 3, 4 and root preserve their shared pages and exact tokens.
func TestKVMMU_CopyOnWriteOnBranchMutation(t *testing.T) {
	t.Run("COWOnPartiallyFilledPage", func(t *testing.T) {
		pool := NewPagePool()
		root := NewV2PTable(pool, "root")

		// Root with 56 tokens (3 full pages of 16 + 1 page of 8 tokens, 4 pages total)
		rootTokens := make([]int, 56)
		for i := 0; i < 56; i++ {
			rootTokens[i] = i + 1
		}
		if err := root.AppendTokens(rootTokens); err != nil {
			t.Fatalf("AppendTokens failed: %v", err)
		}

		// Fork 4 branches
		b1, _ := root.ForkBranch("branch_1")
		b2, _ := root.ForkBranch("branch_2")
		b3, _ := root.ForkBranch("branch_3")
		b4, _ := root.ForkBranch("branch_4")

		sharedPage3ID := root.PageIDs[3]
		p3 := pool.Page(sharedPage3ID)
		if p3.RefCount.Load() != 5 {
			t.Fatalf("expected RefCount 5 on page 3, got %d", p3.RefCount.Load())
		}

		// Branch 1 drafts 12 tokens (8 fill the 4th page triggering COW, 4 allocate a 5th page)
		draftTokens := []int{201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212}
		if err := b1.AppendTokens(draftTokens); err != nil {
			t.Fatalf("b1.AppendTokens failed: %v", err)
		}

		// Verify branch 1 received private pages
		if len(b1.PageIDs) != 5 {
			t.Fatalf("b1 expected 5 pages, got %d", len(b1.PageIDs))
		}
		if b1.PageIDs[3] == sharedPage3ID {
			t.Fatalf("b1 page 3 should have been replaced via COW, but matches old shared ID %d", sharedPage3ID)
		}
		b1Page3 := pool.Page(b1.PageIDs[3])
		if b1Page3.RefCount.Load() != 1 {
			t.Fatalf("b1 COW page 3 should have RefCount 1, got %d", b1Page3.RefCount.Load())
		}
		b1Page4 := pool.Page(b1.PageIDs[4])
		if b1Page4.RefCount.Load() != 1 {
			t.Fatalf("b1 new page 4 should have RefCount 1, got %d", b1Page4.RefCount.Load())
		}

		// Verify old shared page 3 refcount decremented from 5 to 4
		if p3.RefCount.Load() != 4 {
			t.Fatalf("expected old shared page 3 RefCount 4, got %d", p3.RefCount.Load())
		}

		// Pages 0, 1, 2 are still shared by all 5 tables (root + 4 branches)
		for i := 0; i < 3; i++ {
			p := pool.Page(root.PageIDs[i])
			if p.RefCount.Load() != 5 {
				t.Fatalf("expected shared page %d RefCount 5, got %d", i, p.RefCount.Load())
			}
		}

		// Verify root, branch 2, 3, 4 preserve exact original tokens and page IDs
		for _, table := range []*V2PTable{root, b2, b3, b4} {
			if len(table.PageIDs) != 4 {
				t.Fatalf("table %s expected 4 pages, got %d", table.BranchID, len(table.PageIDs))
			}
			if table.PageIDs[3] != sharedPage3ID {
				t.Fatalf("table %s page 3 changed from %d to %d", table.BranchID, sharedPage3ID, table.PageIDs[3])
			}
			tokens := table.ReadTokens()
			if !reflect.DeepEqual(tokens, rootTokens) {
				t.Fatalf("table %s tokens corrupted after b1 mutation", table.BranchID)
			}
		}

		// Verify branch 1 tokens: 56 original tokens + 12 draft tokens
		expectedB1Tokens := append(append([]int{}, rootTokens...), draftTokens...)
		b1Tokens := b1.ReadTokens()
		if !reflect.DeepEqual(b1Tokens, expectedB1Tokens) {
			t.Fatalf("b1 tokens mismatch: expected %d tokens, got %d", len(expectedB1Tokens), len(b1Tokens))
		}
	})

	t.Run("COWOnFullPageBoundary", func(t *testing.T) {
		pool := NewPagePool()
		root := NewV2PTable(pool, "root")

		// 64 tokens (4 full pages)
		rootTokens := make([]int, 64)
		for i := 0; i < 64; i++ {
			rootTokens[i] = i + 1
		}
		_ = root.AppendTokens(rootTokens)

		b1, _ := root.ForkBranch("branch_1")
		b2, _ := root.ForkBranch("branch_2")
		b3, _ := root.ForkBranch("branch_3")
		b4, _ := root.ForkBranch("branch_4")

		// Branch 1 appends 4 draft tokens (allocates a private 5th page)
		draftTokens := []int{301, 302, 303, 304}
		if err := b1.AppendTokens(draftTokens); err != nil {
			t.Fatalf("b1.AppendTokens failed: %v", err)
		}

		if len(b1.PageIDs) != 5 {
			t.Fatalf("b1 expected 5 pages, got %d", len(b1.PageIDs))
		}
		b1Page4 := pool.Page(b1.PageIDs[4])
		if b1Page4.RefCount.Load() != 1 {
			t.Fatalf("b1 page 4 expected RefCount 1, got %d", b1Page4.RefCount.Load())
		}

		// Root, b2, b3, b4 preserve all 4 shared pages
		for _, table := range []*V2PTable{root, b2, b3, b4} {
			if len(table.PageIDs) != 4 {
				t.Fatalf("table %s expected 4 pages, got %d", table.BranchID, len(table.PageIDs))
			}
			if !reflect.DeepEqual(table.ReadTokens(), rootTokens) {
				t.Fatalf("table %s tokens corrupted", table.BranchID)
			}
		}
	})
}

// TestKVMMU_BranchSquashReleasesMemory squashes branches 2, 3, 4, verifies physical page
// refcounts decrement cleanly, then squashes branch 1 and verifies private pages are returned to pool.
func TestKVMMU_BranchSquashReleasesMemory(t *testing.T) {
	pool := NewPagePool()
	root := NewV2PTable(pool, "root")

	// 56 tokens: 4 pages (3 full + 1 partial)
	rootTokens := make([]int, 56)
	for i := 0; i < 56; i++ {
		rootTokens[i] = i + 1
	}
	_ = root.AppendTokens(rootTokens)

	b1, _ := root.ForkBranch("b1")
	b2, _ := root.ForkBranch("b2")
	b3, _ := root.ForkBranch("b3")
	b4, _ := root.ForkBranch("b4")

	// b1 drafts 12 tokens, creating 2 private pages (one via COW, one new page)
	draft := []int{101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112}
	_ = b1.AppendTokens(draft)

	sharedP0 := pool.Page(root.PageIDs[0])
	sharedP3 := pool.Page(root.PageIDs[3]) // Old shared 4th page
	b1PrivateP3 := pool.Page(b1.PageIDs[3])
	b1PrivateP4 := pool.Page(b1.PageIDs[4])

	// Total active pages: 4 (root) + 2 (b1 private) = 6 pages
	if pool.ActivePageCount() != 6 {
		t.Fatalf("expected 6 active pages, got %d", pool.ActivePageCount())
	}

	// Step 1: Squash branches 2, 3, 4
	start := time.Now()
	for _, b := range []*V2PTable{b2, b3, b4} {
		if err := b.Squash(); err != nil {
			t.Fatalf("Squash failed: %v", err)
		}
		if b.TotalTokens != 0 || len(b.PageIDs) != 0 {
			t.Fatalf("branch not cleanly reset after squash: %v", b)
		}
	}
	squashDuration := time.Since(start)
	if squashDuration > 50*time.Millisecond {
		t.Errorf("squash took unusually long: %v (expected < 50µs)", squashDuration)
	}

	// Verifies physical page refcounts decrement cleanly:
	// Page 0 was referenced by root + b1 + b2 + b3 + b4 (5). With b2, b3, b4 squashed, it's held by root + b1 (2).
	if sharedP0.RefCount.Load() != 2 {
		t.Fatalf("expected shared page 0 RefCount 2, got %d", sharedP0.RefCount.Load())
	}
	// Old Page 3 was referenced by root + b2 + b3 + b4 (4). With b2, b3, b4 squashed, it's held by root only (1).
	if sharedP3.RefCount.Load() != 1 {
		t.Fatalf("expected shared page 3 RefCount 1, got %d", sharedP3.RefCount.Load())
	}

	// b1 private pages are untouched
	if b1PrivateP3.RefCount.Load() != 1 || b1PrivateP4.RefCount.Load() != 1 {
		t.Fatalf("b1 private pages unexpectedly modified")
	}
	if pool.ActivePageCount() != 6 {
		t.Fatalf("expected 6 active pages before squashing b1, got %d", pool.ActivePageCount())
	}

	// Step 2: Squash branch 1; verifies private pages are returned to pool
	if err := b1.Squash(); err != nil {
		t.Fatalf("b1.Squash failed: %v", err)
	}
	if b1.TotalTokens != 0 || len(b1.PageIDs) != 0 {
		t.Fatalf("b1 not cleanly reset after squash")
	}

	// b1 private pages should now have RefCount == 0
	if b1PrivateP3.RefCount.Load() != 0 {
		t.Fatalf("b1 private page 3 RefCount expected 0, got %d", b1PrivateP3.RefCount.Load())
	}
	if b1PrivateP4.RefCount.Load() != 0 {
		t.Fatalf("b1 private page 4 RefCount expected 0, got %d", b1PrivateP4.RefCount.Load())
	}

	// Page 0 RefCount dropped from 2 to 1 (only root holds it)
	if sharedP0.RefCount.Load() != 1 {
		t.Fatalf("shared page 0 RefCount expected 1, got %d", sharedP0.RefCount.Load())
	}

	// All 4 root pages have RefCount == 1
	for idx, id := range root.PageIDs {
		p := pool.Page(id)
		if p.RefCount.Load() != 1 {
			t.Fatalf("root page %d (id %d) expected RefCount 1, got %d", idx, id, p.RefCount.Load())
		}
	}

	// Pool active page count must be back to 4 (the 2 private pages were freed back to pool)
	if pool.ActivePageCount() != 4 {
		t.Fatalf("expected 4 active pages in pool after b1 squash, got %d", pool.ActivePageCount())
	}

	// Root tokens are completely intact
	if !reflect.DeepEqual(root.ReadTokens(), rootTokens) {
		t.Fatalf("root tokens corrupted after branch squashes")
	}
}

// TestKVMMU_CommitPromotesAuthoritative verifies that when a winning branch commits,
// root updates to the winning branch and token integrity is strictly preserved.
func TestKVMMU_CommitPromotesAuthoritative(t *testing.T) {
	pool := NewPagePool()
	root := NewV2PTable(pool, "root")

	// Prompt: 20 tokens (spans 2 pages: 16 + 4)
	prompt := make([]int, 20)
	for i := 0; i < 20; i++ {
		prompt[i] = i + 10
	}
	_ = root.AppendTokens(prompt)

	// Speculative candidate branches
	cand1, _ := root.ForkBranch("cand_medusa_0")
	cand2, _ := root.ForkBranch("cand_medusa_1")

	draft1 := []int{101, 102, 103, 104, 105}
	draft2 := []int{201, 202}

	_ = cand1.AppendTokens(draft1)
	_ = cand2.AppendTokens(draft2)

	// Losing branch cand2 squashes
	if err := cand2.Squash(); err != nil {
		t.Fatalf("cand2.Squash failed: %v", err)
	}

	// Winning branch cand1 commits
	if err := cand1.Commit(); err != nil {
		t.Fatalf("cand1.Commit failed: %v", err)
	}
	if !cand1.Authoritative {
		t.Fatalf("expected cand1.Authoritative == true")
	}

	// Root updates to winning branch
	if err := root.UpdateFrom(cand1); err != nil {
		t.Fatalf("root.UpdateFrom failed: %v", err)
	}
	if !root.Authoritative {
		t.Fatalf("expected root.Authoritative == true")
	}

	// Verify token integrity
	expectedTokens := append(append([]int{}, prompt...), draft1...)
	rootTokens := root.ReadTokens()
	if !reflect.DeepEqual(rootTokens, expectedTokens) {
		t.Fatalf("root tokens do not match winning branch:\ngot:  %v\nwant: %v", rootTokens, expectedTokens)
	}

	// cand1 can now safely be squashed without affecting root
	if err := cand1.Squash(); err != nil {
		t.Fatalf("cand1.Squash failed: %v", err)
	}
	if !reflect.DeepEqual(root.ReadTokens(), expectedTokens) {
		t.Fatalf("root tokens corrupted after squashing committed branch")
	}

	// Root can continue appending subsequent tokens
	furtherTokens := []int{998, 999}
	if err := root.AppendTokens(furtherTokens); err != nil {
		t.Fatalf("root.AppendTokens failed: %v", err)
	}
	expectedFurther := append(expectedTokens, furtherTokens...)
	if !reflect.DeepEqual(root.ReadTokens(), expectedFurther) {
		t.Fatalf("root tokens mismatch after further append")
	}
}

func BenchmarkKVMMU_ForkBranch(b *testing.B) {
	pool := NewPagePool()
	root := NewV2PTable(pool, "root")
	tokens := make([]int, 64)
	_ = root.AppendTokens(tokens)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		child, _ := root.ForkBranch("bench_fork")
		_ = child.Squash()
	}
}

func BenchmarkKVMMU_Squash(b *testing.B) {
	pool := NewPagePool()
	root := NewV2PTable(pool, "root")
	tokens := make([]int, 64)
	_ = root.AppendTokens(tokens)

	children := make([]*V2PTable, b.N)
	for i := 0; i < b.N; i++ {
		children[i], _ = root.ForkBranch("bench_squash")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = children[i].Squash()
	}
}
