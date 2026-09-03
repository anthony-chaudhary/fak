package agentopt

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestPriorityKVEviction(t *testing.T) {
	t.Run("root instructions and pinned tools remain resident under budget pressure", func(t *testing.T) {
		table := NewPriorityKVTable()

		// 1. Setup blocks across all priority rungs:
		// Root system prompt (1000 tokens, PriorityRoot, auto-pinned)
		rootBlock := NewRootBlock("sys-root", 1000)
		if err := table.AddBlock(rootBlock); err != nil {
			t.Fatalf("unexpected error adding root block: %v", err)
		}

		// Shared tool definition (500 tokens, PriorityTool, pinned)
		pinnedTool := NewToolBlock("tool-shared-search", 500, true)
		if err := table.AddBlock(pinnedTool); err != nil {
			t.Fatalf("unexpected error adding pinned tool: %v", err)
		}

		// Ephemeral tool definition (400 tokens, PriorityTool, unpinned)
		ephemeralTool := NewToolBlock("tool-ephemeral-calc", 400, false)
		if err := table.AddBlock(ephemeralTool); err != nil {
			t.Fatalf("unexpected error adding ephemeral tool: %v", err)
		}

		// Conversational turns (turn-1 older than turn-2)
		now := time.Now()
		turn1 := NewTurnBlock("turn-1", 600)
		turn1.LastAccess = now.Add(-10 * time.Minute)
		if err := table.AddBlock(turn1); err != nil {
			t.Fatalf("unexpected error adding turn 1: %v", err)
		}

		turn2 := NewTurnBlock("turn-2", 700)
		turn2.LastAccess = now.Add(-5 * time.Minute)
		if err := table.AddBlock(turn2); err != nil {
			t.Fatalf("unexpected error adding turn 2: %v", err)
		}

		// Tool / model outputs (large output should be pruned before small output)
		outputLarge := NewOutputBlock("output-large", 2000)
		outputLarge.LastAccess = now.Add(-2 * time.Minute)
		if err := table.AddBlock(outputLarge); err != nil {
			t.Fatalf("unexpected error adding large output: %v", err)
		}

		outputSmall := NewOutputBlock("output-small", 300)
		outputSmall.LastAccess = now.Add(-1 * time.Minute)
		if err := table.AddBlock(outputSmall); err != nil {
			t.Fatalf("unexpected error adding small output: %v", err)
		}

		initialTokens := 1000 + 500 + 400 + 600 + 700 + 2000 + 300 // 5500 tokens
		if table.TotalTokens() != initialTokens {
			t.Fatalf("expected initial tokens %d, got %d", initialTokens, table.TotalTokens())
		}
		if table.BlockCount() != 7 {
			t.Fatalf("expected 7 blocks, got %d", table.BlockCount())
		}

		// 2. Enforce moderate budget of 3400 tokens.
		// Expected prune sequence:
		// - outputLarge (2000 tokens, PriorityOutput) -> total becomes 3500 > 3400
		// - outputSmall (300 tokens, PriorityOutput)  -> total becomes 3200 <= 3400 (pruning stops)
		report1 := table.EnforceBudget(3400)

		if report1.PrunedNodesCount != 2 {
			t.Fatalf("expected 2 pruned blocks, got %d", report1.PrunedNodesCount)
		}
		if len(report1.RemovedNodes) != 2 || report1.RemovedNodes[0] != "output-large" || report1.RemovedNodes[1] != "output-small" {
			t.Fatalf("expected [output-large, output-small] removed, got %v", report1.RemovedNodes)
		}
		if table.TotalTokens() != 3200 {
			t.Fatalf("expected 3200 tokens remaining, got %d", table.TotalTokens())
		}

		// Verify root and pinned blocks remain resident
		if !table.HasBlock("sys-root") {
			t.Fatalf("root instruction block must remain resident")
		}
		if !table.HasBlock("tool-shared-search") {
			t.Fatalf("pinned tool block must remain resident")
		}
		if !table.HasBlock("tool-ephemeral-calc") {
			t.Fatalf("tool definition block must remain resident")
		}
		if !table.HasBlock("turn-1") || !table.HasBlock("turn-2") {
			t.Fatalf("turn blocks should remain resident under moderate budget")
		}

		// 3. Enforce tighter budget of 2000 tokens.
		// Previous remaining: 3200 (sys-root:1000, tool-shared:500, tool-ephemeral:400, turn-1:600, turn-2:700)
		// Expected prune sequence:
		// - turn-1 (600 tokens, older PriorityTurn) -> total becomes 2600 > 2000
		// - turn-2 (700 tokens, newer PriorityTurn) -> total becomes 1900 <= 2000 (pruning stops)
		report2 := table.EnforceBudget(2000)

		if report2.PrunedNodesCount != 2 {
			t.Fatalf("expected 2 pruned blocks, got %d", report2.PrunedNodesCount)
		}
		if len(report2.RemovedNodes) != 2 || report2.RemovedNodes[0] != "turn-1" || report2.RemovedNodes[1] != "turn-2" {
			t.Fatalf("expected [turn-1, turn-2] removed, got %v", report2.RemovedNodes)
		}
		if table.TotalTokens() != 1900 {
			t.Fatalf("expected 1900 tokens remaining, got %d", table.TotalTokens())
		}

		// Root and tool blocks remain resident
		if !table.HasBlock("sys-root") {
			t.Fatalf("root instruction block must remain resident")
		}
		if !table.HasBlock("tool-shared-search") {
			t.Fatalf("pinned tool block must remain resident")
		}
		if !table.HasBlock("tool-ephemeral-calc") {
			t.Fatalf("ephemeral tool block must remain resident")
		}

		// 4. Enforce extreme budget of 1200 tokens (less than sys-root + tool-shared = 1500).
		// Previous remaining: 1900 (sys-root:1000, tool-shared:500, tool-ephemeral:400)
		// Expected prune sequence:
		// - tool-ephemeral-calc (400 tokens, unpinned PriorityTool) -> total becomes 1500
		// - sys-root (PriorityRoot) and tool-shared-search (Pinned) cannot be pruned.
		// Pruning halts because all remaining blocks are immune.
		report3 := table.EnforceBudget(1200)

		if report3.PrunedNodesCount != 1 {
			t.Fatalf("expected 1 pruned block, got %d", report3.PrunedNodesCount)
		}
		if len(report3.RemovedNodes) != 1 || report3.RemovedNodes[0] != "tool-ephemeral-calc" {
			t.Fatalf("expected [tool-ephemeral-calc] removed, got %v", report3.RemovedNodes)
		}
		if table.TotalTokens() != 1500 {
			t.Fatalf("expected 1500 tokens remaining (immune blocks), got %d", table.TotalTokens())
		}
		if !table.HasBlock("sys-root") {
			t.Fatalf("root block must NEVER be pruned, even under impossible budget pressure")
		}
		if !table.HasBlock("tool-shared-search") {
			t.Fatalf("pinned tool block must NEVER be pruned")
		}
	})

	t.Run("touch block updates recency and alters pruning order", func(t *testing.T) {
		table := NewPriorityKVTable()

		baseTime := time.Now().Add(-1 * time.Hour)
		turnA := NewTurnBlock("turn-a", 500)
		turnA.LastAccess = baseTime
		_ = table.AddBlock(turnA)

		turnB := NewTurnBlock("turn-b", 500)
		turnB.LastAccess = baseTime.Add(10 * time.Minute)
		_ = table.AddBlock(turnB)

		// Touch turn-a to make it more recently accessed than turn-b
		if !table.TouchBlock("turn-a") {
			t.Fatalf("expected TouchBlock to return true for existing block")
		}

		// Enforce budget requiring 1 block to be discarded
		report := table.EnforceBudget(500)
		if report.PrunedNodesCount != 1 {
			t.Fatalf("expected 1 block pruned, got %d", report.PrunedNodesCount)
		}
		// turn-b should be pruned since turn-a was recently touched
		if report.RemovedNodes[0] != "turn-b" {
			t.Fatalf("expected older turn-b to be pruned first, got %s", report.RemovedNodes[0])
		}
		if !table.HasBlock("turn-a") {
			t.Fatalf("recently touched turn-a should remain resident")
		}
	})

	t.Run("manual pin and unpin behavior", func(t *testing.T) {
		table := NewPriorityKVTable()

		outBlock := NewOutputBlock("output-pinned", 800)
		_ = table.AddBlock(outBlock)

		// Explicitly pin the output block
		if !table.PinBlock("output-pinned") {
			t.Fatalf("expected PinBlock to return true")
		}

		// Enforce budget of 100 tokens: pinned output block must be immune
		report := table.EnforceBudget(100)
		if report.PrunedNodesCount != 0 {
			t.Fatalf("expected 0 blocks pruned for pinned block, got %d", report.PrunedNodesCount)
		}
		if !table.HasBlock("output-pinned") {
			t.Fatalf("pinned output block must remain resident")
		}

		// Unpin the block
		if !table.UnpinBlock("output-pinned") {
			t.Fatalf("expected UnpinBlock to return true")
		}

		// Enforce budget again: now it should be pruned
		report2 := table.EnforceBudget(100)
		if report2.PrunedNodesCount != 1 || report2.RemovedNodes[0] != "output-pinned" {
			t.Fatalf("expected unpinned output block to be pruned")
		}

		// Root blocks cannot be unpinned
		rootBlock := NewRootBlock("sys-root-guard", 500)
		_ = table.AddBlock(rootBlock)
		if table.UnpinBlock("sys-root-guard") {
			t.Fatalf("expected UnpinBlock on PriorityRoot to return false")
		}
		b, _ := table.GetBlock("sys-root-guard")
		if !b.Pinned {
			t.Fatalf("root instruction must remain pinned")
		}
	})

	t.Run("detailed KV report and helper methods", func(t *testing.T) {
		table := NewPriorityKVManager()

		_ = table.AddBlock(NewRootBlock("root-1", 500))
		_ = table.AddBlock(NewTurnBlock("turn-1", 300))
		_ = table.AddBlock(NewOutputBlock("out-1", 400))

		kvReport := table.EnforceBudgetDetailed(600)
		if kvReport.InitialTokens != 1200 {
			t.Fatalf("expected InitialTokens 1200, got %d", kvReport.InitialTokens)
		}
		if kvReport.PrunedTokens != 700 {
			t.Fatalf("expected PrunedTokens 700 (400 output + 300 turn), got %d", kvReport.PrunedTokens)
		}
		if kvReport.RemainingTokens != 500 {
			t.Fatalf("expected RemainingTokens 500, got %d", kvReport.RemainingTokens)
		}
		if len(kvReport.PrunedBlocks) != 2 {
			t.Fatalf("expected 2 pruned blocks in report, got %d", len(kvReport.PrunedBlocks))
		}

		pruned := table.PrunedBlocks()
		if len(pruned) != 2 {
			t.Fatalf("expected 2 pruned blocks from PrunedBlocks(), got %d", len(pruned))
		}
		prunedIDs := table.PrunedBlockIDs()
		if len(prunedIDs) != 2 || prunedIDs[0] != "out-1" || prunedIDs[1] != "turn-1" {
			t.Fatalf("expected [out-1, turn-1] pruned IDs, got %v", prunedIDs)
		}

		blocks := table.Blocks()
		if len(blocks) != 1 || blocks[0].BlockID != "root-1" {
			t.Fatalf("expected only root-1 block in resident blocks snapshot, got %v", blocks)
		}
	})

	t.Run("validation and edge cases", func(t *testing.T) {
		table := NewPriorityKVTable()

		// Empty block ID
		if err := table.AddBlock(KVBlock{BlockID: "", TokenCount: 100}); err == nil {
			t.Fatalf("expected error for empty block ID")
		}

		// Negative tokens
		if err := table.AddBlock(KVBlock{BlockID: "valid", TokenCount: -5}); err == nil {
			t.Fatalf("expected error for negative token count")
		}

		// Add block then update with new token count
		_ = table.AddBlock(KVBlock{BlockID: "update-me", TokenCount: 100, Priority: PriorityTurn})
		if table.TotalTokens() != 100 {
			t.Fatalf("expected 100 tokens, got %d", table.TotalTokens())
		}
		_ = table.AddBlock(KVBlock{BlockID: "update-me", TokenCount: 250, Priority: PriorityTurn})
		if table.TotalTokens() != 250 {
			t.Fatalf("expected 250 tokens after update, got %d", table.TotalTokens())
		}
		if table.BlockCount() != 1 {
			t.Fatalf("expected 1 block count after update, got %d", table.BlockCount())
		}

		// Budget greater than total tokens
		report := table.EnforceBudget(1000)
		if report.PrunedNodesCount != 0 {
			t.Fatalf("expected 0 pruned blocks when under budget, got %d", report.PrunedNodesCount)
		}

		// DiscardBlock
		if !table.DiscardBlock("update-me") {
			t.Fatalf("expected DiscardBlock to succeed for unpinned block")
		}
		if table.TotalTokens() != 0 || table.BlockCount() != 0 {
			t.Fatalf("expected table to be empty after discard")
		}

		// Discard non-existent block
		if table.DiscardBlock("non-existent") {
			t.Fatalf("expected DiscardBlock to return false for non-existent block")
		}
	})

	t.Run("concurrent access safety", func(t *testing.T) {
		table := NewPriorityKVTable()
		_ = table.AddBlock(NewRootBlock("sys-root", 1000))

		var wg sync.WaitGroup
		numWorkers := 8
		iterations := 100

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for i := 0; i < iterations; i++ {
					blockID := fmt.Sprintf("worker-%d-block-%d", workerID, i)
					_ = table.AddBlock(NewTurnBlock(blockID, 50))
					table.TouchBlock(blockID)
					_ = table.TotalTokens()
					_ = table.HasBlock(blockID)
					if i%10 == 0 {
						table.EnforceBudget(3000)
					}
				}
			}(w)
		}

		wg.Wait()

		// Final check: root must still be resident
		if !table.HasBlock("sys-root") {
			t.Fatalf("root block must remain resident after concurrent execution")
		}
	})
}
