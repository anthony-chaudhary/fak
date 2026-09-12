package ctxmmu_test

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestCOWSubagentPrefixSharingWitness is the deterministic witness for #12313:
// forking 8 subagent session branches off one coordinator prefix must share the
// resident KV page blocks with zero duplicate physical allocations, and the
// zero-copy savings must be reported through the ctxmmu subagent telemetry.
//
// It pins three falsifiable properties of the physical sharing after the fork
// storm:
//  1. >80% of each subagent's page blocks are shared with the parent/peers,
//  2. zero duplicate physical page blocks were allocated by the forks,
//  3. the reported DeduplicatedBytes/DedupRatio/PrefixHitRate agree with the
//     observed block identity, so the telemetry cannot drift from the memory.
func TestCOWSubagentPrefixSharingWitness(t *testing.T) {
	table := ctxmmu.NewCOWPageTable()

	// 32k-token coordinator prefix = 512 blocks at the default 64-token capacity.
	const prefixTokens = 32768
	const numSubagents = 8

	tokens := make([]int, prefixTokens)
	for i := range tokens {
		tokens[i] = 1000 + (i % 5000)
	}

	if _, err := table.CreateSession("coordinator"); err != nil {
		t.Fatalf("CreateSession failed: %v", err)
	}
	if err := table.AppendTokens("coordinator", tokens); err != nil {
		t.Fatalf("AppendTokens failed: %v", err)
	}

	initialBlocks := table.PhysicalBlockCount()
	initialBytes := table.TotalAllocatedBytes()
	if initialBlocks != 512 {
		t.Fatalf("expected 512 resident prefix blocks, got %d", initialBlocks)
	}

	// Fork the subagents through the telemetry-bearing seam.
	var totalSharedPages int
	children := make([]*ctxmmu.SessionBranch, 0, numSubagents)
	for i := 0; i < numSubagents; i++ {
		childID := fmt.Sprintf("subagent-%d", i)
		child, telem, err := table.ForkSubagent("coordinator", childID)
		if err != nil {
			t.Fatalf("ForkSubagent %s failed: %v", childID, err)
		}
		children = append(children, child)

		if telem.ParentID != "coordinator" {
			t.Fatalf("subagent %s telemetry parent = %q, want coordinator", childID, telem.ParentID)
		}
		if telem.DuplicatePages != 0 {
			t.Fatalf("subagent %s allocated %d duplicate pages, want 0", childID, telem.DuplicatePages)
		}
		if telem.SharedPages != initialBlocks {
			t.Fatalf("subagent %s shared %d pages, want %d", childID, telem.SharedPages, initialBlocks)
		}
		if telem.PrefixHitRate < 0.8 {
			t.Fatalf("subagent %s prefix hit rate %f, want >= 0.8", childID, telem.PrefixHitRate)
		}
		totalSharedPages += telem.SharedPages
	}

	// Property 1: >80% shared pages per subagent over the physical block count.
	for i, child := range children {
		total := child.PageCount()
		if total == 0 {
			t.Fatalf("subagent %d has no page blocks", i)
		}
		sharedRatio := float64(child.SharedPages()) / float64(total)
		if sharedRatio <= 0.80 {
			t.Fatalf("subagent %d shared page ratio %f, want > 0.80", i, sharedRatio)
		}
	}

	// Property 2: zero duplicate allocations across the whole fork storm.
	if got := table.PhysicalBlockCount(); got != initialBlocks {
		t.Fatalf("fork storm allocated %d duplicate physical blocks (was %d)", got, initialBlocks)
	}
	if got := table.TotalAllocatedBytes(); got != initialBytes {
		t.Fatalf("fork storm allocated %d bytes (was %d)", got, initialBytes)
	}
	if got := table.DuplicatePhysicalPagesAllocated(); got != 0 {
		t.Fatalf("DuplicatePhysicalPagesAllocated = %d, want 0", got)
	}

	// Property 3: aggregate telemetry agrees with observed sharing.
	report := table.SubagentSharingReport()
	if report.ForkedSubagents != numSubagents {
		t.Fatalf("report.ForkedSubagents = %d, want %d", report.ForkedSubagents, numSubagents)
	}
	if report.ActiveBranches != numSubagents+1 {
		t.Fatalf("report.ActiveBranches = %d, want %d", report.ActiveBranches, numSubagents+1)
	}
	if report.TotalSharedPages != initialBlocks {
		t.Fatalf("report.TotalSharedPages = %d, want %d", report.TotalSharedPages, initialBlocks)
	}
	if report.TotalPhysicalBlocks != initialBlocks {
		t.Fatalf("report.TotalPhysicalBlocks = %d, want %d (no duplication)", report.TotalPhysicalBlocks, initialBlocks)
	}
	if report.DuplicatePagesAlloc != 0 {
		t.Fatalf("report.DuplicatePagesAlloc = %d, want 0", report.DuplicatePagesAlloc)
	}
	if report.DedupRatio <= 0.80 {
		t.Fatalf("report.DedupRatio = %f, want > 0.80", report.DedupRatio)
	}
	if report.DeduplicatedBytes <= 0 {
		t.Fatalf("report.DeduplicatedBytes = %d, want > 0", report.DeduplicatedBytes)
	}
	if report.PrefixHitRate < 0.8 {
		t.Fatalf("report.PrefixHitRate = %f, want >= 0.8", report.PrefixHitRate)
	}
	if totalSharedPages != numSubagents*initialBlocks {
		t.Fatalf("total shared pages across subagents = %d, want %d", totalSharedPages, numSubagents*initialBlocks)
	}
}
