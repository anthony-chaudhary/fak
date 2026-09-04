package ctxmmu_test

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// TestDemandPage50TurnTrajectory verifies that across a 50-turn interactive trajectory:
// 1. The prefix anchor bytes are 100% stable across all turn boundaries (prefix cache hit guarantee).
// 2. Zero loss of pinned user constraints occurs.
// 3. Zero loss of negative search findings occurs.
// 4. Older tool outputs are paged out to digest-bound cards and can be faulted back in on demand.
func TestDemandPage50TurnTrajectory(t *testing.T) {
	vctx := ctxmmu.NewVirtualContext()

	// Turn 0: Register immutable prefix anchors
	sysPrompt := []byte("System instructions: You are an autonomous Go agent operating inside the fak repository kernel.")
	vctx.AddSystemInstructions(sysPrompt)

	c1 := []byte("Constraint 1: Never edit files outside internal/ctxmmu and docs/specs.")
	c2 := []byte("Constraint 2: Maintain Go 1.26 compatibility and zero external dependencies.")
	c3 := []byte("Constraint 3: All tests must pass with -race enabled.")
	vctx.AddUserConstraint(c1, true)
	vctx.AddUserConstraint(c2, true)
	vctx.AddUserConstraint(c3, true)

	var firstPrefixBytes []byte
	negativeTurns := map[int]struct {
		query   string
		finding string
	}{
		3:  {query: "grep 'deprecated_symbol'", finding: "0 matches across 140 files"},
		7:  {query: "find -name 'legacy_driver.go'", finding: "File not found in repository"},
		12: {query: "git log -S 'old_token'", finding: "No commits matching string"},
		19: {query: "fak preflight --tool dangerous_exec", finding: "Refused: POLICY_BLOCK"},
		28: {query: "grep 'TODO(security)' internal/", finding: "0 matches found"},
		37: {query: "read /etc/passwd", finding: "Permission denied: SANDBOX_RESTRICTION"},
		45: {query: "grep 'flaky_test'", finding: "0 matches found"},
		49: {query: "fak arbitrate --lane busy_lane", finding: "Refused: LANE_LOCKED"},
	}

	expectedNegativeCount := 0

	// Execute 50 interactive turns
	for turn := 1; turn <= 50; turn++ {
		// User turn
		vctx.AddUserTurn(turn, []byte(fmt.Sprintf("Turn %d: Please proceed with task execution step %d.", turn, turn)))

		// Assistant reasoning
		vctx.AddAssistantTurn(turn, []byte(fmt.Sprintf("Turn %d: Analyzing current workspace state and executing necessary inspection tools.", turn)))

		// Tool outputs: generate small and large outputs
		if turn%2 == 0 {
			// Large verbose tool output (~3000 bytes)
			var largeOutput bytes.Buffer
			for line := 1; line <= 80; line++ {
				fmt.Fprintf(&largeOutput, "Line %d: checked package internal/ctxmmu/subpkg_%d OK\n", line, line)
			}
			vctx.AddToolOutput(turn, "bash", largeOutput.Bytes())
		} else {
			// Normal tool output
			vctx.AddToolOutput(turn, "read", []byte(fmt.Sprintf("File content for turn %d: status=ready, version=1.26", turn)))
		}

		// Inject negative search finding on designated turns
		if nf, ok := negativeTurns[turn]; ok {
			vctx.AddNegativeFinding(turn, nf.query, []byte(nf.finding))
			expectedNegativeCount++
		}

		// Project context view under a standard token budget (e.g. 6000 tokens)
		view := vctx.ProjectView(6000)

		// 1. Prefix Anchor Invariance Assertion
		if turn == 1 {
			firstPrefixBytes = make([]byte, len(view.PrefixBytes))
			copy(firstPrefixBytes, view.PrefixBytes)
			if len(firstPrefixBytes) == 0 {
				t.Fatalf("Turn 1: PrefixBytes is empty")
			}
		} else {
			if !bytes.Equal(view.PrefixBytes, firstPrefixBytes) {
				t.Fatalf("Turn %d: PrefixBytes diverged from Turn 1 prefix bytes (cache invalidation defect!)", turn)
			}
		}

		// Verify that RenderedBytes strictly starts with PrefixBytes
		if !view.HasPrefixAnchor(view.PrefixBytes) {
			t.Fatalf("Turn %d: RenderedBytes does not start with PrefixBytes", turn)
		}
		if !bytes.HasPrefix(view.RenderedBytes, firstPrefixBytes) {
			t.Fatalf("Turn %d: RenderedBytes does not start with firstPrefixBytes", turn)
		}

		// 2. Pinned Constraints Zero-Loss Invariant Assertion
		if view.PinnedConstraintCount() != 3 {
			t.Fatalf("Turn %d: Expected 3 pinned constraints, got %d", turn, view.PinnedConstraintCount())
		}
		renderedStr := view.String()
		if !strings.Contains(renderedStr, string(c1)) {
			t.Fatalf("Turn %d: Pinned constraint 1 missing from rendered context", turn)
		}
		if !strings.Contains(renderedStr, string(c2)) {
			t.Fatalf("Turn %d: Pinned constraint 2 missing from rendered context", turn)
		}
		if !strings.Contains(renderedStr, string(c3)) {
			t.Fatalf("Turn %d: Pinned constraint 3 missing from rendered context", turn)
		}

		// 3. Negative Findings Zero-Loss Invariant Assertion
		if view.NegativeFindingCount() != expectedNegativeCount {
			t.Fatalf("Turn %d: Expected %d negative findings, got %d", turn, expectedNegativeCount, view.NegativeFindingCount())
		}
		for tIdx, nf := range negativeTurns {
			if tIdx <= turn {
				if !strings.Contains(renderedStr, nf.query) {
					t.Fatalf("Turn %d: Negative finding query %q (from turn %d) lost from rendered context", turn, nf.query, tIdx)
				}
				if !strings.Contains(renderedStr, nf.finding) {
					t.Fatalf("Turn %d: Negative finding result %q (from turn %d) lost from rendered context", turn, nf.finding, tIdx)
				}
			}
		}
	}

	// 4. Assert Demand-Paging and Fault-In Behavior after 50 turns
	finalView := vctx.ProjectView(6000)
	if finalView.PagedOutCount == 0 {
		t.Fatalf("Expected older tool outputs to be paged out after 50 turns, got 0")
	}

	// Find a paged-out tool cell and fault it back in via PageIn
	var foundPagedCell *ctxmmu.ContextCell
	for i := range finalView.Cells {
		if finalView.Cells[i].PagedOut && finalView.Cells[i].Kind == ctxmmu.CellKindToolOutput {
			foundPagedCell = &finalView.Cells[i]
			break
		}
	}
	if foundPagedCell == nil {
		t.Fatalf("Could not locate any paged-out tool cell in final projection")
	}

	faultedBytes, err := vctx.PageIn(foundPagedCell.DigestHex)
	if err != nil {
		t.Fatalf("PageIn failed for digest %s: %v", foundPagedCell.DigestHex, err)
	}
	if len(faultedBytes) != foundPagedCell.OriginalBytes {
		t.Fatalf("Faulted bytes length mismatch: expected %d, got %d", foundPagedCell.OriginalBytes, len(faultedBytes))
	}
	if sha256.Sum256(faultedBytes) != foundPagedCell.Digest {
		t.Fatalf("Faulted bytes SHA256 checksum mismatch")
	}
}

// TestDemandPageTokenBudgets verifies context projection under various token constraints.
func TestDemandPageTokenBudgets(t *testing.T) {
	vctx := ctxmmu.NewVirtualContext()

	vctx.AddSystemInstructions([]byte("System: You are an agent."))
	vctx.AddUserConstraint([]byte("Constraint: Obey all tool limits."), true)

	// Add 10 turns with large tool outputs
	for turn := 1; turn <= 10; turn++ {
		vctx.AddUserTurn(turn, []byte(fmt.Sprintf("User query turn %d", turn)))
		vctx.AddAssistantTurn(turn, []byte(fmt.Sprintf("Assistant thought turn %d", turn)))

		// 4000-byte tool output (~1000 tokens)
		data := bytes.Repeat([]byte(fmt.Sprintf("payload-turn-%d-", turn)), 200)
		vctx.AddToolOutput(turn, "bash", data)
	}

	vctx.AddNegativeFinding(5, "find /opt/bin", []byte("No such file or directory"))

	// Test 1: Unconstrained / High Budget (e.g. 50,000 tokens)
	viewUnconstrained := vctx.ProjectView(50000)
	if viewUnconstrained.PagedOutCount > 0 {
		t.Errorf("Expected 0 paged out cells under 50k budget, got %d", viewUnconstrained.PagedOutCount)
	}

	// Test 2: Moderate Budget (e.g. 5,000 tokens)
	viewModerate := vctx.ProjectView(5000)
	if viewModerate.TotalTokens > 5000 {
		t.Errorf("Expected total tokens <= 5000, got %d", viewModerate.TotalTokens)
	}
	if viewModerate.PagedOutCount == 0 {
		t.Errorf("Expected some older tool outputs to be paged out under 5000 budget")
	}
	if !viewModerate.HasPrefixAnchor(viewModerate.PrefixBytes) {
		t.Errorf("viewModerate missing prefix anchor")
	}
	if viewModerate.NegativeFindingCount() != 1 {
		t.Errorf("Expected 1 negative finding preserved, got %d", viewModerate.NegativeFindingCount())
	}

	// Test 3: Tight Budget (e.g. 1,500 tokens)
	viewTight := vctx.ProjectView(1500)
	if viewTight.TotalTokens > 1500 {
		t.Errorf("Expected total tokens <= 1500, got %d", viewTight.TotalTokens)
	}
	if !viewTight.HasPrefixAnchor(viewTight.PrefixBytes) {
		t.Errorf("viewTight missing prefix anchor")
	}
	if viewTight.PinnedConstraintCount() != 1 {
		t.Errorf("Expected 1 pinned constraint preserved, got %d", viewTight.PinnedConstraintCount())
	}
	if viewTight.NegativeFindingCount() != 1 {
		t.Errorf("Expected 1 negative finding preserved, got %d", viewTight.NegativeFindingCount())
	}

	// Prefix bytes must be identical between all budget projections
	if !bytes.Equal(viewUnconstrained.PrefixBytes, viewModerate.PrefixBytes) {
		t.Errorf("PrefixBytes diverged between unconstrained and moderate budgets")
	}
	if !bytes.Equal(viewModerate.PrefixBytes, viewTight.PrefixBytes) {
		t.Errorf("PrefixBytes diverged between moderate and tight budgets")
	}
}

// TestDemandPageFaultIn verifies retrieval from the Content-Addressed Store and error handling.
func TestDemandPageFaultIn(t *testing.T) {
	vctx := ctxmmu.NewVirtualContext()

	payload := []byte("git diff --stat: 4 files changed, 25 insertions(+), 3 deletions(-)")
	id := vctx.AddToolOutput(1, "git", payload)
	if id == 0 {
		t.Fatalf("AddToolOutput returned id 0")
	}

	sum := sha256.Sum256(payload)
	hexDigest := fmt.Sprintf("%x", sum)

	// Fetch without sha256 prefix
	retrieved, err := vctx.PageIn(hexDigest)
	if err != nil {
		t.Fatalf("PageIn failed: %v", err)
	}
	if !bytes.Equal(retrieved, payload) {
		t.Fatalf("Retrieved payload does not match original")
	}

	// Fetch with sha256: prefix
	retrievedWithPrefix, err := vctx.PageIn("sha256:" + hexDigest)
	if err != nil {
		t.Fatalf("PageIn with prefix failed: %v", err)
	}
	if !bytes.Equal(retrievedWithPrefix, payload) {
		t.Fatalf("Retrieved payload with prefix does not match original")
	}

	// Fetch non-existent digest
	_, err = vctx.PageIn("0000000000000000000000000000000000000000000000000000000000000000")
	if err == nil {
		t.Fatalf("Expected error when paging in non-existent digest, got nil")
	}

	// Defensive copy verification: modifying retrieved slice does not corrupt store
	retrieved[0] = 'X'
	fresh, err := vctx.PageIn(hexDigest)
	if err != nil {
		t.Fatalf("PageIn failed: %v", err)
	}
	if fresh[0] == 'X' {
		t.Fatalf("CAS store was mutated; expected defensive copy")
	}
}

// TestDemandPageConcurrentAccess verifies thread-safety under concurrent access.
func TestDemandPageConcurrentAccess(t *testing.T) {
	vctx := ctxmmu.NewVirtualContext()
	vctx.AddSystemInstructions([]byte("Concurrent System Prompt"))
	vctx.AddUserConstraint([]byte("Concurrent Pinned Constraint"), true)

	var wg sync.WaitGroup
	workers := 10
	iterations := 25

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				turn := workerID*iterations + i + 1

				vctx.AddUserTurn(turn, []byte(fmt.Sprintf("User msg w=%d i=%d", workerID, i)))
				vctx.AddAssistantTurn(turn, []byte(fmt.Sprintf("Assistant msg w=%d i=%d", workerID, i)))
				vctx.AddToolOutput(turn, "bash", []byte(fmt.Sprintf("Output bytes for w=%d i=%d payload", workerID, i)))

				if i%5 == 0 {
					vctx.AddNegativeFinding(turn, fmt.Sprintf("query_%d_%d", workerID, i), []byte("not found"))
				}

				view := vctx.ProjectView(3000)
				if len(view.PrefixBytes) == 0 {
					t.Errorf("Empty prefix bytes in worker %d", workerID)
				}
				if !view.HasPrefixAnchor(view.PrefixBytes) {
					t.Errorf("Missing prefix anchor in worker %d", workerID)
				}
			}
		}(w)
	}

	wg.Wait()
}
