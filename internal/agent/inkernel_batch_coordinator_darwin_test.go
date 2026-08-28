//go:build darwin && arm64 && cgo

package agent

import (
	"context"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInKernelPlannerCoalescesRealQwenMetalTurns is the planner-owned execution
// witness for #9075. It deliberately installs no shared-receipt probe: nonzero
// receipt values can only come back from BatchSession's real Qwen Metal path.
func TestInKernelPlannerCoalescesRealQwenMetalTurns(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := model.NewQwen35HybridQ4KMetalFixture(t)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Fatalf("close model weights: %v", err)
		}
	})

	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "on")
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-qwen35-metal-cohort", true, nil, true)
	p.maxNew = 2
	p.batchDecode = true

	const cohortSize = 4 // the real hybrid Q4_K path admits B=4..8
	ready := make(chan struct{})
	p.coalesceReadyHook = func() { <-ready }
	var batchMu sync.Mutex
	var batches []int
	p.coalesceBatchHook = func(n int) {
		batchMu.Lock()
		batches = append(batches, n)
		batchMu.Unlock()
	}
	messages := make([][]Message, cohortSize)
	for i := range messages {
		messages[i] = []Message{{Role: RoleUser, Content: string(rune('a' + i))}}
	}
	type answer struct {
		completion *Completion
		err        error
	}
	answers := make([]answer, cohortSize)
	var wg sync.WaitGroup
	for i := range answers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			answers[i].completion, answers[i].err = p.Complete(context.Background(), messages[i], nil)
		}(i)
	}
	deadline := time.After(10 * time.Second)
	for {
		p.coalesceMu.Lock()
		n := len(p.coalesceReady)
		p.coalesceMu.Unlock()
		if n == cohortSize {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("planner queued %d/%d requests", n, cohortSize)
		default:
			runtime.Gosched()
		}
	}
	close(ready)
	wg.Wait()

	var cohortID uint64
	for i, answer := range answers {
		if answer.err != nil {
			t.Fatalf("coalesced[%d]: %v", i, answer.err)
		}
		if answer.completion == nil || answer.completion.InKernelBatch == nil {
			t.Fatalf("coalesced[%d] missing receipt", i)
		}
		receipt := answer.completion.InKernelBatch
		if receipt.CohortSize != cohortSize || receipt.SharedPanels == 0 || receipt.SharedMACs == 0 || receipt.SessionCloses != cohortSize {
			t.Fatalf("coalesced[%d] real Metal receipt=%+v", i, receipt)
		}
		if cohortID == 0 {
			cohortID = receipt.CohortID
		} else if receipt.CohortID != cohortID {
			t.Fatalf("coalesced[%d] cohort=%d, want %d", i, receipt.CohortID, cohortID)
		}
	}
	batchMu.Lock()
	defer batchMu.Unlock()
	if len(batches) == 0 || batches[0] != cohortSize {
		t.Fatalf("planner batches=%v, want first batch B=%d", batches, cohortSize)
	}
}
