package modelengine

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestSubagentDynamicAdmissionAndCompletion verifies dynamic admission and iteration-level
// continuous batching retirement for B=1..8 subagents with irregular turn lengths (20..300 tokens).
// When a subagent completes its turn, it is retired immediately without stalling active generation.
func TestSubagentDynamicAdmissionAndCompletion(t *testing.T) {
	cfg := DefaultSubagentSchedulerConfig()
	cfg.MaxConcurrency = 4 // Concurrency pool of 4 slots

	sched, err := NewSubagentScheduler(cfg)
	if err != nil {
		t.Fatalf("NewSubagentScheduler failed: %v", err)
	}
	defer sched.Close()

	ctx := context.Background()

	// 8 subagents with irregular sequence lengths
	turnLengths := []int{10, 25, 5, 30, 8, 15, 20, 12}
	sessions := make([]*SubagentSession, len(turnLengths))

	for i, length := range turnLengths {
		id := fmt.Sprintf("subagent-%d", i)
		prompt := []int{100 + i, 200 + i}
		sub, err := sched.Admit(ctx, id, prompt, length)
		if err != nil {
			t.Fatalf("Admit subagent %d failed: %v", i, err)
		}
		sessions[i] = sub
	}

	// Initial check: exactly 4 active, 4 waiting
	if got := sched.ActiveCount(); got != 4 {
		t.Fatalf("ActiveCount initial = %d, want 4", got)
	}
	if got := sched.WaitingCount(); got != 4 {
		t.Fatalf("WaitingCount initial = %d, want 4", got)
	}

	// Step iterations until all subagents complete
	maxSteps := 100
	stepsRun := 0
	for stepsRun < maxSteps {
		allDone := true
		for _, s := range sessions {
			if s.TokensGenerated() < s.TargetTokens {
				allDone = false
				break
			}
		}
		if allDone {
			break
		}

		batch, err := sched.StepIteration()
		if err != nil {
			t.Fatalf("StepIteration failed at step %d: %v", stepsRun, err)
		}

		// At no point should active count exceed MaxConcurrency
		if batch.ActiveCount > cfg.MaxConcurrency {
			t.Fatalf("Step %d: ActiveCount %d exceeded MaxConcurrency %d", stepsRun, batch.ActiveCount, cfg.MaxConcurrency)
		}
		stepsRun++
	}

	// Verify all 8 subagents completed with exact target token counts
	for i, sub := range sessions {
		select {
		case <-sub.Done():
		default:
			t.Fatalf("subagent %d (%s) did not signal Done", i, sub.ID)
		}

		if got, want := sub.TokensGenerated(), turnLengths[i]; got != want {
			t.Fatalf("subagent %d tokens = %d, want %d", i, got, want)
		}
		if sub.State != SubagentStateCompleted {
			t.Fatalf("subagent %d state = %v, want %v", i, sub.State, SubagentStateCompleted)
		}
	}

	receipt := sched.Receipt()
	if receipt.TotalCompleted != len(turnLengths) {
		t.Fatalf("Receipt.TotalCompleted = %d, want %d", receipt.TotalCompleted, len(turnLengths))
	}
	if receipt.PeakConcurrency > cfg.MaxConcurrency {
		t.Fatalf("Receipt.PeakConcurrency = %d, exceeded cap %d", receipt.PeakConcurrency, cfg.MaxConcurrency)
	}
	if receipt.CompactedBatchesRun == 0 {
		t.Fatalf("Receipt.CompactedBatchesRun is 0")
	}
}

// TestSubagentYieldIOAndResumeZeroEviction verifies subagent tool-wait gap exploitation:
// 1. YieldIO vacates active decode slot while keeping KV cache resident in UMA (0 ms eviction, 0 bytes swapped).
// 2. Resume re-inserts the session into active decode pool with 0 re-prefill tokens.
func TestSubagentYieldIOAndResumeZeroEviction(t *testing.T) {
	cfg := DefaultSubagentSchedulerConfig()
	cfg.MaxConcurrency = 4

	sched, err := NewSubagentScheduler(cfg)
	if err != nil {
		t.Fatalf("NewSubagentScheduler: %v", err)
	}
	defer sched.Close()

	ctx := context.Background()
	subA, err := sched.Admit(ctx, "agent-A", []int{1, 2}, 30)
	if err != nil {
		t.Fatalf("Admit agent-A: %v", err)
	}
	subB, err := sched.Admit(ctx, "agent-B", []int{3, 4}, 30)
	if err != nil {
		t.Fatalf("Admit agent-B: %v", err)
	}

	// Generate 5 tokens concurrently
	for i := 0; i < 5; i++ {
		_, err := sched.StepIteration()
		if err != nil {
			t.Fatalf("StepIteration %d: %v", i, err)
		}
	}

	if got := subA.TokensGenerated(); got != 5 {
		t.Fatalf("agent-A tokens before yield = %d, want 5", got)
	}
	if got := subB.TokensGenerated(); got != 5 {
		t.Fatalf("agent-B tokens before yield = %d, want 5", got)
	}

	// Agent A initiates external tool call (e.g. bash or web fetch) -> YieldIO
	err = sched.YieldIO("agent-A", "web_fetch")
	if err != nil {
		t.Fatalf("YieldIO failed: %v", err)
	}

	// Verify YieldIO invariants:
	// 1. Active decode slot vacated
	if got := sched.ActiveCount(); got != 1 {
		t.Fatalf("ActiveCount after YieldIO = %d, want 1", got)
	}
	if got := sched.YieldedCount(); got != 1 {
		t.Fatalf("YieldedCount after YieldIO = %d, want 1", got)
	}
	// 2. Stationarity in UMA with 0 ms eviction and 0 bytes swapped
	if subA.EvictionDuration != 0 {
		t.Fatalf("agent-A EvictionDuration = %v, want 0 ms", subA.EvictionDuration)
	}
	if subA.BytesSwapped != 0 {
		t.Fatalf("agent-A BytesSwapped = %d, want 0", subA.BytesSwapped)
	}
	if !subA.KVCacheStationary {
		t.Fatalf("agent-A KVCacheStationary = false, want true")
	}
	if subA.State != SubagentStateYielded {
		t.Fatalf("agent-A State = %v, want %v", subA.State, SubagentStateYielded)
	}
	if subA.CurrentTool != "web_fetch" {
		t.Fatalf("agent-A CurrentTool = %q, want 'web_fetch'", subA.CurrentTool)
	}

	// Run 3 steps: Agent B decodes, while Agent A stays stationary without advancing
	for i := 0; i < 3; i++ {
		batch, err := sched.StepIteration()
		if err != nil {
			t.Fatalf("StepIteration during tool wait: %v", err)
		}
		if batch.ActiveCount != 1 {
			t.Fatalf("ActiveCount during tool wait = %d, want 1", batch.ActiveCount)
		}
	}

	if got := subA.TokensGenerated(); got != 5 {
		t.Fatalf("agent-A tokens during yield = %d, want 5 (must not advance)", got)
	}
	if got := subB.TokensGenerated(); got != 8 {
		t.Fatalf("agent-B tokens during yield = %d, want 8", got)
	}

	// Tool call finishes -> Resume
	err = sched.Resume("agent-A")
	if err != nil {
		t.Fatalf("Resume failed: %v", err)
	}

	// Verify Resume invariants:
	// 1. Zero re-prefill tokens needed because KV cache stayed in UMA
	if subA.ReprefillTokens != 0 {
		t.Fatalf("agent-A ReprefillTokens = %d, want 0", subA.ReprefillTokens)
	}
	if subA.State != SubagentStateActive {
		t.Fatalf("agent-A State after resume = %v, want %v", subA.State, SubagentStateActive)
	}
	if sched.ActiveCount() != 2 {
		t.Fatalf("ActiveCount after resume = %d, want 2", sched.ActiveCount())
	}
	if sched.YieldedCount() != 0 {
		t.Fatalf("YieldedCount after resume = %d, want 0", sched.YieldedCount())
	}

	// Continue stepping: both agents now decode in continuous batch
	for i := 0; i < 5; i++ {
		_, err := sched.StepIteration()
		if err != nil {
			t.Fatalf("StepIteration post resume %d: %v", i, err)
		}
	}

	if got := subA.TokensGenerated(); got != 10 {
		t.Fatalf("agent-A tokens post resume = %d, want 10", got)
	}
	if got := subB.TokensGenerated(); got != 13 {
		t.Fatalf("agent-B tokens post resume = %d, want 13", got)
	}

	// Audit receipts
	receipt := sched.Receipt()
	if receipt.ZeroEvictionProofCount < 1 {
		t.Fatalf("ZeroEvictionProofCount = %d, want >= 1", receipt.ZeroEvictionProofCount)
	}
	if receipt.ZeroReprefillProofCount < 1 {
		t.Fatalf("ZeroReprefillProofCount = %d, want >= 1", receipt.ZeroReprefillProofCount)
	}
}

// TestSubagentToolWaitSlotHandoff verifies that when an active subagent yields for I/O,
// its active decode slot is immediately handed off to a queued waiting subagent.
func TestSubagentToolWaitSlotHandoff(t *testing.T) {
	cfg := DefaultSubagentSchedulerConfig()
	cfg.MaxConcurrency = 2 // Exactly 2 decode slots

	sched, err := NewSubagentScheduler(cfg)
	if err != nil {
		t.Fatalf("NewSubagentScheduler: %v", err)
	}
	defer sched.Close()

	ctx := context.Background()

	// Fill slots with agent-1 and agent-2
	sub1, err := sched.Admit(ctx, "agent-1", []int{10}, 20)
	if err != nil {
		t.Fatalf("Admit agent-1: %v", err)
	}
	_, err = sched.Admit(ctx, "agent-2", []int{20}, 20)
	if err != nil {
		t.Fatalf("Admit agent-2: %v", err)
	}

	// Queue agent-3 into waiting
	sub3, err := sched.Admit(ctx, "agent-3", []int{30}, 20)
	if err != nil {
		t.Fatalf("Admit agent-3: %v", err)
	}

	if sched.ActiveCount() != 2 {
		t.Fatalf("ActiveCount = %d, want 2", sched.ActiveCount())
	}
	if sched.WaitingCount() != 1 {
		t.Fatalf("WaitingCount = %d, want 1", sched.WaitingCount())
	}
	if sub3.State != SubagentStateWaiting {
		t.Fatalf("sub3 state = %v, want %v", sub3.State, SubagentStateWaiting)
	}

	// agent-1 yields execution for tool call
	err = sched.YieldIO("agent-1", "bash")
	if err != nil {
		t.Fatalf("YieldIO: %v", err)
	}

	// Active slot handoff: agent-3 must be immediately promoted to active!
	if sub1.State != SubagentStateYielded {
		t.Fatalf("sub1 state = %v, want %v", sub1.State, SubagentStateYielded)
	}
	if sub3.State != SubagentStateActive {
		t.Fatalf("sub3 state after handoff = %v, want %v", sub3.State, SubagentStateActive)
	}
	if sched.ActiveCount() != 2 {
		t.Fatalf("ActiveCount after handoff = %d, want 2", sched.ActiveCount())
	}
	if sched.WaitingCount() != 0 {
		t.Fatalf("WaitingCount after handoff = %d, want 0", sched.WaitingCount())
	}

	// Step a turn: agent-2 and agent-3 advance
	batch, err := sched.StepIteration()
	if err != nil {
		t.Fatalf("StepIteration: %v", err)
	}
	if batch.ActiveCount != 2 {
		t.Fatalf("batch.ActiveCount = %d, want 2", batch.ActiveCount)
	}
	if sub3.TokensGenerated() != 1 {
		t.Fatalf("sub3 tokens = %d, want 1", sub3.TokensGenerated())
	}
}

// TestSubagentRaggedBatchCompaction verifies that ragged batch tensor compaction gathers
// strictly active lanes into the batch forward pass, skipping paused/finished lanes
// to avoid wasted FLOPs.
func TestSubagentRaggedBatchCompaction(t *testing.T) {
	cfg := DefaultSubagentSchedulerConfig()
	cfg.MaxConcurrency = 8 // Concurrency B=8

	sched, err := NewSubagentScheduler(cfg)
	if err != nil {
		t.Fatalf("NewSubagentScheduler: %v", err)
	}
	defer sched.Close()

	ctx := context.Background()

	// Admit 6 subagents
	for i := 0; i < 6; i++ {
		_, err := sched.Admit(ctx, fmt.Sprintf("lane-%d", i), []int{i + 1}, 50)
		if err != nil {
			t.Fatalf("Admit lane %d: %v", i, err)
		}
	}

	// Yield 2 lanes into tool-wait
	if err := sched.YieldIO("lane-1", "tool-grep"); err != nil {
		t.Fatalf("YieldIO lane-1: %v", err)
	}
	if err := sched.YieldIO("lane-3", "tool-curl"); err != nil {
		t.Fatalf("YieldIO lane-3: %v", err)
	}

	// Compact ragged batch
	batch := sched.CompactRaggedBatch()

	// Verify compaction metrics:
	// TotalSlots = 8, Active = 4, Inactive = 4
	if batch.TotalSlots != 8 {
		t.Fatalf("batch.TotalSlots = %d, want 8", batch.TotalSlots)
	}
	if batch.ActiveCount != 4 {
		t.Fatalf("batch.ActiveCount = %d, want 4", batch.ActiveCount)
	}
	if batch.InactiveCount != 4 {
		t.Fatalf("batch.InactiveCount = %d, want 4", batch.InactiveCount)
	}
	if len(batch.CompactedIDs) != 4 {
		t.Fatalf("len(CompactedIDs) = %d, want 4", len(batch.CompactedIDs))
	}
	if batch.CompactionRatio != 0.5 {
		t.Fatalf("batch.CompactionRatio = %f, want 0.5", batch.CompactionRatio)
	}

	// Verify FLOPs saved by not executing inactive lanes
	expectedSavedFLOPs := 4.0 * float64(cfg.ModelParams*2)
	if batch.SavedFLOPs != expectedSavedFLOPs {
		t.Fatalf("batch.SavedFLOPs = %f, want %f", batch.SavedFLOPs, expectedSavedFLOPs)
	}
	if batch.SavedFLOPs <= 0 {
		t.Fatalf("batch.SavedFLOPs must be positive")
	}

	// Verify only the 4 active sessions are gathered
	for _, s := range batch.ActiveSessions {
		if s.ID == "lane-1" || s.ID == "lane-3" {
			t.Fatalf("Inactive lane %s found in compacted active sessions", s.ID)
		}
		if s.State != SubagentStateActive {
			t.Fatalf("Session %s in compacted batch is not active", s.ID)
		}
	}
}

// TestSubagentAggregateThroughputAndIntensity verifies arithmetic intensity and aggregate throughput
// scaling on AMD Strix Halo APU for Qwen3.8-14B Q4_K_M:
// 1. Single agent: ~19 tok/s (memory-bandwidth bound at ~3.3 FLOPs/byte).
// 2. 8 subagents: aggregate throughput >80 tok/s.
// 3. Operational intensity pushed into 50-150 FLOPs/byte range.
func TestSubagentAggregateThroughputAndIntensity(t *testing.T) {
	cfg := DefaultSubagentSchedulerConfig()
	sched, err := NewSubagentScheduler(cfg)
	if err != nil {
		t.Fatalf("NewSubagentScheduler: %v", err)
	}
	defer sched.Close()

	// 1. Single-agent baseline
	singleTokSec := sched.AggregateThroughput(1)
	if singleTokSec < 18.0 || singleTokSec > 21.0 {
		t.Fatalf("Single-agent throughput = %f tok/s, want ~19.0 tok/s", singleTokSec)
	}

	singleIntensity := sched.OperationalIntensity(1)
	if singleIntensity < 3.0 || singleIntensity > 4.0 {
		t.Fatalf("Single-agent operational intensity = %f FLOPs/byte, want ~3.3 FLOPs/byte", singleIntensity)
	}

	// 2. 8 subagents aggregate throughput: must exceed >80 tok/s
	tps8 := sched.AggregateThroughput(8)
	if tps8 <= 80.0 {
		t.Fatalf("Aggregate throughput across 8 subagents = %f tok/s, want >80.0 tok/s", tps8)
	}

	// 3. Operational intensity pushed into 50-150 FLOPs/byte range for B=8..32
	batchSizes := []int{8, 12, 16, 24, 32}
	for _, b := range batchSizes {
		intensity := sched.ArithmeticIntensity(b)
		if intensity < 50.0 || intensity > 150.0 {
			t.Fatalf("Arithmetic intensity at B=%d is %f FLOPs/byte, want in [50.0, 150.0] range", b, intensity)
		}

		tps := sched.AggregateThroughput(b)
		if tps <= 80.0 {
			t.Fatalf("Throughput at B=%d is %f tok/s, want >80.0 tok/s", b, tps)
		}
	}

	// 4. Scaling monotonicity check: B=1 < B=2 < B=4 < B=8 < B=16 < B=32
	testScales := []int{1, 2, 4, 8, 16, 32}
	var prevTPS float64
	for _, b := range testScales {
		curTPS := sched.AggregateThroughput(b)
		if curTPS <= prevTPS {
			t.Fatalf("Throughput failed monotonicity at B=%d: cur %f <= prev %f", b, curTPS, prevTPS)
		}
		prevTPS = curTPS
	}
}

// TestSubagentDynamicConcurrency1To32 verifies that the scheduler supports dynamic concurrency
// from B=1 up to B=32 concurrent subagent streams.
func TestSubagentDynamicConcurrency1To32(t *testing.T) {
	concurrencies := []int{1, 2, 4, 8, 16, 32}

	for _, b := range concurrencies {
		t.Run(fmt.Sprintf("Concurrency-%d", b), func(t *testing.T) {
			cfg := DefaultSubagentSchedulerConfig()
			cfg.MaxConcurrency = b

			sched, err := NewSubagentScheduler(cfg)
			if err != nil {
				t.Fatalf("NewSubagentScheduler B=%d failed: %v", b, err)
			}
			defer sched.Close()

			ctx := context.Background()

			// Admit b subagents
			for i := 0; i < b; i++ {
				id := fmt.Sprintf("c%d-agent-%d", b, i)
				_, err := sched.Admit(ctx, id, []int{i}, 5)
				if err != nil {
					t.Fatalf("Admit failed for B=%d, agent %d: %v", b, i, err)
				}
			}

			if sched.ActiveCount() != b {
				t.Fatalf("ActiveCount = %d, want %d", sched.ActiveCount(), b)
			}

			// Step 5 iterations to complete all
			for i := 0; i < 5; i++ {
				batch, err := sched.StepIteration()
				if err != nil {
					t.Fatalf("StepIteration %d failed: %v", i, err)
				}
				if batch.ActiveCount > b {
					t.Fatalf("ActiveCount %d exceeded B=%d", batch.ActiveCount, b)
				}
			}

			receipt := sched.Receipt()
			if receipt.TotalCompleted != b {
				t.Fatalf("TotalCompleted = %d, want %d", receipt.TotalCompleted, b)
			}
			if receipt.PeakConcurrency != b {
				t.Fatalf("PeakConcurrency = %d, want %d", receipt.PeakConcurrency, b)
			}
		})
	}
}

// TestSubagentModelExecutionSynthetic verifies end-to-end integration with a real model.Model instance.
func TestSubagentModelExecutionSynthetic(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	cfg := DefaultSubagentSchedulerConfig()
	cfg.MaxConcurrency = 4
	cfg.Model = m

	sched, err := NewSubagentScheduler(cfg)
	if err != nil {
		t.Fatalf("NewSubagentScheduler: %v", err)
	}
	defer sched.Close()

	ctx := context.Background()
	lengths := []int{4, 6, 8, 10}
	sessions := make([]*SubagentSession, len(lengths))

	for i, length := range lengths {
		id := fmt.Sprintf("model-agent-%d", i)
		prompt := []int{10 + i, 20 + i}
		sub, err := sched.Admit(ctx, id, prompt, length)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		sessions[i] = sub
	}

	// Step until all complete
	for step := 0; step < 20; step++ {
		allDone := true
		for _, s := range sessions {
			if s.TokensGenerated() < s.TargetTokens {
				allDone = false
				break
			}
		}
		if allDone {
			break
		}

		_, err := sched.StepIteration()
		if err != nil {
			t.Fatalf("StepIteration step %d: %v", step, err)
		}
	}

	// Verify all subagents completed with real model tokens
	for i, sub := range sessions {
		if sub.TokensGenerated() != lengths[i] {
			t.Fatalf("subagent %d tokens = %d, want %d", i, sub.TokensGenerated(), lengths[i])
		}
		if sub.State != SubagentStateCompleted {
			t.Fatalf("subagent %d state = %v, want %v", i, sub.State, SubagentStateCompleted)
		}
	}
}

// TestSubagentCancellation verifies cancellation of an active or queued subagent cleanly frees slots.
func TestSubagentCancellation(t *testing.T) {
	cfg := DefaultSubagentSchedulerConfig()
	cfg.MaxConcurrency = 2

	sched, err := NewSubagentScheduler(cfg)
	if err != nil {
		t.Fatalf("NewSubagentScheduler: %v", err)
	}
	defer sched.Close()

	ctx := context.Background()
	sub1, _ := sched.Admit(ctx, "sub-1", []int{1}, 20)
	sub2, _ := sched.Admit(ctx, "sub-2", []int{2}, 20)
	sub3, _ := sched.Admit(ctx, "sub-3", []int{3}, 20)

	// sub3 should be in waiting
	if sub3.State != SubagentStateWaiting {
		t.Fatalf("sub3 state = %v, want waiting", sub3.State)
	}

	// Cancel sub1 mid-generation
	err = sched.Cancel("sub-1")
	if err != nil {
		t.Fatalf("Cancel sub-1: %v", err)
	}

	if sub1.State != SubagentStateCancelled {
		t.Fatalf("sub1 state = %v, want cancelled", sub1.State)
	}

	// sub3 should have been promoted into sub1's vacated slot
	if sub3.State != SubagentStateActive {
		t.Fatalf("sub3 state after cancellation handoff = %v, want active", sub3.State)
	}
	if sched.ActiveCount() != 2 {
		t.Fatalf("ActiveCount = %d, want 2", sched.ActiveCount())
	}
	if sched.WaitingCount() != 0 {
		t.Fatalf("WaitingCount = %d, want 0", sched.WaitingCount())
	}

	_ = sub2
}

// TestSubagentConcurrentStartStop verifies autonomous background loop operations.
func TestSubagentConcurrentStartStop(t *testing.T) {
	cfg := DefaultSubagentSchedulerConfig()
	cfg.MaxConcurrency = 4

	sched, err := NewSubagentScheduler(cfg)
	if err != nil {
		t.Fatalf("NewSubagentScheduler: %v", err)
	}

	sched.Start(5 * time.Millisecond)

	ctx := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			sub, err := sched.Admit(ctx, fmt.Sprintf("bg-agent-%d", id), []int{id}, 10)
			if err != nil {
				t.Errorf("Admit bg-agent-%d: %v", id, err)
				return
			}
			select {
			case <-sub.Done():
			case <-time.After(2 * time.Second):
				t.Errorf("bg-agent-%d timed out", id)
			}
		}(i)
	}

	wg.Wait()
	sched.Close()
}
