package workerworktree

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestDebouncedSweepGate_ConcurrentSubagentFanout(t *testing.T) {
	ResetSweepGateForTest()
	defer ResetSweepGateForTest()

	root := t.TempDir()
	wtRoot := filepath.Join(root, "_scratch")
	if err := os.MkdirAll(wtRoot, 0o755); err != nil {
		t.Fatal(err)
	}

	const concurrency = 16
	var wg sync.WaitGroup
	startBarrier := make(chan struct{})
	receipts := make([]TriggerReceipt, concurrency)
	callDurations := make([]time.Duration, concurrency)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			<-startBarrier
			t0 := time.Now()
			receipts[idx] = DebouncedSweepDeadWorktrees(root, wtRoot, nil)
			callDurations[idx] = time.Since(t0)
		}(i)
	}

	// Release all 16 workers simultaneously.
	close(startBarrier)
	wg.Wait()

	var executedCount, skippedCount int
	for i, r := range receipts {
		switch r.Decision {
		case DecisionExecuted:
			executedCount++
			if r.Schema != TriggerReceiptSchema {
				t.Errorf("worker %d: expected schema %q, got %q", i, TriggerReceiptSchema, r.Schema)
			}
			if r.Trigger != TriggerStaleSweep {
				t.Errorf("worker %d: expected trigger %q, got %q", i, TriggerStaleSweep, r.Trigger)
			}
		case DecisionSkippedCooldown, DecisionSkippedLockHeld:
			skippedCount++
			// Verify that skipped calls returned immediately (< 1ms execution duration)
			if r.DurationNS > int64(time.Millisecond) {
				t.Errorf("worker %d: expected receipt DurationNS < 1ms, got %v", i, time.Duration(r.DurationNS))
			}
			if r.RefusalCode != RefusalCooldownActive && r.RefusalCode != RefusalLockContention {
				t.Errorf("worker %d: unexpected refusal code %q", i, r.RefusalCode)
			}
		default:
			t.Errorf("worker %d: unexpected decision %q", i, r.Decision)
		}
	}

	if executedCount != 1 {
		t.Fatalf("expected exactly 1 worker to execute sweep, got %d", executedCount)
	}
	if skippedCount != concurrency-1 {
		t.Fatalf("expected exactly %d workers to be debounced/skipped, got %d", concurrency-1, skippedCount)
	}
}

func TestDebouncedSweepGate_CooldownExpiration(t *testing.T) {
	ResetSweepGateForTest()
	defer ResetSweepGateForTest()

	root := t.TempDir()
	wtRoot := filepath.Join(root, "_scratch")
	if err := os.MkdirAll(wtRoot, 0o755); err != nil {
		t.Fatal(err)
	}

	gate := NewDebouncedSweepGate()
	cooldown := 30 * time.Millisecond

	// 1. First sweep must execute
	r1, ok1 := gate.TrySweep(root, wtRoot, nil, cooldown, false)
	if !ok1 || r1.Decision != DecisionExecuted {
		t.Fatalf("expected initial sweep to execute, got ok=%v decision=%s", ok1, r1.Decision)
	}

	// 2. Second sweep within cooldown must be debounced
	r2, ok2 := gate.TrySweep(root, wtRoot, nil, cooldown, false)
	if ok2 || r2.Decision != DecisionSkippedCooldown {
		t.Fatalf("expected immediate sweep to skip on cooldown, got ok=%v decision=%s", ok2, r2.Decision)
	}
	if r2.RefusalCode != RefusalCooldownActive {
		t.Fatalf("expected refusal code %s, got %s", RefusalCooldownActive, r2.RefusalCode)
	}

	// 3. Forced sweep must bypass cooldown and execute
	r3, ok3 := gate.TrySweep(root, wtRoot, nil, cooldown, true)
	if !ok3 || r3.Decision != DecisionExecuted {
		t.Fatalf("expected forced sweep to execute, got ok=%v decision=%s", ok3, r3.Decision)
	}

	// 4. Wait for cooldown to expire
	time.Sleep(35 * time.Millisecond)

	// 5. Sweep after cooldown expiration must execute
	r4, ok4 := gate.TrySweep(root, wtRoot, nil, cooldown, false)
	if !ok4 || r4.Decision != DecisionExecuted {
		t.Fatalf("expected sweep after cooldown to execute, got ok=%v decision=%s", ok4, r4.Decision)
	}
}

func TestTriggerReceiptSchema(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Millisecond)
	receipt := TriggerReceipt{
		Schema:         TriggerReceiptSchema,
		Trigger:        TriggerStaleSweep,
		Decision:       DecisionExecuted,
		Reason:         "",
		RefusalCode:    "",
		InspectedCount: 12,
		ReapedCount:    3,
		RetainedCount:  9,
		FreedBytes:     104857600,
		DurationNS:     42000000,
		ReapedPaths:    []string{"_scratch/fak-worker-wt-1", "_scratch/fak-worker-wt-2"},
		Timestamp:      now,
	}

	if receipt.Schema != "fak-worktree-trigger/1" {
		t.Fatalf("expected schema %q, got %q", "fak-worktree-trigger/1", receipt.Schema)
	}

	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}

	jsonStr := string(data)
	if !strings.Contains(jsonStr, `"schema":"fak-worktree-trigger/1"`) {
		t.Fatalf("json missing schema: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"trigger":"TriggerStaleSweep"`) {
		t.Fatalf("json missing trigger: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"decision":"EXECUTED"`) {
		t.Fatalf("json missing decision: %s", jsonStr)
	}

	var decoded TriggerReceipt
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal receipt: %v", err)
	}

	if decoded.Schema != receipt.Schema {
		t.Errorf("schema mismatch: %q vs %q", decoded.Schema, receipt.Schema)
	}
	if decoded.Trigger != receipt.Trigger {
		t.Errorf("trigger mismatch: %q vs %q", decoded.Trigger, receipt.Trigger)
	}
	if decoded.Decision != receipt.Decision {
		t.Errorf("decision mismatch: %q vs %q", decoded.Decision, receipt.Decision)
	}
	if decoded.InspectedCount != receipt.InspectedCount {
		t.Errorf("inspected mismatch: %d vs %d", decoded.InspectedCount, receipt.InspectedCount)
	}
	if decoded.ReapedCount != receipt.ReapedCount {
		t.Errorf("reaped mismatch: %d vs %d", decoded.ReapedCount, receipt.ReapedCount)
	}
	if decoded.RetainedCount != receipt.RetainedCount {
		t.Errorf("retained mismatch: %d vs %d", decoded.RetainedCount, receipt.RetainedCount)
	}
	if decoded.FreedBytes != receipt.FreedBytes {
		t.Errorf("freed bytes mismatch: %d vs %d", decoded.FreedBytes, receipt.FreedBytes)
	}
	if decoded.DurationNS != receipt.DurationNS {
		t.Errorf("duration mismatch: %d vs %d", decoded.DurationNS, receipt.DurationNS)
	}
	if len(decoded.ReapedPaths) != len(receipt.ReapedPaths) {
		t.Errorf("reaped paths length mismatch: %d vs %d", len(decoded.ReapedPaths), len(receipt.ReapedPaths))
	}
}

func TestTriggerReceiptSchema_SkippedReceipt(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Millisecond)
	receipt := TriggerReceipt{
		Schema:      TriggerReceiptSchema,
		Trigger:     TriggerStaleSweep,
		Decision:    DecisionSkippedCooldown,
		Reason:      "sweep skipped: cooldown active",
		RefusalCode: RefusalCooldownActive,
		DurationNS:  250,
		Timestamp:   now,
	}

	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("failed to marshal skipped receipt: %v", err)
	}

	var decoded TriggerReceipt
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal skipped receipt: %v", err)
	}

	if decoded.Decision != DecisionSkippedCooldown {
		t.Errorf("expected decision %q, got %q", DecisionSkippedCooldown, decoded.Decision)
	}
	if decoded.RefusalCode != RefusalCooldownActive {
		t.Errorf("expected refusal code %q, got %q", RefusalCooldownActive, decoded.RefusalCode)
	}
}
