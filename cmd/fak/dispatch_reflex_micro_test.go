package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agenttopo"
)

func TestParallelReflexMicroAgentsExecution(t *testing.T) {
	// 1. Simulate 3 parallel micro-agents on disjoint lanes
	tasks := []ReflexMicroTask{
		{
			ID:      "task-alpha",
			Lane:    "lane-alpha",
			Tree:    []string{"internal/alpha/**"},
			Model:   "glm-4.5-air",
			Witness: "witness:alpha:verified",
		},
		{
			ID:      "task-beta",
			Lane:    "lane-beta",
			Tree:    []string{"internal/beta/**"},
			Model:   "qwen3.8-7b",
			Witness: "witness:beta:verified",
		},
		{
			ID:      "task-gamma",
			Lane:    "lane-gamma",
			Tree:    []string{"internal/gamma/**"},
			Model:   "glm-4.5-air",
			Witness: "witness:gamma:verified",
		},
	}

	simulatedDelay := 35 * time.Millisecond
	executor := func(ctx context.Context, task ReflexMicroTask) (agenttopo.ReflexMicroReceipt, error) {
		time.Sleep(simulatedDelay)
		return agenttopo.ReflexMicroReceipt{
			Schema:      agenttopo.ReflexMicroReceiptSchema,
			Lane:        task.Lane,
			Witness:     task.Witness,
			State:       "completed",
			Allowed:     1,
			Denied:      0,
			TokensSaved: 12500,
			ElapsedMs:   simulatedDelay.Milliseconds(),
		}, nil
	}

	req := ReflexMicroDispatchRequest{
		Workspace:   ".",
		Coordinator: "coordinator-root",
		Tasks:       tasks,
		Executor:    executor,
	}

	start := time.Now()
	res, err := ExecuteReflexMicroDispatch(context.Background(), req)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("ExecuteReflexMicroDispatch failed: %v", err)
	}

	// Verify outcome
	if !res.OK {
		t.Fatalf("expected result.OK = true, got false (error: %s)", res.Error)
	}
	if !res.Disjoint {
		t.Fatalf("expected result.Disjoint = true, got false")
	}
	if res.TaskCount != 3 {
		t.Fatalf("expected TaskCount = 3, got %d", res.TaskCount)
	}
	if len(res.Receipts) != 3 {
		t.Fatalf("expected 3 receipts, got %d", len(res.Receipts))
	}
	if res.CombinedState != "completed" {
		t.Fatalf("expected CombinedState = 'completed', got %q", res.CombinedState)
	}
	if res.TotalTokensSaved != 37500 {
		t.Fatalf("expected 37500 tokens saved, got %d", res.TotalTokensSaved)
	}

	// Verify parallel execution (zero coordinator stalls):
	// Total elapsed should be well below sequential execution (3 * 35ms = 105ms)
	if elapsed > 95*time.Millisecond {
		t.Errorf("execution was not parallel: elapsed %v > 95ms (sequential would be ~105ms)", elapsed)
	}

	// Verify each compact receipt schema and properties
	expectedLanes := map[string]bool{
		"lane-alpha": false,
		"lane-beta":  false,
		"lane-gamma": false,
	}
	for _, r := range res.Receipts {
		if r.Schema != "fleet-reflex-micro-receipt/1" {
			t.Errorf("receipt schema = %q, want fleet-reflex-micro-receipt/1", r.Schema)
		}
		if r.State != "completed" {
			t.Errorf("receipt state = %q, want completed", r.State)
		}
		if r.Allowed != 1 || r.Denied != 0 {
			t.Errorf("receipt allowed=%d denied=%d, want 1/0", r.Allowed, r.Denied)
		}
		if !strings.HasPrefix(r.Witness, "witness:") {
			t.Errorf("unexpected witness: %s", r.Witness)
		}
		if _, ok := expectedLanes[r.Lane]; !ok {
			t.Errorf("unexpected lane in receipt: %s", r.Lane)
		}
		expectedLanes[r.Lane] = true
	}
	for lane, seen := range expectedLanes {
		if !seen {
			t.Errorf("missing receipt for lane %s", lane)
		}
	}

	// Verify coordinator context hygiene: only compact receipts collected
	contextBytes := res.CoordinatorContextBytes()
	if contextBytes == 0 || contextBytes > 1000 {
		t.Errorf("coordinator context footprint = %d bytes, expected compact receipt payload (<1000 bytes)", contextBytes)
	}
}

func TestParallelReflexMicroAgentsCollisionRejection(t *testing.T) {
	// Overlapping file trees between lane-alpha and lane-beta
	tasks := []ReflexMicroTask{
		{
			ID:   "task-alpha",
			Lane: "lane-alpha",
			Tree: []string{"internal/alpha/**"},
		},
		{
			ID:   "task-beta",
			Lane: "lane-beta",
			Tree: []string{"internal/alpha/pkg/**"},
		},
	}

	req := ReflexMicroDispatchRequest{
		Workspace:   ".",
		Coordinator: "coordinator-root",
		Tasks:       tasks,
	}

	res, err := ExecuteReflexMicroDispatch(context.Background(), req)
	if err == nil {
		t.Fatal("expected error on overlapping trees, got nil")
	}
	if res.OK {
		t.Error("expected res.OK = false")
	}
	if res.Disjoint {
		t.Error("expected res.Disjoint = false")
	}
	if !strings.Contains(err.Error(), "overlapping tree lease") {
		t.Errorf("expected error to mention 'overlapping tree lease', got: %v", err)
	}
}

func TestParallelReflexMicroAgentsCollidingLaneName(t *testing.T) {
	// Duplicate lane lease name
	tasks := []ReflexMicroTask{
		{
			ID:   "task-1",
			Lane: "lane-alpha",
			Tree: []string{"internal/alpha/**"},
		},
		{
			ID:   "task-2",
			Lane: "lane-alpha",
			Tree: []string{"internal/beta/**"},
		},
	}

	req := ReflexMicroDispatchRequest{
		Workspace:   ".",
		Coordinator: "coordinator-root",
		Tasks:       tasks,
	}

	res, err := ExecuteReflexMicroDispatch(context.Background(), req)
	if err == nil {
		t.Fatal("expected error on duplicate lane name, got nil")
	}
	if res.OK || res.Disjoint {
		t.Errorf("expected OK=false Disjoint=false, got OK=%v Disjoint=%v", res.OK, res.Disjoint)
	}
	if !strings.Contains(err.Error(), "colliding lane lease") {
		t.Errorf("expected error to mention 'colliding lane lease', got: %v", err)
	}
}

func TestRunDispatchReflexCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	argv := []string{
		"--lanes", "lane-alpha,lane-beta,lane-gamma",
		"--trees", "internal/alpha/**,internal/beta/**,internal/gamma/**",
		"--dry-run",
		"--json",
	}

	code := runDispatchReflex(&stdout, &stderr, argv)
	if code != 0 {
		t.Fatalf("runDispatchReflex failed (exit %d): %s", code, stderr.String())
	}

	var result ReflexMicroDispatchResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("failed to decode JSON output: %v\nraw: %s", err, stdout.String())
	}

	if result.Schema != "fleet-reflex-micro-dispatch/1" {
		t.Errorf("schema = %q, want fleet-reflex-micro-dispatch/1", result.Schema)
	}
	if !result.OK || !result.Disjoint || result.TaskCount != 3 {
		t.Errorf("unexpected result: %+v", result)
	}
	if len(result.Receipts) != 3 {
		t.Fatalf("expected 3 receipts, got %d", len(result.Receipts))
	}
	for _, r := range result.Receipts {
		if r.Schema != "fleet-reflex-micro-receipt/1" {
			t.Errorf("receipt schema = %q, want fleet-reflex-micro-receipt/1", r.Schema)
		}
		if r.State != "dry_run" {
			t.Errorf("receipt state = %q, want dry_run", r.State)
		}
	}
}
