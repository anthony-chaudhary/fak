package agentopt

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestSpeculativeToolExecution(t *testing.T) {
	ctx := context.Background()
	engine := NewSpeculativeEngine("Read", "Glob", "Grep")

	var execCount atomic.Int32
	mockExec := func(ctx context.Context) (string, error) {
		execCount.Add(1)
		time.Sleep(10 * time.Millisecond)
		return "file contents of main.go", nil
	}

	call := ToolCall{
		ID:       "call-1",
		Name:     "Read",
		Args:     map[string]any{"path": "main.go"},
		ReadOnly: true,
	}

	// 1. Speculative execution runs in advance.
	specReceipt, err := engine.Speculate(ctx, call, mockExec)
	if err != nil {
		t.Fatalf("speculative execution failed: %v", err)
	}
	if !specReceipt.Speculative || specReceipt.Hit {
		t.Fatalf("expected speculative=true, hit=false, got %+v", specReceipt)
	}
	if execCount.Load() != 1 {
		t.Fatalf("expected execCount=1, got %d", execCount.Load())
	}
	if engine.CacheSize() != 1 {
		t.Fatalf("expected CacheSize=1, got %d", engine.CacheSize())
	}

	// 2. Matching call arrives — served instantly with zero duration and hit=true.
	actualReceipt, err := engine.ExecuteOrHit(ctx, call, mockExec)
	if err != nil {
		t.Fatalf("execute or hit failed: %v", err)
	}
	if !actualReceipt.Hit {
		t.Fatalf("expected cached hit, got hit=false")
	}
	if actualReceipt.Duration != 0 {
		t.Fatalf("expected zero wait on speculative hit, got duration=%v", actualReceipt.Duration)
	}
	if actualReceipt.Output != "file contents of main.go" {
		t.Fatalf("output mismatch: %q", actualReceipt.Output)
	}
	if execCount.Load() != 1 {
		t.Fatalf("expected no additional execution, execCount=%d", execCount.Load())
	}

	// 3. Non-read-only tool rejected from speculative execution.
	mutatingCall := ToolCall{
		ID:       "call-2",
		Name:     "Write",
		Args:     map[string]any{"path": "foo.go", "content": "package foo"},
		ReadOnly: false,
	}
	if _, err := engine.Speculate(ctx, mutatingCall, mockExec); err == nil {
		t.Fatal("expected error speculating non-read-only tool call")
	}

	// 4. Invalidation clears cache upon mutation.
	engine.InvalidateTool("Read")
	if engine.CacheSize() != 0 {
		t.Fatalf("expected empty cache after invalidation, got %d", engine.CacheSize())
	}

	// 5. Subsequent call misses cache and re-executes.
	freshReceipt, err := engine.ExecuteOrHit(ctx, call, mockExec)
	if err != nil {
		t.Fatalf("execute after invalidation failed: %v", err)
	}
	if freshReceipt.Hit {
		t.Fatalf("expected cache miss after invalidation, got hit=true")
	}
	if execCount.Load() != 2 {
		t.Fatalf("expected execCount=2 after miss, got %d", execCount.Load())
	}
}

func TestSpeculativeEngineErrorHandling(t *testing.T) {
	ctx := context.Background()
	engine := NewSpeculativeEngine("Grep")

	errExec := func(ctx context.Context) (string, error) {
		return "", errors.New("file not found")
	}

	call := ToolCall{
		ID:       "call-err",
		Name:     "Grep",
		Args:     map[string]any{"pattern": "missing"},
		ReadOnly: true,
	}

	receipt, err := engine.Speculate(ctx, call, errExec)
	if err == nil || receipt.Error != "file not found" {
		t.Fatalf("expected error preserved in receipt, got err=%v receipt=%+v", err, receipt)
	}
}
