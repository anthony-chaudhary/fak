package harnesswarm

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func createTestWorkspace(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	files := map[string]string{
		"go.mod":         "module example.com/testproject\n\ngo 1.26\n",
		".gitignore":     "bin/\n*.tmp\nvendor/\n",
		".fakignore":     "scratch/\n",
		"main.go":        "package main\n\nfunc main() {}\nfunc Helper() string { return \"ok\" }\ntype Config struct{}\n",
		"pkg/calc.go":    "package pkg\n\nfunc Add(a, b int) int { return a + b }\n",
		"scripts/run.py": "def run_task():\n    pass\n\nclass Runner:\n    pass\n",
		"web/index.ts":   "export function renderApp() {}\nexport class AppView {}\n",
		"README.md":      "# Test Project\nSample workspace for warming tests.\n",
	}

	for rel, content := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", filepath.Dir(full), err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", full, err)
		}
	}

	return dir
}

func TestNonBlockingInitializationAndProgressiveCompletion(t *testing.T) {
	ws := createTestWorkspace(t)

	engine := NewEngine(ws, Options{})
	defer engine.Close()

	// Initial check: all stages should be pending
	snap := engine.Snapshot()
	if snap.AllWarm {
		t.Fatalf("expected initial AllWarm to be false")
	}
	for _, s := range AllStages {
		if st := snap.Status(s); st != StatusPending {
			t.Fatalf("expected initial status for %s to be %s, got %s", s, StatusPending, st)
		}
		if engine.IsWarm(s) {
			t.Fatalf("expected stage %s to not be warm initially", s)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	start := time.Now()
	engine.Start(ctx)
	elapsed := time.Since(start)

	// Start must return without blocking (< 50ms)
	if elapsed > 50*time.Millisecond {
		t.Fatalf("Start() blocked for %v, want non-blocking return", elapsed)
	}

	// Progressive stage completion check
	for _, stage := range AllStages {
		if err := engine.WaitStage(ctx, stage); err != nil {
			t.Fatalf("WaitStage(%s) failed: %v", stage, err)
		}
		if !engine.IsWarm(stage) {
			t.Fatalf("expected stage %s to report IsWarm == true after WaitStage", stage)
		}
	}

	finalSnap := engine.Snapshot()
	if !finalSnap.AllWarm {
		t.Fatalf("expected all stages to be warm in final snapshot")
	}

	if len(finalSnap.Files) == 0 {
		t.Fatalf("expected files inventory to be populated")
	}
	if len(finalSnap.Ignore) == 0 {
		t.Fatalf("expected ignore patterns to be populated")
	}
	if len(finalSnap.Manifests) == 0 {
		t.Fatalf("expected manifests to be populated")
	}
	if len(finalSnap.Semantic) == 0 {
		t.Fatalf("expected semantic symbol hints to be populated")
	}

	// Check specific content
	foundGoMod := false
	for _, m := range finalSnap.Manifests {
		if m == "go.mod" {
			foundGoMod = true
			break
		}
	}
	if !foundGoMod {
		t.Errorf("manifests missing go.mod: %v", finalSnap.Manifests)
	}

	// Check Go symbols
	mainSyms := finalSnap.Semantic["main.go"]
	if len(mainSyms) == 0 {
		t.Errorf("expected symbols extracted for main.go")
	}
}

func TestEarlyQueryUnblocking(t *testing.T) {
	ws := createTestWorkspace(t)

	// Add deliberate delay to StageSemantic to prove StageFiles unblocks while
	// StageSemantic is actively warming.
	semanticDelay := 200 * time.Millisecond
	engine := NewEngine(ws, Options{
		StageDelays: map[Stage]time.Duration{
			StageSemantic: semanticDelay,
		},
	})
	defer engine.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	start := time.Now()
	engine.Start(ctx)

	// Wait specifically for StageFiles
	if err := engine.WaitStage(ctx, StageFiles); err != nil {
		t.Fatalf("WaitStage(StageFiles) failed: %v", err)
	}
	filesElapsed := time.Since(start)

	// StageFiles must unblock quickly, well before semantic delay completes
	if filesElapsed >= semanticDelay {
		t.Fatalf("StageFiles unblocked after %v, expected earlier than semantic delay %v", filesElapsed, semanticDelay)
	}

	if !engine.IsWarm(StageFiles) {
		t.Fatalf("StageFiles must be warm")
	}

	// CRITICAL REQUIREMENT: queries for StageFiles return while StageSemantic is still warming
	if engine.IsWarm(StageSemantic) {
		t.Fatalf("StageSemantic should NOT be warm yet while files stage just completed")
	}

	snap := engine.Snapshot()
	if snap.Status(StageSemantic) != StatusWarming && snap.Status(StageSemantic) != StatusPending {
		t.Fatalf("expected StageSemantic to be warming or pending, got %s", snap.Status(StageSemantic))
	}

	// Now wait for StageSemantic to complete
	if err := engine.WaitStage(ctx, StageSemantic); err != nil {
		t.Fatalf("WaitStage(StageSemantic) failed: %v", err)
	}

	if !engine.IsWarm(StageSemantic) {
		t.Fatalf("StageSemantic must be warm after WaitStage")
	}
}

func TestSelectiveInvalidation(t *testing.T) {
	ws := createTestWorkspace(t)

	engine := NewEngine(ws, Options{
		StageDelays: map[Stage]time.Duration{
			StageSemantic:  100 * time.Millisecond,
			StageManifests: 100 * time.Millisecond,
			StageIgnore:    100 * time.Millisecond,
		},
	})
	defer engine.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	engine.Start(ctx)

	// Wait for everything to become warm first
	for _, s := range AllStages {
		if err := engine.WaitStage(ctx, s); err != nil {
			t.Fatalf("initial WaitStage(%s) failed: %v", s, err)
		}
	}

	// Case 1: Invalidate manifest only (e.g. go.mod changed)
	engine.Invalidate("go.mod")

	if engine.IsWarm(StageManifests) {
		t.Errorf("StageManifests should NOT be warm immediately after invalidate")
	}
	if !engine.IsWarm(StageFiles) {
		t.Errorf("StageFiles should remain warm when only manifest invalidated")
	}
	if !engine.IsWarm(StageIgnore) {
		t.Errorf("StageIgnore should remain warm when only manifest invalidated")
	}
	if !engine.IsWarm(StageSemantic) {
		t.Errorf("StageSemantic should remain warm when only manifest invalidated")
	}

	// Wait for StageManifests to re-warm
	if err := engine.WaitStage(ctx, StageManifests); err != nil {
		t.Fatalf("WaitStage(StageManifests) after invalidate failed: %v", err)
	}
	if !engine.IsWarm(StageManifests) {
		t.Errorf("StageManifests should be warm after re-warm")
	}

	// Case 2: Invalidate source code only (e.g. main.go changed)
	engine.Invalidate("main.go")

	if engine.IsWarm(StageSemantic) {
		t.Errorf("StageSemantic should NOT be warm immediately after invalidate")
	}
	if !engine.IsWarm(StageFiles) {
		t.Errorf("StageFiles should remain warm when only source invalidated")
	}
	if !engine.IsWarm(StageIgnore) {
		t.Errorf("StageIgnore should remain warm when only source invalidated")
	}
	if !engine.IsWarm(StageManifests) {
		t.Errorf("StageManifests should remain warm when only source invalidated")
	}

	// Wait for StageSemantic to re-warm
	if err := engine.WaitStage(ctx, StageSemantic); err != nil {
		t.Fatalf("WaitStage(StageSemantic) after invalidate failed: %v", err)
	}
	if !engine.IsWarm(StageSemantic) {
		t.Errorf("StageSemantic should be warm after re-warm")
	}

	// Case 3: Invalidate ignore rules only (e.g. .gitignore changed)
	engine.Invalidate(".gitignore")

	if engine.IsWarm(StageIgnore) {
		t.Errorf("StageIgnore should NOT be warm immediately after invalidate")
	}
	if !engine.IsWarm(StageFiles) {
		t.Errorf("StageFiles should remain warm when only ignore invalidated")
	}
	if !engine.IsWarm(StageManifests) {
		t.Errorf("StageManifests should remain warm when only ignore invalidated")
	}
	if !engine.IsWarm(StageSemantic) {
		t.Errorf("StageSemantic should remain warm when only ignore invalidated")
	}

	if err := engine.WaitStage(ctx, StageIgnore); err != nil {
		t.Fatalf("WaitStage(StageIgnore) after invalidate failed: %v", err)
	}
	if !engine.IsWarm(StageIgnore) {
		t.Errorf("StageIgnore should be warm after re-warm")
	}
}

func TestConcurrentSafety(t *testing.T) {
	ws := createTestWorkspace(t)

	engine := NewEngine(ws, Options{
		StageDelays: map[Stage]time.Duration{
			StageSemantic: 10 * time.Millisecond,
		},
	})
	defer engine.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	engine.Start(ctx)

	var wg sync.WaitGroup
	var readOps atomic.Int64
	var invalOps atomic.Int64

	// Spawn readers
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			stage := AllStages[workerID%len(AllStages)]
			for {
				select {
				case <-ctx.Done():
					return
				default:
					_ = engine.IsWarm(stage)
					_ = engine.Snapshot()
					readCtx, readCancel := context.WithTimeout(ctx, 15*time.Millisecond)
					_ = engine.WaitStage(readCtx, stage)
					readCancel()
					readOps.Add(1)
					time.Sleep(1 * time.Millisecond)
				}
			}
		}(i)
	}

	// Spawn invalidators
	invalidations := [][]string{
		{"main.go"},
		{"go.mod"},
		{".gitignore"},
		{"pkg/calc.go"},
		{"scripts/run.py"},
		{"README.md"},
	}

	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				default:
					target := invalidations[(workerID+int(invalOps.Load()))%len(invalidations)]
					engine.Invalidate(target...)
					invalOps.Add(1)
					time.Sleep(5 * time.Millisecond)
				}
			}
		}(i)
	}

	wg.Wait()

	if readOps.Load() == 0 {
		t.Errorf("expected read operations to execute")
	}
	if invalOps.Load() == 0 {
		t.Errorf("expected invalidation operations to execute")
	}
}

func TestStatusTransitions(t *testing.T) {
	ws := createTestWorkspace(t)

	type transition struct {
		stage Stage
		from  Status
		to    Status
	}

	var mu sync.Mutex
	var transitions []transition

	engine := NewEngine(ws, Options{
		OnStageTransition: func(stage Stage, from Status, to Status) {
			mu.Lock()
			transitions = append(transitions, transition{stage: stage, from: from, to: to})
			mu.Unlock()
		},
	})
	defer engine.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	engine.Start(ctx)

	for _, s := range AllStages {
		if err := engine.WaitStage(ctx, s); err != nil {
			t.Fatalf("WaitStage(%s) failed: %v", s, err)
		}
	}

	engine.Invalidate("main.go")
	if err := engine.WaitStage(ctx, StageSemantic); err != nil {
		t.Fatalf("WaitStage(StageSemantic) after invalidate failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	hasWarm := false
	hasStale := false
	for _, tr := range transitions {
		if tr.to == StatusWarm {
			hasWarm = true
		}
		if tr.to == StatusStale {
			hasStale = true
		}
	}

	if !hasWarm {
		t.Errorf("expected StatusWarm transitions in log")
	}
	if !hasStale {
		t.Errorf("expected StatusStale transitions in log after Invalidate")
	}
}

func TestWaitStageContextCancellation(t *testing.T) {
	ws := createTestWorkspace(t)

	engine := NewEngine(ws, Options{
		StageDelays: map[Stage]time.Duration{
			StageSemantic: 1 * time.Second,
		},
	})
	defer engine.Close()

	ctx, cancel := context.WithCancel(context.Background())
	engine.Start(ctx)

	// Cancel context quickly
	shortCtx, shortCancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer shortCancel()

	err := engine.WaitStage(shortCtx, StageSemantic)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded, got %v", err)
	}
	cancel()
}

func TestCloseUnblocksWaiters(t *testing.T) {
	ws := createTestWorkspace(t)

	engine := NewEngine(ws, Options{
		StageDelays: map[Stage]time.Duration{
			StageSemantic: 2 * time.Second,
		},
	})

	ctx := context.Background()
	engine.Start(ctx)

	errCh := make(chan error, 1)
	go func() {
		errCh <- engine.WaitStage(ctx, StageSemantic)
	}()

	time.Sleep(20 * time.Millisecond)
	engine.Close()

	select {
	case err := <-errCh:
		if !errors.Is(err, ErrClosed) {
			t.Fatalf("expected ErrClosed on engine.Close(), got %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("WaitStage did not unblock when engine was closed")
	}
}

func TestNonExistentRootFailsGracefully(t *testing.T) {
	nonExistent := filepath.Join(t.TempDir(), "does_not_exist")

	engine := NewEngine(nonExistent, Options{})
	defer engine.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	engine.Start(ctx)

	err := engine.WaitStage(ctx, StageFiles)
	if err == nil {
		t.Fatalf("expected error for non-existent root, got nil")
	}

	snap := engine.Snapshot()
	if snap.Status(StageFiles) != StatusFailed {
		t.Fatalf("expected StatusFailed for StageFiles, got %s", snap.Status(StageFiles))
	}
}
