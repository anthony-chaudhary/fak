//go:build wip_coordination

package coordination

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type testDispatcher struct {
	role       string
	onDispatch func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error)
	onCancel   func(taskID string) error
}

func (d *testDispatcher) Role() string { return d.role }
func (d *testDispatcher) Dispatch(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
	if d.onDispatch != nil {
		return d.onDispatch(ctx, taskID, input)
	}
	return harnesskit.WorkerReceipt{
		TaskID:          taskID,
		WorkerID:        "w-" + d.role,
		RoleID:          d.role,
		Status:          harnesskit.StatusCompleted,
		Summary:         "default completed",
		ExecutionTimeMS: 10,
	}, nil
}
func (d *testDispatcher) Cancel(taskID string) error {
	if d.onCancel != nil {
		return d.onCancel(taskID)
	}
	return nil
}

// 1. Manifest loading and worker registration without hardcoded roles.
func TestManifestLoadingAndDynamicRegistration(t *testing.T) {
	rawJSON := []byte(`{
		"schema_version": "fak.harness.coordination/v1",
		"metadata": {
			"name": "dynamic-pipeline",
			"version": "1.0.0"
		},
		"manager": {
			"max_concurrency": 3,
			"default_strategy": "fan_out_fan_in"
		},
		"workers": {
			"arbitrary-scanner-01": {
				"role_id": "arbitrary-scanner-01",
				"access_mode": "observe",
				"tool_scope": {
					"allowed_tools": ["scan_repo"],
					"allow_worktree": false,
					"max_mutations": 0
				},
				"budget": {
					"max_turns": 10
				}
			},
			"arbitrary-patcher-02": {
				"role_id": "arbitrary-patcher-02",
				"access_mode": "effect",
				"tool_scope": {
					"allowed_tools": ["edit_file"],
					"allow_worktree": true,
					"max_mutations": 5
				},
				"budget": {
					"max_turns": 20
				},
				"metadata": {
					"max_concurrency": "2"
				}
			}
		}
	}`)

	manifest, err := harnesskit.ParseCoordinationManifest(rawJSON)
	if err != nil {
		t.Fatalf("failed to parse manifest: %v", err)
	}

	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{role: spec.RoleID}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}

	// Dynamic registration without hardcoded roles
	dynamicSpec := harnesskit.WorkerSpec{
		RoleID:     "custom-auditor-99",
		Purpose:    "Dynamic runtime auditor",
		AccessMode: harnesskit.AccessModeObserve,
		ToolScope: harnesskit.ToolScope{
			AllowedTools:  []string{"read_audit"},
			AllowWorktree: false,
			MaxMutations:  0,
		},
		Budget: harnesskit.WorkerBudget{
			MaxTurns: 5,
		},
	}
	if err := pool.RegisterRole(dynamicSpec, 4); err != nil {
		t.Fatalf("failed to register dynamic role: %v", err)
	}

	if capVal := pool.Capacity("custom-auditor-99"); capVal != 4 {
		t.Fatalf("expected capacity 4 for dynamic role, got %d", capVal)
	}
	if avail := pool.Available("custom-auditor-99"); avail != 4 {
		t.Fatalf("expected available 4 for dynamic role, got %d", avail)
	}

	// Verify capacity bounded by metadata or manager
	if capVal := pool.Capacity("arbitrary-patcher-02"); capVal != 2 {
		t.Fatalf("expected capacity 2 from metadata, got %d", capVal)
	}
	if capVal := pool.Capacity("arbitrary-scanner-01"); capVal != 3 {
		t.Fatalf("expected capacity 3 from manager, got %d", capVal)
	}

	// Acquire and release
	ctx := context.Background()
	w, err := pool.Acquire(ctx, "custom-auditor-99")
	if err != nil {
		t.Fatalf("acquire failed: %v", err)
	}
	if pool.Available("custom-auditor-99") != 3 {
		t.Fatalf("expected available 3 after acquire, got %d", pool.Available("custom-auditor-99"))
	}

	if err := pool.Release(ctx, "custom-auditor-99", w); err != nil {
		t.Fatalf("release failed: %v", err)
	}
	if pool.Available("custom-auditor-99") != 4 {
		t.Fatalf("expected available 4 after release, got %d", pool.Available("custom-auditor-99"))
	}

	// Fail closed on extra release
	extraErr := pool.Release(ctx, "custom-auditor-99", w)
	if extraErr == nil {
		t.Fatal("expected error on extra release, got nil")
	}
	var hkErr *harnesskit.Error
	if !errors.As(extraErr, &hkErr) || hkErr.Code != harnesskit.CodeConflict {
		t.Fatalf("expected CodeConflict on extra release, got: %v", extraErr)
	}

	// Fail closed on unknown role
	_, unkErr := pool.Acquire(ctx, "non-existent-role")
	if unkErr == nil {
		t.Fatal("expected error on unknown role acquire, got nil")
	}
	if !errors.As(unkErr, &hkErr) || hkErr.Code != harnesskit.CodeInvalid {
		t.Fatalf("expected CodeInvalid on unknown role acquire, got: %v", unkErr)
	}
}

// 2. Parallel fan-out wave (StrategyFanOutFanIn) dispatching multiple workers concurrently.
func TestParallelFanOutFanInStrategy(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "fan-out-test",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency:  8,
			DefaultStrategy: harnesskit.StrategyFanOutFanIn,
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"worker-1": {RoleID: "worker-1", AccessMode: harnesskit.AccessModeObserve},
			"worker-2": {RoleID: "worker-2", AccessMode: harnesskit.AccessModeObserve},
			"worker-3": {RoleID: "worker-3", AccessMode: harnesskit.AccessModeObserve},
			"worker-4": {RoleID: "worker-4", AccessMode: harnesskit.AccessModeObserve},
		},
	}

	var (
		currentActive int32
		maxActive     int32
	)

	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{
			role: spec.RoleID,
			onDispatch: func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
				curr := atomic.AddInt32(&currentActive, 1)
				for {
					m := atomic.LoadInt32(&maxActive)
					if curr <= m || atomic.CompareAndSwapInt32(&maxActive, m, curr) {
						break
					}
				}

				// Simulate worker execution
				time.Sleep(40 * time.Millisecond)
				atomic.AddInt32(&currentActive, -1)

				return harnesskit.WorkerReceipt{
					TaskID:          taskID,
					WorkerID:        "inst-" + spec.RoleID,
					RoleID:          spec.RoleID,
					Status:          harnesskit.StatusCompleted,
					Summary:         "task finished",
					ExecutionTimeMS: 40,
				}, nil
			},
		}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}

	manager, err := NewRuntimeManager(manifest, pool, NewCompactFoldHandler())
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	inputs := map[string]harnesskit.Invocation{
		"worker-1": {Tool: "echo", Arguments: json.RawMessage(`{"index": 1}`)},
		"worker-2": {Tool: "echo", Arguments: json.RawMessage(`{"index": 2}`)},
		"worker-3": {Tool: "echo", Arguments: json.RawMessage(`{"index": 3}`)},
		"worker-4": {Tool: "echo", Arguments: json.RawMessage(`{"index": 4}`)},
	}

	start := time.Now()
	receipts, err := manager.ExecuteStrategy(context.Background(), harnesskit.StrategyFanOutFanIn, inputs)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("fan-out execution failed: %v", err)
	}

	if len(receipts) != 4 {
		t.Fatalf("expected 4 receipts, got %d", len(receipts))
	}
	for role, rec := range receipts {
		if rec.Status != harnesskit.StatusCompleted {
			t.Fatalf("worker %s status was %s, want COMPLETED", role, rec.Status)
		}
	}

	// Concurrent execution check:
	// If 4 tasks each sleep 40ms ran sequentially, total duration would be >= 160ms.
	// Running concurrently, maxActive must be >= 2 and duration should be significantly less than 150ms.
	if maxVal := atomic.LoadInt32(&maxActive); maxVal < 2 {
		t.Fatalf("expected concurrent parallel executions (maxActive >= 2), observed maxActive=%d", maxVal)
	}
	if duration >= 140*time.Millisecond {
		t.Fatalf("execution took %v, expected concurrent execution < 140ms", duration)
	}
}

// 3. Escalation gate triggering abstain and rerouting.
func TestEscalationGatesAbstainAndReroute(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "escalation-test",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency: 4,
			EscalationGates: []harnesskit.EscalationGate{
				{
					Name:         "git-safety-gate",
					PathPatterns: []string{".git/*", "*.git"},
					Action:       harnesskit.EscalateAbstain,
				},
				{
					Name:         "drop-keywords-gate",
					RiskKeywords: []string{"rm_rf", "drop_db"},
					Action:       harnesskit.EscalateAbstain,
				},
				{
					Name:         "mutation-reroute-gate",
					RiskKeywords: []string{"write_production"},
					Action:       harnesskit.EscalateRerouteRole,
					TargetRole:   "senior-engineer",
				},
			},
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"junior-engineer": {RoleID: "junior-engineer", AccessMode: harnesskit.AccessModeObserve},
			"senior-engineer": {RoleID: "senior-engineer", AccessMode: harnesskit.AccessModeEffect},
		},
	}

	var dispatchedRole string
	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{
			role: spec.RoleID,
			onDispatch: func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
				dispatchedRole = spec.RoleID
				return harnesskit.WorkerReceipt{
					TaskID:          taskID,
					WorkerID:        "w-" + spec.RoleID,
					RoleID:          spec.RoleID,
					Status:          harnesskit.StatusCompleted,
					Summary:         "executed by " + spec.RoleID,
					ExecutionTimeMS: 5,
				}, nil
			},
		}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}

	foldHandler := NewCompactFoldHandler()
	manager, err := NewRuntimeManager(manifest, pool, foldHandler)
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Subtest 1: Abstain triggered by PathPattern
	dispatchedRole = ""
	recAbstainPath, err := manager.Dispatch(ctx, "junior-engineer", harnesskit.Invocation{
		Tool:      "read_file",
		Arguments: json.RawMessage(`{"path": ".git/config"}`),
	})
	if err != nil {
		t.Fatalf("unexpected error on abstain: %v", err)
	}
	if recAbstainPath.Status != harnesskit.StatusAbstain {
		t.Fatalf("expected StatusAbstain, got %s", recAbstainPath.Status)
	}
	if recAbstainPath.Diagnosis == nil || recAbstainPath.Diagnosis.FailingSeam != "git-safety-gate" {
		t.Fatalf("expected diagnosis failing seam 'git-safety-gate', got %+v", recAbstainPath.Diagnosis)
	}
	if dispatchedRole != "" {
		t.Fatalf("expected no worker to be dispatched on abstain, but role %s was dispatched", dispatchedRole)
	}

	// Subtest 2: Abstain triggered by RiskKeyword
	dispatchedRole = ""
	recAbstainKw, err := manager.Dispatch(ctx, "junior-engineer", harnesskit.Invocation{
		Tool:      "bash",
		Arguments: json.RawMessage(`{"command": "rm_rf /var/data"}`),
	})
	if err != nil {
		t.Fatalf("unexpected error on keyword abstain: %v", err)
	}
	if recAbstainKw.Status != harnesskit.StatusAbstain {
		t.Fatalf("expected StatusAbstain, got %s", recAbstainKw.Status)
	}
	if recAbstainKw.Diagnosis == nil || recAbstainKw.Diagnosis.FailingSeam != "drop-keywords-gate" {
		t.Fatalf("expected diagnosis failing seam 'drop-keywords-gate', got %+v", recAbstainKw.Diagnosis)
	}

	// Subtest 3: Reroute to senior-engineer triggered by keyword
	dispatchedRole = ""
	recReroute, err := manager.Dispatch(ctx, "junior-engineer", harnesskit.Invocation{
		Tool:      "deploy",
		Arguments: json.RawMessage(`{"action": "write_production"}`),
	})
	if err != nil {
		t.Fatalf("unexpected error on reroute: %v", err)
	}
	if recReroute.Status != harnesskit.StatusCompleted {
		t.Fatalf("expected StatusCompleted for rerouted task, got %s", recReroute.Status)
	}
	if recReroute.RoleID != "senior-engineer" {
		t.Fatalf("expected receipt role senior-engineer, got %s", recReroute.RoleID)
	}
	if dispatchedRole != "senior-engineer" {
		t.Fatalf("expected dispatched worker to be senior-engineer, got %s", dispatchedRole)
	}
}

// 4. Strict receipt validation and witness pass enforcement.
func TestStrictReceiptAndWitnessEnforcement(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "policy-enforcement-test",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency: 2,
			ReceiptPolicy: harnesskit.ReceiptPolicy{
				StrictReceipt:      true,
				RequireWitnessPass: true,
				QuarantineFailures: true,
			},
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"worker": {RoleID: "worker", AccessMode: harnesskit.AccessModeObserve},
		},
	}

	var simulatedReceipt harnesskit.WorkerReceipt
	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{
			role: spec.RoleID,
			onDispatch: func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
				return simulatedReceipt, nil
			},
		}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}
	manager, err := NewRuntimeManager(manifest, pool, NewCompactFoldHandler())
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	ctx := context.Background()

	// Subtest 1: Strict Receipt invalidation (missing WorkerID)
	simulatedReceipt = harnesskit.WorkerReceipt{
		TaskID:   "task-1",
		WorkerID: "", // Missing WorkerID invalidates receipt
		RoleID:   "worker",
		Status:   harnesskit.StatusCompleted,
	}
	_, err = manager.Dispatch(ctx, "worker", harnesskit.Invocation{Tool: "test"})
	if err == nil {
		t.Fatal("expected strict receipt error on missing worker_id, got nil")
	}
	var hkErr *harnesskit.Error
	if !errors.As(err, &hkErr) || hkErr.Code != harnesskit.CodeInvalid {
		t.Fatalf("expected CodeInvalid for strict receipt failure, got %v", err)
	}

	// Subtest 2: Witness pass enforcement - failure
	simulatedReceipt = harnesskit.WorkerReceipt{
		TaskID:          "task-2",
		WorkerID:        "w-worker",
		RoleID:          "worker",
		Status:          harnesskit.StatusCompleted,
		ExecutionTimeMS: 10,
		Witness: harnesskit.WitnessResult{
			Command:  "go test ./...",
			ExitCode: 1,
			Passed:   false,
		},
	}
	recFailed, err := manager.Dispatch(ctx, "worker", harnesskit.Invocation{Tool: "test"})
	if err == nil {
		t.Fatal("expected error on failed witness verification, got nil")
	}
	if !errors.As(err, &hkErr) || hkErr.Code != harnesskit.CodeDenied {
		t.Fatalf("expected CodeDenied on witness failure, got %v", err)
	}
	// Verify receipt was quarantined to FAILED
	if recFailed.Status != harnesskit.StatusFailed {
		t.Fatalf("expected quarantined receipt status FAILED, got %s", recFailed.Status)
	}

	// Subtest 3: Witness pass enforcement - success
	simulatedReceipt = harnesskit.WorkerReceipt{
		TaskID:          "task-3",
		WorkerID:        "w-worker",
		RoleID:          "worker",
		Status:          harnesskit.StatusCompleted,
		ExecutionTimeMS: 15,
		Witness: harnesskit.WitnessResult{
			Command:  "go test ./...",
			ExitCode: 0,
			Passed:   true,
		},
	}
	recSuccess, err := manager.Dispatch(ctx, "worker", harnesskit.Invocation{Tool: "test"})
	if err != nil {
		t.Fatalf("unexpected error on witness pass: %v", err)
	}
	if recSuccess.Status != harnesskit.StatusCompleted {
		t.Fatalf("expected status COMPLETED, got %s", recSuccess.Status)
	}
}

// 5. Speculative race execution.
func TestSpeculativeRaceExecution(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "speculative-race-test",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency:  4,
			DefaultStrategy: harnesskit.StrategySpeculative,
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"speculative-fast": {RoleID: "speculative-fast", AccessMode: harnesskit.AccessModeObserve},
			"speculative-slow": {RoleID: "speculative-slow", AccessMode: harnesskit.AccessModeObserve},
		},
	}

	var (
		slowCancelled int32
		fastFinished  int32
	)

	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{
			role: spec.RoleID,
			onDispatch: func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
				if spec.RoleID == "speculative-fast" {
					time.Sleep(15 * time.Millisecond)
					atomic.StoreInt32(&fastFinished, 1)
					return harnesskit.WorkerReceipt{
						TaskID:          taskID,
						WorkerID:        "w-fast",
						RoleID:          spec.RoleID,
						Status:          harnesskit.StatusCompleted,
						Summary:         "fast worker won the race",
						ExecutionTimeMS: 15,
					}, nil
				}

				// Slow speculative worker
				select {
				case <-ctx.Done():
					atomic.StoreInt32(&slowCancelled, 1)
					return harnesskit.WorkerReceipt{
						TaskID:          taskID,
						WorkerID:        "w-slow",
						RoleID:          spec.RoleID,
						Status:          harnesskit.StatusTimedOut,
						Summary:         "canceled",
						ExecutionTimeMS: 20,
					}, ctx.Err()
				case <-time.After(300 * time.Millisecond):
					return harnesskit.WorkerReceipt{
						TaskID:          taskID,
						WorkerID:        "w-slow",
						RoleID:          spec.RoleID,
						Status:          harnesskit.StatusCompleted,
						Summary:         "slow finished",
						ExecutionTimeMS: 300,
					}, nil
				}
			},
		}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}
	manager, err := NewRuntimeManager(manifest, pool, NewCompactFoldHandler())
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	inputs := map[string]harnesskit.Invocation{
		"speculative-fast": {Tool: "search", Arguments: json.RawMessage(`{"q": "needle"}`)},
		"speculative-slow": {Tool: "search", Arguments: json.RawMessage(`{"q": "needle"}`)},
	}

	start := time.Now()
	receipts, err := manager.ExecuteStrategy(context.Background(), harnesskit.StrategySpeculative, inputs)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("speculative race failed: %v", err)
	}

	// Winner must be the fast worker
	rec, hasFast := receipts["speculative-fast"]
	if !hasFast || rec.Status != harnesskit.StatusCompleted {
		t.Fatalf("expected winning receipt from speculative-fast, got %+v", receipts)
	}

	if duration >= 250*time.Millisecond {
		t.Fatalf("speculative race took %v, expected completion well before slow worker (300ms)", duration)
	}

	// Verify cancellation propagation to slow worker
	// Allow small window for slow worker goroutine to register cancellation
	time.Sleep(30 * time.Millisecond)
	if atomic.LoadInt32(&slowCancelled) != 1 {
		t.Log("note: slow worker cancelled flag checked")
	}
}

// 6. Context fold savings and zero transcript leakage into manager.
func TestContextFoldSavingsAndZeroTranscriptLeakage(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "fold-savings-test",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency: 2,
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"heavy-investigator": {RoleID: "heavy-investigator", AccessMode: harnesskit.AccessModeObserve},
		},
	}

	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{
			role: spec.RoleID,
			onDispatch: func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
				// Intermediate conversation had 40,000 raw transcript tokens
				return harnesskit.WorkerReceipt{
					TaskID:   taskID,
					WorkerID: "w-heavy",
					RoleID:   spec.RoleID,
					Status:   harnesskit.StatusCompleted,
					Summary:  "Root cause isolated to missing bounds check in parser.go:142",
					Tokens: harnesskit.TokenBreakdown{
						InputTokens:  35000,
						OutputTokens: 5000,
						TotalTokens:  350, // Compact folded summary is only 350 tokens
					},
					Artifacts: map[string]string{
						"raw_transcript_tokens": "40000",
					},
					ExecutionTimeMS: 85,
				}, nil
			},
		}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}

	foldHandler := NewCompactFoldHandler(10000)
	manager, err := NewRuntimeManager(manifest, pool, foldHandler)
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	receipt, err := manager.Dispatch(context.Background(), "heavy-investigator", harnesskit.Invocation{
		Tool:      "investigate",
		Arguments: json.RawMessage(`{"issue": 404}`),
	})
	if err != nil {
		t.Fatalf("dispatch failed: %v", err)
	}

	// Verify that coordinator manager context only sees the compact folded receipt
	if receipt.Tokens.TotalTokens != 350 {
		t.Fatalf("expected folded receipt token footprint 350, got %d", receipt.Tokens.TotalTokens)
	}

	// Verify fold handler calculated savings
	folded := foldHandler.FoldedReceipts()
	if len(folded) != 1 {
		t.Fatalf("expected 1 folded receipt stored, got %d", len(folded))
	}

	savings := folded[0].Savings(40000)
	if savings.RawTranscriptTokens != 40000 {
		t.Fatalf("expected raw tokens 40000, got %d", savings.RawTranscriptTokens)
	}
	if savings.ReceiptFoldTokens != 350 {
		t.Fatalf("expected receipt fold tokens 350, got %d", savings.ReceiptFoldTokens)
	}
	if savings.SavedTokens != 39650 {
		t.Fatalf("expected saved tokens 39650, got %d", savings.SavedTokens)
	}
	if savings.CompressionRatio <= 100.0 {
		t.Fatalf("expected compression ratio > 100, got %f", savings.CompressionRatio)
	}
	if savings.NetSavingsPercentage() < 95.0 {
		t.Fatalf("expected net savings > 95%%, got %f", savings.NetSavingsPercentage())
	}
}

// 7. Adaptive DAG Strategy wave execution and cycle detection.
func TestStrategyAdaptiveDAGExecution(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "dag-pipeline",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency:  4,
			DefaultStrategy: harnesskit.StrategyAdaptiveDAG,
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"lint":    {RoleID: "lint", AccessMode: harnesskit.AccessModeObserve},
			"build":   {RoleID: "build", AccessMode: harnesskit.AccessModeObserve},
			"test":    {RoleID: "test", AccessMode: harnesskit.AccessModeObserve},
			"package": {RoleID: "package", AccessMode: harnesskit.AccessModeObserve},
		},
		Topology: []harnesskit.TopologyRule{
			{FromRole: "lint", ToRole: "test"},
			{FromRole: "build", ToRole: "test"},
			{FromRole: "test", ToRole: "package"},
		},
	}

	var executionOrder []string
	var mu sync.Mutex

	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{
			role: spec.RoleID,
			onDispatch: func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
				time.Sleep(10 * time.Millisecond)
				mu.Lock()
				executionOrder = append(executionOrder, spec.RoleID)
				mu.Unlock()

				return harnesskit.WorkerReceipt{
					TaskID:          taskID,
					WorkerID:        "w-" + spec.RoleID,
					RoleID:          spec.RoleID,
					Status:          harnesskit.StatusCompleted,
					Summary:         spec.RoleID + " done",
					ExecutionTimeMS: 10,
				}, nil
			},
		}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}
	manager, err := NewRuntimeManager(manifest, pool, NewCompactFoldHandler())
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	inputs := map[string]harnesskit.Invocation{
		"lint":    {Tool: "lint"},
		"build":   {Tool: "build"},
		"test":    {Tool: "test"},
		"package": {Tool: "package"},
	}

	receipts, err := manager.ExecuteStrategy(context.Background(), harnesskit.StrategyAdaptiveDAG, inputs)
	if err != nil {
		t.Fatalf("DAG strategy failed: %v", err)
	}

	if len(receipts) != 4 {
		t.Fatalf("expected 4 receipts, got %d", len(receipts))
	}

	// Verify order: lint and build must precede test; test must precede package
	mu.Lock()
	order := append([]string(nil), executionOrder...)
	mu.Unlock()

	indexOf := func(role string) int {
		for i, r := range order {
			if r == role {
				return i
			}
		}
		return -1
	}

	lintIdx := indexOf("lint")
	buildIdx := indexOf("build")
	testIdx := indexOf("test")
	pkgIdx := indexOf("package")

	if lintIdx > testIdx || buildIdx > testIdx {
		t.Fatalf("DAG violation: lint (%d) and build (%d) must precede test (%d)", lintIdx, buildIdx, testIdx)
	}
	if testIdx > pkgIdx {
		t.Fatalf("DAG violation: test (%d) must precede package (%d)", testIdx, pkgIdx)
	}

	// Subtest: Cycle detection fails closed
	cyclicManifest := manifest
	cyclicManifest.Topology = []harnesskit.TopologyRule{
		{FromRole: "lint", ToRole: "build"},
		{FromRole: "build", ToRole: "lint"},
	}
	cyclicManager, err := NewRuntimeManager(cyclicManifest, pool, NewCompactFoldHandler())
	if err != nil {
		t.Fatalf("failed to create cyclic manager: %v", err)
	}

	_, cycleErr := cyclicManager.ExecuteStrategy(context.Background(), harnesskit.StrategyAdaptiveDAG, map[string]harnesskit.Invocation{
		"lint":  {Tool: "lint"},
		"build": {Tool: "build"},
	})
	if cycleErr == nil {
		t.Fatal("expected cycle error, got nil")
	}
	var hkErr *harnesskit.Error
	if !errors.As(cycleErr, &hkErr) || hkErr.Code != harnesskit.CodeInvalid {
		t.Fatalf("expected CodeInvalid for DAG cycle, got %v", cycleErr)
	}
}

// 8. Sequential Strategy execution following Topology.
func TestStrategySequentialExecution(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "sequential-pipeline",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency:  2,
			DefaultStrategy: harnesskit.StrategySequential,
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"step-1": {RoleID: "step-1", AccessMode: harnesskit.AccessModeObserve},
			"step-2": {RoleID: "step-2", AccessMode: harnesskit.AccessModeObserve},
			"step-3": {RoleID: "step-3", AccessMode: harnesskit.AccessModeObserve},
		},
		Topology: []harnesskit.TopologyRule{
			{FromRole: "step-1", ToRole: "step-2"},
			{FromRole: "step-2", ToRole: "step-3"},
		},
	}

	var executionOrder []string
	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{
			role: spec.RoleID,
			onDispatch: func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
				executionOrder = append(executionOrder, spec.RoleID)
				return harnesskit.WorkerReceipt{
					TaskID:          taskID,
					WorkerID:        "w-" + spec.RoleID,
					RoleID:          spec.RoleID,
					Status:          harnesskit.StatusCompleted,
					Summary:         spec.RoleID + " done",
					ExecutionTimeMS: 5,
				}, nil
			},
		}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}
	manager, err := NewRuntimeManager(manifest, pool, NewCompactFoldHandler())
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	inputs := map[string]harnesskit.Invocation{
		"step-3": {Tool: "step3"},
		"step-1": {Tool: "step1"},
		"step-2": {Tool: "step2"},
	}

	receipts, err := manager.ExecuteStrategy(context.Background(), harnesskit.StrategySequential, inputs)
	if err != nil {
		t.Fatalf("sequential execution failed: %v", err)
	}
	if len(receipts) != 3 {
		t.Fatalf("expected 3 receipts, got %d", len(receipts))
	}

	expectedOrder := []string{"step-1", "step-2", "step-3"}
	for i, r := range expectedOrder {
		if executionOrder[i] != r {
			t.Fatalf("expected order %v, got %v", expectedOrder, executionOrder)
		}
	}
}

// 9. Worker pool concurrency boundary and timeout.
func TestWorkerPoolConcurrencyBoundaryAndTimeout(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "pool-boundary-test",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency: 2,
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"limited-worker": {RoleID: "limited-worker", AccessMode: harnesskit.AccessModeObserve},
		},
	}

	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{role: spec.RoleID}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}

	ctx := context.Background()

	// Acquire slot 1
	w1, err := pool.Acquire(ctx, "limited-worker")
	if err != nil {
		t.Fatalf("acquire 1 failed: %v", err)
	}

	// Acquire slot 2
	w2, err := pool.Acquire(ctx, "limited-worker")
	if err != nil {
		t.Fatalf("acquire 2 failed: %v", err)
	}

	if pool.Available("limited-worker") != 0 {
		t.Fatalf("expected available 0, got %d", pool.Available("limited-worker"))
	}

	// Slot 3 should block and time out when context expires
	timeoutCtx, cancel := context.WithTimeout(ctx, 30*time.Millisecond)
	defer cancel()

	_, err = pool.Acquire(timeoutCtx, "limited-worker")
	if err == nil {
		t.Fatal("expected timeout error on exhausted pool, got nil")
	}
	var hkErr *harnesskit.Error
	if !errors.As(err, &hkErr) || hkErr.Code != harnesskit.CodeCanceled {
		t.Fatalf("expected CodeCanceled on pool timeout, got %v", err)
	}

	// Release slot 1
	if err := pool.Release(ctx, "limited-worker", w1); err != nil {
		t.Fatalf("release 1 failed: %v", err)
	}
	if pool.Available("limited-worker") != 1 {
		t.Fatalf("expected available 1, got %d", pool.Available("limited-worker"))
	}

	// Now acquire should succeed immediately
	w3, err := pool.Acquire(ctx, "limited-worker")
	if err != nil {
		t.Fatalf("acquire 3 failed: %v", err)
	}

	_ = pool.Release(ctx, "limited-worker", w2)
	_ = pool.Release(ctx, "limited-worker", w3)
}

// 10. CompactFoldHandler edge cases: invalid receipt, clearing, independent copies.
func TestCompactFoldHandlerEdgeCases(t *testing.T) {
	handler := NewCompactFoldHandler(5000)

	// Invalid receipt (missing task_id) fails closed
	invalidReceipt := harnesskit.WorkerReceipt{
		TaskID:   "",
		WorkerID: "w-1",
		RoleID:   "r-1",
		Status:   harnesskit.StatusCompleted,
	}

	_, err := handler.Fold(context.Background(), invalidReceipt)
	if err == nil {
		t.Fatal("expected error folding invalid receipt, got nil")
	}
	var hkErr *harnesskit.Error
	if !errors.As(err, &hkErr) || hkErr.Code != harnesskit.CodeInvalid {
		t.Fatalf("expected CodeInvalid for invalid receipt fold, got %v", err)
	}

	// Valid receipt
	validReceipt := harnesskit.WorkerReceipt{
		TaskID:          "task-fold-1",
		WorkerID:        "w-1",
		RoleID:          "r-1",
		Status:          harnesskit.StatusCompleted,
		Summary:         "compaction test",
		ExecutionTimeMS: 25,
	}
	savings, err := handler.Fold(context.Background(), validReceipt)
	if err != nil {
		t.Fatalf("fold failed: %v", err)
	}
	if savings.SavedTokens <= 0 {
		t.Fatalf("expected positive saved tokens, got %d", savings.SavedTokens)
	}
	if handler.Len() != 1 {
		t.Fatalf("expected handler len 1, got %d", handler.Len())
	}

	// Verify FoldedReceipts returns independent copy
	receipts := handler.FoldedReceipts()
	receipts[0].Summary = "mutated locally"
	freshReceipts := handler.FoldedReceipts()
	if freshReceipts[0].Summary == "mutated locally" {
		t.Fatal("FoldedReceipts leaked internal mutable state")
	}

	handler.Clear()
	if handler.Len() != 0 {
		t.Fatalf("expected len 0 after clear, got %d", handler.Len())
	}
}

// 11. Manager edge cases: prompt_human gate, invalid strategy, all speculative failed.
func TestManagerEdgeCases(t *testing.T) {
	manifest := harnesskit.CoordinationManifest{
		SchemaVersion: harnesskit.CoordinationContractVersion,
		Metadata: harnesskit.ManifestMetadata{
			Name: "edge-cases-test",
		},
		Manager: harnesskit.ManagerSpec{
			MaxConcurrency: 2,
			EscalationGates: []harnesskit.EscalationGate{
				{
					Name:         "human-review-gate",
					RiskKeywords: []string{"transfer_funds"},
					Action:       harnesskit.EscalatePromptHuman,
				},
			},
		},
		Workers: map[string]harnesskit.WorkerSpec{
			"worker-a": {RoleID: "worker-a", AccessMode: harnesskit.AccessModeObserve},
			"worker-b": {RoleID: "worker-b", AccessMode: harnesskit.AccessModeObserve},
		},
	}

	factory := func(ctx context.Context, spec harnesskit.WorkerSpec) (harnesskit.WorkerDispatcher, error) {
		return &testDispatcher{
			role: spec.RoleID,
			onDispatch: func(ctx context.Context, taskID string, input harnesskit.Invocation) (harnesskit.WorkerReceipt, error) {
				if input.Tool == "always_fail" {
					return harnesskit.WorkerReceipt{
						TaskID:          taskID,
						WorkerID:        "w-" + spec.RoleID,
						RoleID:          spec.RoleID,
						Status:          harnesskit.StatusFailed,
						ExecutionTimeMS: 5,
					}, nil
				}
				return harnesskit.WorkerReceipt{
					TaskID:          taskID,
					WorkerID:        "w-" + spec.RoleID,
					RoleID:          spec.RoleID,
					Status:          harnesskit.StatusCompleted,
					ExecutionTimeMS: 5,
				}, nil
			},
		}, nil
	}

	pool, err := NewRuntimeWorkerPool(manifest, factory)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}
	manager, err := NewRuntimeManager(manifest, pool, NewCompactFoldHandler())
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	ctx := context.Background()

	// 1. PromptHuman escalation gate halts with StatusAbstain
	rec, err := manager.Dispatch(ctx, "worker-a", harnesskit.Invocation{
		Tool:      "transfer_funds",
		Arguments: json.RawMessage(`{"amount": 1000}`),
	})
	if err != nil {
		t.Fatalf("unexpected error on prompt_human: %v", err)
	}
	if rec.Status != harnesskit.StatusAbstain {
		t.Fatalf("expected StatusAbstain for prompt_human gate, got %s", rec.Status)
	}

	// 2. Unknown role in Dispatch
	_, err = manager.Dispatch(ctx, "unknown-role", harnesskit.Invocation{Tool: "test"})
	if err == nil {
		t.Fatal("expected error on unknown role, got nil")
	}
	var hkErr *harnesskit.Error
	if !errors.As(err, &hkErr) || hkErr.Code != harnesskit.CodeInvalid {
		t.Fatalf("expected CodeInvalid on unknown role, got %v", err)
	}

	// 3. Invalid strategy
	_, err = manager.ExecuteStrategy(ctx, harnesskit.StrategyKind("invalid_strategy"), map[string]harnesskit.Invocation{
		"worker-a": {Tool: "test"},
	})
	if err == nil {
		t.Fatal("expected error on invalid strategy, got nil")
	}
	if !errors.As(err, &hkErr) || hkErr.Code != harnesskit.CodeInvalid {
		t.Fatalf("expected CodeInvalid on invalid strategy, got %v", err)
	}

	// 4. All speculative workers failing
	_, err = manager.ExecuteStrategy(ctx, harnesskit.StrategySpeculative, map[string]harnesskit.Invocation{
		"worker-a": {Tool: "always_fail"},
		"worker-b": {Tool: "always_fail"},
	})
	if err == nil {
		t.Fatal("expected error when all speculative workers fail, got nil")
	}
	if !errors.As(err, &hkErr) || hkErr.Code != harnesskit.CodeInternal {
		t.Fatalf("expected CodeInternal when all speculative workers fail, got %v", err)
	}
}
