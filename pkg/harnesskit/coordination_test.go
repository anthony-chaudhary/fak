//go:build wip_coordination

package harnesskit

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

func TestWorkerSpecSerializationAndRoundtrip(t *testing.T) {
	spec := WorkerSpec{
		RoleID:              "general",
		Purpose:             "Autonomous code implementation worker",
		InstructionSnapshot: "You are an autonomous code implementation worker with scoped mutation rights.",
		InstructionTemplate: "Execute task: {{.TaskID}}",
		AccessMode:          AccessModeEffect,
		ToolScope: ToolScope{
			AllowedTools:  []string{"read_file", "write_file", "edit_file", "bash"},
			DeniedTools:   []string{"git_push", "rm_rf"},
			MaxMutations:  25,
			AllowNetwork:  false,
			AllowWorktree: true,
		},
		Budget: WorkerBudget{
			MaxTurns:        30,
			MaxInputTokens:  128000,
			MaxOutputTokens: 16000,
			Timeout:         10 * time.Minute,
		},
		Witness: WitnessRequirements{
			RequireIndependentWitness: true,
			WitnessCommand:            "go test ./...",
			RequireZeroExitCode:       true,
			WitnessTimeout:            90 * time.Second,
			VerifyArtifactIntegrity:   true,
		},
		Placement: ModelPlacement{
			Provider:       "anthropic",
			Model:          "claude-3-7-sonnet",
			Effort:         "high",
			ThinkingBudget: 8000,
			Temperature:    0.2,
		},
		Metadata: map[string]string{
			"lane": "implementation",
			"tier": "primary",
		},
	}

	if err := spec.Validate(); err != nil {
		t.Fatalf("valid spec failed validation: %v", err)
	}

	data, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("failed to marshal spec: %v", err)
	}

	var roundTrip WorkerSpec
	if err := json.Unmarshal(data, &roundTrip); err != nil {
		t.Fatalf("failed to unmarshal spec: %v", err)
	}

	if err := roundTrip.Validate(); err != nil {
		t.Fatalf("unmarshaled spec failed validation: %v", err)
	}

	if !reflect.DeepEqual(spec, roundTrip) {
		t.Fatalf("roundtrip mismatch:\ngot:  %+v\nwant: %+v", roundTrip, spec)
	}
}

func sampleValidManifest() CoordinationManifest {
	return CoordinationManifest{
		SchemaVersion: CoordinationContractVersion,
		Metadata: ManifestMetadata{
			Name:        "code-factory",
			Version:     "1.0.0",
			Description: "Autonomous multi-role coordination manifest",
			Labels: map[string]string{
				"environment": "production",
			},
		},
		Manager: ManagerSpec{
			MaxConcurrency:  4,
			DefaultStrategy: StrategyFanOutFanIn,
			ReceiptPolicy: ReceiptPolicy{
				StrictReceipt:      true,
				QuarantineFailures: true,
				RequireWitnessPass: true,
				MaxFoldTokens:      500,
			},
			EscalationGates: []EscalationGate{
				{
					Name:         "git-push-gate",
					PathPatterns: []string{".git/*"},
					RiskKeywords: []string{"push", "remote"},
					Action:       EscalatePromptHuman,
				},
				{
					Name:         "security-review-gate",
					PathPatterns: []string{"internal/crypto/*"},
					RiskKeywords: []string{"secret", "private_key"},
					Action:       EscalateRerouteRole,
					TargetRole:   "reviewer",
				},
			},
			TokenCap: 1000000,
		},
		Workers: map[string]WorkerSpec{
			"explorer": {
				RoleID:     "explorer",
				Purpose:    "Codebase exploration and discovery",
				AccessMode: AccessModeObserve,
				ToolScope: ToolScope{
					AllowedTools:  []string{"read_file", "glob", "grep"},
					DeniedTools:   []string{"bash", "write_file"},
					MaxMutations:  0,
					AllowNetwork:  false,
					AllowWorktree: false,
				},
				Budget: WorkerBudget{
					MaxTurns: 15,
				},
			},
			"general": {
				RoleID:     "general",
				Purpose:    "Code implementation and refactoring",
				AccessMode: AccessModeEffect,
				ToolScope: ToolScope{
					AllowedTools:  []string{"read_file", "write_file", "edit_file", "bash"},
					MaxMutations:  50,
					AllowNetwork:  false,
					AllowWorktree: true,
				},
				Budget: WorkerBudget{
					MaxTurns: 30,
				},
				Witness: WitnessRequirements{
					RequireIndependentWitness: true,
					WitnessCommand:            "go test ./...",
					RequireZeroExitCode:       true,
				},
			},
			"reviewer": {
				RoleID:     "reviewer",
				Purpose:    "Quality audit and diff verification",
				AccessMode: AccessModeObserve,
				ToolScope: ToolScope{
					AllowedTools:  []string{"read_file", "git_diff", "git_status"},
					MaxMutations:  0,
					AllowNetwork:  false,
					AllowWorktree: false,
				},
				Budget: WorkerBudget{
					MaxTurns: 10,
				},
			},
		},
		Topology: []TopologyRule{
			{
				FromRole: "explorer",
				ToRole:   "general",
				Required: true,
			},
			{
				FromRole:  "general",
				ToRole:    "reviewer",
				Condition: "witness_passed",
				Required:  true,
			},
		},
	}
}

func TestCoordinationManifestSerializationAndRoundtrip(t *testing.T) {
	manifest := sampleValidManifest()
	if err := manifest.Validate(); err != nil {
		t.Fatalf("sample manifest validation failed: %v", err)
	}

	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal manifest: %v", err)
	}

	parsed, err := ParseCoordinationManifest(data)
	if err != nil {
		t.Fatalf("ParseCoordinationManifest failed: %v", err)
	}

	if !reflect.DeepEqual(manifest, parsed) {
		t.Fatalf("parsed manifest mismatch:\ngot:  %+v\nwant: %+v", parsed, manifest)
	}
}

func TestWorkerReceiptSerializationAndRoundtrip(t *testing.T) {
	receipt := WorkerReceipt{
		TaskID:       "task-build-123",
		WorkerID:     "worker-general-42",
		RoleID:       "general",
		Status:       StatusCompleted,
		Summary:      "Successfully modified coordination primitives and verified all package tests pass.",
		TouchedFiles: []string{"pkg/harnesskit/coordination.go", "pkg/harnesskit/coordination_test.go"},
		GitOID:       "e5f6a7b8c9d0123456789abcdef0123456789abc",
		Diff: DiffSummary{
			FilesChanged: 2,
			Insertions:   220,
			Deletions:    15,
		},
		Witness: WitnessResult{
			Command:      "go test ./pkg/harnesskit/...",
			ExitCode:     0,
			Duration:     350 * time.Millisecond,
			OutputDigest: "sha256:d8e8fca2dc0f896fd7cb4cb0031ba249",
			Passed:       true,
		},
		Artifacts: map[string]string{
			"coverage_profile": "/tmp/cov.out",
		},
		Tokens: TokenBreakdown{
			InputTokens:  18500,
			OutputTokens: 2100,
			TotalTokens:  20600,
		},
		Diagnosis: &FailureDiagnosis{
			ReasonCategory: "none",
			Message:        "All witness assertions succeeded without regressions",
		},
		ExecutionTimeMS: 4500,
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("valid receipt failed validation: %v", err)
	}

	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}

	var roundTrip WorkerReceipt
	if err := json.Unmarshal(data, &roundTrip); err != nil {
		t.Fatalf("failed to unmarshal receipt: %v", err)
	}

	if err := roundTrip.Validate(); err != nil {
		t.Fatalf("unmarshaled receipt failed validation: %v", err)
	}

	if !reflect.DeepEqual(receipt, roundTrip) {
		t.Fatalf("receipt roundtrip mismatch:\ngot:  %+v\nwant: %+v", roundTrip, receipt)
	}
}

func TestParseCoordinationManifestStrictUnknownFields(t *testing.T) {
	manifest := sampleValidManifest()
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}

	// 1. Unknown field at top level
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal map failed: %v", err)
	}
	raw["unexpected_top_level"] = "forbidden"
	corrupted, _ := json.Marshal(raw)
	if _, err := ParseCoordinationManifest(corrupted); err == nil {
		t.Fatal("expected error on unknown top-level field, got nil")
	}

	// 2. Unknown field inside manager
	delete(raw, "unexpected_top_level")
	mgrRaw, ok := raw["manager"].(map[string]any)
	if !ok {
		t.Fatal("manager is not map")
	}
	mgrRaw["unknown_manager_property"] = 999
	corrupted, _ = json.Marshal(raw)
	if _, err := ParseCoordinationManifest(corrupted); err == nil {
		t.Fatal("expected error on unknown manager field, got nil")
	}

	// 3. Unknown field inside worker tool scope
	delete(mgrRaw, "unknown_manager_property")
	workersRaw, ok := raw["workers"].(map[string]any)
	if !ok {
		t.Fatal("workers is not map")
	}
	explorerRaw, ok := workersRaw["explorer"].(map[string]any)
	if !ok {
		t.Fatal("explorer worker is not map")
	}
	toolScopeRaw, ok := explorerRaw["tool_scope"].(map[string]any)
	if !ok {
		t.Fatal("tool_scope is not map")
	}
	toolScopeRaw["nonexistent_tool_scope_option"] = true
	corrupted, _ = json.Marshal(raw)
	if _, err := ParseCoordinationManifest(corrupted); err == nil {
		t.Fatal("expected error on unknown tool_scope field, got nil")
	}

	// 4. Unexpected trailing JSON content
	validBytes, _ := json.Marshal(manifest)
	withTrailing := append(validBytes, []byte(` {"extra": 1}`)...)
	if _, err := ParseCoordinationManifest(withTrailing); err == nil {
		t.Fatal("expected error on trailing JSON content, got nil")
	}
}

func TestValidationLogic(t *testing.T) {
	t.Run("missing role_id", func(t *testing.T) {
		w := WorkerSpec{
			RoleID:     "",
			AccessMode: AccessModeObserve,
		}
		if err := w.Validate(); err == nil {
			t.Fatal("expected error for empty role_id, got nil")
		}
	})

	t.Run("invalid access_mode", func(t *testing.T) {
		w := WorkerSpec{
			RoleID:     "worker-1",
			AccessMode: AccessMode("unrestricted"),
		}
		if err := w.Validate(); err == nil {
			t.Fatal("expected error for invalid access_mode, got nil")
		}
	})

	t.Run("negative budget turns", func(t *testing.T) {
		w := WorkerSpec{
			RoleID:     "worker-1",
			AccessMode: AccessModeEffect,
			Budget: WorkerBudget{
				MaxTurns: -5,
			},
		}
		if err := w.Validate(); err == nil {
			t.Fatal("expected error for negative max_turns, got nil")
		}
	})

	t.Run("negative budget tokens", func(t *testing.T) {
		w := WorkerSpec{
			RoleID:     "worker-1",
			AccessMode: AccessModeEffect,
			Budget: WorkerBudget{
				MaxInputTokens: -100,
			},
		}
		if err := w.Validate(); err == nil {
			t.Fatal("expected error for negative max_input_tokens, got nil")
		}
	})

	t.Run("manager spec concurrency defaults and validation", func(t *testing.T) {
		m := ManagerSpec{
			MaxConcurrency: 0,
		}
		m.ApplyDefaults()
		if m.MaxConcurrency != DefaultMaxConcurrency {
			t.Fatalf("expected concurrency default %d, got %d", DefaultMaxConcurrency, m.MaxConcurrency)
		}
		if m.DefaultStrategy != StrategyFanOutFanIn {
			t.Fatalf("expected strategy default %s, got %s", StrategyFanOutFanIn, m.DefaultStrategy)
		}

		mNeg := ManagerSpec{
			MaxConcurrency: -2,
		}
		if err := mNeg.Validate(); err == nil {
			t.Fatal("expected error for negative concurrency, got nil")
		}
	})

	t.Run("receipt validation", func(t *testing.T) {
		r := WorkerReceipt{
			TaskID:          "",
			WorkerID:        "w1",
			RoleID:          "general",
			Status:          StatusCompleted,
			ExecutionTimeMS: 100,
		}
		if err := r.Validate(); err == nil {
			t.Fatal("expected error on empty task_id")
		}

		r.TaskID = "t1"
		r.Status = ReceiptStatus("UNKNOWN_STATUS")
		if err := r.Validate(); err == nil {
			t.Fatal("expected error on invalid status")
		}

		r.Status = StatusCompleted
		r.ExecutionTimeMS = -10
		if err := r.Validate(); err == nil {
			t.Fatal("expected error on negative execution_time_ms")
		}
	})

	t.Run("manifest validation integrity", func(t *testing.T) {
		m := sampleValidManifest()
		m.SchemaVersion = "wrong.version/v99"
		if err := m.Validate(); err == nil {
			t.Fatal("expected error on unsupported schema_version")
		}

		m = sampleValidManifest()
		m.Metadata.Name = ""
		if err := m.Validate(); err == nil {
			t.Fatal("expected error on empty metadata.name")
		}

		m = sampleValidManifest()
		m.Workers = nil
		if err := m.Validate(); err == nil {
			t.Fatal("expected error on empty workers map")
		}

		m = sampleValidManifest()
		badWorker := m.Workers["general"]
		badWorker.RoleID = "mismatched"
		m.Workers["general"] = badWorker
		if err := m.Validate(); err == nil {
			t.Fatal("expected error when map key doesn't match role_id")
		}

		m = sampleValidManifest()
		m.Topology = append(m.Topology, TopologyRule{
			FromRole: "nonexistent",
			ToRole:   "general",
		})
		if err := m.Validate(); err == nil {
			t.Fatal("expected error on topology referencing unknown role")
		}

		m = sampleValidManifest()
		m.Manager.EscalationGates = append(m.Manager.EscalationGates, EscalationGate{
			Name:       "bad-gate",
			Action:     EscalateRerouteRole,
			TargetRole: "unknown_target",
		})
		if err := m.Validate(); err == nil {
			t.Fatal("expected error on escalation gate referencing unknown target role")
		}
	})
}

func TestAccessModeSemantics(t *testing.T) {
	// 1. Observe mode helper methods
	if !AccessModeObserve.IsValid() {
		t.Fatal("AccessModeObserve should be valid")
	}
	if AccessModeObserve.AllowsMutations() {
		t.Fatal("AccessModeObserve must not allow mutations")
	}
	if AccessModeObserve.AllowsWorktree() {
		t.Fatal("AccessModeObserve must not allow worktree modifications")
	}

	// 2. Observe mode rejects AllowWorktree: true
	wObserveWorktree := WorkerSpec{
		RoleID:     "observer",
		AccessMode: AccessModeObserve,
		ToolScope: ToolScope{
			AllowWorktree: true,
		},
	}
	if err := wObserveWorktree.Validate(); err == nil {
		t.Fatal("expected error when observe worker has AllowWorktree=true")
	}

	// 3. Observe mode rejects MaxMutations > 0
	wObserveMutations := WorkerSpec{
		RoleID:     "observer",
		AccessMode: AccessModeObserve,
		ToolScope: ToolScope{
			MaxMutations: 5,
		},
	}
	if err := wObserveMutations.Validate(); err == nil {
		t.Fatal("expected error when observe worker has MaxMutations > 0")
	}

	// 4. Observe mode rejects MutabilityMutating
	wObserveMutability := WorkerSpec{
		RoleID:     "observer",
		AccessMode: AccessModeObserve,
		ToolScope: ToolScope{
			Mutability: MutabilityMutating,
		},
	}
	if err := wObserveMutability.Validate(); err == nil {
		t.Fatal("expected error when observe worker has MutabilityMutating")
	}

	// 5. Observe mode clean validation
	wObserveClean := WorkerSpec{
		RoleID:     "observer",
		AccessMode: AccessModeObserve,
		ToolScope: ToolScope{
			AllowedTools:  []string{"read_file"},
			MaxMutations:  0,
			AllowWorktree: false,
		},
	}
	if err := wObserveClean.Validate(); err != nil {
		t.Fatalf("clean observe worker failed validation: %v", err)
	}

	// 6. Effect mode helper methods and permissions
	if !AccessModeEffect.IsValid() {
		t.Fatal("AccessModeEffect should be valid")
	}
	if !AccessModeEffect.AllowsMutations() {
		t.Fatal("AccessModeEffect must allow mutations")
	}
	if !AccessModeEffect.AllowsWorktree() {
		t.Fatal("AccessModeEffect must allow worktree")
	}

	wEffect := WorkerSpec{
		RoleID:     "effector",
		AccessMode: AccessModeEffect,
		ToolScope: ToolScope{
			AllowedTools:  []string{"write_file"},
			MaxMutations:  10,
			AllowWorktree: true,
		},
	}
	if err := wEffect.Validate(); err != nil {
		t.Fatalf("valid effect worker failed validation: %v", err)
	}
}

func TestReceiptSavingsAndCompressionRatio(t *testing.T) {
	// Standard high compression scenario: 50,000 raw turns -> 500 token receipt fold
	savings := CalculateContextSavings(50000, 500)
	if savings.RawTranscriptTokens != 50000 {
		t.Fatalf("expected 50000 raw tokens, got %d", savings.RawTranscriptTokens)
	}
	if savings.ReceiptFoldTokens != 500 {
		t.Fatalf("expected 500 fold tokens, got %d", savings.ReceiptFoldTokens)
	}
	if savings.SavedTokens != 49500 {
		t.Fatalf("expected 49500 saved tokens, got %d", savings.SavedTokens)
	}
	if savings.CompressionRatio != 100.0 {
		t.Fatalf("expected compression ratio 100.0, got %f", savings.CompressionRatio)
	}
	if pct := savings.NetSavingsPercentage(); pct != 99.0 {
		t.Fatalf("expected 99.0%% net savings, got %f%%", pct)
	}

	// Edge case: zero fold tokens
	zeroFold := CalculateContextSavings(1000, 0)
	if zeroFold.SavedTokens != 1000 || zeroFold.CompressionRatio != 1000.0 {
		t.Fatalf("unexpected zero-fold calculation: %+v", zeroFold)
	}

	// Edge case: zero raw tokens
	zeroRaw := CalculateContextSavings(0, 500)
	if zeroRaw.SavedTokens != 0 || zeroRaw.NetSavingsPercentage() != 0.0 {
		t.Fatalf("unexpected zero-raw calculation: %+v", zeroRaw)
	}

	// Edge case: fold exceeds raw (anomaly guard)
	negativeGuard := CalculateContextSavings(100, 300)
	if negativeGuard.SavedTokens != 0 {
		t.Fatalf("saved tokens should not be negative: %d", negativeGuard.SavedTokens)
	}

	// Method invocation via WorkerReceipt
	receipt := WorkerReceipt{
		TaskID:   "t1",
		WorkerID: "w1",
		RoleID:   "general",
		Status:   StatusCompleted,
		Tokens: TokenBreakdown{
			InputTokens:  18000,
			OutputTokens: 2000,
			TotalTokens:  20000,
		},
	}
	receiptSavings := receipt.Savings(200000)
	if receiptSavings.RawTranscriptTokens != 200000 || receiptSavings.ReceiptFoldTokens != 20000 {
		t.Fatalf("unexpected receipt savings tokens: %+v", receiptSavings)
	}
	if receiptSavings.SavedTokens != 180000 {
		t.Fatalf("expected 180000 saved tokens, got %d", receiptSavings.SavedTokens)
	}
	if receiptSavings.CompressionRatio != 10.0 {
		t.Fatalf("expected compression ratio 10.0, got %f", receiptSavings.CompressionRatio)
	}
}

func TestPublicCoordinationContract(t *testing.T) {
	contract := PublicCoordinationContract()

	if contract.SchemaVersion != CoordinationContractVersion {
		t.Fatalf("expected schema version %s, got %s", CoordinationContractVersion, contract.SchemaVersion)
	}

	expectedStrategies := []string{
		string(StrategyFanOutFanIn),
		string(StrategySequential),
		string(StrategyAdaptiveDAG),
		string(StrategySpeculative),
	}
	for _, s := range expectedStrategies {
		found := false
		for _, cs := range contract.Strategies {
			if cs == s {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing strategy %s in contract", s)
		}
	}

	expectedModes := []string{string(AccessModeObserve), string(AccessModeEffect)}
	for _, m := range expectedModes {
		found := false
		for _, cm := range contract.AccessModes {
			if cm == m {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing access mode %s in contract", m)
		}
	}

	expectedStatuses := []string{
		string(StatusCompleted),
		string(StatusFailed),
		string(StatusAbstain),
		string(StatusTimedOut),
	}
	for _, st := range expectedStatuses {
		found := false
		for _, cst := range contract.ReceiptStatuses {
			if cst == st {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("missing receipt status %s in contract", st)
		}
	}

	if contract.Isolation == "" || contract.ReceiptFolding == "" || contract.Security == "" || contract.Cancellation == "" {
		t.Fatalf("incomplete coordination contract invariants: %+v", contract)
	}

	if len(contract.Errors) == 0 {
		t.Fatal("contract errors must not be empty")
	}

	// Also verify PublicContract() includes it identically
	pub := PublicContract()
	if !reflect.DeepEqual(pub.Coordination, contract) {
		t.Fatalf("PublicContract().Coordination mismatch:\ngot:  %+v\nwant: %+v", pub.Coordination, contract)
	}
}

// TestInterfacesCompile verifies mock implementations satisfy declared coordination interfaces.
type mockManager struct {
	manifest CoordinationManifest
}

func (m *mockManager) Manifest() CoordinationManifest { return m.manifest }
func (m *mockManager) Dispatch(ctx context.Context, roleID string, input Invocation) (WorkerReceipt, error) {
	return WorkerReceipt{
		TaskID:   "task-mock",
		WorkerID: "worker-mock",
		RoleID:   roleID,
		Status:   StatusCompleted,
	}, nil
}
func (m *mockManager) ExecuteStrategy(ctx context.Context, strategy StrategyKind, inputs map[string]Invocation) (map[string]WorkerReceipt, error) {
	res := make(map[string]WorkerReceipt)
	for k := range inputs {
		res[k] = WorkerReceipt{TaskID: k, RoleID: k, Status: StatusCompleted}
	}
	return res, nil
}

type mockWorkerPool struct{}

func (p *mockWorkerPool) Acquire(ctx context.Context, roleID string) (WorkerDispatcher, error) {
	return &mockDispatcher{role: roleID}, nil
}
func (p *mockWorkerPool) Release(ctx context.Context, roleID string, worker WorkerDispatcher) error {
	return nil
}
func (p *mockWorkerPool) Available(roleID string) int { return 5 }
func (p *mockWorkerPool) Capacity(roleID string) int  { return 10 }

type mockDispatcher struct{ role string }

func (d *mockDispatcher) Role() string { return d.role }
func (d *mockDispatcher) Dispatch(ctx context.Context, taskID string, input Invocation) (WorkerReceipt, error) {
	return WorkerReceipt{
		TaskID:   taskID,
		WorkerID: "w-mock",
		RoleID:   d.role,
		Status:   StatusCompleted,
	}, nil
}
func (d *mockDispatcher) Cancel(taskID string) error { return nil }

type mockFoldHandler struct{}

func (f *mockFoldHandler) Fold(ctx context.Context, receipt WorkerReceipt) (ContextSavings, error) {
	return receipt.Savings(10000), nil
}

func TestInterfacesCompile(t *testing.T) {
	var _ Manager = (*mockManager)(nil)
	var _ WorkerPool = (*mockWorkerPool)(nil)
	var _ WorkerDispatcher = (*mockDispatcher)(nil)
	var _ ReceiptFoldHandler = (*mockFoldHandler)(nil)
}

func TestStrategyAndStatusHelpers(t *testing.T) {
	for _, strat := range []StrategyKind{StrategyFanOutFanIn, StrategySequential, StrategyAdaptiveDAG, StrategySpeculative} {
		if !strat.IsValid() {
			t.Fatalf("strategy %s should be valid", strat)
		}
	}
	if StrategyKind("random_strategy").IsValid() {
		t.Fatal("random strategy should not be valid")
	}

	for _, st := range []ReceiptStatus{StatusCompleted, StatusFailed, StatusAbstain, StatusTimedOut} {
		if !st.IsValid() {
			t.Fatalf("receipt status %s should be valid", st)
		}
		if !st.IsTerminal() {
			t.Fatalf("receipt status %s should be terminal", st)
		}
	}
	if !StatusCompleted.IsSuccess() {
		t.Fatal("StatusCompleted should be success")
	}
	if StatusFailed.IsSuccess() || StatusAbstain.IsSuccess() || StatusTimedOut.IsSuccess() {
		t.Fatal("non-completed status must not report success")
	}

	for _, act := range []RiskEscalationAction{EscalateAbstain, EscalatePromptHuman, EscalateRerouteRole} {
		if !act.IsValid() {
			t.Fatalf("escalation action %s should be valid", act)
		}
	}
	if RiskEscalationAction("ignore").IsValid() {
		t.Fatal("unrecognized escalation action should not be valid")
	}
}
