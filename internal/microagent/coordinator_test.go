package microagent_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestCoordinator10TaskBatchExecutionTokenReduction(t *testing.T) {
	coord := microagent.NewCoordinator(microagent.CoordinatorConfig{
		TokenCap:           32768,
		QuarantineFailures: true,
	})

	const numTasks = 10
	for i := 1; i <= numTasks; i++ {
		taskID := fmt.Sprintf("TASK-%03d", i)
		err := coord.RegisterTask(microagent.CoordinatorTask{
			ID:             taskID,
			Deliverable:    fmt.Sprintf("Implement bounded feature %d with unit test", i),
			TargetFiles:    []string{fmt.Sprintf("internal/pkg%d/feature.go", i), fmt.Sprintf("internal/pkg%d/feature_test.go", i)},
			WitnessCommand: fmt.Sprintf("go test -v -race ./internal/pkg%d -run TestFeature", i),
			Subsystem:      "feature",
		})
		if err != nil {
			t.Fatalf("RegisterTask(%s): %v", taskID, err)
		}
	}

	// Simulate 10 worker subagents completing their tasks.
	// Each worker generates a realistic, verbose raw transcript (compiler logs, test runs, tool outputs)
	// of ~10,000 characters (~2,500 tokens).
	for i := 1; i <= numTasks; i++ {
		taskID := fmt.Sprintf("TASK-%03d", i)
		rawTranscript := fmt.Sprintf(
			"=== RUN   TestFeature%d\n"+
				"%s\n"+
				"--- PASS: TestFeature%d (0.45s)\n"+
				"PASS\n"+
				"ok  	github.com/anthony-chaudhary/fak/internal/pkg%d	0.512s\n"+
				"Compiler build log: emitting object files, linking binaries, checking symbols...\n"+
				"%s\n",
			i,
			strings.Repeat(fmt.Sprintf("step %d: executing tool read_file, inspecting AST, modifying lines;\n", i), 80),
			i,
			i,
			strings.Repeat("debug: symbols resolved; no escape to heap detected;\n", 60),
		)

		receipt := microagent.WorkerReceipt{
			TaskID:          taskID,
			Status:          microagent.StatusCompleted,
			TouchedFiles:    []string{fmt.Sprintf("internal/pkg%d/feature.go", i), fmt.Sprintf("internal/pkg%d/feature_test.go", i)},
			WitnessCommand:  fmt.Sprintf("go test -v -race ./internal/pkg%d -run TestFeature", i),
			WitnessExitCode: 0,
			GitSHA:          fmt.Sprintf("c0ffee%02d", i),
			Summary:         fmt.Sprintf("Implemented feature %d with complete test coverage", i),
			TokensUsed:      2400,
		}

		if err := coord.IngestReceipt(receipt, rawTranscript); err != nil {
			t.Fatalf("IngestReceipt(%s): %v", taskID, err)
		}
	}

	savings := coord.ContextSavings()
	if len(coord.CompletedTasks()) != numTasks {
		t.Fatalf("completed tasks = %d, want %d", len(coord.CompletedTasks()), numTasks)
	}

	t.Logf("10-Task Batch Savings: RawTokens=%d, ReceiptTokensInContext=%d, SavedTokens=%d, Reduction=%.2f%%",
		savings.RawTokens, savings.ReceiptTokens, savings.SavedTokens, savings.ReductionRatio*100)

	if savings.ReductionRatio < 0.90 {
		t.Fatalf("coordinator context reduction ratio = %.2f%%, want >= 90.00%%", savings.ReductionRatio*100)
	}

	if coord.Context().Tokens() > int(float64(savings.RawTokens)*0.10) {
		t.Fatalf("coordinator active context (%d tokens) exceeded 10%% of raw transcript tokens (%d)",
			coord.Context().Tokens(), savings.RawTokens)
	}
}

func TestCoordinatorStructuredAbstainOnHighRiskBoundaries(t *testing.T) {
	coord := microagent.NewCoordinator(microagent.CoordinatorConfig{
		TokenCap: 16384,
	})

	highRiskCases := []struct {
		taskID       string
		deliverable  string
		targetFiles  []string
		witnessCmd   string
		subsystem    string
		expectedRisk microagent.RiskCategory
		rationale    string
	}{
		{
			taskID:       "TASK-RISK-01",
			deliverable:  "Modify invariant constant in frozen ABI",
			targetFiles:  []string{"internal/abi/agent_invariants.go"},
			witnessCmd:   "go test ./internal/abi -run TestInvariants",
			subsystem:    "abi",
			expectedRisk: microagent.RiskFrozenABI,
			rationale:    "Refusing modification to frozen ABI internal/abi/agent_invariants.go; requires human operator signoff",
		},
		{
			taskID:       "TASK-RISK-02",
			deliverable:  "Refactor mutex lock ordering and concurrency invariants",
			targetFiles:  []string{"internal/microagent/lock_order.go"},
			witnessCmd:   "go test ./internal/microagent -run TestLockOrder",
			subsystem:    "concurrency",
			expectedRisk: microagent.RiskConcurrencyLocks,
			rationale:    "Refusing modification of complex concurrency lock invariants; risk of deadlock",
		},
		{
			taskID:       "TASK-RISK-03",
			deliverable:  "Optimize CUDA GEMM dispatch kernel",
			targetFiles:  []string{"internal/engine/cuda_kernel.cu"},
			witnessCmd:   "go test ./internal/engine -run TestCUDA",
			subsystem:    "gpu-kernel",
			expectedRisk: microagent.RiskLowLevelKernels,
			rationale:    "Low-level GPU CUDA kernel mechanics exceed small worker envelope",
		},
		{
			taskID:       "TASK-RISK-04",
			deliverable:  "Relax capability floor security policy gate",
			targetFiles:  []string{"internal/policy/gate.go"},
			witnessCmd:   "go test ./internal/policy -run TestGate",
			subsystem:    "security-gate",
			expectedRisk: microagent.RiskSecurityPolicy,
			rationale:    "Refusing unvetted modification to security policy capability floor",
		},
		{
			taskID:       "TASK-RISK-05",
			deliverable:  "Protocol migration for cross-subsystem wire format",
			targetFiles:  []string{"internal/wire/migration.go"},
			witnessCmd:   "go test ./internal/wire -run TestMigration",
			subsystem:    "protocol",
			expectedRisk: microagent.RiskProtocolMigration,
			rationale:    "Breaking wire format protocol migration requires architectural steward approval",
		},
	}

	for _, tc := range highRiskCases {
		err := coord.RegisterTask(microagent.CoordinatorTask{
			ID:             tc.taskID,
			Deliverable:    tc.deliverable,
			TargetFiles:    tc.targetFiles,
			WitnessCommand: tc.witnessCmd,
			Subsystem:      tc.subsystem,
		})
		if err != nil {
			t.Fatalf("RegisterTask(%s): %v", tc.taskID, err)
		}

		receipt := microagent.WorkerReceipt{
			TaskID:           tc.taskID,
			Status:           microagent.StatusAbstain,
			TouchedFiles:     tc.targetFiles,
			WitnessCommand:   tc.witnessCmd,
			WitnessExitCode:  0,
			AbstainRationale: tc.rationale,
			Summary:          "Abstained from high-risk boundary",
			TokensUsed:       350,
		}

		rawOutput := fmt.Sprintf("Worker evaluated %s: detected boundary %s. Refusing to speculate.", tc.taskID, tc.expectedRisk)
		if err := coord.IngestReceipt(receipt, rawOutput); err != nil {
			t.Fatalf("IngestReceipt(%s): %v", tc.taskID, err)
		}
	}

	escalations := coord.Escalations()
	if len(escalations) != len(highRiskCases) {
		t.Fatalf("escalations count = %d, want %d", len(escalations), len(highRiskCases))
	}

	for i, tc := range highRiskCases {
		esc := escalations[i]
		if esc.TaskID != tc.taskID {
			t.Errorf("escalation[%d].TaskID = %q, want %q", i, esc.TaskID, tc.taskID)
		}
		if esc.Risk != tc.expectedRisk {
			t.Errorf("escalation[%d].Risk = %q, want %q", i, esc.Risk, tc.expectedRisk)
		}
		if esc.EscalateTo != "higher_capability_model" {
			t.Errorf("escalation[%d].EscalateTo = %q, want 'higher_capability_model'", i, esc.EscalateTo)
		}
		if esc.Rationale != tc.rationale {
			t.Errorf("escalation[%d].Rationale = %q, want %q", i, esc.Rationale, tc.rationale)
		}
	}

	if len(coord.AbstainedTasks()) != len(highRiskCases) {
		t.Fatalf("abstained tasks count = %d, want %d", len(coord.AbstainedTasks()), len(highRiskCases))
	}
}

func TestCoordinatorZeroContextPollutionFromFailedSubtasks(t *testing.T) {
	coord := microagent.NewCoordinator(microagent.CoordinatorConfig{
		TokenCap:           16384,
		QuarantineFailures: true,
	})

	// Task 1: Success
	task1 := microagent.CoordinatorTask{
		ID:             "TASK-CLEAN-01",
		Deliverable:    "Deliver valid parser enhancement",
		TargetFiles:    []string{"internal/parser/ast.go"},
		WitnessCommand: "go test ./internal/parser -run TestAST",
	}
	if err := coord.RegisterTask(task1); err != nil {
		t.Fatalf("RegisterTask(task1): %v", err)
	}

	// Task 2: Failure with catastrophic crash / panic dump
	task2 := microagent.CoordinatorTask{
		ID:             "TASK-CLEAN-02",
		Deliverable:    "Attempted dangerous change causing compilation crash",
		TargetFiles:    []string{"internal/parser/bad.go"},
		WitnessCommand: "go test ./internal/parser -run TestBad",
	}
	if err := coord.RegisterTask(task2); err != nil {
		t.Fatalf("RegisterTask(task2): %v", err)
	}

	// Task 3: Success
	task3 := microagent.CoordinatorTask{
		ID:             "TASK-CLEAN-03",
		Deliverable:    "Deliver tokenizer fix",
		TargetFiles:    []string{"internal/parser/token.go"},
		WitnessCommand: "go test ./internal/parser -run TestToken",
	}
	if err := coord.RegisterTask(task3); err != nil {
		t.Fatalf("RegisterTask(task3): %v", err)
	}

	// 1. Ingest Task 1 (Completed)
	r1 := microagent.WorkerReceipt{
		TaskID:          task1.ID,
		Status:          microagent.StatusCompleted,
		TouchedFiles:    task1.TargetFiles,
		WitnessCommand:  task1.WitnessCommand,
		WitnessExitCode: 0,
		GitSHA:          "sha001",
		Summary:         "Parsed AST nodes cleanly",
		TokensUsed:      1200,
	}
	if err := coord.IngestReceipt(r1, "=== RUN TestAST\nPASS\nok internal/parser 0.1s"); err != nil {
		t.Fatalf("IngestReceipt(r1): %v", err)
	}

	// 2. Ingest Task 2 (Failed with 50,000 characters of panic/compiler stack trace)
	catastrophicPanic := fmt.Sprintf(
		"panic: runtime error: invalid memory address or nil pointer dereference\n"+
			"[signal SIGSEGV: segmentation violation code=0x2 addr=0x0 pc=0x102a3b4c]\n"+
			"goroutine 42 [running]:\n"+
			"%s\n"+
			"FAIL	github.com/anthony-chaudhary/fak/internal/parser [build failed]\n",
		strings.Repeat("internal/parser.MutateState(0x0, 0x1029)\n\t/fak/internal/parser/bad.go:99 +0x12a\n", 400),
	)

	r2 := microagent.WorkerReceipt{
		TaskID:          task2.ID,
		Status:          microagent.StatusFailed,
		TouchedFiles:    task2.TargetFiles,
		WitnessCommand:  task2.WitnessCommand,
		WitnessExitCode: 2,
		Summary:         "Build failed with SIGSEGV",
		TokensUsed:      3500,
	}
	if err := coord.IngestReceipt(r2, catastrophicPanic); err != nil {
		t.Fatalf("IngestReceipt(r2): %v", err)
	}

	// 3. Ingest Task 3 (Completed)
	r3 := microagent.WorkerReceipt{
		TaskID:          task3.ID,
		Status:          microagent.StatusCompleted,
		TouchedFiles:    task3.TargetFiles,
		WitnessCommand:  task3.WitnessCommand,
		WitnessExitCode: 0,
		GitSHA:          "sha003",
		Summary:         "Fixed token boundaries",
		TokensUsed:      1100,
	}
	if err := coord.IngestReceipt(r3, "=== RUN TestToken\nPASS\nok internal/parser 0.1s"); err != nil {
		t.Fatalf("IngestReceipt(r3): %v", err)
	}

	// Assertions on failure isolation & zero context pollution
	if len(coord.CompletedTasks()) != 2 {
		t.Fatalf("completed tasks = %d, want 2", len(coord.CompletedTasks()))
	}
	if len(coord.FailedTasks()) != 1 {
		t.Fatalf("failed tasks = %d, want 1", len(coord.FailedTasks()))
	}

	// Zero pollution check 1: Ensure no message in context contains the raw panic/traceback text
	for idx, msg := range coord.Context().Messages() {
		if strings.Contains(msg.Content, "panic: runtime error") {
			t.Fatalf("context polluted at msg[%d]: contains raw panic text", idx)
		}
		if strings.Contains(msg.Content, "SIGSEGV") {
			t.Fatalf("context polluted at msg[%d]: contains SIGSEGV compiler dump", idx)
		}
		if strings.Contains(msg.Content, "goroutine 42") {
			t.Fatalf("context polluted at msg[%d]: contains goroutine stack trace", idx)
		}
	}

	// Zero pollution check 2: With QuarantineFailures=true, failed task contributes exactly 0 messages
	// to the active coordinator context. Only Task 1 and Task 3 are present.
	msgs := coord.Context().Messages()
	if len(msgs) != 2 {
		t.Fatalf("coordinator message count = %d, want exactly 2 (Task 1 and Task 3)", len(msgs))
	}
	if strings.Contains(msgs[0].Content, task2.ID) || strings.Contains(msgs[1].Content, task2.ID) {
		t.Fatalf("coordinator context contains failed task ID %q despite quarantine", task2.ID)
	}
	if !strings.Contains(msgs[0].Content, task1.ID) || !strings.Contains(msgs[1].Content, task3.ID) {
		t.Fatalf("coordinator context missing expected successful tasks: msgs=%+v", msgs)
	}
}

func TestCoordinatorAtomicS0S1ConstraintsValidation(t *testing.T) {
	coord := microagent.NewCoordinator(microagent.CoordinatorConfig{})

	// 1. Rejects 0 target files
	err := coord.RegisterTask(microagent.CoordinatorTask{
		ID:             "TASK-ZERO-FILES",
		Deliverable:    "Empty file list",
		TargetFiles:    []string{},
		WitnessCommand: "go test ./...",
	})
	if err == nil || !strings.Contains(err.Error(), "1 to 3 files") {
		t.Fatalf("expected error for 0 target files, got: %v", err)
	}

	// 2. Rejects >3 target files (atomic bound violation)
	err = coord.RegisterTask(microagent.CoordinatorTask{
		ID:             "TASK-FOUR-FILES",
		Deliverable:    "Too many files",
		TargetFiles:    []string{"a.go", "b.go", "c.go", "d.go"},
		WitnessCommand: "go test ./...",
	})
	if err == nil || !strings.Contains(err.Error(), "1 to 3 files") {
		t.Fatalf("expected error for 4 target files, got: %v", err)
	}

	// 3. Rejects empty witness command
	err = coord.RegisterTask(microagent.CoordinatorTask{
		ID:             "TASK-NO-WITNESS",
		Deliverable:    "No witness command",
		TargetFiles:    []string{"a.go"},
		WitnessCommand: "",
	})
	if err == nil || !strings.Contains(err.Error(), "require exactly one witness command") {
		t.Fatalf("expected error for empty witness command, got: %v", err)
	}

	// 4. Rejects chained witness commands (&&, ;)
	err = coord.RegisterTask(microagent.CoordinatorTask{
		ID:             "TASK-CHAINED-WITNESS-1",
		Deliverable:    "Chained witness command",
		TargetFiles:    []string{"a.go"},
		WitnessCommand: "go test ./... && go vet ./...",
	})
	if err == nil || !strings.Contains(err.Error(), "not a chained pipeline") {
		t.Fatalf("expected error for chained && witness command, got: %v", err)
	}

	err = coord.RegisterTask(microagent.CoordinatorTask{
		ID:             "TASK-CHAINED-WITNESS-2",
		Deliverable:    "Chained witness command",
		TargetFiles:    []string{"a.go"},
		WitnessCommand: "go test ./...; make ci",
	})
	if err == nil || !strings.Contains(err.Error(), "not a chained pipeline") {
		t.Fatalf("expected error for chained ; witness command, got: %v", err)
	}

	// 5. Valid task registration
	validTask := microagent.CoordinatorTask{
		ID:             "TASK-VALID",
		Deliverable:    "Valid atomic deliverable",
		TargetFiles:    []string{"a.go", "a_test.go"},
		WitnessCommand: "go test -v ./a -run TestA",
	}
	if err := coord.RegisterTask(validTask); err != nil {
		t.Fatalf("RegisterTask(validTask) failed: %v", err)
	}

	// 6. Duplicate task ID rejected
	if err := coord.RegisterTask(validTask); err == nil {
		t.Fatal("expected error on duplicate task ID, got nil")
	}

	// 7. Receipt validation: completed with non-zero exit code rejected
	badReceipt := microagent.WorkerReceipt{
		TaskID:          validTask.ID,
		Status:          microagent.StatusCompleted,
		TouchedFiles:    validTask.TargetFiles,
		WitnessCommand:  validTask.WitnessCommand,
		WitnessExitCode: 1, // Non-zero exit code with COMPLETED status
		Summary:         "Invalid completion",
	}
	if err := coord.IngestReceipt(badReceipt); err == nil {
		t.Fatal("expected error for COMPLETED status with non-zero exit code, got nil")
	}

	// 8. Receipt validation: abstain without rationale rejected
	abstainNoRationale := microagent.WorkerReceipt{
		TaskID:          validTask.ID,
		Status:          microagent.StatusAbstain,
		TouchedFiles:    validTask.TargetFiles,
		WitnessCommand:  validTask.WitnessCommand,
		WitnessExitCode: 0,
		Summary:         "Abstained without rationale",
	}
	if err := coord.IngestReceipt(abstainNoRationale); err == nil {
		t.Fatal("expected error for ABSTAIN status without rationale, got nil")
	}
}
