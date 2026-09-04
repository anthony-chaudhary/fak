package agentopt

import (
	"sync"
	"testing"
)

func TestEffortTierBasics(t *testing.T) {
	tiers := []EffortTier{EffortNone, EffortLow, EffortMedium, EffortHigh}

	// Verify valid tiers and string representation
	for _, tier := range tiers {
		if !tier.IsValid() {
			t.Fatalf("tier %s should be valid", tier)
		}
		if tier.String() != string(tier) {
			t.Fatalf("tier.String() = %q, want %q", tier.String(), string(tier))
		}
		if tier.ProviderEffort() != string(tier) {
			t.Fatalf("tier.ProviderEffort() = %q, want %q", tier.ProviderEffort(), string(tier))
		}
	}

	invalidTier := EffortTier("super-high")
	if invalidTier.IsValid() {
		t.Fatalf("invalid tier %s should not be valid", invalidTier)
	}

	// Verify ThinkingBudget mappings
	expectedBudgets := map[EffortTier]int{
		EffortNone:   0,
		EffortLow:    256,
		EffortMedium: 1024,
		EffortHigh:   2048,
	}
	for tier, expected := range expectedBudgets {
		if got := tier.ThinkingBudget(); got != expected {
			t.Fatalf("ThinkingBudget(%s) = %d, want %d", tier, got, expected)
		}
	}
	if got := invalidTier.ThinkingBudget(); got != 0 {
		t.Fatalf("ThinkingBudget(invalid) = %d, want 0", got)
	}

	// Verify provider representations
	providers := []string{"anthropic", "openai", "gemini", "generic"}
	for _, p := range providers {
		for _, tier := range tiers {
			rep := tier.ProviderRepresentation(p)
			if rep != string(tier) {
				t.Fatalf("ProviderRepresentation(%s, %s) = %q, want %q", tier, p, rep, string(tier))
			}
		}
	}

	// Verify ParseEffortTier
	parseCases := []struct {
		input    string
		expected EffortTier
		hasErr   bool
	}{
		{"none", EffortNone, false},
		{"0", EffortNone, false},
		{"off", EffortNone, false},
		{"low", EffortLow, false},
		{"medium", EffortMedium, false},
		{"med", EffortMedium, false},
		{"high", EffortHigh, false},
		{"HIGH", EffortHigh, false},
		{"  Medium  ", EffortMedium, false},
		{"unknown", EffortNone, true},
		{"", EffortNone, true},
	}
	for _, tc := range parseCases {
		res, err := ParseEffortTier(tc.input)
		if tc.hasErr && err == nil {
			t.Fatalf("ParseEffortTier(%q) expected error, got nil", tc.input)
		}
		if !tc.hasErr && err != nil {
			t.Fatalf("ParseEffortTier(%q) unexpected error: %v", tc.input, err)
		}
		if !tc.hasErr && res != tc.expected {
			t.Fatalf("ParseEffortTier(%q) = %s, want %s", tc.input, res, tc.expected)
		}
	}
}

func TestOperationalCategoryBasics(t *testing.T) {
	cats := []OperationalCategory{
		CategoryPlanAndDecompose,
		CategoryRoutineToolInvocation,
		CategoryDiagnosticAndVerification,
		CategoryErrorRecovery,
	}
	for _, c := range cats {
		if c.String() != string(c) {
			t.Fatalf("category.String() = %q, want %q", c.String(), string(c))
		}
	}
}

func TestSyntheticTurnTrajectories(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	t.Run("Trajectory_PlanningPrompt", func(t *testing.T) {
		tc := TurnContext{
			TurnIndex: 0,
			Prompt:    "Plan and decompose the authentication system into 3 phases: schema, jwt, and middleware",
		}
		class := router.Classify(tc)
		if class.Effort != EffortHigh {
			t.Fatalf("planning prompt effort = %s, want %s", class.Effort, EffortHigh)
		}
		if class.Category != CategoryPlanAndDecompose {
			t.Fatalf("planning prompt category = %s, want %s", class.Category, CategoryPlanAndDecompose)
		}
		if class.ThinkingBudget != 2048 {
			t.Fatalf("planning prompt thinking budget = %d, want 2048", class.ThinkingBudget)
		}
		if router.ClassifyTurn(tc) != EffortHigh {
			t.Fatalf("ClassifyTurn mismatch: got %s, want %s", router.ClassifyTurn(tc), EffortHigh)
		}
	})

	t.Run("Trajectory_RoutineFileRead", func(t *testing.T) {
		tc := TurnContext{
			TurnIndex: 1,
			ToolName:  "read",
			ToolArgs:  map[string]any{"filePath": "internal/agentopt/intra_model_router.go"},
			ToolOutput: `package agentopt
// package contents line 1
// package contents line 2`,
		}
		class := router.Classify(tc)
		if class.Effort != EffortNone {
			t.Fatalf("routine file read effort = %s, want %s (reason: %s)", class.Effort, EffortNone, class.Reason)
		}
		if class.Category != CategoryRoutineToolInvocation {
			t.Fatalf("routine file read category = %s, want %s", class.Category, CategoryRoutineToolInvocation)
		}
		if class.ThinkingBudget != 0 {
			t.Fatalf("routine file read thinking budget = %d, want 0", class.ThinkingBudget)
		}
		if router.ClassifyTurn(tc) != EffortNone {
			t.Fatalf("ClassifyTurn mismatch: got %s, want %s", router.ClassifyTurn(tc), EffortNone)
		}
	})

	t.Run("Trajectory_TestFailureTrace", func(t *testing.T) {
		tc := TurnContext{
			TurnIndex: 2,
			ToolName:  "bash",
			ToolArgs:  map[string]any{"command": "go test ./internal/agentopt"},
			ToolOutput: `=== RUN   TestAuth
--- FAIL: TestAuth (0.01s)
    auth_test.go:42: expected token but got nil
FAIL
FAIL	github.com/anthony-chaudhary/fak/internal/agentopt	0.035s`,
		}
		class := router.Classify(tc)
		if class.Effort != EffortHigh {
			t.Fatalf("test failure trace effort = %s, want %s (reason: %s)", class.Effort, EffortHigh, class.Reason)
		}
		if class.Category != CategoryErrorRecovery {
			t.Fatalf("test failure trace category = %s, want %s", class.Category, CategoryErrorRecovery)
		}
		if class.ThinkingBudget != 2048 {
			t.Fatalf("test failure trace thinking budget = %d, want 2048", class.ThinkingBudget)
		}
		if router.ClassifyTurn(tc) != EffortHigh {
			t.Fatalf("ClassifyTurn mismatch: got %s, want %s", router.ClassifyTurn(tc), EffortHigh)
		}
	})

	t.Run("Trajectory_SynthesisPrompt", func(t *testing.T) {
		tc := TurnContext{
			TurnIndex: 3,
			Prompt:    "Synthesize the findings from the investigation and summarize the root cause of the memory spike",
		}
		class := router.Classify(tc)
		if class.Effort != EffortMedium {
			t.Fatalf("synthesis prompt effort = %s, want %s (reason: %s)", class.Effort, EffortMedium, class.Reason)
		}
		if class.Category != CategoryDiagnosticAndVerification {
			t.Fatalf("synthesis prompt category = %s, want %s", class.Category, CategoryDiagnosticAndVerification)
		}
		if class.ThinkingBudget != 1024 {
			t.Fatalf("synthesis prompt thinking budget = %d, want 1024", class.ThinkingBudget)
		}
		if router.ClassifyTurn(tc) != EffortMedium {
			t.Fatalf("ClassifyTurn mismatch: got %s, want %s", router.ClassifyTurn(tc), EffortMedium)
		}
	})
}

func TestEdgeCases(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	t.Run("EmptyInput", func(t *testing.T) {
		tc := TurnContext{}
		class := router.Classify(tc)
		if class.Effort != EffortNone {
			t.Fatalf("empty input effort = %s, want %s", class.Effort, EffortNone)
		}
		if class.Category != CategoryRoutineToolInvocation {
			t.Fatalf("empty input category = %s, want %s", class.Category, CategoryRoutineToolInvocation)
		}
		if class.ThinkingBudget != 0 {
			t.Fatalf("empty input budget = %d, want 0", class.ThinkingBudget)
		}
		if class.Reason != "empty turn context defaults to no effort" {
			t.Fatalf("unexpected reason: %s", class.Reason)
		}
	})

	t.Run("MixedToolLogs_RoutineAndFailure", func(t *testing.T) {
		// Output contains routine logs, a passed test, but also a failing test.
		tc := TurnContext{
			ToolName: "bash",
			ToolOutput: `[INFO] Read 120 lines from configuration file
=== RUN   TestInitialSetup
--- PASS: TestInitialSetup (0.00s)
=== RUN   TestTokenValidation
--- FAIL: TestTokenValidation (0.02s)
    token_test.go:88: signature mismatch
FAIL
exit status 1`,
		}
		class := router.Classify(tc)
		if class.Effort != EffortHigh {
			t.Fatalf("mixed log with failure effort = %s, want %s (reason: %s)", class.Effort, EffortHigh, class.Reason)
		}
		if class.Category != CategoryErrorRecovery {
			t.Fatalf("mixed log with failure category = %s, want %s", class.Category, CategoryErrorRecovery)
		}
	})

	t.Run("StackTrace_GoPanic", func(t *testing.T) {
		tc := TurnContext{
			ToolName: "bash",
			ToolOutput: `panic: runtime error: index out of range [5] with length 3

goroutine 1 [running]:
main.processTokens(0x1400010c000, 0x3, 0x3)
	/workspace/cmd/fak/main.go:42 +0x68
main.main()
	/workspace/cmd/fak/main.go:12 +0x24`,
		}
		class := router.Classify(tc)
		if class.Effort != EffortHigh {
			t.Fatalf("go panic stack trace effort = %s, want %s (reason: %s)", class.Effort, EffortHigh, class.Reason)
		}
		if class.Category != CategoryErrorRecovery {
			t.Fatalf("go panic category = %s, want %s", class.Category, CategoryErrorRecovery)
		}
	})

	t.Run("StackTrace_PythonTraceback", func(t *testing.T) {
		tc := TurnContext{
			ToolName: "bash",
			ToolOutput: `Traceback (most recent call last):
  File "train.py", line 45, in <module>
    optimizer.step()
  File "torch/optim/adam.py", line 120, in step
    RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB`,
		}
		class := router.Classify(tc)
		if class.Effort != EffortHigh {
			t.Fatalf("python traceback effort = %s, want %s (reason: %s)", class.Effort, EffortHigh, class.Reason)
		}
		if class.Category != CategoryErrorRecovery {
			t.Fatalf("python traceback category = %s, want %s", class.Category, CategoryErrorRecovery)
		}
	})

	t.Run("ExplicitErrorFlags", func(t *testing.T) {
		// HasError flag
		tc1 := TurnContext{HasError: true, ToolName: "read", ToolOutput: "content"}
		if router.ClassifyTurn(tc1) != EffortHigh {
			t.Fatalf("HasError=true should be EffortHigh")
		}

		// ExitCode != 0
		tc2 := TurnContext{ExitCode: 1, ToolOutput: "process exited with code 1"}
		if router.ClassifyTurn(tc2) != EffortHigh {
			t.Fatalf("ExitCode=1 should be EffortHigh")
		}

		// ToolError populated
		tc3 := TurnContext{ToolError: "fatal: remote error: repository not found"}
		if router.ClassifyTurn(tc3) != EffortHigh {
			t.Fatalf("ToolError should be EffortHigh")
		}

		// RecentErrors present
		tc4 := TurnContext{RecentErrors: []string{"dial tcp 127.0.0.1:8080: connect: connection refused"}}
		if router.ClassifyTurn(tc4) != EffortHigh {
			t.Fatalf("RecentErrors should be EffortHigh")
		}
	})
}

func TestDiagnosticAndVerification(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	t.Run("TestPass_Go", func(t *testing.T) {
		tc := TurnContext{
			ToolName: "bash",
			ToolOutput: `=== RUN   TestIntraModelEffortRouter
--- PASS: TestIntraModelEffortRouter (0.02s)
PASS
ok  	github.com/anthony-chaudhary/fak/internal/agentopt	0.035s`,
		}
		class := router.Classify(tc)
		if class.Effort != EffortLow {
			t.Fatalf("test pass effort = %s, want %s (reason: %s)", class.Effort, EffortLow, class.Reason)
		}
		if class.Category != CategoryDiagnosticAndVerification {
			t.Fatalf("test pass category = %s, want %s", class.Category, CategoryDiagnosticAndVerification)
		}
		if class.ThinkingBudget != 256 {
			t.Fatalf("test pass budget = %d, want 256", class.ThinkingBudget)
		}
	})

	t.Run("LinterCheck_Tool", func(t *testing.T) {
		lintTools := []string{"golangci-lint", "ruff", "eslint", "flake8", "clippy", "govet"}
		for _, tool := range lintTools {
			tc := TurnContext{
				ToolName:   tool,
				ToolOutput: "0 issues reported",
			}
			class := router.Classify(tc)
			if class.Effort != EffortLow {
				t.Fatalf("linter tool %s effort = %s, want %s", tool, class.Effort, EffortLow)
			}
			if class.Category != CategoryDiagnosticAndVerification {
				t.Fatalf("linter tool %s category = %s, want %s", tool, class.Category, CategoryDiagnosticAndVerification)
			}
		}
	})

	t.Run("LinterCheck_Shell", func(t *testing.T) {
		tc := TurnContext{
			ToolName:   "bash",
			ToolArgs:   map[string]any{"command": "golangci-lint run ./..."},
			ToolOutput: "Checked 42 packages, 0 errors",
		}
		class := router.Classify(tc)
		if class.Effort != EffortLow {
			t.Fatalf("linter in shell effort = %s, want %s (reason: %s)", class.Effort, EffortLow, class.Reason)
		}
		if class.Category != CategoryDiagnosticAndVerification {
			t.Fatalf("linter in shell category = %s, want %s", class.Category, CategoryDiagnosticAndVerification)
		}
	})

	t.Run("DiffInspection_Tool", func(t *testing.T) {
		tc := TurnContext{
			ToolName: "git_diff",
			ToolOutput: `diff --git a/internal/agentopt/intra_model_router.go b/internal/agentopt/intra_model_router.go
index 1234567..89abcdef 100644
--- a/internal/agentopt/intra_model_router.go
+++ b/internal/agentopt/intra_model_router.go
@@ -10,3 +10,4 @@
+func NewIntraModelEffortRouter()`,
		}
		class := router.Classify(tc)
		if class.Effort != EffortMedium {
			t.Fatalf("diff tool effort = %s, want %s (reason: %s)", class.Effort, EffortMedium, class.Reason)
		}
		if class.Category != CategoryDiagnosticAndVerification {
			t.Fatalf("diff tool category = %s, want %s", class.Category, CategoryDiagnosticAndVerification)
		}
		if class.ThinkingBudget != 1024 {
			t.Fatalf("diff tool budget = %d, want 1024", class.ThinkingBudget)
		}
	})

	t.Run("DiffInspection_Prompt", func(t *testing.T) {
		tc := TurnContext{
			Prompt: "Review diff before staging files to confirm no unintended whitespace changes",
		}
		class := router.Classify(tc)
		if class.Effort != EffortMedium {
			t.Fatalf("review diff prompt effort = %s, want %s", class.Effort, EffortMedium)
		}
		if class.Category != CategoryDiagnosticAndVerification {
			t.Fatalf("review diff prompt category = %s, want %s", class.Category, CategoryDiagnosticAndVerification)
		}
	})

	t.Run("BenchmarkDiagnostics", func(t *testing.T) {
		tc := TurnContext{
			ToolOutput: "BenchmarkClassifier-8   5000000   240 ns/op   48 B/op   2 allocs/op",
		}
		class := router.Classify(tc)
		if class.Effort != EffortMedium {
			t.Fatalf("benchmark diagnostic effort = %s, want %s", class.Effort, EffortMedium)
		}
		if class.Category != CategoryDiagnosticAndVerification {
			t.Fatalf("benchmark diagnostic category = %s, want %s", class.Category, CategoryDiagnosticAndVerification)
		}
	})
}

func TestRoutineToolInvocations(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	tools := []string{
		"read", "read_file", "cat", "head", "tail",
		"glob", "find", "list_files",
		"grep", "search", "rg", "ripgrep",
		"listdir", "list_dir", "ls",
		"format", "fmt", "gofmt", "prettier", "black", "ruff_format",
		"stat", "file_info", "wc", "pwd", "which",
	}

	for _, tool := range tools {
		tc := TurnContext{
			ToolName: tool,
			ToolArgs: map[string]any{"path": "foo/bar.txt"},
		}
		class := router.Classify(tc)
		if class.Effort != EffortNone {
			t.Fatalf("routine tool %s classified as %s, want %s", tool, class.Effort, EffortNone)
		}
		if class.Category != CategoryRoutineToolInvocation {
			t.Fatalf("routine tool %s category = %s, want %s", tool, class.Category, CategoryRoutineToolInvocation)
		}
		if class.ThinkingBudget != 0 {
			t.Fatalf("routine tool %s budget = %d, want 0", tool, class.ThinkingBudget)
		}
	}

	t.Run("RoutineBashCommands", func(t *testing.T) {
		commands := []string{
			"ls -la",
			"cat internal/agentopt/intra_model_router.go",
			"head -n 20 main.go",
			"tail -f log.txt",
			"grep -rn 'func' .",
			"rg 'EffortTier' internal/",
			"gofmt -w .",
			"git status",
			"git diff --stat",
		}
		for _, cmd := range commands {
			tc := TurnContext{
				ToolName: "bash",
				ToolArgs: map[string]any{"command": cmd},
			}
			class := router.Classify(tc)
			if class.Effort != EffortNone {
				t.Fatalf("routine bash command %q classified as %s, want %s (reason: %s)", cmd, class.Effort, EffortNone, class.Reason)
			}
			if class.Category != CategoryRoutineToolInvocation {
				t.Fatalf("routine bash command %q category = %s, want %s", cmd, class.Category, CategoryRoutineToolInvocation)
			}
		}
	})

	t.Run("BatchToolCalls", func(t *testing.T) {
		tc := TurnContext{
			ToolCalls: []ToolCall{
				{Name: "read", Args: map[string]any{"filePath": "a.go"}},
				{Name: "glob", Args: map[string]any{"pattern": "*.go"}},
				{Name: "grep", Args: map[string]any{"pattern": "type"}},
			},
		}
		class := router.Classify(tc)
		if class.Effort != EffortNone {
			t.Fatalf("batch routine tool calls effort = %s, want %s", class.Effort, EffortNone)
		}
		if class.Category != CategoryRoutineToolInvocation {
			t.Fatalf("batch routine tool calls category = %s, want %s", class.Category, CategoryRoutineToolInvocation)
		}
	})

	t.Run("RoutineInspectionPrompt", func(t *testing.T) {
		prompts := []string{
			"read internal/agentopt/intra_model_router.go",
			"cat /etc/hosts",
			"glob internal/**/*.go",
			"grep for EffortTier in internal/agentopt",
			"list files in directory",
		}
		for _, p := range prompts {
			tc := TurnContext{Prompt: p}
			class := router.Classify(tc)
			if class.Effort != EffortNone {
				t.Fatalf("routine prompt %q classified as %s, want %s", p, class.Effort, EffortNone)
			}
			if class.Category != CategoryRoutineToolInvocation {
				t.Fatalf("routine prompt %q category = %s, want %s", p, class.Category, CategoryRoutineToolInvocation)
			}
		}
	})
}

func TestPlanAndDecompose(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	t.Run("IsReplanFlag", func(t *testing.T) {
		tc := TurnContext{
			IsReplan: true,
			Prompt:   "Our first attempt failed due to cyclic imports, please replan the component division",
		}
		class := router.Classify(tc)
		if class.Effort != EffortHigh {
			t.Fatalf("IsReplan effort = %s, want %s", class.Effort, EffortHigh)
		}
		if class.Category != CategoryPlanAndDecompose {
			t.Fatalf("IsReplan category = %s, want %s", class.Category, CategoryPlanAndDecompose)
		}
	})

	t.Run("IsInitialFlag", func(t *testing.T) {
		tc := TurnContext{
			IsInitial: true,
			Prompt:    "Scaffold the new microservice",
		}
		class := router.Classify(tc)
		if class.Effort != EffortHigh {
			t.Fatalf("IsInitial effort = %s, want %s", class.Effort, EffortHigh)
		}
		if class.Category != CategoryPlanAndDecompose {
			t.Fatalf("IsInitial category = %s, want %s", class.Category, CategoryPlanAndDecompose)
		}
	})

	t.Run("TurnIndexZeroWithPrompt", func(t *testing.T) {
		tc := TurnContext{
			TurnIndex: 0,
			Prompt:    "Implement Issue #11186: feat(agentopt): IntraModelEffortRouter",
		}
		class := router.Classify(tc)
		if class.Effort != EffortHigh {
			t.Fatalf("TurnIndex 0 effort = %s, want %s", class.Effort, EffortHigh)
		}
		if class.Category != CategoryPlanAndDecompose {
			t.Fatalf("TurnIndex 0 category = %s, want %s", class.Category, CategoryPlanAndDecompose)
		}
	})

	t.Run("HighLevelQueries", func(t *testing.T) {
		queries := []string{
			"Design the system architecture for our multi-region data replication",
			"Propose a plan to migrate internal/agentopt to Go 1.26",
			"Break down this feature into sequential implementation steps",
			"How should we approach optimizing the cache invalidation algorithm?",
			"Formulate a plan for zero-downtime deployment",
		}
		for _, q := range queries {
			tc := TurnContext{TurnIndex: 5, Prompt: q}
			class := router.Classify(tc)
			if class.Effort != EffortHigh {
				t.Fatalf("query %q classified as %s, want %s", q, class.Effort, EffortHigh)
			}
			if class.Category != CategoryPlanAndDecompose {
				t.Fatalf("query %q category = %s, want %s", q, class.Category, CategoryPlanAndDecompose)
			}
		}
	})
}

func TestErrorRecoveryPatterns(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	cases := []struct {
		name   string
		output string
	}{
		{"GoCompilerSyntaxError", "main.go:14:2: syntax error: non-declaration statement outside function body"},
		{"GoCompilerUndefined", "main.go:25:12: undefined: IntraModelEffortRouter"},
		{"GoCannotUseType", "cannot use val (variable of type string) as int value in struct literal"},
		{"CannotFindPackage", "cannot find package \"github.com/anthony-chaudhary/missing\" in any of:"},
		{"BuildFailed", "make: *** [build] Error 2\nbuild failed"},
		{"PolicyBlock", "DENY (POLICY_BLOCK): tool refund_payment refused by structure"},
		{"PermissionDenied", "open /etc/shadow: permission denied"},
		{"AccessDenied", "Access denied for user 'agent'@'localhost'"},
		{"UnwitnessedClaim", "guard refusal: CLAIM_UNWITNESSED: diff did not witness claim in commit message"},
		{"InvariantViolation", "invariant violation: tree was not disjoint with concurrent exclusive lease"},
		{"DivergenceDetected", "divergence detected between committed trunk and local worktree"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tc := TurnContext{ToolOutput: c.output}
			class := router.Classify(tc)
			if class.Effort != EffortHigh {
				t.Fatalf("%s effort = %s, want %s (reason: %s)", c.name, class.Effort, EffortHigh, class.Reason)
			}
			if class.Category != CategoryErrorRecovery {
				t.Fatalf("%s category = %s, want %s", c.name, class.Category, CategoryErrorRecovery)
			}
		})
	}
}

func TestCustomConfigurationAndOptions(t *testing.T) {
	router := NewIntraModelEffortRouter(
		WithCustomBudget(EffortHigh, 4096),
		WithCustomBudget(EffortLow, 128),
		WithToolCategory("custom_scanner", CategoryDiagnosticAndVerification),
		WithDefaultEffort(EffortMedium),
	)

	if got := router.ThinkingBudget(EffortHigh); got != 4096 {
		t.Fatalf("custom EffortHigh budget = %d, want 4096", got)
	}
	if got := router.ThinkingBudget(EffortLow); got != 128 {
		t.Fatalf("custom EffortLow budget = %d, want 128", got)
	}
	if got := router.ThinkingBudget(EffortMedium); got != 1024 {
		t.Fatalf("default EffortMedium budget = %d, want 1024", got)
	}

	tcCustomTool := TurnContext{ToolName: "custom_scanner"}
	class := router.Classify(tcCustomTool)
	if class.Category != CategoryDiagnosticAndVerification {
		t.Fatalf("custom tool category = %s, want %s", class.Category, CategoryDiagnosticAndVerification)
	}
	if class.Effort != EffortMedium {
		t.Fatalf("custom tool effort = %s, want %s", class.Effort, EffortMedium)
	}

	// Dynamic registration
	router.RegisterBudget(EffortNone, 10)
	if got := router.ThinkingBudget(EffortNone); got != 10 {
		t.Fatalf("dynamically registered EffortNone budget = %d, want 10", got)
	}

	router.RegisterToolCategory("custom_panic_watcher", CategoryErrorRecovery)
	tcWatcher := TurnContext{ToolName: "custom_panic_watcher"}
	if router.ClassifyTurn(tcWatcher) != EffortHigh {
		t.Fatalf("dynamically registered tool should classify as EffortHigh")
	}

	// Helper method
	if b := router.ThinkingBudgetForTurn(tcWatcher); b != 4096 {
		t.Fatalf("ThinkingBudgetForTurn = %d, want 4096", b)
	}
}

func TestDeterminismAndConcurrency(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	testContexts := []TurnContext{
		{TurnIndex: 0, Prompt: "Plan and decompose the new architecture"},
		{ToolName: "read", ToolOutput: "package main\nfunc main() {}"},
		{ToolName: "bash", ToolOutput: "--- FAIL: TestFoo (0.01s)\nFAIL"},
		{Prompt: "Synthesize findings from diagnostic run"},
		{ToolName: "bash", ToolOutput: "--- PASS: TestFoo (0.01s)\nPASS\nok foo 0.01s"},
		{},
	}

	expectedTiers := []EffortTier{
		EffortHigh,
		EffortNone,
		EffortHigh,
		EffortMedium,
		EffortLow,
		EffortNone,
	}

	// 100% Determinism over 100 sequential passes
	for iter := 0; iter < 100; iter++ {
		for i, tc := range testContexts {
			got := router.ClassifyTurn(tc)
			if got != expectedTiers[i] {
				t.Fatalf("iter %d, case %d: got %s, want %s", iter, i, got, expectedTiers[i])
			}
		}
	}

	// Concurrency test with race detector
	var wg sync.WaitGroup
	for worker := 0; worker < 20; worker++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for iter := 0; iter < 50; iter++ {
				for i, tc := range testContexts {
					got := router.ClassifyTurn(tc)
					if got != expectedTiers[i] {
						t.Errorf("worker %d: got %s, want %s", w, got, expectedTiers[i])
					}
				}
			}
		}(worker)
	}
	wg.Wait()
}

func TestPackageLevelHelpers(t *testing.T) {
	tc := TurnContext{TurnIndex: 0, Prompt: "Decompose task"}
	if ClassifyTurn(tc) != EffortHigh {
		t.Fatalf("ClassifyTurn() = %s, want %s", ClassifyTurn(tc), EffortHigh)
	}

	class := Classify(tc)
	if class.Effort != EffortHigh || class.Category != CategoryPlanAndDecompose {
		t.Fatalf("Classify() = %+v, want EffortHigh, CategoryPlanAndDecompose", class)
	}
}

func TestIntraModelEffortRouter_TiersAndMultipliers(t *testing.T) {
	tiers := []EffortTier{EffortNone, EffortLow, EffortMedium, EffortHigh}

	// 1. IsValid, String, ProviderEffort, ProviderRepresentation
	for _, tier := range tiers {
		if !tier.IsValid() {
			t.Errorf("tier %s should be valid", tier)
		}
		if tier.String() != string(tier) {
			t.Errorf("tier.String() = %q, want %q", tier.String(), string(tier))
		}
		if tier.ProviderEffort() != string(tier) {
			t.Errorf("tier.ProviderEffort() = %q, want %q", tier.ProviderEffort(), string(tier))
		}
		for _, provider := range []string{"anthropic", "openai", "gemini", "generic"} {
			if rep := tier.ProviderRepresentation(provider); rep != string(tier) {
				t.Errorf("ProviderRepresentation(%s, %s) = %q, want %q", tier, provider, rep, string(tier))
			}
		}
	}

	invalidTier := EffortTier("bogus")
	if invalidTier.IsValid() {
		t.Errorf("invalid tier %s should not be valid", invalidTier)
	}

	// 2. ThinkingBudget
	expectedBudgets := map[EffortTier]int{
		EffortNone:   0,
		EffortLow:    256,
		EffortMedium: 1024,
		EffortHigh:   2048,
	}
	for tier, want := range expectedBudgets {
		if got := tier.ThinkingBudget(); got != want {
			t.Errorf("ThinkingBudget(%s) = %d, want %d", tier, got, want)
		}
	}
	if got := invalidTier.ThinkingBudget(); got != 0 {
		t.Errorf("ThinkingBudget(invalid) = %d, want 0", got)
	}

	// 3. ParseEffortTier
	parseTests := []struct {
		input string
		want  EffortTier
		err   bool
	}{
		{"none", EffortNone, false},
		{"0", EffortNone, false},
		{"off", EffortNone, false},
		{"low", EffortLow, false},
		{"medium", EffortMedium, false},
		{"med", EffortMedium, false},
		{"high", EffortHigh, false},
		{"HIGH", EffortHigh, false},
		{"  Medium  ", EffortMedium, false},
		{"invalid", EffortNone, true},
		{"", EffortNone, true},
	}
	for _, tc := range parseTests {
		got, err := ParseEffortTier(tc.input)
		if tc.err && err == nil {
			t.Errorf("ParseEffortTier(%q) expected error, got nil", tc.input)
		}
		if !tc.err && err != nil {
			t.Errorf("ParseEffortTier(%q) unexpected error: %v", tc.input, err)
		}
		if !tc.err && got != tc.want {
			t.Errorf("ParseEffortTier(%q) = %s, want %s", tc.input, got, tc.want)
		}
	}

	// 4. VolumeMultiplier: EffortNone: 0.4, EffortLow: 0.7, EffortMedium: 1.0, EffortHigh: 1.6 (default: 1.0)
	expectedMultipliers := map[EffortTier]float64{
		EffortNone:   0.4,
		EffortLow:    0.7,
		EffortMedium: 1.0,
		EffortHigh:   1.6,
	}
	for tier, want := range expectedMultipliers {
		if got := tier.VolumeMultiplier(); got != want {
			t.Errorf("VolumeMultiplier(%s) = %f, want %f", tier, got, want)
		}
	}
	// Verify unknown tier defaults to 1.0 (neutral anchor)
	if got := invalidTier.VolumeMultiplier(); got != 1.0 {
		t.Errorf("VolumeMultiplier(invalid) = %f, want 1.0", got)
	}
	if got := EffortTier("").VolumeMultiplier(); got != 1.0 {
		t.Errorf("VolumeMultiplier(\"\") = %f, want 1.0", got)
	}

	// 5. Verify Decision and IntraModelEffortRouter link to VolumeMultiplier
	router := DefaultIntraModelEffortRouter()
	for tier, want := range expectedMultipliers {
		if got := router.VolumeMultiplier(tier); got != want {
			t.Errorf("router.VolumeMultiplier(%s) = %f, want %f", tier, got, want)
		}
		if got := VolumeMultiplier(tier); got != want {
			t.Errorf("VolumeMultiplier(%s) = %f, want %f", tier, got, want)
		}
	}

	tc := TurnContext{Prompt: "Plan architecture"}
	var d Decision = router.Classify(tc)
	if got := d.VolumeMultiplier(); got != 1.6 {
		t.Errorf("Decision.VolumeMultiplier() = %f, want 1.6", got)
	}
	if got := router.VolumeMultiplierForTurn(tc); got != 1.6 {
		t.Errorf("router.VolumeMultiplierForTurn() = %f, want 1.6", got)
	}
	if got := VolumeMultiplierForTurn(tc); got != 1.6 {
		t.Errorf("VolumeMultiplierForTurn() = %f, want 1.6", got)
	}
}

func TestIntraModelEffortRouter_Classification(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	t.Run("PlanAndDecompose_EffortHigh", func(t *testing.T) {
		cases := []TurnContext{
			{TurnIndex: 0, Prompt: "Plan the implementation of issue #11186 in internal/agentopt"},
			{Prompt: "Decompose this refactoring task into sequential steps"},
			{IsInitial: true, Prompt: "Scaffold new component"},
			{IsPlanning: true, Prompt: "Analyze architecture options"},
			{IsReplan: true, Prompt: "Replan after strategy change"},
		}
		for _, tc := range cases {
			class := router.Classify(tc)
			if class.Category != CategoryPlanAndDecompose {
				t.Errorf("category for prompt %q = %s, want %s", tc.Prompt, class.Category, CategoryPlanAndDecompose)
			}
			if class.Effort != EffortHigh {
				t.Errorf("effort for prompt %q = %s, want %s", tc.Prompt, class.Effort, EffortHigh)
			}
			if class.VolumeMultiplier() != 1.6 {
				t.Errorf("VolumeMultiplier for prompt %q = %f, want 1.6", tc.Prompt, class.VolumeMultiplier())
			}
		}
	})

	t.Run("RoutineTool_EffortNone", func(t *testing.T) {
		cases := []TurnContext{
			{ToolName: "read", ToolArgs: map[string]any{"filePath": "intra_model_router.go"}},
			{ToolName: "glob", ToolArgs: map[string]any{"pattern": "*.go"}},
			{ToolName: "grep", ToolArgs: map[string]any{"pattern": "EffortTier"}},
			{ToolName: "bash", ToolArgs: map[string]any{"command": "ls -la"}},
			{ToolName: "gofmt", ToolArgs: map[string]any{"file": "intra_model_router.go"}},
			{Prompt: "read internal/agentopt/intra_model_router.go"},
		}
		for _, tc := range cases {
			class := router.Classify(tc)
			if class.Category != CategoryRoutineToolInvocation {
				t.Errorf("category = %s, want %s (tc: %+v)", class.Category, CategoryRoutineToolInvocation, tc)
			}
			if class.Effort != EffortNone {
				t.Errorf("effort = %s, want %s (tc: %+v)", class.Effort, EffortNone, tc)
			}
			if class.VolumeMultiplier() != 0.4 {
				t.Errorf("VolumeMultiplier = %f, want 0.4 (tc: %+v)", class.VolumeMultiplier(), tc)
			}
		}
	})

	t.Run("Diagnostic_EffortLowAndMedium", func(t *testing.T) {
		// Low effort: test pass or linter
		lowCases := []TurnContext{
			{ToolName: "bash", ToolOutput: "=== RUN   TestPass\n--- PASS: TestPass (0.01s)\nPASS\nok  internal/agentopt 0.02s"},
			{ToolName: "golangci-lint", ToolOutput: "0 issues found"},
			{ToolName: "govet", ToolOutput: "vet check clean"},
		}
		for _, tc := range lowCases {
			class := router.Classify(tc)
			if class.Category != CategoryDiagnosticAndVerification {
				t.Errorf("category = %s, want %s (tc: %+v)", class.Category, CategoryDiagnosticAndVerification, tc)
			}
			if class.Effort != EffortLow {
				t.Errorf("effort = %s, want %s (tc: %+v)", class.Effort, EffortLow, tc)
			}
			if class.VolumeMultiplier() != 0.7 {
				t.Errorf("VolumeMultiplier = %f, want 0.7 (tc: %+v)", class.VolumeMultiplier(), tc)
			}
		}

		// Medium effort: diff inspection, synthesis, benchmark
		medCases := []TurnContext{
			{ToolName: "git_diff", ToolOutput: "diff --git a/foo.go b/foo.go\n--- a/foo.go\n+++ b/foo.go\n@@ -1,3 +1,4 @@"},
			{Prompt: "Review diff before committing changes"},
			{Prompt: "Synthesize the verification results and summarize findings"},
			{ToolOutput: "BenchmarkRouter-8  5000000  240 ns/op  48 B/op"},
		}
		for _, tc := range medCases {
			class := router.Classify(tc)
			if class.Category != CategoryDiagnosticAndVerification {
				t.Errorf("category = %s, want %s (tc: %+v)", class.Category, CategoryDiagnosticAndVerification, tc)
			}
			if class.Effort != EffortMedium {
				t.Errorf("effort = %s, want %s (tc: %+v)", class.Effort, EffortMedium, tc)
			}
			if class.VolumeMultiplier() != 1.0 {
				t.Errorf("VolumeMultiplier = %f, want 1.0 (tc: %+v)", class.VolumeMultiplier(), tc)
			}
		}
	})

	t.Run("ErrorRecovery_EffortHigh", func(t *testing.T) {
		cases := []TurnContext{
			{ToolName: "bash", ToolOutput: "--- FAIL: TestFoo (0.01s)\nFAIL"},
			{ToolName: "bash", ToolOutput: "main.go:12:3: syntax error: unexpected newline"},
			{ToolName: "bash", ToolOutput: "panic: runtime error: index out of range [0] with length 0"},
			{ToolName: "bash", ToolOutput: "DENY (POLICY_BLOCK): tool refund_payment refused by structure"},
			{HasError: true, ErrorMessage: "unexpected EOF"},
			{ExitCode: 1, ToolOutput: "process failed"},
			{ToolError: "fatal: compilation failed"},
			{RecentErrors: []string{"connection reset by peer"}},
		}
		for _, tc := range cases {
			class := router.Classify(tc)
			if class.Category != CategoryErrorRecovery {
				t.Errorf("category = %s, want %s (tc: %+v)", class.Category, CategoryErrorRecovery, tc)
			}
			if class.Effort != EffortHigh {
				t.Errorf("effort = %s, want %s (tc: %+v)", class.Effort, EffortHigh, tc)
			}
			if class.VolumeMultiplier() != 1.6 {
				t.Errorf("VolumeMultiplier = %f, want 1.6 (tc: %+v)", class.VolumeMultiplier(), tc)
			}
		}
	})
}

func TestIntraModelEffortRouter_BudgetMapping(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	// Direct tier thinking budget mapping
	tierBudgets := []struct {
		tier   EffortTier
		budget int
	}{
		{EffortNone, 0},
		{EffortLow, 256},
		{EffortMedium, 1024},
		{EffortHigh, 2048},
	}
	for _, tb := range tierBudgets {
		if got := tb.tier.ThinkingBudget(); got != tb.budget {
			t.Errorf("tier %s ThinkingBudget = %d, want %d", tb.tier, got, tb.budget)
		}
		if got := router.ThinkingBudget(tb.tier); got != tb.budget {
			t.Errorf("router ThinkingBudget(%s) = %d, want %d", tb.tier, got, tb.budget)
		}
	}

	// Category budget mapping: [2048, 0, 256/1024, 2048]
	// 1. Plan & Decompose -> 2048
	planTurn := TurnContext{TurnIndex: 0, Prompt: "Plan feature implementation"}
	planClass := router.Classify(planTurn)
	if planClass.Category != CategoryPlanAndDecompose || planClass.ThinkingBudget != 2048 {
		t.Errorf("Plan & Decompose budget = %d, want 2048", planClass.ThinkingBudget)
	}
	if planClass.AllocatedBudget != 2048 {
		t.Errorf("Plan & Decompose AllocatedBudget = %d, want 2048", planClass.AllocatedBudget)
	}

	// 2. Routine Tool -> 0
	routineTurn := TurnContext{ToolName: "read", ToolArgs: map[string]any{"path": "file.go"}}
	routineClass := router.Classify(routineTurn)
	if routineClass.Category != CategoryRoutineToolInvocation || routineClass.ThinkingBudget != 0 {
		t.Errorf("Routine Tool budget = %d, want 0", routineClass.ThinkingBudget)
	}
	if routineClass.AllocatedBudget != 0 {
		t.Errorf("Routine Tool AllocatedBudget = %d, want 0", routineClass.AllocatedBudget)
	}

	// 3. Diagnostic & Verification -> 256 (EffortLow) / 1024 (EffortMedium)
	diagLowTurn := TurnContext{ToolName: "govet", ToolOutput: "ok"}
	diagLowClass := router.Classify(diagLowTurn)
	if diagLowClass.Category != CategoryDiagnosticAndVerification || diagLowClass.ThinkingBudget != 256 {
		t.Errorf("Diagnostic (low) budget = %d, want 256", diagLowClass.ThinkingBudget)
	}

	diagMedTurn := TurnContext{ToolName: "git_diff", ToolOutput: "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@"}
	diagMedClass := router.Classify(diagMedTurn)
	if diagMedClass.Category != CategoryDiagnosticAndVerification || diagMedClass.ThinkingBudget != 1024 {
		t.Errorf("Diagnostic (med) budget = %d, want 1024", diagMedClass.ThinkingBudget)
	}

	// 4. Error Recovery -> 2048
	errTurn := TurnContext{ToolName: "bash", ToolOutput: "--- FAIL: TestX\nFAIL"}
	errClass := router.Classify(errTurn)
	if errClass.Category != CategoryErrorRecovery || errClass.ThinkingBudget != 2048 {
		t.Errorf("Error Recovery budget = %d, want 2048", errClass.ThinkingBudget)
	}
	if errClass.AllocatedBudget != 2048 {
		t.Errorf("Error Recovery AllocatedBudget = %d, want 2048", errClass.AllocatedBudget)
	}

	// ThinkingBudgetForTurn helper
	if b := router.ThinkingBudgetForTurn(planTurn); b != 2048 {
		t.Errorf("ThinkingBudgetForTurn(planTurn) = %d, want 2048", b)
	}
	if b := router.ThinkingBudgetForTurn(routineTurn); b != 0 {
		t.Errorf("ThinkingBudgetForTurn(routineTurn) = %d, want 0", b)
	}
}

func TestIntraModelEffortRouter_SyntheticTrajectories(t *testing.T) {
	router := DefaultIntraModelEffortRouter()

	type trajectoryStep struct {
		name       string
		turn       TurnContext
		wantEffort EffortTier
		wantCat    OperationalCategory
		wantBudget int
		wantMult   float64
	}

	trajectory := []trajectoryStep{
		{
			name: "Turn 0: Initial prompt planning and decomposition",
			turn: TurnContext{
				TurnIndex: 0,
				Prompt:    "Implement Issue #11186: add VolumeMultiplier to EffortTier in internal/agentopt",
			},
			wantEffort: EffortHigh,
			wantCat:    CategoryPlanAndDecompose,
			wantBudget: 2048,
			wantMult:   1.6,
		},
		{
			name: "Turn 1: Routine tool read file inspection",
			turn: TurnContext{
				TurnIndex:  1,
				ToolName:   "read",
				ToolArgs:   map[string]any{"filePath": "internal/agentopt/intra_model_router.go"},
				ToolOutput: "package agentopt\n...",
			},
			wantEffort: EffortNone,
			wantCat:    CategoryRoutineToolInvocation,
			wantBudget: 0,
			wantMult:   0.4,
		},
		{
			name: "Turn 2: Run test encountering failure trace",
			turn: TurnContext{
				TurnIndex: 2,
				ToolName:  "bash",
				ToolArgs:  map[string]any{"command": "go test ./internal/agentopt"},
				ToolOutput: `=== RUN   TestIntraModelEffortRouter_TiersAndMultipliers
--- FAIL: TestIntraModelEffortRouter_TiersAndMultipliers (0.00s)
    router_test.go:40: tier.VolumeMultiplier undefined
FAIL
FAIL	github.com/anthony-chaudhary/fak/internal/agentopt	0.012s`,
			},
			wantEffort: EffortHigh,
			wantCat:    CategoryErrorRecovery,
			wantBudget: 2048,
			wantMult:   1.6,
		},
		{
			name: "Turn 3: Inspect git diff patch for proposed changes",
			turn: TurnContext{
				TurnIndex: 3,
				ToolName:  "git_diff",
				ToolOutput: `diff --git a/internal/agentopt/intra_model_router.go b/internal/agentopt/intra_model_router.go
--- a/internal/agentopt/intra_model_router.go
+++ b/internal/agentopt/intra_model_router.go
@@ -52,3 +52,15 @@
+func (e EffortTier) VolumeMultiplier() float64 {`,
			},
			wantEffort: EffortMedium,
			wantCat:    CategoryDiagnosticAndVerification,
			wantBudget: 1024,
			wantMult:   1.0,
		},
		{
			name: "Turn 4: Re-run test with passing verification",
			turn: TurnContext{
				TurnIndex: 4,
				ToolName:  "bash",
				ToolArgs:  map[string]any{"command": "go test ./internal/agentopt"},
				ToolOutput: `=== RUN   TestIntraModelEffortRouter_TiersAndMultipliers
--- PASS: TestIntraModelEffortRouter_TiersAndMultipliers (0.01s)
PASS
ok  	github.com/anthony-chaudhary/fak/internal/agentopt	0.018s`,
			},
			wantEffort: EffortLow,
			wantCat:    CategoryDiagnosticAndVerification,
			wantBudget: 256,
			wantMult:   0.7,
		},
		{
			name: "Turn 5: Synthesize and summarize verification results",
			turn: TurnContext{
				TurnIndex: 5,
				Prompt:    "Synthesize the verification findings and summarize test coverage for the new VolumeMultiplier",
			},
			wantEffort: EffortMedium,
			wantCat:    CategoryDiagnosticAndVerification,
			wantBudget: 1024,
			wantMult:   1.0,
		},
	}

	for _, step := range trajectory {
		t.Run(step.name, func(t *testing.T) {
			decision := router.Classify(step.turn)
			if decision.Effort != step.wantEffort {
				t.Errorf("%s: got effort %s, want %s (reason: %s)", step.name, decision.Effort, step.wantEffort, decision.Reason)
			}
			if decision.Category != step.wantCat {
				t.Errorf("%s: got category %s, want %s", step.name, decision.Category, step.wantCat)
			}
			if decision.ThinkingBudget != step.wantBudget {
				t.Errorf("%s: got ThinkingBudget %d, want %d", step.name, decision.ThinkingBudget, step.wantBudget)
			}
			if decision.VolumeMultiplier() != step.wantMult {
				t.Errorf("%s: got VolumeMultiplier %f, want %f", step.name, decision.VolumeMultiplier(), step.wantMult)
			}
			if router.ClassifyTurn(step.turn) != step.wantEffort {
				t.Errorf("%s: ClassifyTurn mismatch", step.name)
			}
			if router.VolumeMultiplierForTurn(step.turn) != step.wantMult {
				t.Errorf("%s: VolumeMultiplierForTurn mismatch", step.name)
			}
		})
	}
}
