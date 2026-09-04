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
