package agent

import (
	"context"
	"strings"
	"testing"
)

// TestEffortSampleOptions verifies functional options WithReasoningEffort and WithThinkingBudget.
func TestEffortSampleOptions(t *testing.T) {
	t.Run("WithReasoningEffort", func(t *testing.T) {
		sp := applySampleOpts(WithReasoningEffort(EffortTierBalanced))
		if sp.ReasoningEffort != EffortTierBalanced {
			t.Fatalf("expected ReasoningEffort %q, got %q", EffortTierBalanced, sp.ReasoningEffort)
		}
	})

	t.Run("WithThinkingBudget", func(t *testing.T) {
		sp := applySampleOpts(WithThinkingBudget(512))
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != 512 {
			t.Fatalf("expected ThinkingBudget 512, got %v", sp.ThinkingBudget)
		}
	})

	t.Run("WithThinkingBudgetZeroPreserved", func(t *testing.T) {
		sp := applySampleOpts(WithThinkingBudget(0))
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != 0 {
			t.Fatalf("expected ThinkingBudget pointer to 0, got %v", sp.ThinkingBudget)
		}
	})

	t.Run("combined options", func(t *testing.T) {
		sp := applySampleOpts(
			WithReasoningEffort(EffortTierHigh),
			WithThinkingBudget(2048),
		)
		if sp.ReasoningEffort != EffortTierHigh {
			t.Errorf("expected ReasoningEffort %q, got %q", EffortTierHigh, sp.ReasoningEffort)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != 2048 {
			t.Errorf("expected ThinkingBudget 2048, got %v", sp.ThinkingBudget)
		}
	})
}

// TestResolveEffortBudgetTiers verifies static effort tier resolution.
func TestResolveEffortBudgetTiers(t *testing.T) {
	cases := []struct {
		effort string
		want   int
	}{
		{EffortTierNone, BudgetTierNone},
		{"NONE", BudgetTierNone},
		{EffortTierLow, BudgetTierLow},
		{"Low", BudgetTierLow},
		{EffortTierMedium, BudgetTierMedium},
		{"MEDIUM", BudgetTierMedium},
		{EffortTierHigh, BudgetTierHigh},
		{"High", BudgetTierHigh},
		{EffortTierBalanced, BudgetBalancedDefault},
		{"Balanced", BudgetBalancedDefault},
		{EffortTierAdaptive, BudgetBalancedDefault},
		{"Adaptive", BudgetBalancedDefault},
		{"", 0},
		{"unknown", 0},
	}

	for _, tc := range cases {
		got := ResolveEffortBudget(tc.effort, nil)
		if got != tc.want {
			t.Errorf("ResolveEffortBudget(%q, nil) = %d, want %d", tc.effort, got, tc.want)
		}
	}
}

// TestResolveEffortExplicitBudgetOverride verifies explicit budget overrides win when set and >= 0.
func TestResolveEffortExplicitBudgetOverride(t *testing.T) {
	zero := 0
	fiveHundred := 500
	negative := -1

	// Explicit 0 overrides high effort
	if got := ResolveEffortBudget(EffortTierHigh, &zero); got != 0 {
		t.Errorf("expected explicit budget 0 to override high effort, got %d", got)
	}

	// Explicit 500 overrides low effort
	if got := ResolveEffortBudget(EffortTierLow, &fiveHundred); got != 500 {
		t.Errorf("expected explicit budget 500 to override low effort, got %d", got)
	}

	// Explicit 500 overrides none effort
	if got := ResolveEffortBudget(EffortTierNone, &fiveHundred); got != 500 {
		t.Errorf("expected explicit budget 500 to override none effort, got %d", got)
	}

	// Negative explicit budget falls through to effort resolution
	if got := ResolveEffortBudget(EffortTierMedium, &negative); got != BudgetTierMedium {
		t.Errorf("expected negative budget to fall through to medium effort (%d), got %d", BudgetTierMedium, got)
	}
}

// TestResolveEffortBalancedTitration verifies balanced/adaptive budgeting for routine, error, and normal turns.
func TestResolveEffortBalancedTitration(t *testing.T) {
	for _, tier := range []string{EffortTierBalanced, EffortTierAdaptive} {
		t.Run(tier+"/default_without_assessment", func(t *testing.T) {
			got := ResolveEffortBudget(tier, nil)
			if got != BudgetBalancedDefault {
				t.Fatalf("expected default %d, got %d", BudgetBalancedDefault, got)
			}
		})

		t.Run(tier+"/routine_tool_turns", func(t *testing.T) {
			routineTools := []string{"read", "glob", "grep", "cat", "view", "read_file", "list_dir", "find", "file_search"}
			for _, tool := range routineTools {
				ta := TurnAssessment{ToolName: tool}
				got := ResolveEffortBudget(tier, nil, ta)
				if got != BudgetBalancedRoutineTool {
					t.Errorf("expected routine tool %s to yield %d, got %d", tool, BudgetBalancedRoutineTool, got)
				}
			}

			// Explicit flag
			got := ResolveEffortBudget(tier, nil, TurnAssessment{IsRoutineTool: true})
			if got != BudgetBalancedRoutineTool {
				t.Errorf("expected IsRoutineTool: true to yield %d, got %d", BudgetBalancedRoutineTool, got)
			}
		})

		t.Run(tier+"/error_recovery_turns", func(t *testing.T) {
			errorCases := []TurnAssessment{
				{ErrorMessage: "exit status 2: compiler error: undefined: Foo"},
				{ErrorMessage: "FAIL: test failure in TestBar"},
				{ErrorMessage: "panic: runtime error: invalid memory address"},
				{ErrorMessage: "policy block: execution denied by capability boundary"},
				{ErrorMessage: "policy_block: refused"},
				{IsErrorRecovery: true},
				// Error recovery on a routine tool name still gets error budget
				{ToolName: "read", ErrorMessage: "read error: file not found"},
			}

			for i, ta := range errorCases {
				got := ResolveEffortBudget(tier, nil, ta)
				if got != BudgetBalancedError {
					t.Errorf("case %d: expected error turn to yield %d, got %d", i, BudgetBalancedError, got)
				}
			}
		})

		t.Run(tier+"/normal_turns", func(t *testing.T) {
			normalCases := []TurnAssessment{
				{ToolName: "bash"},
				{ToolName: "custom_analyzer"},
				{},
			}

			for i, ta := range normalCases {
				got := ResolveEffortBudget(tier, nil, ta)
				if got != BudgetBalancedDefault {
					t.Errorf("case %d: expected normal turn to yield %d, got %d", i, BudgetBalancedDefault, got)
				}
			}
		})
	}

	t.Run("AssessTranscriptTurn", func(t *testing.T) {
		// Routine tool
		routineMsgs := []Message{
			{Role: RoleUser, Content: "look at file"},
			{Role: RoleAssistant, Content: "reading"},
			{Role: RoleTool, Name: "read", Content: "file content here"},
		}
		ta, ok := AssessTranscriptTurn(routineMsgs)
		if !ok || !ta.IsRoutine() {
			t.Fatalf("expected routine assessment, got ok=%v, ta=%+v", ok, ta)
		}
		if got := ResolveEffortBudget(EffortTierBalanced, nil, ta); got != BudgetBalancedRoutineTool {
			t.Errorf("expected 0 for routine transcript, got %d", got)
		}

		// Error tool
		errMsgs := []Message{
			{Role: RoleUser, Content: "run test"},
			{Role: RoleAssistant, Content: "running"},
			{Role: RoleTool, Name: "bash", Content: "--- FAIL: TestXYZ\ncompiler error: syntax error"},
		}
		taErr, ok := AssessTranscriptTurn(errMsgs)
		if !ok || !taErr.IsError() {
			t.Fatalf("expected error assessment, got ok=%v, ta=%+v", ok, taErr)
		}
		if got := ResolveEffortBudget(EffortTierBalanced, nil, taErr); got != BudgetBalancedError {
			t.Errorf("expected 1536 for error transcript, got %d", got)
		}

		// Multi-tool batch with earlier error followed by routine read
		multiToolMsgs := []Message{
			{Role: RoleUser, Content: "run test"},
			{Role: RoleAssistant, Content: "running"},
			{Role: RoleTool, Name: "bash", Content: "--- FAIL: TestXYZ\nexit status 1"},
			{Role: RoleTool, Name: "read", Content: "package main\n..."},
		}
		taMulti, ok := AssessTranscriptTurn(multiToolMsgs)
		if !ok || !taMulti.IsError() {
			t.Fatalf("expected error assessment for multi-tool batch containing error, got ok=%v, ta=%+v", ok, taMulti)
		}
		if got := ResolveEffortBudget(EffortTierBalanced, nil, taMulti); got != BudgetBalancedError {
			t.Errorf("expected 1536 for multi-tool batch with error, got %d", got)
		}
		userMsgs := []Message{
			{Role: RoleUser, Content: "just starting"},
		}
		_, ok = AssessTranscriptTurn(userMsgs)
		if ok {
			t.Fatalf("expected ok=false for non-tool trailing message")
		}
	})
}

// TestEffortInKernelRenderSuppression verifies inKernelSuppressQwenThinking for "none" vs "balanced" vs "high".
func TestEffortInKernelRenderSuppression(t *testing.T) {
	// Hybrid config representing a reasoning model
	cfg := bonsaiConfig()

	// 1. "none" suppresses thinking
	spNone := SampleParams{ReasoningEffort: EffortTierNone}
	if !inKernelSuppressQwenThinking(cfg, spNone) {
		t.Errorf("inKernelSuppressQwenThinking must return true for effort %q", EffortTierNone)
	}

	// Explicit 0 budget suppresses thinking
	zero := 0
	spZeroBudget := SampleParams{ThinkingBudget: &zero}
	if !inKernelSuppressQwenThinking(cfg, spZeroBudget) {
		t.Errorf("inKernelSuppressQwenThinking must return true for budget 0")
	}

	// 2. "balanced" does NOT suppress thinking
	spBalanced := SampleParams{ReasoningEffort: EffortTierBalanced}
	if inKernelSuppressQwenThinking(cfg, spBalanced) {
		t.Errorf("inKernelSuppressQwenThinking must return false for effort %q", EffortTierBalanced)
	}

	// 3. "high" does NOT suppress thinking
	spHigh := SampleParams{ReasoningEffort: EffortTierHigh}
	if inKernelSuppressQwenThinking(cfg, spHigh) {
		t.Errorf("inKernelSuppressQwenThinking must return false for effort %q", EffortTierHigh)
	}

	// Explicit positive budget does NOT suppress thinking
	positive := 1024
	spPositiveBudget := SampleParams{ThinkingBudget: &positive}
	if inKernelSuppressQwenThinking(cfg, spPositiveBudget) {
		t.Errorf("inKernelSuppressQwenThinking must return false for budget %d", positive)
	}

	// Test prompt rendering with renderInKernelChatMLRequest
	msgs := []Message{{Role: RoleUser, Content: "Solve this puzzle."}}

	chatNone := renderInKernelChatMLRequest(msgs, nil, cfg, nil, nil, spNone)
	if !strings.HasSuffix(chatNone, qwenNoThinkAssistantSeed) {
		t.Errorf("renderInKernelChatMLRequest with effort=none must end with %q, got:\n%q", qwenNoThinkAssistantSeed, chatNone)
	}

	chatBalanced := renderInKernelChatMLRequest(msgs, nil, cfg, nil, nil, spBalanced)
	if strings.Contains(chatBalanced, qwenNoThinkAssistantSeed) {
		t.Errorf("renderInKernelChatMLRequest with effort=balanced must NOT contain %q, got:\n%q", qwenNoThinkAssistantSeed, chatBalanced)
	}

	chatHigh := renderInKernelChatMLRequest(msgs, nil, cfg, nil, nil, spHigh)
	if strings.Contains(chatHigh, qwenNoThinkAssistantSeed) {
		t.Errorf("renderInKernelChatMLRequest with effort=high must NOT contain %q, got:\n%q", qwenNoThinkAssistantSeed, chatHigh)
	}
}

// TestEffortResidualThinkingCleanUp verifies that residual think tags in content are properly cleaned.
func TestEffortResidualThinkingCleanUp(t *testing.T) {
	// Simulate an output where early </think> was forced, and the model emitted residual thinking
	// followed by another </think> before the answer.
	in := "<think>step 1\n</think>\n\nstep 2</think>The final answer is 42."
	reasoning, content := splitReasoning(in)
	for strings.Contains(content, thinkClose) {
		idx := strings.Index(content, thinkClose)
		residual := strings.TrimSpace(content[:idx])
		residual = strings.TrimPrefix(residual, thinkOpen)
		residual = strings.TrimSpace(residual)
		if residual != "" {
			if reasoning != "" {
				reasoning = strings.TrimSpace(reasoning + "\n" + residual)
			} else {
				reasoning = residual
			}
		}
		content = strings.TrimSpace(content[idx+len(thinkClose):])
	}
	content = StripReasoning(content)

	wantReasoning := "step 1\nstep 2"
	wantContent := "The final answer is 42."

	if reasoning != wantReasoning {
		t.Errorf("reasoning = %q, want %q", reasoning, wantReasoning)
	}
	if content != wantContent {
		t.Errorf("content = %q, want %q", content, wantContent)
	}
}

type mockEffortPlanner struct {
	model        string
	capturedOpts []SampleParams
}

func (p *mockEffortPlanner) Model() string { return p.model }

func (p *mockEffortPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, opts ...SampleOpt) (*Completion, error) {
	sp := applySampleOpts(opts...)
	p.capturedOpts = append(p.capturedOpts, sp)
	return &Completion{
		Message: Message{
			Role:    RoleAssistant,
			Content: "Done.",
		},
	}, nil
}

// TestEffortTurnLoopResolution verifies reasoning effort and thinking budget options wired into runArm.
func TestEffortTurnLoopResolution(t *testing.T) {
	ctx := context.Background()

	t.Run("BalancedDefaultBudget", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		_, err := RunArm(ctx, p, "task", false, 1, nil, WithRunReasoningEffort(EffortTierBalanced))
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierBalanced {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierBalanced)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != BudgetBalancedDefault {
			t.Errorf("ThinkingBudget = %v, want %d", sp.ThinkingBudget, BudgetBalancedDefault)
		}
	})

	t.Run("BalancedRoutineToolBudget", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		msgs := []Message{
			{Role: RoleUser, Content: "look at file"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Function: Func{Name: "read"}}}},
			{Role: RoleTool, Name: "read", Content: "file contents"},
		}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithRunReasoningEffort(EffortTierBalanced),
			WithConversation(msgs),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierBalanced {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierBalanced)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != BudgetBalancedRoutineTool {
			t.Errorf("ThinkingBudget = %v, want %d", sp.ThinkingBudget, BudgetBalancedRoutineTool)
		}
	})

	t.Run("BalancedErrorRecoveryBudget", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		msgs := []Message{
			{Role: RoleUser, Content: "run build"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Function: Func{Name: "bash"}}}},
			{Role: RoleTool, Name: "bash", Content: "compiler error: undefined identifier"},
		}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithRunReasoningEffort(EffortTierBalanced),
			WithConversation(msgs),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierBalanced {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierBalanced)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != BudgetBalancedError {
			t.Errorf("ThinkingBudget = %v, want %d", sp.ThinkingBudget, BudgetBalancedError)
		}
	})

	t.Run("BalancedWithExplicitBudgetOverride", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithRunReasoningEffort(EffortTierBalanced),
			WithRunThinkingBudget(512),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierBalanced {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierBalanced)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != 512 {
			t.Errorf("ThinkingBudget = %v, want 512", sp.ThinkingBudget)
		}
	})

	t.Run("StaticEffortHigh", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithRunReasoningEffort(EffortTierHigh),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierHigh {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierHigh)
		}
		if sp.ThinkingBudget != nil {
			t.Errorf("ThinkingBudget = %v, want nil", sp.ThinkingBudget)
		}
	})

	t.Run("ExplicitBudgetOnly", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithRunThinkingBudget(1024),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != "" {
			t.Errorf("ReasoningEffort = %q, want empty", sp.ReasoningEffort)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != 1024 {
			t.Errorf("ThinkingBudget = %v, want 1024", sp.ThinkingBudget)
		}
	})
}
