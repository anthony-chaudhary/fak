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

		// Routine tool reading file content with error terms should stay routine
		routineWithContent := []Message{
			{Role: RoleUser, Content: "grep errors"},
			{Role: RoleAssistant, Content: "grepping"},
			{Role: RoleTool, Name: "grep", Content: "compiler error handling logic on line 42"},
		}
		taRoutineContent, ok := AssessTranscriptTurn(routineWithContent)
		if !ok || taRoutineContent.IsError() || !taRoutineContent.IsRoutine() {
			t.Fatalf("routine grep containing error phrase should stay routine, got ok=%v, ta=%+v", ok, taRoutineContent)
		}

		// Tool returning exit status 0 is success, not error
		exitZeroMsgs := []Message{
			{Role: RoleUser, Content: "run test"},
			{Role: RoleAssistant, Content: "running"},
			{Role: RoleTool, Name: "bash", Content: "PASS: all tests passed\nexit status 0"},
		}
		taExitZero, ok := AssessTranscriptTurn(exitZeroMsgs)
		if !ok || taExitZero.IsError() {
			t.Fatalf("exit status 0 should not trigger error recovery, got ok=%v, ta=%+v", ok, taExitZero)
		}

		// Tool with benign "exit status" phrase in output without non-zero code
		benignExitMsgs := []Message{
			{Role: RoleUser, Content: "git log"},
			{Role: RoleAssistant, Content: "checking log"},
			{Role: RoleTool, Name: "bash", Content: "commit abc: improve exit status handling in worker"},
		}
		taBenignExit, ok := AssessTranscriptTurn(benignExitMsgs)
		if !ok || taBenignExit.IsError() {
			t.Fatalf("benign exit status phrase should not trigger error recovery, got ok=%v, ta=%+v", ok, taBenignExit)
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

// TestResolveReasoningProfile verifies profile resolution to effort tier and budget ceiling.
func TestResolveReasoningProfile(t *testing.T) {
	cases := []struct {
		profile    string
		wantEffort string
		wantBudget int
	}{
		{ReasoningProfileDefault, EffortTierMedium, BudgetTierMedium},
		{ReasoningProfileBaseline, EffortTierMedium, BudgetTierMedium},
		{ReasoningProfileDeepReason, EffortTierHigh, BudgetTierHigh},
		{"DEFAULT", EffortTierMedium, BudgetTierMedium},
		{"Baseline", EffortTierMedium, BudgetTierMedium},
		{"deep-reason", EffortTierHigh, BudgetTierHigh},
		{"DEEP-REASON", EffortTierHigh, BudgetTierHigh},
		{"deepreason", EffortTierHigh, BudgetTierHigh},
		{"deep_reason", EffortTierHigh, BudgetTierHigh},
		{"", EffortTierMedium, BudgetTierMedium},
		{"unknown-profile", EffortTierMedium, BudgetTierMedium},
		{"high", EffortTierHigh, BudgetTierHigh},
		{"medium", EffortTierMedium, BudgetTierMedium},
		{"low", EffortTierLow, BudgetTierLow},
		{"none", EffortTierNone, BudgetTierNone},
	}

	for _, tc := range cases {
		gotEffort, gotBudget := ResolveReasoningProfile(tc.profile)
		if gotEffort != tc.wantEffort || gotBudget != tc.wantBudget {
			t.Errorf("ResolveReasoningProfile(%q) = (%q, %d), want (%q, %d)",
				tc.profile, gotEffort, gotBudget, tc.wantEffort, tc.wantBudget)
		}
	}

	if !IsValidReasoningProfile("default") || !IsValidReasoningProfile("baseline") || !IsValidReasoningProfile("deep-reason") {
		t.Error("expected default, baseline, and deep-reason to be valid reasoning profiles")
	}
	if IsValidReasoningProfile("invalid_profile") {
		t.Error("expected invalid_profile to not be valid")
	}
	valid := ValidReasoningProfiles()
	if len(valid) != 3 {
		t.Errorf("expected 3 valid reasoning profiles, got %d", len(valid))
	}
}

// TestResolveReasoningProfileBudget verifies dynamic budget mapping under named profiles.
func TestResolveReasoningProfileBudget(t *testing.T) {
	for _, profile := range []string{ReasoningProfileDefault, ReasoningProfileBaseline, ""} {
		t.Run("DefaultProfile/RoutineToolClampedToZero", func(t *testing.T) {
			ta := TurnAssessment{ToolName: "read", IsRoutineTool: true}
			eff, budget := ResolveReasoningProfileBudget(profile, nil, ta)
			if eff != EffortTierMedium {
				t.Errorf("effort = %q, want %q", eff, EffortTierMedium)
			}
			if budget != BudgetBalancedRoutineTool {
				t.Errorf("budget = %d, want %d (zero overhead for routine turn)", budget, BudgetBalancedRoutineTool)
			}
		})

		t.Run("DefaultProfile/ErrorRecoveryElevated", func(t *testing.T) {
			ta := TurnAssessment{ErrorMessage: "compiler error: syntax error", IsErrorRecovery: true}
			eff, budget := ResolveReasoningProfileBudget(profile, nil, ta)
			if eff != EffortTierMedium {
				t.Errorf("effort = %q, want %q", eff, EffortTierMedium)
			}
			if budget != BudgetBalancedError {
				t.Errorf("budget = %d, want %d", budget, BudgetBalancedError)
			}
		})

		t.Run("DefaultProfile/NormalTurnMediumBudget", func(t *testing.T) {
			ta := TurnAssessment{ToolName: "bash"}
			eff, budget := ResolveReasoningProfileBudget(profile, nil, ta)
			if eff != EffortTierMedium {
				t.Errorf("effort = %q, want %q", eff, EffortTierMedium)
			}
			if budget != BudgetTierMedium {
				t.Errorf("budget = %d, want %d", budget, BudgetTierMedium)
			}
		})

		t.Run("DefaultProfile/ExplicitBudgetOverride", func(t *testing.T) {
			override := 512
			ta := TurnAssessment{ToolName: "read", IsRoutineTool: true}
			eff, budget := ResolveReasoningProfileBudget(profile, &override, ta)
			if eff != EffortTierMedium {
				t.Errorf("effort = %q, want %q", eff, EffortTierMedium)
			}
			if budget != 512 {
				t.Errorf("budget = %d, want 512", budget)
			}
		})
	}

	t.Run("DeepReasonProfile/HighBudgetMaintained", func(t *testing.T) {
		// Even for routine tool turns, deep-reason retains high capacity for complex delegation
		ta := TurnAssessment{ToolName: "read", IsRoutineTool: true}
		eff, budget := ResolveReasoningProfileBudget(ReasoningProfileDeepReason, nil, ta)
		if eff != EffortTierHigh {
			t.Errorf("effort = %q, want %q", eff, EffortTierHigh)
		}
		if budget != BudgetTierHigh {
			t.Errorf("budget = %d, want %d", budget, BudgetTierHigh)
		}
	})
}

// TestRoutineTurnClassification verifies routine turn classification helpers.
func TestRoutineTurnClassification(t *testing.T) {
	routineTools := []string{"read", "glob", "grep", "cat", "view", "read_file", "list_dir", "find", "file_search"}
	for _, tool := range routineTools {
		if !IsRoutineToolName(tool) {
			t.Errorf("expected %q to be classified as routine tool name", tool)
		}
	}

	nonRoutineTools := []string{"bash", "write", "edit", "custom_eval", "test_runner"}
	for _, tool := range nonRoutineTools {
		if IsRoutineToolName(tool) {
			t.Errorf("expected %q to NOT be classified as routine tool name", tool)
		}
	}

	// IsRoutineTurn with routine message transcript
	routineTranscript := []Message{
		{Role: RoleUser, Content: "show files"},
		{Role: RoleAssistant, Content: "checking"},
		{Role: RoleTool, Name: "glob", Content: "file1.go\nfile2.go"},
	}
	if !IsRoutineTurn(routineTranscript) {
		t.Error("expected routineTranscript to be classified as routine turn")
	}

	// Transcript with error is NOT a routine turn
	errorTranscript := []Message{
		{Role: RoleUser, Content: "show files"},
		{Role: RoleAssistant, Content: "checking"},
		{Role: RoleTool, Name: "read", Content: "compiler error: syntax error"},
	}
	if IsRoutineTurn(errorTranscript) {
		t.Error("expected errorTranscript to NOT be classified as routine turn")
	}

	// Non-tool trailing message is NOT a routine turn
	userTranscript := []Message{
		{Role: RoleUser, Content: "hello"},
	}
	if IsRoutineTurn(userTranscript) {
		t.Error("expected userTranscript to NOT be classified as routine turn")
	}
}

// TestReasoningProfileTurnLoop verifies end-to-end RunArm wiring of WithReasoningProfile.
func TestReasoningProfileTurnLoop(t *testing.T) {
	ctx := context.Background()

	t.Run("DefaultProfileRoutineToolTurn", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		msgs := []Message{
			{Role: RoleUser, Content: "check directory"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Function: Func{Name: "glob"}}}},
			{Role: RoleTool, Name: "glob", Content: "a.go\nb.go"},
		}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithReasoningProfile(ReasoningProfileDefault),
			WithConversation(msgs),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierMedium {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierMedium)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != BudgetBalancedRoutineTool {
			t.Errorf("ThinkingBudget = %v, want %d (zero overhead on routine tool turn)", sp.ThinkingBudget, BudgetBalancedRoutineTool)
		}
	})

	t.Run("DefaultProfileErrorTurn", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		msgs := []Message{
			{Role: RoleUser, Content: "run build"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Function: Func{Name: "bash"}}}},
			{Role: RoleTool, Name: "bash", Content: "panic: nil pointer dereference"},
		}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithReasoningProfile(ReasoningProfileDefault),
			WithConversation(msgs),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierMedium {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierMedium)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != BudgetBalancedError {
			t.Errorf("ThinkingBudget = %v, want %d", sp.ThinkingBudget, BudgetBalancedError)
		}
	})

	t.Run("DefaultProfileNormalTurn", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		msgs := []Message{
			{Role: RoleUser, Content: "solve problem"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Function: Func{Name: "bash"}}}},
			{Role: RoleTool, Name: "bash", Content: "output without error"},
		}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithReasoningProfile(ReasoningProfileDefault),
			WithConversation(msgs),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierMedium {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierMedium)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != BudgetTierMedium {
			t.Errorf("ThinkingBudget = %v, want %d", sp.ThinkingBudget, BudgetTierMedium)
		}
	})

	t.Run("DeepReasonProfile", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		msgs := []Message{
			{Role: RoleUser, Content: "audit concurrency lock ordering"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Function: Func{Name: "read"}}}},
			{Role: RoleTool, Name: "read", Content: "package kernel\n..."},
		}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithReasoningProfile(ReasoningProfileDeepReason),
			WithConversation(msgs),
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
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != BudgetTierHigh {
			t.Errorf("ThinkingBudget = %v, want %d", sp.ThinkingBudget, BudgetTierHigh)
		}
	})

	t.Run("DefaultProfileWithExplicitBudgetOverride", func(t *testing.T) {
		p := &mockEffortPlanner{model: "test-model"}
		msgs := []Message{
			{Role: RoleUser, Content: "check file"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Function: Func{Name: "read"}}}},
			{Role: RoleTool, Name: "read", Content: "data"},
		}
		_, err := RunArm(ctx, p, "task", false, 1, nil,
			WithReasoningProfile(ReasoningProfileDefault),
			WithRunThinkingBudget(512),
			WithConversation(msgs),
		)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != EffortTierMedium {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, EffortTierMedium)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != 512 {
			t.Errorf("ThinkingBudget = %v, want 512", sp.ThinkingBudget)
		}
	})
}
