package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestReasoningProfileFlags(t *testing.T) {
	testReasoningProfileFlags(t)
}

func TestAgentReasoningProfileFlags(t *testing.T) {
	testReasoningProfileFlags(t)
}

func testReasoningProfileFlags(t *testing.T) {
	t.Run("FlagRegistrationAgent", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		flg := fs.Lookup("reasoning-profile")
		if flg == nil {
			t.Fatal("flag --reasoning-profile not registered on agent flag set")
		}
		if flg.DefValue != agent.ReasoningProfileDefault {
			t.Errorf("expected default %q, got %q", agent.ReasoningProfileDefault, flg.DefValue)
		}
		if af.reasoningProfile == nil {
			t.Fatal("af.reasoningProfile pointer is nil")
		}
	})

	t.Run("FlagRegistrationChat", func(t *testing.T) {
		fs, cf := newChatFlagSet()
		flg := fs.Lookup("reasoning-profile")
		if flg == nil {
			t.Fatal("flag --reasoning-profile not registered on chat flag set")
		}
		if flg.DefValue != agent.ReasoningProfileDefault {
			t.Errorf("expected default %q, got %q", agent.ReasoningProfileDefault, flg.DefValue)
		}
		if cf.reasoningProfile == nil {
			t.Fatal("cf.reasoningProfile pointer is nil")
		}
	})

	t.Run("Validation", func(t *testing.T) {
		valid := []string{"default", "baseline", "deep-reason", "DEFAULT", "Baseline", "DEEP-REASON"}
		for _, p := range valid {
			if err := validateReasoningProfile(p); err != nil {
				t.Errorf("validateReasoningProfile(%q) unexpectedly returned error: %v", p, err)
			}
		}

		invalid := []string{"unsupported", "extreme", "turbo", "slow"}
		for _, p := range invalid {
			if err := validateReasoningProfile(p); err == nil {
				t.Errorf("validateReasoningProfile(%q) expected error, got nil", p)
			}
		}
	})

	t.Run("AgentFlagParsingAndOptions", func(t *testing.T) {
		cases := []struct {
			args        []string
			wantProfile string
			wantOptsLen int
		}{
			{args: []string{}, wantProfile: agent.ReasoningProfileDefault, wantOptsLen: 0},
			{args: []string{"--reasoning-profile", "deep-reason"}, wantProfile: "deep-reason", wantOptsLen: 0},
			{args: []string{"--reasoning-profile", "baseline"}, wantProfile: "baseline", wantOptsLen: 0},
			{args: []string{"--reasoning-profile", "deep-reason", "--effort", "high"}, wantProfile: "deep-reason", wantOptsLen: 1},
			{args: []string{"--reasoning-profile", "default", "--thinking-budget", "512"}, wantProfile: "default", wantOptsLen: 1},
		}

		for _, tc := range cases {
			fs, af := newAgentFlagSet()
			if err := fs.Parse(tc.args); err != nil {
				t.Fatalf("Parse(%v) error: %v", tc.args, err)
			}
			if *af.reasoningProfile != tc.wantProfile {
				t.Errorf("for args %v: got profile %q, want %q", tc.args, *af.reasoningProfile, tc.wantProfile)
			}
			opts := agentEffortRunOptions(af)
			if len(opts) != tc.wantOptsLen {
				t.Errorf("for args %v: got %d options, want %d", tc.args, len(opts), tc.wantOptsLen)
			}
			profOpt := agentReasoningProfileRunOption(af)
			if profOpt == nil {
				t.Errorf("for args %v: expected non-nil profile option", tc.args)
			}
		}
	})

	t.Run("ChatFlagParsing", func(t *testing.T) {
		fs, cf := newChatFlagSet()
		if err := fs.Parse([]string{"--reasoning-profile", "deep-reason"}); err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		if *cf.reasoningProfile != "deep-reason" {
			t.Errorf("expected profile %q, got %q", "deep-reason", *cf.reasoningProfile)
		}
	})

	t.Run("ExecutionWithReasoningProfile", func(t *testing.T) {
		ctx := context.Background()
		p := &capturingPlanner{model: "mock-model"}

		// Test default profile on routine turn clamps thinking budget to 0
		msgs := []agent.Message{
			{Role: agent.RoleUser, Content: "inspect files"},
			{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "c1", Function: agent.Func{Name: "read"}}}},
			{Role: agent.RoleTool, Name: "read", Content: "file content"},
		}

		fs, af := newAgentFlagSet()
		if err := fs.Parse([]string{"--reasoning-profile", "default"}); err != nil {
			t.Fatalf("Parse: %v", err)
		}
		var runOpts []agent.RunOption
		runOpts = append(runOpts, agentEffortRunOptions(af)...)
		if opt := agentReasoningProfileRunOption(af); opt != nil {
			runOpts = append(runOpts, opt)
		}
		opts := append(runOpts, agent.WithConversation(msgs))

		_, err := agent.RunArm(ctx, p, "task", false, 1, nil, opts...)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := p.capturedOpts[0]
		if sp.ReasoningEffort != agent.EffortTierMedium {
			t.Errorf("expected ReasoningEffort %q, got %q", agent.EffortTierMedium, sp.ReasoningEffort)
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != agent.BudgetBalancedRoutineTool {
			t.Errorf("expected ThinkingBudget %d, got %v", agent.BudgetBalancedRoutineTool, sp.ThinkingBudget)
		}

		// Test deep-reason profile keeps high budget
		p.capturedOpts = nil
		fsDeep, afDeep := newAgentFlagSet()
		if err := fsDeep.Parse([]string{"--reasoning-profile", "deep-reason"}); err != nil {
			t.Fatalf("Parse: %v", err)
		}
		var runOptsDeep []agent.RunOption
		runOptsDeep = append(runOptsDeep, agentEffortRunOptions(afDeep)...)
		if opt := agentReasoningProfileRunOption(afDeep); opt != nil {
			runOptsDeep = append(runOptsDeep, opt)
		}
		optsDeep := append(runOptsDeep, agent.WithConversation(msgs))

		_, err = agent.RunArm(ctx, p, "task", false, 1, nil, optsDeep...)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(p.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		spDeep := p.capturedOpts[0]
		if spDeep.ReasoningEffort != agent.EffortTierHigh {
			t.Errorf("expected ReasoningEffort %q, got %q", agent.EffortTierHigh, spDeep.ReasoningEffort)
		}
		if spDeep.ThinkingBudget == nil || *spDeep.ThinkingBudget != agent.BudgetTierHigh {
			t.Errorf("expected ThinkingBudget %d, got %v", agent.BudgetTierHigh, spDeep.ThinkingBudget)
		}
	})
}
