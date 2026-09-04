package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// capturingPlanner records the SampleOpts passed to Complete.
type capturingPlanner struct {
	model        string
	capturedOpts []agent.SampleParams
}

func (p *capturingPlanner) Model() string { return p.model }

func (p *capturingPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	var sp agent.SampleParams
	for _, opt := range opts {
		if opt != nil {
			opt(&sp)
		}
	}
	p.capturedOpts = append(p.capturedOpts, sp)
	return &agent.Completion{
		Message: agent.Message{
			Role:    agent.RoleAssistant,
			Content: "Done.",
		},
	}, nil
}

func TestAgentEffortFlags(t *testing.T) {
	t.Run("FlagRegistration", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		effortFlag := fs.Lookup("effort")
		if effortFlag == nil {
			t.Fatal("flag --effort not registered on agent flag set")
		}
		if effortFlag.DefValue != "" {
			t.Errorf("expected --effort default %q, got %q", "", effortFlag.DefValue)
		}
		if af.effort == nil {
			t.Fatal("af.effort pointer is nil")
		}

		budgetFlag := fs.Lookup("thinking-budget")
		if budgetFlag == nil {
			t.Fatal("flag --thinking-budget not registered on agent flag set")
		}
		if budgetFlag.DefValue != "-1" {
			t.Errorf("expected --thinking-budget default \"-1\", got %q", budgetFlag.DefValue)
		}
		if af.thinkingBudget == nil {
			t.Fatal("af.thinkingBudget pointer is nil")
		}
	})

	t.Run("FlagParsingExplicitZero", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		err := fs.Parse([]string{"--thinking-budget", "0"})
		if err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		if *af.thinkingBudget != 0 {
			t.Errorf("expected thinking-budget 0, got %d", *af.thinkingBudget)
		}
		runOpts := agentEffortRunOptions(af)
		if len(runOpts) != 1 {
			t.Fatalf("expected 1 runOpt for explicit budget 0, got %d", len(runOpts))
		}
	})

	t.Run("FlagParsing", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		err := fs.Parse([]string{"--effort", "balanced", "--thinking-budget", "512"})
		if err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		if *af.effort != "balanced" {
			t.Errorf("expected effort %q, got %q", "balanced", *af.effort)
		}
		if *af.thinkingBudget != 512 {
			t.Errorf("expected thinking-budget 512, got %d", *af.thinkingBudget)
		}
	})

	t.Run("PropagationToRunOpts", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		err := fs.Parse([]string{"--effort", "balanced", "--thinking-budget", "512"})
		if err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		runOpts := agentEffortRunOptions(af)
		if len(runOpts) != 2 {
			t.Fatalf("expected 2 runOpts, got %d", len(runOpts))
		}

		planner := &capturingPlanner{model: "test-model"}
		_, err = agent.RunArm(context.Background(), planner, "test task", false, 1, nil, runOpts...)
		if err != nil {
			t.Fatalf("RunArm error: %v", err)
		}
		if len(planner.capturedOpts) == 0 {
			t.Fatal("expected at least one captured SampleParams")
		}
		got := planner.capturedOpts[0]
		if got.ReasoningEffort != "balanced" {
			t.Errorf("expected ReasoningEffort %q, got %q", "balanced", got.ReasoningEffort)
		}
		if got.ThinkingBudget == nil || *got.ThinkingBudget != 512 {
			t.Errorf("expected ThinkingBudget 512, got %v", got.ThinkingBudget)
		}
	})

	t.Run("DefaultsProduceNoOpts", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		err := fs.Parse([]string{})
		if err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		runOpts := agentEffortRunOptions(af)
		if len(runOpts) != 0 {
			t.Errorf("expected 0 runOpts for defaults, got %d", len(runOpts))
		}
	})
}

func TestAgentEffortHonored(t *testing.T) {
	t.Run("EffortBalancedAndThinkingBudget512Honored", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		if err := fs.Parse([]string{"--effort", "balanced", "--thinking-budget", "512"}); err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		runOpts := agentEffortRunOptions(af)

		planner := &capturingPlanner{model: "test-model"}
		_, err := agent.RunArm(context.Background(), planner, "test task", false, 1, nil, runOpts...)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(planner.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := planner.capturedOpts[0]
		if sp.ReasoningEffort != "balanced" {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, "balanced")
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != 512 {
			t.Errorf("ThinkingBudget = %v, want 512", sp.ThinkingBudget)
		}
	})

	t.Run("EffortBalancedDynamicBudgetHonored", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		if err := fs.Parse([]string{"--effort", "balanced"}); err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		runOpts := agentEffortRunOptions(af)

		planner := &capturingPlanner{model: "test-model"}
		_, err := agent.RunArm(context.Background(), planner, "test task", false, 1, nil, runOpts...)
		if err != nil {
			t.Fatalf("RunArm: %v", err)
		}
		if len(planner.capturedOpts) == 0 {
			t.Fatal("expected Complete to be called")
		}
		sp := planner.capturedOpts[0]
		if sp.ReasoningEffort != "balanced" {
			t.Errorf("ReasoningEffort = %q, want %q", sp.ReasoningEffort, "balanced")
		}
		if sp.ThinkingBudget == nil || *sp.ThinkingBudget != agent.BudgetBalancedDefault {
			t.Errorf("ThinkingBudget = %v, want %d", sp.ThinkingBudget, agent.BudgetBalancedDefault)
		}
	})

	t.Run("CmdAgentOfflineEndToEnd", func(t *testing.T) {
		out := filepath.Join(t.TempDir(), "native_effort.json")
		captureAgentStdio(t, func() {
			cmdAgent([]string{
				"--native",
				"--offline",
				"--effort", "balanced",
				"--thinking-budget", "512",
				"--out", out,
			})
		})
		if _, err := os.Stat(out); err != nil {
			t.Fatalf("expected output receipt file to exist: %v", err)
		}
	})
}
