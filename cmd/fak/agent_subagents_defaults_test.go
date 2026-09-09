package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestAgentSubagentsDefaults(t *testing.T) {
	t.Run("SubagentsFlagDefaultsToTrue", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		flag := fs.Lookup("subagents")
		if flag == nil {
			t.Fatal("flag --subagents not registered on agent flag set")
		}
		if flag.DefValue != "true" {
			t.Errorf("expected --subagents default %q, got %q", "true", flag.DefValue)
		}
		if af.subagents == nil {
			t.Fatal("af.subagents pointer is nil")
		}
		if !*af.subagents {
			t.Errorf("expected *af.subagents to default to true, got false")
		}
	})

	t.Run("SubagentsExplicitFalse", func(t *testing.T) {
		fs, af := newAgentFlagSet()
		err := fs.Parse([]string{"--subagents=false"})
		if err != nil {
			t.Fatalf("Parse error: %v", err)
		}
		if af.subagents == nil {
			t.Fatal("af.subagents pointer is nil")
		}
		if *af.subagents {
			t.Errorf("expected *af.subagents to be false, got true")
		}
	})
}

func TestAgentEffortUltraAndUltracodeProfiles(t *testing.T) {
	for _, effort := range []string{"ultra", "ultracode"} {
		t.Run(effort, func(t *testing.T) {
			fs, af := newAgentFlagSet()
			err := fs.Parse([]string{"--effort", effort})
			if err != nil {
				t.Fatalf("Parse error for --effort %s: %v", effort, err)
			}
			if af.effort == nil || *af.effort != effort {
				t.Fatalf("expected af.effort = %q, got %v", effort, af.effort)
			}

			prof := agent.ResolveSubagentEffortProfile(*af.effort)
			if !prof.IsUltra {
				t.Errorf("expected prof.IsUltra == true for effort %s", effort)
			}
			if prof.MaxActiveTasks != 16 {
				t.Errorf("expected MaxActiveTasks == 16, got %d", prof.MaxActiveTasks)
			}
			if prof.MaxBacklogTasks != 64 {
				t.Errorf("expected MaxBacklogTasks == 64, got %d", prof.MaxBacklogTasks)
			}
			if prof.DefaultFanout != 8 {
				t.Errorf("expected DefaultFanout == 8, got %d", prof.DefaultFanout)
			}
			if !prof.PipelineCohorts {
				t.Errorf("expected PipelineCohorts == true for effort %s", effort)
			}
			if !prof.RequireLeases {
				t.Errorf("expected RequireLeases == true for effort %s", effort)
			}
			if !prof.SubagentsEnabled {
				t.Errorf("expected SubagentsEnabled == true for effort %s", effort)
			}
			if prof.Tier != agent.EffortTierUltra {
				t.Errorf("expected Tier %q, got %q", agent.EffortTierUltra, prof.Tier)
			}

			eff, budget := agent.ResolveReasoningProfile(*af.effort)
			if eff != agent.EffortTierUltra || budget != agent.BudgetTierUltra {
				t.Errorf("ResolveReasoningProfile(%q) = (%q, %d), want (%q, %d)",
					*af.effort, eff, budget, agent.EffortTierUltra, agent.BudgetTierUltra)
			}
			if b := agent.ResolveEffortBudget(*af.effort, nil); b != agent.BudgetTierUltra {
				t.Errorf("ResolveEffortBudget(%q) = %d, want %d", *af.effort, b, agent.BudgetTierUltra)
			}
		})
	}
}

func TestAgentArmTaskToolsWithLimits(t *testing.T) {
	defer agent.DisarmTaskTools()

	t.Run("ExplicitLimits", func(t *testing.T) {
		defer agent.DisarmTaskTools()

		_, err := agent.ArmTaskToolsWithLimits(16, 64)
		if err != nil {
			t.Fatalf("ArmTaskToolsWithLimits(16, 64) failed: %v", err)
		}

		st := agent.GetActiveTaskState()
		if st == nil {
			t.Fatal("expected active task state after ArmTaskToolsWithLimits, got nil")
		}

		maxActive, maxBacklog := st.Limits()
		if maxActive != 16 {
			t.Errorf("expected maxActive 16, got %d", maxActive)
		}
		if maxBacklog != 64 {
			t.Errorf("expected maxBacklog 64, got %d", maxBacklog)
		}
	})

	t.Run("EffortProfileWiring", func(t *testing.T) {
		defer agent.DisarmTaskTools()

		for _, effort := range []string{"ultra", "ultracode"} {
			fs, af := newAgentFlagSet()
			err := fs.Parse([]string{"--effort", effort})
			if err != nil {
				t.Fatalf("Parse error: %v", err)
			}
			if !*af.subagents {
				t.Fatalf("expected subagents to be enabled by default")
			}

			prof := agent.ResolveSubagentEffortProfile(*af.effort)
			if !prof.SubagentsEnabled {
				t.Fatalf("expected SubagentsEnabled for %s", effort)
			}

			_, err = agent.ArmTaskToolsWithLimits(prof.MaxActiveTasks, prof.MaxBacklogTasks)
			if err != nil {
				t.Fatalf("ArmTaskToolsWithLimits: %v", err)
			}

			st := agent.GetActiveTaskState()
			if st == nil {
				t.Fatalf("expected active task state")
			}

			maxActive, maxBacklog := st.Limits()
			if maxActive != prof.MaxActiveTasks {
				t.Errorf("maxActive = %d, want %d", maxActive, prof.MaxActiveTasks)
			}
			if maxBacklog != prof.MaxBacklogTasks {
				t.Errorf("maxBacklog = %d, want %d", maxBacklog, prof.MaxBacklogTasks)
			}
		}
	})
}
