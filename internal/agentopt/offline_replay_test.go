package agentopt

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestOfflineTrajectoryReplay(t *testing.T) {
	ctx := context.Background()

	// Helper to build a standard 3-turn baseline trajectory.
	buildBaselineTrajectory := func() Trajectory {
		return Trajectory{
			ID:          "traj-baseline-001",
			Description: "Historical baseline for log investigation",
			Prompt:      "Inspect server.log for error code 500 and examine handler",
			Turns: []TrajectoryTurn{
				{
					TurnIndex: 0,
					Prompt:    "Inspect server.log for error code 500",
					ToolCalls: []ToolCall{
						{
							ID:   "call-grep-1",
							Name: "Grep",
							Args: map[string]any{
								"path":    "server.log",
								"pattern": "HTTP 500",
							},
							ReadOnly: true,
						},
					},
					Results: []ToolResult{
						{CallID: "call-grep-1", Output: "line 104: HTTP 500 Internal Server Error in /api/checkout"},
					},
					Observations: []EnvironmentObservation{
						{Source: "Grep", Key: "call-grep-1", Content: "line 104: HTTP 500 Internal Server Error in /api/checkout"},
					},
				},
				{
					TurnIndex: 1,
					Prompt:    "Examine checkout handler around line 104",
					ToolCalls: []ToolCall{
						{
							ID:   "call-read-1",
							Name: "Read",
							Args: map[string]any{
								"filePath": "checkout.go",
								"offset":   100,
								"limit":    20,
							},
							ReadOnly: true,
						},
					},
					Results: []ToolResult{
						{CallID: "call-read-1", Output: "func checkoutHandler() { panic(\"db connection failed\") }"},
					},
					Observations: []EnvironmentObservation{
						{Source: "Read", Key: "call-read-1", Content: "func checkoutHandler() { panic(\"db connection failed\") }"},
					},
				},
				{
					TurnIndex: 2,
					Prompt:    "Summarize finding",
					Output:    "Root cause identified: database connection failure in checkoutHandler at line 104.",
				},
			},
		}
	}

	t.Run("deterministic replay matches baseline with zero divergence", func(t *testing.T) {
		baseline := buildBaselineTrajectory()
		env := NewHermeticEnvironment()
		env.RegisterTool("Grep", func(ctx context.Context, call ToolCall) (string, error) {
			return "line 104: HTTP 500 Internal Server Error in /api/checkout", nil
		})
		env.RegisterTool("Read", func(ctx context.Context, call ToolCall) (string, error) {
			return "func checkoutHandler() { panic(\"db connection failed\") }", nil
		})

		evaluator := NewOfflineTrajectoryEvaluator(env)
		replayed, report, err := evaluator.Replay(ctx, baseline, ReplayConfig{MaxTurns: 10})
		if err != nil {
			t.Fatalf("replay failed unexpectedly: %v", err)
		}

		if report.Diverged {
			t.Fatalf("expected zero divergence, got diverged with summary: %s", report.Summary)
		}
		if report.RegressionDetected {
			t.Fatal("expected regression_detected = false")
		}
		if report.RegressionScore != 0.0 {
			t.Fatalf("expected regression score = 0, got %f", report.RegressionScore)
		}
		if report.TurnCountDelta != 0 {
			t.Fatalf("expected turn count delta = 0, got %d", report.TurnCountDelta)
		}
		if len(report.ToolSelectionChanges) != 0 {
			t.Fatalf("expected 0 tool selection changes, got %d", len(report.ToolSelectionChanges))
		}
		if len(report.ArgumentMutations) != 0 {
			t.Fatalf("expected 0 argument mutations, got %d", len(report.ArgumentMutations))
		}
		if len(replayed.Turns) != len(baseline.Turns) {
			t.Fatalf("replayed turn count = %d, want %d", len(replayed.Turns), len(baseline.Turns))
		}
		if !strings.Contains(report.Summary, "deterministic match") {
			t.Fatalf("expected summary to indicate deterministic match, got: %q", report.Summary)
		}
	})

	t.Run("detects tool selection divergence", func(t *testing.T) {
		baseline := buildBaselineTrajectory()
		env := NewHermeticEnvironment()
		env.RegisterTool("Grep", func(ctx context.Context, call ToolCall) (string, error) {
			return "line 104: HTTP 500", nil
		})
		env.RegisterTool("Bash", func(ctx context.Context, call ToolCall) (string, error) {
			return "cat checkout.go output", nil
		})

		evaluator := NewOfflineTrajectoryEvaluator(env)

		// Replayed agent regresses at turn 1: uses "Bash" instead of "Read".
		replayedAgent := func(ctx context.Context, turnIndex int, history []TrajectoryTurn, env *HermeticEnvironment) (TrajectoryTurn, bool, error) {
			switch turnIndex {
			case 0:
				return baseline.Turns[0], false, nil
			case 1:
				return TrajectoryTurn{
					TurnIndex: 1,
					ToolCalls: []ToolCall{
						{
							ID:       "call-bash-1",
							Name:     "Bash",
							Args:     map[string]any{"command": "cat checkout.go"},
							ReadOnly: false,
						},
					},
				}, false, nil
			case 2:
				return baseline.Turns[2], true, nil
			default:
				return TrajectoryTurn{}, true, nil
			}
		}

		_, report, err := evaluator.Replay(ctx, baseline, ReplayConfig{
			AgentFunc: replayedAgent,
			MaxTurns:  5,
		})
		if err != nil {
			t.Fatalf("replay failed: %v", err)
		}

		if !report.Diverged {
			t.Fatal("expected diverged = true on tool selection change")
		}
		if !report.RegressionDetected {
			t.Fatal("expected regression_detected = true")
		}
		if len(report.ToolSelectionChanges) != 1 {
			t.Fatalf("expected 1 tool selection change, got %d", len(report.ToolSelectionChanges))
		}

		tsc := report.ToolSelectionChanges[0]
		if tsc.TurnIndex != 1 {
			t.Fatalf("expected tool selection change at turn 1, got %d", tsc.TurnIndex)
		}
		if len(tsc.BaselineTools) != 1 || tsc.BaselineTools[0] != "Read" {
			t.Fatalf("expected baseline tool Read, got %v", tsc.BaselineTools)
		}
		if len(tsc.ReplayedTools) != 1 || tsc.ReplayedTools[0] != "Bash" {
			t.Fatalf("expected replayed tool Bash, got %v", tsc.ReplayedTools)
		}
		if report.RegressionScore <= 0 {
			t.Fatalf("expected positive regression score, got %f", report.RegressionScore)
		}
	})

	t.Run("detects argument mutations: modified, added, removed", func(t *testing.T) {
		baseline := buildBaselineTrajectory()
		env := NewHermeticEnvironment()
		env.RegisterTool("Grep", func(ctx context.Context, call ToolCall) (string, error) {
			return "matched", nil
		})
		env.RegisterTool("Read", func(ctx context.Context, call ToolCall) (string, error) {
			return "read content", nil
		})

		evaluator := NewOfflineTrajectoryEvaluator(env)

		// Replayed mutates arguments:
		// Turn 0: modifies "pattern" from "HTTP 500" to "HTTP 404", adds "max_results": 5
		// Turn 1: removes "limit", modifies "offset" from 100 to 200
		replayedAgent := func(ctx context.Context, turnIndex int, history []TrajectoryTurn, env *HermeticEnvironment) (TrajectoryTurn, bool, error) {
			switch turnIndex {
			case 0:
				return TrajectoryTurn{
					TurnIndex: 0,
					ToolCalls: []ToolCall{
						{
							ID:   "call-grep-1",
							Name: "Grep",
							Args: map[string]any{
								"path":        "server.log",
								"pattern":     "HTTP 404", // modified
								"max_results": 5,          // added
							},
							ReadOnly: true,
						},
					},
				}, false, nil
			case 1:
				return TrajectoryTurn{
					TurnIndex: 1,
					ToolCalls: []ToolCall{
						{
							ID:   "call-read-1",
							Name: "Read",
							Args: map[string]any{
								"filePath": "checkout.go",
								"offset":   200, // modified (was 100)
								// "limit" removed
							},
							ReadOnly: true,
						},
					},
				}, false, nil
			case 2:
				return baseline.Turns[2], true, nil
			default:
				return TrajectoryTurn{}, true, nil
			}
		}

		_, report, err := evaluator.Replay(ctx, baseline, ReplayConfig{
			AgentFunc: replayedAgent,
			MaxTurns:  5,
		})
		if err != nil {
			t.Fatalf("replay failed: %v", err)
		}

		if !report.Diverged {
			t.Fatal("expected diverged = true")
		}
		if len(report.ArgumentMutations) != 4 {
			t.Fatalf("expected 4 argument mutations, got %d: %+v", len(report.ArgumentMutations), report.ArgumentMutations)
		}

		mutationsByKey := make(map[string]ArgumentMutation)
		for _, m := range report.ArgumentMutations {
			mutationsByKey[m.ArgKey] = m
		}

		// Check "pattern" was modified
		if m, ok := mutationsByKey["pattern"]; !ok || m.MutationType != "modified" || m.ReplayedVal != "HTTP 404" {
			t.Fatalf("expected modified pattern, got %+v", m)
		}
		// Check "max_results" was added
		if m, ok := mutationsByKey["max_results"]; !ok || m.MutationType != "added" {
			t.Fatalf("expected added max_results, got %+v", m)
		}
		// Check "limit" was removed
		if m, ok := mutationsByKey["limit"]; !ok || m.MutationType != "removed" {
			t.Fatalf("expected removed limit, got %+v", m)
		}
		// Check "offset" was modified
		if m, ok := mutationsByKey["offset"]; !ok || m.MutationType != "modified" {
			t.Fatalf("expected modified offset, got %+v", m)
		}
	})

	t.Run("detects turn count delta on early exit and turn overrun", func(t *testing.T) {
		baseline := buildBaselineTrajectory()

		// 1. Early exit: candidate finishes in 1 turn instead of 3.
		candidateEarly := Trajectory{
			ID:    "cand-early",
			Turns: []TrajectoryTurn{baseline.Turns[0]},
		}
		repEarly := EvaluateDivergence(baseline, candidateEarly)
		if repEarly.TurnCountDelta != -2 {
			t.Fatalf("expected turn count delta = -2, got %d", repEarly.TurnCountDelta)
		}
		if !repEarly.Diverged || !repEarly.RegressionDetected {
			t.Fatal("expected divergence and regression on early termination")
		}

		// 2. Turn overrun: candidate takes 5 turns instead of 3.
		candidateOverrun := Trajectory{
			ID: "cand-overrun",
			Turns: append(baseline.Turns,
				TrajectoryTurn{
					TurnIndex: 3,
					ToolCalls: []ToolCall{{Name: "Grep", Args: map[string]any{"pattern": "extra"}}},
				},
				TrajectoryTurn{
					TurnIndex: 4,
					ToolCalls: []ToolCall{{Name: "Read", Args: map[string]any{"filePath": "extra.go"}}},
				},
			),
		}
		repOverrun := EvaluateDivergence(baseline, candidateOverrun)
		if repOverrun.TurnCountDelta != 2 {
			t.Fatalf("expected turn count delta = 2, got %d", repOverrun.TurnCountDelta)
		}
		if !repOverrun.Diverged || !repOverrun.RegressionDetected {
			t.Fatal("expected divergence and regression on turn overrun")
		}
	})

	t.Run("replaying against updated prompt detects behavior regression", func(t *testing.T) {
		baseline := buildBaselineTrajectory()
		env := NewHermeticEnvironment()
		env.RegisterTool("Grep", func(ctx context.Context, call ToolCall) (string, error) {
			return "grep result", nil
		})
		env.RegisterTool("Read", func(ctx context.Context, call ToolCall) (string, error) {
			return "read result", nil
		})

		evaluator := NewOfflineTrajectoryEvaluator(env)

		// Updated prompt directs candidate to look for a different error code.
		updatedPrompt := "Inspect server.log for error code 503 and examine handler"

		candidatePromptAgent := func(ctx context.Context, turnIndex int, history []TrajectoryTurn, env *HermeticEnvironment) (TrajectoryTurn, bool, error) {
			switch turnIndex {
			case 0:
				// Agent follows the updated prompt to search for 503
				pattern := "HTTP 500"
				if strings.Contains(updatedPrompt, "503") {
					pattern = "HTTP 503"
				}
				return TrajectoryTurn{
					TurnIndex: 0,
					Prompt:    updatedPrompt,
					ToolCalls: []ToolCall{
						{
							ID:       "call-grep-1",
							Name:     "Grep",
							Args:     map[string]any{"path": "server.log", "pattern": pattern},
							ReadOnly: true,
						},
					},
				}, false, nil
			case 1:
				return baseline.Turns[1], false, nil
			case 2:
				return baseline.Turns[2], true, nil
			default:
				return TrajectoryTurn{}, true, nil
			}
		}

		replayed, report, err := evaluator.Replay(ctx, baseline, ReplayConfig{
			UpdatedPrompt: updatedPrompt,
			AgentFunc:     candidatePromptAgent,
		})
		if err != nil {
			t.Fatalf("replay failed: %v", err)
		}

		if replayed.Prompt != updatedPrompt {
			t.Fatalf("expected replayed prompt %q, got %q", updatedPrompt, replayed.Prompt)
		}
		if !report.Diverged {
			t.Fatal("expected divergence when prompt update alters tool arguments")
		}
		if len(report.ArgumentMutations) != 1 {
			t.Fatalf("expected 1 argument mutation for altered pattern, got %d", len(report.ArgumentMutations))
		}
		if report.ArgumentMutations[0].ArgKey != "pattern" || report.ArgumentMutations[0].ReplayedVal != "HTTP 503" {
			t.Fatalf("expected pattern mutation to HTTP 503, got %+v", report.ArgumentMutations[0])
		}
	})

	t.Run("hermetic mock environment executes tools and records observations without external side effects", func(t *testing.T) {
		baseline := buildBaselineTrajectory()
		env := NewHermeticEnvironment()

		// Pre-populate recorded results from historical baseline trajectory
		env.LoadTrajectory(baseline)

		// Executing a tool matching recorded trajectory uses cached/mocked output
		res := env.ExecuteTool(ctx, ToolCall{
			ID:   "call-grep-1",
			Name: "Grep",
			Args: map[string]any{"path": "server.log", "pattern": "HTTP 500"},
		})
		if res.Error != "" {
			t.Fatalf("expected clean execution from recorded trajectory, got error: %s", res.Error)
		}
		if !strings.Contains(res.Output, "Internal Server Error") {
			t.Fatalf("unexpected output from recorded execution: %q", res.Output)
		}

		// Non-mocked tool returns deterministic hermetic error
		unmocked := env.ExecuteTool(ctx, ToolCall{
			Name: "UnregisteredTool",
			Args: map[string]any{"foo": "bar"},
		})
		if unmocked.Error == "" || !strings.Contains(unmocked.Error, "not mocked") {
			t.Fatalf("expected hermetic fallback error, got %+v", unmocked)
		}

		// Tool error handling
		env.RegisterTool("FailingTool", func(ctx context.Context, call ToolCall) (string, error) {
			return "", errors.New("simulated tool failure")
		})
		failRes := env.ExecuteTool(ctx, ToolCall{Name: "FailingTool"})
		if failRes.Error != "simulated tool failure" {
			t.Fatalf("expected tool failure captured, got %q", failRes.Error)
		}

		// State management in hermetic environment
		env.SetState("user_session", "session-xyz")
		val, ok := env.GetState("user_session")
		if !ok || val != "session-xyz" {
			t.Fatalf("expected state retrieval session-xyz, got %v (ok=%v)", val, ok)
		}

		// Dispatched calls are tracked
		calls := env.GetDispatchedCalls()
		if len(calls) < 3 {
			t.Fatalf("expected at least 3 dispatched calls tracked, got %d", len(calls))
		}
	})

	t.Run("trajectory and divergence report JSON serialization roundtrips cleanly", func(t *testing.T) {
		baseline := buildBaselineTrajectory()
		data, err := baseline.ToJSON()
		if err != nil {
			t.Fatalf("failed to serialize trajectory: %v", err)
		}

		restored, err := TrajectoryFromJSON(data)
		if err != nil {
			t.Fatalf("failed to deserialize trajectory: %v", err)
		}
		if restored.ID != baseline.ID || len(restored.Turns) != len(baseline.Turns) {
			t.Fatalf("roundtrip mismatch: restored %+v, want %+v", restored, baseline)
		}

		rep := EvaluateDivergence(baseline, baseline)
		repData, err := rep.ToJSON()
		if err != nil {
			t.Fatalf("failed to serialize divergence report: %v", err)
		}
		if len(repData) == 0 {
			t.Fatal("empty divergence report json")
		}
	})
}
