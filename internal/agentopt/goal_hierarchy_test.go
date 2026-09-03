package agentopt

import (
	"testing"
)

func TestGoalHierarchyValidator(t *testing.T) {
	t.Run("valid diamond DAG with disjoint parallel tasks", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "root",
				Description: "Initialize shared interfaces",
				DependsOn:   nil,
				TargetPaths: []string{"pkg/api/types.go"},
				MaxTurns:    5,
			},
			{
				ID:          "branch-left",
				Description: "Implement left component",
				DependsOn:   []string{"root"},
				TargetPaths: []string{"pkg/left/worker.go"},
				MaxTurns:    10,
			},
			{
				ID:          "branch-right",
				Description: "Implement right component",
				DependsOn:   []string{"root"},
				TargetPaths: []string{"pkg/right/worker.go"},
				MaxTurns:    15,
			},
			{
				ID:          "aggregator",
				Description: "Integrate components",
				DependsOn:   []string{"branch-left", "branch-right"},
				TargetPaths: []string{"pkg/integrate/main.go"},
				MaxTurns:    8,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if !report.Valid {
			t.Fatalf("expected valid DAG report, got violations: %v", report.Violations)
		}
		if report.CyclesDetected {
			t.Fatalf("unexpected cycles detected: %v", report.CyclePaths)
		}
		if len(report.ScopeOverlaps) != 0 {
			t.Fatalf("expected no scope overlaps, got %d", len(report.ScopeOverlaps))
		}
		if len(report.ParallelStages) != 3 {
			t.Fatalf("expected 3 parallel stages, got %d (%v)", len(report.ParallelStages), report.ParallelStages)
		}
		if len(report.ExecutionOrder) != 4 {
			t.Fatalf("expected 4 tasks in execution order, got %d", len(report.ExecutionOrder))
		}
		// Critical path: root (5) + branch-right (15) + aggregator (8) = 28 turns
		if report.CriticalPathTurns != 28 {
			t.Fatalf("expected 28 critical path turns, got %d", report.CriticalPathTurns)
		}
		if report.TotalTurnBudget != 38 {
			t.Fatalf("expected 38 total turn budget, got %d", report.TotalTurnBudget)
		}
	})

	t.Run("cycle detection: direct two-node mutual dependency", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "task-1",
				Description: "Task one",
				DependsOn:   []string{"task-2"},
				TargetPaths: []string{"pkg/a/a.go"},
				MaxTurns:    5,
			},
			{
				ID:          "task-2",
				Description: "Task two",
				DependsOn:   []string{"task-1"},
				TargetPaths: []string{"pkg/b/b.go"},
				MaxTurns:    5,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected cycle validation failure, got valid report")
		}
		if !report.CyclesDetected {
			t.Fatal("expected CyclesDetected to be true")
		}
		if len(report.CyclePaths) == 0 {
			t.Fatal("expected non-empty CyclePaths")
		}
	})

	t.Run("cycle detection: three-node transitive cycle", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "step-a",
				Description: "Step A",
				DependsOn:   []string{"step-c"},
				TargetPaths: []string{"pkg/a.go"},
				MaxTurns:    5,
			},
			{
				ID:          "step-b",
				Description: "Step B",
				DependsOn:   []string{"step-a"},
				TargetPaths: []string{"pkg/b.go"},
				MaxTurns:    5,
			},
			{
				ID:          "step-c",
				Description: "Step C",
				DependsOn:   []string{"step-b"},
				TargetPaths: []string{"pkg/c.go"},
				MaxTurns:    5,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected cycle failure, got valid")
		}
		if !report.CyclesDetected {
			t.Fatal("expected CyclesDetected == true")
		}
	})

	t.Run("cycle detection: self-referential loop", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "loop-task",
				Description: "Self referential",
				DependsOn:   []string{"loop-task"},
				TargetPaths: []string{"pkg/self.go"},
				MaxTurns:    4,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected failure on self-referential task")
		}
		if !report.CyclesDetected {
			t.Fatal("expected CyclesDetected to be true")
		}
	})

	t.Run("scope disjointness: parallel tasks with exact path collision", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "subtask-alpha",
				Description: "Alpha modification",
				DependsOn:   nil,
				TargetPaths: []string{"internal/core/auth.go"},
				MaxTurns:    5,
			},
			{
				ID:          "subtask-beta",
				Description: "Beta modification",
				DependsOn:   nil,
				TargetPaths: []string{"internal/core/auth.go"},
				MaxTurns:    5,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected scope collision failure, got valid")
		}
		if len(report.ScopeOverlaps) == 0 {
			t.Fatal("expected ScopeOverlaps to contain conflict")
		}
		overlap := report.ScopeOverlaps[0]
		if overlap.TaskA != "subtask-alpha" || overlap.TaskB != "subtask-beta" {
			t.Fatalf("unexpected tasks in overlap: %+v", overlap)
		}
	})

	t.Run("scope disjointness: parallel tasks with prefix directory overlap", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "tree-writer",
				Description: "Bulk tree write",
				DependsOn:   nil,
				TargetPaths: []string{"internal/storage/**"},
				MaxTurns:    10,
			},
			{
				ID:          "file-writer",
				Description: "Single file write",
				DependsOn:   nil,
				TargetPaths: []string{"internal/storage/sqlite/db.go"},
				MaxTurns:    10,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected prefix overlap failure, got valid")
		}
		if len(report.ScopeOverlaps) != 1 {
			t.Fatalf("expected 1 scope overlap, got %d", len(report.ScopeOverlaps))
		}
	})

	t.Run("scope disjointness: sequential tasks with same target path allowed", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "producer",
				Description: "Generate artifact",
				DependsOn:   nil,
				TargetPaths: []string{"build/output.json"},
				MaxTurns:    5,
			},
			{
				ID:          "consumer",
				Description: "Format artifact",
				DependsOn:   []string{"producer"},
				TargetPaths: []string{"build/output.json"},
				MaxTurns:    5,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if !report.Valid {
			t.Fatalf("sequential tasks should be permitted on shared paths, got violations: %v", report.Violations)
		}
		if len(report.ScopeOverlaps) != 0 {
			t.Fatalf("expected 0 scope overlaps for sequential dependency, got %d", len(report.ScopeOverlaps))
		}
	})

	t.Run("atomicity checks: empty ID and empty description", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "",
				Description: "Has no ID",
				MaxTurns:    5,
			},
			{
				ID:          "task-no-desc",
				Description: "   ",
				MaxTurns:    5,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected atomic violations for empty ID and description")
		}
		if len(report.AtomicViolations) < 2 {
			t.Fatalf("expected at least 2 atomic violations, got %d (%v)", len(report.AtomicViolations), report.AtomicViolations)
		}
	})

	t.Run("atomicity checks: duplicate ID and unknown dependency", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "same-id",
				Description: "First instance",
				MaxTurns:    5,
			},
			{
				ID:          "same-id",
				Description: "Second instance",
				MaxTurns:    5,
			},
			{
				ID:          "dependent-task",
				Description: "Depends on missing task",
				DependsOn:   []string{"non-existent-id"},
				MaxTurns:    5,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected duplicate ID and unknown dependency failures")
		}
	})

	t.Run("turn budget bounds: subtask turn limit exceeded", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "heavy-task",
				Description: "Requires too many turns",
				MaxTurns:    80,
			},
		}

		validator := &GoalHierarchyValidator{
			MaxTurnsPerSubtask: 30,
		}
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected subtask turn budget violation")
		}
		if len(report.TurnBudgetViolations) == 0 {
			t.Fatal("expected TurnBudgetViolations to record excess turns")
		}
	})

	t.Run("turn budget bounds: non-positive turns rejected", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "zero-turn-task",
				Description: "Zero turns allocated",
				MaxTurns:    0,
			},
			{
				ID:          "negative-turn-task",
				Description: "Negative turns allocated",
				MaxTurns:    -3,
			},
		}

		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected failure for non-positive turn budgets")
		}
		if len(report.TurnBudgetViolations) != 2 {
			t.Fatalf("expected 2 turn violations, got %d", len(report.TurnBudgetViolations))
		}
	})

	t.Run("turn budget bounds: cumulative budget exceeded", func(t *testing.T) {
		nodes := []GoalNode{
			{
				ID:          "task-1",
				Description: "First chunk",
				MaxTurns:    15,
			},
			{
				ID:          "task-2",
				Description: "Second chunk",
				MaxTurns:    15,
			},
		}

		validator := &GoalHierarchyValidator{
			MaxTurnsPerSubtask: 20,
			MaxTotalTurns:      25,
		}
		report := validator.ValidateGoalDAG(nodes)

		if report.Valid {
			t.Fatal("expected total turn budget violation")
		}
		if !report.TurnBudgetExceeded {
			t.Fatal("expected TurnBudgetExceeded == true")
		}
	})

	t.Run("empty nodes returns clean report", func(t *testing.T) {
		validator := NewGoalHierarchyValidator()
		report := validator.ValidateGoalDAG(nil)

		if !report.Valid {
			t.Fatalf("expected empty nodes to be valid, got violations: %v", report.Violations)
		}
		if len(report.Subtasks) != 0 || len(report.ExecutionOrder) != 0 {
			t.Fatal("expected empty subtasks and execution order")
		}
	})
}
