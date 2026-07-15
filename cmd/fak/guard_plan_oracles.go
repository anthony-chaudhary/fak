package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/operatorquestion"
	"github.com/anthony-chaudhary/fak/internal/planresolve"
)

// planCommandRunner is the read-only process seam used by the production plan
// oracles. Tests replace it; no oracle mutates leases, the index, or the worktree.
type planCommandRunner interface {
	Run(context.Context, string, ...string) ([]byte, error)
}

type execPlanCommandRunner struct{}

func (execPlanCommandRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).CombinedOutput()
}

type guardPlanOracleSet struct {
	runner planCommandRunner
}

func resolveGuardOperatorQuestionPlan(ctx context.Context, q operatorquestion.OperatorQuestion) (planresolve.Verdict, error) {
	return planresolve.Resolve(ctx, q, guardPlanOracleSet{runner: execPlanCommandRunner{}})
}

func (o guardPlanOracleSet) TreeDisjoint(ctx context.Context, tree []string) (planresolve.OracleResult, error) {
	tree = cleanPlanTree(tree)
	if len(tree) == 0 {
		return planresolve.OracleResult{Reason: planresolve.ReasonTreeCollision}, fmt.Errorf("plan tree is empty")
	}
	args := []string{"arbitrate", "--lane", "plan-approval", "--kind", "keyword", "--mode", "exclusive", "--tree"}
	args = append(args, tree...)
	args = append(args, "--output", "json")
	out, err := o.runner.Run(ctx, "dos", args...)
	var row struct {
		Outcome string `json:"outcome"`
		Reason  string `json:"reason"`
	}
	if decodeErr := json.Unmarshal(out, &row); decodeErr != nil {
		if err != nil {
			return planresolve.OracleResult{}, fmt.Errorf("dos arbitrate: %w: %s", err, strings.TrimSpace(string(out)))
		}
		return planresolve.OracleResult{}, fmt.Errorf("decode dos arbitrate: %w", decodeErr)
	}
	witness := "dos arbitrate: " + strings.TrimSpace(row.Reason)
	switch strings.ToLower(strings.TrimSpace(row.Outcome)) {
	case "acquire":
		return planresolve.OracleResult{OK: true, Witness: witness}, nil
	case "refuse":
		return planresolve.OracleResult{OK: false, Reason: planresolve.ReasonTreeCollision, Witness: witness}, nil
	default:
		return planresolve.OracleResult{}, fmt.Errorf("dos arbitrate returned outcome %q", row.Outcome)
	}
}

func (o guardPlanOracleSet) DirectionAllowed(ctx context.Context, tree []string) (planresolve.OracleResult, error) {
	tree = cleanPlanTree(tree)
	for _, path := range tree {
		slash := strings.TrimPrefix(filepath.ToSlash(path), "./")
		if slash == "internal/abi" || strings.HasPrefix(slash, "internal/abi/") {
			return planresolve.OracleResult{
				OK: false, Reason: planresolve.ReasonWrongDirection,
				Witness: "core-lock: internal/abi is frozen and human-owned",
			}, nil
		}
	}
	// architest is the repository's executable direction oracle. Run the committed
	// architecture checks read-only; a red means the direction cannot auto-approve.
	out, err := o.runner.Run(ctx, "go", "test", "./internal/architest", "-run", "Test(NoUpwardImports|PureRootLeavesStayPure|KernelImportsOnlyABI)$", "-count=1")
	if err != nil {
		return planresolve.OracleResult{OK: false, Reason: planresolve.ReasonWrongDirection, Witness: "architest: " + boundedOracleOutput(out)}, nil
	}
	return planresolve.OracleResult{OK: true, Witness: "architest direction checks: PASS"}, nil
}

func (o guardPlanOracleSet) DoneVerifiable(ctx context.Context, criterion string) (planresolve.OracleResult, error) {
	criterion = strings.TrimSpace(criterion)
	if criterion == "" {
		return planresolve.OracleResult{OK: false, Reason: planresolve.ReasonDoneUnverifiable, Witness: "done criterion is empty"}, nil
	}
	// A criterion is auto-approvable only when it names a registered executable
	// witness form. This is structural validation before execution; dos verify remains
	// the shipped-phase read-back once a plan/phase pair exists.
	lower := strings.ToLower(criterion)
	registered := []string{"go test ", "fak buildcheck", "fak ci-preflight", "dos verify ", "dos commit-audit ", "-selfcheck"}
	for _, marker := range registered {
		if strings.Contains(lower, marker) {
			return planresolve.OracleResult{OK: true, Witness: "registered done witness: " + marker}, nil
		}
	}
	return planresolve.OracleResult{OK: false, Reason: planresolve.ReasonDoneUnverifiable, Witness: "no registered executable witness in done criterion"}, nil
}

func cleanPlanTree(tree []string) []string {
	out := make([]string, 0, len(tree))
	seen := map[string]bool{}
	for _, path := range tree {
		path = strings.TrimSpace(path)
		if path == "" || seen[path] {
			continue
		}
		seen[path] = true
		out = append(out, path)
	}
	return out
}

func boundedOracleOutput(out []byte) string {
	value := strings.TrimSpace(string(out))
	const limit = 400
	if len(value) > limit {
		value = value[len(value)-limit:]
	}
	if value == "" {
		return "FAIL"
	}
	return value
}

var _ planresolve.OracleSet = guardPlanOracleSet{}
var _ = os.DevNull
