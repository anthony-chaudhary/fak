package atif

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

// Scoreboard refusals. Each is distinct so a caller can tell why a run set
// cannot be summarized; none degrade to a zero value.
var (
	ErrScoreboardEmpty        = errors.New("atif: scoreboard: no trajectories")
	ErrScoreboardMixed        = errors.New("atif: scoreboard: mixed harness, model, or reasoning effort")
	ErrScoreboardMissingCost  = errors.New("atif: scoreboard: final_metrics.total_cost_usd missing")
	ErrScoreboardBadTimestamp = errors.New("atif: scoreboard: step timestamp missing or unparseable")
	ErrScoreboardEmptyArgv    = errors.New("atif: scoreboard: argv is empty")
)

// Row is one scoreboard line summarizing a homogeneous set of Harbor
// trajectories: one harness, one model, one reasoning effort.
type Row struct {
	Harness string `json:"harness"` // agent name@version
	Model   string `json:"model"`
	// ReasoningEffort is empty when no step in the set records one.
	ReasoningEffort    string   `json:"reasoning_effort"`
	Tasks              int      `json:"tasks"`
	TotalCostUSD       float64  `json:"total_cost_usd"`
	CostUSDPerTask     float64  `json:"cost_usd_per_task"`
	WallMinutesTotal   float64  `json:"wall_minutes_total"`
	WallMinutesPerTask float64  `json:"wall_minutes_per_task"`
	Argv               []string `json:"argv"`
	ArgvSHA256         string   `json:"argv_sha256"`
}

// ScoreboardRow folds root trajectories (one per task) into a scoreboard row.
// Subagent trajectories are not counted as tasks; their cost is expected to be
// rolled into the root's final_metrics. A task's wall time spans its earliest
// to latest step timestamp.
func ScoreboardRow(trajs []HarborTrajectory, argv []string) (Row, error) {
	if len(trajs) == 0 {
		return Row{}, ErrScoreboardEmpty
	}
	if len(argv) == 0 {
		return Row{}, ErrScoreboardEmptyArgv
	}
	var row Row
	var wall time.Duration
	for i, t := range trajs {
		harness := t.Agent.Name + "@" + t.Agent.Version
		model, err := trajectoryModel(t)
		if err != nil {
			return Row{}, fmt.Errorf("trajectories[%d]: %w", i, err)
		}
		effort, err := trajectoryEffort(t)
		if err != nil {
			return Row{}, fmt.Errorf("trajectories[%d]: %w", i, err)
		}
		if i == 0 {
			row.Harness, row.Model, row.ReasoningEffort = harness, model, effort
		} else if harness != row.Harness || model != row.Model || effort != row.ReasoningEffort {
			return Row{}, fmt.Errorf("trajectories[%d] %s/%s/%q vs %s/%s/%q: %w",
				i, harness, model, effort, row.Harness, row.Model, row.ReasoningEffort, ErrScoreboardMixed)
		}
		if t.FinalMetrics == nil || t.FinalMetrics.TotalCostUSD == nil ||
			math.IsNaN(*t.FinalMetrics.TotalCostUSD) || math.IsInf(*t.FinalMetrics.TotalCostUSD, 0) {
			return Row{}, fmt.Errorf("trajectories[%d]: %w", i, ErrScoreboardMissingCost)
		}
		row.TotalCostUSD += *t.FinalMetrics.TotalCostUSD
		span, err := trajectoryWall(t)
		if err != nil {
			return Row{}, fmt.Errorf("trajectories[%d]: %w", i, err)
		}
		wall += span
	}
	row.Tasks = len(trajs)
	row.CostUSDPerTask = row.TotalCostUSD / float64(row.Tasks)
	row.WallMinutesTotal = wall.Minutes()
	row.WallMinutesPerTask = row.WallMinutesTotal / float64(row.Tasks)
	row.Argv = append([]string(nil), argv...)
	sum := sha256.Sum256([]byte(strings.Join(argv, "\n")))
	row.ArgvSHA256 = hex.EncodeToString(sum[:])
	return row, nil
}

// trajectoryModel prefers agent.model_name and otherwise requires every step
// that names a model to agree.
func trajectoryModel(t HarborTrajectory) (string, error) {
	if t.Agent.ModelName != nil {
		return *t.Agent.ModelName, nil
	}
	model := ""
	for _, s := range t.Steps {
		if s.ModelName == nil || *s.ModelName == "" {
			continue
		}
		if model != "" && *s.ModelName != model {
			return "", fmt.Errorf("steps name %q and %q: %w", model, *s.ModelName, ErrScoreboardMixed)
		}
		model = *s.ModelName
	}
	return model, nil
}

func trajectoryEffort(t HarborTrajectory) (string, error) {
	effort := ""
	for _, s := range t.Steps {
		if s.ReasoningEffort == nil {
			continue
		}
		e := fmt.Sprint(s.ReasoningEffort)
		if e == "" {
			continue
		}
		if effort != "" && e != effort {
			return "", fmt.Errorf("steps carry effort %q and %q: %w", effort, e, ErrScoreboardMixed)
		}
		effort = e
	}
	return effort, nil
}

func trajectoryWall(t HarborTrajectory) (time.Duration, error) {
	if len(t.Steps) == 0 {
		return 0, fmt.Errorf("no steps: %w", ErrScoreboardBadTimestamp)
	}
	var first, last time.Time
	for i, s := range t.Steps {
		if s.Timestamp == nil {
			return 0, fmt.Errorf("steps[%d]: %w", i, ErrScoreboardBadTimestamp)
		}
		ts, err := time.Parse(time.RFC3339Nano, *s.Timestamp)
		if err != nil {
			return 0, fmt.Errorf("steps[%d] %q: %w", i, *s.Timestamp, ErrScoreboardBadTimestamp)
		}
		if i == 0 || ts.Before(first) {
			first = ts
		}
		if i == 0 || ts.After(last) {
			last = ts
		}
	}
	return last.Sub(first), nil
}
