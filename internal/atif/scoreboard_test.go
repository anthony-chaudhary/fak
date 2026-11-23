package atif

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"testing"
)

func scoreFloat(f float64) *float64 { return &f }

func scoreStep(id int, ts string, model string, effort any) StepV17 {
	s := StepV17{StepID: id, Source: "agent", Message: "m", ReasoningEffort: effort}
	if ts != "" {
		s.Timestamp = v17str(ts)
	}
	if model != "" {
		s.ModelName = v17str(model)
	}
	return s
}

func scoreTraj(cost *float64, steps ...StepV17) HarborTrajectory {
	return HarborTrajectory{
		SchemaVersion: HarborVersion,
		Agent:         AgentV17{Name: "fak", Version: "0.9.0", ModelName: v17str("qwen3")},
		Steps:         steps,
		FinalMetrics:  &FinalMetricsV17{TotalCostUSD: cost},
	}
}

func TestScoreboardRow(t *testing.T) {
	t.Parallel()
	argv := []string{"harbor", "run", "--agent", "fak"}
	a := scoreTraj(scoreFloat(0.25),
		scoreStep(1, "2026-11-22T10:00:00Z", "", "high"),
		scoreStep(2, "2026-11-22T10:06:00Z", "", "high"))
	b := scoreTraj(scoreFloat(0.75),
		scoreStep(1, "2026-11-22T11:00:00.5Z", "", "high"),
		scoreStep(2, "2026-11-22T11:10:00.5Z", "", nil))

	row, err := ScoreboardRow([]HarborTrajectory{a, b}, argv)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256([]byte("harbor\nrun\n--agent\nfak"))
	want := Row{
		Harness:            "fak@0.9.0",
		Model:              "qwen3",
		ReasoningEffort:    "high",
		Tasks:              2,
		TotalCostUSD:       1.0,
		CostUSDPerTask:     0.5,
		WallMinutesTotal:   16,
		WallMinutesPerTask: 8,
		ArgvSHA256:         hex.EncodeToString(sum[:]),
	}
	if row.Harness != want.Harness || row.Model != want.Model || row.ReasoningEffort != want.ReasoningEffort ||
		row.Tasks != want.Tasks || row.ArgvSHA256 != want.ArgvSHA256 ||
		!scoreClose(row.TotalCostUSD, want.TotalCostUSD) || !scoreClose(row.CostUSDPerTask, want.CostUSDPerTask) ||
		!scoreClose(row.WallMinutesTotal, want.WallMinutesTotal) || !scoreClose(row.WallMinutesPerTask, want.WallMinutesPerTask) {
		t.Fatalf("row = %+v, want %+v", row, want)
	}
	if len(row.Argv) != len(argv) || &row.Argv[0] == &argv[0] {
		t.Fatalf("argv not copied: %v", row.Argv)
	}
}

func TestScoreboardRowStepModelAndNoEffort(t *testing.T) {
	t.Parallel()
	tr := scoreTraj(scoreFloat(0),
		scoreStep(1, "2026-11-22T10:00:00Z", "gpt-x", nil),
		scoreStep(2, "2026-11-22T10:01:00Z", "gpt-x", nil))
	tr.Agent.ModelName = nil
	row, err := ScoreboardRow([]HarborTrajectory{tr}, []string{"run"})
	if err != nil {
		t.Fatal(err)
	}
	if row.Model != "gpt-x" || row.ReasoningEffort != "" || row.TotalCostUSD != 0 || !scoreClose(row.WallMinutesTotal, 1) {
		t.Fatalf("row = %+v", row)
	}
}

func TestScoreboardRowRefusals(t *testing.T) {
	t.Parallel()
	ok := func() HarborTrajectory {
		return scoreTraj(scoreFloat(0.1),
			scoreStep(1, "2026-11-22T10:00:00Z", "", "low"),
			scoreStep(2, "2026-11-22T10:01:00Z", "", "low"))
	}
	otherVersion := ok()
	otherVersion.Agent.Version = "1.0.0"
	otherModel := ok()
	otherModel.Agent.ModelName = v17str("llama")
	otherEffort := ok()
	otherEffort.Steps[0].ReasoningEffort = "high"
	otherEffort.Steps[1].ReasoningEffort = "high"
	splitStepModel := ok()
	splitStepModel.Agent.ModelName = nil
	splitStepModel.Steps[0].ModelName = v17str("a")
	splitStepModel.Steps[1].ModelName = v17str("b")
	nilMetrics := ok()
	nilMetrics.FinalMetrics = nil
	nilCost := ok()
	nilCost.FinalMetrics.TotalCostUSD = nil
	nanCost := ok()
	nanCost.FinalMetrics.TotalCostUSD = scoreFloat(math.NaN())
	noTimestamp := ok()
	noTimestamp.Steps[1].Timestamp = nil
	badTimestamp := ok()
	badTimestamp.Steps[0].Timestamp = v17str("yesterday")
	noSteps := ok()
	noSteps.Steps = nil

	argv := []string{"run"}
	cases := []struct {
		name  string
		trajs []HarborTrajectory
		argv  []string
		want  error
	}{
		{"empty input", nil, argv, ErrScoreboardEmpty},
		{"empty argv", []HarborTrajectory{ok()}, nil, ErrScoreboardEmptyArgv},
		{"mixed harness", []HarborTrajectory{ok(), otherVersion}, argv, ErrScoreboardMixed},
		{"mixed model", []HarborTrajectory{ok(), otherModel}, argv, ErrScoreboardMixed},
		{"mixed effort", []HarborTrajectory{ok(), otherEffort}, argv, ErrScoreboardMixed},
		{"mixed step model", []HarborTrajectory{splitStepModel}, argv, ErrScoreboardMixed},
		{"missing final metrics", []HarborTrajectory{ok(), nilMetrics}, argv, ErrScoreboardMissingCost},
		{"missing total cost", []HarborTrajectory{nilCost}, argv, ErrScoreboardMissingCost},
		{"NaN total cost", []HarborTrajectory{nanCost}, argv, ErrScoreboardMissingCost},
		{"missing timestamp", []HarborTrajectory{noTimestamp}, argv, ErrScoreboardBadTimestamp},
		{"unparseable timestamp", []HarborTrajectory{badTimestamp}, argv, ErrScoreboardBadTimestamp},
		{"no steps", []HarborTrajectory{noSteps}, argv, ErrScoreboardBadTimestamp},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			row, err := ScoreboardRow(tc.trajs, tc.argv)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
			if row.Tasks != 0 || row.ArgvSHA256 != "" {
				t.Fatalf("refusal leaked a partial row: %+v", row)
			}
		})
	}
}

func scoreClose(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
