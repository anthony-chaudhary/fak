// Command climbprobe is a hill-climb written entirely outside fak: its own
// proposer, its own metric, and fak's engine deciding what is kept. It imports
// only pkg/climb.
//
// The search is a coordinate walk toward the minimum of (x-7)^2. Two of the
// proposals are deliberately bad: one is worse than the baseline and one
// improves the metric with a red suite. The engine must revert both.
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/pkg/climb"
)

type proposal struct {
	x     float64
	green bool
}

func cost(x float64) float64 { return (x - 7) * (x - 7) }

func main() {
	dir, err := os.MkdirTemp("", "climbprobe-")
	if err != nil {
		fail(err)
	}
	journal, err := climb.NewJournal(filepath.Join(dir, "journal.jsonl"))
	if err != nil {
		os.RemoveAll(dir)
		fail(err)
	}
	cleanup = func() {
		journal.Close()
		os.RemoveAll(dir)
	}
	defer cleanup()

	h := climb.Harness{
		MetricName:      "squared_error",
		LowerBetter:     true,
		BaselineRefName: "start",
		BaselineMetric:  func() (float64, string, error) { return cost(0), "x=0", nil },
		Candidates: func() []climb.Candidate {
			return []climb.Candidate{
				{Label: "step to 4", Payload: proposal{x: 4, green: true}},
				{Label: "step back to 1", Payload: proposal{x: 1, green: true}},
				{Label: "step to 6, suite red", Payload: proposal{x: 6, green: false}},
				{Label: "step to 6", Payload: proposal{x: 6, green: true}},
				{Label: "step to 7", Payload: proposal{x: 7, green: true}},
			}
		},
		Measure: func(c climb.Candidate) (climb.Measurement, error) {
			p := c.Payload.(proposal)
			return climb.Measurement{Metric: cost(p.x), SuiteGreen: p.green, TruthClean: true}, nil
		},
	}

	res, err := climb.Run(h, journal, 10, 0)
	if err != nil {
		fail(err)
	}
	if res.Kept != 3 || res.FinalBaseline != 0 || res.Cycles != 5 {
		fail(fmt.Errorf("want 3 kept of 5 and a final baseline of 0, got kept=%d cycles=%d baseline=%v",
			res.Kept, res.Cycles, res.FinalBaseline))
	}
	fmt.Printf("climbprobe: kept %d of %d candidates, baseline %v -> %v\n", res.Kept, res.Cycles, cost(0), res.FinalBaseline)
}

// cleanup removes the journal; fail calls it because os.Exit skips defers.
var cleanup = func() {}

func fail(err error) {
	cleanup()
	fmt.Fprintln(os.Stderr, "climbprobe:", err)
	os.Exit(1)
}
