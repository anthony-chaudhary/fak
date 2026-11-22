package climb_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/climb"
)

const modulePath = "github.com/anthony-chaudhary/fak"

func quietJournal(t *testing.T) *climb.Journal {
	t.Helper()
	j, err := climb.NewJournal(filepath.Join(t.TempDir(), "journal.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { j.Close() })
	return j
}

// A loop author supplies proposals and measurements but never the verdict: the
// engine keeps only an improving candidate with a green suite and a clean tree.
func TestRunKeepsOnlyWitnessedGains(t *testing.T) {
	type try struct {
		metric       float64
		green, clean bool
	}
	h := climb.Harness{
		MetricName:     "latency_ms",
		LowerBetter:    true,
		BaselineMetric: func() (float64, string, error) { return 100, "base", nil },
		Candidates: func() []climb.Candidate {
			return []climb.Candidate{
				{Label: "better", Payload: try{90, true, true}},
				{Label: "worse", Payload: try{95, true, true}},
				{Label: "better but suite red", Payload: try{50, false, true}},
				{Label: "better but tree dirty", Payload: try{50, true, false}},
				{Label: "better again", Payload: try{80, true, true}},
			}
		},
		Measure: func(c climb.Candidate) (climb.Measurement, error) {
			p := c.Payload.(try)
			return climb.Measurement{Metric: p.metric, SuiteGreen: p.green, TruthClean: p.clean}, nil
		},
	}

	res, err := climb.Run(h, quietJournal(t), 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Kept != 2 {
		t.Errorf("kept %d candidates, want 2", res.Kept)
	}
	if res.FinalBaseline != 80 {
		t.Errorf("final baseline %v, want 80: a kept candidate must become the bar for the next", res.FinalBaseline)
	}
	var kept []string
	for _, row := range res.Rows {
		if row.Kept {
			kept = append(kept, row.Candidate)
		}
	}
	if got := strings.Join(kept, ","); got != "better,better again" {
		t.Errorf("kept %q, want %q", got, "better,better again")
	}
}

func TestEvaluateIsTheSameKeepBit(t *testing.T) {
	d, w := climb.Evaluate(climb.Witness{Class: climb.ClassFull, Before: 10, After: 12, SuiteGreen: true, TruthClean: true})
	if d != climb.KEEP || !w.Kept() {
		t.Errorf("an improving, green, clean witness must be kept; got %v", d)
	}
	d, _ = climb.Evaluate(climb.Witness{Class: climb.ClassFull, Before: 10, After: 12, SuiteGreen: false, TruthClean: true})
	if d != climb.REVERT {
		t.Errorf("a red suite must revert; got %v", d)
	}
}

// TestLoopFromForeignModule builds and runs a hill-climb under a module path
// outside the fak prefix, where Go's internal/ rule applies as it does to a
// third party.
func TestLoopFromForeignModule(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns the go toolchain")
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH")
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}

	// The probe is copied out so the build never writes into the source tree.
	dir := t.TempDir()
	src, err := os.ReadFile(filepath.Join("testdata", "climbprobe", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), src, 0o644); err != nil {
		t.Fatal(err)
	}
	gomod := "module example.com/fakclimbprobe\n\ngo 1.26.0\n\n" +
		"require " + modulePath + " v0.0.0\n\n" +
		"replace " + modulePath + " => " + filepath.ToSlash(root) + "\n"
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(gomod), 0o644); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command(goBin, "run", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-mod=mod")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("foreign-module loop failed: %v\n%s", err, out)
	}
	if !strings.Contains(string(out), "climbprobe: kept 3 of 5 candidates") {
		t.Fatalf("unexpected probe output:\n%s", out)
	}
}
