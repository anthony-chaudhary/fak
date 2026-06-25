package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

func TestLadder(t *testing.T) {
	// Always includes 1 and max, strictly increasing.
	got := ladder(16)
	if got[0] != 1 || got[len(got)-1] != 16 {
		t.Fatalf("ladder(16) endpoints = %v", got)
	}
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Fatalf("ladder not strictly increasing: %v", got)
		}
	}
	// A non-power-of-two max is appended so the frontier width itself is always searched.
	if got := ladder(12); got[len(got)-1] != 12 {
		t.Fatalf("ladder(12) must include the max: %v", got)
	}
	if got := ladder(0); !reflect.DeepEqual(got, []int{1}) {
		t.Fatalf("ladder(0) = %v, want [1]", got)
	}
}

func TestParseGrid(t *testing.T) {
	if got := parseGrid("1,4,16"); !reflect.DeepEqual(got, []int{1, 4, 16}) {
		t.Fatalf("parseGrid = %v", got)
	}
	if got := parseGrid(" 2 , 8 ,"); !reflect.DeepEqual(got, []int{2, 8}) {
		t.Fatalf("parseGrid skips empties/trims = %v", got)
	}
	if got := parseGrid(""); got != nil {
		t.Fatalf("empty parseGrid = %v, want nil", got)
	}
}

func TestProfileByName(t *testing.T) {
	for _, name := range []string{"research", "write-heavy", "no-share", "WH", "control"} {
		if _, ok := profileByName(name); !ok {
			t.Errorf("profileByName(%q) not resolved", name)
		}
	}
	if _, ok := profileByName("nope"); ok {
		t.Errorf("profileByName(nope) unexpectedly resolved")
	}
}

// TestTopobenchEndToEnd drives the real CLI path: main() runs the #541 fleet-topology search
// and writes the JSON + CSV artifacts. It then re-parses the JSON report and asserts the honest
// invariants — ZERO model calls, a non-empty Pareto frontier, and a best topology that strictly
// improves credited savings over the hand-frozen baseline — plus that the CSV carries its grid
// header. This is the end-to-end witness that the search is invokable and emits its artifact.
func TestTopobenchEndToEnd(t *testing.T) {
	dir := t.TempDir()
	oldArgs := os.Args
	oldWd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { os.Args = oldArgs; _ = os.Chdir(oldWd) }()
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}

	os.Args = []string{"topobench",
		"--profile", "research",
		"--frontier-width", "8",
		"--baseline-width", "4", "--baseline-lanes", "1", "--baseline-sub-turns", "4",
		"--named-topology", "star", "--named-width", "4",
		"--widths", "2,4,8", "--lanes", "1,4,8",
		"--trials", "4", "--seed", "5",
		"--out", "topo.json", "--csv", "topo.csv",
	}
	main()

	jb, err := os.ReadFile(filepath.Join(dir, "topo.json"))
	if err != nil {
		t.Fatalf("json artifact not written: %v", err)
	}
	var rep turnbench.TopologySearchReport
	if err := json.Unmarshal(jb, &rep); err != nil {
		t.Fatalf("unmarshal report: %v", err)
	}
	if rep.ModelCallsSpent != 0 {
		t.Errorf("topology search must spend ZERO model calls, got %d", rep.ModelCallsSpent)
	}
	if rep.NamedTopology == nil {
		t.Fatalf("named topology score missing from report")
	}
	if rep.NamedTopology.Name != "named-star" || rep.NamedTopology.Summary["declared_edges"] != "3" {
		t.Errorf("named topology score = %+v", rep.NamedTopology)
	}
	if len(rep.Frontier) == 0 {
		t.Errorf("report must surface a non-empty Pareto frontier")
	}
	if rep.Best.Fitness.CreditedSavingsTokens <= rep.Baseline.Fitness.CreditedSavingsTokens {
		t.Errorf("search must IMPROVE credited savings over the baseline (%d), got best=%d",
			rep.Baseline.Fitness.CreditedSavingsTokens, rep.Best.Fitness.CreditedSavingsTokens)
	}

	cb, err := os.ReadFile(filepath.Join(dir, "topo.csv"))
	if err != nil {
		t.Fatalf("csv artifact not written: %v", err)
	}
	if !strings.Contains(string(cb), "credited_savings_tokens") {
		t.Errorf("csv artifact missing its grid header: %s", string(cb))
	}
}
