package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopindex"
)

func TestLoopIndexIssue1155GreenGateBudgetFeedsS0(t *testing.T) {
	rep := loopindex.Score(collectLoopIndex(repoRoot()))
	ship := rep.StageDetail[4]
	if ship.Name != loopindex.StageShip {
		t.Fatalf("stage[4] = %q, want ship", ship.Name)
	}

	probes := map[string]bool{}
	for _, p := range ship.Probes {
		probes[p.Name] = p.Pass
	}
	for _, name := range []string{
		"commit_verb",
		"green_gate_budget",
	} {
		if !probes[name] {
			t.Fatalf("ship probe %s = false; issue #1155 is not witnessed by the tree", name)
		}
	}

	var shipKPI loopindex.KPI
	for _, k := range rep.KPIs {
		if k.Name == loopindex.StageShip {
			shipKPI = k
			break
		}
	}
	if !shipKPI.Wired || shipKPI.Debt != 0 {
		t.Fatalf("ship KPI = %+v, want wired with no debt for #1155", shipKPI)
	}

	root := repoRoot()
	makefile := readText(t, filepath.Join(root, "Makefile"))
	for _, want := range []string{"VERIFY_LOOP_BUDGET ?= 30s", "fak affected --budget $(VERIFY_LOOP_BUDGET)"} {
		if !strings.Contains(makefile, want) {
			t.Fatalf("Makefile missing %q; verify-loop latency is not a tracked hard budget", want)
		}
	}
	doc := readText(t, filepath.Join(root, "docs", "fak", "green-gate-budget.md"))
	for _, want := range []string{"incremental warm", "incremental cold", "full warm", "full cold", "GATE_LATENCY_REGRESSION"} {
		if !strings.Contains(doc, want) {
			t.Fatalf("green-gate budget doc missing %q", want)
		}
	}
}

func readText(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
