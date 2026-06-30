package turnbench

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionreset"
)

func TestManagedContextSoakManyResetsReportsResetCountAndBounds(t *testing.T) {
	cfg := DefaultManagedContextSoakConfig()
	rep := RunManagedContextSoak(cfg)

	if !rep.OK() {
		t.Fatalf("managed-context soak failed: continuity=%v budget=%v query=%v",
			rep.ContinuityFailures, rep.BudgetViolations, rep.QueryFailures)
	}
	if rep.Schema != ManagedContextSoakVersion {
		t.Fatalf("schema = %q, want %q", rep.Schema, ManagedContextSoakVersion)
	}
	if rep.ResetCount != cfg.ResetCycles {
		t.Fatalf("reset_count = %d, want %d", rep.ResetCount, cfg.ResetCycles)
	}
	if rep.WarmPrefixHits != cfg.ResetCycles {
		t.Fatalf("warm_prefix hits = %d, want one per reset (%d)", rep.WarmPrefixHits, cfg.ResetCycles)
	}
	wantQueryChecks := cfg.ResetCycles / cfg.QueryEvery
	if rep.QueryChecks != wantQueryChecks {
		t.Fatalf("query_checks = %d, want %d", rep.QueryChecks, wantQueryChecks)
	}
	if rep.PeakResidentTokens <= 0 {
		t.Fatalf("peak resident tokens not reported: %d", rep.PeakResidentTokens)
	}
	if rep.PeakResidentTokens > rep.ResidentBoundTokens {
		t.Fatalf("peak resident tokens = %d, bound = %d", rep.PeakResidentTokens, rep.ResidentBoundTokens)
	}
	if len(rep.JSON()) == 0 {
		t.Fatal("empty managed-context soak JSON artifact")
	}
}

func TestManagedContextSoakDeterministic(t *testing.T) {
	cfg := DefaultManagedContextSoakConfig()
	cfg.ResetCycles = 25
	cfg.QueryEvery = 5

	a := RunManagedContextSoak(cfg)
	b := RunManagedContextSoak(cfg)
	if string(a.JSON()) != string(b.JSON()) {
		t.Fatal("managed-context soak artifact is not deterministic")
	}
}

func TestManagedContextSoakReportsContinuityFailure(t *testing.T) {
	cfg := ManagedContextSoakConfig{
		ResetCycles:          1,
		ContextBudgetTokens:  64,
		ResidentBudgetTokens: 512,
		Goal:                 "must survive",
		AssumptionNeedles:    []string{"critical assumption"},
		QueryEvery:           1,
		InitialMessages: []sessionreset.Msg{
			{Role: "system", Content: "fak managed-context stable prefix."},
			{Role: "user", Content: "unrelated work only"},
			{Role: "assistant", Content: "no relevant context present"},
		},
	}
	rep := RunManagedContextSoak(cfg)

	if rep.OK() {
		t.Fatalf("soak unexpectedly passed despite missing goal, assumption, and query marker: %+v", rep)
	}
	var sawGoal, sawAssumption, sawQuery bool
	for _, f := range rep.ContinuityFailures {
		sawGoal = sawGoal || f.Kind == "goal"
		sawAssumption = sawAssumption || f.Kind == "assumption" && strings.Contains(f.Detail, "critical assumption")
	}
	for _, f := range rep.QueryFailures {
		sawQuery = sawQuery || f.Kind == "query_answer"
	}
	if !sawGoal || !sawAssumption || !sawQuery {
		t.Fatalf("missing expected failure rows: continuity=%v query=%v", rep.ContinuityFailures, rep.QueryFailures)
	}
}

func TestManagedContextSoakReportsResidentBoundViolation(t *testing.T) {
	cfg := DefaultManagedContextSoakConfig()
	cfg.ResetCycles = 1
	cfg.ResidentBudgetTokens = 1

	rep := RunManagedContextSoak(cfg)
	if rep.OK() {
		t.Fatalf("soak unexpectedly passed with resident bound %d and peak %d",
			rep.ResidentBoundTokens, rep.PeakResidentTokens)
	}
	if len(rep.BudgetViolations) == 0 {
		t.Fatalf("missing resident budget violation: %+v", rep)
	}
	if rep.BudgetViolations[0].Kind != "resident_bound" {
		t.Fatalf("first budget violation kind = %q, want resident_bound", rep.BudgetViolations[0].Kind)
	}
}
