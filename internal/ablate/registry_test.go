package ablate

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

func TestRegisterOpensConceptSetAndPreservesFailLoud(t *testing.T) {
	const token = "zz_registry_test_concept"
	var set bool
	estimate := 1.25
	Register(Concept{Token: token, Runtime: func(on bool) { set = on }, Owner: "fak", Reversible: true, PrefixStable: true,
		Correctness:            func() GateResult { return GateResult{Verdict: "pass"} },
		ChildAOwnServeEstimate: func() *float64 { return &estimate }})
	if _, err := BuildSweep([]string{"unregistered_registry_test_concept"}); err == nil {
		t.Fatal("BuildSweep accepted an unregistered token")
	}
	configs, err := BuildSweep([]string{token})
	if err != nil {
		t.Fatal(err)
	}
	if got := configs[1].Descriptor()[token]; got != "on" {
		t.Fatalf("registered concept descriptor = %q, want on", got)
	}
	reset := applyRuntimeConcepts(configs[1])
	if !set {
		t.Fatal("registered runtime setter was not called")
	}
	reset()
	if set {
		t.Fatal("runtime reset did not restore off")
	}
}

func TestBuildSweepPlanMainAndPairwise(t *testing.T) {
	known := KnownFeatures()
	main, err := BuildSweepPlan(SweepPlanMain, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(main) != len(known)+1 {
		t.Fatalf("main arms = %d, want baseline + %d concepts", len(main), len(known))
	}
	for _, c := range main {
		if len(c.Descriptor()) != len(known) {
			t.Fatalf("arm %q carries %d concepts, want %d", c.Name, len(c.Descriptor()), len(known))
		}
	}
	ranked := []string{known[0], known[1], known[2]}
	pairwise, err := BuildSweepPlan(SweepPlanPairwise, 3, ranked)
	if err != nil {
		t.Fatal(err)
	}
	if want := len(main) + 3; len(pairwise) != want {
		t.Fatalf("pairwise arms = %d, want %d", len(pairwise), want)
	}
	got := []string{pairwise[len(main)].Name, pairwise[len(main)+1].Name, pairwise[len(main)+2].Name}
	want := []string{"pair-" + ranked[0] + "+" + ranked[1], "pair-" + ranked[0] + "+" + ranked[2], "pair-" + ranked[1] + "+" + ranked[2]}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("pair names = %v, want %v", got, want)
	}
}

func structArm(in, out int64) metrics.Arm { return metrics.Arm{InTokens: in, OutTokens: out} }

func TestConceptRowsCarryRequiredDerivedFields(t *testing.T) {
	token := KnownFeatures()[0]
	rep := &Report{Baseline: "all-off", Runs: []AblationRun{{ArmID: "all-off", Arm: structArm(100, 20)}, {ArmID: token, Features: map[string]string{token: "on"}, Arm: structArm(80, 10), MechanismSavings: gateway.MechanismSavings{FakCompactionShedTokens: 30}}}}
	rep.annotateConceptRows()
	row := rep.Runs[1]
	if row.TokenDelta != -30 {
		t.Fatalf("token delta = %d, want -30", row.TokenDelta)
	}
	if row.FakTokenDelta != 30 || row.ProviderTokenDelta != 0 {
		t.Fatalf("owner split = provider %v fak %v", row.ProviderTokenDelta, row.FakTokenDelta)
	}
	if strings.TrimSpace(row.Correctness.Verdict) == "" {
		t.Fatal("correctness verdict is empty")
	}
}
