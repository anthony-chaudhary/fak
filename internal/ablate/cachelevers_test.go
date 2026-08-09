package ablate

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metrics"
)

func TestCacheLeverDescriptorsAreRegisteredAndGated(t *testing.T) {
	levers := CacheLevers()
	if len(levers) != 6 {
		t.Fatalf("cache lever count = %d, want 6", len(levers))
	}
	seen := map[string]bool{}
	for _, lever := range levers {
		if seen[lever.Name()] {
			t.Fatalf("duplicate lever %q", lever.Name())
		}
		seen[lever.Name()] = true
		concept, ok := registeredConcept(lever.Name())
		if !ok {
			t.Fatalf("lever %q is not in CHILD-C registry", lever.Name())
		}
		if !concept.StreamXform || concept.Correctness == nil {
			t.Fatalf("lever %q registration = %#v, want stream transform with gate", lever.Name(), concept)
		}
		if got := concept.Correctness(); got.Verdict != "pass" {
			t.Fatalf("lever %q correctness = %#v, want pass", lever.Name(), got)
		}
	}
}

func TestCacheLeverSweepReportsTokenDeltaAndStaysDefaultOff(t *testing.T) {
	for _, lever := range CacheLevers() {
		configs, err := BuildSweep([]string{lever.Name()})
		if err != nil {
			t.Fatalf("BuildSweep(%q): %v", lever.Name(), err)
		}
		if len(configs) != 2 || configs[0].Descriptor()[lever.Name()] != "off" || configs[1].Descriptor()[lever.Name()] != "on" {
			t.Fatalf("%q arms = %#v, want default-off then on", lever.Name(), configs)
		}
		report := Report{
			Baseline: configs[0].Name,
			Runs: []AblationRun{
				{ArmID: configs[0].Name, Features: configs[0].Descriptor(), Arm: metrics.Arm{InTokens: 100, OutTokens: 10}},
				{ArmID: configs[1].Name, Features: configs[1].Descriptor(), Arm: metrics.Arm{InTokens: 75, OutTokens: 10}},
			},
		}
		report.annotateConceptRows()
		if len(report.Runs) != 2 || report.Runs[1].TokenDelta != -25 {
			t.Fatalf("%q token delta rows = %#v, want -25", lever.Name(), report.Runs)
		}
		if report.Runs[1].Correctness.Verdict != "pass" {
			t.Fatalf("%q sweep correctness = %#v", lever.Name(), report.Runs[1].Correctness)
		}
	}
}
