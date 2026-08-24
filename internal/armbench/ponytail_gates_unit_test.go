package armbench

import (
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestPonytailGateUnknownFailsClosed(t *testing.T) {
	ok, reason := runPinnedGate(t.Context(), "", GateScenario{ID: "up.unknown"}, "anything")
	if ok || reason != "unknown gate" {
		t.Fatalf("got %v %q", ok, reason)
	}
	if _, err := benchmarkProviderArgs(PonytailGateOptions{}, "unknown", "task"); err == nil || !strings.Contains(err.Error(), "unknown Ponytail gate arm") {
		t.Fatalf("provider dispatch did not fail closed: %v", err)
	}
}

func TestPonytailGateNativeMediumUsesCanonicalRenderedFragment(t *testing.T) {
	canonical := syspromptmmu.DescribeWorkProfile(syspromptmmu.WorkProfilePonytailNativeMed)
	args, err := benchmarkProviderArgs(PonytailGateOptions{Model: "haiku"}, ponytailNativeMediumArm, "task")
	if err != nil {
		t.Fatal(err)
	}
	wantPair := []string{"--system-prompt", canonical.Segment}
	found := false
	for i := 0; i+1 < len(args); i++ {
		if slices.Equal(args[i:i+2], wantPair) {
			found = true
		}
		if args[i] == "--system-prompt-file" {
			t.Fatalf("native arm used a prompt file: %q", args)
		}
	}
	if !found {
		t.Fatalf("native arm omitted canonical segment: %q", args)
	}
	if canonical.Profile != syspromptmmu.WorkProfilePonytailNativeMed || canonical.Witness != ponytailNativeMediumDigest {
		t.Fatalf("unexpected canonical identity: %+v", canonical)
	}
}

func TestPonytailArmIdentitysIdentifyNativeMedium(t *testing.T) {
	arms, err := benchmarkArmIdentities([]GateSource{
		{Path: "benchmarks/arms/caveman-SKILL.md", SHA256: strings.Repeat("c", 64)},
		{Path: "skills/ponytail/SKILL.md", SHA256: strings.Repeat("p", 64)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(arms) != 4 {
		t.Fatalf("arms=%+v", arms)
	}
	for i, want := range []string{"baseline", "caveman", "ponytail", ponytailNativeMediumArm} {
		if arms[i].Arm != want {
			t.Fatalf("arm[%d]=%q want %q", i, arms[i].Arm, want)
		}
	}
	got := arms[3]
	if got.Arm != ponytailNativeMediumArm || got.Implementation != "fak_native" || got.CanonicalProfile != syspromptmmu.WorkProfilePonytailNativeMed || got.FragmentDigest != ponytailNativeMediumDigest {
		t.Fatalf("native receipt=%+v", got)
	}
}
func TestPonytailGateSummaryDoesNotHideCategoryRegression(t *testing.T) {
	sc := []GateScenario{{ID: "b", Category: "behavior", RequiresProvider: true}, {ID: "c", Category: "correctness", RequiresProvider: true}}
	cells := []GateCell{{ScenarioID: "b", Arm: "ponytail", Category: "behavior", Pass: true}, {ScenarioID: "c", Arm: "ponytail", Category: "correctness", Pass: false}, {ScenarioID: "r", Arm: "deterministic", Category: "correctness-regression", Pass: true}}
	sums, overall := summarizeGates(sc, cells, true, 1)
	if overall {
		t.Fatal("aggregate concealed regression")
	}
	found := false
	for _, s := range sums {
		if s.Arm == "ponytail" && s.Category == "correctness" {
			found = true
			if s.GatePass || s.Failed != 1 {
				t.Fatalf("bad %+v", s)
			}
		}
	}
	if !found {
		t.Fatal("missing category")
	}
}
func TestExtensionFixturesAreSeparateDetectorPasses(t *testing.T) {
	for _, c := range extensionFixtureCells() {
		if c.Category != "extension" || !c.Pass || c.Arm != "detector" {
			t.Fatalf("bad extension %+v", c)
		}
	}
}
