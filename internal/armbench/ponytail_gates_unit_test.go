package armbench

import (
	"fmt"
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
	args, err := benchmarkProviderArgs(PonytailGateOptions{Model: "haiku", NativeMedium: NativeProfile{Identity: canonical.Profile, Segment: canonical.Segment}}, ponytailNativeMediumArm, "task")
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
	}, NativeProfile{Identity: syspromptmmu.WorkProfilePonytailNativeMed, Segment: syspromptmmu.DescribeWorkProfile(syspromptmmu.WorkProfilePonytailNativeMed).Segment})
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

func TestPonytailEvaluationModePreSpendValidation(t *testing.T) {
	cases := []struct {
		name string
		opts PonytailGateOptions
		want string
	}{
		{"trials", PonytailGateOptions{EvaluationMode: true, Trials: 4, Model: "claude-haiku-4-5-20251001", Account: "seat-a"}, "--trials >= 5"},
		{"model empty", PonytailGateOptions{EvaluationMode: true, Trials: 5, Account: "seat-a"}, "exact model snapshot"},
		{"model alias", PonytailGateOptions{EvaluationMode: true, Trials: 5, Model: "haiku", Account: "seat-a"}, "exact model snapshot"},
		{"account", PonytailGateOptions{EvaluationMode: true, Trials: 5, Model: "claude-haiku-4-5-20251001"}, "account identity"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateEvaluationModeOptions(tc.opts)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v want %q", err, tc.want)
			}
		})
	}
}

func TestPonytailEvaluationModeScheduleIsDeterministicAndCounterbalanced(t *testing.T) {
	seenFirst := map[string]bool{}
	var prior []string
	for trial := 1; trial <= len(ponytailBenchmarkArms); trial++ {
		arms, seed := counterbalancedArmOrder("up.behavior.hardware-calibration", trial)
		if len(seed) != 64 || len(arms) != len(ponytailBenchmarkArms) { //boundarylint:ignore CHANGE_DETECTOR_TEST the seeded SHA-256 identifier is 64 hex characters while arm cardinality is checked against its source list
			t.Fatalf("trial %d arms=%v seed=%q", trial, arms, seed)
		}
		if seenFirst[arms[0]] {
			t.Fatalf("first arm repeated before balance completed: %v", arms)
		}
		seenFirst[arms[0]] = true
		again, againSeed := counterbalancedArmOrder("up.behavior.hardware-calibration", trial)
		if !slices.Equal(arms, again) || seed != againSeed {
			t.Fatalf("schedule not deterministic: %v/%q vs %v/%q", arms, seed, again, againSeed)
		}
		if trial > 1 && slices.Equal(prior, arms) {
			t.Fatalf("trial order did not rotate: %v", arms)
		}
		prior = arms
	}
}

func TestPonytailDecisionAssessmentFailClosed(t *testing.T) {
	scenarios := []GateScenario{
		{ID: "b", Category: "behavior", RequiresProvider: true},
		{ID: "c", Category: "correctness", RequiresProvider: true},
		{ID: "r", Category: "robustness", RequiresProvider: true},
	}
	summaries := func(nativeBehavior, nativeCorrectness, nativeRobustness int) []GateSummary {
		var out []GateSummary
		for _, category := range []string{"behavior", "correctness", "robustness"} {
			out = append(out,
				GateSummary{Arm: "baseline", Category: category, Passed: 4, Failed: 1},
				GateSummary{Arm: "ponytail", Category: category, Passed: 4, Failed: 1})
		}
		out = append(out,
			GateSummary{Arm: ponytailNativeMediumArm, Category: "behavior", Passed: nativeBehavior, Failed: 5 - nativeBehavior},
			GateSummary{Arm: ponytailNativeMediumArm, Category: "correctness", Passed: nativeCorrectness, Failed: 5 - nativeCorrectness},
			GateSummary{Arm: ponytailNativeMediumArm, Category: "robustness", Passed: nativeRobustness, Failed: 5 - nativeRobustness})
		return out
	}
	if got := assessPonytailProfile(scenarios, summaries(5, 4, 4), true, 5); got.Decision != "keep" {
		t.Fatalf("complete non-regression plus improvement = %+v", got)
	}
	if got := assessPonytailProfile(scenarios, summaries(5, 4, 3), true, 5); got.Decision != "revert" {
		t.Fatalf("robustness regression = %+v", got)
	}
	if got := assessPonytailProfile(scenarios, summaries(5, 3, 4), true, 5); got.Decision != "tune" {
		t.Fatalf("mixed categories = %+v", got)
	}
	if got := assessPonytailProfile(scenarios, summaries(5, 4, 4)[:8], true, 5); got.Decision != "tune" || got.Complete {
		t.Fatalf("missing category = %+v", got)
	}
	if got := assessPonytailProfile(scenarios, summaries(5, 4, 4), true, 4); got.Decision != "tune" || got.Sufficient {
		t.Fatalf("insufficient trials = %+v", got)
	}
}

func TestValidatePonytailReplayRejectsMissingCell(t *testing.T) {
	cells := ponytailReplayCellsForTest(2)
	missingKey := trialCellIdentity("up.behavior.hardware-calibration", "ponytail", 2)
	for i, cell := range cells {
		if trialCellIdentity(cell.ScenarioID, cell.Arm, 0) == missingKey {
			cells = append(cells[:i], cells[i+1:]...)
			break
		}
	}

	_, err := validatePonytailReplay(ponytailReplayScenariosForTest(), cells, 2)
	if err == nil {
		t.Fatal("missing replay cell was accepted")
	}
	for _, want := range []string{`scenario="up.behavior.hardware-calibration"`, `arm="ponytail"`, "trial=2"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q does not identify %q", err, want)
		}
	}
}

func TestValidatePonytailReplayRejectsErroredCell(t *testing.T) {
	cells := ponytailReplayCellsForTest(2)
	erroredKey := trialCellIdentity("up.behavior.hardware-calibration", ponytailNativeMediumArm, 1)
	for i := range cells {
		if trialCellIdentity(cells[i].ScenarioID, cells[i].Arm, 0) == erroredKey {
			cells[i].Error = "provider timeout"
			break
		}
	}

	_, err := validatePonytailReplay(ponytailReplayScenariosForTest(), cells, 2)
	if err == nil {
		t.Fatal("errored replay cell was accepted")
	}
	for _, want := range []string{`scenario="up.behavior.hardware-calibration"`, `arm="native_medium"`, "trial=1", "provider timeout"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q does not identify %q", err, want)
		}
	}
}

func TestValidatePonytailReplayAcceptsCompleteTrials(t *testing.T) {
	for _, trials := range []int{1, 2} {
		t.Run(fmt.Sprintf("trials-%d", trials), func(t *testing.T) {
			replay, err := validatePonytailReplay(ponytailReplayScenariosForTest(), ponytailReplayCellsForTest(trials), trials)
			if err != nil {
				t.Fatal(err)
			}
			if len(replay) != trials*len(ponytailBenchmarkArms) {
				t.Fatalf("replay cells collapsed: %+v", replay)
			}
			for trial := 1; trial <= trials; trial++ {
				for _, arm := range ponytailBenchmarkArms {
					got, ok := replay[replayCellIdentity("up.behavior.hardware-calibration", arm, trial, trials)]
					want := fmt.Sprintf("%s trial %d", arm, trial)
					if !ok || got.Output != want {
						t.Fatalf("trial %d arm %s replay = %+v, %v; want %q", trial, arm, got, ok, want)
					}
				}
			}
		})
	}
}

func ponytailReplayScenariosForTest() []GateScenario {
	return []GateScenario{
		{ID: "up.behavior.hardware-calibration", RequiresProvider: true},
		{ID: "up.correctness-regression.fenced-email-passes", RequiresProvider: false},
	}
}

func ponytailReplayCellsForTest(trials int) []GateCell {
	var cells []GateCell
	for trial := 1; trial <= trials; trial++ {
		for _, arm := range ponytailBenchmarkArms {
			scenarioID := "up.behavior.hardware-calibration"
			if trials > 1 {
				scenarioID = fmt.Sprintf("%s.trial-%02d", scenarioID, trial)
			}
			cells = append(cells, GateCell{ScenarioID: scenarioID, Arm: arm, Output: fmt.Sprintf("%s trial %d", arm, trial)})
		}
	}
	return cells
}
