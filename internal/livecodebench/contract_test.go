package livecodebench

import (
	"encoding/json"
	"strings"
	"testing"
)

func readyContractInput() OfficialRunContractInput {
	return OfficialRunContractInput{
		GeneratedAt:     "2026-07-08T00:00:00Z",
		Issue:           "#3060",
		SuitePath:       "experiments/livecodebench/suite.json",
		ReleaseSelector: "release_v6",
		Scenario:        ScenarioCodeGeneration,
		StartDate:       "2025-01-01",
		EndDate:         "2025-06-30",
		Model:           "glm-5.2",
		ServingBackend:  "SGLang W4AFP8",
		Gateway:         "http://127.0.0.1:8080/v1",
		RunDir:          "experiments/livecodebench/glm52-run",
		Suite: &Suite{
			Schema:         SuiteSchema,
			Benchmark:      Benchmark,
			ReleaseVersion: "release_v6",
			Problems: []Problem{
				{QuestionID: "q-002", Scenario: ScenarioCodeGeneration, Prompt: "p"},
				{QuestionID: "q-001", Scenario: ScenarioCodeGeneration, Prompt: "p"},
				{QuestionID: "q-repair", Scenario: ScenarioSelfRepair, Prompt: "p"},
			},
		},
	}
}

// The load-bearing honesty invariant of #2110: a contract emits, but it can
// never carry a score. result_claim_allowed must be false regardless of how
// complete the run is pinned.
func TestOfficialRunContractResultClaimNeverAllowed(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   OfficialRunContractInput
	}{
		{"fully-pinned", readyContractInput()},
		{"empty", OfficialRunContractInput{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := BuildOfficialRunContract(tc.in)
			if c.ResultClaimAllowed {
				t.Fatalf("result_claim_allowed must always be false, got true for %s", tc.name)
			}
			if c.Schema != OfficialRunContractSchema {
				t.Fatalf("schema = %q, want %q", c.Schema, OfficialRunContractSchema)
			}
		})
	}
}

func TestOfficialRunContractReadyWhenPinned(t *testing.T) {
	c := BuildOfficialRunContract(readyContractInput())
	if c.Status != ContractReady {
		var bad []string
		for _, g := range c.Gates {
			if !g.OK {
				bad = append(bad, g.Name+": "+g.Detail)
			}
		}
		t.Fatalf("status = %q, want %q; failing gates: %s", c.Status, ContractReady, strings.Join(bad, "; "))
	}
	// Constants must be pinned and the release resolved to a concrete version.
	if c.Constants.ReleaseVersion != "release_v6" {
		t.Fatalf("release_version = %q, want release_v6", c.Constants.ReleaseVersion)
	}
}

// The contract must enumerate the exact official grading command (acceptance:
// "enumerates the exact official grading command").
func TestOfficialRunContractEnumeratesOfficialGrading(t *testing.T) {
	c := BuildOfficialRunContract(readyContractInput())
	if !strings.Contains(c.Grading.CustomEvaluatorCommand, "lcb_runner.runner.custom_evaluator") {
		t.Fatalf("custom evaluator command missing lcb_runner: %q", c.Grading.CustomEvaluatorCommand)
	}
	if !strings.Contains(c.Grading.CustomEvaluatorCommand, "--release_version release_v6") {
		t.Fatalf("custom evaluator command must pin the release: %q", c.Grading.CustomEvaluatorCommand)
	}
	if !strings.Contains(c.Grading.ComputeScoresCommand, "lcb_runner.evaluation.compute_scores") {
		t.Fatalf("compute_scores command missing: %q", c.Grading.ComputeScoresCommand)
	}
	if !strings.Contains(c.Grading.ComputeScoresCommand, "--start_date 2025-01-01") || !strings.Contains(c.Grading.ComputeScoresCommand, "--end_date 2025-06-30") {
		t.Fatalf("compute_scores must carry the date window: %q", c.Grading.ComputeScoresCommand)
	}
	if !strings.Contains(c.Grading.RawEvaluateCommand, "--evaluate") {
		t.Fatalf("raw evaluate command must pass --evaluate: %q", c.Grading.RawEvaluateCommand)
	}
}

func TestOfficialRunContractArmsShapeBothGenerationArms(t *testing.T) {
	c := BuildOfficialRunContract(readyContractInput())
	if len(c.Arms) != 2 {
		t.Fatalf("want 2 arms, got %d", len(c.Arms))
	}
	byName := map[string]ContractArm{}
	for _, a := range c.Arms {
		byName[a.Name] = a
	}
	raw, ok := byName["raw-lcb_runner"]
	if !ok {
		t.Fatalf("missing raw-lcb_runner arm: %v", c.Arms)
	}
	if !strings.Contains(strings.Join(raw.Commands, " "), "lcb_runner.runner.main") {
		t.Fatalf("raw arm must call lcb_runner.runner.main: %v", raw.Commands)
	}
	fak, ok := byName["fak-native"]
	if !ok {
		t.Fatalf("missing fak-native arm: %v", c.Arms)
	}
	joined := strings.Join(fak.Commands, "\n")
	if !strings.Contains(joined, "fak livecodebench generate") {
		t.Fatalf("fak arm must call `fak livecodebench generate`: %v", fak.Commands)
	}
	if !strings.Contains(joined, "export --format custom-evaluator") {
		t.Fatalf("fak arm must export to the custom-evaluator shape: %v", fak.Commands)
	}
	// Neither generation arm may grade.
	for _, a := range c.Arms {
		if strings.Contains(strings.Join(a.Commands, " "), "--evaluate") || strings.Contains(strings.Join(a.Commands, " "), "custom_evaluator") {
			t.Fatalf("arm %q must not grade: %v", a.Name, a.Commands)
		}
	}
}

// release_latest / empty release must fail the explicit-pin gate and leave the
// contract INCOMPLETE — an implicit release cannot back a published result.
func TestOfficialRunContractRefusesImplicitRelease(t *testing.T) {
	for _, sel := range []string{"", "release_latest"} {
		in := readyContractInput()
		in.ReleaseSelector = sel
		c := BuildOfficialRunContract(in)
		if gateOK(c, "release_pinned_explicit") {
			t.Fatalf("release selector %q must fail release_pinned_explicit", sel)
		}
		if c.Status != ContractIncomplete {
			t.Fatalf("release selector %q must yield INCOMPLETE_CONTRACT, got %q", sel, c.Status)
		}
	}
}

func TestOfficialRunContractDateWindowGate(t *testing.T) {
	cases := []struct {
		name       string
		start, end string
		wantOK     bool
	}{
		{"good", "2025-01-01", "2025-06-30", true},
		{"missing-end", "2025-01-01", "", false},
		{"reversed", "2025-06-30", "2025-01-01", false},
		{"malformed", "2025-1-1", "2025-06-30", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := readyContractInput()
			in.StartDate, in.EndDate = tc.start, tc.end
			c := BuildOfficialRunContract(in)
			if gateOK(c, "date_window_recorded") != tc.wantOK {
				t.Fatalf("date_window_recorded ok = %v, want %v", gateOK(c, "date_window_recorded"), tc.wantOK)
			}
		})
	}
}

// The candidate question_ids must come from the pinned suite, filtered to the
// scenario, sorted and de-duplicated.
func TestOfficialRunContractPinsSuiteProblemIDs(t *testing.T) {
	c := BuildOfficialRunContract(readyContractInput())
	got := c.ProblemSelection.CandidateProblemIDs
	want := []string{"q-001", "q-002"} // q-repair is a different scenario
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("candidate problem ids = %v, want %v", got, want)
	}
	if !gateOK(c, "candidate_problem_ids") {
		t.Fatalf("candidate_problem_ids gate should be OK when a suite is pinned")
	}

	// Without a suite the gate asks the operator to supply one.
	in := readyContractInput()
	in.Suite = nil
	noSuite := BuildOfficialRunContract(in)
	if gateOK(noSuite, "candidate_problem_ids") {
		t.Fatalf("candidate_problem_ids should not be OK without a suite")
	}
	if noSuite.Status != ContractIncomplete {
		t.Fatalf("no-suite contract should be INCOMPLETE, got %q", noSuite.Status)
	}
}

func TestOfficialRunContractJSONRoundTrips(t *testing.T) {
	c := BuildOfficialRunContract(readyContractInput())
	raw, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back OfficialRunContract
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Schema != c.Schema || back.ResultClaimAllowed != c.ResultClaimAllowed || back.Status != c.Status {
		t.Fatalf("round-trip mismatch: %+v vs %+v", back, c)
	}
	// The false must be present in the wire form, not omitted.
	if !strings.Contains(string(raw), `"result_claim_allowed":false`) {
		t.Fatalf("result_claim_allowed must serialize explicitly as false: %s", raw)
	}
}

func TestRenderOfficialRunContractMarkdown(t *testing.T) {
	c := BuildOfficialRunContract(readyContractInput())
	md := RenderOfficialRunContractMarkdown(c)
	for _, want := range []string{
		"# LiveCodeBench Official-Run Contract",
		"Result claim allowed: `false`",
		"lcb_runner.runner.custom_evaluator",
		"fak livecodebench generate",
		"## Required Before Any Result Claim",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q\n---\n%s", want, md)
		}
	}
}

func gateOK(c OfficialRunContract, name string) bool {
	for _, g := range c.Gates {
		if g.Name == name {
			return g.OK
		}
	}
	return false
}
