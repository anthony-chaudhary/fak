package livecodebench

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

func validSuite() Suite {
	return Suite{
		Schema:         SuiteSchema,
		Benchmark:      Benchmark,
		Model:          "test-model",
		ReleaseVersion: "release_v6",
		Problems: []Problem{
			{
				QuestionID:  "lc-0001",
				Scenario:    ScenarioCodeGeneration,
				Platform:    "leetcode",
				Difficulty:  "easy",
				ContestDate: "2025-11-03",
				Prompt:      "Given an array, return the sum.",
				StarterCode: "class Solution:\n    def sum(self, nums):",
				PublicTests: []TestCase{{Input: "[1,2]", Output: "3", TestType: "functional"}},
				PrivateTests: []TestCase{
					{Input: "[1,2,3]", Output: "6", TestType: "functional"},
				},
			},
			{
				QuestionID:  "ac-0002",
				Scenario:    ScenarioCodeExecution,
				Platform:    "atcoder",
				Difficulty:  "medium",
				ContestDate: "2025-12-15",
				Prompt:      "What does this program print?",
			},
		},
	}
}

func TestSuiteJSONRoundTrip(t *testing.T) {
	want := validSuite()
	b, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := LoadSuite(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("round trip mismatch:\ngot  %#v\nwant %#v", got, want)
	}
}

func TestLoadSuiteRejectsUnknownFields(t *testing.T) {
	body := `{"schema":"fak.livecodebench-suite.v1","benchmark":"livecodebench","release_version":"release_v6","problems":[],"pass_1":0.99}`
	_, err := LoadSuite(strings.NewReader(body))
	if err == nil || !strings.Contains(err.Error(), "pass_1") {
		t.Fatalf("err = %v, want unknown-field refusal", err)
	}
}

func TestSuiteValidateRejections(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Suite)
		want   string
	}{
		{"wrong schema", func(s *Suite) { s.Schema = "fak.livecodebench.fixture.v1" }, "schema"},
		{"wrong benchmark", func(s *Suite) { s.Benchmark = "swebench" }, "benchmark"},
		{"missing release", func(s *Suite) { s.ReleaseVersion = " " }, "release_version is required"},
		{"no problems", func(s *Suite) { s.Problems = nil }, "at least one problem"},
		{"missing question id", func(s *Suite) { s.Problems[0].QuestionID = "" }, "question_id is required"},
		{"unknown scenario", func(s *Suite) { s.Problems[0].Scenario = "codereview" }, "not supported"},
		{"duplicate id in scenario", func(s *Suite) {
			s.Problems[1] = s.Problems[0]
		}, "duplicate question_id"},
		{"missing prompt", func(s *Suite) { s.Problems[0].Prompt = "" }, "prompt is required"},
		{"unknown platform", func(s *Suite) { s.Problems[0].Platform = "topcoder" }, "platform"},
		{"unknown difficulty", func(s *Suite) { s.Problems[0].Difficulty = "extreme" }, "difficulty"},
		{"bad contest date", func(s *Suite) { s.Problems[0].ContestDate = "11/03/2025" }, "YYYY-MM-DD"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := validSuite()
			tc.mutate(&s)
			err := s.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestSuiteAllowsSameQuestionAcrossScenarios(t *testing.T) {
	s := validSuite()
	s.Problems[1].QuestionID = s.Problems[0].QuestionID
	if err := s.Validate(); err != nil {
		t.Fatalf("same question_id across scenarios must be allowed: %v", err)
	}
}

func TestNewReportIsHonestByDefault(t *testing.T) {
	rep := NewReport(validSuite(), time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC))
	if err := rep.Validate(); err != nil {
		t.Fatal(err)
	}
	if rep.ResultClaimAllowed {
		t.Fatal("a fresh report must not allow a result claim")
	}
	if rep.EvidenceClass != EvidenceLocalUngraded {
		t.Fatalf("evidence class = %q", rep.EvidenceClass)
	}
	if rep.Summary.Problems != 2 || len(rep.Summary.Scenarios) != 2 {
		t.Fatalf("summary = %+v", rep.Summary)
	}
}

func TestReportPassRatesRequireEvidenceClass(t *testing.T) {
	rep := NewReport(validSuite(), time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC))
	rep.EvidenceClass = ""
	rep.Arms = []ArmResult{{Arm: "fak", Scenario: ScenarioCodeGeneration, Problems: 1, Generations: 5, Graded: 5, Pass1: 0.6}}
	err := rep.Validate()
	if err == nil || !strings.Contains(err.Error(), "evidence_class") {
		t.Fatalf("err = %v, want evidence_class refusal", err)
	}
	rep.EvidenceClass = EvidenceOfficialLCBRunner
	if err := rep.Validate(); err != nil {
		t.Fatalf("officially graded results must validate: %v", err)
	}
}

func TestReportClaimRequiresOfficialGrading(t *testing.T) {
	rep := NewReport(validSuite(), time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC))
	rep.Arms = []ArmResult{{Arm: "raw", Scenario: ScenarioCodeGeneration, Problems: 1, Generations: 5, Graded: 5, Pass1: 0.4}}
	rep.ResultClaimAllowed = true
	err := rep.Validate()
	if err == nil || !strings.Contains(err.Error(), EvidenceOfficialLCBRunner) {
		t.Fatalf("err = %v, want official-grading refusal", err)
	}

	rep.EvidenceClass = EvidenceOfficialLCBRunner
	if err := rep.Validate(); err != nil {
		t.Fatal(err)
	}

	rep.Arms = nil
	err = rep.Validate()
	if err == nil || !strings.Contains(err.Error(), "graded results") {
		t.Fatalf("err = %v, want graded-results refusal", err)
	}
}

func TestReportArmValidation(t *testing.T) {
	base := func() Report {
		rep := NewReport(validSuite(), time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC))
		rep.Arms = []ArmResult{{Arm: "fak", Scenario: ScenarioCodeGeneration, Problems: 1, Generations: 5, Graded: 5, Pass1: 0.6}}
		return rep
	}
	cases := []struct {
		name   string
		mutate func(*Report)
		want   string
	}{
		{"unknown arm", func(r *Report) { r.Arms[0].Arm = "vanilla" }, "raw|fak"},
		{"unknown scenario", func(r *Report) { r.Arms[0].Scenario = "codereview" }, "not supported"},
		{"graded exceeds generations", func(r *Report) { r.Arms[0].Graded = 6 }, "exceeds generations"},
		{"pass rate out of range", func(r *Report) { r.Arms[0].Pass1 = 1.5 }, "[0,1]"},
		{"bad generated_at", func(r *Report) { r.GeneratedAt = "yesterday" }, "RFC3339"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rep := base()
			tc.mutate(&rep)
			err := rep.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want substring %q", err, tc.want)
			}
		})
	}
}
