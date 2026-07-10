package livecodebench

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"
)

// The adapter schema (#2087) is the stable shape every downstream LiveCodeBench
// scenario builds on: #2090 (data load) writes it, #2096/#2097/#2099/#2100
// (codegen/self-repair/code-execution/grader) fill it, #2105/#2106 (fak-arm /
// sampling) run over it, and #2114 (honesty) gates its result fields. The
// round-trip test in suite_test.go compares Go structs, so a *symmetric* json
// tag or schema-string rename would still pass it while silently breaking the
// wire contract and every persisted suite/report. These tests pin the wire
// contract itself so such a break cannot land unwitnessed.

// TestSchemaConstantsPinV1Wire pins the two schema-version strings and the
// benchmark tag that identify a LiveCodeBench suite/report on the wire (#2087).
func TestSchemaConstantsPinV1Wire(t *testing.T) {
	for _, tc := range []struct{ name, got, want string }{
		{"SuiteSchema", SuiteSchema, "fak.livecodebench-suite.v1"},
		{"ReportSchema", ReportSchema, "fak.livecodebench-report.v1"},
		{"Benchmark", Benchmark, "livecodebench"},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %q, want the pinned v1 wire string %q (a rename is a wire break)", tc.name, tc.got, tc.want)
		}
	}
}

// TestScenarioConstantsPinUpstreamNames pins the four upstream LiveCodeBench
// scenario identifiers (#2087). These strings key into the upstream datasets, so
// a typo here silently mis-loads or drops problems rather than failing loudly.
func TestScenarioConstantsPinUpstreamNames(t *testing.T) {
	for _, tc := range []struct {
		got  Scenario
		want string
	}{
		{ScenarioCodeGeneration, "codegeneration"},
		{ScenarioSelfRepair, "selfrepair"},
		{ScenarioTestOutputPrediction, "testoutputprediction"},
		{ScenarioCodeExecution, "codeexecution"},
	} {
		if string(tc.got) != tc.want {
			t.Errorf("scenario constant = %q, want upstream name %q", tc.got, tc.want)
		}
	}
}

// TestAdapterTypesCarryDocumentedWireFields pins the JSON field names of the six
// named adapter types #2087 defines (Suite, Problem, Scenario, Report,
// ArmResult, Summary), including the LCB-native fields the scope calls out:
// question_id, platform, difficulty, release_version, contest_date,
// public/private test cases, and starter_code.
func TestAdapterTypesCarryDocumentedWireFields(t *testing.T) {
	suite := validSuite()

	// Suite: schema tag + the release pin every downstream reader keys on.
	assertWireFields(t, "Suite", suite, "schema", "benchmark", "release_version", "provenance", "problems")

	// Problem: the LCB-native fields. validSuite's first problem sets them all,
	// so even omitempty fields must appear.
	assertWireFields(t, "Problem", suite.Problems[0],
		"question_id", "scenario", "platform", "difficulty",
		"contest_date", "prompt", "starter_code",
		"public_test_cases", "private_test_cases")

	// Report + ArmResult: the honesty seam and the (arm, scenario) result cell.
	rep := NewReport(suite, time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC))
	rep.Arms = []ArmResult{{Arm: "fak", Scenario: ScenarioCodeGeneration, Problems: 1, Generations: 5, Graded: 5, Pass1: 0.6, Pass5: 0.8}}
	assertWireFields(t, "Report", rep,
		"schema", "generated_at", "evidence_class", "result_claim_allowed",
		"arms", "summary", "official_harness")
	assertWireFields(t, "ArmResult", rep.Arms[0],
		"arm", "scenario", "problems", "generations", "graded", "pass_1", "pass_5")

	// Summary: the folded suite shape a report ran over.
	assertWireFields(t, "Summary",
		Summary{Problems: 2, Graded: 1, Scenarios: []ScenarioReport{{Scenario: "codegeneration", Questions: 2}}},
		"problems", "graded", "scenarios")
}

func assertWireFields(t *testing.T, typ string, v any, fields ...string) {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal %s: %v", typ, err)
	}
	for _, f := range fields {
		if !bytes.Contains(b, []byte(`"`+f+`"`)) {
			t.Errorf("%s JSON missing documented wire field %q:\n%s", typ, f, b)
		}
	}
}
