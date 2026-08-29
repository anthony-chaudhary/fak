package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentic"
)

func TestAgenticJSONIsDeterministicAndParseable(t *testing.T) {
	args := []string{"--json", "--objective", "Add a CLI and tests for internal/widget/widget.go"}
	var first, firstErr bytes.Buffer
	if code := runAgentic(&first, &firstErr, args); code != 0 {
		t.Fatalf("first code=%d stderr=%s", code, firstErr.String())
	}
	var second, secondErr bytes.Buffer
	if code := runAgentic(&second, &secondErr, args); code != 0 {
		t.Fatalf("second code=%d stderr=%s", code, secondErr.String())
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatalf("repeated JSON differs:\n%s\n%s", first.Bytes(), second.Bytes())
	}
	var plan agentic.Plan
	if err := json.Unmarshal(first.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if plan.Schema != agentic.Schema || plan.Inference.Domain != agentic.DomainDevelopment || !plan.Mode.ReadOnly || !plan.Mode.Offline {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestAgenticPositionalObjectiveRemainsCompatible(t *testing.T) {
	var explicit, explicitErr bytes.Buffer
	if code := runAgentic(&explicit, &explicitErr, []string{"--json", "--objective=Add a CLI and tests"}); code != 0 {
		t.Fatalf("explicit code=%d stderr=%s", code, explicitErr.String())
	}
	var positional, positionalErr bytes.Buffer
	if code := runAgentic(&positional, &positionalErr, []string{"--json", "Add", "a", "CLI", "and", "tests"}); code != 0 {
		t.Fatalf("positional code=%d stderr=%s", code, positionalErr.String())
	}
	if !bytes.Equal(explicit.Bytes(), positional.Bytes()) {
		t.Fatalf("explicit and positional JSON differ:\n%s\n%s", explicit.Bytes(), positional.Bytes())
	}
}

func TestAgenticNativeJSONPreservesNativeContract(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runAgentic(&out, &errOut, []string{"--json", "--objective", "Optimize Qwen3.8 fak-native CUDA decode throughput"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var plan agentic.Plan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if plan.Native == nil || plan.Native.Engine != "fak-native" || plan.Native.DefaultModel != "Qwen3.8" || plan.Native.Receipt.Schema != "fak-native-performance-receipt/v2" || !plan.Native.MatchedWorkload {
		t.Fatalf("native contract = %+v", plan.Native)
	}
}

func TestAgenticTextRendersActionablePlan(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runAgentic(&out, &errOut, []string{"Fix", "the", "widget", "CLI"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"FAK agentic plan objective-", "stage expand:", "issue-ready work:", "model=gpt-5.6-sol", "read_only=true offline=true"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("text output missing %q:\n%s", want, out.String())
		}
	}
}

func TestAgenticInvalidInputReturnsUsageError(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "missing objective", args: nil, want: "objective is required"},
		{name: "blank objective", args: []string{"   "}, want: "objective is required"},
		{name: "unknown flag", args: []string{"--bogus"}, want: `unexpected option "--bogus"`},
		{name: "unknown flag after positional", args: []string{"Fix", "the", "CLI", "--bogus"}, want: `unexpected option "--bogus"`},
		{name: "short option after positional", args: []string{"Fix", "the", "CLI", "-x"}, want: `unexpected option "-x"`},
		{name: "known option after positional", args: []string{"Fix", "the", "CLI", "--json"}, want: `unexpected option "--json" after positional objective text`},
		{name: "option terminator is unexpected", args: []string{"Fix", "the", "CLI", "--", "extra"}, want: `unexpected option "--"`},
		{name: "explicit conflicts with positional", args: []string{"--objective", "Fix the CLI", "extra"}, want: "--objective conflicts with positional objective text"},
		{name: "explicit after positional conflicts", args: []string{"Fix", "--objective", "the CLI"}, want: "--objective conflicts with positional objective text"},
		{name: "duplicate objective", args: []string{"--objective", "one", "--objective", "two"}, want: "--objective may be specified only once"},
		{name: "duplicate json", args: []string{"--json", "--json", "objective"}, want: "--json may be specified only once"},
		{name: "missing objective value", args: []string{"--objective", "--json"}, want: "--objective requires a value"},
		{name: "blank objective value", args: []string{"--objective="}, want: "objective is required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, errOut bytes.Buffer
			if code := runAgentic(&out, &errOut, tt.args); code != 2 {
				t.Fatalf("code=%d, want 2; stdout=%s stderr=%s", code, out.String(), errOut.String())
			}
			if out.Len() != 0 || !strings.Contains(errOut.String(), tt.want) {
				t.Fatalf("stdout=%q stderr=%q, want stderr containing %q", out.String(), errOut.String(), tt.want)
			}
		})
	}
}

func TestAgenticBroad9896DogfoodIsMixedBoundedScopedAndByteStable(t *testing.T) {
	objective := "Build 100x better agentic performance processes across development and native model running: smaller issue-sized work, concurrent experiments, expand-contract cycles, outcome learning, OSS research, trigger graphs, and typed effects."
	args := []string{"--json", "--objective", objective}
	var first, firstErr bytes.Buffer
	if code := runAgentic(&first, &firstErr, args); code != 0 {
		t.Fatalf("first code=%d stderr=%s", code, firstErr.String())
	}
	var second, secondErr bytes.Buffer
	if code := runAgentic(&second, &secondErr, args); code != 0 {
		t.Fatalf("second code=%d stderr=%s", code, secondErr.String())
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatalf("repeated dogfood JSON differs:\n%s\n%s", first.Bytes(), second.Bytes())
	}
	var plan agentic.Plan
	if err := json.Unmarshal(first.Bytes(), &plan); err != nil {
		t.Fatal(err)
	}
	if plan.Inference.Domain != agentic.DomainMixed || len(plan.Cohorts) != 2 {
		t.Fatalf("mixed plan = inference=%+v cohorts=%+v", plan.Inference, plan.Cohorts)
	}
	if plan.Native == nil || plan.Native.Engine != "fak-native" || plan.Native.DefaultModel != "Qwen3.8" || plan.Native.Receipt.Schema != "fak-native-performance-receipt/v2" {
		t.Fatalf("native contract = %+v", plan.Native)
	}
	directionCount := 0
	cohortExperiments := map[string]int{}
	for _, unit := range plan.WorkUnits {
		if unit.Stage == "expand" {
			directionCount++
		}
		if unit.Stage != "experiment" {
			continue
		}
		cohortExperiments[unit.Cohort]++
		if unit.Witness.SelectionEvidence == "" || unit.Witness.RejectionEvidence == "" || len(unit.DependsOn) == 0 {
			t.Fatalf("experiment omits direction decision evidence: %+v", unit)
		}
		encoded, err := json.Marshal(unit)
		if err != nil {
			t.Fatal(err)
		}
		if unit.Cohort == agentic.DomainDevelopment && containsAnyString(string(encoded), "Qwen3.8", "fak-native", "receipt v2", "sanctioned hardware", "llama.cpp") {
			t.Fatalf("native controls leaked into development unit %s: %s", unit.ID, encoded)
		}
	}
	if directionCount < 3 || directionCount > 6 || directionCount > plan.Bounds.MaxDirections {
		t.Fatalf("directions=%d bounds=%+v", directionCount, plan.Bounds)
	}
	if cohortExperiments[agentic.DomainDevelopment] == 0 || cohortExperiments[agentic.DomainNativeModel] == 0 {
		t.Fatalf("cohort experiments = %v", cohortExperiments)
	}
}

func containsAnyString(s string, values ...string) bool {
	for _, value := range values {
		if strings.Contains(s, value) {
			return true
		}
	}
	return false
}
