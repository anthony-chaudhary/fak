package agentic

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func TestCompileDevelopmentPlanIsBoundedIssueReadyAndOffline(t *testing.T) {
	plan, err := Compile("  Add a CLI and tests for internal/widget/widget.go and cmd/fak/widget.go.  ")
	if err != nil {
		t.Fatal(err)
	}
	if plan.Schema != Schema || plan.Learning.Schema != LearningSchema {
		t.Fatalf("schemas = %q, %q", plan.Schema, plan.Learning.Schema)
	}
	if plan.Objective != "Add a CLI and tests for internal/widget/widget.go and cmd/fak/widget.go." {
		t.Fatalf("normalized objective = %q", plan.Objective)
	}
	if plan.Inference.Domain != DomainDevelopment || plan.Native != nil {
		t.Fatalf("development classification = %+v, native=%+v", plan.Inference, plan.Native)
	}
	if !plan.Mode.ReadOnly || !plan.Mode.Offline {
		t.Fatalf("planning mode = %+v", plan.Mode)
	}
	if len(plan.Stages) != 3 || plan.Stages[0].Name != "expand" || plan.Stages[1].Name != "experiment" || plan.Stages[2].Name != "contract" {
		t.Fatalf("stages = %+v", plan.Stages)
	}
	assertBoundedUnits(t, plan)
	if len(plan.WorkUnits) < 3 || plan.WorkUnits[len(plan.WorkUnits)-1].Stage != "contract" {
		t.Fatalf("work units omit the contract stage: %+v", plan.WorkUnits)
	}
	for _, unit := range plan.WorkUnits {
		if unit.ID == "" || unit.Title == "" || unit.Body == "" || unit.Witness.Class == "" || unit.Witness.Evidence == "" || unit.Witness.CommandHint == "" {
			t.Fatalf("unit is not issue-ready: %+v", unit)
		}
		for _, section := range []string{"## Objective", "## Bound", "## Scope hints", "## Witness"} {
			if !strings.Contains(unit.Body, section) {
				t.Fatalf("unit %s body missing %q:\n%s", unit.ID, section, unit.Body)
			}
		}
	}
	if got, want := plan.WorkUnits[0].ScopeHints[:2], []string{"cmd/fak/widget.go", "internal/widget/widget.go"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("sorted scope hints = %q, want %q", got, want)
	}
	if plan.Handoff.Model != "gpt-5.6-sol" || plan.Handoff.Profile != "ultracode" || !reflect.DeepEqual(plan.Handoff.Args, []string{"--task-text", plan.Objective, "--json"}) {
		t.Fatalf("handoff = %+v", plan.Handoff)
	}
	if plan.Learning.Outcome != "pending" || plan.Learning.Hypothesis == "" || plan.Learning.Adjustment == "" || plan.Learning.Evidence == nil {
		t.Fatalf("learning hook = %+v", plan.Learning)
	}
}

func TestCompileNativePlanPreservesExactExecutionContract(t *testing.T) {
	plan, err := Compile("Optimize Qwen3.8 CUDA decode throughput in internal/engine and benchmark the result")
	if err != nil {
		t.Fatal(err)
	}
	if plan.Inference.Domain != DomainNativeModel || plan.Native == nil {
		t.Fatalf("native classification = %+v, contract=%+v", plan.Inference, plan.Native)
	}
	native := plan.Native
	if native.Engine != "fak-native" || native.DefaultModel != "Qwen3.8" {
		t.Fatalf("native identity = engine %q model %q", native.Engine, native.DefaultModel)
	}
	if native.Receipt.Schema != nativeperf.ReceiptSchemaV2 || !native.Receipt.Required {
		t.Fatalf("receipt = %+v", native.Receipt)
	}
	wantFields := []string{
		"schema", "role", "envelope_id", "changed_lever_id", "revision",
		"artifact_sha256", "machine", "controls", "unchanged_controls",
		"changed_axes", "repetitions", "memory", "execution", "quality",
		"module_versions", "commands", "profiler_artifacts",
	}
	if !reflect.DeepEqual(native.Receipt.RequiredFields, wantFields) {
		t.Fatalf("canonical receipt fields = %q, want %q", native.Receipt.RequiredFields, wantFields)
	}
	for _, alias := range []string{"engine", "model", "artifact", "hardware", "request", "load", "state", "outcome", "setup", "recovery", "verification"} {
		if containsString(native.Receipt.RequiredFields, alias) {
			t.Fatalf("conceptual alias %q leaked into canonical receipt fields", alias)
		}
	}
	if native.Fallback.Engine != "llama.cpp" || native.Fallback.SilentFallback || len(native.Fallback.ExplicitExceptions) != 3 {
		t.Fatalf("fallback contract = %+v", native.Fallback)
	}
	if !native.MatchedQuality || !native.MatchedWorkload || !native.MatchedOperatingEnvelope {
		t.Fatalf("matched envelope contract = %+v", native)
	}
	if native.Accounting.Method != "net_true" || !reflect.DeepEqual(native.Accounting.Includes, []string{"setup", "recovery", "verification"}) {
		t.Fatalf("accounting = %+v", native.Accounting)
	}
	if !native.Hardware.Required || native.Hardware.Reference != "docs/fleet-compute-nodes.md" {
		t.Fatalf("hardware = %+v", native.Hardware)
	}
	assertBoundedUnits(t, plan)
	stageCounts := map[string]int{}
	for _, unit := range plan.WorkUnits {
		stageCounts[unit.Stage]++
	}
	if stageCounts["expand"] < 3 || stageCounts["expand"] > 6 || stageCounts["experiment"] != 2 || stageCounts["contract"] != 1 {
		t.Fatalf("native stage counts = %v", stageCounts)
	}
	if plan.WorkUnits[len(plan.WorkUnits)-1].Witness.Class != "contract" {
		t.Fatalf("native contract witness missing: %+v", plan.WorkUnits)
	}
}

func TestCompileNativeTechnologyWithoutExecutionIntentStaysDevelopment(t *testing.T) {
	tests := []string{
		"Document Qwen3.8 configuration and receipt v2 fields",
		"Add a CUDA model selector to the API client UI",
		"Update GGUF configuration for Qwen3.8",
		"Build dashboard for GPU performance metrics",
		"Improve docs for Qwen3.8 performance settings",
		"Document native model running configuration",
		"API client fields for Qwen3.8 benchmark results",
	}
	for _, objective := range tests {
		t.Run(objective, func(t *testing.T) {
			plan, err := Compile(objective)
			if err != nil {
				t.Fatal(err)
			}
			if plan.Inference.Domain != DomainDevelopment || !reflect.DeepEqual(plan.Inference.Domains, []string{DomainDevelopment}) || plan.Native != nil {
				t.Fatalf("technology noun triggered native execution: inference=%+v native=%+v", plan.Inference, plan.Native)
			}
			for _, unit := range plan.WorkUnits {
				if unit.Cohort == DomainNativeModel {
					t.Fatalf("development objective emitted native unit: %+v", unit)
				}
			}
		})
	}
}

func TestCompileDevelopmentSurfaceWithExecutionIntentBecomesMixed(t *testing.T) {
	for _, objective := range []string{
		"Build an API client UI that runs Qwen3.8 and benchmarks the model",
		"Build a UI to benchmark Qwen3.8 throughput",
		"Benchmark Qwen3.8 through the API client",
	} {
		plan, err := Compile(objective)
		if err != nil {
			t.Fatal(err)
		}
		if plan.Inference.Domain != DomainMixed || !reflect.DeepEqual(plan.Inference.Domains, []string{DomainDevelopment, DomainNativeModel}) {
			t.Fatalf("execution-bearing development surface %q = %+v", objective, plan.Inference)
		}
	}
}

func TestCompileMixedDogfoodPreservesCohortsDirectionsAndContracts(t *testing.T) {
	objective := "Build 100x better agentic performance processes across development and native model running: smaller issue-sized work, concurrent experiments, expand-contract cycles, outcome learning, OSS research, trigger graphs, and typed effects."
	plan, err := Compile(objective)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Inference.Domain != DomainMixed || !reflect.DeepEqual(plan.Inference.Domains, []string{DomainDevelopment, DomainNativeModel}) {
		t.Fatalf("mixed inference = %+v", plan.Inference)
	}
	if len(plan.Cohorts) != 2 || plan.Cohorts[0].Domain != DomainDevelopment || plan.Cohorts[1].Domain != DomainNativeModel {
		t.Fatalf("cohorts = %+v", plan.Cohorts)
	}
	if plan.Native == nil || plan.Native.Engine != "fak-native" || plan.Native.DefaultModel != "Qwen3.8" || plan.Native.Receipt.Schema != nativeperf.ReceiptSchemaV2 {
		t.Fatalf("native contract = %+v", plan.Native)
	}

	var directionIDs []string
	candidates := map[string]bool{}
	for _, unit := range plan.WorkUnits {
		if unit.Stage != "expand" {
			continue
		}
		directionIDs = append(directionIDs, unit.ID)
		if unit.Direction == nil || unit.Direction.Candidate == "" || unit.Direction.Question == "" || unit.Direction.Decision != "pending" {
			t.Fatalf("direction is not explicit: %+v", unit)
		}
		if candidates[unit.Direction.Candidate] {
			t.Fatalf("duplicate candidate direction %q", unit.Direction.Candidate)
		}
		candidates[unit.Direction.Candidate] = true
		if unit.Witness.SelectionEvidence == "" || unit.Witness.RejectionEvidence == "" {
			t.Fatalf("direction omits decision evidence: %+v", unit.Witness)
		}
	}
	if len(directionIDs) < 3 || len(directionIDs) > 6 || len(directionIDs) > plan.Bounds.MaxDirections {
		t.Fatalf("candidate directions = %d bounds=%+v", len(directionIDs), plan.Bounds)
	}

	cohortExperiments := map[string]int{}
	directionIDsByCohort := map[string][]string{}
	for _, unit := range plan.WorkUnits {
		if unit.Stage == "expand" {
			directionIDsByCohort[unit.Cohort] = append(directionIDsByCohort[unit.Cohort], unit.ID)
		}
	}
	for _, unit := range plan.WorkUnits {
		switch unit.Stage {
		case "experiment":
			cohortExperiments[unit.Cohort]++
			if !reflect.DeepEqual(unit.DependsOn, directionIDsByCohort[unit.Cohort]) {
				t.Fatalf("experiment %s dependencies = %q, want cohort directions %q", unit.ID, unit.DependsOn, directionIDsByCohort[unit.Cohort])
			}
			if unit.Witness.SelectionEvidence == "" || unit.Witness.RejectionEvidence == "" {
				t.Fatalf("experiment %s omits direction decision evidence: %+v", unit.ID, unit.Witness)
			}
			encoded, err := json.Marshal(unit)
			if err != nil {
				t.Fatal(err)
			}
			scopedControls := string(encoded)
			if unit.Cohort == DomainDevelopment && containsAny(strings.ToLower(scopedControls), "qwen3.8", "fak-native", "receipt v2", "sanctioned hardware", "llama.cpp") {
				t.Fatalf("native controls leaked into development unit %s: %s", unit.ID, encoded)
			}
			if unit.Cohort == DomainNativeModel {
				for _, want := range []string{"fak-native", "Qwen3.8", "receipt v2"} {
					if !strings.Contains(scopedControls, want) {
						t.Fatalf("native unit %s missing %q: %s", unit.ID, want, scopedControls)
					}
				}
			}
		case "contract":
			if unit.Cohort != "all" || unit.Witness.SelectionEvidence == "" || unit.Witness.RejectionEvidence == "" {
				t.Fatalf("contract does not preserve direction decisions: %+v", unit)
			}
			wantDeps := append(append([]string{}, directionIDs...), experimentIDs(plan)...)
			if !reflect.DeepEqual(unit.DependsOn, wantDeps) {
				t.Fatalf("contract dependencies = %q, want direction and experiment evidence %q", unit.DependsOn, wantDeps)
			}
			for _, want := range []string{"development", "fak-native", "receipt v2", "matched-envelope", "net-true"} {
				if !strings.Contains(unit.Witness.Evidence, want) {
					t.Fatalf("contract evidence missing %q: %s", want, unit.Witness.Evidence)
				}
			}
		}
	}
	if cohortExperiments[DomainDevelopment] == 0 || cohortExperiments[DomainNativeModel] == 0 {
		t.Fatalf("mixed experiments = %v", cohortExperiments)
	}
	assertBoundedUnits(t, plan)

	first, err := Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	recompiled, err := Compile(objective)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Marshal(recompiled)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("dogfood JSON is not byte deterministic:\n%s\n%s", first, second)
	}
}

func experimentIDs(plan Plan) []string {
	var ids []string
	for _, unit := range plan.WorkUnits {
		if unit.Stage == "experiment" {
			ids = append(ids, unit.ID)
		}
	}
	return ids
}

func TestCompileAndMarshalAreByteDeterministic(t *testing.T) {
	objective := "Implement Bob's $(unsafe) JSON workflow in cmd/fak/example.go"
	first, err := Compile(objective)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compile(objective)
	if err != nil {
		t.Fatal(err)
	}
	a, err := Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Marshal(second)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("identical input produced different JSON:\n%s\n%s", a, b)
	}
	if !json.Valid(a) || a[len(a)-1] != '\n' {
		t.Fatalf("marshal did not emit newline-terminated JSON: %q", a)
	}
	if first.ObjectiveID != second.ObjectiveID || !strings.HasPrefix(first.ObjectiveID, "objective-") {
		t.Fatalf("objective IDs = %q, %q", first.ObjectiveID, second.ObjectiveID)
	}
	wantCommand := "fak ultracode --task-text 'Implement Bob'\"'\"'s $(unsafe) JSON workflow in cmd/fak/example.go' --json"
	if first.Handoff.Command != wantCommand {
		t.Fatalf("handoff command = %q, want %q", first.Handoff.Command, wantCommand)
	}
}

func TestCompileRejectsInvalidObjectives(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{name: "empty", in: " \t\n", want: "objective is required"},
		{name: "invalid UTF-8", in: string([]byte{0xff}), want: "objective must be valid UTF-8"},
		{name: "control character", in: "build\x00widget", want: "objective contains an unsupported control character"},
		{name: "too large", in: strings.Repeat("x", maxObjectiveBytes+1), want: "objective exceeds 16384 bytes"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Compile(tt.in)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Compile error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestCompileUnicodeSummaryStaysValidUTF8(t *testing.T) {
	plan, err := Compile("Implement " + strings.Repeat("界", 100) + " safely")
	if err != nil {
		t.Fatal(err)
	}
	for _, unit := range plan.WorkUnits {
		if !utf8.ValidString(unit.Title) {
			t.Fatalf("invalid UTF-8 title: %q", unit.Title)
		}
	}
}

func TestRenderIncludesDecisionsAndOfflinePosture(t *testing.T) {
	plan, err := Compile("Benchmark Qwen3.8 on a GPU")
	if err != nil {
		t.Fatal(err)
	}
	text := Render(plan)
	for _, want := range []string{
		"stage expand:", "stage experiment:", "stage contract:",
		"issue-ready work:", "model=gpt-5.6-sol", "engine=fak-native",
		"receipt=fak-native-performance-receipt/v2", "matched_workload=true",
		"llama.cpp silent=false", "net_true", "docs/fleet-compute-nodes.md",
		"schema=fak-agentic-learning/1", "read_only=true offline=true",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("render missing %q:\n%s", want, text)
		}
	}
}

func assertBoundedUnits(t *testing.T, plan Plan) {
	t.Helper()
	if plan.Bounds.MaxCycles != 1 || len(plan.WorkUnits) > plan.Bounds.MaxWorkUnits {
		t.Fatalf("global bounds violated: bounds=%+v units=%d", plan.Bounds, len(plan.WorkUnits))
	}
	stageMax := map[string]int{}
	for _, stage := range plan.Stages {
		if stage.MaxItems <= 0 || stage.MaxConcurrent <= 0 || stage.MaxConcurrent > stage.MaxItems || stage.MaxConcurrent > plan.Bounds.MaxConcurrency || stage.ExitCriterion == "" {
			t.Fatalf("invalid stage bound: %+v", stage)
		}
		stageMax[stage.Name] = stage.MaxItems
	}
	stageCounts := map[string]int{}
	for _, unit := range plan.WorkUnits {
		stageCounts[unit.Stage]++
		if unit.Bounds.MaxFiles <= 0 || unit.Bounds.MaxDeliverables <= 0 || unit.Bounds.MaxExperiments < 0 || unit.Bounds.MaxExperiments > plan.Bounds.MaxExperiments {
			t.Fatalf("invalid unit bounds: %+v", unit)
		}
	}
	for stage, count := range stageCounts {
		if count > stageMax[stage] {
			t.Fatalf("stage %s has %d units, max %d", stage, count, stageMax[stage])
		}
	}
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
