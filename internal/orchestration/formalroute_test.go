package orchestration

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func completeFormalTask() TaskSpec {
	return TaskSpec{
		Schema:    "fak-orchestration-task/1",
		ID:        "formal-proof",
		WorkClass: WorkRigor,
		FormalPacket: &FormalPacket{
			Schema:                    FormalPacketSchemaVersion,
			TaskKinds:                 []FormalTaskKind{FormalProof},
			DefinitionsAndAssumptions: "For finite state set S, transition relation R is total.",
			ExactProposition:          "Prove every reachable state has exactly one canonical successor.",
			RequiredOutputForm:        "DEFINITIONS, PROPOSITION, PROOF, COUNTEREXAMPLES, WITNESS",
			DeterministicWitness:      "go test ./internal/orchestration -run TestFormalPacket",
			Surfaces:                  []string{"internal/orchestration/**"},
		},
	}
}

func TestFormalPacketAutomaticallyRoutesAstra(t *testing.T) {
	raw, err := json.Marshal(completeFormalTask())
	if err != nil {
		t.Fatal(err)
	}
	task, err := ParseTask(raw)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, task, nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	if got.Resolved.AstraRoute == nil || !got.Resolved.AstraRoute.Eligible {
		t.Fatalf("astra route = %+v, want eligible", got.Resolved.AstraRoute)
	}
	if got.Resolved.SOLRoute.WorkerModel != AstraWorkerModel || got.Resolved.SOLRoute.WorkerReasoningEffort != AstraWorkerEffort {
		t.Fatalf("worker route = model %q effort %q, want %q/%q", got.Resolved.SOLRoute.WorkerModel, got.Resolved.SOLRoute.WorkerReasoningEffort, AstraWorkerModel, AstraWorkerEffort)
	}
	if got.Resolved.AstraRoute.Source != AstraRouteSourceFormalPacket || !got.Resolved.AstraRoute.Selected || got.Resolved.AstraRoute.Model != AstraWorkerModel {
		t.Fatalf("astra receipt = %+v", got.Resolved.AstraRoute)
	}
}

func TestFormalPacketAllowsEveryClosedTaskKind(t *testing.T) {
	for _, kind := range []FormalTaskKind{FormalProof, InvariantAudit, StateMachineAudit, NumericalCorrectnessDerivation} {
		t.Run(string(kind), func(t *testing.T) {
			task := completeFormalTask()
			task.FormalPacket.TaskKinds = []FormalTaskKind{kind}
			route := AssessAstraRoute(task)
			if route == nil || !route.Eligible || len(route.Reasons) != 0 {
				t.Fatalf("route = %+v, want eligible", route)
			}
		})
	}
	for _, surfaces := range [][]string{{"internal/orchestration"}, {"internal/orchestration", "internal/modelroute", "cmd/fak/orchestration.go"}} {
		task := completeFormalTask()
		task.FormalPacket.Surfaces = surfaces
		if route := AssessAstraRoute(task); route == nil || !route.Eligible {
			t.Fatalf("bounded surfaces %v refused: %+v", surfaces, route)
		}
	}
}

func TestFormalPacketFailsClosedForEveryRequiredField(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*FormalPacket)
		reason AstraRouteReason
	}{
		{"schema", func(p *FormalPacket) { p.Schema = "" }, AstraReasonSchemaInvalid},
		{"task kinds", func(p *FormalPacket) { p.TaskKinds = nil }, AstraReasonTaskKindRequired},
		{"definitions and assumptions", func(p *FormalPacket) { p.DefinitionsAndAssumptions = "" }, AstraReasonDefinitionsAndAssumptions},
		{"exact proposition", func(p *FormalPacket) { p.ExactProposition = "" }, AstraReasonExactProposition},
		{"required output form", func(p *FormalPacket) { p.RequiredOutputForm = "" }, AstraReasonRequiredOutputForm},
		{"deterministic witness", func(p *FormalPacket) { p.DeterministicWitness = "" }, AstraReasonDeterministicWitness},
		{"surfaces", func(p *FormalPacket) { p.Surfaces = nil }, AstraReasonSurfaceCount},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			task := completeFormalTask()
			tc.mutate(task.FormalPacket)
			assertAstraRefusal(t, task, tc.reason)
		})
	}
}

func TestFormalPacketSchemaRequiresExactVersionIdentity(t *testing.T) {
	for _, schema := range []string{" fak-formal-packet/1", "fak-formal-packet/1 ", "FAK-FORMAL-PACKET/1", "fak-formal-packet/2"} {
		t.Run(schema, func(t *testing.T) {
			task := completeFormalTask()
			task.FormalPacket.Schema = schema
			assertAstraRefusal(t, task, AstraReasonSchemaInvalid)
		})
	}
}

func TestFormalPacketRejectsKindsOutsideExclusiveClosedSet(t *testing.T) {
	cases := []struct {
		name   string
		kinds  []FormalTaskKind
		reason AstraRouteReason
	}{
		{"blank", []FormalTaskKind{""}, AstraReasonTaskKindRequired},
		{"duplicate", []FormalTaskKind{FormalProof, FormalProof}, AstraReasonTaskKindDuplicate},
		{"case folded duplicate", []FormalTaskKind{FormalProof, "FORMAL_PROOF"}, AstraReasonTaskKindDuplicate},
		{"excluded", []FormalTaskKind{FormalProof, "implementation"}, AstraReasonTaskKindExcluded},
		{"unknown", []FormalTaskKind{"difficult_coding"}, AstraReasonTaskKindUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			task := completeFormalTask()
			task.FormalPacket.TaskKinds = tc.kinds
			assertAstraRefusal(t, task, tc.reason)
		})
	}
	for _, kind := range []FormalTaskKind{"documentation", "exploration", "general_review", "implementation", "issue_triage", "test_execution"} {
		t.Run("excluded-"+string(kind), func(t *testing.T) {
			task := completeFormalTask()
			task.FormalPacket.TaskKinds = []FormalTaskKind{FormalProof, kind}
			assertAstraRefusal(t, task, AstraReasonTaskKindExcluded)
		})
	}
}

func TestFormalPacketRejectsPlaceholders(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*FormalPacket)
		reason AstraRouteReason
	}{
		{"definitions", func(p *FormalPacket) { p.DefinitionsAndAssumptions = "TBD" }, AstraReasonDefinitionsAndAssumptions},
		{"proposition", func(p *FormalPacket) { p.ExactProposition = "<TODO>" }, AstraReasonExactProposition},
		{"output", func(p *FormalPacket) { p.RequiredOutputForm = "fill this in" }, AstraReasonRequiredOutputForm},
		{"witness", func(p *FormalPacket) { p.DeterministicWitness = "placeholder" }, AstraReasonDeterministicWitness},
		{"todo prefix", func(p *FormalPacket) { p.ExactProposition = "TODO: later" }, AstraReasonExactProposition},
		{"tbd token", func(p *FormalPacket) { p.RequiredOutputForm = "proof; TBD pending format" }, AstraReasonRequiredOutputForm},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			task := completeFormalTask()
			tc.mutate(task.FormalPacket)
			assertAstraRefusal(t, task, tc.reason)
		})
	}

	task := completeFormalTask()
	task.FormalPacket.DefinitionsAndAssumptions = "The methodology fixes all definitions before proof."
	if route := AssessAstraRoute(task); route == nil || !route.Eligible {
		t.Fatalf("placeholder substring false positive: %+v", route)
	}
}

func TestFormalPacketRequiresOneToThreeDistinctBoundedSurfaces(t *testing.T) {
	cases := []struct {
		name     string
		surfaces []string
		reason   AstraRouteReason
	}{
		{"none", nil, AstraReasonSurfaceCount},
		{"too many", []string{"a", "b", "c", "d"}, AstraReasonSurfaceCount},
		{"repository root", []string{"."}, AstraReasonSurfaceUnbounded},
		{"parent escape", []string{"../other"}, AstraReasonSurfaceUnbounded},
		{"slash root", []string{"/x"}, AstraReasonSurfaceUnbounded},
		{"backslash root", []string{`\x`}, AstraReasonSurfaceUnbounded},
		{"unc", []string{`\\server\share`}, AstraReasonSurfaceUnbounded},
		{"windows absolute", []string{`C:\workspace\file.go`}, AstraReasonSurfaceUnbounded},
		{"windows drive relative", []string{`C:workspace\file.go`}, AstraReasonSurfaceUnbounded},
		{"wildcard region", []string{"internal/*/file.go"}, AstraReasonSurfaceUnbounded},
		{"duplicate", []string{"internal/orchestration", `internal\\orchestration`}, AstraReasonSurfaceDuplicate},
		{"canonical duplicate", []string{"internal/./orchestration", `INTERNAL\orchestration`}, AstraReasonSurfaceDuplicate},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			task := completeFormalTask()
			task.FormalPacket.Surfaces = tc.surfaces
			assertAstraRefusal(t, task, tc.reason)
		})
	}

	for _, surface := range []string{
		"internal/orchestration",
		"cmd/fak/orchestration.go",
		`internal\orchestration`,
		`cmd\fak\orchestration.go`,
		"internal/orchestration/**",
		`internal\orchestration\**`,
	} {
		t.Run("accept-"+surface, func(t *testing.T) {
			task := completeFormalTask()
			task.FormalPacket.Surfaces = []string{surface}
			if route := AssessAstraRoute(task); route == nil || !route.Eligible {
				t.Fatalf("bounded surface %q refused: %+v", surface, route)
			}
		})
	}
}

func TestFormalPacketRequiresRigorAndTaskTextCannotInventPacket(t *testing.T) {
	task := completeFormalTask()
	task.WorkClass = WorkGrind
	assertAstraRefusal(t, task, AstraReasonWorkClassRequired)

	fromText, err := TaskFromText("write a formal proof of the scheduler invariant with a deterministic witness")
	if err != nil {
		t.Fatal(err)
	}
	if fromText.FormalPacket != nil || AssessAstraRoute(fromText) != nil {
		t.Fatalf("task text invented formal eligibility: %+v", fromText)
	}
	resolved, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, fromText, nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Resolved.AstraRoute != nil || resolved.Resolved.SOLRoute.WorkerModel == AstraWorkerModel {
		t.Fatalf("task-text route selected Astra: %+v", resolved.Resolved)
	}
}

func TestFormalPacketExplicitTaskPinsWinAndReceiptNamesSource(t *testing.T) {
	task := completeFormalTask()
	task.Pins.Model = "gpt-5.6-sol"
	task.Pins.Effort = "high"
	got, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, task, nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	route := got.Resolved.AstraRoute
	if route == nil || !route.Eligible || route.Selected || route.Source != AstraRouteSourceTaskPin || route.Model != task.Pins.Model || route.ReasoningEffort != task.Pins.Effort || route.ReasoningEffortSource != AstraRouteSourceTaskPin {
		t.Fatalf("astra receipt did not preserve explicit pins: %+v", route)
	}
	if got.Resolved.SOLRoute.WorkerModel != task.Pins.Model || got.Resolved.SOLRoute.WorkerReasoningEffort != task.Pins.Effort {
		t.Fatalf("effective worker route ignored pins: %+v", got.Resolved.SOLRoute)
	}

	// An explicit Astra pin remains authoritative even when the packet itself is
	// ineligible; the receipt must not relabel the override as automatic admission.
	task.FormalPacket.TaskKinds = []FormalTaskKind{"implementation"}
	task.Pins.Model = AstraWorkerModel
	got, err = Resolve(OrchestrationProfile{Name: ProfileAuto}, task, nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	route = got.Resolved.AstraRoute
	if route == nil || route.Eligible || !route.Selected || route.Source != AstraRouteSourceTaskPin || route.Model != AstraWorkerModel {
		t.Fatalf("explicit Astra override was misattributed: %+v", route)
	}
}

func TestFormalPacketSeparateTaskPinsPreserveReceiptProvenance(t *testing.T) {
	t.Run("model only Astra alias", func(t *testing.T) {
		task := completeFormalTask()
		task.Pins.Model = "openai/astra"
		got, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, task, nativeCaps())
		if err != nil {
			t.Fatal(err)
		}
		route := got.Resolved.AstraRoute
		if route == nil || !route.Eligible || !route.Selected || route.Source != AstraRouteSourceTaskPin || route.Model != task.Pins.Model {
			t.Fatalf("model-only alias receipt = %+v", route)
		}
		if route.ReasoningEffort != AstraWorkerEffort || route.ReasoningEffortSource != AstraRouteSourceFormalPacket {
			t.Fatalf("model-only pin changed automatic effort provenance: %+v", route)
		}
	})

	t.Run("effort only", func(t *testing.T) {
		task := completeFormalTask()
		task.Pins.Effort = "high"
		got, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, task, nativeCaps())
		if err != nil {
			t.Fatal(err)
		}
		route := got.Resolved.AstraRoute
		if route == nil || !route.Eligible || !route.Selected || route.Source != AstraRouteSourceFormalPacket || route.Model != AstraWorkerModel {
			t.Fatalf("effort-only receipt changed model provenance: %+v", route)
		}
		if route.ReasoningEffort != task.Pins.Effort || route.ReasoningEffortSource != AstraRouteSourceTaskPin {
			t.Fatalf("effort-only receipt = %+v", route)
		}
	})
}

func TestFormalPacketTaskPinsReplaceAutomaticProvenance(t *testing.T) {
	type wantProvenance struct {
		source string
		value  string
	}
	tests := []struct {
		name   string
		model  string
		effort string
		want   map[string]wantProvenance
	}{
		{
			name:  "model only",
			model: AstraWorkerModel,
			want: map[string]wantProvenance{
				"sol_route.worker_model":            {AstraRouteSourceTaskPin, AstraWorkerModel},
				"sol_route.worker_reasoning_effort": {AstraRouteSourceFormalPacket, AstraWorkerEffort},
			},
		},
		{
			name:   "effort only",
			effort: "high",
			want: map[string]wantProvenance{
				"sol_route.worker_model":            {AstraRouteSourceFormalPacket, AstraWorkerModel},
				"sol_route.worker_reasoning_effort": {AstraRouteSourceTaskPin, "high"},
			},
		},
		{
			name:   "both",
			model:  "gpt-5.6-sol",
			effort: "medium",
			want: map[string]wantProvenance{
				"sol_route.worker_model":            {AstraRouteSourceTaskPin, "gpt-5.6-sol"},
				"sol_route.worker_reasoning_effort": {AstraRouteSourceTaskPin, "medium"},
			},
		},
		{
			name:  "Astra alias",
			model: "openai/astra",
			want: map[string]wantProvenance{
				"sol_route.worker_model":            {AstraRouteSourceTaskPin, "openai/astra"},
				"sol_route.worker_reasoning_effort": {AstraRouteSourceFormalPacket, AstraWorkerEffort},
			},
		},
		{
			name:  "non-Astra",
			model: "gpt-5.6-terra",
			want: map[string]wantProvenance{
				"sol_route.worker_model":            {AstraRouteSourceTaskPin, "gpt-5.6-terra"},
				"sol_route.worker_reasoning_effort": {AstraRouteSourceFormalPacket, AstraWorkerEffort},
			},
		},
	}

	baseline, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, completeFormalTask(), nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	unrelated := func(all []Provenance) []Provenance {
		kept := make([]Provenance, 0, len(all))
		for _, entry := range all {
			if entry.Field != "sol_route.worker_model" && entry.Field != "sol_route.worker_reasoning_effort" {
				kept = append(kept, entry)
			}
		}
		return kept
	}
	baselineUnrelated := unrelated(baseline.Overrides)

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			task := completeFormalTask()
			task.Pins.Model = tc.model
			task.Pins.Effort = tc.effort
			got, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, task, nativeCaps())
			if err != nil {
				t.Fatal(err)
			}
			if gotUnrelated := unrelated(got.Overrides); !reflect.DeepEqual(gotUnrelated, baselineUnrelated) {
				t.Fatalf("unrelated provenance order changed\ngot:  %+v\nwant: %+v", gotUnrelated, baselineUnrelated)
			}
			for field, want := range tc.want {
				count := 0
				var entry Provenance
				for _, candidate := range got.Overrides {
					if candidate.Field == field {
						count++
						entry = candidate
					}
				}
				if count != 1 || entry.Source != want.source || entry.Value != want.value {
					t.Fatalf("provenance %q count=%d entry=%+v, want exactly one source=%q value=%q", field, count, entry, want.source, want.value)
				}
			}
		})
	}
}

func TestFormalPacketRouteAuthorityInvariant(t *testing.T) {
	tests := []struct {
		name         string
		profile      Profile
		configure    func(*TaskSpec)
		eligible     bool
		selected     bool
		model        string
		effort       string
		modelSource  string
		effortSource string
		reason       AstraRouteReason
	}{
		{"auto", ProfileAuto, nil, true, true, AstraWorkerModel, AstraWorkerEffort, AstraRouteSourceFormalPacket, AstraRouteSourceFormalPacket, ""},
		{"off direct", ProfileOff, nil, true, false, AstraWorkerModel, AstraWorkerEffort, AstraRouteSourceFormalPacket, AstraRouteSourceFormalPacket, ""},
		{"fast launch declined", ProfileFast, func(task *TaskSpec) {
			task.FastIntent = &FastIntent{Schema: FastIntentSchemaVersion, LatencyClass: "interactive", QualityFloor: "accepted", CostCeiling: "bounded", CachePolicy: "preserve", FallbackPolicy: "degrade"}
		}, true, false, AstraWorkerModel, AstraWorkerEffort, AstraRouteSourceFormalPacket, AstraRouteSourceFormalPacket, ""},
		{"canonical Astra task pin", ProfileAuto, func(task *TaskSpec) { task.Pins.Model = AstraWorkerModel }, true, true, AstraWorkerModel, AstraWorkerEffort, AstraRouteSourceTaskPin, AstraRouteSourceFormalPacket, ""},
		{"alias Astra task pin", ProfileAuto, func(task *TaskSpec) { task.Pins.Model = "openai/astra" }, true, true, "openai/astra", AstraWorkerEffort, AstraRouteSourceTaskPin, AstraRouteSourceFormalPacket, ""},
		{"non-Astra task pin", ProfileAuto, func(task *TaskSpec) { task.Pins.Model = "gpt-5.6-sol" }, true, false, "gpt-5.6-sol", AstraWorkerEffort, AstraRouteSourceTaskPin, AstraRouteSourceFormalPacket, ""},
		{"effort-only task pin", ProfileAuto, func(task *TaskSpec) { task.Pins.Effort = "high" }, true, true, AstraWorkerModel, "high", AstraRouteSourceFormalPacket, AstraRouteSourceTaskPin, ""},
		{"explicit observe access", ProfileAuto, func(task *TaskSpec) {
			task.WorkerAccess = []WorkerAccessSpec{{RoleID: "worker-1", Access: ChildAccess{Mode: ChildAccessObserve}}}
		}, true, true, AstraWorkerModel, AstraWorkerEffort, AstraRouteSourceFormalPacket, AstraRouteSourceFormalPacket, ""},
		{"effect access", ProfileAuto, func(task *TaskSpec) {
			task.WorkerAccess = []WorkerAccessSpec{{RoleID: "worker-1", Access: ChildAccess{Mode: ChildAccessEffect, WriteSet: []string{"internal/orchestration"}}}}
		}, false, false, "", "", "", "", AstraReasonAnalysisOnly},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			task := completeFormalTask()
			if tc.configure != nil {
				tc.configure(&task)
			}
			got, err := Resolve(OrchestrationProfile{Name: tc.profile}, task, nativeCaps())
			if err != nil {
				t.Fatal(err)
			}
			route := got.Resolved.AstraRoute
			if route == nil || route.Eligible != tc.eligible || route.Selected != tc.selected || route.Model != tc.model || route.ReasoningEffort != tc.effort {
				t.Fatalf("route = %+v, want eligible=%v selected=%v model=%q effort=%q", route, tc.eligible, tc.selected, tc.model, tc.effort)
			}
			if tc.reason != "" && !containsAstraReason(route.Reasons, tc.reason) {
				t.Fatalf("route reasons = %v, want %q", route.Reasons, tc.reason)
			}

			lastProvenance := func(field string) (Provenance, bool) {
				for i := len(got.Overrides) - 1; i >= 0; i-- {
					if got.Overrides[i].Field == field {
						return got.Overrides[i], true
					}
				}
				return Provenance{}, false
			}
			if tc.profile != ProfileFast {
				for _, forbidden := range []string{"fast.model", "fast.effort"} {
					if _, ok := lastProvenance(forbidden); ok {
						t.Fatalf("formal route emitted forbidden provenance field %q: %+v", forbidden, got.Overrides)
					}
				}
			}
			for field, wantSource := range map[string]string{
				"sol_route.worker_model":            tc.modelSource,
				"sol_route.worker_reasoning_effort": tc.effortSource,
			} {
				entry, ok := lastProvenance(field)
				if wantSource == "" {
					if ok {
						t.Fatalf("provenance %q = %+v, want absent", field, entry)
					}
					continue
				}
				if !ok || entry.Source != wantSource {
					t.Fatalf("provenance %q = %+v present=%v, want source %q", field, entry, ok, wantSource)
				}
			}
		})
	}
}

func TestFormalPacketStableJSONRoundTripAndAbsentCompatibility(t *testing.T) {
	got, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, completeFormalTask(), nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	first, err := StableJSON(got)
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := ParseResolution(first)
	if err != nil {
		t.Fatal(err)
	}
	second, err := StableJSON(roundTrip)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) || roundTrip.Resolved.AstraRoute == nil || !reflect.DeepEqual(roundTrip.Resolved.AstraRoute, got.Resolved.AstraRoute) {
		t.Fatalf("unstable formal route\nfirst=%s\nsecond=%s", first, second)
	}

	absent, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, TaskSpec{Schema: "fak-orchestration-task/1", ID: "ordinary"}, nativeCaps())
	if err != nil {
		t.Fatal(err)
	}
	absentJSON, err := StableJSON(absent)
	if err != nil {
		t.Fatal(err)
	}
	if absent.Resolved.AstraRoute != nil || strings.Contains(string(absentJSON), "astra_route") {
		t.Fatalf("absent packet changed resolution shape: %s", absentJSON)
	}
	rawTask, err := json.Marshal(TaskSpec{Schema: "fak-orchestration-task/1", ID: "ordinary"})
	if err != nil || strings.Contains(string(rawTask), "formal_packet") {
		t.Fatalf("absent packet changed task shape: %s err=%v", rawTask, err)
	}
}

func TestFormalPacketReasonOrderIsClosedAndDeterministic(t *testing.T) {
	task := TaskSpec{
		WorkClass: WorkDefault,
		FormalPacket: &FormalPacket{
			TaskKinds: []FormalTaskKind{"implementation", "mystery", "implementation", ""},
			Surfaces:  []string{".", "../escape", "internal/orchestration", "internal/orchestration"},
		},
		WorkerAccess: []WorkerAccessSpec{{Access: ChildAccess{Mode: ChildAccessEffect}}},
	}
	route := AssessAstraRoute(task)
	want := []AstraRouteReason{
		AstraReasonSchemaInvalid,
		AstraReasonWorkClassRequired,
		AstraReasonAnalysisOnly,
		AstraReasonTaskKindRequired,
		AstraReasonTaskKindDuplicate,
		AstraReasonTaskKindExcluded,
		AstraReasonTaskKindUnknown,
		AstraReasonDefinitionsAndAssumptions,
		AstraReasonExactProposition,
		AstraReasonRequiredOutputForm,
		AstraReasonDeterministicWitness,
		AstraReasonSurfaceCount,
		AstraReasonSurfaceUnbounded,
		AstraReasonSurfaceDuplicate,
	}
	if route == nil || !reflect.DeepEqual(route.Reasons, want) {
		t.Fatalf("reasons = %v, want %v", route.Reasons, want)
	}
}

func containsAstraReason(reasons []AstraRouteReason, want AstraRouteReason) bool {
	for _, reason := range reasons {
		if reason == want {
			return true
		}
	}
	return false
}

func assertAstraRefusal(t *testing.T, task TaskSpec, reason AstraRouteReason) {
	t.Helper()
	route := AssessAstraRoute(task)
	if route == nil || route.Eligible || route.Selected {
		t.Fatalf("route = %+v, want refused", route)
	}
	for _, got := range route.Reasons {
		if got == reason {
			return
		}
	}
	t.Fatalf("reasons = %v, want %q", route.Reasons, reason)
}
