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
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			task := completeFormalTask()
			tc.mutate(task.FormalPacket)
			assertAstraRefusal(t, task, tc.reason)
		})
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
		{"absolute", []string{`C:\\workspace\\file.go`}, AstraReasonSurfaceUnbounded},
		{"wildcard region", []string{"internal/*/file.go"}, AstraReasonSurfaceUnbounded},
		{"duplicate", []string{"internal/orchestration", `internal\\orchestration`}, AstraReasonSurfaceDuplicate},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			task := completeFormalTask()
			task.FormalPacket.Surfaces = tc.surfaces
			assertAstraRefusal(t, task, tc.reason)
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
	}
	route := AssessAstraRoute(task)
	want := []AstraRouteReason{
		AstraReasonSchemaInvalid,
		AstraReasonWorkClassRequired,
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
