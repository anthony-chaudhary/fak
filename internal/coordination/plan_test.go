package coordination

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

func validInput() Input {
	return Input{
		HarnessIntent: HarnessIntent{
			Kind:    HarnessKindFakNative,
			Task:    "run the agent",
			Outcome: "the task is complete",
		},
		ContextState: ContextState{
			Pressure: 0.2,
		},
		ComputeState: ComputeState{
			Engine:    "fak_native",
			Available: true,
		},
		ServeState: ServeState{
			Admitted:     true,
			Backpressure: 0.2,
		},
	}
}

func TestBuildRefusesInvalidInput(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Input)
	}{
		{name: "unknown harness", mutate: func(in *Input) { in.HarnessIntent.Kind = "unknown" }},
		{name: "empty task", mutate: func(in *Input) { in.HarnessIntent.Task = "" }},
		{name: "whitespace task", mutate: func(in *Input) { in.HarnessIntent.Task = " \t\n" }},
		{name: "empty outcome", mutate: func(in *Input) { in.HarnessIntent.Outcome = "" }},
		{name: "whitespace outcome", mutate: func(in *Input) { in.HarnessIntent.Outcome = " \t\n" }},
		{name: "negative context pressure", mutate: func(in *Input) { in.ContextState.Pressure = -0.01 }},
		{name: "context pressure above one", mutate: func(in *Input) { in.ContextState.Pressure = 1.01 }},
		{name: "NaN context pressure", mutate: func(in *Input) { in.ContextState.Pressure = math.NaN() }},
		{name: "negative serve backpressure", mutate: func(in *Input) { in.ServeState.Backpressure = -0.01 }},
		{name: "serve backpressure above one", mutate: func(in *Input) { in.ServeState.Backpressure = 1.01 }},
		{name: "NaN serve backpressure", mutate: func(in *Input) { in.ServeState.Backpressure = math.NaN() }},
		{name: "wrong engine", mutate: func(in *Input) { in.ComputeState.Engine = "other" }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := validInput()
			tt.mutate(&input)
			plan := Build(input)
			if plan.Disposition != PlanDispositionRefuse {
				t.Fatalf("disposition = %q, want %q", plan.Disposition, PlanDispositionRefuse)
			}
			if !strings.HasPrefix(plan.FusedValue, "refused: ") {
				t.Fatalf("refusal explanation = %q, want refused prefix", plan.FusedValue)
			}
			assertEvidenceContract(t, plan)
		})
	}
}

func TestBuildPressureRangeBoundariesAreValid(t *testing.T) {
	tests := []struct {
		name     string
		pressure float64
	}{
		{name: "zero", pressure: 0},
		{name: "defer threshold", pressure: 0.8},
		{name: "one", pressure: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := validInput()
			input.ContextState.Pressure = tt.pressure
			input.ServeState.Backpressure = tt.pressure
			plan := Build(input)
			want := PlanDispositionExecute
			if tt.pressure > 0.8 {
				want = PlanDispositionDefer
			}
			if plan.Disposition != want {
				t.Fatalf("Build with pressure %v disposition = %q, want %q", tt.pressure, plan.Disposition, want)
			}
		})
	}
}

func TestBuildExecutesSupportedHarnesses(t *testing.T) {
	for _, kind := range []HarnessKind{HarnessKindFakNative, HarnessKindExternal} {
		t.Run(string(kind), func(t *testing.T) {
			input := validInput()
			input.HarnessIntent.Kind = kind
			plan := Build(input)
			if plan.Disposition != PlanDispositionExecute {
				t.Fatalf("disposition = %q, want %q", plan.Disposition, PlanDispositionExecute)
			}
			if plan.SelectedHarness != kind {
				t.Fatalf("selected harness = %q, want %q", plan.SelectedHarness, kind)
			}
			if plan.SelectedEngine != "fak_native" {
				t.Fatalf("selected engine = %q, want fak_native", plan.SelectedEngine)
			}
			assertEvidenceContract(t, plan)
		})
	}
}

func TestBuildDefersWhenCapacityCannotServe(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Input)
		want   string
	}{
		{name: "compute unavailable", mutate: func(in *Input) { in.ComputeState.Available = false }, want: "compute unavailable"},
		{name: "serve not admitted", mutate: func(in *Input) { in.ServeState.Admitted = false }, want: "serve not admitted"},
		{name: "context pressure above threshold", mutate: func(in *Input) { in.ContextState.Pressure = 0.81 }, want: "high cache pressure"},
		{name: "serve backpressure above threshold", mutate: func(in *Input) { in.ServeState.Backpressure = 0.81 }, want: "high serve backpressure"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := validInput()
			tt.mutate(&input)
			plan := Build(input)
			if plan.Disposition != PlanDispositionDefer {
				t.Fatalf("disposition = %q, want %q", plan.Disposition, PlanDispositionDefer)
			}
			if !strings.Contains(plan.FusedValue, tt.want) {
				t.Fatalf("defer explanation = %q, want it to contain %q", plan.FusedValue, tt.want)
			}
			assertEvidenceContract(t, plan)
		})
	}
}

func TestBuildManagedReusablePrefixControlsReuse(t *testing.T) {
	tests := []struct {
		name    string
		managed bool
		bytes   int
		reuse   bool
		want    string
	}{
		{name: "managed positive prefix", managed: true, bytes: 12345, reuse: true, want: "12345 bytes"},
		{name: "unmanaged positive prefix", managed: false, bytes: 12345, want: "without context reuse"},
		{name: "managed zero prefix", managed: true, bytes: 0, want: "without context reuse"},
		{name: "managed negative prefix", managed: true, bytes: -1, want: "without context reuse"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := validInput()
			input.ContextState.Managed = tt.managed
			input.ContextState.ReusablePrefixBytes = tt.bytes
			plan := Build(input)
			if plan.ContextReuse != tt.reuse {
				t.Fatalf("context reuse = %t, want %t", plan.ContextReuse, tt.reuse)
			}
			if !strings.Contains(plan.FusedValue, tt.want) {
				t.Fatalf("explanation = %q, want it to contain %q", plan.FusedValue, tt.want)
			}
		})
	}
}

func TestBuildAlwaysRequiresOutcomeAndEffectsEvidence(t *testing.T) {
	for _, requirements := range []EvidenceRequirements{
		{},
		{RequireOutcome: true},
		{RequireEffects: true},
		{RequireOutcome: true, RequireEffects: true},
	} {
		input := validInput()
		input.EvidenceRequirements = requirements
		assertEvidenceContract(t, Build(input))
	}
}

func TestPlanJSONIsDeterministic(t *testing.T) {
	input := validInput()
	input.ContextState.Managed = true
	input.ContextState.ReusablePrefixBytes = 1234

	want := `{"disposition":"execute","fusedValueExplanation":"execute: fak-native harness with context reuse (1234 bytes)","selectedHarness":"fak_native","selectedEngine":"fak_native","contextReuse":true,"requiredEvidence":["agent_outcome","effects"],"rawModelEvidenceSufficient":false}`
	for i := 0; i < 10; i++ {
		got, err := json.Marshal(Build(input))
		if err != nil {
			t.Fatalf("marshal plan: %v", err)
		}
		if string(got) != want {
			t.Fatalf("JSON = %s, want %s", got, want)
		}
	}
}

func assertEvidenceContract(t *testing.T, plan Plan) {
	t.Helper()
	wantEvidence := []string{"agent_outcome", "effects"}
	if !reflect.DeepEqual(plan.RequiredEvidence, wantEvidence) {
		t.Fatalf("required evidence = %#v, want %#v", plan.RequiredEvidence, wantEvidence)
	}
	if plan.RawModelEvidenceSufficient {
		t.Fatal("raw model evidence unexpectedly marked sufficient")
	}
}
