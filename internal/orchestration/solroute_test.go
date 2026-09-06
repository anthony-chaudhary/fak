package orchestration

import "testing"

func TestSelectSOLRoute(t *testing.T) {
	tests := []struct {
		name        string
		task        string
		profile     Profile
		workClass   WorkClass
		wantMode    SOLMode
		wantEffort  string
		wantMulti   bool
		wantConsult bool
	}{
		{name: "small issue", task: "fix the typo", profile: ProfileAuto, workClass: WorkDefault, wantMode: SOLStandard, wantEffort: "high"},
		{name: "rigor", task: "audit an uncertain security invariant", profile: ProfileAuto, workClass: WorkRigor, wantMode: SOLMax, wantEffort: "xhigh"},
		{name: "wave", task: "run a fleet wave over independent issues", profile: ProfileAuto, workClass: WorkGrind, wantMode: SOLUltra, wantEffort: "high", wantMulti: true},
		{name: "ultracode profile", task: "implement the feature", profile: ProfileUltracode, workClass: WorkGrind, wantMode: SOLUltra, wantEffort: "high", wantMulti: true},
		{name: "parallel rigor", task: "audit an uncertain security invariant in parallel", profile: ProfileAuto, workClass: WorkRigor, wantMode: SOLUltra, wantEffort: "xhigh", wantMulti: true},
		{name: "ultracode rigor", task: "verify the migration", profile: ProfileUltracode, workClass: WorkRigor, wantMode: SOLUltra, wantEffort: "xhigh", wantMulti: true},
		{name: "pro is consult only", task: "consult pro for an adversarial review", profile: ProfileAuto, workClass: WorkRigor, wantMode: SOLPro, wantEffort: "xhigh", wantConsult: true},
		{name: "off", task: "audit an uncertain invariant", profile: ProfileOff, workClass: WorkRigor, wantMode: SOLStandard, wantEffort: "medium"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SelectSOLRoute(tt.task, tt.profile, tt.workClass, "gpt-5.6-sol")
			if got.Mode != tt.wantMode || got.ReasoningEffort != tt.wantEffort || got.MultiAgent != tt.wantMulti || got.ConsultOnly != tt.wantConsult {
				t.Fatalf("SelectSOLRoute() = %+v", got)
			}
			if got.Mode == SOLPro && got.ReasoningMode != "pro" {
				t.Fatalf("Pro route reasoning mode = %q", got.ReasoningMode)
			}
		})
	}
}

func TestRouteResolutionPreservesEffortPin(t *testing.T) {
	// 1. Valid effort pin reaches SOLRoute.ReasoningEffort
	for _, effort := range []string{"low", "medium", "high", "xhigh"} {
		task := TaskSpec{
			ID:        "test-task",
			WorkClass: WorkDefault,
			Pins:      FastPins{Effort: effort},
		}
		res, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, task, HarnessCapabilities{})
		if err != nil {
			t.Fatalf("Resolve with effort %q: %v", effort, err)
		}
		RouteResolution(&res, "audit an uncertain security invariant", "gpt-5.6-sol")
		if res.Resolved.SOLRoute.ReasoningEffort != effort {
			t.Fatalf("effort pin = %q, got %q", effort, res.Resolved.SOLRoute.ReasoningEffort)
		}
	}

	// 2. Invalid effort pin is rejected
	badTask := TaskSpec{
		ID:        "bad-task",
		WorkClass: WorkDefault,
		Pins:      FastPins{Effort: "unsupported"},
	}
	if _, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, badTask, HarnessCapabilities{}); err == nil {
		t.Fatal("expected error on invalid effort pin, got nil")
	}
}

func TestSelectSOLRouteAstraManagerDefaults(t *testing.T) {
	for _, astraModel := range []string{"gpt-6-astra", "astra", "gpt-6", "openai/gpt-6-astra", "astra-gpt-6", "astra gpt 6", "gpt-6 astra"} {
		got := SelectSOLRoute("implement feature", ProfileUltracode, WorkGrind, astraModel)
		if got.WorkerModel != DefaultAstraChildWorkerModel {
			t.Errorf("model %q: got WorkerModel %q, want %q", astraModel, got.WorkerModel, DefaultAstraChildWorkerModel)
		}
		if got.WorkerReasoningEffort != DefaultAstraChildWorkerEffort {
			t.Errorf("model %q: got WorkerReasoningEffort %q, want %q", astraModel, got.WorkerReasoningEffort, DefaultAstraChildWorkerEffort)
		}
	}
}

func TestRouteResolutionAstraManagerPreservesExplicitWorkerOverrides(t *testing.T) {
	task := TaskSpec{
		ID:        "override-task",
		WorkClass: WorkGrind,
		Pins: FastPins{
			Model:  "custom-worker-model",
			Effort: "high",
		},
	}
	res, err := Resolve(OrchestrationProfile{Name: ProfileUltracode}, task, HarnessCapabilities{})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	RouteResolution(&res, "implement feature in parallel", "gpt-6-astra")
	if res.Resolved.SOLRoute.WorkerModel != "custom-worker-model" {
		t.Errorf("got WorkerModel %q, want custom-worker-model", res.Resolved.SOLRoute.WorkerModel)
	}
	if res.Resolved.SOLRoute.WorkerReasoningEffort != "high" {
		t.Errorf("got WorkerReasoningEffort %q, want high", res.Resolved.SOLRoute.WorkerReasoningEffort)
	}
}

func TestSelectSOLRouteSubagentsSignal(t *testing.T) {
	got := SelectSOLRoute("work on goal using sub agents", ProfileAuto, WorkDefault, "gpt-6-astra")
	if got.Mode != SOLUltra {
		t.Errorf("got Mode %q, want %q", got.Mode, SOLUltra)
	}
	if !got.MultiAgent {
		t.Errorf("got MultiAgent = false, want true")
	}
	if got.WorkerModel != DefaultAstraChildWorkerModel {
		t.Errorf("got WorkerModel %q, want %q", got.WorkerModel, DefaultAstraChildWorkerModel)
	}
	if got.WorkerReasoningEffort != DefaultAstraChildWorkerEffort {
		t.Errorf("got WorkerReasoningEffort %q, want %q", got.WorkerReasoningEffort, DefaultAstraChildWorkerEffort)
	}
}
