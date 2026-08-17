package toolcallcontrol

import (
	"encoding/json"
	"testing"
)

func TestRuntimeShadowObservesWithoutApplying(t *testing.T) {
	in := RuntimeInput{Tool: "Read", Args: json.RawMessage(`{"file_path":"a"}`), ReadOnly: true, CallID: "c1", Succeeded: true, PromptUnits: 128000}
	state := After(RuntimeState{}, in, 8)
	got := Before(ModeShadow, state, in)
	if got.Action != Reuse || got.Applied || got.ResultRef != "c1" || got.ReplayUnitsSaved != 128000 || got.ReplaySquaredSaved != "16384000000" || got.PromptBucket != "gte128k" {
		t.Fatalf("verdict=%+v", got)
	}
}

func TestRuntimeEnforceReusesUntilMutation(t *testing.T) {
	read := RuntimeInput{Tool: "Read", Args: json.RawMessage(`{"file_path":"a"}`), ReadOnly: true, CallID: "c1", Succeeded: true}
	state := After(RuntimeState{}, read, 8)
	if got := Before(ModeEnforce, state, read); got.Action != Reuse || !got.Applied {
		t.Fatalf("repeat=%+v", got)
	}
	state = After(state, RuntimeInput{Tool: "Write", Args: json.RawMessage(`{"file_path":"a"}`)}, 8)
	if got := Before(ModeEnforce, state, read); got.Action != Allow || got.Reason != "novel_at_epoch" {
		t.Fatalf("after mutation=%+v", got)
	}
}

func TestRuntimeNeverSuppressesMutationOrMalformedInput(t *testing.T) {
	for _, in := range []RuntimeInput{
		{Tool: "Bash", Args: json.RawMessage(`{"command":"rm x"}`)},
		{Tool: "Read", ReadOnly: true},
	} {
		if got := Before(ModeEnforce, RuntimeState{}, in); got.Action != Allow || got.Applied {
			t.Fatalf("input=%+v verdict=%+v", in, got)
		}
	}
}

func TestRuntimeObservationBound(t *testing.T) {
	state := RuntimeState{}
	for _, path := range []string{"a", "b", "c"} {
		state = After(state, RuntimeInput{Tool: "Read", Args: json.RawMessage(`{"file_path":"` + path + `"}`), ReadOnly: true, Succeeded: true}, 2)
	}
	if len(state.Observations) != 2 {
		t.Fatalf("observations=%d", len(state.Observations))
	}
}

func TestParseModeFailsOff(t *testing.T) {
	if ParseMode("ENFORCE") != ModeEnforce || ParseMode("bogus") != ModeOff {
		t.Fatal("mode parser did not enforce closed vocabulary")
	}
}

func TestRuntimeOutcomeClassesPreserveFailureVisibility(t *testing.T) {
	exit1, exit124 := 1, 124
	tests := []struct {
		name       string
		input      RuntimeInput
		wantClass  OutcomeClass
		wantUnder  OutcomeClass
		projection OutcomeProjection
	}{
		{
			name:      "success",
			input:     RuntimeInput{Succeeded: true, Output: json.RawMessage(`{"stdout":"ok"}`)},
			wantClass: OutcomeSuccess, projection: ProjectionSuccess,
		},
		{
			name: "structural expected negative",
			input: RuntimeInput{ExitCode: &exit1, Output: json.RawMessage(`{"stderr":"no matches"}`),
				Declaration: OutcomeDeclaration{ExpectedNegative: true, ExpectedNegativeSet: true}},
			wantClass: OutcomeExpectedNegative, wantUnder: OutcomeUnexpectedCommandFailure, projection: ProjectionExpectedNegative,
		},
		{
			name:      "guard refusal",
			input:     RuntimeInput{Output: json.RawMessage(`{"error":true,"reason":"POLICY_BLOCK"}`)},
			wantClass: OutcomeGuardRefusal, projection: ProjectionUnexpectedFailure,
		},
		{
			name: "test failure",
			input: RuntimeInput{Args: json.RawMessage(`{"command":"go test ./internal/widget"}`), ExitCode: &exit1,
				Output: json.RawMessage(`{"stderr":"FAIL widget"}`)},
			wantClass: OutcomeTestFailure, projection: ProjectionUnexpectedFailure,
		},
		{
			name:      "timeout interruption",
			input:     RuntimeInput{ExitCode: &exit124, Output: json.RawMessage(`{"timed_out":true}`)},
			wantClass: OutcomeTimeoutInterruption, projection: ProjectionUnexpectedFailure,
		},
		{
			name:      "contract defect",
			input:     RuntimeInput{Output: json.RawMessage(`{"error":true,"message":"invalid tool arguments"}`)},
			wantClass: OutcomeContractDefect, projection: ProjectionUnexpectedFailure,
		},
		{
			name:      "unexpected command failure",
			input:     RuntimeInput{ExitCode: &exit1, Output: json.RawMessage(`{"stderr":"connection reset"}`)},
			wantClass: OutcomeUnexpectedCommandFailure, projection: ProjectionUnexpectedFailure,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyOutcome(tc.input)
			if got.Class != tc.wantClass || got.UnderlyingClass != tc.wantUnder || got.Projection != tc.projection {
				t.Fatalf("receipt=%+v, want class=%s underlying=%s projection=%s", got, tc.wantClass, tc.wantUnder, tc.projection)
			}
			if string(got.Output) != string(tc.input.Output) {
				t.Fatalf("output changed: got=%s want=%s", got.Output, tc.input.Output)
			}
			if tc.input.ExitCode != nil && (got.ExitCode == nil || *got.ExitCode != *tc.input.ExitCode) {
				t.Fatalf("exit code not retained: got=%v want=%d", got.ExitCode, *tc.input.ExitCode)
			}
		})
	}
}

func TestRuntimeOutcomeAmbiguityAndUnknownDeclarationsFailClosed(t *testing.T) {
	exit124 := 124
	for _, tc := range []struct {
		name  string
		input RuntimeInput
	}{
		{
			name: "ambiguous test timeout",
			input: RuntimeInput{Args: json.RawMessage(`{"command":"go test ./..."}`), ExitCode: &exit124,
				Output: json.RawMessage(`{"timed_out":true}`)},
		},
		{
			name:  "unknown declared class",
			input: RuntimeInput{Declaration: OutcomeDeclaration{Class: OutcomeClass("probably_ok")}},
		},
		{
			name: "expected failing witness unexpectedly succeeds",
			input: RuntimeInput{Succeeded: true,
				Declaration: OutcomeDeclaration{ExpectedNegative: true, ExpectedNegativeSet: true}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyOutcome(tc.input)
			if got.Class != OutcomeUnknown || got.Projection != ProjectionUnexpectedFailure {
				t.Fatalf("receipt=%+v, want fail-closed unknown unexpected failure", got)
			}
			if tc.input.Declaration.ExpectedNegative && !got.ExpectedNegative {
				t.Fatalf("receipt lost declared expectedness while failing closed: %+v", got)
			}
		})
	}
}

func TestRuntimeHeuristicsCannotCreateExpectedNegative(t *testing.T) {
	got := ClassifyOutcome(RuntimeInput{Output: json.RawMessage(`{"error":true,"reason":"expected grep miss"}`)})
	if got.Class == OutcomeExpectedNegative || got.Projection != ProjectionUnexpectedFailure || got.ExpectedNegative {
		t.Fatalf("text heuristics marked an outcome healthy: %+v", got)
	}
}
