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
