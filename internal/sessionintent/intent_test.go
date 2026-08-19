package sessionintent

import (
	"strings"
	"testing"
	"time"
)

func baseIntent() Intent {
	return Intent{Version: "fak.session-intent/v1alpha1", Objective: "inventory session needs", Trigger: Trigger{Kind: TriggerImmediate}}
}

func TestValidateAcceptsIndependentActiveAndElapsedBounds(t *testing.T) {
	i := baseIntent()
	i.Effort = []EffortBound{{Kind: BoundMinimum, Clock: ClockActive, Duration: 2 * time.Hour}, {Kind: BoundTarget, Clock: ClockActive, Duration: 3 * time.Hour}, {Kind: BoundMaximum, Clock: ClockElapsed, Duration: 10 * time.Hour}}
	if err := i.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRejectsContradictoryBounds(t *testing.T) {
	i := baseIntent()
	i.Effort = []EffortBound{{Kind: BoundMinimum, Clock: ClockActive, Duration: 2 * time.Hour}, {Kind: BoundMaximum, Clock: ClockActive, Duration: time.Hour}}
	if err := i.Validate(); err == nil || !strings.Contains(err.Error(), "minimum exceeds maximum") {
		t.Fatalf("got %v", err)
	}
}

func TestValidateRequiresTypedSchedulePolicies(t *testing.T) {
	i := baseIntent()
	i.Recurrence = &Recurrence{Every: time.Hour}
	if err := i.Validate(); err == nil || !strings.Contains(err.Error(), "overlap policy") {
		t.Fatalf("got %v", err)
	}
	i.Recurrence.OverlapPolicy, i.Recurrence.MisfirePolicy = "forbid", "skip"
	if err := i.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateHookTimeoutAndFailurePolicy(t *testing.T) {
	i := baseIntent()
	i.Hooks = []Hook{{Event: "before_stop", Action: "witness", Timeout: 30 * time.Second, FailurePolicy: "block"}}
	if err := i.Validate(); err != nil {
		t.Fatal(err)
	}
	i.Hooks[0].Timeout = 0
	if err := i.Validate(); err == nil {
		t.Fatal("expected zero timeout rejection")
	}
}
