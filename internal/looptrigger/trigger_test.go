package looptrigger

import (
	"encoding/json"
	"testing"
	"time"
)

func base() Input {
	return Input{Loop: "super-loop", ObservedAt: time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC), EligibleUnits: 8, OldestAge: 2 * time.Hour, SourceAge: 20 * time.Second, MaxSourceAge: 5 * time.Minute, OfferedCapacity: 4, RequiredCapacity: 1, SinceLastRun: 2 * time.Hour, Cooldown: time.Hour, ServiceWindow: 4 * time.Hour, ExpectedValue: 10, ValueFloor: 5, EstimatedWall: 15 * time.Minute, EstimatedAttention: 2 * time.Minute, EvidenceRefs: []string{"issue:10352", "lease:loops"}}
}

func TestEvaluateClassifications(t *testing.T) {
	tests := []struct {
		name     string
		mutate   func(*Input)
		decision Decision
		reason   Reason
		timing   string
	}{
		{"no-demand", func(i *Input) { i.EligibleUnits = 0 }, Skip, NoDemand, "TIMELY"},
		{"stale", func(i *Input) { i.SourceAge = 6 * time.Minute }, Defer, InputStale, "TIMELY"},
		{"overlap", func(i *Input) { i.OverlapCount = 1 }, Merge, AlreadyOwned, "TIMELY"},
		{"no-capacity", func(i *Input) { i.OfferedCapacity = 0 }, Defer, NoCapacity, "TIMELY"},
		{"too-early", func(i *Input) { i.SinceLastRun = 30 * time.Minute }, Defer, Cooldown, "EARLY"},
		{"below-value", func(i *Input) { i.ExpectedValue = 4 }, Skip, BelowValueFloor, "TIMELY"},
		{"timely", func(i *Input) {}, Run, DemandReady, "TIMELY"},
		{"overdue", func(i *Input) { i.OldestAge = 5 * time.Hour }, Run, Deadline, "OVERDUE"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := base()
			tt.mutate(&in)
			got := Evaluate(in)
			if got.Decision != tt.decision || got.Reason != tt.reason || got.Timing.State != tt.timing {
				t.Fatalf("got %s/%s/%s", got.Decision, got.Reason, got.Timing.State)
			}
		})
	}
}

func TestEvaluatePrecedenceAndStableJSON(t *testing.T) {
	in := base()
	in.EligibleUnits = 0
	in.SourceAge = time.Hour
	in.OverlapCount = 1
	got := Evaluate(in)
	if got.Reason != NoDemand {
		t.Fatalf("precedence=%s", got.Reason)
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Receipt
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Schema != Schema || decoded.ObservedAt != "2026-08-31T12:00:00Z" || len(decoded.EvidenceRefs) != 2 {
		t.Fatalf("unstable receipt: %s", b)
	}
}
