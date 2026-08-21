package orchestration

import (
	"reflect"
	"testing"
)

func TestObservationValidityAgainstChangeFeed(t *testing.T) {
	observation := ObservationSnapshot{
		ID:         "observer-result-1",
		StateEpoch: "git:abc123",
		ReadSet:    []string{`internal\orchestration\**`, "docs/agent-runtime.md"},
	}
	wantReadSet := []string{"docs/agent-runtime.md", "internal/orchestration"}

	unchanged := DecideObservationValidity(observation, nil)
	if !unchanged.Current || unchanged.Reason != "" {
		t.Fatalf("unchanged decision = %+v, want current", unchanged)
	}
	if unchanged.ObservationID != observation.ID || unchanged.StateEpoch != observation.StateEpoch || !reflect.DeepEqual(unchanged.ReadSet, wantReadSet) {
		t.Fatalf("unchanged evidence binding = %+v, want id/epoch/read set retained", unchanged)
	}

	unrelated := DecideObservationValidity(observation, []ObservationChange{{
		Sequence:     41,
		StateEpoch:   "git:def456",
		ChangedPaths: []string{"internal/model/sampler.go"},
	}})
	if !unrelated.Current || unrelated.Reason != "" || len(unrelated.InvalidatingChanges) != 0 {
		t.Fatalf("unrelated-change decision = %+v, want current", unrelated)
	}

	stale := DecideObservationValidity(observation, []ObservationChange{
		{Sequence: 43, StateEpoch: "git:fedcba", ChangedPaths: []string{"README.md"}},
		{Sequence: 42, StateEpoch: "git:def456", ChangedPaths: []string{"internal/orchestration/orchestration.go", "internal/model/sampler.go"}},
	})
	if stale.Current || stale.Reason != ObservationStale {
		t.Fatalf("relevant-change decision = %+v, want %s", stale, ObservationStale)
	}
	if stale.Guidance != ObservationStaleGuidance {
		t.Fatalf("stale guidance = %q, want %q", stale.Guidance, ObservationStaleGuidance)
	}
	wantChanges := []ObservationChange{{
		Sequence:     42,
		StateEpoch:   "git:def456",
		ChangedPaths: []string{"internal/orchestration/orchestration.go"},
	}}
	if !reflect.DeepEqual(stale.InvalidatingChanges, wantChanges) {
		t.Fatalf("invalidating changes = %+v, want %+v", stale.InvalidatingChanges, wantChanges)
	}
	if stale.ObservationID != observation.ID || stale.StateEpoch != observation.StateEpoch || !reflect.DeepEqual(stale.ReadSet, wantReadSet) {
		t.Fatalf("stale evidence binding = %+v, want original observation receipt", stale)
	}
}

func TestObservationValidityFailsClosedWithoutSnapshotBinding(t *testing.T) {
	decision := DecideObservationValidity(ObservationSnapshot{ID: "result"}, nil)
	if decision.Current || decision.Reason != ObservationInvalid || decision.Guidance == "" {
		t.Fatalf("invalid observation decision = %+v, want typed fail-closed refusal", decision)
	}
}
