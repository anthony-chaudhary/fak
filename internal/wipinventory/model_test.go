package wipinventory

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"
)

func TestWIPUnitIDIsVersionedOpaque(t *testing.T) {
	first, err := NewWIPUnitID()
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewWIPUnitID()
	if err != nil {
		t.Fatal(err)
	}
	if first == second {
		t.Fatalf("random IDs collided: %q", first)
	}
	if err := first.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, invalid := range []string{"10436", "main/path", "wip:v2:00000000000000000000000000000001", "wip:v1:10436"} {
		if _, err := ParseWIPUnitID(invalid); err == nil {
			t.Errorf("ParseWIPUnitID(%q) succeeded", invalid)
		}
	}
}

func TestLifecycleFixtureRoundTripsDeterministically(t *testing.T) {
	fixture, err := os.ReadFile("testdata/lifecycle.json")
	if err != nil {
		t.Fatal(err)
	}
	var history History
	if err := json.Unmarshal(fixture, &history); err != nil {
		t.Fatal(err)
	}
	first, err := MarshalDeterministic(history)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip History
	if err := json.Unmarshal(first, &roundTrip); err != nil {
		t.Fatal(err)
	}
	second, err := MarshalDeterministic(roundTrip)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("serialization changed:\n%s\n%s", first, second)
	}
}

func TestEverySurfaceReferenceKind(t *testing.T) {
	refs := []SurfaceReference{
		{Kind: SurfaceIssue, Issue: &IssueReference{Repository: "owner/repo", Number: 1}},
		{Kind: SurfaceDispatchSession, DispatchSession: &DispatchSessionReference{SessionID: "session"}},
		{Kind: SurfaceCheckpoint, Checkpoint: &CheckpointReference{CheckpointID: "checkpoint"}},
		{Kind: SurfaceLaneLease, LaneLease: &LaneLeaseReference{Lane: "lane", LeaseID: "lease"}},
		{Kind: SurfaceManagedWorktree, ManagedWorktree: &ManagedWorktreeReference{WorktreeID: "worktree"}},
		{Kind: SurfaceWitnessedRetirement, WitnessedRetirement: &WitnessedRetirementReference{RetirementID: "retirement", Witness: "commit"}},
	}
	for _, ref := range refs {
		if err := ref.Validate(); err != nil {
			t.Errorf("%s: %v", ref.Kind, err)
		}
	}
}
