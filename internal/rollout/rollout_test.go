package rollout

import (
	"fmt"
	"testing"
)

var (
	generationA = Generation{ID: "gen-a", Digest: "sha256:aaa"}
	generationB = Generation{ID: "gen-b", Digest: "sha256:bbb"}
	generationC = Generation{ID: "gen-c", Digest: "sha256:ccc"}
)

func baseState() State { return State{Stable: generationA, LastKnownGood: generationA} }

func stagedState(t *testing.T, basisPoints, cap int) State {
	t.Helper()
	state, err := baseState().Stage(generationB, basisPoints, cap, "rollout-test")
	if err != nil {
		t.Fatalf("Stage() error = %v", err)
	}
	return state
}

func TestOrdinarySessionsRemainStable(t *testing.T) {
	selection, err := baseState().Select("session-ordinary", 0)
	if err != nil {
		t.Fatalf("Select() error = %v", err)
	}
	if selection.Generation != generationA || selection.Canary || selection.Reason != ReasonStableNoCanary {
		t.Fatalf("Select() = %+v, want stable no-canary selection", selection)
	}
}

func TestSelectUsesDeterministicCohort(t *testing.T) {
	state := stagedState(t, 5_000, 10)
	var canarySession, stableSession string
	for i := 0; i < 100 && (canarySession == "" || stableSession == ""); i++ {
		sessionID := fmt.Sprintf("session-%d", i)
		selection, err := state.Select(sessionID, 0)
		if err != nil {
			t.Fatalf("Select(%q) error = %v", sessionID, err)
		}
		if selection.Canary {
			canarySession = sessionID
		} else if selection.Reason == ReasonStableOutsideCohort {
			stableSession = sessionID
		}
	}
	if canarySession == "" || stableSession == "" {
		t.Fatalf("did not find both cohort outcomes: canary=%q stable=%q", canarySession, stableSession)
	}

	first, err := state.Select(canarySession, 0)
	if err != nil {
		t.Fatal(err)
	}
	second, err := state.Select(canarySession, 0)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || !first.Canary || first.Generation != generationB || first.Reason != ReasonCanarySelected {
		t.Fatalf("deterministic canary selection mismatch: first=%+v second=%+v", first, second)
	}
}

func TestSelectEnforcesAbsoluteExposureCap(t *testing.T) {
	state := stagedState(t, 10_000, 2)
	selection, err := state.Select("always-canary", 2)
	if err != nil {
		t.Fatalf("Select() error = %v", err)
	}
	if selection.Generation != generationA || selection.Canary || selection.Reason != ReasonStableExposureCap {
		t.Fatalf("Select() = %+v, want stable cap-reached selection", selection)
	}
}

func TestAbortRestoresLastKnownGoodAndRejectsCanary(t *testing.T) {
	state := stagedState(t, 10_000, 2)
	state.LastKnownGood = generationC

	aborted, err := state.Abort()
	if err != nil {
		t.Fatalf("Abort() error = %v", err)
	}
	if aborted.Stable != generationC || !aborted.Canary.isZero() {
		t.Fatalf("Abort() stable/canary = %+v/%+v, want LKG/empty", aborted.Stable, aborted.Canary)
	}
	if aborted.ExposureBasisPoints != 0 || aborted.ExposureConcurrentCap != 0 || !aborted.ExposureKilled {
		t.Fatalf("Abort() did not stop canary admission: %+v", aborted)
	}
	if !aborted.RejectedGenerationIDs[generationB.ID] || !aborted.RejectedDigests[generationB.Digest] {
		t.Fatalf("Abort() did not record canary rejection: %+v", aborted)
	}

	selection, err := aborted.Select("post-abort", 0)
	if err != nil {
		t.Fatalf("post-abort Select() error = %v", err)
	}
	if selection.Generation != generationC || selection.Reason != ReasonStableCanaryKilled {
		t.Fatalf("post-abort Select() = %+v, want killed canary fallback to LKG", selection)
	}
}

func TestRejectedGenerationOrDigestCannotBeRestaged(t *testing.T) {
	aborted, err := stagedState(t, 10_000, 1).Abort()
	if err != nil {
		t.Fatalf("Abort() error = %v", err)
	}
	if _, err := aborted.Stage(generationB, 100, 1, "retry"); err == nil {
		t.Fatal("Stage() accepted rejected generation and digest")
	}
	alias := Generation{ID: "gen-b-alias", Digest: generationB.Digest}
	if _, err := aborted.Stage(alias, 100, 1, "retry-alias"); err == nil {
		t.Fatal("Stage() accepted rejected digest under a new generation ID")
	}
}

func TestPromoteMakesCanaryStableAndPriorStableLastKnownGood(t *testing.T) {
	promoted, err := stagedState(t, 1_000, 3).Promote()
	if err != nil {
		t.Fatalf("Promote() error = %v", err)
	}
	if promoted.Stable != generationB || promoted.LastKnownGood != generationA {
		t.Fatalf("Promote() stable/LKG = %+v/%+v, want canary/prior stable", promoted.Stable, promoted.LastKnownGood)
	}
	if !promoted.Canary.isZero() || promoted.ExposureBasisPoints != 0 || promoted.ExposureConcurrentCap != 0 {
		t.Fatalf("Promote() retained canary admission state: %+v", promoted)
	}
}

func TestValidateRejectsInvalidStates(t *testing.T) {
	tests := map[string]State{
		"missing stable": {LastKnownGood: generationA},
		"identical canary": {
			Stable: generationA, LastKnownGood: generationC, Canary: generationA,
		},
		"basis points too high": {
			Stable: generationA, LastKnownGood: generationA, ExposureBasisPoints: 10_001,
		},
		"negative cap": {
			Stable: generationA, LastKnownGood: generationA, ExposureConcurrentCap: -1,
		},
		"positive percentage without cap": {
			Stable: generationA, LastKnownGood: generationA, Canary: generationB,
			ExposureBasisPoints: 1, CohortSalt: "salt",
		},
	}
	for name, state := range tests {
		t.Run(name, func(t *testing.T) {
			if err := state.Validate(); err == nil {
				t.Fatal("Validate() accepted invalid state")
			}
		})
	}
}
