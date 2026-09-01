package workerworktree

import (
	"reflect"
	"strings"
	"testing"
)

func TestAssessCapacityBelowAtAndAboveSetpoint(t *testing.T) {
	for _, test := range []struct {
		name       string
		count      int
		status     CapacityStatus
		above      bool
		wantPrompt bool
	}{
		{name: "below", count: 49, status: CapacityWithinSetpoint},
		{name: "at", count: 50, status: CapacityWithinSetpoint},
		{name: "above", count: 51, status: CapacityJustificationNeeded, above: true, wantPrompt: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			got := AssessCapacity(test.count, test.count, true, "", nil)
			if !got.Allowed || got.Status != test.status || got.AboveSetpoint != test.above {
				t.Fatalf("advisory = %+v", got)
			}
			if strings.Contains(got.Message, "--capacity-reason") != test.wantPrompt {
				t.Fatalf("message = %q", got.Message)
			}
			if got.ContractionRecommendations == nil {
				t.Fatal("empty recommendations must encode as []")
			}
		})
	}
}

func TestAssessCapacityRecordsJustifiedGrowth(t *testing.T) {
	got := AssessCapacity(50, 51, true, "  burst   for issue #10459  ", nil)
	if !got.Allowed || got.Status != CapacityGrowthJustified || got.Reason != "burst for issue #10459" {
		t.Fatalf("advisory = %+v", got)
	}
	if !strings.Contains(got.Message, got.Reason) {
		t.Fatalf("message does not carry deterministic evidence: %q", got.Message)
	}
}

func TestAssessCapacityRecommendsOnlySafeContractionAndPreservesDirty(t *testing.T) {
	got := AssessCapacity(54, 54, true, "", []RetainedTree{
		{Path: "/wt/z-dirty", ColdReapable: true, OwnerDead: true, Clean: false},
		{Path: "/wt/c-owner-dead", OwnerDead: true, Clean: true},
		{Path: "/wt/b-cold", ColdReapable: true, OwnerDead: true, Clean: true},
		{Path: "/wt/a-live-clean", Clean: true},
		{Path: "/wt/b-cold", ColdReapable: true, OwnerDead: true, Clean: true},
	})
	want := []ContractionRecommendation{
		{Path: "/wt/b-cold", Basis: "COLD_REAPABLE", Action: "fak worktree worker reap --all-cold"},
		{Path: "/wt/c-owner-dead", Basis: "OWNER_DEAD_CLEAN", Action: "fak worktree worker gc --dry-run"},
	}
	if !reflect.DeepEqual(got.ContractionRecommendations, want) {
		t.Fatalf("recommendations = %+v, want %+v", got.ContractionRecommendations, want)
	}
	for _, recommendation := range got.ContractionRecommendations {
		if strings.Contains(recommendation.Path, "dirty") || strings.Contains(recommendation.Action, "--apply") || strings.Contains(recommendation.Action, "--even-if-unlanded") {
			t.Fatalf("unsafe destructive recommendation: %+v", recommendation)
		}
	}
}

func TestAssessCapacityUnknownInventoryFailsOpenWithWarning(t *testing.T) {
	got := AssessCapacity(0, 1, false, "", []RetainedTree{{Path: "/wt/cold", ColdReapable: true, Clean: true}})
	if !got.Allowed || got.InventoryKnown || got.Status != CapacityInventoryUnknown || got.AboveSetpoint {
		t.Fatalf("advisory = %+v", got)
	}
	if !strings.Contains(got.Message, "failed open") || len(got.ContractionRecommendations) != 0 {
		t.Fatalf("unknown inventory warning/recommendations = %+v", got)
	}
}

func TestCapacityCensusForDistinguishesUnknownAndSortsSanctionedPaths(t *testing.T) {
	unknown := CapacityCensusFor("/repo", func(string, []string) (int, string) { return 1, "unavailable" })
	if unknown.Known || unknown.Paths == nil || len(unknown.Paths) != 0 {
		t.Fatalf("unknown census = %+v", unknown)
	}

	porcelain := "worktree /repo\nHEAD a\n\n" +
		"worktree /wt/fak-worker-wt-z-bbbbbbbbbbbb\nHEAD b\n\n" +
		"worktree /wt/fak-worker-wt-a-aaaaaaaaaaaa\nHEAD c\n\n"
	known := CapacityCensusFor("/repo", func(string, []string) (int, string) { return 0, porcelain })
	want := []string{"/wt/fak-worker-wt-a-aaaaaaaaaaaa", "/wt/fak-worker-wt-z-bbbbbbbbbbbb"}
	if !known.Known || !reflect.DeepEqual(known.Paths, want) {
		t.Fatalf("known census = %+v, want paths %v", known, want)
	}
}
