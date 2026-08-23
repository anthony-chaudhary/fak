package polymodel

import (
	"reflect"
	"testing"
)

func TestAcceptanceProfileFixedFixture(t *testing.T) {
	var profile acceptanceProfile
	for i := range []int{3, 1, 0} {
		profile.record([]int{4, 2, 1}[i], []int{3, 1, 0}[i])
	}
	got := profile.snapshot()
	wantCounts := [][2]int{{3, 2}, {2, 1}, {1, 1}, {1, 0}}
	wantRates := []float64{2.0 / 3.0, 0.5, 1, 0}
	for i := range wantCounts {
		if got[i].Proposed != wantCounts[i][0] || got[i].Accepted != wantCounts[i][1] {
			t.Fatalf("position %d counts = %d/%d, want %d/%d", i, got[i].Accepted, got[i].Proposed, wantCounts[i][1], wantCounts[i][0])
		}
		if got[i].Rate == nil || *got[i].Rate != wantRates[i] {
			t.Fatalf("position %d rate = %v, want %v", i, got[i].Rate, wantRates[i])
		}
	}
}

func TestAcceptanceProfileUnavailableWithoutProposal(t *testing.T) {
	var profile acceptanceProfile
	profile.record(2, 1)
	profile.proposed = append(profile.proposed, 0)
	profile.accepted = append(profile.accepted, 0)
	got := profile.snapshot()
	if got[2].Rate != nil {
		t.Fatalf("zero-proposal rate = %v, want unavailable", *got[2].Rate)
	}
}

func TestAcceptanceProfileObservationDoesNotChangeCounts(t *testing.T) {
	var profile acceptanceProfile
	beforeProposed := []int{4, 2, 1}
	beforeAccepted := []int{3, 1, 0}
	for i := range beforeProposed {
		profile.record(beforeProposed[i], beforeAccepted[i])
	}
	if !reflect.DeepEqual(beforeProposed, []int{4, 2, 1}) || !reflect.DeepEqual(beforeAccepted, []int{3, 1, 0}) {
		t.Fatal("recording mutated decode outcomes")
	}
}

func TestSpecDecodeAndTreeExposeSamePositionSemantics(t *testing.T) {
	chain, err := SpecDecode(nil,
		func([]int) []int { return []int{10, 11, 12} },
		func([]int, []int) []int { return []int{10, 11, 99, 100} },
		SpecDecodeConfig{MaxNewTokens: 3, MaxDraft: 3},
	)
	if err != nil {
		t.Fatal(err)
	}
	tree, err := SpecDecodeTree(nil,
		func([]int) SpecTree {
			return SpecTree{Nodes: []TreeNode{
				{Children: []int{1}},
				{Token: 10, Children: []int{2}},
				{Token: 11, Children: []int{3}},
				{Token: 12},
			}}
		},
		func([]int, SpecTree) []int { return []int{10, 11, 99, 100} },
		SpecDecodeTreeConfig{MaxNewTokens: 3, MaxNodes: 3},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(chain.AcceptanceProfile, tree.AcceptanceProfile) {
		t.Fatalf("chain profile %#v != tree profile %#v", chain.AcceptanceProfile, tree.AcceptanceProfile)
	}
	want := []AcceptancePosition{
		{Position: 0, Proposed: 1, Accepted: 1, Rate: ratePointer(1)},
		{Position: 1, Proposed: 1, Accepted: 1, Rate: ratePointer(1)},
		{Position: 2, Proposed: 1, Accepted: 0, Rate: ratePointer(0)},
	}
	if !reflect.DeepEqual(chain.AcceptanceProfile, want) {
		t.Fatalf("profile = %#v, want %#v", chain.AcceptanceProfile, want)
	}
	if chain.AcceptedDrafts != 2 || tree.AcceptedNodes != 2 || chain.MeanAcceptanceLength != 3 || tree.MeanAcceptanceLength != 3 {
		t.Fatalf("scalar accounting changed: chain=%+v tree=%+v", chain, tree)
	}
}

func ratePointer(value float64) *float64 { return &value }
