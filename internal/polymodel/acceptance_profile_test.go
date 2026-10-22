package polymodel

import (
	"math"
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
	profile.reached = append(profile.reached, 0)
	got := profile.snapshot()
	if got[2].Rate != nil || got[2].ConditionalRate != nil {
		t.Fatalf("zero-proposal rates = %v/%v, want unavailable", got[2].Rate, got[2].ConditionalRate)
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
		{Position: 0, Proposed: 1, Accepted: 1, Rate: ratePointer(1), Reached: 1, ConditionalRate: ratePointer(1)},
		{Position: 1, Proposed: 1, Accepted: 1, Rate: ratePointer(1), Reached: 1, ConditionalRate: ratePointer(1)},
		{Position: 2, Proposed: 1, Accepted: 0, Rate: ratePointer(0), Reached: 1, ConditionalRate: ratePointer(0)},
	}
	if !reflect.DeepEqual(chain.AcceptanceProfile, want) {
		t.Fatalf("profile = %#v, want %#v", chain.AcceptanceProfile, want)
	}
	if chain.AcceptedDrafts != 2 || tree.AcceptedNodes != 2 || chain.MeanAcceptanceLength != 3 || tree.MeanAcceptanceLength != 3 {
		t.Fatalf("scalar accounting changed: chain=%+v tree=%+v", chain, tree)
	}
}

func ratePointer(value float64) *float64 { return &value }

func geometricYield(rates []float64) float64 {
	yield, prefix := 1.0, 1.0
	for _, rate := range rates {
		prefix *= rate
		yield += prefix
	}
	return yield
}

func conditionalRates(t *testing.T, positions []AcceptancePosition) []float64 {
	t.Helper()
	rates := make([]float64, len(positions))
	for i, row := range positions {
		if row.ConditionalRate == nil {
			t.Fatalf("position %d conditional rate unavailable", i)
		}
		rates[i] = *row.ConditionalRate
	}
	return rates
}

func assertPositionCounts(t *testing.T, got []AcceptancePosition, proposed, reached, accepted []int) {
	t.Helper()
	if len(got) != len(proposed) {
		t.Fatalf("positions = %d, want %d", len(got), len(proposed))
	}
	for i, row := range got {
		if row.Position != i || row.Proposed != proposed[i] || row.Reached != reached[i] || row.Accepted != accepted[i] {
			t.Fatalf("position %d = %+v, want proposed/reached/accepted %d/%d/%d", i, row, proposed[i], reached[i], accepted[i])
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAcceptanceProfileCensoredConditionalRate(t *testing.T) {
	var profile acceptanceProfile
	for accepted, rounds := range []int{125, 100, 80, 64, 256} {
		for range rounds {
			profile.record(4, accepted)
		}
	}
	got := profile.snapshot()
	assertPositionCounts(t, got, []int{625, 625, 625, 625}, []int{625, 500, 400, 320}, []int{500, 400, 320, 256})
	for i, wantRate := range []float64{0.8, 0.64, 0.512, 0.4096} {
		if got[i].Rate == nil || *got[i].Rate != wantRate {
			t.Fatalf("position %d marginal rate = %v, want %v", i, got[i].Rate, wantRate)
		}
		if got[i].ConditionalRate == nil || *got[i].ConditionalRate != 0.8 {
			t.Fatalf("position %d conditional rate = %v, want 0.8", i, got[i].ConditionalRate)
		}
	}

	const observedYield = 2101.0 / 625.0
	conditionalYield := geometricYield(conditionalRates(t, got))
	if math.Abs(conditionalYield-observedYield) > 1e-12 {
		t.Fatalf("conditional-fed yield = %v, want observed %v", conditionalYield, observedYield)
	}
	marginal := 1476.0 / 2500.0
	marginalYield := geometricYield([]float64{marginal, marginal, marginal, marginal})
	if math.Abs(marginalYield-2.2662717139910656) > 1e-12 {
		t.Fatalf("marginal-fed yield = %v, want 2.2662717139910656", marginalYield)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAcceptanceProfileConditionalRateVariableDepth(t *testing.T) {
	var profile acceptanceProfile
	profile.record(4, 4)
	profile.record(2, 2)
	profile.record(3, 1)
	got := profile.snapshot()
	assertPositionCounts(t, got, []int{3, 3, 2, 1}, []int{3, 3, 1, 1}, []int{3, 2, 1, 1})
	wantConditional := []float64{1, 2.0 / 3.0, 1, 1}
	wantMarginal := []float64{1, 2.0 / 3.0, 0.5, 1}
	for i := range got {
		if *got[i].ConditionalRate != wantConditional[i] || *got[i].Rate != wantMarginal[i] {
			t.Fatalf("position %d rates = %v/%v, want conditional %v marginal %v", i, *got[i].ConditionalRate, *got[i].Rate, wantConditional[i], wantMarginal[i])
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAcceptanceProfileConditionalRateNilWhenUnreached(t *testing.T) {
	var profile acceptanceProfile
	profile.record(3, 0)
	got := profile.snapshot()
	assertPositionCounts(t, got, []int{1, 1, 1}, []int{1, 0, 0}, []int{0, 0, 0})
	if got[0].ConditionalRate == nil || *got[0].ConditionalRate != 0 {
		t.Fatalf("reached position conditional rate = %v, want 0", got[0].ConditionalRate)
	}
	for _, row := range got[1:] {
		if row.ConditionalRate != nil || row.Rate == nil || *row.Rate != 0 {
			t.Fatalf("unreached position %d = %+v, want nil conditional and zero marginal", row.Position, row)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAcceptanceProfileTreeReachedCountsRounds(t *testing.T) {
	var profile acceptanceProfile
	profile.recordCounts([]int{2, 1}, 1)
	got := profile.snapshot()
	assertPositionCounts(t, got, []int{2, 1}, []int{1, 1}, []int{1, 0})
	if *got[0].ConditionalRate != 1 || *got[1].ConditionalRate != 0 || *got[0].Rate != 0.5 || *got[1].Rate != 0 {
		t.Fatalf("tree rates = %+v", got)
	}
}
