package dispatchtick

import (
	"reflect"
	"testing"
)

func TestLaneScorerRegistryCombinesIndependentSignals(t *testing.T) {
	priority := LaneScorerFunc{"priority", 2, func(c LaneCandidate) float64 { return float64(c.Weight) / 1000 }}
	affinity := LaneScorerFunc{"affinity", 1, func(c LaneCandidate) float64 {
		if c.Number == 2 {
			return 1
		}
		return 0
	}}
	got := NewLaneScorerRegistry(priority, affinity).Order([]LaneCandidate{{Number: 1, Weight: 400}, {Number: 2, Weight: 60}}, false)
	if want := []int{2, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("order=%v want %v", got, want)
	}
}

func TestLaneScorerRegistryClampsPluginScores(t *testing.T) {
	broken := LaneScorerFunc{"broken", 1, func(c LaneCandidate) float64 {
		if c.Number == 1 {
			return 99
		}
		return -99
	}}
	bounded := LaneScorerFunc{"bounded", 2, func(c LaneCandidate) float64 {
		if c.Number == 2 {
			return 1
		}
		return 0
	}}
	got := NewLaneScorerRegistry(broken, bounded).Order([]LaneCandidate{{Number: 1, Weight: 0}, {Number: 2, Weight: 0}}, false)
	if want := []int{2, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("order=%v want %v", got, want)
	}
}

func TestDefaultLaneScorersPreservePriorityOrdering(t *testing.T) {
	got := OrderLaneCandidates([]LaneCandidate{{Number: 4, Weight: 60}, {Number: 3, Weight: 150}, {Number: 2, Weight: 400}, {Number: 1, Weight: 1000}}, false)
	if want := []int{1, 2, 3, 4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("order=%v want %v", got, want)
	}
}
