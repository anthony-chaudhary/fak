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
	got := NewLaneScorerRegistry(priority, affinity).Order([]LaneCandidate{{1, 400}, {2, 60}}, false)
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
	got := NewLaneScorerRegistry(broken, bounded).Order([]LaneCandidate{{1, 0}, {2, 0}}, false)
	if want := []int{2, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("order=%v want %v", got, want)
	}
}

func TestDefaultLaneScorersPreservePriorityOrdering(t *testing.T) {
	got := OrderLaneCandidates([]LaneCandidate{{4, 60}, {3, 150}, {2, 400}, {1, 1000}}, false)
	if want := []int{1, 2, 3, 4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("order=%v want %v", got, want)
	}
}
