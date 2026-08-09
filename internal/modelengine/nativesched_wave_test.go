package modelengine

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestWavePromotionConsumesDispatchHintAndCoPromotesReadySet(t *testing.T) {
	c := []PromotionCandidate{
		{Index: 0, Hint: dispatchtick.NewWaveHint("agent-a", "early", "wave-old", 4, 0)},
		{Index: 1, Hint: dispatchtick.NewWaveHint("agent-b", "left", "wave-ready", 1, 1)},
		{Index: 2, Hint: dispatchtick.NewWaveHint("agent-c", "right", "wave-ready", 1, 1)},
	}
	if got, want := WavePromotionPicker(c, 2, nil), []int{1, 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("promotion=%v want same dispatch-computed wave %v", got, want)
	}
}

func TestWavePromotionSessionAffinityAndWorkStealing(t *testing.T) {
	c := []PromotionCandidate{
		{Index: 0, Hint: dispatchtick.NewWaveHint("returning", "a", "cold", 1, 2)},
		{Index: 1, Hint: dispatchtick.NewWaveHint("returning", "b", "warm", 1, 7)},
		{Index: 2, Hint: dispatchtick.NewWaveHint("other", "c", "steal", 1, 3)},
	}
	got := WavePromotionPicker(c, 2, map[string]int{"returning": 7})
	if got[0] != 1 {
		t.Fatalf("first=%d want affinity worker lane 1", got[0])
	}
	if len(got) != 2 {
		t.Fatalf("promotion=%v want work stealing to fill spare slot", got)
	}
}

func TestWavePromotionFallbackIsFIFO(t *testing.T) {
	c := []PromotionCandidate{{Index: 0}, {Index: 1}, {Index: 2}}
	if got, want := WavePromotionPicker(c, 2, nil), []int{0, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("fallback=%v want FIFO %v", got, want)
	}
}
