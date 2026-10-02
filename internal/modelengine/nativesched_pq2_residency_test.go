package modelengine

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// A Bonsai-2 PQ2_0 checkpoint need not contain any Q4_K tensor. Scheduler
// construction must still select the resident session and Qwen prefill lane.
func TestNativeSchedulerPQ2OnlySelectsResidentSession(t *testing.T) {
	b := model.NewQuantBuilder(model.Config{
		ModelType:        "qwen35",
		HiddenSize:       128,
		IntermediateSize: 128,
		NumLayers:        1,
	}, false)
	if err := b.AddResidentQ2("model.layers.0.mlp.up_proj.weight", []int{128, 128}, make([]byte, 128*34)); err != nil {
		t.Fatal(err)
	}
	m, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	if m.Q2Count() != 1 || m.Q4KCount() != 0 {
		t.Fatalf("fixture: Q2=%d Q4K=%d, want 1/0", m.Q2Count(), m.Q4KCount())
	}
	s := NewNativeScheduler(m)
	if !s.residentQ4K || s.qwenPrefillCap == nil {
		t.Fatalf("PQ2-only scheduler: resident=%t Qwen prefill=%t, want both true", s.residentQ4K, s.qwenPrefillCap != nil)
	}
}
