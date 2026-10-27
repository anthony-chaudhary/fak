package model

import (
	"errors"
	"slices"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmEmbeddingForward(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	s, control := m.NewSession(), m.NewSession()
	t.Cleanup(s.Close)
	t.Cleanup(control.Close)
	var stages []string
	s.hostStageObserver = func(stage string) {
		if stage == "embedding-panel" {
			stages = append(stages, stage)
		}
	}
	v41ScoreTestBits(t, s.Prefill([]int{1, 2}), control.Prefill([]int{1, 2}))
	v41ScoreTestBits(t, s.Step(3), control.Step(3))
	if !slices.Equal(stages, []string{"embedding-panel", "embedding-panel"}) {
		t.Fatalf("executed embedding stages = %v", stages)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestV41HostArmEmbeddingBoundaries(t *testing.T) {
	m := v41IncrementalPlainModel(t, 1)
	ids := []int{2, 1, 2}
	want, err := m.v41EmbeddingPanel(ids)
	if err != nil {
		t.Fatal(err)
	}
	for _, selected := range []bool{false, true} {
		var stages []string
		s := &Session{M: m, hostStageObserver: func(stage string) { stages = append(stages, stage) }}
		ops := &v41DeviceGatherOps{owner: s}
		if selected {
			ops.embedGather = func([]int) ([]float32, error) { return slices.Clone(want), nil }
		}
		got, err := m.v41EmbeddingPanelOn(ids, ops)
		if err != nil {
			t.Fatal(err)
		}
		v41ScoreTestBits(t, got, want)
		if selected && len(stages) != 0 || !selected && !slices.Equal(stages, []string{"embedding-panel"}) {
			t.Fatalf("selected=%t stages=%v", selected, stages)
		}
	}
	for _, fault := range []string{"selected-error", "selected-width", "empty", "missing-table"} {
		t.Run(fault, func(t *testing.T) {
			var stages []string
			s := &Session{hostStageObserver: func(stage string) { stages = append(stages, stage) }}
			ops := &v41DeviceGatherOps{owner: s}
			model, input := m, ids
			cause := errors.New("selected embedding failure")
			switch fault {
			case "selected-error":
				ops.embedGather = func([]int) ([]float32, error) { return nil, cause }
			case "selected-width":
				ops.embedGather = func([]int) ([]float32, error) { return want[:1], nil }
				cause = errV41EmbeddingGatherResult
			case "empty":
				input = nil
			case "missing-table":
				model = &Model{Cfg: m.Cfg}
				cause = ErrV41ForwardStage
			}
			got, err := model.v41EmbeddingPanelOn(input, ops)
			if fault == "empty" {
				if err != nil || len(got) != 0 {
					t.Fatalf("empty embedding: values=%d err=%v", len(got), err)
				}
			} else if got != nil || !errors.Is(err, cause) {
				t.Fatalf("fault=%s values=%d err=%v", fault, len(got), err)
			}
			if len(stages) != 0 {
				t.Fatalf("unexecuted scalar gather observed: %v", stages)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41HostArmEmbeddingPacked(t *testing.T) {
	const vocab, hidden = 5, 256
	packed, err := NewQ2KEmbedding(v41Q2KEmbeddingTestRaw(vocab, hidden, 14006), vocab, hidden)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Cfg: Config{HiddenSize: hidden, VocabSize: vocab, EmbedScale: 2}, Q2KEmbedding: packed}
	ids := []int{3, 0, 3}
	want, err := m.v41EmbeddingPanel(ids)
	if err != nil {
		t.Fatal(err)
	}
	var stages []string
	s := &Session{hostStageObserver: func(stage string) { stages = append(stages, stage) }}
	got, err := m.v41EmbeddingPanelOn(ids, &v41DeviceGatherOps{owner: s})
	if err != nil {
		t.Fatal(err)
	}
	v41ScoreTestBits(t, got, want)
	if !slices.Equal(stages, []string{"embedding-panel"}) {
		t.Fatalf("packed scalar stages = %v", stages)
	}
}
