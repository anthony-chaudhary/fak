package model

import (
	"errors"
	"testing"
)

// The GGUF ships the Engram mixing norms as [hc][hidden]; admission accepts that
// rank-2 declaration by cell count, as the consumer reads it, and still refuses a
// wrong cell count at the Engram stage.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime fast est=100ms lane=default
func TestV41EngramNormsAdmittedByCellCount(t *testing.T) {
	H := v41FullGeometryConfig(t).HiddenSize
	for _, c := range []struct {
		shape []int
		ok    bool
	}{
		{[]int{4 * H}, true},
		{[]int{4, H}, true},
		{[]int{3, H}, false},
		{[]int{4 * H, 1, 1}, false},
	} {
		for _, leaf := range []string{"engram_q_norm.weight", "engram_k_norm.weight"} {
			m, _ := v41FullEngramModel(t)
			name := layerName(0, leaf)
			meta := m.manifest[name]
			meta.Shape = c.shape
			m.manifest[name] = meta
			err := m.v41EngramForwardAdmitted()
			if c.ok {
				if err != nil {
					t.Errorf("%s shape %v refused: %v", leaf, c.shape, err)
				}
				continue
			}
			var stage *V41ForwardError
			if !errors.As(err, &stage) || stage.Stage != v41StageEngram || stage.Layer != 0 || !errors.Is(err, ErrV41ForwardStage) {
				t.Errorf("%s shape %v admitted or refused off-stage: %v", leaf, c.shape, err)
			}
		}
	}
}
