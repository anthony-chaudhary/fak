package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41RoutedOutputExhaustedCompletion(t *testing.T) {
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	b := &v41RoutedOutputOwnerWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true}, freed: make(map[compute.Buffer]bool)}
	s := &Session{M: &Model{Cfg: Config{HiddenSize: 4, MoEIntermediateSize: 3, HeadDim: v41KVLoraRank, NumLayers: 1}}, Backend: b}
	scope := s.beginBackendExecution()
	defer scope.finish()
	defer s.closeBackendOwned()
	picks := [][]routePick{{{expert: 0, weight: 1}, {expert: 1, weight: 1}}}
	o := newV41RoutedOutputOwner(s, 0, 4, picks)
	if o == nil {
		t.Fatal("owner declined")
	}
	defer o.release(0, true)
	h := o.handoff([]v41ExpertRow{{0, 0}, {0, 1}})
	if h.canCompleteUnretained(2) || h.completeUnretained(2) == nil {
		t.Fatal("unfilled owner bypass")
	}
	rows := []compute.Tensor{compute.NewF32(b, []int{4}, []float32{1, 2, 3, 4}), compute.NewF32(b, []int{4}, []float32{5, 6, 7, 8})}
	if n, e := h.adopt(rows); e != nil || n != 2 {
		t.Fatal(n, e)
	}
	if !h.canCompleteUnretained(2) || h.completeUnretained(2) != nil {
		t.Fatal("exhausted owner refused")
	}
	if n, e := h.adopt(make([]compute.Tensor, 2)); e != nil || n != 0 {
		t.Fatal("existing zero-transfer equivalence", n, e)
	}
	if len(o.rows) != 2 || len(o.buffers) != 2 || len(o.index) != 2 || b.freeRows != 0 {
		t.Fatal("completion mutated retained ownership")
	}
	if h.canCompleteUnretained(1) || h.completeUnretained(1) == nil {
		t.Fatal("wrong logical count")
	}
	o.capRows = 3
	if h.canCompleteUnretained(2) {
		t.Fatal("crossing cohort")
	}
	o.capRows = 2
	o.closed = true
	if h.canCompleteUnretained(2) || h.completeUnretained(2) == nil {
		t.Fatal("stale owner")
	}
	o.closed = false
	scope.live = false
	if h.canCompleteUnretained(2) {
		t.Fatal("stale scope")
	}
	scope.live = true
	s.M.Cfg.HiddenSize = 5
	if h.canCompleteUnretained(2) {
		t.Fatal("changed geometry")
	}
	s.M.Cfg.HiddenSize = 4
}
