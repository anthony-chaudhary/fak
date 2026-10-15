package model

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestV41RoutedSumBinding(t *testing.T) {
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	b := &v41RoutedOutputOwnerWitness{v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend()}}
	m := &Model{Cfg: Config{HiddenSize: 4, MoEIntermediateSize: 3, HeadDim: v41KVLoraRank, NumLayers: 1}}
	s := &Session{M: m, Backend: b}
	scope := s.beginBackendExecution()
	defer scope.finish()
	defer s.closeBackendOwned()
	bound := s.v41CanonicalRoutedSumBinding()
	if bound == nil || bound.native == nil || bound.sumFunc() == nil {
		t.Fatal("factory did not capture canonical backend provenance")
	}
	st := &v41ForwardState{routedSumBinding: bound}
	scratch := &v41ProjScratch{}
	scratch.copyRoutedSumFrom(st)
	picks := [][]routePick{{{expert: 0, weight: 1}}}
	if owner := scratch.routedOutputOwner(s, 0, 4, picks); owner == nil {
		t.Fatal("trusted copy lost canonical authority")
	} else {
		owner.release(0, true)
	}
	// Even the same extracted ordinary callback loses native authority.
	st.setRoutedSum(st.routedSumFunc())
	scratch.copyRoutedSumFrom(st)
	if scratch.routedSumFunc() == nil || scratch.routedOutputOwner(s, 0, 4, picks) != nil {
		t.Fatal("ordinary replacement inherited native authority")
	}
	scratch.routedSumBinding = bound
	oldBackend := s.Backend
	s.Backend = newV41SumTestBackend()
	if scratch.routedOutputOwner(s, 0, 4, picks) != nil {
		t.Fatal("old factory provenance followed a replacement backend")
	}
	s.Backend = oldBackend
	m.Cfg.MoEIntermediateSize++
	if scratch.routedOutputOwner(s, 0, 4, picks) != nil {
		t.Fatal("old factory provenance followed changed projection geometry")
	}
	m.Cfg.MoEIntermediateSize--
	// The final capability probe may execute user backend code. Recheck
	// authority after it, before any publication can be retained.
	b.ownerQuery = func() { s.Backend = newV41SumTestBackend() }
	if scratch.routedOutputOwner(s, 0, 4, picks) != nil || s.v41RoutedOutputs != nil {
		t.Fatal("capability callback transferred stale factory authority")
	}
	b.ownerQuery = nil
	s.Backend = oldBackend
	scratch.setRoutedSum(bound.sumFunc())
	if scratch.routedOutputOwner(s, 0, 4, picks) != nil {
		t.Fatal("scratch replacement retained authority")
	}
	scratch.copyRoutedSumFrom(nil)
	if scratch.routedSumFunc() != nil {
		t.Fatal("nil trusted copy kept callback")
	}
}
