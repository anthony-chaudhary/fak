package model

import (
	"errors"
	"reflect"
	"testing"
)

// Real compressed source/reader execution witnesses capture after a cache restore.
// The estimate is unmeasured; this establishes ownership, not physical speed.
// fak-test:runtime medium est=1s lane=default
func TestV41RecaptureImmutablePrefix(t *testing.T) {
	m := v41CompressorTestFixtureIndex(t, 128)
	m.Cfg.Window = []int{2, 2}
	capture := func(s *Session) *PrefixSnapshot {
		t.Helper()
		p, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(p.Close)
		return p
	}
	restore := func(p *PrefixSnapshot) *Session {
		t.Helper()
		s := m.NewSession()
		t.Cleanup(s.Close)
		owned := p.v41
		if err := p.Restore(s); err != nil {
			t.Fatal(err)
		}
		if p.v41 != nil || p.Cache != nil || owned.history != nil || owned.attn != nil || owned.layers != nil ||
			s.v41Forward.callbackOwner != s || s.v41Forward.finalNorm != nil {
			t.Fatal("restore did not transfer ownership and bind destination callbacks")
		}
		p.Close()
		return s
	}
	states := func(n *v41ForwardSnapshot) []*V41AttentionState {
		return append([]*V41AttentionState{n.attn}, n.layers...)
	}
	source := m.NewSession()
	t.Cleanup(source.Close)
	source.Prefill([]int{1, 2, 3})
	seed := capture(source)
	first := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
	second := v41AttentionPublicationKey{sourceLayer: 0, start: 1, end: 2}
	if len(seed.v41.attn.kvPublications[first]) != m.Cfg.HeadDim || len(seed.v41.attn.indexPublications[first]) != m.Cfg.IndexHeadDim ||
		&seed.v41.attn.kvPublications[first][0] == &source.v41Forward.attn.kvPublications[first][0] ||
		&seed.v41.attn.indexPublications[first][0] == &source.v41Forward.attn.indexPublications[first][0] {
		t.Fatal("ordinary capture lacks real publications or shares writable rows")
	}
	seed.PrepareForCache()
	if seed.v41.attn.publicationPrefix == nil || seed.v41.layers[0].publicationPrefix == nil {
		t.Fatal("fixture did not seal both registry and source-layer publications")
	}
	fork, err := seed.CloneForRestore()
	if err != nil {
		t.Fatal(err)
	}
	parent := restore(fork)
	parent.Step(4)
	parent.Step(5) // Retain both a completed local publication and a partial group.
	pristine := captureV41ForwardSnapshot(parent.v41Forward)
	parent.v41Forward.finalNorm = func([]float32) ([]float32, error) {
		return nil, errors.New("source callback must not follow its snapshot")
	}
	recaptured := capture(parent)
	parent.v41Forward.finalNorm = nil
	if !reflect.DeepEqual(recaptured.v41.clone(), pristine) ||
		&recaptured.v41.history[0] == &parent.v41Forward.history[0] {
		t.Fatal("recapture changed continuation or shared writable history")
	}
	live := append([]*V41AttentionState{parent.v41Forward.attn}, parent.v41Forward.layers...)
	for i, got := range states(recaptured.v41) {
		want := live[i]
		if got == want || got.publicationPrefix != want.publicationPrefix {
			t.Fatalf("state %d copied the sealed base or shared writable state", i)
		}
		for key, row := range want.kvPublications {
			if &got.kvPublications[key][0] == &row[0] {
				t.Fatalf("state %d shares a writable KV overlay", i)
			}
		}
		for key, row := range want.indexPublications {
			if &got.indexPublications[key][0] == &row[0] {
				t.Fatalf("state %d shares a writable index overlay", i)
			}
		}
	}
	layer, original := recaptured.v41.layers[0], parent.v41Forward.layers[0]
	if len(layer.partialInputs) != 1 || len(layer.partialPositions) != 1 ||
		len(recaptured.v41.attn.kvPublications[second]) != m.Cfg.HeadDim ||
		len(recaptured.v41.attn.indexPublications[second]) != m.Cfg.IndexHeadDim ||
		&layer.window[0][0] == &original.window[0][0] ||
		&layer.partialInputs[0][0] == &original.partialInputs[0][0] ||
		&layer.partialPositions[0] == &original.partialPositions[0] {
		t.Fatal("recapture lacks the mutable suffix or shares temporal storage")
	}
	public, err := recaptured.Clone()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(public.Close)
	bytes := recaptured.ResidentBytes()
	if bytes != public.ResidentBytes() || !reflect.DeepEqual(public.v41, pristine) ||
		public.v41.attn.publicationPrefix != nil ||
		&public.v41.attn.kvPublications[first][0] == &recaptured.v41.attn.publicationPrefix.kv[first][0] {
		t.Fatal("public Clone or conservative byte accounting changed")
	}
	public.v41.attn.kvPublications[first][0]++
	public.v41.attn.indexPublications[second][0]++
	if !reflect.DeepEqual(recaptured.v41.clone(), pristine) {
		t.Fatal("public Clone exposed captured payload for mutation")
	}
	public.Close()
	// Successful admission flattens descriptors, keeping old sealed payload and
	// moving only this capture's copied overlays into its new immutable owner.
	base := recaptured.v41.attn.publicationPrefix
	local := recaptured.v41.attn.kvPublications[second]
	recaptured.PrepareForCache()
	if recaptured.v41.attn.publicationPrefix == base || parent.v41Forward.attn.publicationPrefix != base ||
		&recaptured.v41.attn.publicationPrefix.kv[first][0] != &base.kv[first][0] ||
		&recaptured.v41.attn.publicationPrefix.kv[second][0] != &local[0] ||
		recaptured.ResidentBytes() != bytes || !reflect.DeepEqual(recaptured.v41.clone(), pristine) {
		t.Fatal("recapture admission copied sealed payload or changed source/state/accounting")
	}
	peer, err := recaptured.CloneForRestore()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(peer.Close)
	deep, err := recaptured.Clone()
	if err != nil {
		t.Fatal(err)
	}
	branch, control := restore(recaptured), restore(deep)
	parent.Step(6)
	parent.v41Forward.attn.Reset()
	parent.v41Forward.layers[0].Reset()
	parent.Close()
	seed.Close()
	source.Close()
	if !reflect.DeepEqual(captureV41ForwardSnapshot(branch.v41Forward), pristine) || !reflect.DeepEqual(peer.v41.clone(), pristine) {
		t.Fatal("source continuation, reset, or close changed a recaptured branch")
	}
	failure := errors.New("late failure after recaptured source publication")
	sawPublication := false
	branch.v41Forward.finalNorm = func([]float32) ([]float32, error) {
		sawPublication = branch.v41Forward.attn.kvPublishedEnd[0] == 3 && len(branch.v41Forward.attn.kvPublications) == 1
		return nil, failure
	}
	var stats v41StepStats
	var returned error
	panicked := recoverError(func() { _, stats, returned = m.forwardV41Step(6, branch.v41Forward, nil) })
	var projection *V41ProjectionOperationError
	if returned != nil || !errors.Is(panicked, failure) || !errors.As(panicked, &projection) || stats.Committed || !sawPublication ||
		!reflect.DeepEqual(captureV41ForwardSnapshot(branch.v41Forward), pristine) || !reflect.DeepEqual(peer.v41.clone(), pristine) {
		t.Fatalf("recaptured branch rollback leaked: returned=%v panic=%v", returned, panicked)
	}
	branch.v41Forward.finalNorm = nil
	if got, want := branch.Step(6), control.Step(6); !reflect.DeepEqual(got, want) {
		t.Fatal("recaptured continuation differs from its deep-owned control")
	}
	peerControl, err := peer.Clone()
	if err != nil {
		t.Fatal(err)
	}
	sibling, siblingControl := restore(peer), restore(peerControl)
	branch.v41Forward.attn.Reset()
	branch.v41Forward.layers[0].Reset()
	branch.Close()
	if got, want := sibling.Step(7), siblingControl.Step(7); !reflect.DeepEqual(got, want) {
		t.Fatal("divergent sibling lost its prefix after recaptured branch reset/close")
	}
}
