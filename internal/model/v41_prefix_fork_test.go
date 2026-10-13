package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"
)

// CPU ownership/continuation witness using the existing compressed source-reader
// fixture. The estimate is unmeasured; no physical cache-hit speed is claimed.
// fak-test:runtime medium est=3s lane=default
func TestV41ImmutablePrefixFork(t *testing.T) {
	t.Run("sealed_forks_restore_rollback_and_reset_independently", func(t *testing.T) {
		m := v41CompressorTestFixtureIndex(t, 128)
		source := m.NewSession()
		t.Cleanup(source.Close)
		source.Prefill([]int{1, 2, 3})
		cached, err := source.PrefixSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(cached.Close)
		pristine := cached.v41.clone()
		bytes := cached.ResidentBytes()
		key := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
		row := cached.v41.attn.kvPublications[key]
		if len(row) != 512 || len(cached.v41.attn.indexPublications[key]) != 128 {
			t.Fatal("fixture lacks the real compressed publication widths")
		}
		cached.PrepareForCache()
		base := cached.v41.attn.publicationPrefix
		if base == nil || &base.kv[key][0] != &row[0] || len(cached.v41.attn.kvPublications) != 0 ||
			cached.ResidentBytes() != bytes || !reflect.DeepEqual(cached.v41.clone(), pristine) {
			t.Fatal("sealing copied payload, retained writable rows, or changed state/accounting")
		}
		cached.PrepareForCache()
		if cached.v41.attn.publicationPrefix != base {
			t.Fatal("repeated admission replaced an already sealed owner")
		}

		// Cache publication precedes hits. Concurrent hits only read the sealed
		// maps, retaining the same row allocations and independent mutable state.
		var forks [4]*PrefixSnapshot
		var errs [4]error
		var wg sync.WaitGroup
		for i := range forks {
			wg.Add(1)
			go func() {
				defer wg.Done()
				forks[i], errs[i] = cached.CloneForRestore()
			}()
		}
		wg.Wait()
		for i, fork := range forks {
			if errs[i] != nil || fork == nil {
				t.Fatalf("fork %d: %v", i, errs[i])
			}
			t.Cleanup(fork.Close)
			state := fork.v41.attn
			if state == cached.v41.attn || state.publicationPrefix != base || len(state.kvPublications) != 0 ||
				fork.ResidentBytes() != bytes || &fork.v41.layers[0].window[0][0] == &cached.v41.layers[0].window[0][0] ||
				&fork.v41.history[0] == &cached.v41.history[0] {
				t.Fatal("fork copied sealed payload or shared writable continuation state")
			}
		}
		public, err := forks[0].Clone()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(public.Close)
		if public.v41.attn.publicationPrefix != nil || !reflect.DeepEqual(public.v41, pristine) ||
			&public.v41.attn.kvPublications[key][0] == &base.kv[key][0] {
			t.Fatal("public Clone did not materialize independently writable rows")
		}
		public.v41.attn.kvPublications[key][0] = 999
		inspected, _ := forks[0].v41.attn.KVSourceRows(0)
		inspected[0][0] = 999
		if base.kv[key][0] != pristine.attn.kvPublications[key][0] {
			t.Fatal("public clone or reader exposed a writable sealed row")
		}

		newBranch := func(snapshot *PrefixSnapshot) *Session {
			t.Helper()
			s := m.NewSession()
			t.Cleanup(s.Close)
			if err := snapshot.Restore(s); err != nil {
				t.Fatal(err)
			}
			snapshot.Close()
			return s
		}
		left, right := newBranch(forks[0]), newBranch(forks[1])
		controlSnapshot, err := cached.Clone()
		if err != nil {
			t.Fatal(err)
		}
		leftControl := newBranch(controlSnapshot)
		controlSnapshot, err = cached.Clone()
		if err != nil {
			t.Fatal(err)
		}
		rightControl := newBranch(controlSnapshot)
		cached.Close()
		source.Close()
		forks[2].Close()

		before := captureV41ForwardSnapshot(left.v41Forward)
		failure := errors.New("late head after compressed source publication")
		sawPublication := false
		left.v41Forward.finalNorm = func([]float32) ([]float32, error) {
			sawPublication = left.v41Forward.attn.kvPublishedEnd[0] == 2 && len(left.v41Forward.attn.kvPublications) == 1
			return nil, failure
		}
		var stats v41StepStats
		var returned error
		panicked := recoverError(func() { _, stats, returned = m.forwardV41Step(4, left.v41Forward, nil) })
		var projection *V41ProjectionOperationError
		if returned != nil || !errors.Is(panicked, failure) || !errors.As(panicked, &projection) || stats.Committed || !sawPublication ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(left.v41Forward), before) ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(right.v41Forward), pristine) {
			t.Fatalf("fork rollback changed a prefix or sibling: returned=%v panic=%v", returned, panicked)
		}
		left.v41Forward.finalNorm = nil
		for _, pair := range []struct {
			branch, control *Session
			token           int
		}{{left, leftControl, 4}, {right, rightControl, 5}} {
			if got, want := pair.branch.Step(pair.token), pair.control.Step(pair.token); !reflect.DeepEqual(got, want) {
				t.Fatal("restored fork continuation differs from the deep-cloned control")
			}
		}
		if !reflect.DeepEqual(forks[3].v41.clone(), pristine) || &left.v41Forward.attn.kvPublications[v41AttentionPublicationKey{sourceLayer: 0, start: 1, end: 2}][0] ==
			&right.v41Forward.attn.kvPublications[v41AttentionPublicationKey{sourceLayer: 0, start: 1, end: 2}][0] {
			t.Fatal("branch writes changed the sealed cache or shared an append row")
		}
		left.v41Forward.attn.Reset()
		left.v41Forward.layers[0].Reset()
		if left.v41Forward.attn.publicationPrefix != nil || left.v41Forward.layers[0].publicationPrefix != nil ||
			right.v41Forward.attn.publicationPrefix != base {
			t.Fatal("Reset did not detach only its own immutable prefix")
		}
		left.Close()
		if got, want := right.Step(6), rightControl.Step(6); !reflect.DeepEqual(got, want) {
			t.Fatal("closing/resetting a sibling invalidated the retained branch")
		}
	})

	t.Run("shadow_accounting_materialization_and_reseal", func(t *testing.T) {
		state, _ := NewV41AttentionState(2, 2)
		key := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
		state.publish([]V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 0}, Latent: []float32{1, 2}, IndexKey: []float32{3, 4}}})
		state.sealPublications()
		base := state.publicationPrefix
		// An overwritten local key must charge and materialize only its visible
		// value. Different lengths make accidental double-counting non-vacuous.
		state.kvPublications[key] = []float32{5, 6, 7}
		bytes, canonical := state.residentBytes(), state.clone()
		if bytes != canonical.residentBytes() {
			t.Fatal("shadowed base row was charged twice")
		}
		state.sealPublications()
		if state.residentBytes() != bytes || !reflect.DeepEqual(state.clone(), canonical) ||
			!reflect.DeepEqual(base.kv[key], []float32{1, 2}) {
			t.Fatal("resealing changed accounting, visible rows, or a prior immutable owner")
		}
		state.indexPublications = nil
		prefix := state.publicationPrefix
		state.sealPublications()
		if state.indexPublications != nil || state.publicationPrefix != prefix {
			t.Fatal("sealing repaired a malformed publication store")
		}
	})
}
