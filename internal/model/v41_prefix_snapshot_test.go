package model

// v41_prefix_snapshot_test.go is the fak#13342 acceptance witness: PrefixSnapshot
// must carry the COMPLETE DeepSeek V4.1 continuation state (the committed token
// history plus every layer's bounded temporal V41AttentionState and the shared
// source-publication registry), not just the generic KVCache. Before this leaf
// the V4.1 capability guard refused prefix reuse precisely because the snapshot
// omitted that state (v41_prefix_capability_test.go); this witness proves the
// omission is closed for capture/clone/restore/close and byte accounting.
//
// Independence: it drives the REAL reduced V4.1 assembly (v41ReducedModelLayers)
// to build the continuation state, then compares a branch carried THROUGH the
// snapshot against an unsnapshotted branch by continuing both and requiring the
// logits to agree. A snapshot that dropped history, a layer's window/partial
// group, or a shared publication would diverge here.

import (
	"reflect"
	"testing"
)

// v41PrefixSnapshotSession builds a reduced V4.1 session with an explicitly sized
// causal window per layer, a real Backend-less Cache (NewSession) and a primed
// continuation state, so PrefixSnapshot has both a Cache and V4.1 state to carry.
func v41PrefixSnapshotSession(t *testing.T, layers int, window []int, prompt []int) *Session {
	t.Helper()
	m := v41DecodeStateModel(t, layers)
	if window != nil {
		m.Cfg.Window = window
	}
	s := m.NewSession()
	if got := s.Prefill(prompt); len(got) == 0 {
		t.Fatalf("priming prefill of %v returned no logits", prompt)
	}
	if s.v41Forward == nil {
		t.Fatal("V4.1 prefill did not install a continuation state")
	}
	if s.Cache == nil {
		t.Fatal("session has no KV cache to snapshot")
	}
	return s
}

// TestV41CompletePrefixSnapshot is the named fak#13342 witness. It proves a
// snapshot captured mid-decode restores a branch that continues IDENTICALLY to
// the original, that clone/close isolate branches, that generation/geometry
// mismatches are refused atomically, and that resident-byte accounting includes
// the complete state (a count-only or nil snapshot cannot qualify).
func TestV41CompletePrefixSnapshot(t *testing.T) {
	const (
		prompt = 4
		suffix = 3
	)
	base := []int{1, 3, 5, 7}

	t.Run("restore continues identically and owns complete state", func(t *testing.T) {
		s := v41PrefixSnapshotSession(t, 2, []int{1, 2}, base)

		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		if snap.v41 == nil {
			t.Fatal("snapshot carries no V4.1 continuation state")
		}
		if got, want := snap.v41.history, base; !reflect.DeepEqual(got, want) {
			t.Fatalf("snapshot history = %v, want committed %v", got, want)
		}
		// The temporal layer states are the #13305/#13313 contract; a snapshot
		// that lost them would continue from a bare token history and replay the
		// prefix rather than reuse prepared state.
		if len(snap.v41.layers) != 2 || snap.v41.layers[0] == nil || snap.v41.layers[1] == nil {
			t.Fatalf("snapshot layers = %v, want two populated temporal states", snap.v41.layers)
		}
		// The reduced fixture declares no KV-source layer, so the shared registry
		// is legitimately absent (nil). When a source layer DID publish, the
		// registry must be carried -- exercised by the registry subtest below.
		if s.v41Forward.attn != nil && snap.v41.attn == nil {
			t.Fatal("snapshot dropped the shared source-publication registry")
		}

		// Deep ownership: captured rows must not alias the live session's rows.
		if &snap.v41.history[0] == &s.v41Forward.history[0] {
			t.Fatal("snapshot history aliases the live session history")
		}
		if snap.v41.layers[0] == s.v41Forward.layerState(0) {
			t.Fatal("snapshot layer state aliases the live session layer state")
		}

		// Continue the ORIGINAL and the RESTORED branch over the same suffix and
		// require identical logits. Any dropped continuation field diverges here.
		branch := &Session{M: s.M, Cache: s.Cache.Clone()}
		if err := snap.Restore(branch); err != nil {
			t.Fatalf("Restore: %v", err)
		}
		want := s.Step(6)
		got := branch.Step(6)
		assertV41LogitsClose(t, got, want, "restored-vs-original")

		wantAgain := s.Step(2)
		gotAgain := branch.Step(2)
		assertV41LogitsClose(t, gotAgain, wantAgain, "restored-vs-original second step")

		// Restore transferred ownership: the snapshot no longer holds device or
		// state rows, so a second restore of the same snapshot cannot be a silent
		// reuse of a live branch.
		if snap.v41 != nil {
			t.Fatal("Restore did not clear the snapshot's ownership")
		}
	})

	t.Run("shared publications and selections survive capture", func(t *testing.T) {
		// Drive the registry and temporal state directly so the shared compressed
		// index publications, the incomplete compressor group, and the latest
		// candidate/top-k selections are all present at capture. The reduced
		// fixture's single decoder layer never declares a KV source, so this is
		// the only way to witness those fields are carried (fak#13342 checklist).
		state, err := NewV41AttentionState(2, 8)
		if err != nil {
			t.Fatal(err)
		}
		if err := state.PublishCandidates(4, []bool{true, false, true}); err != nil {
			t.Fatalf("PublishCandidates: %v", err)
		}
		if err := state.PublishTopK(4, [][]int32{{0, 2}, {1, 3}}); err != nil {
			t.Fatalf("PublishTopK: %v", err)
		}
		if err := state.publishUpdates([]V41AttentionStateUpdate{
			{Ref: V41AttentionStateRef{LayerID: 2, Ratio: 4, IsKVSource: true, IsIndexSource: true}, Latent: []float32{1, 2}, IndexKey: []float32{11, 12}},
			{Ref: V41AttentionStateRef{LayerID: 2, Ratio: 4, IsKVSource: true, IsIndexSource: true}, Latent: []float32{3, 4}, IndexKey: []float32{13, 14}},
		}); err != nil {
			t.Fatalf("publishUpdates: %v", err)
		}

		snap := &PrefixSnapshot{owner: nil, Cache: NewKVCache(Config{}), v41: &v41ForwardSnapshot{
			history:  []int{1, 3},
			attn:     state,
			hadState: true,
		}}
		clone, err := snap.Clone()
		if err != nil {
			t.Fatalf("Clone: %v", err)
		}
		got, ok := clone.v41.attn.KVSourceRows(2)
		if !ok || !reflect.DeepEqual(got, [][]float32{{1, 2}, {3, 4}}) {
			t.Fatalf("carried KV publications = %v, %t, want [[1 2] [3 4]]", got, ok)
		}
		idx, ok := clone.v41.attn.IndexKeys(2)
		if !ok || !reflect.DeepEqual(idx, [][]float32{{11, 12}, {13, 14}}) {
			t.Fatalf("carried index publications = %v, %t", idx, ok)
		}
		topk, ratio, ok := clone.v41.attn.TopK()
		if !ok || ratio != 4 || !reflect.DeepEqual(topk, [][]int32{{0, 2}, {1, 3}}) {
			t.Fatalf("carried top-k = %v ratio %d set %t", topk, ratio, ok)
		}
		cand, cRatio, ok := clone.v41.attn.Candidates()
		if !ok || cRatio != 4 || !reflect.DeepEqual(cand, []bool{true, false, true}) {
			t.Fatalf("carried candidates = %v ratio %d set %t", cand, cRatio, ok)
		}
		// Deep isolation: mutating the clone's publication must not touch the source.
		clone.v41.attn.kvPublications[v41AttentionPublicationKey{sourceLayer: 2, start: 0, end: 1}][0] = 99
		if got := snap.v41.attn.kvPublications[v41AttentionPublicationKey{sourceLayer: 2, start: 0, end: 1}][0]; got == 99 {
			t.Fatal("clone shares publication rows with its source")
		}
	})

	t.Run("clone isolates branches and close is safe", func(t *testing.T) {
		s := v41PrefixSnapshotSession(t, 2, []int{1, 2}, base)
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		second, err := snap.Clone()
		if err != nil {
			t.Fatalf("Clone: %v", err)
		}
		if second == nil || second.v41 == nil {
			t.Fatal("Clone produced no V4.1 state")
		}
		// Mutating one clone's history/rows must not alter the other.
		second.v41.history[0] = 999
		if snap.v41.history[0] == 999 {
			t.Fatal("clone shares mutable history with its source")
		}
		second.v41.layers[0].window[0][0] = -7
		if snap.v41.layers[0].window[0][0] == -7 {
			t.Fatal("clone shares mutable window rows with its source")
		}
		// Closing one branch leaves the other usable and does not panic.
		second.Close()
		if second.v41 != nil {
			t.Fatal("Close did not release the V4.1 state")
		}
		third, err := snap.Clone()
		if err != nil {
			t.Fatalf("Clone after sibling Close: %v", err)
		}
		third.Close()
	})

	t.Run("failed restore is atomic and mismatched geometry refused", func(t *testing.T) {
		s := v41PrefixSnapshotSession(t, 2, []int{1, 2}, base)
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		// A nil target session is refused without consuming the snapshot.
		if err := snap.Restore(nil); err == nil {
			t.Fatal("Restore(nil) succeeded")
		}
		// An epoch mismatch (cache rebuilt after the snapshot) is refused and the
		// snapshot's state survives so a valid target can still consume it.
		s.cacheGeometryMu.Lock()
		s.cacheGeometryEpoch++
		s.cacheGeometryMu.Unlock()
		target := m2Session(t, s)
		if err := snap.Restore(target); err == nil {
			t.Fatal("Restore after cache rebuild succeeded, want stale-epoch refusal")
		}
		if snap.v41 == nil {
			t.Fatal("failed restore consumed the snapshot's V4.1 state")
		}
	})

	t.Run("resident bytes include the complete state", func(t *testing.T) {
		s := v41PrefixSnapshotSession(t, 2, []int{1, 2}, base)
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		if got := snap.v41ResidentBytes(); got <= 0 {
			t.Fatalf("v41ResidentBytes = %d, want a positive payload for a populated snapshot", got)
		}
		// A snapshot of a session with no V4.1 state must not inflate the ledger.
		bare := &PrefixSnapshot{Cache: NewKVCache(s.M.Cfg)}
		if got := bare.v41ResidentBytes(); got != 0 {
			t.Fatalf("bare snapshot v41ResidentBytes = %d, want 0", got)
		}
		// The complete-state bytes are counted in the top-level ResidentBytes too.
		if snap.ResidentBytes() < snap.v41ResidentBytes() {
			t.Fatalf("ResidentBytes %d < V4.1 payload %d", snap.ResidentBytes(), snap.v41ResidentBytes())
		}
	})
}

// m2Session builds a second session over the same model for restore-target tests.
func m2Session(t *testing.T, s *Session) *Session {
	t.Helper()
	return &Session{M: s.M, Cache: s.Cache.Clone()}
}
