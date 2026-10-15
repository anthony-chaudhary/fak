package model

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestV41SourcePublicationCopyBoundary(t *testing.T) {
	m := &Model{Cfg: Config{DeepSeekV41: &DeepSeekV41Config{IndexSourceLayerIDs: []int{0}}}}
	plan := V41AttentionPlan{Layer: 0, KVSourceLayer: 0, Ratio: 2}
	newState := func(t *testing.T) *V41AttentionState {
		t.Helper()
		s, err := NewV41AttentionState(4, 8)
		if err != nil {
			t.Fatal(err)
		}
		s.indexHeadDim = 2
		return s
	}
	key := func(group int) v41AttentionPublicationKey {
		return v41AttentionPublicationKey{sourceLayer: 0, start: group, end: group + 1}
	}

	t.Run("borrow-until-owned-publication", func(t *testing.T) {
		latents := [][]float32{{1, 2, 3, 4, 99}, {5, 6, 7, 8, 99}}
		keys := [][]float32{{11, 12, 99}, {13, 14, 99}}
		for i := range latents {
			latents[i] = latents[i][:4]
			keys[i] = keys[i][:2]
		}
		states := []*V41AttentionState{newState(t), newState(t)}
		for _, s := range states {
			updates := m.v41AttentionSourceUpdates(plan, latents, nil, keys)
			if len(updates) != 2 {
				t.Fatalf("updates = %d, want two completed groups", len(updates))
			}
			for i, up := range updates {
				if len(up.Latent) != 4 || cap(up.Latent) != 4 || &up.Latent[0] != &latents[i][0] || len(up.IndexKey) != 2 || cap(up.IndexKey) != 2 || &up.IndexKey[0] != &keys[i][0] {
					t.Fatal("builder copied a payload or exposed capacity beyond the completed row")
				}
			}
			if err := s.publishUpdates(updates); err != nil {
				t.Fatal(err)
			}
		}
		for i := range latents {
			latents[i][0], keys[i][0] = -1, -2
		}
		wantLatents := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
		wantKeys := [][]float32{{11, 12}, {13, 14}}
		for _, s := range states {
			if s.kvPublishedEnd[0] != 2 || s.indexPublishedEnd[0] != 2 || len(s.kvPublications) != 2 || len(s.indexPublications) != 2 {
				t.Fatal("publication lost completed groups")
			}
			for i := range wantLatents {
				if !reflect.DeepEqual(s.kvPublications[key(i)], wantLatents[i]) || !reflect.DeepEqual(s.indexPublications[key(i)], wantKeys[i]) {
					t.Fatal("owned publication changed after source mutation")
				}
			}
		}
		states[0].kvPublications[key(0)][0] = -3
		states[0].indexPublications[key(0)][0] = -4
		if !reflect.DeepEqual(states[1].kvPublications[key(0)], wantLatents[0]) || !reflect.DeepEqual(states[1].indexPublications[key(0)], wantKeys[0]) {
			t.Fatal("owner and registry publications share payload storage")
		}
	})

	t.Run("late-invalid-update-is-atomic", func(t *testing.T) {
		s := newState(t)
		seed := m.v41AttentionSourceUpdates(plan, [][]float32{{1, 2, 3, 4}}, nil, [][]float32{{5, 6}})
		if err := s.publishUpdates(seed); err != nil {
			t.Fatal(err)
		}
		updates := m.v41AttentionSourceUpdates(plan, [][]float32{{7, 8, 9, 10}, {11, 12, 13, 14}}, nil, [][]float32{{15, 16}, {17, float32(math.NaN())}})
		if err := s.publishUpdates(updates); err == nil {
			t.Fatal("non-finite late key was accepted")
		}
		if s.kvPublishedEnd[0] != 1 || s.indexPublishedEnd[0] != 1 || len(s.kvPublications) != 1 || len(s.indexPublications) != 1 || !reflect.DeepEqual(s.kvPublications[key(0)], []float32{1, 2, 3, 4}) || !reflect.DeepEqual(s.indexPublications[key(0)], []float32{5, 6}) {
			t.Fatal("invalid batch changed an earlier publication or counter")
		}
	})

	t.Run("empty-payload-normalization", func(t *testing.T) {
		updates := m.v41AttentionSourceUpdates(plan, [][]float32{nil, {}}, nil, [][]float32{{}, nil})
		if len(updates) != 2 {
			t.Fatal("empty payloads changed descriptor count")
		}
		for _, up := range updates {
			if up.Latent != nil || up.IndexKey != nil {
				t.Fatal("zero-length payload lost clone-to-nil semantics")
			}
		}
		s := newState(t)
		if err := s.publishUpdates(updates); err != nil {
			t.Fatal(err)
		}
		if len(s.kvPublications) != 0 || len(s.indexPublications) != 0 || s.kvPublishedEnd[0] != 0 || s.indexPublishedEnd[0] != 0 {
			t.Fatal("empty payload published a completed row")
		}
		if got := m.v41AttentionSourceUpdates(plan, nil, [][]float32{{1, 2}}); got != nil {
			t.Fatal("empty compressed stream published query latents")
		}
	})
}
