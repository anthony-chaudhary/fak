package model

import (
	"fmt"
	"testing"
)

const v41RowViewSources = 2

func v41RowViewRow(source, i, salt int) []float32 {
	return []float32{float32(source*10000+i) * 0.125, float32(salt) - 0.5}
}

func v41RowViewPublish(t testing.TB, s *V41AttentionState, n, salt int) {
	t.Helper()
	for i := 0; i < n; i++ {
		for source := 0; source < v41RowViewSources; source++ {
			at := s.kvPublishedEnd[source]
			if err := s.publishUpdates([]V41AttentionStateUpdate{{
				Ref:    V41AttentionStateRef{LayerID: source, Ratio: 2, IsKVSource: true, IsIndexSource: true},
				Latent: v41RowViewRow(source, at, salt), IndexKey: v41RowViewRow(source, at, salt+1),
			}}); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func v41RowViewState(t testing.TB, n int) *V41AttentionState {
	t.Helper()
	s, err := NewV41AttentionState(2, 8)
	if err != nil {
		t.Fatal(err)
	}
	v41RowViewPublish(t, s, n, 0)
	return s
}

// v41RowViewMapRows is the pre-view borrow: one publication lookup per row.
func v41RowViewMapRows(s *V41AttentionState, index bool, source int) ([][]float32, bool) {
	end := s.kvPublishedEnd[source]
	if index {
		end = s.indexPublishedEnd[source]
	}
	if end == 0 {
		return nil, false
	}
	out := make([][]float32, 0, end)
	for i := 0; i < end; i++ {
		key := v41AttentionPublicationKey{sourceLayer: source, start: i, end: i + 1}
		row, ok := s.kvPublication(key)
		if index {
			row, ok = s.indexPublication(key)
		}
		if !ok {
			return nil, false
		}
		out = append(out, row)
	}
	return out, true
}

func v41RowViewBorrow(s *V41AttentionState, index bool, source int) ([][]float32, bool) {
	if index {
		return s.borrowIndexKeys(source)
	}
	return s.borrowKVSourceRows(source)
}

func v41RowViewCountRebuilds(t *testing.T) *int {
	t.Helper()
	rebuilds := 0
	prior := v41RowViewRebuildProbe
	v41RowViewRebuildProbe = func(bool, int, int) { rebuilds++ }
	t.Cleanup(func() { v41RowViewRebuildProbe = prior })
	return &rebuilds
}

// v41RowViewSame requires every borrow to return exactly the map path's row
// pointers, with a capacity that forces a consumer append to reallocate, and the
// view to retain no row beyond the published end.
func v41RowViewSame(t *testing.T, edge string, s *V41AttentionState) {
	t.Helper()
	for _, index := range []bool{false, true} {
		for source := 0; source < v41RowViewSources; source++ {
			want, wantOK := v41RowViewMapRows(s, index, source)
			got, ok := v41RowViewBorrow(s, index, source)
			if ok != wantOK || len(got) != len(want) || cap(got) != len(got) {
				t.Fatalf("%s index=%t source=%d: borrow ok=%t len=%d cap=%d, map ok=%t len=%d", edge, index, source, ok, len(got), cap(got), wantOK, len(want))
			}
			for i := range want {
				if len(got[i]) != len(want[i]) || &got[i][0] != &want[i][0] {
					t.Fatalf("%s index=%t source=%d: row %d is not the published row", edge, index, source, i)
				}
			}
			views := s.kvRowViews
			if index {
				views = s.indexRowViews
			}
			if view, armed := views[source]; armed && len(view) != len(want) {
				t.Fatalf("%s index=%t source=%d: view retains %d rows for %d published", edge, index, source, len(view), len(want))
			}
		}
	}
}

// The view serves exactly the map path's rows across every lifecycle edge and
// only the first borrow after a state is created, forked, restored or reset
// walks the maps; publish and rollback keep the view without a rebuild.
// fak-test:runtime fast est=20ms lane=default
func TestV41AttentionStateRowViewLifecycle(t *testing.T) {
	rebuilds := v41RowViewCountRebuilds(t)
	const perState = 2 * v41RowViewSources
	expect := func(edge string, want int) {
		t.Helper()
		if *rebuilds != want {
			t.Fatalf("%s: %d map walks, want %d", edge, *rebuilds, want)
		}
	}
	s := v41RowViewState(t, 64)
	v41RowViewSame(t, "first borrow", s)
	expect("first borrow", perState)
	v41RowViewSame(t, "repeat borrow", s)
	v41RowViewPublish(t, s, 3, 0)
	v41RowViewSame(t, "publish", s)
	expect("publish", perState)

	t.Run("rollback", func(t *testing.T) {
		before := *rebuilds
		undo := stageV41StepRegistry(s, v41RowViewSources)
		v41RowViewPublish(t, s, 1, 7)
		rolled, _ := s.borrowKVSourceRows(0)
		dropped := rolled[len(rolled)-1]
		undo.restore()
		v41RowViewSame(t, "rollback", s)
		v41RowViewPublish(t, s, 1, 9)
		v41RowViewSame(t, "republish after rollback", s)
		if got, _ := s.borrowKVSourceRows(0); &got[len(got)-1][0] == &dropped[0] {
			t.Fatal("rolled-back row survived its replacement")
		}
		undo = stageV41StepRegistry(s, v41RowViewSources)
		v41RowViewPublish(t, s, 1, 11)
		undo.restore()
		v41RowViewPublish(t, s, 1, 13)
		v41RowViewSame(t, "rollback without an intermediate borrow", s)
		expect("rollback", before)
	})

	t.Run("seal", func(t *testing.T) {
		before := *rebuilds
		s.sealPublications()
		v41RowViewSame(t, "seal", s)
		v41RowViewPublish(t, s, 2, 0)
		v41RowViewSame(t, "publish over sealed prefix", s)
		expect("seal", before)
	})

	t.Run("fork-and-restore", func(t *testing.T) {
		for _, shareSealed := range []bool{false, true} {
			before := *rebuilds
			fork := s.cloneWithPublications(shareSealed)
			if fork.kvRowViews != nil || fork.indexRowViews != nil {
				t.Fatal("fork carried the live row view")
			}
			v41RowViewSame(t, "fork", fork)
			v41RowViewPublish(t, fork, 2, 21)
			v41RowViewSame(t, "fork publish", fork)
			v41RowViewSame(t, "origin after fork publish", s)
			expect("fork", before+perState)

			snap := captureV41ForwardSnapshotWithPublications(&v41ForwardState{attn: s, layers: []*V41AttentionState{s}}, shareSealed)
			restored := snap.restore()
			for _, state := range append([]*V41AttentionState{restored.attn}, restored.layers...) {
				if state.kvRowViews != nil || state.indexRowViews != nil {
					t.Fatal("restored state carried the live row view")
				}
				v41RowViewSame(t, "restore", state)
				v41RowViewPublish(t, state, 1, 31)
				v41RowViewSame(t, "restore publish", state)
			}
			expect("restore", before+3*perState)
		}
	})

	t.Run("reset-and-reseed", func(t *testing.T) {
		before := *rebuilds
		s.Reset()
		if s.kvRowViews != nil || s.indexRowViews != nil {
			t.Fatal("reset kept the row view")
		}
		v41RowViewSame(t, "reset", s)
		v41RowViewPublish(t, s, 4, 41)
		v41RowViewSame(t, "republish after reset", s)
		expect("reset", before+perState)
		if err := s.seedTemporal([][]float32{{1, 2}, {3, 4}}, 2, -1); err != nil {
			t.Fatal(err)
		}
		if s.kvRowViews != nil || s.indexRowViews != nil {
			t.Fatal("reseed kept the row view")
		}
		v41RowViewSame(t, "reseed", s)
	})
}

// Malformed registries still refuse through the map walk even when the live
// state already holds an armed view, and copying readers stay deep copies.
// fak-test:runtime fast est=10ms lane=default
func TestV41AttentionStateRowViewRefusals(t *testing.T) {
	t.Run("missing-store", func(t *testing.T) {
		s := v41RowViewState(t, 8)
		v41RowViewSame(t, "armed", s)
		kv, index := s.kvPublications, s.indexPublications
		s.kvPublications, s.indexPublications = nil, nil
		for _, idx := range []bool{false, true} {
			if rows, ok := v41RowViewBorrow(s, idx, 0); ok || rows != nil {
				t.Fatalf("index=%t: armed view served a missing publication store", idx)
			}
		}
		s.kvPublications, s.indexPublications = kv, index
		v41RowViewSame(t, "store restored", s)
	})
	t.Run("hole", func(t *testing.T) {
		s := v41RowViewState(t, 8)
		v41RowViewSame(t, "armed", s)
		fork := s.clone()
		delete(fork.kvPublications, v41AttentionPublicationKey{sourceLayer: 1, start: 3, end: 4})
		delete(fork.indexPublications, v41AttentionPublicationKey{sourceLayer: 1, start: 3, end: 4})
		for _, idx := range []bool{false, true} {
			if rows, ok := v41RowViewBorrow(fork, idx, 1); ok || rows != nil {
				t.Fatalf("index=%t: borrow served a publication set with a hole", idx)
			}
		}
		v41RowViewSame(t, "unholed source", fork)
	})
	t.Run("stale-anchor", func(t *testing.T) {
		s := v41RowViewState(t, 8)
		v41RowViewSame(t, "armed", s)
		key := v41AttentionPublicationKey{sourceLayer: 0, start: 7, end: 8}
		s.kvPublications[key] = []float32{5, 6}
		s.indexPublications[key] = []float32{7, 8}
		v41RowViewSame(t, "replaced terminal row", s)
		s.kvPublishedEnd[0], s.indexPublishedEnd[0] = 9, 9
		for _, idx := range []bool{false, true} {
			if rows, ok := v41RowViewBorrow(s, idx, 0); ok || rows != nil {
				t.Fatalf("index=%t: borrow served an end beyond the publications", idx)
			}
		}
	})
	t.Run("copying-readers", func(t *testing.T) {
		s := v41RowViewState(t, 8)
		v41RowViewSame(t, "armed", s)
		for _, read := range []func(int) ([][]float32, bool){s.KVSourceRows, s.IndexKeys} {
			copied, ok := read(0)
			if !ok || len(copied) != 8 {
				t.Fatal("copying reader refused a complete source")
			}
			kv, _ := s.borrowKVSourceRows(0)
			keys, _ := s.borrowIndexKeys(0)
			for i, row := range copied {
				if &row[0] == &kv[i][0] || &row[0] == &keys[i][0] {
					t.Fatal("copying reader returned a borrowed row")
				}
			}
		}
	})
}

// Steady-state borrows allocate nothing and walk no publications at any
// history length; a forced map walk allocates, so the witness discriminates.
// fak-test:runtime fast est=50ms lane=default
func TestV41AttentionStateRowViewBorrowConstant(t *testing.T) {
	rebuilds := v41RowViewCountRebuilds(t)
	for _, n := range []int{64, 4096} {
		s := v41RowViewState(t, n)
		v41RowViewSame(t, fmt.Sprintf("n=%d", n), s)
		*rebuilds = 0
		borrow := func() {
			for source := 0; source < v41RowViewSources; source++ {
				if rows, ok := s.borrowKVSourceRows(source); !ok || len(rows) != n {
					t.Fatal("borrow refused")
				}
				if keys, ok := s.borrowIndexKeys(source); !ok || len(keys) != n {
					t.Fatal("borrow refused")
				}
			}
		}
		if allocs := testing.AllocsPerRun(100, borrow); allocs != 0 || *rebuilds != 0 {
			t.Fatalf("n=%d: borrow allocated %v times and walked the maps %d times, want 0 and 0", n, allocs, *rebuilds)
		}
		walk := testing.AllocsPerRun(10, func() {
			s.kvRowViews = nil
			if _, ok := s.borrowKVSourceRows(0); !ok {
				t.Fatal("map walk refused")
			}
		})
		if walk < 1 || *rebuilds == 0 {
			t.Fatalf("n=%d: forced map walk allocated %v times over %d walks", n, walk, *rebuilds)
		}
	}
}

// The decode role step is the view's caller: across consecutive positions it
// serves every source and reader borrow from the view, and stays bit-identical
// to a run whose views are dropped before every layer.
// fak-test:runtime fast est=3s lane=default
func TestV41AttentionStateRowViewRoleStep(t *testing.T) {
	for _, indexed := range []bool{false, true} {
		m := v41ReaderWidthFixture(t)
		if indexed {
			m = v41IncrementalExpertFixture(t, true, false)
		}
		rebuilds := v41RowViewCountRebuilds(t)
		run := func(view bool) (int, [][]float32) {
			st := v41RoleContinuationSeed(t, m, []int{1, 2, 3, 4})
			var outs [][]float32
			walks := 0
			for pos := 4; pos < 12; pos++ {
				x, streams := v41LayerStepInputs(t, m, 1+pos%4, true)
				scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
				for l := 0; l < m.Cfg.NumLayers; l++ {
					plan, err := m.v41AttentionPlan(l)
					if err != nil {
						t.Fatal(err)
					}
					if !view {
						for _, state := range []*V41AttentionState{st.layerState(l), st.attn} {
							state.kvRowViews, state.indexRowViews = nil, nil
						}
					}
					*rebuilds = 0
					work, err := m.prepareV41LayerStepRole(l, plan, x, streams, pos, st.layerState(l), st.attn, scratch)
					if err != nil {
						t.Fatal(err)
					}
					if err = work.complete(nil); err != nil {
						t.Fatal(err)
					}
					if pos > 4 {
						walks += *rebuilds
					}
				}
				outs = append(outs, append([]float32(nil), x...))
				for _, stream := range streams {
					outs = append(outs, append([]float32(nil), stream...))
				}
			}
			return walks, outs
		}
		controlWalks, control := run(false)
		viewWalks, got := run(true)
		if len(got) != len(control) {
			t.Fatalf("indexed=%t: view run took a different number of steps", indexed)
		}
		for i := range got {
			if !v41SameBits(got[i], control[i]) {
				t.Fatalf("indexed=%t: row view changed step output %d", indexed, i)
			}
		}
		if controlWalks == 0 || viewWalks != 0 {
			t.Fatalf("indexed=%t: map walks after the first decode step: view=%d control=%d, want 0 and >0", indexed, viewWalks, controlWalks)
		}
	}
}

func BenchmarkV41AttentionStateRowViewBorrow(b *testing.B) {
	for _, n := range []int{64, 4096} {
		s := v41RowViewState(b, n)
		for _, mode := range []string{"view", "map-walk"} {
			b.Run(fmt.Sprintf("n=%d/%s", n, mode), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if mode == "map-walk" {
						s.kvRowViews, s.indexRowViews = nil, nil
					}
					for source := 0; source < v41RowViewSources; source++ {
						if _, ok := s.borrowKVSourceRows(source); !ok {
							b.Fatal("borrow refused")
						}
						if _, ok := s.borrowIndexKeys(source); !ok {
							b.Fatal("borrow refused")
						}
					}
				}
			})
		}
	}
}
