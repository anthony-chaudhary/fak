package ctxplan

import (
	"context"
	"testing"
)

// chosenSet folds a ProbeResult's chosen paths into a set for assertions.
func chosenSet(r ProbeResult) map[AccessPath]bool {
	m := make(map[AccessPath]bool, len(r.Chosen))
	for _, p := range r.Chosen {
		m[p] = true
	}
	return m
}

// estimateFor returns the cost estimate the chooser read for one path.
func estimateFor(r ProbeResult, p AccessPath) pathEstimate {
	for _, e := range r.Estimates {
		if e.Path == p {
			return e
		}
	}
	return pathEstimate{Path: p}
}

// allPathsProbe is the unconditional union (every access path scanned) — the baseline a
// cost-selected probe must equal on the qualifying-row set. It is exactly what Index.Probe
// returns; we name it here to make the equivalence assertions read as "cost-selected == union".
func allPathsProbe(ix *Index, f Forecast, opts ProbeOptions) []Span {
	return ix.Probe(f, opts)
}

// recencyAllRelevantStore builds a store whose recency TAIL is entirely relevant (every span in
// the window matches the forecast intent) while a block of irrelevant noise sits in the MIDDLE,
// outside the window. With such a store a selective forecast's relevance path already reaches
// every span the recency tail would, so the cost model can PROVE the recency seq-scan is
// redundant and drop it — the Index-Scan-instead-of-Seq-Scan win.
func recencyAllRelevantStore(noise, tail int) *MemStore {
	st := NewMemStore()
	st.Add("user", DurabilityDurable, []byte("auth token rotation runbook anchor"), false) // span:0 relevant+durable
	for i := 0; i < noise; i++ {
		st.Add("Bash", DurabilityTurn, []byte("build log line "+itoaTest(i)+" compiled quietly"), false)
	}
	for i := 0; i < tail; i++ {
		// Every tail span matches the "auth token" intent, so the relevance path reaches it.
		st.Add("Read", DurabilitySession, []byte("auth token note "+itoaTest(i)+" for the service"), false)
	}
	return st
}

// TestAccessPathDropsRecencyWhenRelevanceCovers is the cheapest-single-path witness: a highly
// selective forecast over a store whose recency tail is entirely relevant lets the cost model
// drop the recency Seq Scan — the inverted Index Scan alone reaches the load-bearing spans, and
// the chosen-path probe returns the SAME candidate set as the full union at fewer scanned paths.
func TestAccessPathDropsRecencyWhenRelevanceCovers(t *testing.T) {
	ctx := context.Background()
	st := recencyAllRelevantStore(200, 8)
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)
	// Window covers exactly the relevant tail; every windowed span is a relevance hit.
	opts := ProbeOptions{RecencyWindow: 8}
	f := Forecast{Intents: []string{"auth token note service"}}

	res := ix.ProbePlan(f, opts)
	chosen := chosenSet(res)

	if chosen[PathRecency] {
		t.Errorf("a selective forecast whose relevance path covers the whole window must DROP the recency Seq Scan; chosen=%v", res.Chosen)
	}
	if !chosen[PathRelevance] {
		t.Errorf("the inverted Index Scan must be chosen for a forecast with matches; chosen=%v", res.Chosen)
	}
	if !chosen[PathPin] || !chosen[PathDurable] {
		t.Errorf("pin and durable paths are always cheap-and-kept; chosen=%v", res.Chosen)
	}

	// CORRECTNESS PRESERVED: the cost-selected candidate set equals the full union's.
	assertSameSpanSet(t, res.Spans, allPathsProbe(ix, f, opts), "drop-recency probe")
}

// TestAccessPathDropsRelevanceWhenForecastEmpty is the empty-forecast witness: with no usable
// inverted-index match the cost model drops the dead relevance path and falls back to the
// recency + durable Seq/Index scans — and still returns the same candidate set as the union
// (the relevance path contributed nothing to the union either).
func TestAccessPathDropsRelevanceWhenForecastEmpty(t *testing.T) {
	ctx := context.Background()
	st := goodPlusNoiseStore(100)
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)
	opts := ProbeOptions{RecencyWindow: 16}
	f := Forecast{Intents: nil} // no prediction — the inverted index is unusable

	res := ix.ProbePlan(f, opts)
	chosen := chosenSet(res)

	if chosen[PathRelevance] {
		t.Errorf("an empty forecast has no inverted-index match; the relevance path must be dropped; chosen=%v", res.Chosen)
	}
	if !chosen[PathRecency] {
		t.Errorf("the recency Seq Scan is the fallback when no index is usable; chosen=%v", res.Chosen)
	}
	if est := estimateFor(res, PathRelevance); est.Cardinality != 0 {
		t.Errorf("the relevance path should estimate cardinality 0 for an empty forecast, got %d", est.Cardinality)
	}

	assertSameSpanSet(t, res.Spans, allPathsProbe(ix, f, opts), "empty-forecast probe")
}

// TestAccessPathKeepsAllPathsWhenNeeded is the multi-path witness: a selective forecast over a
// store with irrelevant noise INSIDE the recency window cannot prove the recency tail redundant
// (the windowed noise is reachable only by the recency path), so the cost model keeps EVERY
// path — and the result equals the full union (here, trivially, since it IS the full union).
func TestAccessPathKeepsAllPathsWhenNeeded(t *testing.T) {
	ctx := context.Background()
	st := goodPlusNoiseStore(300) // the tail mixes relevant + irrelevant spans
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)
	opts := ProbeOptions{RecencyWindow: 32} // window holds irrelevant noise the relevance path misses
	f := Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"span:0", "span:1"}}

	res := ix.ProbePlan(f, opts)
	chosen := chosenSet(res)

	for _, p := range []AccessPath{PathPin, PathRelevance, PathRecency, PathDurable} {
		if !chosen[p] {
			t.Errorf("when the recency tail holds spans no other path reaches, every path must be kept; missing %v, chosen=%v", p.pathName(), res.Chosen)
		}
	}
	assertSameSpanSet(t, res.Spans, allPathsProbe(ix, f, opts), "all-paths probe")
}

// TestAccessPathSelectedPlanEqualsUnionPlan is the END-TO-END correctness witness: for
// representative queries the cost-SELECTED probe, run through the full budgeted planner, yields
// the IDENTICAL Selected set as the full-UNION probe through the same planner. Path selection
// optimizes WHICH paths are scanned, never WHICH rows qualify — proven on the actual plan, not
// just the probe.
func TestAccessPathSelectedPlanEqualsUnionPlan(t *testing.T) {
	ctx := context.Background()
	budget := Budget{Tokens: 64}

	cases := []struct {
		name  string
		store *MemStore
		opts  ProbeOptions
		f     Forecast
	}{
		{
			name:  "selective/recency-redundant",
			store: recencyAllRelevantStore(200, 8),
			opts:  ProbeOptions{RecencyWindow: 8},
			f:     Forecast{Intents: []string{"auth token note service"}, Horizon: 3},
		},
		{
			name:  "empty-forecast",
			store: goodPlusNoiseStore(120),
			opts:  ProbeOptions{RecencyWindow: 16},
			f:     Forecast{Intents: nil, Pins: []string{"span:0"}, Horizon: 2},
		},
		{
			name:  "multi-path-needed",
			store: goodPlusNoiseStore(300),
			opts:  ProbeOptions{RecencyWindow: 32},
			f:     Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"span:0", "span:1"}, Horizon: 4},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spans, _ := tc.store.Spans(ctx)
			ix := BuildIndex(spans)

			// Full-union plan: probe every path, then plan under budget.
			union := PlanCells(ix.Probe(tc.f, tc.opts), tc.f, budget, nil)

			// Cost-selected plan: probe only the chosen paths, then the SAME planner.
			res := ix.ProbePlan(tc.f, tc.opts)
			sel := PlanCells(res.Spans, tc.f, budget, nil)

			u, s := selectedIDs(union), selectedIDs(sel)
			if len(u) != len(s) {
				t.Fatalf("%s: cost-selected plan kept %d spans, union plan kept %d — selection changed the result", tc.name, len(s), len(u))
			}
			for id := range u {
				if !s[id] {
					t.Errorf("%s: union selected %s but the cost-selected plan dropped it — path pruning changed which rows qualify", tc.name, id)
				}
			}
			for id := range s {
				if !u[id] {
					t.Errorf("%s: cost-selected plan added %s the union never had — impossible if pruning only removes paths", tc.name, id)
				}
			}
		})
	}
}

// TestAccessPathProbeIsDeterministic pins replay-stability of the cost-based probe: two builds
// and ProbePlan calls over the same spans and forecast yield the same chosen paths and the same
// candidate slice — no map-iteration or RNG dependence in the cost model or the executor.
func TestAccessPathProbeIsDeterministic(t *testing.T) {
	ctx := context.Background()
	st := recencyAllRelevantStore(150, 6)
	spans, _ := st.Spans(ctx)
	opts := ProbeOptions{RecencyWindow: 6}
	f := Forecast{Intents: []string{"auth token note service"}, Pins: []string{"span:0"}}

	a := BuildIndex(spans).ProbePlan(f, opts)
	b := BuildIndex(spans).ProbePlan(f, opts)

	if len(a.Chosen) != len(b.Chosen) {
		t.Fatalf("non-deterministic chosen-path count: %d vs %d", len(a.Chosen), len(b.Chosen))
	}
	for i := range a.Chosen {
		if a.Chosen[i] != b.Chosen[i] {
			t.Fatalf("non-deterministic chosen path at %d: %v vs %v", i, a.Chosen[i], b.Chosen[i])
		}
	}
	if len(a.Spans) != len(b.Spans) {
		t.Fatalf("non-deterministic probe size: %d vs %d", len(a.Spans), len(b.Spans))
	}
	for i := range a.Spans {
		if a.Spans[i].ID != b.Spans[i].ID {
			t.Fatalf("non-deterministic probe order at %d: %s vs %s", i, a.Spans[i].ID, b.Spans[i].ID)
		}
	}
}

// TestProbePathsAllEqualsProbe is the no-drift witness: the access-path executor with every
// path enabled is byte-identical to Index.Probe (the unconditional union). This is what lets
// the cost-based probe SHARE the union's correctness — a cost-selected probe is this executor
// with redundant paths removed, so any equivalence it preserves rests on this identity.
func TestProbePathsAllEqualsProbe(t *testing.T) {
	ctx := context.Background()
	st := goodPlusNoiseStore(250)
	spans, _ := st.Spans(ctx)
	ix := BuildIndex(spans)
	opts := ProbeOptions{RecencyWindow: 24}
	f := Forecast{Intents: []string{"auth token rotation"}, Pins: []string{"span:0", "span:1"}}

	want := ix.Probe(f, opts)
	got := ix.probePaths(f, opts.orDefaults(), map[AccessPath]bool{
		PathPin: true, PathRelevance: true, PathRecency: true, PathDurable: true,
	})
	if len(want) != len(got) {
		t.Fatalf("probePaths(all) size %d != Probe size %d", len(got), len(want))
	}
	for i := range want {
		if want[i].ID != got[i].ID {
			t.Fatalf("probePaths(all) diverged from Probe at %d: %s vs %s", i, got[i].ID, want[i].ID)
		}
	}
}

// assertSameSpanSet fails unless two probe slices carry the identical id set.
func assertSameSpanSet(t *testing.T, got, want []Span, label string) {
	t.Helper()
	g, w := probeIDset(got), probeIDset(want)
	if len(g) != len(w) {
		t.Fatalf("%s: cost-selected probe has %d spans, union has %d — sets differ", label, len(g), len(w))
	}
	for id := range w {
		if !g[id] {
			t.Errorf("%s: union probed %s but the cost-selected probe dropped it — a qualifying row was lost", label, id)
		}
	}
	for id := range g {
		if !w[id] {
			t.Errorf("%s: cost-selected probe has %s the union never had — impossible", label, id)
		}
	}
}
