package model

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

const syntheticRouteTrace = `{"schema":"fak.expert-route.trace/v1","layers":2,"experts":8,"top_k":2}
{"layer":0,"token":0,"phase":"prefill","experts":[0,1]}
{"layer":0,"token":1,"phase":"prefill","experts":[0,2]}
{"layer":0,"token":2,"phase":"prefill","experts":[0,1]}
{"layer":1,"token":0,"phase":"prefill","experts":[5,6]}
{"layer":1,"token":1,"phase":"prefill","experts":[7,5]}
{"layer":1,"token":2,"phase":"prefill","experts":[6,7]}
{"layer":0,"token":1,"phase":"decode","experts":[4,4]}
{"layer":0,"token":3,"phase":"decode","experts":[0,3]}
{"layer":1,"token":3,"phase":"decode","experts":[5,6]}
`

func syntheticRouteAnalysis(t *testing.T, opts ExpertRouteAnalyzeOptions) ExpertRouteAnalysis {
	t.Helper()
	tr, err := ReadExpertRouteTrace(strings.NewReader(syntheticRouteTrace))
	if err != nil {
		t.Fatal(err)
	}
	rep, err := AnalyzeExpertRouteTrace(tr, opts)
	if err != nil {
		t.Fatal(err)
	}
	return rep
}

func routeNear(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestAnalyzeExpertRouteTraceKnownAnswers(t *testing.T) {
	rep := syntheticRouteAnalysis(t, ExpertRouteAnalyzeOptions{
		Windows: []int{1, 2, 8}, TopN: []int{1, 2}, Capacities: []int{1, 2, 4},
	})
	if rep.Routes != 8 || rep.RepeatsDropped != 1 || rep.Picks != 16 || rep.Layers != 2 || rep.TopK != 2 {
		t.Fatalf("routes=%d dropped=%d picks=%d layers=%d topk=%d, want 8 1 16 2 2", rep.Routes, rep.RepeatsDropped, rep.Picks, rep.Layers, rep.TopK)
	}
	if !reflect.DeepEqual(rep.PhaseRoutes, map[string]int{"prefill": 6, "decode": 2}) {
		t.Fatalf("phase routes = %v", rep.PhaseRoutes)
	}
	if rep.UniqueExperts != 7 || !routeNear(rep.MeanUniquePerLayer, 3.5) || rep.MaxUniquePerLayer != 4 {
		t.Fatalf("unique=%d mean=%v max=%d, want 7 3.5 4", rep.UniqueExperts, rep.MeanUniquePerLayer, rep.MaxUniquePerLayer)
	}
	wantWin := []ExpertRouteWindowStat{{1, 2, 2}, {2, 3, 3}, {8, 3.5, 4}}
	if !reflect.DeepEqual(rep.Windows, wantWin) {
		t.Fatalf("windows = %+v, want %+v", rep.Windows, wantWin)
	}
	if !routeNear(rep.TopNShare[0].Share, 7.0/16) || !routeNear(rep.TopNShare[1].Share, 12.0/16) {
		t.Fatalf("top-n share = %+v, want 7/16, 12/16", rep.TopNShare)
	}
	wantReuse := []ExpertRouteReuseBucket{{1, 1, 6}, {2, 3, 3}}
	if rep.ColdPicks != 7 || !reflect.DeepEqual(rep.Reuse, wantReuse) {
		t.Fatalf("cold=%d reuse=%+v, want 7 %+v", rep.ColdPicks, rep.Reuse, wantReuse)
	}
	wantCache := []ExpertRouteCacheRow{
		{CapacityPerLayer: 1, TotalExperts: 2, LRUHitRate: 0, BeladyHitRate: 0},
		{CapacityPerLayer: 2, TotalExperts: 4, LRUHitRate: 3.0 / 16, BeladyHitRate: 6.0 / 16},
		{CapacityPerLayer: 4, TotalExperts: 8, LRUHitRate: 9.0 / 16, BeladyHitRate: 9.0 / 16},
	}
	if !reflect.DeepEqual(rep.Cache, wantCache) {
		t.Fatalf("cache = %+v, want %+v", rep.Cache, wantCache)
	}
	wantHot := []ExpertUsageCount{{0, 0, 4}, {0, 1, 2}, {0, 2, 1}, {1, 5, 3}, {1, 6, 3}, {1, 7, 2}}
	if rep.HotSetSize != 6 || !reflect.DeepEqual(rep.HotSet, wantHot) {
		t.Fatalf("hot set %d %+v, want 6 %+v", rep.HotSetSize, rep.HotSet, wantHot)
	}
}

func TestAnalyzeExpertRouteTracePhaseFilterAfterDedupe(t *testing.T) {
	rep := syntheticRouteAnalysis(t, ExpertRouteAnalyzeOptions{Phase: "decode"})
	if rep.Routes != 2 || rep.RepeatsDropped != 1 || rep.Picks != 4 {
		t.Fatalf("decode routes=%d dropped=%d picks=%d, want 2 1 4", rep.Routes, rep.RepeatsDropped, rep.Picks)
	}
	keep := syntheticRouteAnalysis(t, ExpertRouteAnalyzeOptions{KeepRepeat: true})
	if keep.Routes != 9 || keep.RepeatsDropped != 0 || keep.Picks != 17 {
		t.Fatalf("keep-repeat routes=%d dropped=%d picks=%d, want 9 0 17 (duplicate id folded)", keep.Routes, keep.RepeatsDropped, keep.Picks)
	}
}

func TestReadExpertRouteTraceRefusesMalformed(t *testing.T) {
	for name, in := range map[string]string{
		"schema": `{"schema":"other/v1"}` + "\n" + `{"layer":0,"token":0,"experts":[1]}`,
		"empty":  `{"schema":"fak.expert-route.trace/v1"}`,
		"json":   `{"layer":`,
		"neg":    `{"layer":0,"token":0,"experts":[-1]}`,
	} {
		if _, err := ReadExpertRouteTrace(strings.NewReader(in)); !errors.Is(err, ErrExpertRouteTrace) {
			t.Fatalf("%s: err = %v, want ErrExpertRouteTrace", name, err)
		}
	}
}

func TestExpertRouteTraceToStrix(t *testing.T) {
	tr, err := ReadExpertRouteTrace(strings.NewReader(syntheticRouteTrace))
	if err != nil {
		t.Fatal(err)
	}
	st, err := ExpertRouteTraceToStrix(tr, ExpertRouteAnalyzeOptions{}, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if st.Schema != StrixMoECacheTraceSchema || st.Layers != 2 || st.Experts != 8 || st.TopK != 2 || len(st.Events) != 16 {
		t.Fatalf("strix trace = %s layers=%d experts=%d topk=%d events=%d", st.Schema, st.Layers, st.Experts, st.TopK, len(st.Events))
	}
	if st.Events[0] != (StrixMoECacheEvent{Layer: 0, Expert: 0, Bytes: 4096}) {
		t.Fatalf("first event = %+v", st.Events[0])
	}
	if _, err := ExpertRouteTraceToStrix(tr, ExpertRouteAnalyzeOptions{}, 0); !errors.Is(err, ErrExpertRouteTrace) {
		t.Fatalf("zero bytes err = %v, want ErrExpertRouteTrace", err)
	}
}

func TestExpertRouteTraceSinkRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "route.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	m := &Model{Cfg: Config{NumLayers: 3, NumExperts: 16, NumExpertsPerTok: 2}}
	sink := &expertRouteTraceSink{f: f}
	sink.write(m, ExpertRouteEvent{Layer: 0, Token: 0, Phase: "prefill", Experts: []int{3, 9}})
	sink.write(m, ExpertRouteEvent{Layer: 2, Token: 0, Phase: "prefill", Experts: []int{1, 4}})
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	tr, err := ReadExpertRouteTrace(raw)
	if err != nil {
		t.Fatal(err)
	}
	want := ExpertRouteTraceHeader{Schema: ExpertRouteTraceSchema, Layers: 3, Experts: 16, TopK: 2}
	if tr.Header != want || len(tr.Events) != 2 || !reflect.DeepEqual(tr.Events[1].Experts, []int{1, 4}) {
		t.Fatalf("round trip = %+v", tr)
	}
}
