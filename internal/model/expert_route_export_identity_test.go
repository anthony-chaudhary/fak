package model

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s
func TestExpertRouteExportNormalizesPickIdentity(t *testing.T) {
	t.Parallel()
	trace := expertRouteIdentityRead(t, `{"layer":0,"token":0,"phase":"prefill","experts":[5,2,5,1,2]}
{"layer":1,"token":0,"phase":"prefill","experts":[2,2,5]}
{"layer":0,"token":1,"phase":"decode","experts":[1,1,9,5,9]}
`)
	const bytes = int64(4096)
	input := append([]int(nil), trace.Events[0].Experts...)
	report, exported := expertRouteIdentityAnalyzeExport(t, trace, ExpertRouteAnalyzeOptions{}, bytes)
	want := []StrixMoECacheEvent{
		{Layer: 0, Expert: 5, Bytes: bytes},
		{Layer: 0, Expert: 2, Bytes: bytes},
		{Layer: 0, Expert: 1, Bytes: bytes},
		{Layer: 1, Expert: 2, Bytes: bytes},
		{Layer: 1, Expert: 5, Bytes: bytes},
		{Layer: 0, Expert: 1, Bytes: bytes},
		{Layer: 0, Expert: 9, Bytes: bytes},
		{Layer: 0, Expert: 5, Bytes: bytes},
	}
	if !reflect.DeepEqual(exported.Events, want) {
		t.Fatalf("exported access order = %v, want %v", exported.Events, want)
	}
	if report.Routes != 3 || report.Picks != 8 || report.Layers != 2 || report.UniqueExperts != 6 || report.ColdPicks != 6 {
		t.Fatalf("report routes/picks/layers/unique/cold = %d/%d/%d/%d/%d, want 3/8/2/6/6",
			report.Routes, report.Picks, report.Layers, report.UniqueExperts, report.ColdPicks)
	}
	if report.TopK != 3 || exported.TopK != 3 {
		t.Fatalf("inferred top-k report/export = %d/%d, want 3/3", report.TopK, exported.TopK)
	}
	if exported.Schema != StrixMoECacheTraceSchema || exported.Layers != 2 || exported.Experts != 10 || exported.ExpertBytes != bytes {
		t.Fatalf("export schema/layers/experts/bytes = %s/%d/%d/%d, want %s/2/10/%d",
			exported.Schema, exported.Layers, exported.Experts, exported.ExpertBytes, StrixMoECacheTraceSchema, bytes)
	}
	if !reflect.DeepEqual(trace.Events[0].Experts, input) {
		t.Fatalf("input experts changed = %v", trace.Events[0].Experts)
	}
}

// fak-test:runtime fast est=1s
func TestExpertRouteExportPreservesUnknownTokenRepeats(t *testing.T) {
	t.Parallel()
	trace := expertRouteIdentityRead(t, `{"layer":0,"token":-1,"phase":"decode","experts":[3,1,3]}
{"layer":0,"token":-1,"phase":"decode","experts":[3,1,3]}
{"layer":0,"token":7,"phase":"decode","experts":[3,3]}
{"layer":0,"token":7,"phase":"decode","experts":[1,1]}
{"layer":1,"token":7,"phase":"decode","experts":[3,3]}
`)
	const bytes = int64(512)
	for _, tc := range []struct {
		name    string
		keep    bool
		routes  int
		dropped int
		want    []StrixMoECacheEvent
	}{
		{
			name: "default", routes: 4, dropped: 1,
			want: []StrixMoECacheEvent{
				{Layer: 0, Expert: 3, Bytes: bytes}, {Layer: 0, Expert: 1, Bytes: bytes},
				{Layer: 0, Expert: 3, Bytes: bytes}, {Layer: 0, Expert: 1, Bytes: bytes},
				{Layer: 0, Expert: 3, Bytes: bytes}, {Layer: 1, Expert: 3, Bytes: bytes},
			},
		},
		{
			name: "keep-repeat", keep: true, routes: 5,
			want: []StrixMoECacheEvent{
				{Layer: 0, Expert: 3, Bytes: bytes}, {Layer: 0, Expert: 1, Bytes: bytes},
				{Layer: 0, Expert: 3, Bytes: bytes}, {Layer: 0, Expert: 1, Bytes: bytes},
				{Layer: 0, Expert: 3, Bytes: bytes}, {Layer: 0, Expert: 1, Bytes: bytes},
				{Layer: 1, Expert: 3, Bytes: bytes},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report, exported := expertRouteIdentityAnalyzeExport(t, trace, ExpertRouteAnalyzeOptions{KeepRepeat: tc.keep}, bytes)
			if report.Routes != tc.routes || report.RepeatsDropped != tc.dropped || report.Picks != len(tc.want) {
				t.Fatalf("routes/dropped/picks = %d/%d/%d, want %d/%d/%d",
					report.Routes, report.RepeatsDropped, report.Picks, tc.routes, tc.dropped, len(tc.want))
			}
			if !reflect.DeepEqual(exported.Events, tc.want) {
				t.Fatalf("exported events = %v, want %v", exported.Events, tc.want)
			}
			if report.UniqueExperts != 3 || report.ColdPicks != 3 {
				t.Fatalf("unique/cold = %d/%d, want 3/3 for distinct layer identities", report.UniqueExperts, report.ColdPicks)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestExpertRouteExportPhaseSelectionAgreesWithAnalysis(t *testing.T) {
	t.Parallel()
	trace := expertRouteIdentityRead(t, `{"layer":0,"token":0,"phase":"prefill","experts":[7,2,7]}
{"layer":0,"token":0,"phase":"decode","experts":[9,9]}
{"layer":0,"token":1,"phase":"decode","experts":[2,6,2]}
{"layer":1,"token":0,"phase":"decode","experts":[7,7]}
{"layer":0,"token":-1,"phase":"decode","experts":[6,6]}
{"layer":0,"token":-1,"phase":"decode","experts":[6,6]}
`)
	const bytes = int64(64)
	want := []StrixMoECacheEvent{
		{Layer: 0, Expert: 2, Bytes: bytes}, {Layer: 0, Expert: 6, Bytes: bytes},
		{Layer: 1, Expert: 7, Bytes: bytes}, {Layer: 0, Expert: 6, Bytes: bytes},
		{Layer: 0, Expert: 6, Bytes: bytes},
	}
	for _, keep := range []bool{false, true} {
		report, exported := expertRouteIdentityAnalyzeExport(t, trace, ExpertRouteAnalyzeOptions{Phase: "decode", KeepRepeat: keep}, bytes)
		wantEvents, wantRoutes, wantDropped := want, 4, 1
		if keep {
			wantEvents = append([]StrixMoECacheEvent{{Layer: 0, Expert: 9, Bytes: bytes}}, want...)
			wantRoutes, wantDropped = 5, 0
		}
		if report.Routes != wantRoutes || report.RepeatsDropped != wantDropped || report.PhaseRoutes["decode"] != wantRoutes || len(report.PhaseRoutes) != 1 {
			t.Fatalf("keep=%v routes/dropped/decode/phases = %d/%d/%d/%d, want %d/%d/%d/1",
				keep, report.Routes, report.RepeatsDropped, report.PhaseRoutes["decode"], len(report.PhaseRoutes), wantRoutes, wantDropped, wantRoutes)
		}
		if !reflect.DeepEqual(exported.Events, wantEvents) {
			t.Fatalf("keep=%v exported events = %v, want %v", keep, exported.Events, wantEvents)
		}
	}
}

// fak-test:runtime fast est=1s
func TestExpertRouteExportReplayFixtures(t *testing.T) {
	t.Parallel()
	const bytes = int64(64)
	for _, tc := range []struct {
		name       string
		experts    []int
		phaseShift int
		lruHits    int
		oracleHits int
		cold       int
		reuse      []ExpertRouteReuseBucket
	}{
		{name: "reuse", experts: []int{0, 1, 0, 1, 0, 1}, lruHits: 4, oracleHits: 4, cold: 2, reuse: []ExpertRouteReuseBucket{{Min: 2, Max: 3, Count: 4}}},
		{name: "scan", experts: []int{0, 1, 2, 3, 4, 5}, cold: 6},
		{name: "phase-shift", experts: []int{0, 1, 0, 1, 2, 3, 2, 3}, phaseShift: 4, lruHits: 4, oracleHits: 4, cold: 4, reuse: []ExpertRouteReuseBucket{{Min: 2, Max: 3, Count: 4}}},
		{name: "held-out-irregular", experts: []int{3, 0, 2, 3, 1, 0, 2, 1, 2, 0}, lruHits: 1, oracleHits: 3, cold: 4, reuse: []ExpertRouteReuseBucket{{Min: 2, Max: 3, Count: 3}, {Min: 4, Max: 7, Count: 3}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			trace := ExpertRouteTrace{}
			for token, expert := range tc.experts {
				phase := "prefill"
				if tc.phaseShift > 0 && token >= tc.phaseShift {
					phase = "decode"
				}
				trace.Events = append(trace.Events, ExpertRouteEvent{Layer: 0, Token: token, Phase: phase, Experts: []int{expert, expert}})
			}
			report, exported := expertRouteIdentityAnalyzeExport(t, trace, ExpertRouteAnalyzeOptions{Capacities: []int{2}}, bytes)
			if report.Picks != len(tc.experts) || report.TopK != 1 || exported.TopK != 1 || report.ColdPicks != tc.cold {
				t.Fatalf("picks/top-k/export-top-k/cold = %d/%d/%d/%d, want %d/1/1/%d",
					report.Picks, report.TopK, exported.TopK, report.ColdPicks, len(tc.experts), tc.cold)
			}
			if len(report.Reuse) != len(tc.reuse) || !reflect.DeepEqual(report.Reuse, tc.reuse) {
				t.Fatalf("reuse buckets = %v, want %v", report.Reuse, tc.reuse)
			}
			for i, event := range exported.Events {
				if event != (StrixMoECacheEvent{Layer: 0, Expert: tc.experts[i], Bytes: bytes}) {
					t.Fatalf("exported event %d = %v, want expert %d in layer 0", i, event, tc.experts[i])
				}
			}
			replayTrace := ExpertAccessTrace{
				Schema: ExpertReplayTraceSchema, Name: tc.name, Source: "fixed-route-export-contract", BudgetBytes: 2 * bytes,
			}
			for _, event := range exported.Events {
				replayTrace.Events = append(replayTrace.Events, ExpertAccessTraceEvent{Layer: event.Layer, Expert: event.Expert, WeightBytes: event.Bytes})
			}
			replayed, err := ReplayExpertAccessTrace(replayTrace)
			if err != nil {
				t.Fatal(err)
			}
			if !replayed.Oracle.Exact || replayed.PagedRingLRU.HitTokens != tc.lruHits*int(bytes) || replayed.Oracle.HitTokens != tc.oracleHits*int(bytes) {
				t.Fatalf("replay exact/LRU/oracle = %v/%d/%d, want true/%d/%d",
					replayed.Oracle.Exact, replayed.PagedRingLRU.HitTokens, replayed.Oracle.HitTokens, tc.lruHits*int(bytes), tc.oracleHits*int(bytes))
			}
			cache := report.Cache[0]
			if cache.CapacityPerLayer != 2 || cache.TotalExperts != 2 || math.Abs(cache.LRUHitRate-float64(tc.lruHits)/float64(report.Picks)) > 1e-9 || math.Abs(cache.BeladyHitRate-float64(tc.oracleHits)/float64(report.Picks)) > 1e-9 {
				t.Fatalf("analyzer capacity/total/LRU/oracle = %d/%d/%g/%g, want 2/2/%g/%g",
					cache.CapacityPerLayer, cache.TotalExperts, cache.LRUHitRate, cache.BeladyHitRate,
					float64(tc.lruHits)/float64(report.Picks), float64(tc.oracleHits)/float64(report.Picks))
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestExpertRouteTraceTruncatedFinalRecord(t *testing.T) {
	t.Parallel()
	const prefix = "{\"layer\":0,\"token\":0,\"phase\":\"decode\",\"experts\":[3]}\n"
	const truncated = `{"layer":0,"token":1,"phase":"decode","experts":[4`
	for _, ending := range []string{"", "\n"} {
		if _, err := ReadExpertRouteTrace(strings.NewReader(prefix + truncated + ending)); !errors.Is(err, ErrExpertRouteTrace) {
			t.Fatalf("truncated record error = %v, want ErrExpertRouteTrace", err)
		}
	}
	// A prefix ending at a complete JSONL record is a valid shorter trace. The schema
	// has no terminal marker, so the reader cannot infer intended capture completeness.
	trace := expertRouteIdentityRead(t, prefix)
	if len(trace.Events) != 1 || trace.Events[0].Token != 0 || !reflect.DeepEqual(trace.Events[0].Experts, []int{3}) {
		t.Fatalf("whole-record prefix events = %v, want the one complete route", trace.Events)
	}
}

func expertRouteIdentityRead(t *testing.T, raw string) ExpertRouteTrace {
	t.Helper()
	trace, err := ReadExpertRouteTrace(strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	return trace
}

func expertRouteIdentityAnalyzeExport(t *testing.T, trace ExpertRouteTrace, opts ExpertRouteAnalyzeOptions, bytes int64) (ExpertRouteAnalysis, StrixMoECacheTrace) {
	t.Helper()
	report, err := AnalyzeExpertRouteTrace(trace, opts)
	if err != nil {
		t.Fatal(err)
	}
	exported, err := ExpertRouteTraceToStrix(trace, opts, bytes)
	if err != nil {
		t.Fatal(err)
	}
	if report.Picks != len(exported.Events) {
		t.Fatalf("analysis picks/export events = %d/%d, want matching normalized expert touches", report.Picks, len(exported.Events))
	}
	return report, exported
}
