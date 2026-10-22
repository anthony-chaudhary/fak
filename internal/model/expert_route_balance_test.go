package model

import (
	"errors"
	"strings"
	"testing"
)

func readRouteTrace(t *testing.T, body string) ExpertRouteTrace {
	t.Helper()
	tr, err := ReadExpertRouteTrace(strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	return tr
}

// fak-test:runtime fast est=5ms
func TestExpertRouteRankBalanceHeldOutKnownAnswers(t *testing.T) {
	tr := readRouteTrace(t, `{"schema":"fak.expert-route.trace/v1","layers":1,"experts":4,"top_k":2}
{"layer":0,"token":0,"phase":"decode","experts":[0,1]}
{"layer":0,"token":1,"phase":"decode","experts":[0,1]}
{"layer":0,"token":2,"phase":"decode","experts":[0,1]}
{"layer":0,"token":3,"phase":"decode","experts":[2,3]}
`)
	rep, err := AnalyzeExpertRouteTrace(tr, ExpertRouteAnalyzeOptions{Ranks: []int{2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.RankBalance) != 1 {
		t.Fatalf("rank balance rows = %d, want 1", len(rep.RankBalance))
	}
	b := rep.RankBalance[0]
	if b.Ranks != 2 || b.FitSteps != 2 || b.LoadBanded.Steps != 2 || b.CountEvenAll.Steps != 4 {
		t.Fatalf("steps = %+v", b)
	}
	for name, got := range map[string][2]float64{
		"count_even":            {b.CountEven.MoeLB, 2},
		"load_banded":           {b.LoadBanded.MoeLB, 1.5},
		"load_banded_p99":       {b.LoadBanded.P99StepRatio, 2},
		"load_banded_mean":      {b.LoadBanded.MeanStepRatio, 1.5},
		"floor":                 {b.Floor.MoeLB, 1},
		"count_even_all":        {b.CountEvenAll.MoeLB, 2},
		"load_banded_in_sample": {b.LoadBandedInSample.MoeLB, 1.25},
	} {
		if !routeNear(got[0], got[1]) {
			t.Errorf("%s = %v, want %v", name, got[0], got[1])
		}
	}
	if none, err := AnalyzeExpertRouteTrace(tr, ExpertRouteAnalyzeOptions{}); err != nil || none.RankBalance != nil {
		t.Fatalf("no ranks requested: err=%v rows=%v", err, none.RankBalance)
	}
	if _, err := AnalyzeExpertRouteTrace(tr, ExpertRouteAnalyzeOptions{Ranks: []int{5}}); !errors.Is(err, ErrExpertRouteTrace) {
		t.Fatalf("ranks > experts err = %v, want ErrExpertRouteTrace", err)
	}
}

// fak-test:runtime fast est=5ms
func TestAnalyzeExpertRouteStreamsSharedReads(t *testing.T) {
	a := readRouteTrace(t, `{"layer":0,"token":0,"phase":"decode","experts":[0,1]}
{"layer":0,"token":1,"phase":"decode","experts":[2,3]}
{"layer":0,"token":2,"phase":"decode","experts":[1,2]}
`)
	b := readRouteTrace(t, `{"schema":"fak.expert-route.trace/v1","layers":1,"experts":4,"top_k":2}
{"layer":0,"token":5,"phase":"prefill","experts":[3,1]}
{"layer":0,"token":6,"phase":"decode","experts":[0,2]}
{"layer":0,"token":7,"phase":"decode","experts":[2,3]}
`)
	rep, err := AnalyzeExpertRouteStreams([]ExpertRouteTrace{a, b}, ExpertRouteAnalyzeOptions{Phase: "decode", Ranks: []int{2}})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Schema != ExpertRouteOverlapSchema || rep.Streams != 2 || rep.Layers != 1 || rep.Steps != 2 || rep.Picks != 8 || rep.UniqueReads != 5 {
		t.Fatalf("overlap = %+v", rep)
	}
	if !routeNear(rep.SharedFraction, 0.375) || !routeNear(rep.MeanUnion, 2.5) {
		t.Fatalf("shared=%v union=%v, want 0.375 and 2.5", rep.SharedFraction, rep.MeanUnion)
	}
	if len(rep.RankBalance) != 1 || !routeNear(rep.RankBalance[0].CountEven.MoeLB, 2) || rep.RankBalance[0].CountEven.Steps != 1 {
		t.Fatalf("merged balance = %+v", rep.RankBalance)
	}
	if _, err := AnalyzeExpertRouteStreams([]ExpertRouteTrace{a}, ExpertRouteAnalyzeOptions{}); !errors.Is(err, ErrExpertRouteTrace) {
		t.Fatalf("single stream err = %v, want ErrExpertRouteTrace", err)
	}
}
