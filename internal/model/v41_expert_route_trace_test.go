package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV41ForwardEmitsExpertRoutePerTokenLayer(t *testing.T) {
	m := v41IncrementalExpertFixture(t, false, false)
	var events []ExpertRouteEvent
	legacy := 0
	m.SetExpertRouteEventObserver(func(ev ExpertRouteEvent) { events = append(events, ev) })
	m.SetExpertRouteObserver(func(int, []int) { legacy++ })
	s, err := m.NewBackendSessionChecked(compute.Default())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)

	layers, topK := m.Cfg.NumLayers, m.Cfg.NumExpertsPerTok
	type key struct{ token, layer int }
	check := func(evs []ExpertRouteEvent, phase string, tokens []int) {
		t.Helper()
		if want := len(tokens) * layers; len(evs) != want {
			t.Fatalf("%s events = %d, want %d (tokens x layers)", phase, len(evs), want)
		}
		seen := map[key]bool{}
		picks := 0
		for _, ev := range evs {
			if ev.Phase != phase {
				t.Fatalf("event phase = %q, want %q", ev.Phase, phase)
			}
			if len(ev.Experts) != topK {
				t.Fatalf("event experts = %d, want top-k %d", len(ev.Experts), topK)
			}
			for _, e := range ev.Experts {
				if e < 0 || e >= m.Cfg.NumExperts {
					t.Fatalf("expert id %d outside [0,%d)", e, m.Cfg.NumExperts)
				}
			}
			seen[key{ev.Token, ev.Layer}] = true
			picks += len(ev.Experts)
		}
		for _, tok := range tokens {
			for l := 0; l < layers; l++ {
				if !seen[key{tok, l}] {
					t.Fatalf("%s missing route for token %d layer %d", phase, tok, l)
				}
			}
		}
		if picks != len(tokens)*layers*topK {
			t.Fatalf("%s picks = %d, want %d", phase, picks, len(tokens)*layers*topK)
		}
	}

	s.Prefill([]int{1, 2, 3})
	check(events, "prefill", []int{0, 1, 2})
	if !s.v41IncrementalEligible() {
		t.Fatal("prefill did not seed an incremental decode state")
	}
	n := len(events)
	s.Step(4)
	check(events[n:], "decode", []int{3})
	if legacy != len(events) {
		t.Fatalf("legacy observer saw %d routes, want %d", legacy, len(events))
	}
}

func TestV41FaultLatencyBucketsLog2(t *testing.T) {
	cases := map[int64]int{0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 1023: 9, 1024: 10, 1 << 62: v41FaultLatencyBuckets - 1}
	for nanos, want := range cases {
		if got := v41FaultLatencyBucket(nanos); got != want {
			t.Fatalf("bucket(%d) = %d, want %d", nanos, got, want)
		}
	}
	m := &Model{}
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	for _, ns := range []int64{5, 6, 7, 1000, -3} {
		m.v41NoteExpertFaultDoorNanos(ns)
	}
	d := m.V41ExpertFaultAttribution().Decode
	if d.FaultNanosLog2Buckets[2] != 3 || d.FaultNanosLog2Buckets[9] != 1 || d.FaultDoorNanos != 1018 {
		t.Fatalf("decode buckets[2]=%d buckets[9]=%d sum=%d, want 3, 1, 1018",
			d.FaultNanosLog2Buckets[2], d.FaultNanosLog2Buckets[9], d.FaultDoorNanos)
	}
	if p := m.V41ExpertFaultAttribution().Prefill; p.FaultNanosLog2Buckets != ([v41FaultLatencyBuckets]int{}) {
		t.Fatal("decode fault latency leaked into the prefill histogram")
	}
}
