package inferencetopology

import (
	"context"
	"errors"
	"math"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

func readRuntimeSearchFixture(t *testing.T) RuntimeSearchInput {
	t.Helper()
	f, err := os.Open("testdata/runtime_search.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	in, err := DecodeRuntimeSearch(f)
	if err != nil {
		t.Fatal(err)
	}
	return in
}

// fak-test:runtime fast est=100ms lane=default
func TestTopologySearchDerivesConcurrency(t *testing.T) {
	now := time.Date(2026, 11, 2, 5, 0, 30, 0, time.UTC)
	run := func(in RuntimeSearchInput) RuntimeSearchReport {
		t.Helper()
		r, e := SearchRuntimeTopology(context.Background(), in, now)
		if e != nil {
			t.Fatal(e)
		}
		return r
	}
	chosen := func(r RuntimeSearchReport, id string) {
		t.Helper()
		if r.Chosen == nil || r.Chosen.Config.TopologyID != id || r.Chosen.Reason != SearchEligible || r.Status != "feasible" {
			t.Fatalf("chosen=%v status=%s, want %s", r.Chosen, r.Status, id)
		}
	}
	in := readRuntimeSearchFixture(t)
	chosen(run(in), "colocated")
	in.Workload.Concurrency = 4
	baseline := run(in)
	chosen(baseline, "pd-1p2d")
	if baseline.Chosen.TTFTSeconds != .2 || baseline.Chosen.TPOTSeconds != .02 || baseline.Chosen.UsefulTokensPerSecond != 140 || baseline.Chosen.TransferCriticalPathSeconds != .01 {
		t.Fatal("separate observed metrics lost")
	}
	if baseline.Evidence != "SUPPLIED_OBSERVATIONS_NOT_HARDWARE_QUALIFICATION" {
		t.Fatal("fixture promoted to hardware evidence")
	}
	slices.Reverse(in.Options)
	slices.Reverse(in.Observations)
	for i := range in.Options {
		o := &in.Options[i]
		slices.Reverse(o.Batches)
		slices.Reverse(o.Slots)
		slices.Reverse(o.Topology.Configured.Groups)
		slices.Reverse(o.Topology.Configured.Ranks)
		slices.Reverse(o.Topology.Observed.Devices)
		slices.Reverse(o.Topology.Observed.Links)
	}
	if got := run(in); !reflect.DeepEqual(got, baseline) {
		t.Fatal("discovery order changed receipt")
	}
	for _, tc := range []struct {
		name   string
		change func(*RuntimeSearchInput)
		reason RuntimeSearchReason
	}{
		{"bottleneck_rank", func(x *RuntimeSearchInput) { x.Options[1].Topology.Observed.Devices[2].FreeBytes = 70 }, SearchMemoryInfeasible},
		{"reservation", func(x *RuntimeSearchInput) { x.Options[1].Reservations["r2"] = 80 }, SearchMemoryInfeasible},
		{"unknown_reservation", func(x *RuntimeSearchInput) { delete(x.Options[1].Reservations, "r2") }, SearchMemoryUnknown},
		{"withdrawn", func(x *RuntimeSearchInput) { x.Options[1].Topology.Available[2].State = "unavailable" }, SearchTopologyUnavailable},
		{"slower_link", func(x *RuntimeSearchInput) { x.Options[1].Topology.Observed.Links[0].BytesPerSecond /= 2 }, SearchIdentityMismatch},
		{"epoch", func(x *RuntimeSearchInput) { x.Options[1].Topology.Observed.Epoch++ }, SearchIdentityMismatch},
		{"stale_curve", func(x *RuntimeSearchInput) { x.Observations[3].ExpiresAt = "2026-11-02T05:00:29Z" }, SearchObservationStale},
		{"stale_snapshot", func(x *RuntimeSearchInput) { x.Options[1].Topology.Observed.TTLSeconds = 20 }, SearchTopologyUnavailable},
		{"kernel", func(x *RuntimeSearchInput) { x.Observations[3].Kernel = "other" }, SearchIdentityMismatch},
		{"nonfinite_curve", func(x *RuntimeSearchInput) { x.Observations[3].UsefulTokensPerSecond = math.Inf(1) }, SearchObservationUnknown},
		{"nonfinite_calibration", func(x *RuntimeSearchInput) { x.Observations[3].Decode.StepSeconds = math.NaN() }, SearchObservationUnknown},
		{"duplicate_curve", func(x *RuntimeSearchInput) { x.Observations = append(x.Observations, x.Observations[3]) }, SearchObservationUnknown},
		{"slo", func(x *RuntimeSearchInput) { x.SLO.TPOTMS = 15 }, SearchSLOInfeasible},
	} {
		t.Run(tc.name, func(t *testing.T) {
			x := readRuntimeSearchFixture(t)
			x.Workload.Concurrency = 4
			tc.change(&x)
			r := run(x)
			chosen(r, "colocated")
			found := false
			for _, row := range r.Rows {
				if row.Config.TopologyID == "pd-1p2d" && row.Config.Batch == 4 && row.Config.Slots == 4 {
					found = true
					if row.Reason != tc.reason {
						t.Fatalf("reason=%s want=%s", row.Reason, tc.reason)
					}
				}
			}
			if !found {
				t.Fatal("candidate silently omitted")
			}
		})
	}
	t.Run("pins_incumbent_budget", func(t *testing.T) {
		x := readRuntimeSearchFixture(t)
		x.Workload.Concurrency = 4
		inc := RuntimeSearchConfig{TopologyID: "pd-1p2d", Batch: 4, Chunk: 128, Slots: 4}
		x.Incumbent = &inc
		x.MaxCandidates = 1
		r := run(x)
		chosen(r, "pd-1p2d")
		if r.Evaluated != 1 || r.StopReason != "candidate_budget" || r.Incumbent == nil || r.Incumbent.Config != inc {
			t.Fatal("incumbent not retained within finite budget")
		}
		x.MaxCandidates = 64
		x.Pins.TopologyID = "colocated"
		chosen(run(x), "colocated")
		x.Pins.TopologyID = "absent"
		r = run(x)
		if r.Chosen != nil || r.Status != "infeasible" {
			t.Fatal("hard pin silently relaxed")
		}
		x.Pins = RuntimeSearchConfig{}
		x.Workload.Concurrency = 2
		r = run(x)
		if r.Chosen != nil || r.Status != "unknown" {
			t.Fatal("unobserved workload extrapolated")
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		r, e := SearchRuntimeTopology(ctx, x, now)
		if e != nil || r.Evaluated != 0 || r.StopReason != "cancelled" || r.Chosen != nil {
			t.Fatal("cancel budget ignored")
		}
	})
	t.Run("strict_bounds", func(t *testing.T) {
		x := readRuntimeSearchFixture(t)
		x.MaxCandidates = 0
		if _, e := SearchRuntimeTopology(context.Background(), x, now); !errors.Is(e, ErrInvalid) {
			t.Fatal("missing candidate bound accepted")
		}
		x.MaxCandidates = 1
		x.MaxDurationMS = 0
		if _, e := SearchRuntimeTopology(context.Background(), x, now); !errors.Is(e, ErrInvalid) {
			t.Fatal("missing time bound accepted")
		}
		for _, raw := range []string{`{"unknown":1}`, `{} {}`, strings.Repeat(" ", 16<<20) + "{}"} {
			if _, e := DecodeRuntimeSearch(strings.NewReader(raw)); !errors.Is(e, ErrInvalid) {
				t.Fatal("invalid document accepted")
			}
		}
	})
}
