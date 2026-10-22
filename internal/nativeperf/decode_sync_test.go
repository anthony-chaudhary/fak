package nativeperf

import (
	"context"
	"math"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// syncFixture is a device-work reader whose cumulative counters a test moves.
type syncFixture struct {
	mu sync.Mutex
	w  compute.DeviceWork
}

func newSyncFixture(t *testing.T, backend string) *syncFixture {
	t.Helper()
	f := &syncFixture{w: compute.DeviceWork{SyncObserved: true}}
	t.Cleanup(compute.RegisterDeviceWorkReader(backend, func() compute.DeviceWork {
		f.mu.Lock()
		defer f.mu.Unlock()
		w := f.w
		w.SyncSites = append([]compute.DeviceSyncSite(nil), f.w.SyncSites...)
		return w
	}))
	return f
}

func (f *syncFixture) set(fn func(*compute.DeviceWork)) {
	f.mu.Lock()
	defer f.mu.Unlock()
	fn(&f.w)
}

func approx(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Abs(b)) }

// fak-test:runtime fast est=10ms lane=default
func TestDecodeWindowReportsGPUIdleAndSyncsBySite(t *testing.T) {
	f := newSyncFixture(t, "sync-fixture")
	f.set(func(w *compute.DeviceWork) {
		w.Dispatches, w.DeviceIdleNS = 100, 5_000_000
		w.SyncSites = []compute.DeviceSyncSite{{Site: compute.SyncSiteReadback, Stage: compute.SyncStageNone, Syncs: 7, WaitNS: 70, D2HBytes: 28}}
	})
	ctx, acc := WithDecodeHostWork(context.Background())
	for range 2 {
		span := StartDecodeSpan()
		time.Sleep(2 * time.Millisecond)
		f.set(func(w *compute.DeviceWork) {
			w.Dispatches += 50
			w.DeviceIdleNS += uint64(time.Millisecond)
			w.SyncSites = compute.DeviceSyncSitesAdd(w.SyncSites, []compute.DeviceSyncSite{
				{Site: compute.SyncSiteFlushStatus, Stage: "v41_exact_rmsnorm", Syncs: 8, WaitNS: uint64(400 * time.Microsecond)},
				{Site: compute.SyncSiteReadback, Stage: compute.SyncStageNone, Syncs: 4, WaitNS: uint64(100 * time.Microsecond), D2HBytes: 4096},
			})
		})
		w := span.Finish(ctx, 4)
		wantSites := []compute.DeviceSyncSite{
			{Site: compute.SyncSiteFlushStatus, Stage: "v41_exact_rmsnorm", Syncs: 8, WaitNS: uint64(400 * time.Microsecond)},
			{Site: compute.SyncSiteReadback, Stage: compute.SyncStageNone, Syncs: 4, WaitNS: uint64(100 * time.Microsecond), D2HBytes: 4096},
		}
		if w.Backend != "sync-fixture" || w.DeviceIdle == nil || *w.DeviceIdle != time.Millisecond || !w.SyncsObserved || !reflect.DeepEqual(w.SyncSites, wantSites) {
			t.Fatalf("window = %+v, want 1ms idle and %+v", w, wantSites)
		}
	}
	totals, ok := acc.Totals()
	if !ok || totals.Steps != 8 || totals.DeviceIdle == nil || *totals.DeviceIdle != 2*time.Millisecond || !totals.SyncsObserved {
		t.Fatalf("totals = %+v ok=%v, want 8 steps, 2ms idle, observed syncs", totals, ok)
	}
	if idle, ok := totals.GPUIdleMSPerToken(); !ok || !approx(idle, 0.25) {
		t.Fatalf("gpu idle ms/token = %v ok=%v, want 0.25", idle, ok)
	}
	if syncs, ok := totals.SyncsPerToken(); !ok || !approx(syncs, 3) {
		t.Fatalf("syncs/token = %v ok=%v, want 3", syncs, ok)
	}
	rates, ok := totals.SyncSiteRates()
	wantRates := []DecodeSyncSiteRate{
		{Site: compute.SyncSiteFlushStatus, Stage: "v41_exact_rmsnorm", SyncsPerToken: 2, WaitMSPerToken: 0.1},
		{Site: compute.SyncSiteReadback, Stage: compute.SyncStageNone, SyncsPerToken: 1, WaitMSPerToken: 0.025, D2HBytesPerToken: 1024},
	}
	if !ok || len(rates) != len(wantRates) {
		t.Fatalf("rates = %+v ok=%v, want %+v", rates, ok, wantRates)
	}
	for i, r := range rates {
		w := wantRates[i]
		if r.Site != w.Site || r.Stage != w.Stage || !approx(r.SyncsPerToken, w.SyncsPerToken) || !approx(r.WaitMSPerToken, w.WaitMSPerToken) || !approx(r.D2HBytesPerToken, w.D2HBytesPerToken) {
			t.Fatalf("rate[%d] = %+v, want %+v", i, r, w)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeWindowSyncsAbsentWhenUnobservedResetOrOverlapped(t *testing.T) {
	f := newSyncFixture(t, "sync-fixture")
	cell := compute.DeviceSyncSite{Site: compute.SyncSiteStepEnd, Stage: "v41_mhc", Syncs: 10, WaitNS: 10}
	f.set(func(w *compute.DeviceWork) { w.Dispatches, w.SyncSites = 1, []compute.DeviceSyncSite{cell} })
	window := func(mutate func(*compute.DeviceWork)) DecodeHostWork {
		span := StartDecodeSpan()
		time.Sleep(time.Millisecond)
		f.set(func(w *compute.DeviceWork) {
			w.Dispatches++
			mutate(w)
		})
		return span.Finish(context.Background(), 1)
	}
	cases := map[string]func(*compute.DeviceWork){
		"unobserved":     func(w *compute.DeviceWork) { w.SyncObserved = false },
		"idle_backwards": func(w *compute.DeviceWork) { w.DeviceIdleNS-- },
		"cell_backwards": func(w *compute.DeviceWork) {
			w.SyncSites = []compute.DeviceSyncSite{{Site: cell.Site, Stage: cell.Stage, Syncs: cell.Syncs - 1, WaitNS: cell.WaitNS}}
		},
	}
	for name, mutate := range cases {
		f.set(func(w *compute.DeviceWork) {
			w.SyncObserved, w.DeviceIdleNS, w.SyncSites = true, 1000, []compute.DeviceSyncSite{cell}
		})
		if w := window(mutate); w.DeviceIdle != nil || w.SyncSites != nil || w.SyncsObserved {
			t.Fatalf("%s: window = %+v, want idle and syncs absent", name, w)
		}
	}

	f.set(func(w *compute.DeviceWork) { w.SyncObserved, w.SyncSites = true, []compute.DeviceSyncSite{cell} })
	if w := window(func(w *compute.DeviceWork) { w.DeviceIdleNS += uint64(time.Hour) }); w.DeviceIdle != nil || !w.SyncsObserved || len(w.SyncSites) != 0 {
		t.Fatalf("idle beyond wall: window = %+v, want idle absent and an observed zero-sync delta", w)
	}

	pipelined := map[string]func(*compute.DeviceWork){
		"pipelined_cycle": func(w *compute.DeviceWork) {
			w.DeviceIdleNS += uint64(100 * time.Microsecond)
			w.PipelinedSubmits++
			w.PipelinedDrains++
			w.SyncSites = compute.DeviceSyncSitesAdd(w.SyncSites, []compute.DeviceSyncSite{{Site: compute.SyncSitePipelineDrain, Stage: "add", Syncs: 1, WaitNS: 5}})
		},
		"outstanding_throughout": func(w *compute.DeviceWork) {
			w.SyncSites = compute.DeviceSyncSitesAdd(w.SyncSites, []compute.DeviceSyncSite{{Site: compute.SyncSitePipelineDrain, Stage: "add", Syncs: 1, WaitNS: 5}})
		},
	}
	for name, mutate := range pipelined {
		f.set(func(w *compute.DeviceWork) {
			w.SyncObserved, w.DeviceIdleNS, w.SyncSites = true, 1000, []compute.DeviceSyncSite{cell}
			w.PipelinedSubmits, w.PipelinedDrains = 2, 2
			if name == "outstanding_throughout" {
				w.PipelinedSubmits = 3
			}
		})
		want := []compute.DeviceSyncSite{{Site: compute.SyncSitePipelineDrain, Stage: "add", Syncs: 1, WaitNS: 5}}
		if w := window(mutate); w.DeviceIdle != nil || !w.SyncsObserved || !reflect.DeepEqual(w.SyncSites, want) {
			t.Fatalf("%s: window = %+v, want idle absent and syncs %+v", name, w, want)
		}
	}
	f.set(func(w *compute.DeviceWork) { w.PipelinedSubmits, w.PipelinedDrains = 0, 0 })

	outer := StartDecodeSpan()
	inner := StartDecodeSpan()
	if w := inner.Finish(context.Background(), 1); !w.Overlapped || w.DeviceIdle != nil || w.SyncsObserved {
		t.Fatalf("overlapped window = %+v, want idle and syncs absent", w)
	}
	outer.Abandon()

	ctx, acc := WithDecodeHostWork(context.Background())
	span := StartDecodeSpan()
	time.Sleep(time.Millisecond)
	span.Finish(ctx, 2)
	f.set(func(w *compute.DeviceWork) { w.SyncObserved = false })
	span = StartDecodeSpan()
	span.Finish(ctx, 2)
	totals, _ := acc.Totals()
	if _, ok := totals.SyncsPerToken(); ok || totals.SyncSites != nil {
		t.Fatalf("totals = %+v, want syncs absent once any window lacked them", totals)
	}
	if _, ok := totals.SyncSiteRates(); ok {
		t.Fatal("per-site rates must be absent when a window lacked syncs")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDecodeTotalsDropGPUIdleButKeepSyncsAfterPipelinedWindow(t *testing.T) {
	f := newSyncFixture(t, "sync-fixture")
	ctx, acc := WithDecodeHostWork(context.Background())
	for _, cycles := range []uint64{0, 3} {
		span := StartDecodeSpan()
		time.Sleep(time.Millisecond)
		f.set(func(w *compute.DeviceWork) {
			w.Dispatches++
			w.DeviceIdleNS += uint64(100 * time.Microsecond)
			w.PipelinedSubmits += cycles
			w.PipelinedDrains += cycles
			w.SyncSites = compute.DeviceSyncSitesAdd(w.SyncSites, []compute.DeviceSyncSite{{Site: compute.SyncSiteBatchFlush, Stage: "add", Syncs: 2, WaitNS: 10}})
		})
		span.Finish(ctx, 1)
	}
	totals, ok := acc.Totals()
	if !ok || totals.DeviceIdle != nil {
		t.Fatalf("totals = %+v ok=%v, want device idle absent after a pipelined window", totals, ok)
	}
	if _, ok := totals.GPUIdleMSPerToken(); ok {
		t.Fatal("gpu idle ms/token must be absent once a window ran a pipelined batch")
	}
	if syncs, ok := totals.SyncsPerToken(); !ok || !approx(syncs, 2) {
		t.Fatalf("syncs/token = %v ok=%v, want 2", syncs, ok)
	}
}
