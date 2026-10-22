package compute

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestDeviceSyncSitesDeltaKeepsMovedCellsSorted(t *testing.T) {
	before := []DeviceSyncSite{
		{Site: SyncSiteFlushStatus, Stage: "v41_exact_rmsnorm", Syncs: 10, WaitNS: 100},
		{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 2, WaitNS: 50, D2HBytes: 64},
	}
	after := []DeviceSyncSite{
		{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 5, WaitNS: 80, D2HBytes: 160},
		{Site: SyncSiteFlushStatus, Stage: "v41_exact_rmsnorm", Syncs: 10, WaitNS: 100},
		{Site: SyncSiteBatchFlush, Stage: "add", Syncs: 1, WaitNS: 7},
	}
	got, ok := DeviceSyncSitesDelta(before, after)
	want := []DeviceSyncSite{
		{Site: SyncSiteBatchFlush, Stage: "add", Syncs: 1, WaitNS: 7},
		{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 3, WaitNS: 30, D2HBytes: 96},
	}
	if !ok || !reflect.DeepEqual(got, want) {
		t.Fatalf("delta = %+v ok=%v, want %+v", got, ok, want)
	}
	if got, ok := DeviceSyncSitesDelta(nil, nil); !ok || len(got) != 0 {
		t.Fatalf("empty delta = %+v ok=%v, want observed zero", got, ok)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceSyncSitesDeltaRefusesCounterReset(t *testing.T) {
	base := DeviceSyncSite{Site: SyncSiteStepEnd, Stage: "v41_mhc", Syncs: 4, WaitNS: 40, D2HBytes: 8}
	cases := map[string][]DeviceSyncSite{
		"syncs_backwards":    {{Site: base.Site, Stage: base.Stage, Syncs: 3, WaitNS: 40, D2HBytes: 8}},
		"wait_backwards":     {{Site: base.Site, Stage: base.Stage, Syncs: 4, WaitNS: 39, D2HBytes: 8}},
		"d2h_backwards":      {{Site: base.Site, Stage: base.Stage, Syncs: 4, WaitNS: 40, D2HBytes: 7}},
		"cell_disappeared":   {{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 9}},
		"all_cells_vanished": nil,
	}
	for name, after := range cases {
		if got, ok := DeviceSyncSitesDelta([]DeviceSyncSite{base}, after); ok || got != nil {
			t.Fatalf("%s: delta = %+v ok=%v, want refused", name, got, ok)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceSyncSitesAddMergesCells(t *testing.T) {
	a := []DeviceSyncSite{{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 1, WaitNS: 2, D2HBytes: 3}}
	b := []DeviceSyncSite{
		{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 10, WaitNS: 20, D2HBytes: 30},
		{Site: SyncSiteBatchFlush, Stage: "add", Syncs: 5, WaitNS: 6},
	}
	got := DeviceSyncSitesAdd(a, b)
	want := []DeviceSyncSite{
		{Site: SyncSiteBatchFlush, Stage: "add", Syncs: 5, WaitNS: 6},
		{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 11, WaitNS: 22, D2HBytes: 33},
	}
	if !reflect.DeepEqual(got, want) || a[0].Syncs != 1 || b[0].Syncs != 10 {
		t.Fatalf("sum = %+v (inputs %+v %+v), want %+v with inputs unchanged", got, a, b, want)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceWorkSnapshotCarriesSyncCells(t *testing.T) {
	cells := []DeviceSyncSite{
		{Site: SyncSiteFlushStatus, Stage: "v41_exact_rmsnorm", Syncs: 43, WaitNS: 4300},
		{Site: SyncSiteReadback, Stage: SyncStageNone, Syncs: 2, WaitNS: 900, D2HBytes: 512},
	}
	t.Cleanup(RegisterDeviceWorkReader("fake-device-sync", func() DeviceWork {
		return DeviceWork{Dispatches: 1, DeviceIdleNS: 42, SyncSites: cells, SyncObserved: true}
	}))
	for _, w := range DeviceWorkSnapshot() {
		if w.Backend != "fake-device-sync" {
			continue
		}
		if w.DeviceIdleNS != 42 || !w.SyncObserved || !reflect.DeepEqual(w.SyncSites, cells) {
			t.Fatalf("snapshot = %+v, want idle 42 and the reader's cells", w)
		}
		if syncs, wait := w.HostSyncs(); syncs != 45 || wait != 5200 {
			t.Fatalf("HostSyncs = %d/%d, want 45/5200", syncs, wait)
		}
		return
	}
	t.Fatal("registered backend missing from snapshot")
}

// fak-test:runtime fast est=1ms lane=default
func TestDeviceIdleDeltaRefusesPipelinedOrUnobservedWindows(t *testing.T) {
	before := DeviceWork{DeviceIdleNS: 100, SyncObserved: true, PipelinedSubmits: 3, PipelinedDrains: 3}
	after := before
	after.DeviceIdleNS = 160
	if ns, ok := DeviceIdleDelta(before, after); !ok || ns != 60 {
		t.Fatalf("delta = %d ok=%v, want 60", ns, ok)
	}
	cases := map[string]func(b, a *DeviceWork){
		"before_unobserved":      func(b, _ *DeviceWork) { b.SyncObserved = false },
		"after_unobserved":       func(_, a *DeviceWork) { a.SyncObserved = false },
		"idle_backwards":         func(_, a *DeviceWork) { a.DeviceIdleNS = 99 },
		"pipelined_submit":       func(_, a *DeviceWork) { a.PipelinedSubmits++ },
		"pipelined_cycle":        func(_, a *DeviceWork) { a.PipelinedSubmits, a.PipelinedDrains = 5, 5 },
		"drain_of_outstanding":   func(b, a *DeviceWork) { b.PipelinedSubmits, a.PipelinedSubmits, a.PipelinedDrains = 4, 4, 4 },
		"outstanding_throughout": func(b, a *DeviceWork) { b.PipelinedSubmits, a.PipelinedSubmits = 4, 4 },
		"counters_reset":         func(_, a *DeviceWork) { a.PipelinedSubmits, a.PipelinedDrains = 0, 0 },
	}
	for name, mutate := range cases {
		b, a := before, after
		mutate(&b, &a)
		if ns, ok := DeviceIdleDelta(b, a); ok || ns != 0 {
			t.Fatalf("%s: delta = %d ok=%v, want refused", name, ns, ok)
		}
	}
}
