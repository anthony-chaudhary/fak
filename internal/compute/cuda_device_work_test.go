package compute

import (
	"os"
	"regexp"
	"testing"
)

// fakeCUDAShim is a host-side counter source with the shim's snapshot shape.
type fakeCUDAShim struct {
	counters [cudaDWCounterCount]uint64
	cells    []cudaSyncCell
	stages   map[uint32]string
}

func (f *fakeCUDAShim) read() DeviceWork {
	return cudaDeviceWorkFromShim(f.counters, len(f.cells), f.cells, func(id uint32) string { return f.stages[id] })
}

func newFakeCUDAShim() *fakeCUDAShim {
	return &fakeCUDAShim{stages: map[uint32]string{2: "rms_norm", 3: "argmax"}}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDADeviceWorkFoldsSitesStagesAndIdle(t *testing.T) {
	f := newFakeCUDAShim()
	f.counters = [cudaDWCounterCount]uint64{cudaDWDispatches: 40, cudaDWReadbackBytes: 4096, cudaDWIdleNS: 900}
	f.cells = []cudaSyncCell{
		{Site: 1, Stage: 3, Syncs: 2, WaitNS: 300, D2HBytes: 4096},
		{Site: 0, Stage: 2, Syncs: 1, WaitNS: 100},
		{Site: 2, Stage: 0, Syncs: 1, WaitNS: 50},
	}
	w := f.read()
	if !w.SyncObserved || w.Dispatches != 40 || w.ReadbackBytes != 4096 || w.DeviceIdleNS != 900 {
		t.Fatalf("fold = %+v", w)
	}
	want := []DeviceSyncSite{
		{Site: SyncSiteReadback, Stage: "argmax", Syncs: 2, WaitNS: 300, D2HBytes: 4096},
		{Site: SyncSiteSubmitWait, Stage: "rms_norm", Syncs: 1, WaitNS: 100},
		{Site: SyncSiteUpload, Stage: SyncStageNone, Syncs: 1, WaitNS: 50},
	}
	if len(w.SyncSites) != len(want) {
		t.Fatalf("sites = %+v, want %+v", w.SyncSites, want)
	}
	for i := range want {
		if w.SyncSites[i] != want[i] {
			t.Fatalf("site %d = %+v, want %+v", i, w.SyncSites[i], want[i])
		}
	}
	if syncs, wait := w.HostSyncs(); syncs != 4 || wait != 450 {
		t.Fatalf("HostSyncs = %d, %d; want 4, 450", syncs, wait)
	}
	if w.SubmitWaitObserved {
		t.Fatal("CUDA has no submit-to-signal clock; SubmitWaitObserved must stay false")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDADeviceWorkWindowDeltas(t *testing.T) {
	f := newFakeCUDAShim()
	f.counters = [cudaDWCounterCount]uint64{cudaDWDispatches: 10, cudaDWIdleNS: 1000}
	f.cells = []cudaSyncCell{{Site: 1, Stage: 3, Syncs: 1, WaitNS: 20, D2HBytes: 8}}
	before := f.read()

	f.counters[cudaDWDispatches] = 35
	f.counters[cudaDWReadbackBytes] = 16
	f.counters[cudaDWIdleNS] = 1600
	f.cells = []cudaSyncCell{
		{Site: 1, Stage: 3, Syncs: 2, WaitNS: 50, D2HBytes: 16},
		{Site: 0, Stage: 2, Syncs: 1, WaitNS: 70},
	}
	after := f.read()

	if idle, ok := DeviceIdleDelta(before, after); !ok || idle != 600 {
		t.Fatalf("idle delta = %d, %v; want 600, true", idle, ok)
	}
	sites, ok := DeviceSyncSitesDelta(before.SyncSites, after.SyncSites)
	if !ok || len(sites) != 2 {
		t.Fatalf("site delta = %+v, %v", sites, ok)
	}
	if got := (DeviceSyncSite{Site: SyncSiteReadback, Stage: "argmax", Syncs: 1, WaitNS: 30, D2HBytes: 8}); sites[0] != got {
		t.Fatalf("readback delta = %+v, want %+v", sites[0], got)
	}
	if got := (DeviceSyncSite{Site: SyncSiteSubmitWait, Stage: "rms_norm", Syncs: 1, WaitNS: 70}); sites[1] != got {
		t.Fatalf("submit-wait delta = %+v, want %+v", sites[1], got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDADeviceWorkUnobservedLaunchRefusesWindow(t *testing.T) {
	f := newFakeCUDAShim()
	f.counters = [cudaDWCounterCount]uint64{cudaDWDispatches: 5, cudaDWIdleNS: 100}
	f.cells = []cudaSyncCell{{Site: 0, Stage: 2, Syncs: 1, WaitNS: 10}}
	before := f.read()
	f.counters[cudaDWUnobserved] = 1 // a graph replay bypassed the launch counters
	f.counters[cudaDWIdleNS] = 400
	after := f.read()
	if after.SyncObserved {
		t.Fatal("an unobserved launch must mark the counters unobserved")
	}
	if _, ok := DeviceIdleDelta(before, after); ok {
		t.Fatal("idle delta across an unobserved launch must be unavailable")
	}
	f.counters[cudaDWIdleNS] = 500
	if later := f.read(); later.SyncObserved {
		t.Fatal("unobserved is sticky for the process lifetime")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDADeviceWorkUnresolvableCellsLeaveSyncUnavailable(t *testing.T) {
	for name, tc := range map[string]struct {
		n     int
		cells []cudaSyncCell
	}{
		"torn read":        {n: 2, cells: []cudaSyncCell{{Site: 0, Stage: 2, Syncs: 1}}},
		"refused snapshot": {n: -1, cells: nil},
		"unknown site":     {n: 1, cells: []cudaSyncCell{{Site: uint32(len(cudaSyncSiteByID)), Stage: 2, Syncs: 1}}},
		"unassigned stage": {n: 1, cells: []cudaSyncCell{{Site: 0, Stage: 9, Syncs: 1}}},
	} {
		t.Run(name, func(t *testing.T) {
			f := newFakeCUDAShim()
			counters := [cudaDWCounterCount]uint64{cudaDWDispatches: 7, cudaDWReadbackBytes: 64, cudaDWIdleNS: 50}
			w := cudaDeviceWorkFromShim(counters, tc.n, tc.cells, func(id uint32) string { return f.stages[id] })
			if w.SyncObserved || w.SyncSites != nil || w.DeviceIdleNS != 0 {
				t.Fatalf("sync fields = observed %v sites %+v idle %d; want unavailable", w.SyncObserved, w.SyncSites, w.DeviceIdleNS)
			}
			if w.Dispatches != 7 || w.ReadbackBytes != 64 {
				t.Fatalf("dispatch/readback = %d/%d; want 7/64 kept", w.Dispatches, w.ReadbackBytes)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDADeviceWorkStageLabelsStayBounded(t *testing.T) {
	f := newFakeCUDAShim()
	f.stages[4] = "Bad-Name::with<template>"
	f.cells = []cudaSyncCell{
		{Site: 0, Stage: cudaSyncStageOtherID, Syncs: 1, WaitNS: 5},
		{Site: 0, Stage: 4, Syncs: 2, WaitNS: 7},
		{Site: 3, Stage: 2},
	}
	w := f.read()
	if !w.SyncObserved || len(w.SyncSites) != 1 {
		t.Fatalf("sites = %+v, observed %v; want one merged cell", w.SyncSites, w.SyncObserved)
	}
	if got := (DeviceSyncSite{Site: SyncSiteSubmitWait, Stage: HostFallbackOtherStage, Syncs: 3, WaitNS: 12}); w.SyncSites[0] != got {
		t.Fatalf("merged cell = %+v, want %+v", w.SyncSites[0], got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDADeviceWorkReaderRegistersInSnapshot(t *testing.T) {
	f := newFakeCUDAShim()
	t.Cleanup(RegisterDeviceWorkReader("fake-cuda-device-work", f.read))
	f.counters[cudaDWDispatches] = 3
	for _, w := range DeviceWorkSnapshot() {
		if w.Backend == "fake-cuda-device-work" {
			if w.Dispatches != 3 || !w.SyncObserved {
				t.Fatalf("snapshot row = %+v", w)
			}
			return
		}
	}
	t.Fatal("registered backend missing from snapshot")
}

// The site table is an ABI between the shim and the fold; the CUDA build cannot run on
// this host, so the order is checked against the shim source.
//
// fak-test:runtime fast est=5ms lane=default
func TestCUDADeviceWorkSiteTableMatchesShim(t *testing.T) {
	src, err := os.ReadFile("cuda/kernel_profile.cuh")
	if err != nil {
		t.Fatal(err)
	}
	table := regexp.MustCompile(`kDwSiteNames\[kDwSites\]\s*=\s*\{([^}]*)\}`).FindSubmatch(src)
	if table == nil {
		t.Fatal("kernel_profile.cuh has no kDwSiteNames table")
	}
	var got []string
	for _, m := range regexp.MustCompile(`"([a-z_]+)"`).FindAllSubmatch(table[1], -1) {
		got = append(got, string(m[1]))
	}
	if len(got) != len(cudaSyncSiteByID) {
		t.Fatalf("shim sites %v, Go sites %v", got, cudaSyncSiteByID)
	}
	for i, name := range cudaSyncSiteByID {
		if got[i] != name {
			t.Fatalf("site id %d: shim %q, Go %q", i, got[i], name)
		}
	}
}
