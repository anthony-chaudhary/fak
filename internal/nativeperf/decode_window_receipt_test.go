package nativeperf

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func receiptPhase(t *testing.T, r PhaseReceipt, p Phase) PhaseAccounting {
	t.Helper()
	for _, a := range r.Phases {
		if a.Phase == p {
			return a
		}
	}
	t.Fatalf("receipt has no %q phase", p)
	return PhaseAccounting{}
}

// fak-test:runtime fast est=2ms lane=default
func TestDecodeWindowReceiptBooksSyncWait(t *testing.T) {
	kernel := 4 * time.Millisecond
	for _, tc := range []struct {
		name     string
		totals   DecodeHostWorkTotals
		ok       bool
		kernel   time.Duration
		sync     time.Duration
		decode   time.Duration
		d2hBytes uint64
	}{
		{
			name: "sync_beyond_fence_wall",
			totals: DecodeHostWorkTotals{Wall: 10 * time.Millisecond, Kernel: &kernel, Backend: "cuda", SyncsObserved: true, SyncSites: []compute.DeviceSyncSite{
				{Site: compute.SyncSiteSubmitWait, Stage: "add", Syncs: 4, WaitNS: uint64(4 * time.Millisecond)},
				{Site: compute.SyncSitePipelineDrain, Stage: "add", Syncs: 2, WaitNS: uint64(2 * time.Millisecond)},
				{Site: compute.SyncSiteReadback, Stage: compute.SyncStageNone, Syncs: 1, WaitNS: uint64(time.Millisecond), D2HBytes: 4096},
			}},
			ok: true, kernel: 4 * time.Millisecond, sync: 3 * time.Millisecond, decode: 3 * time.Millisecond, d2hBytes: 4096,
		},
		{
			name: "sync_inside_fence_wall_books_once",
			totals: DecodeHostWorkTotals{Wall: 10 * time.Millisecond, Kernel: &kernel, Backend: "cuda", SyncsObserved: true, SyncSites: []compute.DeviceSyncSite{
				{Site: compute.SyncSiteSubmitWait, Stage: "add", Syncs: 4, WaitNS: uint64(3 * time.Millisecond), D2HBytes: 8},
			}},
			ok: true, kernel: 4 * time.Millisecond, sync: 0, decode: 6 * time.Millisecond, d2hBytes: 8,
		},
		{
			name: "syncs_unobserved_keeps_two_phase_split",
			totals: DecodeHostWorkTotals{Wall: 10 * time.Millisecond, Kernel: &kernel, Backend: "cuda", SyncSites: []compute.DeviceSyncSite{
				{Site: compute.SyncSitePipelineDrain, Stage: "add", Syncs: 2, WaitNS: uint64(5 * time.Millisecond), D2HBytes: 64},
			}},
			ok: true, kernel: 4 * time.Millisecond, sync: 0, decode: 6 * time.Millisecond,
		},
		{
			name: "sync_wait_exceeds_wall_refused",
			totals: DecodeHostWorkTotals{Wall: 10 * time.Millisecond, Kernel: &kernel, Backend: "cuda", SyncsObserved: true, SyncSites: []compute.DeviceSyncSite{
				{Site: compute.SyncSitePipelineDrain, Stage: "add", Syncs: 2, WaitNS: uint64(11 * time.Millisecond)},
			}},
		},
		{
			name: "sync_wait_overflow_refused",
			totals: DecodeHostWorkTotals{Wall: 10 * time.Millisecond, Kernel: &kernel, Backend: "cuda", SyncsObserved: true, SyncSites: []compute.DeviceSyncSite{
				{Site: compute.SyncSitePipelineDrain, Stage: "a", Syncs: 1, WaitNS: ^uint64(0)},
				{Site: compute.SyncSiteReadback, Stage: "b", Syncs: 1, WaitNS: 2},
			}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, ok := tc.totals.DecodeWindowReceipt()
			if ok != tc.ok {
				t.Fatalf("receipt ok=%v, want %v", ok, tc.ok)
			}
			if !ok {
				return
			}
			if err := r.Validate(time.Nanosecond); err != nil {
				t.Fatalf("receipt does not validate: %v", err)
			}
			k, s, d := receiptPhase(t, r, PhaseKernel), receiptPhase(t, r, PhaseSynchronization), receiptPhase(t, r, PhaseDecode)
			if k.Exclusive.Wall != tc.kernel || s.Exclusive.Wall != tc.sync || d.Exclusive.Wall != tc.decode {
				t.Fatalf("kernel/sync/decode = %s/%s/%s, want %s/%s/%s", k.Exclusive.Wall, s.Exclusive.Wall, d.Exclusive.Wall, tc.kernel, tc.sync, tc.decode)
			}
			if s.Exclusive.Wait != tc.sync || s.Exclusive.Active != 0 || (tc.sync > 0 && s.Parent != PhaseDecode) {
				t.Fatalf("synchronization timing = %+v parent %q, want wait-only under decode", s.Exclusive, s.Parent)
			}
			if r.HostDownloadBytes != tc.d2hBytes {
				t.Fatalf("host download bytes = %d, want %d", r.HostDownloadBytes, tc.d2hBytes)
			}
			if dl := receiptPhase(t, r, PhaseHostDownload); dl.Exclusive.Wall != 0 {
				t.Fatalf("host_device_download wall = %s, want 0: bytes never mint a copy duration", dl.Exclusive.Wall)
			}
			raw, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			var wire map[string]any
			if err := json.Unmarshal(raw, &wire); err != nil {
				t.Fatal(err)
			}
			if _, present := wire["host_device_download_bytes"]; present != (tc.d2hBytes > 0) {
				t.Fatalf("host_device_download_bytes present=%v, want %v", present, tc.d2hBytes > 0)
			}
		})
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestDecodeWindowReceiptSyncPhaseReachesMetrics(t *testing.T) {
	kernel := 4 * time.Millisecond
	totals := DecodeHostWorkTotals{Wall: 10 * time.Millisecond, Kernel: &kernel, Backend: "cuda", SyncsObserved: true, SyncSites: []compute.DeviceSyncSite{
		{Site: compute.SyncSitePipelineDrain, Stage: "add", Syncs: 2, WaitNS: uint64(6 * time.Millisecond)},
	}}
	r, ok := totals.DecodeWindowReceipt()
	if !ok {
		t.Fatal("observed syncs within the wall must yield a receipt")
	}
	m := NewReceiptMetrics(0)
	if !m.ObservePhases(r, time.Now()) {
		t.Fatalf("phase receipt rejected: %v", r.Validate(time.Nanosecond))
	}
	if got := receiptPhase(t, r, PhaseSynchronization).Exclusive.Wall; got != 2*time.Millisecond {
		t.Fatalf("synchronization = %s, want 2ms (6ms waits minus 4ms fence wall)", got)
	}
}
