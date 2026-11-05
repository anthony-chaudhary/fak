package gateway

import (
	"context"
	"crypto/sha256"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

// hostOverheadGapFields mirrors the receipt-v3 closed field vocabulary so the
// seam test can check the "<field>=<reason>" grammar without nativeperf's
// unexported state.
var hostOverheadGapFields = map[string]bool{
	"host_phase_us":                true,
	"kernel_launch_count":          true,
	"inter_kernel_gap_sum_us":      true,
	"h2d_copy_bytes":               true,
	"sync_roundtrips":              true,
	"graph_replay_engaged_tokens":  true,
	"graph_replay_degraded_tokens": true,
}

// checkHostOverheadContract asserts one recorded decomposition forms the
// closed contract: every absent field carries exactly one "<field>=<reason>"
// entry with a closed reason, and no gap names a field the repetition records.
func checkHostOverheadContract(t *testing.T, rep *nativeperf.Repetition) {
	t.Helper()
	recorded := map[string]bool{
		"host_phase_us":                rep.HostPhaseUs != nil,
		"kernel_launch_count":          rep.KernelLaunchCount != nil,
		"inter_kernel_gap_sum_us":      rep.InterKernelGapSumUs != nil,
		"h2d_copy_bytes":               rep.H2DCopyBytes != nil,
		"sync_roundtrips":              rep.SyncRoundtrips != nil,
		"graph_replay_engaged_tokens":  rep.GraphReplayEngagedTokens != nil,
		"graph_replay_degraded_tokens": rep.GraphReplayDegradedTokens != nil,
	}
	gapped := map[string]bool{}
	for _, gap := range rep.HostOverheadGaps {
		field, reason, ok := strings.Cut(gap, "=")
		closedReason := reason == nativeperf.HostOverheadGapDisabled || reason == nativeperf.HostOverheadGapUnobserved || reason == nativeperf.HostOverheadGapUnsupported
		if !ok || !hostOverheadGapFields[field] || !closedReason {
			t.Fatalf("gap entry %q violates the closed field/reason grammar", gap)
		}
		if gapped[field] {
			t.Fatalf("gap entries name %q twice", field)
		}
		gapped[field] = true
	}
	for field, rec := range recorded {
		switch {
		case rec && gapped[field]:
			t.Fatalf("gap entry names recorded field %q", field)
		case !rec && !gapped[field]:
			t.Fatalf("absent field %q carries no gap reason", field)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestObserveDecodeWindowReceiptClaimsHostOverheadOnce(t *testing.T) {
	if compute.KernelProfileReaderRegistered() {
		t.Skip("a backend kernel profile reader is registered in this process")
	}
	const backend = "host-overhead-test"

	var profile compute.KernelProfile // cumulative profile; advanced inside the window below
	t.Cleanup(compute.RegisterKernelProfileReader(backend, func() (compute.KernelProfile, bool) { return profile, true }))

	work := compute.DeviceWork{SubmitWaitObserved: true, SyncObserved: true}
	t.Cleanup(compute.RegisterDeviceWorkReader(backend, func() compute.DeviceWork { return work }))

	ctx, _ := nativeperf.WithDecodeHostWork(context.Background())
	span := nativeperf.StartDecodeSpan()
	// Spin past a Windows process-CPU tick (~15.6ms) so the window measures CPU.
	for deadline := time.Now().Add(40 * time.Millisecond); time.Now().Before(deadline); {
	}
	profile = compute.KernelProfile{
		Batches: 2,
		Ops:     10,
		GPUNS:   4000,
		WallNS:  6000,
		Rows: []compute.KernelRow{
			{Kernel: "attention_decode", GX: 1, GY: 1, Calls: 7, NS: 2500},
			{Kernel: "matvec_w4a16", GX: 1, GY: 1, Calls: 3, NS: 1500},
		},
	}
	work = compute.DeviceWork{
		Dispatches:         10,
		SubmitWaitObserved: true,
		SyncObserved:       true,
		SyncSites: []compute.DeviceSyncSite{
			{Site: compute.SyncSiteStepEnd, Stage: "matvec", Syncs: 5, WaitNS: 1000},
		},
	}
	span.Finish(ctx, 4)

	o := &requestObservation{digest: sha256.New(), phaseContext: ctx}
	m := &gatewayMetrics{}

	if _, ok := m.observeDecodeWindowReceipt(perfSource{observation: o}); !ok {
		t.Fatal("first pass returned no phase receipt, want the request's decode window")
	}
	rep, found := o.claimedHostOverhead()
	if !found {
		t.Fatal("the decode-window receipt path did not retain the claimed host-overhead decomposition")
	}
	if !rep.HasHostOverhead() {
		t.Fatal("claimed repetition carries no host-overhead decomposition")
	}
	checkHostOverheadContract(t, rep)
	if rep.KernelLaunchCount == nil || *rep.KernelLaunchCount != 10 {
		t.Fatalf("kernel_launch_count = %v, want 10 (sum of the profile's row calls)", rep.KernelLaunchCount)
	}
	if rep.SyncRoundtrips == nil || *rep.SyncRoundtrips != 5 {
		t.Fatalf("sync_roundtrips = %v, want 5 (the step_end site's sync delta)", rep.SyncRoundtrips)
	}
	if rep.InterKernelGapSumUs == nil || *rep.InterKernelGapSumUs != 2.0 {
		t.Fatalf("inter_kernel_gap_sum_us = %v, want 2 (window wall minus GPU-busy, in µs)", rep.InterKernelGapSumUs)
	}
	if rep.HostPhaseUs == nil || math.IsNaN(*rep.HostPhaseUs) || math.IsInf(*rep.HostPhaseUs, 0) || *rep.HostPhaseUs < 0 {
		t.Fatalf("host_phase_us = %v, want a finite non-negative measurement", rep.HostPhaseUs)
	}

	// Claim-once: a second pass over the same observation books nothing new.
	if _, ok := m.observeDecodeWindowReceipt(perfSource{observation: o}); ok {
		t.Fatal("second pass returned a phase receipt, want none")
	}
	again, found := o.claimedHostOverhead()
	if !found || again != rep {
		t.Fatal("second pass replaced or lost the retained decomposition")
	}

	// A turn that never decoded claims nothing through the same seam.
	plain := &requestObservation{digest: sha256.New(), phaseContext: context.Background()}
	if _, ok := m.observeDecodeWindowReceipt(perfSource{observation: plain}); ok {
		t.Fatal("no-decode turn returned a phase receipt, want none")
	}
	if _, found := plain.claimedHostOverhead(); found {
		t.Fatal("no-decode turn retained a host-overhead decomposition, want none")
	}
}
