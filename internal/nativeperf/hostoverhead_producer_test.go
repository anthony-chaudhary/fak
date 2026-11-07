package nativeperf

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fakeGraphWitness drives model.GraphReplayWitness deterministically through
// the span's swappable reader.
type fakeGraphWitness struct {
	mu       sync.Mutex
	engaged  uint64
	degraded uint64
	armed    bool
}

func (f *fakeGraphWitness) read() (uint64, uint64, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.engaged, f.degraded, f.armed
}

func (f *fakeGraphWitness) add(engaged, degraded uint64, armed bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.engaged += engaged
	f.degraded += degraded
	f.armed = f.armed || armed
}

func installFakeGraphWitness(t *testing.T, f *fakeGraphWitness) {
	t.Helper()
	saved := graphReplayWitness
	graphReplayWitness = f.read
	t.Cleanup(func() { graphReplayWitness = saved })
}

// fakeH2DCensus drives the FAK_H2D_CENSUS snapshot/enabled pair the span diffs.
func installFakeH2DCensus(t *testing.T, on bool, bytes *atomic.Int64) {
	t.Helper()
	savedSnap, savedOn := h2dCensusSnapshot, h2dCensusEnabled
	h2dCensusSnapshot = func() (int64, int64) { return 0, bytes.Load() }
	h2dCensusEnabled = func() bool { return on }
	t.Cleanup(func() { h2dCensusSnapshot, h2dCensusEnabled = savedSnap, savedOn })
}

// fak-test:runtime fast est=20ms lane=default
func TestHostOverheadRejectsContaminatedCounters(t *testing.T) {
	if compute.KernelProfileReaderRegistered() || compute.DeviceBackendRegistered() {
		t.Skip("a backend counter reader is registered in this process")
	}
	for _, tc := range []struct {
		name, at, kind           string
		profile, census, cpuOnly bool
	}{
		{"overlapping_decode", "before", "decode", false, true, false},
		{"active_nondecode_without_profile", "before", "nondecode", false, true, false},
		{"opening_nondecode", "open_device", "nondecode", false, true, false},
		{"closing_cpu", "cpu", "decode", false, true, false},
		{"closing_device", "device", "decode", false, true, false},
		{"closing_kernel", "kernel", "decode", true, true, false},
		{"closing_graph", "graph", "decode", false, true, false},
		{"closing_h2d", "h2d", "decode", false, true, false},
		{"closing_nondecode", "h2d", "nondecode", false, true, false},
		{"sync_only_nondecode", "device", "nondecode", false, false, false},
		{"cpu_only_nondecode", "before", "nondecode", false, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var peer *DecodeSpan
			var nondecode NonDecodeWork
			defer func() { peer.Abandon(); nondecode.End() }()
			closing, fired := false, false
			contaminate := func(at string) {
				if fired || at != tc.at || (!closing && at != "before" && at != "open_device") {
					return
				}
				fired = true
				if tc.kind == "decode" {
					p := StartDecodeSpan()
					peer = &p
				} else {
					nondecode = BeginNonDecodeWork()
					if at == "open_device" {
						nondecode.End()
					}
				}
			}
			oldCPU, oldGraph := processCPUTime, graphReplayWitness
			oldSnapshot, oldEnabled := h2dCensusSnapshot, h2dCensusEnabled
			t.Cleanup(func() {
				processCPUTime, graphReplayWitness = oldCPU, oldGraph
				h2dCensusSnapshot, h2dCensusEnabled = oldSnapshot, oldEnabled
			})
			var cpu time.Duration
			var bytes, engaged int64
			processCPUTime = func() (time.Duration, bool) { contaminate("cpu"); return cpu, true }
			graphReplayWitness = func() (uint64, uint64, bool) { contaminate("graph"); return uint64(engaged), 0, !tc.cpuOnly }
			h2dCensusEnabled = func() bool { return tc.census }
			h2dCensusSnapshot = func() (int64, int64) { contaminate("h2d"); return 1, bytes }
			work := compute.DeviceWork{SubmitWaitObserved: true, SyncObserved: true}
			if !tc.cpuOnly {
				t.Cleanup(compute.RegisterDeviceWorkReader("ownership-test", func() compute.DeviceWork {
					if !closing {
						contaminate("open_device")
					}
					contaminate("device")
					return work
				}))
			}
			var profile compute.KernelProfile
			if tc.profile {
				t.Cleanup(compute.RegisterKernelProfileReader("ownership-test", func() (compute.KernelProfile, bool) {
					contaminate("kernel")
					return profile, true
				}))
			}
			contaminate("before")
			ctx, acc := WithDecodeHostWork(context.Background())
			span := StartDecodeSpan()
			defer span.Abandon()
			cpu, bytes, engaged = time.Millisecond, 123, 2
			work.Dispatches, work.SubmitWaitNS = 1, 1
			work.ReadbackBytes = 17
			work.SyncSites = []compute.DeviceSyncSite{{Site: compute.SyncSiteStepEnd, Syncs: 1, WaitNS: 1}}
			profile = compute.KernelProfile{Batches: 1, Ops: 1, GPUNS: 1, WallNS: 2,
				Rows: []compute.KernelRow{{Kernel: "decode", Calls: 1, NS: 1}}}
			closing = true
			w := span.Finish(ctx, 1)
			if !fired || !(w.Overlapped || w.NonDecodeOverlap) {
				t.Errorf("contamination was not observed: fired=%v window=%+v", fired, w)
			}
			if w.CPU != nil || w.Kernel != nil || w.Kernels != nil || w.DeviceIdle != nil || w.SyncsObserved || len(w.SyncSites) != 0 || w.Dispatches != 0 || w.Readback != 0 ||
				w.H2DCopyBytes != nil || w.GraphObserved || w.GraphEngaged != 0 || w.GraphDegraded != 0 {
				t.Errorf("global measurements attributed to a contaminated request: %+v", w)
			}
			if w.Wall < 0 || w.Steps != 1 || w.GraphArmed == tc.cpuOnly {
				t.Errorf("owned wall/steps or capability evidence lost: %+v", w)
			}
			var rep Repetition
			if !acc.ClaimHostOverhead(&rep) || rep.HostPhaseUs != nil || rep.KernelLaunchCount != nil || rep.InterKernelGapSumUs != nil ||
				rep.H2DCopyBytes != nil || rep.SyncRoundtrips != nil || rep.GraphReplayEngagedTokens != nil || rep.GraphReplayDegradedTokens != nil {
				t.Errorf("contaminated global measurements persisted: %+v", rep)
			}
			h2dReason, graphReason := HostOverheadGapUnobserved, HostOverheadGapUnobserved
			if !tc.census {
				h2dReason = HostOverheadGapDisabled
			}
			if tc.cpuOnly {
				graphReason = HostOverheadGapUnsupported
			}
			wantGaps := map[string]string{
				"host_phase_us": HostOverheadGapUnobserved, "sync_roundtrips": HostOverheadGapUnobserved,
				"kernel_launch_count": HostOverheadGapDisabled, "inter_kernel_gap_sum_us": HostOverheadGapDisabled,
				"h2d_copy_bytes": h2dReason, "graph_replay_engaged_tokens": graphReason, "graph_replay_degraded_tokens": graphReason,
			}
			for _, gap := range rep.HostOverheadGaps {
				field, reason, found := strings.Cut(gap, "=")
				if !found || wantGaps[field] != reason {
					t.Errorf("unexpected gap %q", gap)
				}
				delete(wantGaps, field)
			}
			if len(wantGaps) != 0 || acc.ClaimHostOverhead(&Repetition{}) {
				t.Errorf("missing gaps %v or repeated claim accepted", wantGaps)
			}
			if peer != nil {
				other := peer.Finish(context.Background(), 1)
				if !other.Overlapped || other.H2DCopyBytes != nil || other.GraphObserved {
					t.Errorf("the overlapping peer inherited attribution: %+v", other)
				}
			}
		})
	}

	t.Run("uncontaminated_zero_and_reset", func(t *testing.T) {
		var bytes atomic.Int64
		installFakeH2DCensus(t, true, &bytes)
		ctx, acc := WithDecodeHostWork(context.Background())
		span := StartDecodeSpan()
		w := span.Finish(ctx, 1)
		var rep Repetition
		if w.H2DCopyBytes == nil || *w.H2DCopyBytes != 0 || !acc.ClaimHostOverhead(&rep) || rep.H2DCopyBytes == nil || *rep.H2DCopyBytes != 0 {
			t.Fatalf("uncontaminated measured zero lost: window=%+v repetition=%+v", w, rep)
		}
		bytes.Store(10)
		span = StartDecodeSpan()
		bytes.Store(9)
		if w := span.Finish(context.Background(), 1); w.H2DCopyBytes != nil {
			t.Fatalf("reset census became a measurement: %+v", w)
		}
	})

	t.Run("snapshot_panic_releases_span", func(t *testing.T) {
		var bytes atomic.Int64
		installFakeH2DCensus(t, true, &bytes)
		before := decodeSpansInFlight.Load()
		span := StartDecodeSpan()
		defer span.Abandon()
		h2dCensusSnapshot = func() (int64, int64) { panic("snapshot") }
		func() {
			defer func() {
				if recover() == nil {
					t.Error("snapshot did not panic")
				}
			}()
			span.Finish(context.Background(), 1)
		}()
		if got := decodeSpansInFlight.Load(); got != before {
			t.Errorf("closing snapshot stranded an active span: got %d, want %d", got, before)
		}
	})
}

// fakeDeviceWork is a cumulative device-work reader with observed sync cells.
type fakeDeviceWork struct {
	mu sync.Mutex
	w  compute.DeviceWork
}

func (f *fakeDeviceWork) read() compute.DeviceWork {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.w
}

func (f *fakeDeviceWork) addSyncs(site, stage string, syncs, waitNS, d2h uint64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for i := range f.w.SyncSites {
		if f.w.SyncSites[i].Site == site && f.w.SyncSites[i].Stage == stage {
			f.w.SyncSites[i].Syncs += syncs
			f.w.SyncSites[i].WaitNS += waitNS
			f.w.SyncSites[i].D2HBytes += d2h
			return
		}
	}
	f.w.SyncSites = append(f.w.SyncSites, compute.DeviceSyncSite{Site: site, Stage: stage, Syncs: syncs, WaitNS: waitNS, D2HBytes: d2h})
}

// fak-test:runtime fast est=10ms lane=default
func TestClaimHostOverheadFillsEveryObservableField(t *testing.T) {
	var cpu atomic.Int64
	fakeProcessCPU(t, &cpu, true)
	f := registerFakeKernelProfile(t)
	dev := &fakeDeviceWork{w: compute.DeviceWork{SubmitWaitObserved: true, SyncObserved: true}}
	t.Cleanup(compute.RegisterDeviceWorkReader("vulkan", dev.read))
	witness := &fakeGraphWitness{engaged: 100, degraded: 5, armed: true}
	installFakeGraphWitness(t, witness)
	var h2d atomic.Int64
	installFakeH2DCensus(t, true, &h2d)

	ctx, acc := WithDecodeHostWork(context.Background())
	span := StartDecodeSpan()
	cpu.Add(int64(1500 * time.Microsecond))
	f.step(10)
	dev.addSyncs(compute.SyncSiteReadback, compute.SyncStageNone, 1104, uint64(4*time.Millisecond), 8192)
	witness.add(9, 2, true)
	h2d.Add(2048)
	w := span.Finish(ctx, 10)
	if w.CPU == nil || w.Kernels == nil || !w.SyncsObserved {
		t.Fatalf("window lost measured sources: %+v", w)
	}
	if w.H2DCopyBytes == nil || *w.H2DCopyBytes != 2048 {
		t.Fatalf("window h2d copy bytes = %v, want 2048", w.H2DCopyBytes)
	}
	if w.GraphEngaged != 9 || w.GraphDegraded != 2 || !w.GraphObserved {
		t.Fatalf("window engagement = %d/%d observed=%v, want 9/2 observed", w.GraphEngaged, w.GraphDegraded, w.GraphObserved)
	}

	rep := Repetition{}
	if !acc.ClaimHostOverhead(&rep) {
		t.Fatal("first claim must project the request's decomposition")
	}
	if rep.HostPhaseUs == nil || *rep.HostPhaseUs != 1500 {
		t.Fatalf("host_phase_us = %v, want 1500", rep.HostPhaseUs)
	}
	if rep.KernelLaunchCount == nil || *rep.KernelLaunchCount != 30 {
		t.Fatalf("kernel_launch_count = %v, want 30 (two kernels x 10 steps)", rep.KernelLaunchCount)
	}
	if rep.InterKernelGapSumUs == nil || *rep.InterKernelGapSumUs != 20 {
		t.Fatalf("inter_kernel_gap_sum_us = %v, want 20 (wall 90us - gpu 70us over 10 steps)", rep.InterKernelGapSumUs)
	}
	if rep.H2DCopyBytes == nil || *rep.H2DCopyBytes != 2048 {
		t.Fatalf("h2d_copy_bytes = %v, want 2048", rep.H2DCopyBytes)
	}
	if rep.SyncRoundtrips == nil || *rep.SyncRoundtrips != 1104 {
		t.Fatalf("sync_roundtrips = %v, want 1104", rep.SyncRoundtrips)
	}
	if rep.GraphReplayEngagedTokens == nil || *rep.GraphReplayEngagedTokens != 9 ||
		rep.GraphReplayDegradedTokens == nil || *rep.GraphReplayDegradedTokens != 2 {
		t.Fatalf("engagement witness = %v/%v, want 9/2", rep.GraphReplayEngagedTokens, rep.GraphReplayDegradedTokens)
	}
	if len(rep.HostOverheadGaps) != 0 {
		t.Fatalf("fully observed repetition must carry no gaps: %v", rep.HostOverheadGaps)
	}
	if acc.ClaimHostOverhead(&Repetition{}) {
		t.Fatal("a second claim must not rebook the decomposition")
	}

	// The filled repetition must survive strict decode + validation and keep
	// its values across the JSON round trip.
	r := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	r.Repetitions[0] = rep
	r.Repetitions[0].EndToEndMilliseconds = 100
	r.Repetitions[0].TokensPerSecond = 3
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeReceipt(data)
	if err != nil {
		t.Fatalf("strict decode of a filled v3 receipt: %v", err)
	}
	if err := ValidateReceipt(ActiveGraph(), got); err != nil {
		t.Fatalf("filled v3 receipt must validate: %v", err)
	}
	if got.Repetitions[0].GraphReplayEngagedTokens == nil || *got.Repetitions[0].GraphReplayEngagedTokens != 9 ||
		got.Repetitions[0].InterKernelGapSumUs == nil || *got.Repetitions[0].InterKernelGapSumUs != 20 {
		t.Fatalf("round trip lost decomposition fields: %+v", got.Repetitions[0])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestUnobservableDecompositionFieldsRoundTripAsGapsNotZeros(t *testing.T) {
	if compute.KernelProfileReaderRegistered() || compute.DeviceBackendRegistered() {
		t.Skip("a backend counter reader is registered in this process")
	}
	var cpu atomic.Int64
	fakeProcessCPU(t, &cpu, true)
	witness := &fakeGraphWitness{} // never armed: no capture-once backend ran
	installFakeGraphWitness(t, witness)
	var h2d atomic.Int64
	installFakeH2DCensus(t, false, &h2d) // census off: counters stay empty

	ctx, acc := WithDecodeHostWork(context.Background())
	span := StartDecodeSpan()
	cpu.Add(int64(700 * time.Microsecond))
	w := span.Finish(ctx, 3)
	if w.H2DCopyBytes != nil || w.GraphObserved {
		t.Fatalf("unobservable sources must stay unobserved: %+v", w)
	}

	rep := Repetition{}
	if !acc.ClaimHostOverhead(&rep) {
		t.Fatal("claim must succeed even when most fields are unobservable")
	}
	if rep.HostPhaseUs == nil || *rep.HostPhaseUs != 700 {
		t.Fatalf("measured host_phase_us lost: %v", rep.HostPhaseUs)
	}
	wantGaps := map[string]bool{
		"kernel_launch_count=" + HostOverheadGapDisabled:             true,
		"inter_kernel_gap_sum_us=" + HostOverheadGapDisabled:         true,
		"sync_roundtrips=" + HostOverheadGapUnobserved:               true,
		"h2d_copy_bytes=" + HostOverheadGapDisabled:                  true,
		"graph_replay_engaged_tokens=" + HostOverheadGapUnsupported:  true,
		"graph_replay_degraded_tokens=" + HostOverheadGapUnsupported: true,
	}
	if len(rep.HostOverheadGaps) != len(wantGaps) {
		t.Fatalf("gaps = %v, want %d entries", rep.HostOverheadGaps, len(wantGaps))
	}
	for _, g := range rep.HostOverheadGaps {
		if !wantGaps[g] {
			t.Fatalf("unexpected gap entry %q in %v", g, rep.HostOverheadGaps)
		}
	}
	// Absent stays absent after the strict JSON round trip: unknown, never zero.
	data, err := json.Marshal(Repetition{HostPhaseUs: rep.HostPhaseUs, HostOverheadGaps: rep.HostOverheadGaps})
	if err != nil {
		t.Fatal(err)
	}
	var back Repetition
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.KernelLaunchCount != nil || back.H2DCopyBytes != nil ||
		back.GraphReplayEngagedTokens != nil || back.GraphReplayDegradedTokens != nil {
		t.Fatalf("absent fields must not fabricate zeros: %+v", back)
	}
	if !strings.Contains(string(data), `"host_overhead_gaps"`) {
		t.Fatalf("gap reasons must round-trip machine-readably: %s", data)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestGraphReplayWitnessRollsUpWindowsAndKeepsMeasuredZero(t *testing.T) {
	var cpu atomic.Int64
	fakeProcessCPU(t, &cpu, true)
	witness := &fakeGraphWitness{armed: true}
	installFakeGraphWitness(t, witness)

	ctx, acc := WithDecodeHostWork(context.Background())
	span := StartDecodeSpan()
	cpu.Add(int64(time.Millisecond))
	witness.add(5, 0, true)
	if w := span.Finish(ctx, 4); w.GraphEngaged != 5 || w.GraphDegraded != 0 || !w.GraphObserved {
		t.Fatalf("window one = %d/%d observed=%v, want 5/0 observed", w.GraphEngaged, w.GraphDegraded, w.GraphObserved)
	}
	span = StartDecodeSpan()
	cpu.Add(int64(time.Millisecond))
	witness.add(0, 3, true)
	if w := span.Finish(ctx, 4); w.GraphEngaged != 0 || w.GraphDegraded != 3 || !w.GraphObserved {
		t.Fatalf("window two = %d/%d observed=%v, want 0/3 observed", w.GraphEngaged, w.GraphDegraded, w.GraphObserved)
	}
	totals, ok := acc.Totals()
	if !ok || totals.GraphReplayEngaged == nil || *totals.GraphReplayEngaged != 5 ||
		totals.GraphReplayDegraded == nil || *totals.GraphReplayDegraded != 3 || !totals.GraphBackendSupported {
		t.Fatalf("request rollup = %v/%v supported=%v, want 5/3 supported", totals.GraphReplayEngaged, totals.GraphReplayDegraded, totals.GraphBackendSupported)
	}

	// A window where nothing engaged or degraded is still a MEASURED zero on an
	// armed backend: present pointers to zero, not absent fields.
	other, otherAcc := WithDecodeHostWork(context.Background())
	span = StartDecodeSpan()
	cpu.Add(int64(time.Millisecond))
	if w := span.Finish(other, 2); w.GraphEngaged != 0 || w.GraphDegraded != 0 || !w.GraphObserved {
		t.Fatalf("quiet window = %d/%d observed=%v, want measured 0/0 observed", w.GraphEngaged, w.GraphDegraded, w.GraphObserved)
	}
	zeroTotals, _ := otherAcc.Totals()
	if zeroTotals.GraphReplayEngaged == nil || *zeroTotals.GraphReplayEngaged != 0 ||
		zeroTotals.GraphReplayDegraded == nil || *zeroTotals.GraphReplayDegraded != 0 {
		t.Fatalf("quiet request = %v/%v, want present measured zeros", zeroTotals.GraphReplayEngaged, zeroTotals.GraphReplayDegraded)
	}

	// A counter that went backwards drops the observation instead of minting a
	// negative delta.
	regressing := &fakeGraphWitness{engaged: 10, degraded: 0, armed: true}
	installFakeGraphWitness(t, regressing)
	regressed, regressedAcc := WithDecodeHostWork(context.Background())
	span = StartDecodeSpan()
	cpu.Add(int64(time.Millisecond))
	regressing.engaged = 4 // reset between start and finish
	if w := span.Finish(regressed, 2); w.GraphObserved {
		t.Fatalf("regressed counters must drop the observation: %+v", w)
	}
	regressedTotals, _ := regressedAcc.Totals()
	if regressedTotals.GraphReplayEngaged != nil || regressedTotals.GraphReplayDegraded != nil {
		t.Fatalf("regressed request must keep the witness absent: %v/%v", regressedTotals.GraphReplayEngaged, regressedTotals.GraphReplayDegraded)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestHostOverheadGapGrammarIsClosed(t *testing.T) {
	graph := ActiveGraph()
	badField := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	badField.Repetitions[0].HostOverheadGaps = []string{"bogus_field=disabled"}
	if err := ValidateReceipt(graph, badField); err == nil || !strings.Contains(err.Error(), "closed field/reason grammar") {
		t.Fatalf("unknown gap field must be refused, err=%v", err)
	}
	badReason := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	badReason.Repetitions[0].HostOverheadGaps = []string{"h2d_copy_bytes=mystery"}
	if err := ValidateReceipt(graph, badReason); err == nil || !strings.Contains(err.Error(), "closed field/reason grammar") {
		t.Fatalf("unknown gap reason must be refused, err=%v", err)
	}
	contradiction := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	v := 1500.0
	contradiction.Repetitions[0].HostPhaseUs = &v
	contradiction.Repetitions[0].HostOverheadGaps = []string{"host_phase_us=unobserved"}
	if err := ValidateReceipt(graph, contradiction); err == nil || !strings.Contains(err.Error(), "names recorded field") {
		t.Fatalf("a gap naming a recorded field must be refused, err=%v", err)
	}
	v2WithWitness := validReceipt(t, RoleBaseline, "metal.command-buffer-amortization")
	v2WithWitness.Schema = ReceiptSchemaV2
	e := uint64(1)
	v2WithWitness.Repetitions[0].GraphReplayEngagedTokens = &e
	if err := ValidateReceipt(graph, v2WithWitness); err == nil || !strings.Contains(err.Error(), "require schema v3") {
		t.Fatalf("the engagement witness requires schema v3, err=%v", err)
	}
}
