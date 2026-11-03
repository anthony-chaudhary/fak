package nativeperf

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestSubagentTraceProvenanceAdmission(t *testing.T) {
	measured := func() SubagentTraceReceipt {
		r, err := NewSubagentTraceReceipt(1, "fixture", map[string]float64{
			SubagentPhaseHostDispatch: 1, SubagentPhasePrefixTreeLookup: 1,
			SubagentPhaseKVAllocation: 1, SubagentPhaseGPUKernel: 1,
			SubagentPhaseTokenSampling: 1,
		}, 5)
		if err != nil {
			t.Fatal(err)
		}
		r.Provenance = TraceMeasured
		r.PhaseProvenance = make(map[string]TraceProvenance)
		for phase := range r.PhasesUS {
			r.PhaseProvenance[phase] = TraceMeasured
		}
		return r
	}
	cases := []struct {
		name            string
		mutate          func(*SubagentTraceReceipt)
		valid, admitted bool
	}{
		{"measured", func(r *SubagentTraceReceipt) {}, true, true},
		{"measured-zero", func(r *SubagentTraceReceipt) {
			for phase := range r.PhasesUS {
				r.PhasesUS[phase] = 0
			}
			r.TotalWallUS, r.HostCPUOverheadUS, r.GPUKernelWallUS = 0, 0, 0
			r.HostCPUOverheadPercent, r.GPUKernelWallPercent = 0, 0
		}, true, true},
		{"measured-extra-zero", func(r *SubagentTraceReceipt) {
			r.PhasesUS["extra"] = 0
			r.PhaseProvenance["extra"] = TraceMeasured
		}, true, true},
		{"measured-second-turn", func(r *SubagentTraceReceipt) { r.Turn = 2 }, true, true},
		{"legacy", func(r *SubagentTraceReceipt) { r.Provenance = ""; r.PhaseProvenance = nil }, true, false},
		{"simulated", func(r *SubagentTraceReceipt) {
			if err := r.MarkPhaseSimulated(SubagentPhaseGPUKernel); err != nil {
				t.Fatal(err)
			}
		}, true, false},
		{"missing-phase-provenance", func(r *SubagentTraceReceipt) { delete(r.PhaseProvenance, SubagentPhaseGPUKernel) }, false, false},
		{"unknown-phase", func(r *SubagentTraceReceipt) {
			r.Provenance = TraceUnknown
			r.PhaseProvenance[SubagentPhaseGPUKernel] = TraceUnknown
		}, true, false},
		{"aggregate-only", func(r *SubagentTraceReceipt) { r.PhaseProvenance = nil }, false, false},
		{"unknown-token", func(r *SubagentTraceReceipt) { r.PhaseProvenance[SubagentPhaseGPUKernel] = "calibrated" }, false, false},
		{"bad-sum", func(r *SubagentTraceReceipt) { r.TotalWallUS++ }, false, false},
		{"bad-percent", func(r *SubagentTraceReceipt) { r.GPUKernelWallPercent++ }, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := measured()
			tc.mutate(&r)
			if got := r.Validate() == nil; got != tc.valid {
				t.Fatalf("valid=%t, want %t", got, tc.valid)
			}
			if got := r.IsMeasuredEvidence(); got != tc.admitted {
				t.Fatalf("admitted=%t, want %t", got, tc.admitted)
			}
			if _, err := r.ToPhaseRecorder("inkernel", "cuda", "fixture"); (err == nil) != tc.admitted {
				t.Fatalf("backend projection error=%v, admission=%t", err, tc.admitted)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSubagentTraceTimerProvenanceIsSticky(t *testing.T) {
	timer := NewSubagentTraceTimer(1, "fixture")
	for _, phase := range SubagentPhases() {
		timer.StartPhase(phase)()
	}
	r, err := timer.Finalize()
	if err != nil || !r.IsMeasuredEvidence() {
		t.Fatalf("clock receipt: %v, measured=%t", err, r.IsMeasuredEvidence())
	}
	timer.RecordUS(SubagentPhaseGPUKernel, 0)
	r, err = timer.Finalize()
	if err != nil || r.IsMeasuredEvidence() {
		t.Fatalf("manual duration was upgraded: %v", err)
	}
	timer.MarkPhaseSimulated(SubagentPhaseGPUKernel)
	timer.StartPhase(SubagentPhaseGPUKernel)()
	r, err = timer.Finalize()
	if err != nil || r.Provenance != TraceSimulated {
		t.Fatalf("simulation lost: %v, provenance=%s", err, r.Provenance)
	}
}
