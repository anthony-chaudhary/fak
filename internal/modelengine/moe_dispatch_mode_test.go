package modelengine

import (
	"errors"
	"testing"
)

// TestMoEDispatchMode pins the closed contract of the DeepEP two-mode MoE
// dispatch split: mode selection by phase (prefill -> normal, decode ->
// low-latency), the per-mode buffer accounting (normal = NVLink intranode +
// RDMA internode; low-latency = a single RDMA buffer used for both hops), the
// no-op when a run of phases is homogeneous, and the fail-closed refusal on an
// unknown phase. The expected byte counts mirror the upstream DeepEP formulas
// (Config::get_nvl_buffer_size_hint, Config::get_rdma_buffer_size_hint, and
// get_low_latency_rdma_size_hint) at the pinned commit, so a change to the
// accounting that diverges from upstream fails here.
func TestMoEDispatchMode(t *testing.T) {
	t.Run("mode-selection-by-phase", func(t *testing.T) {
		cases := []struct {
			phase MoEDispatchPhase
			want  MoEDispatchMode
		}{
			{phase: MoEDispatchPhasePrefill, want: MoEDispatchModeNormal},
			{phase: MoEDispatchPhaseDecode, want: MoEDispatchModeLowLatency},
		}
		for _, tc := range cases {
			got, err := MoEDispatchModeForPhase(tc.phase)
			if err != nil {
				t.Fatalf("MoEDispatchModeForPhase(%v) unexpected error: %v", tc.phase, err)
			}
			if got != tc.want {
				t.Fatalf("MoEDispatchModeForPhase(%v) = %v, want %v", tc.phase, got, tc.want)
			}
		}
	})

	t.Run("per-phase-buffer-accounting", func(t *testing.T) {
		// A 16-rank EP group (two 8-rank NVLink domains) with 132 SMs, a
		// 14336-byte hidden width, and a 4096-token per-channel chunk cap.
		const (
			numRanks = 16
			numSMS   = 132
			hidden   = 14336
			chunk    = 4096
		)

		// Prefill -> normal mode: an intranode NVLink buffer plus an
		// internode RDMA buffer.
		prefill, err := MoEDispatchPlanForPhase(MoEDispatchPhasePrefill, numRanks, numSMS, hidden, 4, 256, chunk, chunk)
		if err != nil {
			t.Fatalf("MoEDispatchPlanForPhase(prefill) unexpected error: %v", err)
		}
		if prefill.Mode != MoEDispatchModeNormal {
			t.Fatalf("prefill plan mode = %v, want normal", prefill.Mode)
		}
		if prefill.Accounting.NVLinkBytes != 34360801792 {
			t.Fatalf("prefill NVLinkBytes = %d, want 34360801792", prefill.Accounting.NVLinkBytes)
		}
		if prefill.Accounting.RDMABytes != 17197714048 {
			t.Fatalf("prefill RDMABytes = %d, want 17197714048", prefill.Accounting.RDMABytes)
		}
		if prefill.Accounting.TotalBytes != prefill.Accounting.NVLinkBytes+prefill.Accounting.RDMABytes {
			t.Fatalf("prefill TotalBytes = %d, want NVLink+RDMA = %d", prefill.Accounting.TotalBytes, prefill.Accounting.NVLinkBytes+prefill.Accounting.RDMABytes)
		}

		// Decode -> low-latency mode: a single RDMA buffer used for both the
		// intra- and internode hops, so no separate NVLink buffer.
		decode, err := MoEDispatchPlanForPhase(MoEDispatchPhaseDecode, numRanks, numSMS, hidden, 4, 256, chunk, chunk)
		if err != nil {
			t.Fatalf("MoEDispatchPlanForPhase(decode) unexpected error: %v", err)
		}
		if decode.Mode != MoEDispatchModeLowLatency {
			t.Fatalf("decode plan mode = %v, want low-latency", decode.Mode)
		}
		if decode.Accounting.NVLinkBytes != 0 {
			t.Fatalf("decode NVLinkBytes = %d, want 0 (low-latency uses RDMA for both hops)", decode.Accounting.NVLinkBytes)
		}
		if decode.Accounting.RDMABytes != 119277568 {
			t.Fatalf("decode RDMABytes = %d, want 119277568", decode.Accounting.RDMABytes)
		}
		if decode.Accounting.TotalBytes != decode.Accounting.RDMABytes {
			t.Fatalf("decode TotalBytes = %d, want RDMA = %d", decode.Accounting.TotalBytes, decode.Accounting.RDMABytes)
		}

		// The two modes must account for different buffers: the prefill
		// normal-mode footprint is not the decode low-latency footprint.
		if prefill.Accounting.TotalBytes == decode.Accounting.TotalBytes {
			t.Fatalf("prefill and decode both total %d bytes; the two modes must differ", prefill.Accounting.TotalBytes)
		}
	})

	t.Run("low-latency-rdma-hint-scales-with-dispatch-cap", func(t *testing.T) {
		// The low-latency RDMA buffer grows with the per-rank dispatch cap:
		// a 4-token cap is exactly 4x a 1-token cap at the same hidden/rank/
		// expert shape, and a 0 cap is a 0-byte buffer.
		const (
			hidden  = 7168
			ranks   = 8
			experts = 256
		)
		if got := MoEDispatchLowLatencyRDMABytesHint(4, hidden, ranks, experts); got != 59639808 {
			t.Fatalf("low-latency hint(4) = %d, want 59639808", got)
		}
		if got := MoEDispatchLowLatencyRDMABytesHint(1, hidden, ranks, experts); got != 14911488 {
			t.Fatalf("low-latency hint(1) = %d, want 14911488", got)
		}
		if got := MoEDispatchLowLatencyRDMABytesHint(0, hidden, ranks, experts); got != 0 {
			t.Fatalf("low-latency hint(0) = %d, want 0", got)
		}
	})

	t.Run("normal-rdma-hint-zero-intranode-only", func(t *testing.T) {
		// An EP group at or under the NVLink domain width has no internode
		// hop, so the normal-mode RDMA buffer is 0 bytes.
		if got := MoEDispatchNormalRDMABytesHint(8, 132, 14336, 4096); got != 0 {
			t.Fatalf("normal RDMA hint(8 ranks) = %d, want 0 (intranode-only)", got)
		}
		if got := MoEDispatchNormalNVLinkBytesHint(8, 132, 14336, 4096); got != 34360797568 {
			t.Fatalf("normal NVLink hint(8 ranks) = %d, want 34360797568", got)
		}
	})

	t.Run("no-op-when-phases-homogeneous", func(t *testing.T) {
		// A run of identical phases is already covered by one mode, so the
		// planning step is a no-op: it returns that mode's plan with NoOp set.
		cases := []struct {
			name   string
			phases []MoEDispatchPhase
			want   MoEDispatchMode
		}{
			{name: "all-prefill", phases: []MoEDispatchPhase{MoEDispatchPhasePrefill, MoEDispatchPhasePrefill}, want: MoEDispatchModeNormal},
			{name: "all-decode", phases: []MoEDispatchPhase{MoEDispatchPhaseDecode, MoEDispatchPhaseDecode}, want: MoEDispatchModeLowLatency},
			{name: "single-phase", phases: []MoEDispatchPhase{MoEDispatchPhaseDecode}, want: MoEDispatchModeLowLatency},
		}
		for _, tc := range cases {
			plan, err := MoEDispatchPlanForPhases(tc.phases, 16, 132, 14336, 4, 256, 4096, 4096)
			if err != nil {
				t.Fatalf("%s: MoEDispatchPlanForPhases unexpected error: %v", tc.name, err)
			}
			if !plan.NoOp {
				t.Fatalf("%s: homogeneous run must be a no-op, got NoOp=false", tc.name)
			}
			if plan.Mode != tc.want {
				t.Fatalf("%s: no-op plan mode = %v, want %v", tc.name, plan.Mode, tc.want)
			}
		}

		// An empty run is a no-op with no mode.
		plan, err := MoEDispatchPlanForPhases(nil, 16, 132, 14336, 4, 256, 4096, 4096)
		if err != nil {
			t.Fatalf("empty run: MoEDispatchPlanForPhases unexpected error: %v", err)
		}
		if !plan.NoOp {
			t.Fatalf("empty run must be a no-op, got NoOp=false")
		}
	})

	t.Run("mixed-run-is-not-a-no-op", func(t *testing.T) {
		// A mixed prefill/decode run is not homogeneous: the step returns the
		// first phase's plan with NoOp clear, signaling the caller to split
		// the run by phase and plan each homogeneous segment separately.
		plan, err := MoEDispatchPlanForPhases(
			[]MoEDispatchPhase{MoEDispatchPhasePrefill, MoEDispatchPhaseDecode},
			16, 132, 14336, 4, 256, 4096, 4096)
		if err != nil {
			t.Fatalf("mixed run: MoEDispatchPlanForPhases unexpected error: %v", err)
		}
		if plan.NoOp {
			t.Fatalf("mixed run must not be a no-op, got NoOp=true")
		}
		if plan.Mode != MoEDispatchModeNormal {
			t.Fatalf("mixed run plan mode = %v, want normal (first phase)", plan.Mode)
		}
	})

	t.Run("fail-closed-on-unknown-phase", func(t *testing.T) {
		// A phase that is neither prefill nor decode is refused, and the
		// refusal carries the sentinel so callers can use errors.Is.
		_, err := MoEDispatchModeForPhase(MoEDispatchPhase(99))
		if !errors.Is(err, ErrMoEDispatchPhaseUnknown) {
			t.Fatalf("MoEDispatchModeForPhase(unknown): err = %v, want errors.Is %v", err, ErrMoEDispatchPhaseUnknown)
		}

		// The same refusal propagates through the single-phase planner.
		_, err = MoEDispatchPlanForPhase(MoEDispatchPhase(99), 16, 132, 14336, 4, 256, 4096, 4096)
		if !errors.Is(err, ErrMoEDispatchPhaseUnknown) {
			t.Fatalf("MoEDispatchPlanForPhase(unknown): err = %v, want errors.Is %v", err, ErrMoEDispatchPhaseUnknown)
		}

		// And through the multi-phase planner, even when the bad phase is not
		// the first one.
		_, err = MoEDispatchPlanForPhases(
			[]MoEDispatchPhase{MoEDispatchPhasePrefill, MoEDispatchPhase(99)},
			16, 132, 14336, 4, 256, 4096, 4096)
		if !errors.Is(err, ErrMoEDispatchPhaseUnknown) {
			t.Fatalf("MoEDispatchPlanForPhases(unknown later): err = %v, want errors.Is %v", err, ErrMoEDispatchPhaseUnknown)
		}
	})

	t.Run("default-off-keeps-single-mode", func(t *testing.T) {
		// With the split off (the default), the planning step is a no-op
		// regardless of the phases, so the MoE dispatch path keeps its single
		// historical mode.
		t.Setenv(envMoEDispatchModeSplit, "")
		plan, err := MoEDispatchPlanStep(
			[]MoEDispatchPhase{MoEDispatchPhasePrefill, MoEDispatchPhaseDecode},
			16, 132, 14336, 4, 256, 4096, 4096)
		if err != nil {
			t.Fatalf("MoEDispatchPlanStep (split off) unexpected error: %v", err)
		}
		if !plan.NoOp {
			t.Fatalf("split off: plan must be a no-op, got NoOp=false")
		}

		// With the split on, the same mixed run is not a no-op.
		t.Setenv(envMoEDispatchModeSplit, "1")
		plan, err = MoEDispatchPlanStep(
			[]MoEDispatchPhase{MoEDispatchPhasePrefill, MoEDispatchPhaseDecode},
			16, 132, 14336, 4, 256, 4096, 4096)
		if err != nil {
			t.Fatalf("MoEDispatchPlanStep (split on) unexpected error: %v", err)
		}
		if plan.NoOp {
			t.Fatalf("split on: mixed run must not be a no-op, got NoOp=true")
		}
	})
}
