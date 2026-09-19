package model

// v41_prefill_latency_gate_test.go — the #13294 follow-on witness: the V4.1
// prefill path DECIDES, before it starts a long prefill, whether the first
// token can plausibly arrive within the caller's watchdog window, and fails
// closed with a TYPED, NAMED refusal when it cannot.
//
// Why this is the next bounded step. The physical strix3 run at `fed6a6a37`
// completed the full 40-layer forward and emitted a warmup token, but the
// 30-token prefill took 492.02s (0.1 tok/s) and both real prompts were
// fail-closed at the 60s first-token watchdog with `upstream_stalled`. That
// outcome is NOT attributable at the moment it happens: the caller only learns
// "no first token within 1m0s" after burning the whole window, and the fault
// attribution landed by 08522a548 is *collected* but never *consulted* to
// decide. This leaf closes that loop: given the measured per-token fault cost
// and a declared fault bandwidth, the prefill entry point refuses UP FRONT with
// the exact projected latency and the watchdog window, instead of wedging into
// a timeout the operator has to post-mortem from a serve log.
//
// The load-bearing property is fail-closed correctness, not a new number:
//   - an UNBOUNDED model (no declared fault bandwidth) refuses NOTHING — the
//     historical prefill runs byte-for-byte, so every existing serve is
//     unchanged and no gate is silently lowered;
//   - a bench-clearable projection (projected first-token latency under the
//     window) proceeds;
//   - a projection that cannot clear the window returns the typed
//     ErrV41PrefillLatency refusal naming the projected latency and the window,
//     and moves no farther into the forward.

import (
	"errors"
	"strings"
	"testing"
)

// TestV41PrefillLatencyGateRefusesOverWatchdog is the RED->GREEN witness: a
// prefill whose projected first-token latency exceeds the declared watchdog
// window fails closed with the typed refusal, while the same model with no
// declared fault bandwidth (the default) and a projection that clears the
// window both proceed.
func TestV41PrefillLatencyGateRefusesOverWatchdog(t *testing.T) {
	// The exact physical shape: a 30-token prompt whose measured prefill fault
	// volume projects past a 60s first-token window.
	const (
		tokens          = 30
		faultedPerToken = int64(56 << 20) // ~56 MiB of routed-expert strides per prompt token
		window          = 60.0            // seconds — the gateway's buffered first-token watchdog
	)

	// --- arm 1: the default (no declared bandwidth) refuses nothing. ---
	unbounded := &Model{}
	if _, err := unbounded.v41PrefillLatencyCheck(tokens, faultedPerToken, window); err != nil {
		t.Fatalf("unbounded model refused prefill: %v, want nil (historical path unchanged)", err)
	}

	// --- arm 2: a projection that clears the window proceeds. ---
	// 1 GiB/s of fault bandwidth over 56 MiB/token x 30 tokens = ~1.7s << 60s.
	fast := &Model{}
	fast.v41SetPrefillFaultBandwidth(1 << 30)
	projected, err := fast.v41PrefillLatencyCheck(tokens, faultedPerToken, window)
	if err != nil {
		t.Fatalf("fast model refused a clearable prefill: %v", err)
	}
	if projected <= 0 || projected >= window {
		t.Fatalf("fast projection = %v s, want (0, %v)", projected, window)
	}

	// --- arm 3: the physical projection (disk-tier bandwidth) fails closed. ---
	// The observed physical run ingested 30 tokens in 492 s (0.1 tok/s) while
	// faulting ~56 MiB of routed-expert strides per token. A declared bandwidth
	// in that regime — 4 MiB/s — projects 30 x 56 MiB / 4 MiB/s = 420 s, far
	// beyond the 60 s window: the gate must refuse before starting.
	slow := &Model{}
	slow.v41SetPrefillFaultBandwidth(4 << 20) // 4 MiB/s → 30 x 56 MiB / 4 MiB/s = 420s
	_, err = slow.v41PrefillLatencyCheck(tokens, faultedPerToken, window)
	if err == nil {
		t.Fatalf("slow model proceeded on a prefill that cannot clear the %.0fs window", window)
	}
	if !errors.Is(err, ErrV41PrefillLatency) {
		t.Fatalf("slow model error = %v, want errors.Is(err, ErrV41PrefillLatency)", err)
	}
	if !strings.Contains(err.Error(), "first-token") {
		t.Fatalf("refusal %q does not name the first-token window", err.Error())
	}

	// The refusal must be actionable: it names the projected latency and window.
	var refusal *V41PrefillLatencyError
	if !errors.As(err, &refusal) {
		t.Fatalf("error %v is not a *V41PrefillLatencyError", err)
	}
	if refusal.ProjectedSeconds <= window {
		t.Fatalf("refusal projects %v s, must exceed the %.0fs window it refused on",
			refusal.ProjectedSeconds, window)
	}
	if refusal.WindowSeconds != window {
		t.Fatalf("refusal window = %v, want %v", refusal.WindowSeconds, window)
	}
}

// TestV41PrefillLatencyGateBoundaryIsInclusive pins the exact equality edge the
// gate refuses on: a projection that lands exactly on the window is NOT refused
// (the boundary is inclusive), and one byte/second slower IS. This is the sharp
// edge an off-by-one would silently flip.
func TestV41PrefillLatencyGateBoundaryIsInclusive(t *testing.T) {
	// 30 tokens x 30 MiB = 900 MiB. At 15 MiB/s that projects exactly 60 s.
	const (
		tokens          = 30
		faultedPerToken = int64(30 << 20)
		window          = 60.0
	)
	onBoundary := &Model{}
	onBoundary.v41SetPrefillFaultBandwidth(15 << 20) // 900 MiB / 15 MiB/s = 60s exactly
	if _, err := onBoundary.v41PrefillLatencyCheck(tokens, faultedPerToken, window); err != nil {
		t.Fatalf("projection exactly on the window was refused: %v, want nil (boundary inclusive)", err)
	}

	overBoundary := &Model{}
	overBoundary.v41SetPrefillFaultBandwidth((15 << 20) - 1) // one B/s slower → just over
	if _, err := overBoundary.v41PrefillLatencyCheck(tokens, faultedPerToken, window); err == nil {
		t.Fatalf("projection just over the window proceeded, want the typed refusal")
	}
}

// TestV41PrefillFirstTokenAdmittedNeedsMeasurement is the wiring witness: the
// prefill ENTRY-POINT gate refuses only once a real per-token fault volume has
// been MEASURED. A cold model (declared bandwidth + window, but no prior pass)
// proceeds — there is nothing honest to project from — and a model with a
// measured prefill attribution refuses.
func TestV41PrefillFirstTokenAdmittedNeedsMeasurement(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K) // tier-only: every read faults
	m.v41SetPrefillFaultBandwidth(1 << 20)        // 1 MiB/s: pessimistic
	m.v41SetPrefillFirstTokenWindow(60)

	// Cold: no phase ever ran, so the measured faults/token is 0 and the gate is
	// inert — the historical prefill must proceed.
	if err := m.v41PrefillFirstTokenAdmitted(30); err != nil {
		t.Fatalf("cold model refused prefill: %v, want nil (no measurement to project from)", err)
	}

	// Run one real prefill to MEASURE the fixture's per-token fault volume.
	m.v41SetExpertFaultPhase(V41PhasePrefill)
	if _, err := m.forwardV41([]int{1, 3}, nil); err != nil {
		t.Fatalf("fixture prefill error = %v", err)
	}
	m.v41NoteExpertFaultToken(2)
	m.v41SetExpertFaultPhase(V41PhaseUnknown)

	got := m.V41ExpertFaultAttribution()
	if got.Prefill.FaultedBytesPerToken <= 0 {
		t.Fatalf("measured FaultedBytesPerToken = %v, want > 0 after a real prefill",
			got.Prefill.FaultedBytesPerToken)
	}

	// Warm: the measured volume at 1 MiB/s over a long prompt projects past 60s,
	// so the gate must now refuse with the typed error.
	err := m.v41PrefillFirstTokenAdmitted(1 << 20)
	if err == nil {
		t.Fatalf("measured model proceeded on a prefill that cannot clear the window")
	}
	if !errors.Is(err, ErrV41PrefillLatency) {
		t.Fatalf("wired gate error = %v, want errors.Is(err, ErrV41PrefillLatency)", err)
	}
}

// TestV41PrefillFirstTokenAdmittedDefaultInert pins the default-off contract:
// with neither bandwidth nor window declared, a measured model still refuses
// nothing, so every existing serve is byte-for-byte unchanged.
func TestV41PrefillFirstTokenAdmittedDefaultInert(t *testing.T) {
	m := &Model{}
	if err := m.v41PrefillFirstTokenAdmitted(1 << 20); err != nil {
		t.Fatalf("default model refused prefill: %v, want nil", err)
	}
	// A declared bandwidth without a window is still inert (no window to miss).
	m.v41SetPrefillFaultBandwidth(1 << 20)
	if err := m.v41PrefillFirstTokenAdmitted(1 << 20); err != nil {
		t.Fatalf("bandwidth-only model refused prefill: %v, want nil (no window declared)", err)
	}
}

// TestV41PrefillLatencyGateZeroVolumeIsSafe pins the degenerate contract: a
// prefill that faults nothing (all residents, or an empty prompt) can never be
// refused by the gate, regardless of the declared bandwidth.
func TestV41PrefillLatencyGateZeroVolumeIsSafe(t *testing.T) {
	m := &Model{}
	m.v41SetPrefillFaultBandwidth(1) // the most pessimistic legal bandwidth
	for _, tokens := range []int{0, 1, 30} {
		if _, err := m.v41PrefillLatencyCheck(tokens, 0, 60); err != nil {
			t.Fatalf("tokens=%d faultedPerToken=0 refused: %v, want nil", tokens, err)
		}
	}
}

// TestV41PrefillEntryPointRefusesOverWatchdog is the END-TO-END wiring witness:
// it drives the real Session.Prefill entry point (kv.go's IsDeepSeekV41 branch ->
// prefillV41) and proves the gate fires where it actually matters — the refusal
// is a PANIC carrying ErrV41PrefillLatency, raised BEFORE the forward runs.
//
// Every other test in this file calls v41PrefillFirstTokenAdmitted directly, so
// they witness the PREDICATE but not the WIRING. Delete the two-line call site in
// prefillV41 and those tests still pass; this one cannot. That is the whole point:
// the leaf's claim is "the prefill entry point fails closed up front", and only a
// test that enters through Session.Prefill can hold it to that.
//
// The physical shape is reproduced end to end: a tier-only model (every
// routed-expert read faults, so a real pass MEASURES a real per-token volume),
// then a declared 1 MiB/s bandwidth and a 60 s window — the strix3 0.1 tok/s
// regime — under which the next prompt must be refused rather than wedge.
func TestV41PrefillEntryPointRefusesOverWatchdog(t *testing.T) {
	m := v41TierOnlyModel(t, ExpertCheckpointQ4K)
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("tier-only admission error = %v, want nil", err)
	}
	s := &Session{M: m}

	// Arm 1 — DEFAULT INERT, through the real entry point. No bandwidth and no
	// window declared, so the warmup prefill below must run byte-for-byte and
	// MEASURE the fixture's per-token fault volume.
	s.Prefill([]int{1, 3})
	measured := m.V41ExpertFaultAttribution().Prefill.FaultedBytesPerToken
	if measured <= 0 {
		t.Fatalf("warmup prefill measured FaultedBytesPerToken = %v, want > 0 "+
			"(the tier-only fixture faults on every routed-expert read)", measured)
	}

	// Declare the physical regime: 1 MiB/s of fault bandwidth under a 60 s
	// first-token watchdog. The measured volume over a long prompt cannot clear it.
	m.v41SetPrefillFaultBandwidth(1 << 20)
	m.v41SetPrefillFirstTokenWindow(60)

	// Arm 2 — the wired gate fails closed with the TYPED sentinel. Session.Prefill
	// reaches prefillV41, which panics with the gate's error before any forward
	// math, mirroring how v41ForwardAdmitted refuses a missing stage.
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("Session.Prefill proceeded on a prompt that cannot clear the %.0fs window; "+
				"want the typed ErrV41PrefillLatency refusal", 60.0)
		}
		err, ok := r.(error)
		if !ok {
			t.Fatalf("refusal panic value = %#v (%T), want an error", r, r)
		}
		if !errors.Is(err, ErrV41PrefillLatency) {
			t.Fatalf("refusal = %v, want errors.Is(err, ErrV41PrefillLatency)", err)
		}
		if !strings.Contains(err.Error(), "first-token") {
			t.Fatalf("refusal %q does not name the first-token window", err.Error())
		}
	}()
	s.Prefill(make([]int, 1<<20)) // enough tokens to project far past 60 s at 1 MiB/s
}
