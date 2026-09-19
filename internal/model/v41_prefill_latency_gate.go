package model

// v41_prefill_latency_gate.go — the V4.1 prefill first-token feasibility gate
// (fak#13294 follow-on).
//
// The physical strix3 rung at `fed6a6a37` completed the full native 40-layer
// V4.1 forward and emitted a warmup token, but its 30-token prefill took
// 492.02 s (0.1 tok/s) and both real prompts were fail-closed at the gateway's
// 60 s buffered first-token watchdog with `upstream_stalled`. The cost was
// *observed* (the phase-scoped fault attribution landed in `08522a548`) but
// never *consulted*: the caller learned only "no first token within 1m0s" after
// the whole window had already been burned, leaving an operator to post-mortem
// a serve log.
//
// This seam closes that loop. When a model declares a routed-expert fault
// bandwidth, the prefill entry point can project the first-token latency from
// the prompt length and the measured per-token faulted-expert volume, and fail
// closed BEFORE entering a prefill that cannot clear the watchdog — naming the
// projected latency and the window rather than emitting a bare timeout later.
//
// Default OFF, exactly the historical behavior: a model with no declared
// bandwidth (v41PrefillFaultBandwidth == 0) refuses nothing and the prefill runs
// byte-for-byte as before. The gate never lowers a roofline threshold, never
// substitutes a checkpoint, and never fabricates a denominator (a zero fault
// volume can never be refused, however pessimistic the declared bandwidth).

import (
	"errors"
	"fmt"
)

// ErrV41PrefillLatency reports that the V4.1 prefill's projected first-token
// latency exceeds the caller's first-token watchdog window: proceeding would
// wedge the request into an upstream_stalled timeout instead of producing a
// token. It is the throughput analogue of the budget refusal
// (ErrExpertCheckpointBudget): a bounded, named, pre-emptive "no" rather than an
// unbounded wait.
var ErrV41PrefillLatency = errors.New("model: V4.1 prefill cannot reach a first token within the declared watchdog window")

// V41PrefillLatencyError is the typed, actionable refusal: the prompt-token
// count and per-token faulted-expert volume the projection rests on, the
// projected first-token latency, and the watchdog window it failed to clear.
type V41PrefillLatencyError struct {
	Tokens           int     // prompt tokens the prefill was asked to ingest
	FaultedPerToken  int64   // measured routed-expert stride bytes one prompt token faults
	FaultBandwidth   int64   // declared fault bandwidth, bytes/second
	ProjectedSeconds float64 // projected first-token latency, seconds
	WindowSeconds    float64 // the first-token watchdog window the projection missed
}

func (e *V41PrefillLatencyError) Error() string {
	return fmt.Sprintf("%v: %d prompt tokens at %d faulted bytes/token over %d B/s projects %.1fs "+
		"first-token latency, beyond the %.1fs window",
		ErrV41PrefillLatency, e.Tokens, e.FaultedPerToken, e.FaultBandwidth,
		e.ProjectedSeconds, e.WindowSeconds)
}

// Unwrap exposes the sentinel so errors.Is(err, ErrV41PrefillLatency) holds
// while the caller can still read the projection operands.
func (e *V41PrefillLatencyError) Unwrap() error { return ErrV41PrefillLatency }

// v41PrefillLatencyCheck projects the first-token latency of a prefill that
// ingests `tokens` prompt tokens, each faulting `faultedPerToken` routed-expert
// stride bytes, at the model's declared fault bandwidth. It returns
// (projectedSeconds, nil) when the model declares no bandwidth (the default —
// nothing to project against, historical path preserved), when the fault volume
// is zero (nothing to fault), or when the projection clears the window. It
// returns (projectedSeconds, *V41PrefillLatencyError) when the projection
// exceeds the window.
//
// The projection is deliberately a LOWER BOUND on the real first-token latency:
// it charges only the routed-expert fault IO the prompt provokes and assumes
// every byte moves at the declared bandwidth. Non-expert work (attention,
// dense projections, dequant) only adds time, so a projection that already
// exceeds the window is a safe refusal. A projection that clears the window is
// not a promise — it is the absence of a proven impossibility, which is why
// the check never refuses on it.
func (m *Model) v41PrefillLatencyCheck(tokens int, faultedPerToken int64, windowSeconds float64) (float64, error) {
	if m == nil || m.v41PrefillFaultBandwidth <= 0 || faultedPerToken <= 0 || tokens <= 0 {
		return 0, nil
	}
	volume := float64(faultedPerToken) * float64(tokens)
	projected := volume / float64(m.v41PrefillFaultBandwidth)
	if windowSeconds <= 0 || projected <= windowSeconds {
		return projected, nil
	}
	return projected, &V41PrefillLatencyError{
		Tokens:           tokens,
		FaultedPerToken:  faultedPerToken,
		FaultBandwidth:   m.v41PrefillFaultBandwidth,
		ProjectedSeconds: projected,
		WindowSeconds:    windowSeconds,
	}
}

// v41SetPrefillFaultBandwidth declares the routed-expert fault bandwidth (bytes
// per second) the prefill gate may project against. Zero (the default) leaves
// the gate inert and the prefill on the historical path. A negative value is
// clamped to 0 rather than silently accepted as an impossible ceiling.
func (m *Model) v41SetPrefillFaultBandwidth(bytesPerSecond int64) {
	if m == nil {
		return
	}
	if bytesPerSecond < 0 {
		bytesPerSecond = 0
	}
	m.v41PrefillFaultBandwidth = bytesPerSecond
}

// v41SetPrefillFirstTokenWindow declares the first-token watchdog window
// (seconds) the prefill gate must clear. Zero (the default) means no window was
// declared, so the gate never refuses. A negative value is clamped to 0.
func (m *Model) v41SetPrefillFirstTokenWindow(seconds float64) {
	if m == nil {
		return
	}
	if seconds < 0 {
		seconds = 0
	}
	m.v41PrefillFirstTokenWindow = seconds
}

// v41PrefillFirstTokenAdmitted is the prefill entry-point gate: it projects the
// first-token latency of a `tokens`-token prefill from the model's last MEASURED
// prefill fault volume (the phase-scoped attribution from a prior pass) and the
// declared fault bandwidth, returning the typed ErrV41PrefillLatency refusal
// when the projection cannot clear the declared watchdog window.
//
// It is inert whenever any operand is absent — no declared bandwidth, no
// declared window, no prior measurement (a cold model has nothing to project
// from), or a zero measured fault volume — so the default path is byte-for-byte
// the historical prefill. This is deliberate: the projection MUST rest on a real
// observed fault volume, never a guessed one (no fabricated denominator).
func (m *Model) v41PrefillFirstTokenAdmitted(tokens int) error {
	if m == nil {
		return nil
	}
	if m.v41PrefillFaultBandwidth <= 0 || m.v41PrefillFirstTokenWindow <= 0 {
		return nil
	}
	perToken := m.V41ExpertFaultAttribution().Prefill.FaultedBytesPerToken
	if perToken <= 0 {
		return nil // cold model: no measured fault volume to project from
	}
	_, err := m.v41PrefillLatencyCheck(tokens, int64(perToken), m.v41PrefillFirstTokenWindow)
	return err
}
