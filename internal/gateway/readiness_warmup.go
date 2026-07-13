package gateway

import (
	"sync"
	"time"
)

// readiness_warmup.go is the #3051 warmup-TIMING gate on the local-model
// readiness surface — the sibling of readiness_decode.go's #4247 coherence gate.
// A GLM-5.2 serve pays a one-time ~500s backend warmup — weight load into VRAM,
// CUDA-graph capture, DeepGEMM/torch JIT compile — that finishes AFTER the HTTP
// listener binds. Today /healthz flips ok:true the instant the port binds
// (MarkReady), so the operator's FIRST real turn absorbs the entire tax: measured
// first turns ~511s / ~501s versus warm repeats ~0.62s / ~1.80s on DGX GLM-5.2,
// and a client's cold-request timeout has already canceled a legitimate warmup
// mid-compile (#3051). Liveness (the port is bound) is not readiness (the backend
// is warm).
//
// This gate lets the host declare at boot that a warmup inference is EXPECTED
// (ArmWarmupGate) and mark it done once a synthetic inference has produced its
// first token (MarkWarmupComplete). While armed-and-incomplete, /healthz reports
// ok:false with warmup_pending so a watchdog holds routing until the backend is
// warm; once complete it exposes time_to_ready_ms so the one-time tax is visible.
//
// Default-silent, exactly like the coherence probe: a serve that never arms the
// gate (proxy, mock, or a local model with no warmup step) is byte-for-byte
// unaffected — readiness stays governed only by port-bind + the existing gates.
// The POLICY lives here in the gateway; the host owns the warmup turn itself.

// warmupGate records whether a boot-time warmup inference is expected and, once
// run, how long boot->first-token took. Zero value == not armed (the gate is
// silent and /healthz is unaffected). Guarded by its own mutex so the health read
// never contends with the one-time boot transitions. Mirrors startupDecodeProbe.
type warmupGate struct {
	mu          sync.Mutex
	armed       bool
	complete    bool
	timeToReady time.Duration
}

// arm declares that this serve expects a warmup inference before it is ready.
// Idempotent; arming an already-complete gate is a no-op (a warm serve stays
// warm and never regresses to pending).
func (g *warmupGate) arm() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.complete {
		return
	}
	g.armed = true
}

// markComplete records that the warmup inference returned its first token after d
// from boot, flipping the gate to complete. The first completion wins; a later
// completion does not overwrite time_to_ready. Recording completion also arms the
// gate, so a host that calls only markComplete (an unconditional warmup) still
// exposes time_to_ready_ms and is never reported pending. A negative duration (a
// clock anomaly) is clamped to zero.
func (g *warmupGate) markComplete(d time.Duration) {
	if d < 0 {
		d = 0
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.complete {
		return
	}
	g.armed, g.complete, g.timeToReady = true, true, d
}

// pending reports whether readiness is being HELD for an incomplete warmup — true
// only when the gate was armed and warmup has not yet completed. A never-armed or
// already-complete gate returns false (readiness unaffected / already warm).
func (g *warmupGate) pending() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.armed && !g.complete
}

// ready returns the boot->first-token duration and ok=true once warmup completed,
// so /healthz can expose time_to_ready_ms. ok=false while pending or never armed.
func (g *warmupGate) ready() (d time.Duration, ok bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.complete {
		return 0, false
	}
	return g.timeToReady, true
}

// ArmWarmupGate declares that this serve expects a synthetic warmup inference to
// complete before it reports ready (#3051). Until MarkWarmupComplete is called,
// /healthz reports ok:false with warmup_pending so a watchdog holds routing work
// off a backend still paying its one-time weight-load + CUDA-graph + JIT-compile
// tax — and a client's cold-request timeout cannot cancel a legitimate warmup by
// racing an early ready mark. The host owns the warmup turn (it issues the
// inference); the gateway owns the readiness POLICY. Safe on a nil Server and for
// concurrent use. A serve that never arms the gate is byte-for-byte unaffected.
func (s *Server) ArmWarmupGate() {
	if s == nil {
		return
	}
	s.warmup.arm()
}

// MarkWarmupComplete records that the boot warmup inference returned its first
// token after d from boot, flipping /healthz to ready and exposing
// time_to_ready_ms so the one-time warmup tax is visible. Idempotent — the first
// completion wins. Safe on a nil Server and for concurrent use.
func (s *Server) MarkWarmupComplete(d time.Duration) {
	if s == nil {
		return
	}
	s.warmup.markComplete(d)
}
