package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/kvbudget"
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

// ErrWarmupTimeout is returned by RunWarmup when the backend's synthetic warmup
// inference does not return its first token within the configured boot/warmup
// ceiling. It is a TYPED, bounded failure — the fix for the strix1 hang where a
// non-returning forward (a stale SPIR-V bundle makes the CPU-side q4k/q8 path spin
// before any device dispatch, in a user-space tight loop that ignores ctx) held
// readiness warmup_pending FOREVER with no journal line. RunWarmup calls
// planner.Complete on a worker goroutine and waits with the ceiling, so a
// non-returning backend is reported instead of holding readiness unbounded. The
// gate stays PENDING on a timeout (a hung backend is not warm) exactly as it does
// on an error/cancellation, so a failed boot never advertises readiness. The
// synthetic turn may still be spinning on the leaked goroutine; this is the best a
// bounded wait can do without cooperative cancellation from the backend.
var ErrWarmupTimeout = errors.New("backend startup warmup timed out")

// DefaultWarmupCeiling is the boot/warmup ceiling applied to a synthetic warmup
// inference when the host sets no override. It is generous by design — a local
// 27B serve can legitimately pay a multi-minute first-decode tax (weight load +
// CUDA-graph capture + JIT compile) — while still bounding the pathological
// non-returning case. Override with FAK_WARMUP_CEILING_S.
const DefaultWarmupCeiling = 10 * time.Minute

// warmupCeilingEnv overrides DefaultWarmupCeiling with an integer-seconds value.
// 0 (or unset/unparseable) selects the default. A negative value disables the
// bound entirely (an explicit opt-out for a host that wants an unbounded hold).
const warmupCeilingEnv = "FAK_WARMUP_CEILING_S"

// warmupCeiling resolves the effective boot/warmup ceiling from the environment,
// mirroring durEnv's integer-seconds convention: unset/unparseable keeps def; a
// non-negative integer wins (0 => def, never an accidental instant timeout); a
// negative integer disables the bound (returns 0, meaning "wait indefinitely").
func warmupCeiling(getenv func(string) string) time.Duration {
	if getenv == nil {
		getenv = os.Getenv
	}
	v := getenv(warmupCeilingEnv)
	if v == "" {
		return DefaultWarmupCeiling
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return DefaultWarmupCeiling
	}
	if n < 0 {
		return 0
	}
	if n == 0 {
		return DefaultWarmupCeiling
	}
	return time.Duration(n) * time.Second
}

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
	if s.logf != nil {
		addr := s.BoundAddr()
		if addr != "" {
			s.logf("[READY] server warmup completed in %v — gateway is ready on http://%s", d.Round(time.Millisecond), addr)
		} else {
			s.logf("[READY] server warmup completed in %v — gateway is ready", d.Round(time.Millisecond))
		}
	}
}

// RunWarmup issues one synthetic completion through the server's chat planner and
// marks the warmup gate complete with the boot->first-token elapsed, returning it.
// It is the execution half of the #3051/#3083 warm-start: the serve path arms the
// gate synchronously (ArmWarmupGate) before binding the listener — so /healthz
// reports warmup_pending from the very first probe — then runs THIS in a goroutine
// alongside ListenAndServe, and the operator's first real turn is warm-path, not a
// ~500s cold stall. It calls s.planner.Complete DIRECTLY, deliberately bypassing
// the served-turn admission/session/metrics bookkeeping (s.complete): a synthetic
// warm turn must not debit a budget or mutate session state. The prompt is a fixed,
// code-authored one-token turn — the tax is paid on the FIRST decode, so a single
// token is enough to force weight load + CUDA-graph capture + JIT compile while
// wasting no steady-state compute. Safe on a nil Server or nil planner (a serve
// with no local backend to warm): it releases any armed gate so readiness is never
// stuck pending.
//
// Completion is CONTINGENT on the warmup actually succeeding (#13353): a planner
// error or a cancelled context leaves an armed gate PENDING and returns the
// original error, so a failed boot cannot advertise warm readiness. The gate stays
// armed — and readiness stays held — until a LATER RunWarmup succeeds; a
// subsequent successful call completes it exactly once. This keeps cancellation,
// backend failure and a successful retry distinguishable without adding a retry
// loop. The nil-planner case is explicit and separate: there is no backend to
// warm, so the gate is released (as before) rather than held forever.
//
// The wait is BOUNDED (#13500): planner.Complete runs on a worker goroutine and
// RunWarmup waits at most the boot/warmup ceiling (warmupCeiling, default
// DefaultWarmupCeiling, override FAK_WARMUP_CEILING_S). A backend whose forward
// never returns — the live strix1 stale-SPIR-V hang, a user-space tight loop that
// ignores ctx — can no longer hold readiness warmup_pending forever with no
// journal line. On expiry RunWarmup returns ErrWarmupTimeout (a typed, bounded
// failure a watchdog/operator can branch on) and leaves the gate PENDING (a hung
// backend is not warm). The synthetic turn's goroutine is abandoned; only
// cooperative cancellation from the backend could reclaim it, which is #993's
// shader-attestation territory, not this readiness policy's.
func (s *Server) RunWarmup(ctx context.Context) (time.Duration, error) {
	if s == nil {
		return 0, nil
	}
	if s.planner == nil {
		s.MarkWarmupComplete(0)
		return 0, nil
	}
	msgs := []agent.Message{{Role: agent.RoleUser, Content: "warmup"}}
	start := time.Now()

	ceiling := warmupCeiling(os.Getenv)
	done := make(chan struct{})
	var compErr error
	go func() {
		_, compErr = s.planner.Complete(ctx, msgs, nil, agent.WithMaxTokens(1))
		close(done)
	}()
	if ceiling > 0 {
		timer := time.NewTimer(ceiling)
		defer timer.Stop()
		select {
		case <-done:
		case <-ctx.Done():
			// A cancelled/expired caller context surfaces as before, leaving the gate
			// PENDING; the worker goroutine is abandoned.
			return time.Since(start), ctx.Err()
		case <-timer.C:
			return time.Since(start), fmt.Errorf("%w after %s (ceiling %s); the backend forward never returned its first token — readiness remains pending",
				ErrWarmupTimeout, time.Since(start).Round(time.Millisecond), ceiling)
		}
	} else {
		// A negative FAK_WARMUP_CEILING_S disables the bound: wait indefinitely, but
		// still surface a caller cancellation.
		select {
		case <-done:
		case <-ctx.Done():
			return time.Since(start), ctx.Err()
		}
	}

	d := time.Since(start)
	if compErr != nil {
		return d, compErr
	}
	s.MarkWarmupComplete(d)

	if rep, ok := s.planner.(agent.KVMemoryReporter); ok {
		st := rep.KVMemoryStats()
		if st.BytesPerToken > 0 {
			usable := st.FitBudgetBytes
			if usable <= 0 {
				usable = st.CapacityFreeBytes
			}
			if usable <= 0 {
				usable = st.CapacityTotalBytes
			}
			if usable > 0 {
				s.SetWarmupCapacity(kvbudget.WarmupCapacity{
					UsableBytes:   usable,
					BytesPerToken: st.BytesPerToken,
				})
			}
		}
	} else if rep, ok := s.planner.(WarmupCapacityReporter); ok {
		if c, okCap := rep.WarmupCapacity(); okCap {
			s.SetWarmupCapacity(c)
		}
	} else if rep, ok := s.planner.(interface {
		WarmupCapacity() kvbudget.WarmupCapacity
	}); ok {
		s.SetWarmupCapacity(rep.WarmupCapacity())
	} else if rep, ok := s.planner.(interface {
		WarmupBlockCapacity() (kvbudget.WarmupBlockCapacity, bool)
	}); ok {
		if bc, okBC := rep.WarmupBlockCapacity(); okBC {
			s.SetWarmupBlockCapacity(bc)
		}
	}
	return d, compErr
}

// checkWarmupPending checks if the server warmup gate is still armed and incomplete.
// If warmup is pending, it writes HTTP 503 (StatusServiceUnavailable) with a Retry-After: 1
// header and a typed "warmup_pending" error, returning true.
//
// A CONFIGURED agent-warm profile (#13333) is held the same way: while its cache is
// pending or degraded, served inference answers 503 with a typed "agent_warm_pending"
// error so a first turn cannot land on a cold prefix the serve promised to warm. An
// unconfigured or unsupported cache never blocks — there is no prefix contract to
// enforce, so the serve behaves exactly as before.
func (s *Server) checkWarmupPending(w http.ResponseWriter) bool {
	if s == nil {
		return false
	}
	if blocked, _, reason := s.agentWarm.admit(); blocked {
		w.Header().Set("Retry-After", "1")
		msg := "server agent cache is warming up; please retry shortly"
		if reason != "" {
			msg = "server agent cache is not warm (" + reason + "); please retry shortly"
		}
		writeErrCode(w, http.StatusServiceUnavailable, "agent_warm_pending", msg)
		return true
	}
	if !s.warmup.pending() {
		return false
	}
	w.Header().Set("Retry-After", "1")
	writeErrCode(w, http.StatusServiceUnavailable, "warmup_pending", "server is warming up; please retry shortly")
	return true
}

// ---------------------------------------------------------------------------
// CW-07 (#13333): agent KV-cache warm readiness.
//
// The #3051 warmup gate above answers "is the backend LOADED?" — a synthetic
// inference returned its first token, so weight load + CUDA-graph capture + JIT
// compile are done. It does NOT answer "is a warm PREFIX resident and reusable?":
// its complete bit is a timing observation, not cache-residency evidence. A
// gateway can be fully warmed and still send the operator's first real turn into a
// full prefix prefill.
//
// This gate is the missing second half. The host installs an agent warm profile
// (the descriptor-bearing inputs) via SetAgentWarmProfile, then runs the native
// warm (RunAgentWarmup) through the narrow AgentWarmWarmer interface. Readiness
// for a configured profile is admitted ONLY against a LIVE receipt: the warm
// reported Ready, restored a prefix reaching the descriptor's stable boundary, and
// (when a finite-residency claim is configured) holds a Live claim. A backend
// warmup completion cannot satisfy it, and a failed/expired/invalidated warm can
// never leave a stale warm-ready bit.
//
// Default-silent, exactly like the warmup gate: a serve that never calls
// SetAgentWarmProfile reports the cache as "unconfigured" and readiness is
// unaffected — byte-for-byte the pre-#13333 behavior.
// ---------------------------------------------------------------------------

// Agent-warm status vocabulary — the closed set an agentWarmGate reports. The set
// is deliberately small: a caller (or a /healthz reader) branches on the token and
// never parses prose. "unconfigured" and "unsupported" are BOTH non-blocking (the
// serve is byte-for-byte unaffected); only "pending" and "degraded" hold readiness.
const (
	// AgentWarmUnconfigured means no agent warm profile was installed; the gate is
	// silent and readiness is unaffected.
	AgentWarmUnconfigured = "unconfigured"
	// AgentWarmPending means a profile is configured but no live warm receipt has been
	// observed yet. Readiness is held.
	AgentWarmPending = "pending"
	// AgentWarmReady means a live warm receipt was observed: the prefix is restored to
	// the stable boundary and (when configured) a residency claim is live.
	AgentWarmReady = "ready"
	// AgentWarmDegraded means the warm was attempted and failed to produce a usable
	// live prefix (cold/unsupported/partial restore, a stale identity, a dead claim, or
	// a planner error). Readiness is held and the closed reason names why; this is not
	// a silent success.
	AgentWarmDegraded = "degraded"
	// AgentWarmUnsupported means the planner cannot warm a prefix at all (recompute-only
	// model, no tree, an unqualified backend). This is reported independently and does
	// NOT hold readiness — there is no cache to warm, so the serve is unaffected.
	AgentWarmUnsupported = "unsupported"
)

// AgentWarmProfile is the descriptor-bearing configuration the host installs to
// make the gateway gate readiness on a live warm prefix. It carries the stable
// inputs a warm descriptor is derived from plus the expectation the observed
// receipt must match.
type AgentWarmProfile struct {
	// Tenant is the authenticated cache scope owner; required (a warm with no owner
	// cannot be bounded). Agent is the optional per-agent scope.
	Tenant string
	Agent  string
	// Inputs are the STABLE warm inputs (instructions, resident blocks, ordered tools,
	// KV layout, adapter). They are copied by SetAgentWarmProfile.
	Inputs agent.WarmPrefixInputs
	// RequireLiveClaim, when true, additionally requires the observed receipt to carry
	// a live residency claim. A planner without claim configuration leaves the claim
	// nil; setting this true against such a planner degrades the warm explicitly rather
	// than reporting a readiness the next request cannot rely on.
	RequireLiveClaim bool
}

// AgentWarmWarmer is the narrow seam the gateway calls to derive and materialize the
// configured agent prefix. *agent.InKernelPlanner satisfies it; a witness injects a
// fake without a live model. The gateway adds no cache math of its own — it derives a
// descriptor, warms it, and adjudicates the returned receipt.
type AgentWarmWarmer interface {
	DeriveWarmPrefix(tenant, agent string, in agent.WarmPrefixInputs) (agent.WarmPrefixSpec, error)
	WarmPrefix(ctx context.Context, spec agent.WarmPrefixSpec) (agent.WarmReceipt, error)
}

// agentWarmGate is the agent-warm readiness state machine. It mirrors warmupGate's
// shape (own mutex, value receipt) but its admission rule is the stronger
// live-receipt rule: a configured profile is ready only when a matching, restored,
// live-claimed warm has been observed. Zero value == unconfigured (silent).
type agentWarmGate struct {
	mu         sync.Mutex
	configured bool
	// spec is the derived, bounded descriptor the warm targets. Its Identity bounds
	// the observed receipt: a receipt for a different descriptor (a changed
	// profile/renderer/KV layout) is a mismatch, never a hit.
	spec             agent.WarmPrefixSpec
	requireLiveClaim bool

	status  string
	reason  string
	receipt *agent.WarmReceipt
}

// configure installs a derived descriptor and moves the gate to pending (armed but
// not yet warm). Reconfiguring clears any prior receipt, so a generation change can
// never reuse the previous profile's warm readiness.
func (g *agentWarmGate) configure(spec agent.WarmPrefixSpec, requireLiveClaim bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.configured = true
	g.spec = spec
	g.requireLiveClaim = requireLiveClaim
	g.status = AgentWarmPending
	g.reason = ""
	g.receipt = nil
}

// observe records the result of a native warm attempt and adjudicates readiness from
// the receipt — never from the attempt's mere completion. A receipt that is not
// Ready, whose identity does not match the configured descriptor, whose restored
// prefix does not reach the stable boundary, or (when required) whose claim is not
// live, is DEGRADED with a closed reason. A nil receipt/planner with an error is
// reported per the planner's own unsupported signal.
func (g *agentWarmGate) observe(receipt agent.WarmReceipt, unsupported bool, err error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.configured {
		// A late observation after an unconfigured gate is inert: a serve that never
		// configured a profile stays unaffected.
		g.status = AgentWarmUnconfigured
		g.receipt = nil
		return
	}
	g.receipt = nil
	switch {
	case unsupported:
		// The planner cannot warm a prefix at all: report it independently, do NOT hold
		// readiness (there is no cache to warm), and do NOT fabricate a ready bit.
		g.status = AgentWarmUnsupported
		g.reason = receipt.Reason
		if g.reason == "" {
			g.reason = "unsupported"
		}
		return
	case err != nil:
		g.status = AgentWarmDegraded
		g.reason = receipt.Reason
		if g.reason == "" {
			g.reason = "warm_error"
		}
		return
	}
	// A live receipt requires ALL of: matching descriptor identity, the Ready bit, and
	// a restored prefix reaching the stable boundary. Any miss is an explicit degrade —
	// never a silent hit.
	switch {
	case receipt.Identity == "" || receipt.Identity != g.spec.Identity:
		g.status = AgentWarmDegraded
		g.reason = "identity_mismatch"
		return
	case !receipt.Ready:
		g.status = AgentWarmDegraded
		g.reason = receipt.Reason
		if g.reason == "" {
			g.reason = "not_ready"
		}
		return
	case receipt.RestoredTokens < receipt.RequestedTokens || receipt.RestoredTokens <= 0:
		g.status = AgentWarmDegraded
		g.reason = "partial_restore"
		return
	case receipt.Status != agent.WarmStatusReady:
		g.status = AgentWarmDegraded
		g.reason = "status_not_ready"
		return
	}
	if g.requireLiveClaim {
		// A configured live-claim expectation cannot be satisfied by a receipt with no
		// claim, or a claim whose Live bit is false (expired/released/stale incarnation).
		// Without this the "complete bit" would be the only evidence — the exact trap
		// CW-07 exists to close.
		if receipt.Claim == nil {
			g.status = AgentWarmDegraded
			g.reason = "claim_absent"
			return
		}
		if !receipt.Claim.Live {
			g.status = AgentWarmDegraded
			g.reason = receipt.Claim.Reason
			if g.reason == "" {
				g.reason = "claim_not_live"
			}
			return
		}
	}
	g.status = AgentWarmReady
	g.reason = ""
	r := receipt
	g.receipt = &r
}

// admit reports whether readiness may be admitted for a configured agent-warm profile
// and, if not, the closed blocking status/reason. A ready gate admits. An unconfigured
// or unsupported gate does NOT block (there is no cache contract to enforce). A
// pending or degraded gate blocks with its reason. The bool is "blocked".
func (g *agentWarmGate) admit() (blocked bool, status, reason string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	switch g.status {
	case AgentWarmPending, AgentWarmDegraded:
		return true, g.status, g.reason
	default:
		return false, g.status, g.reason
	}
}

// snapshot returns the gate's current status, closed reason and receipt for a
// read-only reader (e.g. the /healthz agent_warm block). The receipt carries no prompt
// text, so it is safe to expose.
func (g *agentWarmGate) snapshot() (status, reason string, receipt *agent.WarmReceipt, configured bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.status, g.reason, g.receipt, g.configured
}

// SetAgentWarmProfile installs the agent KV-cache warm profile this serve gates
// readiness on (CW-07, #13333). Until RunAgentWarmup observes a live warm, /healthz
// holds readiness with agent_warm_pending. It derives the expected descriptor
// identity from the configured inputs through the installed warmer, so a receipt for
// any other descriptor is refused. A serve that never calls it is byte-for-byte
// unaffected (the cache reports "unconfigured"). Safe on a nil Server.
//
// It returns the derived descriptor identity for the host to log/bind, or an error
// when the warmer cannot derive a bounded descriptor (the gate is then left
// unconfigured so readiness is not held on a profile that cannot be realized).
func (s *Server) SetAgentWarmProfile(spec AgentWarmProfile) (string, error) {
	if s == nil {
		return "", ErrAgentWarmUnconfigured
	}
	warmer, ok := s.planner.(AgentWarmWarmer)
	if !ok {
		return "", ErrAgentWarmUnconfigured
	}
	desc, err := warmer.DeriveWarmPrefix(spec.Tenant, spec.Agent, spec.Inputs)
	if err != nil {
		return "", err
	}
	s.agentWarm.configure(desc, spec.RequireLiveClaim)
	return desc.Identity, nil
}

// RunAgentWarmup derives and materializes the configured agent prefix through the
// installed warmer, then adjudicates readiness from the returned receipt. It is the
// execution half of CW-07: the host calls it at boot (typically alongside the #3051
// RunWarmup) after SetAgentWarmProfile. It returns the observed receipt for the
// host's startup log.
//
// A planner that is not an AgentWarmWarmer, or a gate that was never configured,
// yields ErrAgentWarmUnconfigured. A warm that fails to produce a live prefix leaves
// the gate DEGRADED with a closed reason (readiness held) rather than reporting a
// readiness the first real request cannot use.
func (s *Server) RunAgentWarmup(ctx context.Context) (agent.WarmReceipt, error) {
	if s == nil {
		return agent.WarmReceipt{}, ErrAgentWarmUnconfigured
	}
	warmer, ok := s.planner.(AgentWarmWarmer)
	if !ok {
		s.agentWarm.observe(agent.WarmReceipt{}, true, ErrAgentWarmUnconfigured)
		return agent.WarmReceipt{}, ErrAgentWarmUnconfigured
	}
	spec, configured := s.agentWarm.configuration()
	if !configured {
		return agent.WarmReceipt{}, ErrAgentWarmUnconfigured
	}
	receipt, err := warmer.WarmPrefix(ctx, spec)
	unsupported := errors.Is(err, agent.ErrWarmPrefixUnsupported)
	s.agentWarm.observe(receipt, unsupported, err)
	return receipt, err
}

// configuration returns the installed, derived descriptor as a value copy and whether
// a profile is configured. WarmPrefix receives exactly the descriptor whose identity
// the gate will bind the receipt against.
func (g *agentWarmGate) configuration() (agent.WarmPrefixSpec, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.spec, g.configured
}

// ErrAgentWarmUnconfigured is the closed refusal for an agent-warm call with no
// installed profile or a planner that cannot warm a prefix. It is fail-closed: a
// caller must never read an unconfigured gate as a warm.
var ErrAgentWarmUnconfigured = errors.New("gateway: agent warm profile is not configured")
