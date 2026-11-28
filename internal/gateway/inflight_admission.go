package gateway

// inflight_admission.go — request-count INFLIGHT admission for the served path, ADAPTED
// from TensorFold's opt-in model-request admission (Apache-2.0): ashhart/TensorFold
// THIRD_PARTY_NOTICES.md ("TensorFold").
//
// BORROWED CONTRACT, re-expressed in idiomatic Go (no upstream code copied):
//   - a request is admitted only within a global max_inflight AND a per-key cap; zero
//     limits count nothing — an all-zero policy admits unbounded concurrent holds and
//     refuses nothing;
//   - a bounded table of distinct principals (default 128 slots) bounds the
//     caller-influenced surface; a NEW principal beyond the bound is refused with its
//     own class until a live holder releases;
//   - refusals are typed by closed classes (global / key / capacity), each distinct and
//     all under one parent sentinel, so callers classify with errors.Is — never prose;
//   - an admitted hold is a lease released EXACTLY ONCE at the end (sync.Once,
//     nil-safe); a refused acquire holds nothing and releases nothing.
//
// SHAPE — composition over the existing token-admission seam, not a parallel counter
// machine. The gate keeps no counters that would owe a release hook: every dimension
// rides a TokenRateGate carrier (token_admission.go) whose reserve-on-admit /
// settle-on-truth machinery already decrements on Release/Settle. The global dimension
// is the concurrency cap of one carrier — ratelimit.TokenCaps.Decide does the
// arithmetic — one carrier per capped key, and one inert carrier per live principal
// entry so the distinct-principal table sweeps idle exactly like the per-principal
// book (#5379). beginServedAdmission (admission_server.go) consults this gate FIRST —
// the coarsest bound before the scheduler slot — and ties each hold's release to the
// request's single lease-release path through the TokenReservation linked chain.
// Inert by default: a Server with no installed gate (SetInflightGate(nil)) leaves the
// request path byte-for-byte historical. The gate is PER-SERVER state (#14453), not
// process-global: each gateway owns its own tenants' holds.
//
// HONEST FENCE — what this is NOT. It counts REQUESTS, not tokens: it settles no
// usage, charges no window, and claims no throughput or hardware property. The
// per-second rate bucket (upstream's fourth refusal class) is a sibling borrow-port
// part-out and is deliberately not here.

import (
	"errors"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// DefaultInflightPrincipalSlots bounds how many DISTINCT principals may hold a live
// inflight lease at once when InflightPolicy.PrincipalSlots is unset. It is a memory
// bound on a caller-influenced table, not a tenancy limit — the same posture as
// defaultPrincipalCeiling (#5379): a principal past the bound is refused until a live
// holder releases, never folded into an unbounded map.
const DefaultInflightPrincipalSlots = 128

// ErrInflight is the parent sentinel every inflight refusal satisfies: a caller that
// only needs "the inflight gate said no" classifies against it; a caller that needs
// the class uses one of the three specific sentinels below.
var ErrInflight = errors.New("gateway inflight admission refused")

// The three closed refusal classes, mirroring the upstream Reason enum minus the rate
// class (sibling part-out). Each is distinct; all satisfy errors.Is(err, ErrInflight).
var (
	ErrInflightGlobal   = errors.New("inflight global max_inflight exhausted")
	ErrInflightKey      = errors.New("inflight per-key cap exhausted")
	ErrInflightCapacity = errors.New("inflight principal slot bound exhausted")
)

// inflightRefusal wraps one class sentinel plus the parent so errors.Is matches both —
// classification by class, never by error prose.
type inflightRefusal struct{ cause error }

func (e *inflightRefusal) Error() string { return e.cause.Error() }

func (e *inflightRefusal) Unwrap() []error {
	return []error{e.cause, ErrInflight}
}

// InflightPolicy configures the request-count inflight gate. Every zero value disables
// its dimension (zero limits count nothing): MaxInflight 0 = no global cap, a PerKey
// entry absent or <= 0 = no cap for that key, PerPrincipal 0 = no uniform per-principal
// cap, PrincipalSlots 0 = DefaultInflightPrincipalSlots. A policy with no positive
// limit anywhere is fully inert: Acquire admits unbounded concurrent holds without any
// refusal and tracks nothing.
type InflightPolicy struct {
	MaxInflight int
	PerKey      map[string]int
	// PerPrincipal caps ANY principal's concurrent holds uniformly — the per-tenant
	// default for tenants that are not pre-listed in PerKey (#14453, fak serve
	// --tenant-max-inflight). It composes WITH PerKey: both carriers are enforced and
	// the tighter binds. 0 (the default) leaves every principal uncapped on this axis,
	// byte-for-byte the historical behavior.
	PerPrincipal   int
	PrincipalSlots int
}

// enabled reports whether any dimension actually limits requests — the all-zero policy
// counts nothing, reserves nothing and refuses nothing.
func (p InflightPolicy) enabled() bool {
	if p.MaxInflight > 0 || p.PerPrincipal > 0 {
		return true
	}
	for _, c := range p.PerKey {
		if c > 0 {
			return true
		}
	}
	return false
}

func (p InflightPolicy) slots() int {
	if p.PrincipalSlots > 0 {
		return p.PrincipalSlots
	}
	return DefaultInflightPrincipalSlots
}

// InflightGate is the request-count inflight admission gate. Build with
// NewInflightGate; the zero-value policy is inert. Safe for concurrent use. A nil gate
// admits freely, so an unwired host is unaffected.
type InflightGate struct {
	mu               sync.Mutex
	policy           InflightPolicy
	slots            int
	global           *TokenRateGate            // concurrency carrier: Caps.MaxConcurrent = MaxInflight (0 = unlimited)
	perKey           map[string]*TokenRateGate // one concurrency carrier per capped key
	principals       map[string]*inflightPrincipalEntry
	admitted         uint64
	rejectedGlobal   uint64
	rejectedKey      uint64
	rejectedCapacity uint64
	// sheds counts refusals PER PRINCIPAL (#14453): the per-tenant shed counter the
	// /metrics surface renders as fak_tenant_admission_shed_total{gate="inflight"}.
	// All three closed classes count; a principal with no refusal holds no entry.
	sheds map[string]uint64
}

// inflightPrincipalEntry is one distinct principal's seat in the bounded table. Its
// inert carrier's live-reservation count IS the principal's liveness, which is what
// makes sweepIdleLocked a pure memory reclaim — the same argument as the per-principal
// book's idle sweep (#5379).
type inflightPrincipalEntry struct {
	gate *TokenRateGate
}

// NewInflightGate builds a gate under an owned snapshot of the given policy.
// Callers may reuse or change PerKey after construction, but must not mutate it
// concurrently with construction. Reconfiguration requires a new gate.
func NewInflightGate(p InflightPolicy) *InflightGate {
	if p.PerKey != nil {
		limits := make(map[string]int, len(p.PerKey))
		for key, limit := range p.PerKey {
			limits[key] = limit
		}
		p.PerKey = limits
	}
	return &InflightGate{
		policy:     p,
		slots:      p.slots(),
		global:     NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxConcurrent: p.MaxInflight, TargetUtilization: 1}}),
		perKey:     map[string]*TokenRateGate{},
		principals: map[string]*inflightPrincipalEntry{},
		sheds:      map[string]uint64{},
	}
}

// InflightSnapshot is the gate's current accounting — the observability read a host or
// test makes: live holds, live distinct principals, and the closed-class refusal counts.
type InflightSnapshot struct {
	InFlight         int
	Principals       int
	Admitted         uint64
	RejectedGlobal   uint64
	RejectedKey      uint64
	RejectedCapacity uint64
}

// Snapshot returns the gate's current accounting. A nil gate reports zeros.
func (g *InflightGate) Snapshot() InflightSnapshot {
	if g == nil {
		return InflightSnapshot{}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return InflightSnapshot{
		InFlight:         g.global.Snapshot().InFlight,
		Principals:       len(g.principals),
		Admitted:         g.admitted,
		RejectedGlobal:   g.rejectedGlobal,
		RejectedKey:      g.rejectedKey,
		RejectedCapacity: g.rejectedCapacity,
	}
}

// InflightLease is one admitted request's hold on the inflight counts. Release is
// idempotent (sync.Once) and nil-safe; a refused acquire returns a nil lease that holds
// nothing and releases nothing.
type InflightLease struct {
	gate         *InflightGate
	head         *TokenReservation // the global-dimension carrier; always non-nil on an admitted hold
	keyRes       *TokenReservation // the per-key carrier; nil when the key is uncapped
	principalRes *TokenReservation // the principal entry's carrier
	once         sync.Once
}

// Acquire admits one request for (principal, key) only while active < MaxInflight AND
// active[key] < PerKey[key], refusing with the fired class's sentinel otherwise:
// ErrInflightGlobal on global exhaustion, ErrInflightKey on per-key exhaustion, and
// ErrInflightCapacity for a new principal beyond the slot bound. An inert gate (nil
// receiver or all-zero policy) admits freely and returns a nil lease. On refusal the
// gate holds nothing: every carrier taken before the firing check is released.
func (g *InflightGate) Acquire(principal, key string) (*InflightLease, error) {
	if g == nil || !g.policy.enabled() {
		return nil, nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	lease := &InflightLease{gate: g}
	head, err := g.global.Admit(ratelimit.TokenUsage{})
	if err != nil {
		g.rejectedGlobal++
		g.sheds[principal]++
		return nil, &inflightRefusal{cause: ErrInflightGlobal}
	}
	lease.head = head
	if cap := g.policy.PerKey[key]; cap > 0 {
		kres, err := g.perKeyCarrierLocked(key, cap).Admit(ratelimit.TokenUsage{})
		if err != nil {
			g.rejectedKey++
			g.sheds[principal]++
			lease.Release() // free the global carrier this refused acquire took
			return nil, &inflightRefusal{cause: ErrInflightKey}
		}
		lease.keyRes = kres
	}
	pres, err := g.principalHoldLocked(principal)
	if err != nil {
		// The refusal is typed by principalHoldLocked: the per-principal uniform cap
		// fires as the key class (the served path keys by principal), the slot bound
		// as capacity. Count the fired class, never a guess.
		switch {
		case errors.Is(err, ErrInflightKey):
			g.rejectedKey++
		default:
			g.rejectedCapacity++
		}
		g.sheds[principal]++
		lease.Release()
		return nil, err
	}
	lease.principalRes = pres
	// Served admission retains only head. Link all successful dimensions before
	// publishing it so both settlement and release retire the entire hold.
	lease.head.linked = pres
	if lease.keyRes != nil {
		lease.head.linked = lease.keyRes
		lease.keyRes.linked = pres
	}
	g.admitted++
	return lease, nil
}

// Release frees the hold exactly once: each carrier's settlement drops its dimension's
// count, and the sync.Once makes a second Release a no-op. Nil-safe.
func (l *InflightLease) Release() {
	if l == nil || l.gate == nil {
		return
	}
	l.once.Do(func() {
		l.head.Release()
		l.keyRes.Release()
		l.principalRes.Release()
	})
}

// perKeyCarrierLocked returns (creating if needed) the concurrency carrier for a capped
// key. Caller holds g.mu.
func (g *InflightGate) perKeyCarrierLocked(key string, cap int) *TokenRateGate {
	if gt := g.perKey[key]; gt != nil {
		return gt
	}
	gt := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxConcurrent: cap, TargetUtilization: 1}})
	g.perKey[key] = gt
	return gt
}

// principalHoldLocked takes (creating if needed) the seat a distinct principal holds in
// the bounded table, applying the idle sweep first. The entry's carrier carries the
// policy's uniform PerPrincipal cap (#14453): when set, it is a real concurrency cap on
// THAT principal's holds (refusing with the key class — the served path keys by
// principal), and when unset it stays inert exactly as before. Caller holds g.mu.
func (g *InflightGate) principalHoldLocked(principal string) (*TokenReservation, error) {
	if e := g.principals[principal]; e != nil {
		res, err := e.gate.Admit(ratelimit.TokenUsage{})
		if err != nil {
			return nil, &inflightRefusal{cause: ErrInflightKey}
		}
		return res, nil
	}
	if len(g.principals) >= g.slots {
		g.sweepIdleLocked()
		if len(g.principals) >= g.slots {
			return nil, &inflightRefusal{cause: ErrInflightCapacity}
		}
	}
	caps := ratelimit.TokenCaps{}
	if g.policy.PerPrincipal > 0 {
		caps = ratelimit.TokenCaps{MaxConcurrent: g.policy.PerPrincipal, TargetUtilization: 1}
	}
	e := &inflightPrincipalEntry{gate: NewTokenRateGate(TokenRatePolicy{Caps: caps})}
	g.principals[principal] = e
	res, err := e.gate.Admit(ratelimit.TokenUsage{})
	if err != nil {
		// A fresh carrier starts empty, so this cannot fire for a positive cap; keep
		// the invariant total anyway and drop the seat that holds nothing.
		delete(g.principals, principal)
		return nil, &inflightRefusal{cause: ErrInflightKey}
	}
	return res, nil
}

// sweepIdleLocked drops every principal entry whose carrier holds no live reservation.
// Unlike the per-principal book's idle sweep (#5379), no window-elapsed clause is needed:
// these carriers carry at most a CONCURRENCY cap (the uniform PerPrincipal dimension,
// #14453) and no token caps, so their settled-window accounting constrains nothing — an
// entry with zero live reservations is indistinguishable from a fresh one and dropping it
// forgives no budget. Freeing the seat the moment the last hold releases mirrors the
// upstream slot-reset-on-zero-count, keeping the table tight under caller-churned
// principals. Caller holds g.mu.
func (g *InflightGate) sweepIdleLocked() {
	for p, e := range g.principals {
		if e.gate.Snapshot().InFlight != 0 {
			continue
		}
		delete(g.principals, p)
	}
}

// SetInflightGate installs this Server's inflight gate, consulted ahead of the
// scheduler slot by beginServedAdmission; nil detaches it and the request path goes
// byte-for-byte historical. Per-SERVER state (#14453): each gateway owns its own
// tenants' holds, so two Servers on one process can run different inflight policies
// without cross-talk. Settable after New, guarded by admissionMu alongside
// SetPrincipalTokenRates. A nil receiver is a no-op.
func (s *Server) SetInflightGate(g *InflightGate) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.inflightGate = g
	s.admissionMu.Unlock()
}

// ShedCounts returns the per-principal refusal totals (#14453) — the per-tenant shed
// counter the /metrics surface renders as fak_tenant_admission_shed_total{gate="inflight"}.
// The map is a copy; a nil gate reports an empty map.
func (g *InflightGate) ShedCounts() map[string]uint64 {
	if g == nil {
		return map[string]uint64{}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make(map[string]uint64, len(g.sheds))
	for p, n := range g.sheds {
		if n > 0 {
			out[p] = n
		}
	}
	return out
}
