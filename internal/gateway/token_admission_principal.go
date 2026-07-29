package gateway

// token_admission_principal.go — PER-PRINCIPAL (per-key / per-tenant) token admission
// (#5379, split out of the gateway governance parity audit #3280, workstream C of epic
// #3256).
//
// WHY THIS FILE EXISTS AS A SEPARATE LAYER. token_admission.go's honest fence says it
// plainly: one TokenRateGate is ONE provider budget (one account/seat), and the per-seat
// pool "composes ABOVE this". This file is that ABOVE layer, and it is deliberately built
// by COMPOSITION rather than by teaching TokenRateGate about identity: a principal's
// allotment is itself an ordinary TokenRateGate, so the rolling-window arithmetic, the
// reserve-on-estimate/settle-on-truth shape and the typed shed all stay in exactly one
// place. Nothing in TokenRateGate changes meaning; a host that never installs a book keeps
// the request path byte-for-byte historical, the same inert-by-default posture as
// SetTokenRateGate itself.
//
// THE GAP IT CLOSES. The gateway already authenticates many api keys, each bound to an
// org/project PRINCIPAL that rides the request (keyset.go / #5332, withAuth →
// WithPrincipal → principalFromContext). It also already enforces a rolling-window TPM +
// concurrency budget. Until now the two were uncomposed, so one noisy key could drink the
// whole provider budget and every other tenant on the same gateway got the 429. Here the
// caller's principal selects its OWN allotment, checked BEFORE the shared provider gate:
// a tenant over its own cap is shed without ever reserving a slice of the shared window,
// and a tenant under its cap is unaffected by a neighbour that blew through theirs.
//
// THE UNIDENTIFIED CALLER — a decision, not an oversight. A request that carries no
// principal (no keyset armed, or an internal caller) is charged to ONE shared allotment
// (allotmentUnidentified), capped exactly like a tenant's. The two alternatives are both
// worse. Giving each unidentified request a FRESH allotment makes the whole limit
// bypassable by simply omitting identity — an admission-control limit that evaporates when
// the caller declines to say who they are is not a limit. Leaving unidentified callers
// UNMETERED is the same hole with a friendlier name. Shared fate among unidentified
// callers is a real cost, but it is exactly the state such a deployment was already in
// (one provider budget for everyone), so this direction only ever tightens.
//
// THE MAP BOUND — also a decision. On a gateway with no keyset the principal string is
// caller-supplied (principalFor reads the X-Fak-Principal header), so a map keyed by it
// is an unbounded-growth surface: mint a fresh principal per request and the book is the
// denial of service. Two mechanisms bound it. (1) IDLE SWEEP: an allotment holding no
// in-flight reservation whose window has already elapsed forgives NO budget when dropped —
// the roll would have zeroed its settled usage anyway — so evicting it is semantically
// free, and the live book tracks concurrent tenants rather than historical ones.
// (2) A hard CEILING on NAMED allotments: once the book is full even after a sweep, a
// newly-seen principal folds into a shared allotmentOverflow bucket. Overflow is kept
// distinct from allotmentUnidentified precisely so that minting principals cannot starve
// genuine unidentified callers. Note which way both fallbacks resolve: a caller that
// cannot get its own allotment shares a CAPPED one — never a fresh one, never an
// unmetered one. Uncertainty in an admission path must not resolve toward admitting.

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// defaultPrincipalCeiling bounds how many DISTINCT named principals hold their own
// allotment at once when NewPrincipalTokenRates is given no explicit ceiling. It is a
// memory bound on a caller-influenced map, not a tenancy limit: principals past it are
// still metered, via the shared overflow allotment.
const defaultPrincipalCeiling = 1024

// allotmentClass separates the three budget-bearing caller classes. It is a distinct
// field rather than a magic principal string on purpose: no caller-supplied principal,
// however crafted, can land in (or be confused with) the shared unidentified/overflow
// allotments, because those carry an empty id under a different class.
type allotmentClass uint8

const (
	// allotmentUnidentified is the single shared allotment for callers presenting no
	// principal at all.
	allotmentUnidentified allotmentClass = iota
	// allotmentNamed is one authenticated principal's own allotment.
	allotmentNamed
	// allotmentOverflow is the single shared allotment for named principals seen after
	// the book hit its ceiling.
	allotmentOverflow
)

// allotmentKey identifies one allotment in the book. id is set only for allotmentNamed.
type allotmentKey struct {
	class allotmentClass
	id    string
}

// subject renders the key for the refusal a client sees, so a 429 says WHICH budget
// fired — the actionable difference between "my tenant is over its cap" and "the shared
// unidentified budget is saturated".
func (k allotmentKey) subject() string {
	switch k.class {
	case allotmentNamed:
		return fmt.Sprintf("principal %q", k.id)
	case allotmentOverflow:
		return "shared overflow principal"
	default:
		return "unidentified caller"
	}
}

// PrincipalTokenRates is the per-principal token allotment book: one rolling-window
// TokenRateGate per caller class, selected by the request's authenticated principal.
// Build with NewPrincipalTokenRates; the zero value is not usable. Safe for concurrent
// use. A nil *PrincipalTokenRates admits freely, so an unwired host is unaffected.
type PrincipalTokenRates struct {
	mu sync.Mutex
	// perPrincipal is the budget EACH allotment gets — every principal is issued the same
	// ceiling. Per-tenant differentiation is a policy surface above this seam (#C1) and
	// deliberately out of scope here.
	perPrincipal TokenRatePolicy
	// ceiling bounds the number of live allotmentNamed entries; the two shared classes
	// sit outside it, so the book holds at most ceiling+2 gates.
	ceiling int
	// now is the injectable clock handed to every allotment gate this book creates.
	now  func() time.Time
	book map[allotmentKey]*TokenRateGate
	// named is the live count of allotmentNamed entries, maintained alongside book so the
	// ceiling check never walks the map.
	named int
}

// NewPrincipalTokenRates builds a book issuing every principal the same per-window budget.
// A ceiling <= 0 takes defaultPrincipalCeiling; a Window <= 0 inside the policy takes
// defaultTokenRateWindow, exactly as NewTokenRateGate does.
func NewPrincipalTokenRates(perPrincipal TokenRatePolicy, ceiling int) *PrincipalTokenRates {
	if perPrincipal.Window <= 0 {
		perPrincipal.Window = defaultTokenRateWindow
	}
	if ceiling <= 0 {
		ceiling = defaultPrincipalCeiling
	}
	return &PrincipalTokenRates{
		perPrincipal: perPrincipal,
		ceiling:      ceiling,
		now:          time.Now,
		book:         map[allotmentKey]*TokenRateGate{},
	}
}

// Admit checks one call's estimated footprint against THIS principal's allotment and, on
// admission, reserves it there. A refusal is the same typed served-path *AdmissionError
// (VerdictShed → HTTP 429) the provider gate returns, with the subject naming which
// budget fired. An empty principal is the unidentified caller (see the file header). A nil
// book admits freely.
func (b *PrincipalTokenRates) Admit(principal string, estimate ratelimit.TokenUsage) (*TokenReservation, error) {
	if b == nil {
		return nil, nil
	}
	return b.allotmentFor(principal).Admit(estimate)
}

// AllotmentCounts reports the book's live size: how many named principals hold their own
// allotment and the total gate count including the shared unidentified/overflow classes.
// The invariant this exposes is the whole point of the ceiling — named never exceeds it.
func (b *PrincipalTokenRates) AllotmentCounts() (named, total int) {
	if b == nil {
		return 0, 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.named, len(b.book)
}

// SnapshotFor returns the window accounting of the allotment a principal would be charged
// to, WITHOUT creating one — the observability read a host or test makes. The zero
// snapshot means that principal holds no live allotment.
func (b *PrincipalTokenRates) SnapshotFor(principal string) TokenRateSnapshot {
	if b == nil {
		return TokenRateSnapshot{}
	}
	b.mu.Lock()
	g := b.book[keyForPrincipal(principal)]
	b.mu.Unlock()
	if g == nil {
		return TokenRateSnapshot{}
	}
	return g.Snapshot()
}

// keyForPrincipal maps a raw principal string to its allotment key. Trimming matches
// WithPrincipal / principalFor, so a whitespace-only principal is the unidentified caller
// rather than a distinct tenant named " ".
func keyForPrincipal(principal string) allotmentKey {
	if p := strings.TrimSpace(principal); p != "" {
		return allotmentKey{class: allotmentNamed, id: p}
	}
	return allotmentKey{class: allotmentUnidentified}
}

// allotmentFor resolves (creating if needed) the gate a principal is charged to, applying
// the idle sweep and the ceiling fallback described in the file header. Never nil.
func (b *PrincipalTokenRates) allotmentFor(principal string) *TokenRateGate {
	key := keyForPrincipal(principal)
	b.mu.Lock()
	defer b.mu.Unlock()
	if g := b.book[key]; g != nil {
		return g
	}
	if key.class == allotmentNamed && b.named >= b.ceiling {
		b.sweepIdleLocked()
		if b.named >= b.ceiling {
			key = allotmentKey{class: allotmentOverflow}
			if g := b.book[key]; g != nil {
				return g
			}
		}
	}
	g := NewTokenRateGate(b.perPrincipal)
	g.now = b.now
	g.windowStart = b.now()
	g.subject = key.subject()
	b.book[key] = g
	if key.class == allotmentNamed {
		b.named++
	}
	return g
}

// sweepIdleLocked drops every named allotment that holds no in-flight reservation and
// whose window has already elapsed. Such an allotment is indistinguishable from a fresh
// one — the next roll would zero its settled usage regardless — so the eviction forgives
// no budget and cannot be used to launder a spent window. The shared classes are never
// swept: they sit outside the ceiling and recreating them buys nothing. Caller holds b.mu.
func (b *PrincipalTokenRates) sweepIdleLocked() {
	for k, g := range b.book {
		if k.class != allotmentNamed || !g.idle() {
			continue
		}
		delete(b.book, k)
		b.named--
	}
}

// SetPrincipalTokenRates wires the per-principal token allotment book (#5379) onto the
// Server: a served request reserves against its own principal's allotment BEFORE the
// shared host-level provider gate, and settles both with the provider's real normalized
// usage. nil detaches it (the request path goes back to the single shared budget).
// Settable after New, guarded by admissionMu alongside SetTokenRateGate. A nil receiver
// is a no-op.
func (s *Server) SetPrincipalTokenRates(b *PrincipalTokenRates) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.principalTokenRates = b
	s.admissionMu.Unlock()
}
