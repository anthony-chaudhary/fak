package session

import "sync"

// pool.go — the OPTIONAL fleet-wide budget pool (#744). Each session's Budget is
// otherwise INDEPENDENT: N served sessions each carry their own TokensLeft, and a
// budget-reset (Recontinue) re-arms a session from a fresh PER-SESSION allotment with no
// awareness of its siblings — so N sessions resetting under "150k each" silently consume
// up to N×150k. A Pool turns that into a SHARED ceiling: one token allotment that many
// sessions DRAW from, so a whole fan-out can be held under a single cap (e.g. a 150k pool
// across N sessions) instead of N independent caps that sum without bound.
//
// The pool is deliberately a SEPARATE value from Budget, not a field on it. A Budget is a
// single session's remaining allotment, debited per turn by Decide/DebitUsage; a Pool is
// the cross-session ceiling a RESET draws a fresh allotment OUT OF. Keeping them distinct
// is what lets the pool stay OPTIONAL — a nil *Pool means "no shared cap," and every
// per-session path is byte-identical to today. The pool never touches the per-turn hot
// path (Decide/DebitUsage stay pool-free, lock-free of the pool); it is consulted only at
// a reset boundary, where a fresh window is armed, and read by the Snapshot surface, which
// reports the fleet ceiling alongside the per-session State rows.
//
// Concurrency: a Pool is safe for concurrent use by many sessions' reset boundaries. It
// guards a single remaining counter under a Mutex — the draws are short critical sections
// far off the per-turn path, so a plain Mutex is right (no need for the Table's RWMutex
// read/write split). A nil *Pool is a valid UNBOUNDED pool: every method is nil-safe so a
// host that wired no pool calls the same API and gets the no-shared-cap behavior.
type Pool struct {
	mu        sync.Mutex
	ceiling   int // configured fleet-wide cap; <=0 means UNBOUNDED (no shared ceiling)
	remaining int // tokens still available to draw; meaningful only when ceiling > 0
}

// NewPool returns a fleet-wide budget pool seeded with total tokens. A total <= 0 builds
// an UNBOUNDED pool: Draw always grants the full request and Remaining reports Unbounded,
// so constructing a pool is never, by itself, a tightening — only a positive ceiling
// bounds. The pool starts full (remaining == total).
func NewPool(total int) *Pool {
	if total < 0 {
		total = 0
	}
	return &Pool{ceiling: total, remaining: total}
}

// unbounded reports whether the pool enforces no shared cap. A nil receiver and a
// non-positive ceiling are both unbounded, so every public method routes the
// no-shared-cap case through one predicate.
func (p *Pool) unbounded() bool { return p == nil || p.ceiling <= 0 }

// Draw takes up to want tokens from the pool for a fresh session window, returning how
// many were GRANTED (0..want) and whether the full request was met. An unbounded pool (or
// a nil receiver) grants want in full — a missing pool never tightens. A bounded pool
// grants min(want, remaining) and debits exactly that, so the sum of all live draws can
// never exceed the ceiling: when the pool runs dry a reset gets granted==0/ok==false (and
// a partial grant likewise returns ok==false), so a host that wants a hard fleet stop can
// decline the continuation instead of minting an over-budget window. A non-positive want
// grants 0/true — nothing requested, nothing refused.
func (p *Pool) Draw(want int) (granted int, ok bool) {
	if want <= 0 {
		return 0, true
	}
	if p.unbounded() {
		return want, true
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.remaining <= 0 {
		return 0, false
	}
	if want > p.remaining {
		granted = p.remaining
		p.remaining = 0
		return granted, false // partial grant: pool now dry, request not fully met
	}
	p.remaining -= want
	return want, true
}

// Return puts n tokens back into a bounded pool — the inverse of Draw, for when a drawn
// allotment is released (a session ends without spending it, or a reset is rolled back).
// It never raises remaining above the ceiling (a buggy double-Return cannot inflate the
// cap) and is a no-op on an unbounded/nil pool or a non-positive n.
func (p *Pool) Return(n int) {
	if n <= 0 || p.unbounded() {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.remaining += n
	if p.remaining > p.ceiling {
		p.remaining = p.ceiling
	}
}

// Remaining reports the tokens still available to draw. An unbounded (or nil) pool reports
// Unbounded (-1), distinguishing "no shared cap" from a bounded pool that is exactly dry
// (0).
func (p *Pool) Remaining() int {
	if p.unbounded() {
		return Unbounded
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.remaining
}

// Cap reports the configured fleet-wide ceiling (0 = unbounded). It never changes after
// construction, so it is read without the lock.
func (p *Pool) Cap() int {
	if p == nil {
		return 0
	}
	return p.ceiling
}

// PoolReport is the fleet-wide budget snapshot the scheduler/Snapshot surface emits
// alongside the per-session State rows: how big the shared ceiling is, how much N sessions
// have already drawn, and how much headroom a further reset has. Drawn is Cap-Remaining (0
// for an unbounded pool); Remaining is Unbounded(-1) when there is no shared cap. Bounded
// distinguishes "a ceiling is enforced" from "no pool wired," so a consumer never mistakes
// an unbounded pool's zero Drawn for a fully-spent one.
type PoolReport struct {
	Cap       int  `json:"cap"`       // configured fleet-wide ceiling; 0 = unbounded
	Drawn     int  `json:"drawn"`     // tokens handed out to sessions so far
	Remaining int  `json:"remaining"` // tokens still available; Unbounded(-1) = no shared cap
	Bounded   bool `json:"bounded"`   // whether a shared cap is enforced at all
}

// Report renders the pool's current fleet-wide state for the Snapshot surface. It is a
// consistent read — Cap, Drawn, and Remaining are captured under one lock. A nil/unbounded
// pool reports Bounded=false with Remaining=Unbounded and Drawn=0, so a host that wired no
// pool still emits a well-formed (no-ceiling) row.
func (p *Pool) Report() PoolReport {
	if p.unbounded() {
		return PoolReport{Cap: 0, Drawn: 0, Remaining: Unbounded, Bounded: false}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return PoolReport{
		Cap:       p.ceiling,
		Drawn:     p.ceiling - p.remaining,
		Remaining: p.remaining,
		Bounded:   true,
	}
}

// RecontinuePooled is Recontinue against a shared Pool: it re-arms a budget-drained session
// on a fresh window whose token allotment is DRAWN from the fleet-wide pool, so N sessions
// resetting under one ceiling cannot collectively exceed it. want is the budget the child
// would get with no pool; its TokensLeft axis is clamped to what the pool grants, every
// other axis passes through untouched (the pool caps OUTPUT tokens — the axis a fleet-wide
// "N sessions share 150k" cap is about). The bool is the pool's ok: false means the pool
// could not fully fund the request (dry, or only partially funded). The child is still
// armed with whatever was granted (granted==0 ⇒ TokensLeft 0, which the next Decide drains
// immediately), so a host wanting a hard fleet stop checks ok and declines the continuation
// rather than relying on this to refuse.
//
// An unbounded/nil pool grants want.TokensLeft in full, so RecontinuePooled with no pool is
// byte-identical to Recontinue. An Unbounded want.TokensLeft (no per-session token cap) is
// passed through UNDRAWN: "unbounded per session" opts that session out of the shared cap
// by construction — a fleet token cap is only meaningful for a session that carries a
// finite token budget — so it neither draws nor reports against the pool.
func (t *Table) RecontinuePooled(parent, child string, want Budget, pool *Pool) (State, bool) {
	if want.tokensUnbounded() {
		return t.Recontinue(parent, child, want), true
	}
	granted, ok := pool.Draw(want.TokensLeft)
	want.TokensLeft = granted
	return t.Recontinue(parent, child, want), ok
}
