package microagent

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Idle-agent hibernation (epic #2000 M12, issue #2012).
//
// An enrolled microagent that is parked — waiting on a free admission slot (M19)
// or a slow tool between steps — still costs a goroutine and its in-RAM context.
// Hibernation parks that context to disk and lets the goroutine go, so a host can
// enroll N agents while keeping only R of them resident (goroutine + RAM), the
// rest frozen on disk at ~O(1) RAM each. On wake the context is restored
// byte-identically, so no state is lost across the round-trip.
//
// The load-bearing constraint, stated honestly: an agent blocked INSIDE a
// synchronous Step (a live syscall / tool call) cannot be hibernated — Go does
// not serialize a goroutine stack. Hibernation happens at a STEP BOUNDARY: the
// host freezes an agent that has RETURNED from Step (parked between units of
// work), then releases its goroutine. That boundary is exactly the shape
// internal/microagent already requires of Step (Step advances one unit and
// returns), so a well-behaved step-resumable agent is hibernatable today.
//
// Generation intent: gen/future (#2012) — an OPTION behind an explicit seam,
// mirroring the Host itself (doc.go): nothing in the default fak serve / guard /
// dispatch path constructs a HibernationStore or a ResidentCap. Closing evidence
// for the generation frame:
//
//   - Promotion evidence: hibernate_test.go witnesses (a) a hibernate->wake
//     round-trip that is byte-identical (Wake refuses a lossy Thaw), and (b) N
//     enrolled step-agents driven through an R-slot ResidentCap + a
//     HibernationStore where the resident set never exceeds R yet all N complete
//     — the two #2012 acceptance bullets. The FIRST half of this promotion gate
//     has since shipped: the live Host.run loop CAN now target the store, via
//     Config.Warm (#5072) — a WarmBand (warmband.go) composes this ResidentCap,
//     a WarmReserve and a HibernationStore into one residency cycle, so a Host
//     auto-hibernates a step-parked Restorable agent at its step boundary and
//     N enrolled agents share R resident slots. What still gates promotion is
//     the OTHER half: a density / RSS-vs-R measurement confirming the parked
//     goroutine + context was the binding cost (the #2000 M8-M12 footprint
//     thesis), and a real (non-test) dispatch path electing a banded Host —
//     nothing in the default fak serve / guard / dispatch path constructs one,
//     so the seam stays gen/future.
//   - Demotion / retirement criteria: retire the seam if the footprint benchmark
//     shows a parked microagent's goroutine + bounded context is cheap enough
//     that O(N) residency is affordable (hibernation then buys no density), or if
//     per-agent isolation forces an OS process per agent anyway (#2018), so there
//     is no in-process goroutine to free.
//   - Invalidating assumption: hibernation assumes the agent loop is
//     step-resumable — that a Microagent's whole live state between steps is
//     captured by Freeze and restored by Thaw, with no residue in the goroutine
//     stack, an open file, a working directory, or a process-tree tool. If a real
//     loop parks mid-Step on a slow tool (the exact case in the issue body) its
//     goroutine cannot be freed without the still-open subprocess ToolExec /
//     admission seams (#2003/#2014/M19); Freeze/Thaw covers the between-steps
//     context only. Prior art for the wake-boundary re-validation discipline:
//     docs/notes/RESEARCH-random-time-horizons-dormancy-rehydration-2026-06-28.md
//     (the CRaC afterRestore analog — verify at the restore boundary, don't trust
//     the thawed snapshot).

// DefaultResidentCap is the resident-slot limit a NewResidentCap(0) selects. It
// mirrors DefaultWorkers: by default at most this many agents hold a goroutine.
const DefaultResidentCap = 8

// Structured refusals for the hibernation seam.
var (
	// ErrUnsafeHibernateID refuses an id that is not a single safe path element,
	// so a frozen file can never escape the store directory (no separators, no
	// "." / ".." — an id is one file, <dir>/<id>.hib).
	ErrUnsafeHibernateID = errors.New("microagent: hibernation id must be a single safe path element (no separators, not . or ..)")
	// ErrNotHibernated is returned by Wake for an id with no frozen context on
	// disk (never parked, or already woken).
	ErrNotHibernated = errors.New("microagent: no hibernated context for id")
	// ErrThawMismatch refuses a Thaw that is not the faithful inverse of Freeze:
	// Wake re-Freezes the restored agent and requires the bytes to equal what it
	// read from disk. This is the built-in no-state-loss check — a lossy round
	// trip is refused at the wake boundary, never silently accepted.
	ErrThawMismatch = errors.New("microagent: Thaw did not restore the frozen context byte-identically (lossy round-trip refused)")
	// ErrWakePanicked is returned to a wake — the single-flight leader AND every
	// follower coalesced behind it — when the caller-supplied Hibernable panics
	// inside Thaw/Freeze during the restore. The panic is caught at the wake
	// boundary so it can never wedge the id: a panicking leader that skipped its
	// inflight cleanup would leave the id's wakeCall in the map and its done channel
	// open forever, deadlocking every current and future waiter and leaking their
	// goroutines. Instead the panic becomes this loud error, fanned out to all
	// waiters like any other outcome, matching the boundary's refuse-loudly stance
	// (issue #4034).
	ErrWakePanicked = errors.New("microagent: Hibernable panicked during wake restore (Thaw/Freeze)")
)

// Hibernable is the OPTIONAL seam a step-resumable Microagent implements to be
// parked to disk between steps and restored byte-identically on wake (#2012).
//
//   - Freeze serializes the agent's bounded context (epic #2000 M4 — the bounded
//     linear-history context) to a self-describing byte slice. It MUST be
//     deterministic: two Freeze calls on an unchanged context return equal bytes,
//     so a hibernate->wake round-trip is byte-identical.
//   - Thaw restores that context. Thaw(b) followed by Freeze() MUST return b
//     (HibernationStore.Wake enforces this and refuses ErrThawMismatch otherwise).
//
// Freeze/Thaw cover the between-steps context only — never a goroutine blocked
// inside a live Step (see the package-level note above).
type Hibernable interface {
	Microagent
	Freeze() ([]byte, error)
	Thaw([]byte) error
}

// HibernationStore parks a hibernated agent's frozen context on disk (one file
// per agent, <dir>/<id>.hib) so it stops holding RAM + a goroutine while it
// waits, and wakes it byte-identically. It is the M12 mechanism a resident cap
// stands on: N enrolled agents, at most R resident, the rest frozen here.
//
// It is safe for concurrent use: each agent id is an independent file, and Park
// writes atomically (temp file + rename) so a crash mid-write never leaves a
// torn frozen context a later Wake would restore.
type HibernationStore struct {
	dir      string
	mu       sync.Mutex           // guards the temp-file dance AND the inflight map; ids are independent
	inflight map[string]*wakeCall // per-id single-flight for Wake (#4034); nil entry = no wake in flight
}

// wakeCall is one in-flight Wake for an id. Concurrent wakes of the SAME id join the
// leader's call and share its outcome, so a wake stampede on one hibernated id drives
// exactly one Thaw + one file removal instead of N goroutines racing
// ReadFile→Thaw→re-Freeze→Remove on the same frozen file (issue #4034).
type wakeCall struct {
	done chan struct{} // closed when the leader's restore finishes
	err  error         // the leader's result, fanned out to every waiter
}

// NewHibernationStore roots a store at dir, creating it if needed.
func NewHibernationStore(dir string) (*HibernationStore, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("microagent: hibernation store dir: %w", err)
	}
	return &HibernationStore{dir: dir, inflight: map[string]*wakeCall{}}, nil
}

// path returns the on-disk file for id, refusing an id that is not a single safe
// path element (so a frozen file can never escape dir).
func (s *HibernationStore) path(id string) (string, error) {
	if id == "" || id == "." || id == ".." || filepath.Base(id) != id {
		return "", ErrUnsafeHibernateID
	}
	return filepath.Join(s.dir, id+".hib"), nil
}

// Park freezes h and writes its bytes to <dir>/<id>.hib, returning the number of
// bytes parked. After Park the caller may drop its reference to h and release the
// goroutine — Wake reconstructs a byte-identical context from disk.
func (s *HibernationStore) Park(id string, h Hibernable) (int, error) {
	dst, err := s.path(id)
	if err != nil {
		return 0, err
	}
	b, err := h.Freeze()
	if err != nil {
		return 0, fmt.Errorf("microagent: freeze %q: %w", id, err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return 0, fmt.Errorf("microagent: park %q: %w", id, err)
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return 0, fmt.Errorf("microagent: park %q: %w", id, err)
	}
	return len(b), nil
}

// Wake reads id's frozen context from disk, restores it into h via Thaw, and — as
// the built-in no-state-loss check — re-Freezes h and refuses ErrThawMismatch
// unless the re-frozen bytes equal the bytes read (the CRaC afterRestore analog:
// the restore is verified at the boundary, not trusted). On success it removes
// the file, so a woken id no longer counts against on-disk residency. Wake
// returns ErrNotHibernated when id has no frozen context.
//
// Wake is single-flighted per id (#4034): a burst of concurrent wakes on the SAME
// id coalesces to ONE restore whose result fans out to every waiter, instead of N
// goroutines each racing ReadFile→Thaw→re-Freeze→Remove on the same frozen file (a
// double-restore hazard — two of them could Thaw one context into two live residents
// before either Remove landed). Because an id maps to exactly one hibernated agent,
// concurrent callers should target that one agent: the leader's Thaw is the single
// restore, and the followers observe it and return the leader's error without a second
// Thaw or Remove. Distinct ids still wake fully concurrently — the coalescing is
// per-key, never a global lock. A panic in the caller's Hibernable during the
// restore is caught and returned as ErrWakePanicked (to the leader and every
// follower) so one bad agent cannot wedge the id (leave its inflight slot and done
// channel dangling) and deadlock every later wake of it.
func (s *HibernationStore) Wake(id string, h Hibernable) error {
	if _, err := s.path(id); err != nil {
		return err // refuse an unsafe id up front, before it can pollute the inflight map
	}
	s.mu.Lock()
	if call, ok := s.inflight[id]; ok {
		// A wake of this id is already in flight: join it and share the leader's
		// outcome. No second Thaw, no second Remove — the stampede is coalesced.
		s.mu.Unlock()
		<-call.done
		return call.err
	}
	call := &wakeCall{done: make(chan struct{})}
	s.inflight[id] = call
	s.mu.Unlock()

	// Run the single restore under a recover so a panic in the caller-supplied
	// Hibernable (Thaw/Freeze) cannot wedge the id. A panicking leader that never
	// reached the cleanup below would leave its inflight entry in the map and its
	// done channel open forever, deadlocking every current AND future waiter on this
	// id and leaking their goroutines. The recover turns the panic into a loud
	// ErrWakePanicked — fanned out to every waiter exactly like a normal wake outcome
	// — instead of unwinding past the cleanup or crashing a coalesced worker (#4034).
	func() {
		defer func() {
			if r := recover(); r != nil {
				call.err = fmt.Errorf("microagent: wake %q: %w: %v", id, ErrWakePanicked, r)
			}
		}()
		call.err = s.wakeOnce(id, h)
	}()

	s.mu.Lock()
	delete(s.inflight, id)
	s.mu.Unlock()
	close(call.done)
	return call.err
}

// wakeOnce is the actual single restore of an id's frozen context. It is only ever run
// by the single-flight leader in Wake; followers share its result via wakeCall.
func (s *HibernationStore) wakeOnce(id string, h Hibernable) error {
	src, err := s.path(id)
	if err != nil {
		return err
	}
	b, err := os.ReadFile(src)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("microagent: wake %q: %w", id, ErrNotHibernated)
		}
		return fmt.Errorf("microagent: wake %q: %w", id, err)
	}
	if err := h.Thaw(b); err != nil {
		return fmt.Errorf("microagent: thaw %q: %w", id, err)
	}
	// No-state-loss check: the restored agent must re-freeze to the same bytes.
	again, err := h.Freeze()
	if err != nil {
		return fmt.Errorf("microagent: wake re-freeze %q: %w", id, err)
	}
	if !bytes.Equal(again, b) {
		return fmt.Errorf("microagent: wake %q: %w", id, ErrThawMismatch)
	}
	if err := os.Remove(src); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("microagent: wake %q: remove frozen file: %w", id, err)
	}
	return nil
}

// Parked reports whether id currently has a frozen context on disk. A malformed
// id is reported as not parked (Park/Wake refuse it loudly).
func (s *HibernationStore) Parked(id string) bool {
	p, err := s.path(id)
	if err != nil {
		return false
	}
	_, err = os.Stat(p)
	return err == nil
}

// ResidentCap bounds how many enrolled agents may be RESIDENT (holding a
// goroutine + in-RAM context) at once, independent of how many are enrolled
// (#2012 scope 3). N enrolled agents share Limit resident slots; the rest are
// parked in a HibernationStore. It is a pure counter — no goroutines, no I/O — so
// a host or scheduler composes it with a HibernationStore without this type
// knowing either. It is safe for concurrent use.
type ResidentCap struct {
	mu       sync.Mutex
	limit    int
	low      int // low-water mark of the warm band (#4035); 0 disables warm refill/park
	resident int
	peak     int // high-water resident count ever reached (the O(R) witness)
}

// NewResidentCap builds a resident cap with the given slot limit; limit <= 0
// selects DefaultResidentCap. It has NO warm band (low-water 0), so WarmRefill and
// WarmPark are always 0 and behavior is identical to a plain hard cap.
func NewResidentCap(limit int) *ResidentCap {
	if limit <= 0 {
		limit = DefaultResidentCap
	}
	return &ResidentCap{limit: limit}
}

// NewResidentCapBand builds a resident cap with a two-watermark WARM BAND (#4035): a
// hard high-water admit cap (high, the same gate NewResidentCap enforces) plus a
// low-water refill mark (low). The band is advisory hysteresis a scheduler consults to
// keep the resident set inside [low, high] without thrashing the hibernation store on
// every admit/release — the worker/agent-process twin of the shipped KV-layer prewarm
// (#810):
//
//   - Below low-water, WarmRefill says how many parked agents to warm-wake now,
//     refilling all the way up to high (not just back to low) so a burst of admits pops
//     from warm residents instead of paying a cold Thaw each.
//   - Above low-water, WarmPark says how many idle residents to warm-park now, draining
//     down to low (not to zero) so a warm reserve survives for the next admit.
//
// high <= 0 selects DefaultResidentCap; low is clamped to [0, high]; low <= 0 disables
// the band (WarmRefill/WarmPark return 0), so it degrades to a plain NewResidentCap. The
// hard Admit/Release gate is unchanged — the band never lets residency exceed high, and
// correctness never depends on a warm hit: a miss only ever costs the status-quo cold start.
func NewResidentCapBand(low, high int) *ResidentCap {
	if high <= 0 {
		high = DefaultResidentCap
	}
	if low < 0 {
		low = 0
	}
	if low > high {
		low = high
	}
	return &ResidentCap{limit: high, low: low}
}

// Admit reserves a resident slot, reporting false without reserving when all
// slots are taken (the caller then hibernates the agent instead of making it
// resident). Every true Admit must be paired with exactly one Release.
func (c *ResidentCap) Admit() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.resident >= c.limit {
		return false
	}
	c.resident++
	if c.resident > c.peak {
		c.peak = c.resident
	}
	return true
}

// Release frees a resident slot previously reserved by a true Admit. Releasing
// with no slot held is a programming error and panics, matching the mismatched
// unlock discipline of the standard library.
func (c *ResidentCap) Release() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.resident == 0 {
		panic("microagent: ResidentCap.Release with no resident slot held")
	}
	c.resident--
}

// Resident reports how many resident slots are currently held.
func (c *ResidentCap) Resident() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.resident
}

// Limit reports the resident-slot cap.
func (c *ResidentCap) Limit() int { return c.limit }

// Peak reports the high-water resident count ever reached — the direct evidence
// that residency stayed ~O(R) while N >> R agents were enrolled: Peak() never
// exceeds Limit() by construction.
func (c *ResidentCap) Peak() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.peak
}

// LowWater reports the warm-band low-water mark (#4035). It is 0 for a plain
// NewResidentCap (no band), positive for a NewResidentCapBand.
func (c *ResidentCap) LowWater() int { return c.low }

// WarmRefill is a pure fold (#4035): how many parked agents a scheduler should
// warm-wake right now to refill the warm band. It returns 0 unless the resident set
// has fallen BELOW the low-water mark; once below, it refills up to the high-water cap
// (Limit), so a single crossing triggers a batch refill rather than one warm-wake per
// admit (the hysteresis that keeps the store from thrashing). The answer is bounded by
// parkedAvailable — you cannot warm-wake more agents than are actually parked. With no
// band (low <= 0) it is always 0, so a plain cap never asks for a refill.
func (c *ResidentCap) WarmRefill(parkedAvailable int) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.low <= 0 || c.resident >= c.low {
		return 0
	}
	want := c.limit - c.resident
	if want > parkedAvailable {
		want = parkedAvailable
	}
	if want < 0 {
		want = 0
	}
	return want
}

// WarmPark is a pure fold (#4035): how many currently-idle resident agents a scheduler
// should warm-park right now, draining the resident set DOWN to the low-water mark
// (never below) so a warm reserve is kept for the next admit instead of decommitting to
// zero and paying a cold Thaw on the next wake. It returns 0 unless resident exceeds
// low-water. idle bounds the answer to agents that are actually parkable (returned from
// Step, holding no live work). With no band (low <= 0) it is always 0.
func (c *ResidentCap) WarmPark(idle int) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.low <= 0 || c.resident <= c.low {
		return 0
	}
	want := c.resident - c.low
	if want > idle {
		want = idle
	}
	if want < 0 {
		want = 0
	}
	return want
}
