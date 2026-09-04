package microagent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Warm-band ACQUIRE side: the background refill producer plus the live Host wiring
// (#5072, follow-on to #4035).
//
// #4035 shipped the band's PRIMITIVES: the ResidentCap folds WarmRefill / WarmPark
// (hibernate.go — how many agents to keep warm) and the WarmReserve (warmreserve.go — a
// bounded, per-id pool of still-live agents a Release parks warm and a same-id re-Admit
// pops back at ZERO Thaw). Both are inert on their own: a scheduler had to call the folds
// and Reserve / Take by hand, and nothing composed them with a HibernationStore. WarmBand
// is the missing MECHANISM on the acquire side:
//
//   - It composes the three (ResidentCap + WarmReserve + HibernationStore) behind one
//     per-step residency cycle — Enroll, Acquire, Yield, Retire — so the Host Step loop
//     (microagent.go) drives N enrolled agents through R resident slots with no
//     hand-rolled fold arithmetic at the call site.
//   - It runs the ACQUIRE-SIDE BACKGROUND PRODUCER: one bounded goroutine that, whenever
//     WarmRefill(parkedAvailable) > 0 (residency has fallen below the low-water mark),
//     warm-wakes parked agents off disk into the reserve up to the high-water cap. The
//     next burst of admits then pops warm at 0 Thaw instead of each paying a cold
//     Freeze -> disk -> Thaw. That round-trip is the documented cold-start tax
//     (cmd/tokendemo/cold.go); the producer pays it OFF the critical path, in advance.
//   - It sheds on a staleness horizon: a warm agent that has sat past Horizon is Taken
//     back out of the reserve and cold-parked to disk, so the pool cannot quietly
//     accumulate stale contexts holding RAM.
//
// Honest tension, carried from the issue: a warm agent accrues context/token cost and goes
// stale. The reserve therefore stays tiny (WarmBandConfig.MaxWarm, which the caller sets at
// or below the dispatch-side effective worker cap), is cap-bounded TWICE over — by
// construction (Reserve refuses past its limit, so Warm never exceeds Cap) and by the
// producer, which keeps warm + resident <= the high-water cap even when MaxWarm is set above
// it, since a warm agent holds a whole context and so costs what a resident costs — and is
// horizoned. CORRECTNESS NEVER
// DEPENDS ON A WARM HIT: every Acquire miss falls through to the cold HibernationStore.Wake,
// so a miss costs exactly the status-quo cold start and never a lost agent. A shed
// cold-parks rather than discards — discarding a live enrolled agent mid-flight would lose
// its work, a strictly worse failure than one more Thaw.
//
// Band OFF stays byte-identical to today: a Host with a nil Config.Warm never enrolls, never
// takes residency, and never touches a store — the Step loop is the pre-band loop. A
// WarmBand built with Low 0 / MaxWarm 0 degrades the same way inside the band: WarmPark and
// WarmRefill are always 0 and Reserve always refuses, so every Acquire is a cold Wake.
//
// Generation intent: gen/future (#2012), inherited from the hibernation seam this stands on
// — an OPTION behind an explicit seam. Nothing in the default fak serve / fak guard /
// dispatch path constructs a WarmBand. Closing evidence for the generation frame:
//
//   - Promotion evidence: warmband_test.go drives N enrolled step-agents through the LIVE
//     Host loop with the band on and witnesses the four acceptance bullets — warm hits > 0,
//     resident peak <= R, the reserve never exceeding its cap, and every agent completing
//     byte-identically — plus a cold-Thaw density measurement in which the same workload
//     pays strictly fewer cold Thaws with the band on than with it off, and the band-off run
//     returns byte-identical results. Promote once a real (non-test) dispatch path targets a
//     banded Host AND an RSS/density measurement confirms the parked in-RAM context was the
//     binding cost (the #2000 M8-M12 footprint thesis).
//   - Demotion / retirement criteria: retire the acquire side if that density measurement
//     shows a cold Thaw is cheap relative to the RAM a warm agent holds (the reserve then
//     buys nothing but pressure), or if enrolled agents are re-admitted so rarely that the
//     producer only ever warms contexts the horizon sheds before any Take reaches them.
//   - Invalidating assumption: the producer assumes a parked agent is LIKELY to be
//     re-admitted soon, so pre-warming it pays. It refills in park order (FIFO), which is
//     right for the round-robin step loop a Host runs and wrong for a workload whose next
//     admit is unpredictable — there every refill is wasted RAM plus a shed, and the
//     producer would have to be driven by a real demand signal instead of park order.
//
// WarmBand is safe for concurrent use.

// Structured refusals for the warm band.
var (
	// ErrWarmBandClosed is returned by Enroll/Acquire once Close has run.
	ErrWarmBandClosed = errors.New("microagent: warm band is closed")
	// ErrNotEnrolled refuses an Acquire/Yield for an id the band never enrolled (or has
	// already retired) — the band can only hand back an agent it was given.
	ErrNotEnrolled = errors.New("microagent: warm band has no enrolled agent for id")
	// ErrNilBlank refuses a Restorable whose Blank returned nil: the band cannot Thaw a
	// frozen context without a vessel to restore it into.
	ErrNilBlank = errors.New("microagent: Restorable.Blank returned nil (no vessel for a cold Thaw)")
	// ErrNoStoreDir refuses a WarmBand with no hibernation directory.
	ErrNoStoreDir = errors.New("microagent: WarmBand requires a hibernation store dir")
)

// Restorable is the OPTIONAL extra seam a Hibernable implements to be driven by a WarmBand.
// Blank returns a FRESH, zero-context sibling value — the empty agent a cold
// HibernationStore.Wake Thaws the frozen bytes into. It is exactly what lets the band drop
// its last reference to a parked agent (freeing the in-RAM context, which is the entire
// point of hibernation) and still reconstruct it later: Freeze/Thaw move the state, Blank
// supplies the vessel.
//
// Blank must return a value whose Thaw accepts this agent's frozen bytes. Returning a
// partially-populated value is a bug the store's byte-identity check catches loudly as
// ErrThawMismatch rather than silently resuming a wrong context. A Microagent that does NOT
// implement Restorable is simply never banded — the Host steps it the pre-band way.
type Restorable interface {
	Hibernable
	Blank() Hibernable
}

// WarmBandConfig sizes a WarmBand. The zero value of every field selects a usable default
// or the off posture.
type WarmBandConfig struct {
	// Low and High are the band's two watermarks, handed to NewResidentCapBand: High is the
	// hard resident cap R (<= 0 selects DefaultResidentCap), Low the refill mark (<= 0
	// disables the band, degrading to a plain resident cap).
	Low, High int
	// MaxWarm caps the warm reserve; <= 0 disables it (the byte-identical off posture).
	// Keep it small and at or below BOTH the dispatch-side effective worker cap
	// (dispatchtick.EvaluatePreflight's effective cap) and the per-worker token budget: a
	// warm agent holds its whole in-RAM context, so the reserve trades RAM for Thaws. The
	// bound is taken as a plain int rather than imported, the same way Scheduler takes a
	// plain priority — this package stays decoupled from the dispatch layer.
	//
	// Because it is a plain int, this package cannot validate it against the cap the caller
	// actually sized, so setting it ABOVE High does not buy a bigger warm pool: the producer
	// independently holds warm + resident <= High. An oversized MaxWarm is therefore inert
	// rather than dangerous, and High is the one number that bounds total in-RAM contexts.
	MaxWarm int
	// Dir roots the HibernationStore (one <id>.hib file per parked agent). Required.
	Dir string
	// Horizon sheds a warm agent that has sat in the reserve longer than this, cold-parking
	// it back to disk; it also paces the producer. <= 0 disables shedding, so a warm agent
	// stays warm until it is Taken.
	Horizon time.Duration
	// Now is an injectable clock for the staleness horizon. Nil selects time.Now.
	Now func() time.Time
}

// WarmBandStats is a snapshot of the band's counters — the density witness. Hits vs Thaws
// is the net-true-value readout: a Hit is a re-admit that paid ZERO Thaw, a Thaw is the
// status-quo cold start the band failed (or declined) to remove.
type WarmBandStats struct {
	Hits     int // warm Takes: re-admits served from the reserve at 0 Thaw
	Thaws    int // cold Wakes: full Freeze -> disk -> Thaw round-trips paid
	Refills  int // agents the background producer warm-woke into the reserve
	Sheds    int // warm agents cold-parked again for sitting past the horizon
	Warm     int // agents held warm right now (never exceeds the reserve cap)
	WarmCap  int // the reserve's cap; 0 means the reserve is off
	Parked   int // agents frozen on disk right now
	Resident int // resident slots held right now
	Peak     int // high-water residency ever reached (never exceeds High)
}

// WarmBand composes a ResidentCap, a WarmReserve, and a HibernationStore into one live
// residency cycle, and runs the background refill producer behind it. See the package note
// above for the seam, the honest tension, and the generation frame.
type WarmBand struct {
	rc      *ResidentCap
	reserve *WarmReserve
	store   *HibernationStore
	horizon time.Duration
	now     func() time.Time

	mu      sync.Mutex
	parked  []string                     // ids frozen on disk, in FIFO refill order
	onDisk  map[string]bool              // membership guard for parked
	blanks  map[string]func() Hibernable // per-id vessel factory, kept after the value is dropped
	held    map[string]Hibernable        // ids holding a resident slot right now
	warmAt  map[string]time.Time         // when each reserved id entered the reserve
	warming map[string]bool              // ids with one claimed warm/cold wake resolver
	freed   chan struct{}                // broadcast generation: residency freed or a warm agent landed

	hits, thaws, refills, sheds int
	closed                      bool

	coldMu     sync.RWMutex
	coldStates map[string]*HibernatedState

	nudge chan struct{} // producer kick (buffered 1, coalescing)
	done  chan struct{}
	wg    sync.WaitGroup
}

// NewWarmBand builds a warm band over a fresh HibernationStore rooted at cfg.Dir and starts
// its single background refill producer. The caller owns Close.
func NewWarmBand(cfg WarmBandConfig) (*WarmBand, error) {
	if cfg.Dir == "" {
		return nil, ErrNoStoreDir
	}
	store, err := NewHibernationStore(cfg.Dir)
	if err != nil {
		return nil, err
	}
	nowFn := cfg.Now
	if nowFn == nil {
		nowFn = time.Now
	}
	b := &WarmBand{
		rc:         NewResidentCapBand(cfg.Low, cfg.High),
		reserve:    NewWarmReserve(cfg.MaxWarm),
		store:      store,
		horizon:    cfg.Horizon,
		now:        nowFn,
		onDisk:     map[string]bool{},
		blanks:     map[string]func() Hibernable{},
		held:       map[string]Hibernable{},
		warmAt:     map[string]time.Time{},
		warming:    map[string]bool{},
		coldStates: map[string]*HibernatedState{},
		freed:      make(chan struct{}),
		nudge:      make(chan struct{}, 1),
		done:       make(chan struct{}),
	}
	b.wg.Add(1)
	go b.produce()
	return b, nil
}

// Enroll registers a fresh agent with the band and parks its (turn 0) context to disk, so
// enrollment costs ONE file and no goroutine — the O(R)-residency posture where N enrolled
// agents share R resident slots. The Blank factory is remembered so the band can rebuild the
// agent on a later cold Wake after dropping its last reference to the value. Enrolling an id
// the band already holds refuses with ErrDuplicateID (an id is one agent lifetime).
func (b *WarmBand) Enroll(id string, r Restorable) error {
	if r == nil {
		return ErrNilAgent
	}
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return ErrWarmBandClosed
	}
	if _, dup := b.blanks[id]; dup {
		b.mu.Unlock()
		return ErrDuplicateID
	}
	b.blanks[id] = r.Blank // claim the id before the I/O so a concurrent Enroll cannot race it
	b.mu.Unlock()

	if _, err := b.store.Park(id, r); err != nil {
		b.mu.Lock()
		delete(b.blanks, id)
		b.mu.Unlock()
		return err
	}
	_, _ = b.recordColdState(id, r)
	b.mu.Lock()
	b.addParkedLocked(id)
	b.mu.Unlock()
	b.kick()
	return nil
}

// Recover registers an agent whose frozen context already exists in the band's
// HibernationStore. Unlike Enroll it never rewrites the snapshot. This is the
// process-restart seam: a fresh WarmBand can rebuild its in-memory registry from
// a durable list of ids while preserving each agent's last committed state.
func (b *WarmBand) Recover(id string, r Restorable) error {
	if r == nil {
		return ErrNilAgent
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return ErrWarmBandClosed
	}
	if _, dup := b.blanks[id]; dup {
		return ErrDuplicateID
	}
	if !b.store.Parked(id) {
		return fmt.Errorf("microagent: warm band recover %q: %w", id, ErrNotEnrolled)
	}
	b.blanks[id] = r.Blank
	b.addParkedLocked(id)
	b.kick()
	return nil
}

// Acquire takes a resident slot for id and returns the LIVE agent to Step. It resolves the
// agent in the order that makes a warm hit free and a miss merely ordinary:
//
//  1. Already held — the id is mid-retry and never gave its slot back: the same value, no
//     second admit, no round-trip.
//  2. The warm reserve — a Take hit, the ZERO-Thaw re-admit this whole seam exists for.
//  3. The cold store — a full Wake into a Blank vessel, the status-quo cold start.
//
// It blocks until a resident slot frees (never busy-waits: it parks on a broadcast channel
// signalled by every release) and returns ctx.Err() if the context ends first, or
// ErrNotEnrolled for an id the band does not hold. Every successful Acquire must be matched
// by exactly one Yield (step boundary) or Retire (terminal).
func (b *WarmBand) Acquire(ctx context.Context, id string) (Hibernable, error) {
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		b.mu.Lock()
		if b.closed {
			b.mu.Unlock()
			return nil, ErrWarmBandClosed
		}
		if _, ok := b.blanks[id]; !ok {
			b.mu.Unlock()
			return nil, fmt.Errorf("microagent: warm band acquire %q: %w", id, ErrNotEnrolled)
		}
		if h, ok := b.held[id]; ok {
			b.mu.Unlock()
			return h, nil
		}
		// Read the broadcast generation BEFORE testing the condition, so a release that
		// lands in between wakes us instead of being lost.
		wait, midWake := b.freed, b.warming[id]
		b.mu.Unlock()

		// The producer is mid warm-wake on this exact id: let it finish and land in the
		// reserve rather than racing a second Wake of the same frozen file.
		if midWake {
			if err := b.await(ctx, wait); err != nil {
				return nil, err
			}
			continue
		}
		if !b.rc.Admit() {
			if err := b.await(ctx, wait); err != nil {
				return nil, err
			}
			continue
		}
		// Claim the one live resolver for this id only after obtaining a resident
		// slot, then re-check under the band lock. Without this second check, the
		// refill producer can remove the snapshot after the first warming check but
		// before take reaches the store, leaving this Acquire with no context.
		b.mu.Lock()
		if b.closed {
			b.mu.Unlock()
			b.releaseSlot()
			return nil, ErrWarmBandClosed
		}
		if h, ok := b.held[id]; ok {
			b.mu.Unlock()
			b.releaseSlot()
			return h, nil
		}
		if _, ok := b.blanks[id]; !ok {
			b.mu.Unlock()
			b.releaseSlot()
			return nil, fmt.Errorf("microagent: warm band acquire %q: %w", id, ErrNotEnrolled)
		}
		wait = b.freed
		if b.warming[id] {
			b.mu.Unlock()
			b.releaseSlot()
			if err := b.await(ctx, wait); err != nil {
				return nil, err
			}
			continue
		}
		b.warming[id] = true
		b.mu.Unlock()

		h, err := b.take(id)
		if err != nil {
			b.mu.Lock()
			delete(b.warming, id)
			b.mu.Unlock()
			b.releaseSlot()
			return nil, err
		}
		b.mu.Lock()
		delete(b.warming, id)
		b.held[id] = h
		b.broadcastLocked()
		b.mu.Unlock()
		return h, nil
	}
}

// await parks on the broadcast channel until residency frees or ctx ends. No polling, no
// CPU while waiting.
func (b *WarmBand) await(ctx context.Context, wait <-chan struct{}) error {
	select {
	case <-wait:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// take resolves an admitted id's live agent: the warm reserve first (0 Thaw), else the cold
// store. A warm miss is never lossy — it is exactly the status-quo cold start.
func (b *WarmBand) take(id string) (Hibernable, error) {
	if h, ok := b.reserve.Take(id); ok {
		b.mu.Lock()
		delete(b.warmAt, id)
		b.hits++
		b.mu.Unlock()
		return h, nil
	}
	b.mu.Lock()
	blank := b.blanks[id]
	b.mu.Unlock()
	if blank == nil {
		return nil, fmt.Errorf("microagent: warm band acquire %q: %w", id, ErrNotEnrolled)
	}
	h := blank()
	if h == nil {
		return nil, fmt.Errorf("microagent: warm band acquire %q: %w", id, ErrNilBlank)
	}
	if err := b.store.Wake(id, h); err != nil {
		return nil, err
	}
	b.mu.Lock()
	b.thaws++
	b.dropParkedLocked(id)
	b.mu.Unlock()
	return h, nil
}

// Yield hands back id's resident slot at a clean step boundary (Step returned, not done) and
// decides WHERE the context goes via the #4035 WarmPark fold: while residency is above the
// low-water mark the agent is parked WARM in the reserve (still live, in RAM, 0 Thaw on its
// next Acquire); at or below low-water — or when the reserve is full or already holds the id
// — it is cold-parked to disk exactly as before the band. Either way the band drops its last
// reference to the value, so the goroutine AND the in-RAM context are freed.
//
// The fold is evaluated BEFORE the slot is released, so it sees this agent's own residency.
func (b *WarmBand) Yield(id string) error {
	b.mu.Lock()
	h, ok := b.held[id]
	if !ok {
		b.mu.Unlock()
		return fmt.Errorf("microagent: warm band yield %q: %w", id, ErrNotEnrolled)
	}
	delete(b.held, id)
	closed := b.closed
	b.mu.Unlock()

	// A Yield that races Close cold-parks unconditionally: Close has already drained the
	// reserve, so a warm park here would leave the context in a pool nothing drains again
	// — and with no frozen file behind it, since a Wake removes one. Disk is the posture
	// that never loses state.
	if !closed && b.rc.WarmPark(1) > 0 && b.reserve.Reserve(id, h) {
		b.mu.Lock()
		b.warmAt[id] = b.now()
		b.mu.Unlock()
		b.releaseSlot()
		return nil
	}
	if _, err := b.store.Park(id, h); err != nil {
		b.releaseSlot()
		return err
	}
	_, _ = b.recordColdState(id, h)
	b.mu.Lock()
	b.addParkedLocked(id)
	b.mu.Unlock()
	b.releaseSlot()
	return nil
}

// Retire drops id from the band on a terminal outcome (done, error, or cancel): the resident
// slot is released if one was held, the live value dropped, and any warm residue Taken so the
// reserve does not carry a finished agent. It is idempotent and safe for an id the band never
// enrolled, so a Host can call it on every retirement path unconditionally.
//
// A frozen file for a cancelled id is deliberately LEFT on disk: it is that agent's last
// witnessed context, and deleting it here would discard state the band never owned.
func (b *WarmBand) Retire(id string) {
	b.mu.Lock()
	_, wasHeld := b.held[id]
	delete(b.held, id)
	delete(b.blanks, id)
	delete(b.warmAt, id)
	b.dropParkedLocked(id)
	b.mu.Unlock()

	b.reserve.Take(id) // drop any warm residue; a miss is fine
	if wasHeld {
		b.releaseSlot()
		return
	}
	b.mu.Lock()
	b.broadcastLocked()
	b.mu.Unlock()
	b.kick()
}

// Stats snapshots the band's counters (the density witness).
func (b *WarmBand) Stats() WarmBandStats {
	b.mu.Lock()
	defer b.mu.Unlock()
	return WarmBandStats{
		Hits:     b.hits,
		Thaws:    b.thaws,
		Refills:  b.refills,
		Sheds:    b.sheds,
		Warm:     b.reserve.Len(),
		WarmCap:  b.reserve.Cap(),
		Parked:   len(b.parked),
		Resident: b.rc.Resident(),
		Peak:     b.rc.Peak(),
	}
}

// Close stops the producer, waits for its goroutine to exit, and only then drains the
// reserve back to disk, so no enrolled agent's context is lost to a refill that landed in
// the shutdown window. Idempotent. It does not cancel in-flight Steps; a caller draining a
// Host should Drain/Close the Host first — a Yield that races Close cold-parks rather than
// warm-parks, so even that ordering mistake costs a Thaw and never a context.
func (b *WarmBand) Close() {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	b.closed = true
	close(b.done)
	b.broadcastLocked()
	b.mu.Unlock()

	b.wg.Wait()

	// Snapshot the reserve only AFTER the producer has exited. A refill that landed in
	// the shutdown window (already past its closed check, mid store.Wake) would be
	// missed by a snapshot taken before the wait — and Wake REMOVES the frozen file, so
	// that agent's context would exist nowhere at all. Draining after the join makes the
	// set of warm ids final.
	b.mu.Lock()
	ids := make([]string, 0, len(b.warmAt))
	for id := range b.warmAt {
		ids = append(ids, id)
	}
	b.warmAt = map[string]time.Time{}
	b.mu.Unlock()

	for _, id := range ids {
		h, ok := b.reserve.Take(id)
		if !ok {
			continue
		}
		if _, err := b.store.Park(id, h); err != nil {
			continue // the value is dropped either way; the loud path is the next Acquire
		}
		_, _ = b.recordColdState(id, h)
		b.mu.Lock()
		b.addParkedLocked(id)
		b.mu.Unlock()
	}
}

// produce is the single background refill producer goroutine. It wakes on a kick (an
// enrollment, a yield, a retirement) or on the horizon tick, sheds stale warm agents, then
// refills the reserve while the fold asks for it. One goroutine, never one per agent.
func (b *WarmBand) produce() {
	defer b.wg.Done()
	var tick <-chan time.Time
	if b.horizon > 0 {
		t := time.NewTicker(b.horizon)
		defer t.Stop()
		tick = t.C
	}
	for {
		select {
		case <-b.done:
			return
		case <-b.nudge:
		case <-tick:
		}
		b.shed()
		b.refill()
	}
}

// refill is the acquire-side background production loop: while WarmRefill reports residency
// below the low-water mark, warm-wake parked agents into the reserve one at a time. The batch
// is bounded three ways — by the high-water cap, by how many agents are actually parked, and
// by the reserve's own remaining room — so the warm pool can never exceed EITHER bound.
func (b *WarmBand) refill() {
	for {
		select {
		case <-b.done:
			return
		default:
		}
		b.mu.Lock()
		if b.closed {
			b.mu.Unlock()
			return
		}
		avail := len(b.parked)
		b.mu.Unlock()

		want := b.rc.WarmRefill(avail)
		if want > 0 {
			// Charge the already-warm agents against the high-water cap. The fold answers
			// in RESIDENT-SLOT terms (high-water minus residency) and a refill never moves
			// residency, so on its own the answer is recomputed IDENTICALLY every pass and
			// never bounds the batch: the producer would drain the store all the way to
			// MaxWarm. Whenever MaxWarm is set above the high-water cap (the reserve cap is
			// a caller-supplied int this package cannot validate), that silently pins far
			// more in-RAM contexts than the cap the caller sized from the dispatch-side
			// effective worker cap — the exact overrun the band's honest tension forbids. A
			// warm agent holds its whole context, so it costs what a resident costs; the
			// invariant the producer must keep is warm + resident <= high-water.
			if headroom := b.rc.Limit() - b.rc.Resident() - b.reserve.Len(); want > headroom {
				want = headroom
			}
		}
		if room := b.reserve.Cap() - b.reserve.Len(); want > room {
			want = room
		}
		if want <= 0 || !b.refillOne() {
			return
		}
	}
}

// refillOne warm-wakes the oldest parked agent into the reserve, reporting whether it made
// progress. A failed wake never loses the agent: the id goes back on the parked list and the
// next Acquire pays the cold path, where the same error surfaces loudly to its caller.
func (b *WarmBand) refillOne() bool {
	b.mu.Lock()
	if b.closed || len(b.parked) == 0 {
		b.mu.Unlock()
		return false
	}
	i := -1
	for j, id := range b.parked {
		if !b.warming[id] {
			i = j
			break
		}
	}
	if i < 0 {
		b.mu.Unlock()
		return false
	}
	id := b.parked[i]
	b.parked = append(b.parked[:i], b.parked[i+1:]...)
	delete(b.onDisk, id)
	blank := b.blanks[id]
	if blank == nil { // retired under us — nothing to warm, but keep draining
		b.mu.Unlock()
		return true
	}
	b.warming[id] = true
	b.mu.Unlock()

	h := blank()
	var err error
	if h == nil {
		err = ErrNilBlank
	} else {
		err = b.store.Wake(id, h)
	}

	b.mu.Lock()
	if err != nil {
		delete(b.warming, id)
		b.addParkedLocked(id)
		b.broadcastLocked()
		b.mu.Unlock()
		return false
	}
	if _, enrolled := b.blanks[id]; !enrolled {
		// Retired while this wake was in flight: Retire's own reserve.Take ran before
		// the Reserve below could land, so reserving now would strand a finished
		// agent's context in the pool — pinned in RAM until Close, and handed to no
		// Acquire, since the band no longer knows the id. Drop the value instead and
		// keep draining.
		delete(b.warming, id)
		b.broadcastLocked()
		b.mu.Unlock()
		return true
	}
	if b.reserve.Reserve(id, h) {
		delete(b.warming, id)
		b.warmAt[id] = b.now()
		b.refills++
		b.broadcastLocked()
		b.mu.Unlock()
		return true
	}
	b.mu.Unlock()

	// No room, or the id is already warm: cold-park it straight back rather than hold a
	// live reference the band no longer tracks.
	_, perr := b.store.Park(id, h)
	b.mu.Lock()
	delete(b.warming, id)
	if perr == nil {
		b.addParkedLocked(id)
	}
	b.broadcastLocked()
	b.mu.Unlock()
	return false
}

// shed enforces the staleness horizon: a warm agent that has sat in the reserve longer than
// Horizon is Taken back out and cold-parked to disk. It bounds the honest tension — a warm
// agent holds RAM and its context goes stale — without ever DISCARDING an enrolled agent's
// work: a discard would lose state, while a cold park only costs the next Thaw.
func (b *WarmBand) shed() {
	if b.horizon <= 0 {
		return
	}
	now := b.now()
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return
	}
	var stale []string
	for id, at := range b.warmAt {
		if now.Sub(at) >= b.horizon {
			stale = append(stale, id)
		}
	}
	b.mu.Unlock()

	for _, id := range stale {
		h, ok := b.reserve.Take(id)
		b.mu.Lock()
		delete(b.warmAt, id)
		b.mu.Unlock()
		if !ok {
			continue // Taken by a real Acquire in between: a warm hit beat the shed
		}
		if _, err := b.store.Park(id, h); err != nil {
			continue
		}
		b.mu.Lock()
		b.addParkedLocked(id)
		b.sheds++
		b.broadcastLocked()
		b.mu.Unlock()
	}
}

// releaseSlot returns one resident slot and wakes everyone parked in Acquire.
func (b *WarmBand) releaseSlot() {
	b.rc.Release()
	b.mu.Lock()
	b.broadcastLocked()
	b.mu.Unlock()
	b.kick()
}

// broadcastLocked wakes every current Acquire waiter by closing the generation channel and
// installing a fresh one. Callers must hold b.mu.
func (b *WarmBand) broadcastLocked() {
	close(b.freed)
	b.freed = make(chan struct{})
}

// kick nudges the producer without ever blocking: the channel is buffered 1, so concurrent
// kicks coalesce into one production pass.
func (b *WarmBand) kick() {
	select {
	case b.nudge <- struct{}{}:
	default:
	}
}

// addParkedLocked records id as frozen on disk (idempotent). Callers must hold b.mu.
func (b *WarmBand) addParkedLocked(id string) {
	if b.onDisk[id] {
		return
	}
	b.onDisk[id] = true
	b.parked = append(b.parked, id)
}

// dropParkedLocked forgets id's on-disk entry. Callers must hold b.mu.
func (b *WarmBand) dropParkedLocked(id string) {
	if !b.onDisk[id] {
		return
	}
	delete(b.onDisk, id)
	for i, p := range b.parked {
		if p == id {
			b.parked = append(b.parked[:i], b.parked[i+1:]...)
			return
		}
	}
}
