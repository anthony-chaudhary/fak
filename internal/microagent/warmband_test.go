package microagent_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// warmAgent is a step-resumable Restorable microagent for the warm-band witnesses
// (#5072): its ENTIRE live context is the turn counter plus the history it appends
// to, so Freeze/Thaw is a faithful, deterministic round-trip and the completion of a
// banded run can be compared BYTE FOR BYTE against a never-hibernated one. It ignores
// the gateway — the band witness needs no model turn — so the tests stay hermetic.
type warmAgent struct {
	id    string
	turns int
	took  int
	hist  []string

	// log and bar are test rigging deliberately OUTSIDE the frozen context. Blank is
	// a method value on the enrolled agent, so every fresh vessel the band restores a
	// cold Thaw into still reports to the same rig — which is what lets the rig
	// observe an agent whose live value was dropped and rebuilt mid-run.
	log *warmLog
	bar *warmBarrier
}

type warmState struct {
	ID    string   `json:"id"`
	Turns int      `json:"turns"`
	Took  int      `json:"took"`
	Hist  []string `json:"hist"`
}

func (a *warmAgent) Step(ctx context.Context, _ microagent.Gateway) (bool, error) {
	a.took++
	a.hist = append(a.hist, fmt.Sprintf("%s:turn:%d", a.id, a.took))
	if a.took == 1 && a.bar != nil {
		// Co-residency barrier, first turn only: it pins enough agents inside Step at
		// once that the FIRST Yield after it provably sees residency ABOVE low-water,
		// which is what makes the warm-park -> warm-hit path a determinism rather than
		// a scheduling coincidence.
		a.bar.arrive(ctx)
	}
	if a.took < a.turns {
		return false, nil
	}
	frozen, err := a.Freeze()
	if err != nil {
		return false, err
	}
	a.log.finish(a.id, frozen)
	return true, nil
}

func (a *warmAgent) Freeze() ([]byte, error) {
	return json.Marshal(warmState{ID: a.id, Turns: a.turns, Took: a.took, Hist: a.hist})
}

func (a *warmAgent) Thaw(b []byte) error {
	var s warmState
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	a.id, a.turns, a.took, a.hist = s.ID, s.Turns, s.Took, s.Hist
	return nil
}

// Blank is the #5072 Restorable seam: the fresh, zero-context vessel a cold Wake
// Thaws into after the band dropped its last reference to the live value.
func (a *warmAgent) Blank() microagent.Hibernable {
	return &warmAgent{log: a.log, bar: a.bar}
}

// plainAgent implements Microagent ONLY (no Freeze/Thaw/Blank), so a Host with the
// band on must step it exactly the pre-band way.
type plainAgent struct {
	turns int
	took  int
}

func (a *plainAgent) Step(context.Context, microagent.Gateway) (bool, error) {
	a.took++
	return a.took >= a.turns, nil
}

// warmLog collects each agent's frozen context at the moment it reported done — the
// per-agent completion witness compared across band-on and band-off runs.
type warmLog struct {
	mu    sync.Mutex
	final map[string][]byte
}

func newWarmLog() *warmLog { return &warmLog{final: map[string][]byte{}} }

func (l *warmLog) finish(id string, frozen []byte) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.final[id] = append([]byte(nil), frozen...)
}

func (l *warmLog) snapshot() map[string][]byte {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make(map[string][]byte, len(l.final))
	for id, b := range l.final {
		out[id] = b
	}
	return out
}

// warmBarrier releases once need callers have arrived, then stays open forever.
type warmBarrier struct {
	mu    sync.Mutex
	n     int
	need  int
	once  sync.Once
	ready chan struct{}
}

func newWarmBarrier(need int) *warmBarrier {
	return &warmBarrier{need: need, ready: make(chan struct{})}
}

func (b *warmBarrier) arrive(ctx context.Context) {
	b.mu.Lock()
	b.n++
	if b.n >= b.need {
		b.once.Do(func() { close(b.ready) })
	}
	b.mu.Unlock()
	select {
	case <-b.ready:
	case <-ctx.Done():
	}
}

// warmRun is one workload driven through a live Host: what every agent completed
// with, how each retired, and the band counters sampled DURING the run (not just at
// the end, so a cap breach mid-run cannot hide behind a tidy final snapshot).
type warmRun struct {
	final    map[string][]byte
	results  map[string]microagent.Result
	stats    microagent.WarmBandStats
	peakWarm int
	peakRes  int
}

// driveWarmWorkload spawns agents step-agents on a live Host and drives them to
// completion under warm (nil warm = the band-off posture), returning the completion
// witness plus the band counters. It is the shared rig for every acceptance bullet.
func driveWarmWorkload(t *testing.T, warm *microagent.WarmBand, agents, turns, workers, barrier int) warmRun {
	t.Helper()
	log := newWarmLog()
	bar := newWarmBarrier(barrier)
	host, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: workers, Warm: warm})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}

	// Sample the band while the workload runs: the cap invariants are about every
	// instant, not the final snapshot.
	stop := make(chan struct{})
	var sampler sync.WaitGroup
	var smu sync.Mutex
	peakWarm, peakRes := 0, 0
	if warm != nil {
		sampler.Add(1)
		go func() {
			defer sampler.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				s := warm.Stats()
				smu.Lock()
				if s.Warm > peakWarm {
					peakWarm = s.Warm
				}
				if s.Resident > peakRes {
					peakRes = s.Resident
				}
				smu.Unlock()
				time.Sleep(time.Millisecond)
			}
		}()
	}

	for i := 0; i < agents; i++ {
		id := fmt.Sprintf("a%d", i)
		if err := host.Spawn(id, &warmAgent{id: id, turns: turns, log: log, bar: bar}); err != nil {
			t.Fatalf("Spawn %q: %v", id, err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := host.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v (the fleet did not finish — a banded agent is stuck)", err)
	}
	host.Close()
	close(stop)
	sampler.Wait()

	run := warmRun{final: log.snapshot(), results: map[string]microagent.Result{}}
	for _, r := range host.Reap() {
		run.results[r.ID] = r
	}
	smu.Lock()
	run.peakWarm, run.peakRes = peakWarm, peakRes
	smu.Unlock()
	if warm != nil {
		run.stats = warm.Stats()
	}
	return run
}

// TestWarmBandLiveHostWiring is the #5072 done-condition witness for scope item 2
// (live wiring): N enrolled step-agents are driven through the LIVE Host.run loop
// with the band ON, and the four acceptance bullets are asserted directly —
//
//	(a) warm hits > 0            — the band actually fires (a re-admit paid 0 Thaw)
//	(b) resident peak <= R       — N enrolled agents shared R slots
//	(c) warm <= the reserve cap  — sampled during the run, not just at the end
//	(d) every agent completes byte-identically to a band-OFF run
//
// (a) is a determinism, not a coincidence: the first-turn barrier pins R agents
// inside Step at once, so the first Yield after it evaluates WarmPark with residency
// at the high-water mark — above low-water — and warm-parks; that id's next Acquire
// is then a reserve Take with no disk round-trip.
func TestWarmBandLiveHostWiring(t *testing.T) {
	const (
		agents  = 6
		turns   = 4
		workers = 4
		low     = 1
		high    = 3
		maxWarm = 2
	)
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: low, High: high, MaxWarm: maxWarm, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()

	on := driveWarmWorkload(t, band, agents, turns, workers, high)
	off := driveWarmWorkload(t, nil, agents, turns, workers, high)

	// (a) the band fired.
	if on.stats.Hits <= 0 {
		t.Errorf("warm hits = %d, want > 0 (the band never served a re-admit): %+v", on.stats.Hits, on.stats)
	}
	// (b) residency stayed O(R) while N were enrolled.
	if on.stats.Peak > high {
		t.Errorf("resident peak = %d, want <= R = %d", on.stats.Peak, high)
	}
	if on.peakRes > high {
		t.Errorf("sampled resident peak = %d, want <= R = %d", on.peakRes, high)
	}
	if agents <= high {
		t.Fatalf("test is not a density witness: %d agents must exceed R = %d", agents, high)
	}
	// (c) the reserve never exceeded its cap.
	if on.stats.WarmCap != maxWarm {
		t.Errorf("warm cap = %d, want %d", on.stats.WarmCap, maxWarm)
	}
	if on.peakWarm > maxWarm {
		t.Errorf("sampled warm peak = %d, want <= cap %d", on.peakWarm, maxWarm)
	}
	// (d) every agent completed, and completed byte-identically to the band-off run.
	if len(off.final) != agents {
		t.Fatalf("band-off baseline completed %d agents, want %d", len(off.final), agents)
	}
	if len(on.final) != agents {
		t.Fatalf("banded run completed %d agents, want %d", len(on.final), agents)
	}
	for id, want := range off.final {
		got, ok := on.final[id]
		if !ok {
			t.Errorf("agent %q never completed under the band", id)
			continue
		}
		if !bytes.Equal(got, want) {
			t.Errorf("agent %q completed with a different context under the band:\n band-on:  %s\n band-off: %s", id, got, want)
		}
	}
	for id, want := range off.results {
		got, ok := on.results[id]
		if !ok {
			t.Errorf("agent %q missing from the banded reap", id)
			continue
		}
		if got.Done != want.Done || got.Steps != want.Steps || got.Err != nil || want.Err != nil {
			t.Errorf("agent %q retired differently: band-on %+v, band-off %+v", id, got, want)
		}
	}
	t.Logf("band on: %+v", on.stats)
}

// TestWarmBandColdThawDensity is the #5072 net-true-value witness: the SAME
// re-admit-heavy workload pays strictly fewer cold Thaws on the critical path with
// the band on than with it off, and the band-off posture stays byte-identical.
//
// Honest accounting, stated so the number cannot be read as more than it is: Thaws
// counts the cold Freeze -> disk -> Thaw round-trips a step actually WAITED on. The
// background producer's refills are additional disk wakes — they are paid OFF the
// critical path and are reported separately as Refills, never netted against Thaws.
// The identity Hits + Thaws == one per Step is asserted so no Acquire goes unaccounted.
func TestWarmBandColdThawDensity(t *testing.T) {
	const (
		agents  = 6
		turns   = 4
		workers = 4
		high    = 3
		steps   = agents * turns
	)
	banded, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: 1, High: high, MaxWarm: 2, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand (banded): %v", err)
	}
	defer banded.Close()
	// Low 0 / MaxWarm 0 is the in-band off posture: WarmPark and WarmRefill are always
	// 0 and Reserve always refuses, so every Yield cold-parks and every Acquire is a
	// cold Wake — a plain hibernation store, exactly today's behavior.
	flat, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: 0, High: high, MaxWarm: 0, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand (flat): %v", err)
	}
	defer flat.Close()

	on := driveWarmWorkload(t, banded, agents, turns, workers, high)
	off := driveWarmWorkload(t, flat, agents, turns, workers, high)

	if got := on.stats.Hits + on.stats.Thaws; got != steps {
		t.Errorf("banded acquires accounted = %d (hits %d + thaws %d), want one per step = %d",
			got, on.stats.Hits, on.stats.Thaws, steps)
	}
	if off.stats.Hits != 0 {
		t.Errorf("band-off hits = %d, want 0 (a cap-0 reserve must never serve a re-admit)", off.stats.Hits)
	}
	if off.stats.Refills != 0 {
		t.Errorf("band-off refills = %d, want 0 (a cap-0 reserve must never be refilled)", off.stats.Refills)
	}
	if off.stats.Thaws != steps {
		t.Errorf("band-off thaws = %d, want one cold wake per step = %d", off.stats.Thaws, steps)
	}
	if on.stats.Thaws >= off.stats.Thaws {
		t.Errorf("cold thaws on the critical path: band-on %d, band-off %d — want strictly fewer with the band on",
			on.stats.Thaws, off.stats.Thaws)
	}
	for id, want := range off.final {
		got, ok := on.final[id]
		if !ok {
			t.Errorf("agent %q never completed under the band", id)
			continue
		}
		if !bytes.Equal(got, want) {
			t.Errorf("agent %q: band-on and band-off completions differ:\n on:  %s\n off: %s", id, got, want)
		}
	}
	t.Logf("density: band-on thaws=%d hits=%d refills=%d sheds=%d | band-off thaws=%d",
		on.stats.Thaws, on.stats.Hits, on.stats.Refills, on.stats.Sheds, off.stats.Thaws)
}

// TestWarmBandYieldServesReAdmitAtZeroThaw pins the warm-park -> warm-hit path
// deterministically at the band's own seam, with no Host and no scheduling luck: with
// two residents above a low-water of 1, a Yield warm-parks, and that id's next
// Acquire is served from the reserve with NO additional cold Thaw.
func TestWarmBandYieldServesReAdmitAtZeroThaw(t *testing.T) {
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: 1, High: 2, MaxWarm: 2, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()

	log := newWarmLog()
	for _, id := range []string{"a0", "a1"} {
		if err := band.Enroll(id, &warmAgent{id: id, turns: 4, log: log}); err != nil {
			t.Fatalf("Enroll %q: %v", id, err)
		}
	}
	ctx := context.Background()
	for _, id := range []string{"a0", "a1"} {
		if _, err := band.Acquire(ctx, id); err != nil {
			t.Fatalf("Acquire %q: %v", id, err)
		}
	}
	if got := band.Stats().Resident; got != 2 {
		t.Fatalf("resident = %d after two acquires, want 2", got)
	}
	// Residency (2) is above low-water (1), so the fold warm-parks instead of
	// decommitting to disk. Deltas, not absolutes: the background producer may have
	// pre-warmed either id before the first Acquire, which is exactly its job.
	before := band.Stats()
	if err := band.Yield("a0"); err != nil {
		t.Fatalf("Yield: %v", err)
	}
	if got := band.Stats().Warm; got != 1 {
		t.Fatalf("warm = %d after a yield above low-water, want 1 (it cold-parked instead)", got)
	}
	if _, err := band.Acquire(ctx, "a0"); err != nil {
		t.Fatalf("re-Acquire: %v", err)
	}
	after := band.Stats()
	if after.Hits != before.Hits+1 {
		t.Errorf("hits %d -> %d, want exactly one warm hit on the re-admit", before.Hits, after.Hits)
	}
	if after.Thaws != before.Thaws {
		t.Errorf("thaws %d -> %d, want NO cold round-trip on a warm re-admit", before.Thaws, after.Thaws)
	}
	if after.Warm > after.WarmCap {
		t.Errorf("warm = %d exceeds cap %d", after.Warm, after.WarmCap)
	}
	band.Retire("a0")
	band.Retire("a1")
	if got := band.Stats().Resident; got != 0 {
		t.Errorf("resident = %d after retiring both, want 0", got)
	}
}

// TestWarmBandProducerRefillsBelowLowWater is the #5072 witness for scope item 1: the
// ACQUIRE-SIDE BACKGROUND PRODUCER. Nothing acquires here at all — residency sits at
// 0, below low-water — so any agent that ends up warm was warm-woken off disk by the
// producer goroutine, in advance and off the critical path, bounded by the reserve cap.
func TestWarmBandProducerRefillsBelowLowWater(t *testing.T) {
	const (
		enrolled = 4
		maxWarm  = 3
	)
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: 2, High: 4, MaxWarm: maxWarm, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()

	log := newWarmLog()
	for i := 0; i < enrolled; i++ {
		id := fmt.Sprintf("a%d", i)
		if err := band.Enroll(id, &warmAgent{id: id, turns: 4, log: log}); err != nil {
			t.Fatalf("Enroll %q: %v", id, err)
		}
	}

	s := waitWarm(t, band, maxWarm)
	if s.Warm != maxWarm {
		t.Fatalf("warm = %d, want exactly the cap %d — the producer overfilled the reserve: %+v", s.Warm, maxWarm, s)
	}
	if s.Refills < maxWarm {
		t.Errorf("refills = %d, want >= %d (each warm agent is one producer refill)", s.Refills, maxWarm)
	}
	if s.Warm > s.WarmCap {
		t.Errorf("warm = %d exceeds cap %d — the producer overfilled the reserve", s.Warm, s.WarmCap)
	}
	if s.Parked != enrolled-maxWarm {
		t.Errorf("parked = %d, want %d (the producer must stop at the cap, not drain the store)", s.Parked, enrolled-maxWarm)
	}
	if s.Thaws != 0 {
		t.Errorf("thaws = %d, want 0 — the producer's wakes are paid OFF the critical path", s.Thaws)
	}
	if s.Resident != 0 {
		t.Errorf("resident = %d, want 0 — the producer must not hold resident slots", s.Resident)
	}

	// The point of the pre-warm: the acquire that follows is served from the reserve.
	if _, err := band.Acquire(context.Background(), "a0"); err != nil {
		t.Fatalf("Acquire after refill: %v", err)
	}
	if got := band.Stats(); got.Hits != 1 || got.Thaws != 0 {
		t.Errorf("after a pre-warmed acquire: hits = %d thaws = %d, want 1 and 0", got.Hits, got.Thaws)
	}
}

// TestWarmBandProducerStopsAtHighWater pins the #5072 honest tension as an executable
// bound: the producer must never hold more contexts in RAM than the high-water cap the
// caller sized from the dispatch-side effective worker cap, EVEN when MaxWarm is set
// above it. MaxWarm is a plain caller-supplied int this package cannot validate, so the
// producer — not the reserve — has to be the binding constraint.
//
// The invariant is warm + resident <= High, because a warm agent holds its whole in-RAM
// context and therefore costs what a resident costs. Residency sits at 0 here, so the
// whole cap is available to the reserve and the bound reduces to warm <= High.
//
// This is a REGRESSION witness. Before the fix, WarmRefill's answer was recomputed
// identically on every pass (a refill never moves residency), so it never bounded the
// batch and the producer drained the store all the way to MaxWarm: with these numbers it
// warmed all 8 enrolled agents against a high-water cap of 2.
func TestWarmBandProducerStopsAtHighWater(t *testing.T) {
	const (
		enrolled = 8
		high     = 2
		maxWarm  = 8 // deliberately ABOVE high: the producer must still stop at high
	)
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: 1, High: high, MaxWarm: maxWarm, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()

	log := newWarmLog()
	for i := 0; i < enrolled; i++ {
		id := fmt.Sprintf("hw%d", i)
		if err := band.Enroll(id, &warmAgent{id: id, turns: 4, log: log}); err != nil {
			t.Fatalf("Enroll %q: %v", id, err)
		}
	}

	// The producer must reach the cap (it is genuinely producing, not merely inert)...
	waitWarm(t, band, high)
	// ...and then STOP there. Every Enroll above kicked it, so an unbounded producer
	// overshoots within milliseconds — the settle window catches it far more cheaply
	// than it took to reach the cap in the first place.
	peak := 0
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if w := band.Stats().Warm; w > peak {
			peak = w
		}
		time.Sleep(2 * time.Millisecond)
	}
	if peak > high {
		t.Errorf("warm peaked at %d with residency 0 and a high-water cap of %d — the producer "+
			"pinned more in-RAM contexts than the effective cap allows", peak, high)
	}
	s := band.Stats()
	if s.Warm != high {
		t.Errorf("warm = %d, want exactly the high-water cap %d", s.Warm, high)
	}
	if s.Refills != high {
		t.Errorf("refills = %d, want exactly %d — the producer must stop at the cap, not drain the store",
			s.Refills, high)
	}
	if s.Parked != enrolled-high {
		t.Errorf("parked = %d, want %d (the rest must stay frozen on disk)", s.Parked, enrolled-high)
	}
	if s.Thaws != 0 {
		t.Errorf("thaws = %d, want 0 — nothing acquired, so no cold wake was on any critical path", s.Thaws)
	}
	// The bound must not have cost correctness: a still-parked agent still acquires.
	if _, err := band.Acquire(context.Background(), fmt.Sprintf("hw%d", enrolled-1)); err != nil {
		t.Fatalf("Acquire of an agent the bound left parked: %v", err)
	}
	if got := band.Stats().Thaws; got != 1 {
		t.Errorf("thaws = %d after acquiring a parked agent, want exactly 1 cold wake", got)
	}
}

// waitWarm blocks until the producer has warmed want agents into the reserve, failing
// the test if it never gets there. It is the shared "the pre-warm has happened" gate.
func waitWarm(t *testing.T, band *microagent.WarmBand, want int) microagent.WarmBandStats {
	t.Helper()
	var s microagent.WarmBandStats
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		s = band.Stats()
		if s.Warm >= want {
			return s
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("producer never warmed %d agents: %+v", want, s)
	return s
}

// TestWarmBandCloseDrainsReserveToDisk pins the shutdown contract: a warm agent holds the
// ONLY copy of its context (HibernationStore.Wake removes the frozen file it restored
// from), so Close must hand every warm agent back to disk. After Close each enrolled
// agent is parked again and the reserve is empty — no context lives only in a pool
// nothing will ever drain.
func TestWarmBandCloseDrainsReserveToDisk(t *testing.T) {
	const (
		enrolled = 4
		maxWarm  = 3
	)
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: 2, High: 4, MaxWarm: maxWarm, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	log := newWarmLog()
	for i := 0; i < enrolled; i++ {
		id := fmt.Sprintf("a%d", i)
		if err := band.Enroll(id, &warmAgent{id: id, turns: 4, log: log}); err != nil {
			t.Fatalf("Enroll %q: %v", id, err)
		}
	}
	warmed := waitWarm(t, band, maxWarm)
	if warmed.Parked != enrolled-maxWarm {
		t.Fatalf("parked = %d before close, want %d: %+v", warmed.Parked, enrolled-maxWarm, warmed)
	}

	band.Close()
	band.Close() // idempotent: a second Close must not double-drain or panic

	s := band.Stats()
	if s.Warm != 0 {
		t.Errorf("warm = %d after Close, want 0 (the reserve was not drained)", s.Warm)
	}
	if s.Parked != enrolled {
		t.Errorf("parked = %d after Close, want all %d enrolled contexts back on disk: %+v", s.Parked, enrolled, s)
	}
	// The band refuses work once closed, rather than silently accepting an agent it can
	// no longer produce for.
	if err := band.Enroll("late", &warmAgent{id: "late", turns: 1, log: log}); err != microagent.ErrWarmBandClosed {
		t.Errorf("Enroll after Close = %v, want ErrWarmBandClosed", err)
	}
	if _, err := band.Acquire(context.Background(), "a0"); err != microagent.ErrWarmBandClosed {
		t.Errorf("Acquire after Close = %v, want ErrWarmBandClosed", err)
	}
}

// TestWarmBandYieldAfterCloseColdParks pins the other half of the shutdown contract: a
// Yield that races Close (the Host was closed in the wrong order) must cold-park to disk
// instead of warm-parking into a reserve that has already been drained — otherwise that
// agent's context would sit in a pool nothing drains again, with no frozen file behind it.
func TestWarmBandYieldAfterCloseColdParks(t *testing.T) {
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: 1, High: 2, MaxWarm: 2, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	log := newWarmLog()
	for _, id := range []string{"a0", "a1"} {
		if err := band.Enroll(id, &warmAgent{id: id, turns: 4, log: log}); err != nil {
			t.Fatalf("Enroll %q: %v", id, err)
		}
	}
	ctx := context.Background()
	for _, id := range []string{"a0", "a1"} {
		if _, err := band.Acquire(ctx, id); err != nil {
			t.Fatalf("Acquire %q: %v", id, err)
		}
	}
	band.Close()

	// Residency is 2, above the low-water of 1, so an OPEN band would warm-park here.
	if err := band.Yield("a0"); err != nil {
		t.Fatalf("Yield after Close: %v", err)
	}
	s := band.Stats()
	if s.Warm != 0 {
		t.Errorf("warm = %d after a post-Close yield, want 0 (it warm-parked into a drained reserve)", s.Warm)
	}
	if s.Parked != 1 {
		t.Errorf("parked = %d after a post-Close yield, want 1 (the context must reach disk): %+v", s.Parked, s)
	}
	if s.Resident != 1 {
		t.Errorf("resident = %d after yielding one of two, want 1", s.Resident)
	}
}

// TestWarmBandSkipsNonRestorableAgent pins the band's opt-in edge: an agent that does
// not implement Restorable is never enrolled, never takes residency, and never touches
// the store — it is stepped exactly the pre-band way even with the band configured.
func TestWarmBandSkipsNonRestorableAgent(t *testing.T) {
	band, err := microagent.NewWarmBand(microagent.WarmBandConfig{
		Low: 1, High: 2, MaxWarm: 2, Dir: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()

	host, err := microagent.NewHost(stubPlanner{}, microagent.Config{Workers: 2, Warm: band})
	if err != nil {
		t.Fatalf("NewHost: %v", err)
	}
	if err := host.Spawn("plain", &plainAgent{turns: 3}); err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := host.Drain(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}
	host.Close()

	results := host.Reap()
	if len(results) != 1 || !results[0].Done || results[0].Steps != 3 {
		t.Fatalf("non-Restorable agent retired as %+v, want done in 3 steps", results)
	}
	s := band.Stats()
	if s.Parked != 0 || s.Warm != 0 || s.Thaws != 0 || s.Hits != 0 || s.Peak != 0 {
		t.Errorf("band touched a non-Restorable agent: %+v", s)
	}
}

func TestWarmBandRecoverAfterProcessRestart(t *testing.T) {
	dir := t.TempDir()
	log := newWarmLog()
	original := &warmAgent{id: "restartable", turns: 2, log: log}

	first, err := microagent.NewWarmBand(microagent.WarmBandConfig{Dir: dir, Low: 1, High: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Enroll("restartable", original); err != nil {
		t.Fatal(err)
	}
	h, err := first.Acquire(context.Background(), "restartable")
	if err != nil {
		t.Fatal(err)
	}
	if done, err := h.Step(context.Background(), nil); err != nil || done {
		t.Fatalf("first step done=%v err=%v", done, err)
	}
	if err := first.Yield("restartable"); err != nil {
		t.Fatal(err)
	}
	first.Close()

	second, err := microagent.NewWarmBand(microagent.WarmBandConfig{Dir: dir, Low: 1, High: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	blank := &warmAgent{log: log}
	if err := second.Recover("restartable", blank); err != nil {
		t.Fatal(err)
	}
	h, err = second.Acquire(context.Background(), "restartable")
	if err != nil {
		t.Fatal(err)
	}
	if done, err := h.Step(context.Background(), nil); err != nil || !done {
		t.Fatalf("resumed step done=%v err=%v", done, err)
	}
	second.Retire("restartable")
	if got := second.Stats(); got.Resident != 0 || got.Parked != 0 {
		t.Fatalf("post-retire stats=%+v", got)
	}
	var state warmState
	if err := json.Unmarshal(log.snapshot()["restartable"], &state); err != nil {
		t.Fatal(err)
	}
	if state.Took != 2 || len(state.Hist) != 2 {
		t.Fatalf("restored state=%+v", state)
	}
}
