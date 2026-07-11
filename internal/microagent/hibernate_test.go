package microagent_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// histAgent is a step-resumable Hibernable microagent whose bounded context
// (epic #2000 M4) is a linear history plus a turn counter. It ignores the
// gateway — the hibernation witness needs no model turn — so the tests stay
// hermetic. Freeze/Thaw are a deterministic JSON round-trip: an unchanged
// context always freezes to the same bytes, the byte-identity Wake enforces.
type histAgent struct {
	id    string
	turns int
	took  int
	hist  []string
}

type histState struct {
	ID    string   `json:"id"`
	Turns int      `json:"turns"`
	Took  int      `json:"took"`
	Hist  []string `json:"hist"`
}

func (a *histAgent) Step(_ context.Context, _ microagent.Gateway) (bool, error) {
	a.took++
	a.hist = append(a.hist, fmt.Sprintf("%s:turn:%d", a.id, a.took))
	return a.took >= a.turns, nil
}

func (a *histAgent) Freeze() ([]byte, error) {
	return json.Marshal(histState{ID: a.id, Turns: a.turns, Took: a.took, Hist: a.hist})
}

func (a *histAgent) Thaw(b []byte) error {
	var s histState
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	a.id, a.turns, a.took, a.hist = s.ID, s.Turns, s.Took, s.Hist
	return nil
}

// lossyAgent's Thaw drops its state, so re-freezing after a Wake yields
// different bytes — it drives the ErrThawMismatch no-state-loss guard.
type lossyAgent struct{ n int }

func (a *lossyAgent) Step(context.Context, microagent.Gateway) (bool, error) { return true, nil }
func (a *lossyAgent) Freeze() ([]byte, error)                                { return []byte(fmt.Sprintf("%d", a.n)), nil }
func (a *lossyAgent) Thaw([]byte) error                                      { a.n = 0; return nil }

// TestHibernationRoundTripByteIdentical is the #2012 acceptance witness for
// "hibernate->wake round-trips a context with no state loss": a partially-run
// agent is frozen to disk, its goroutine + RAM dropped, then woken into a FRESH
// value that re-freezes byte-identically and resumes to the exact same
// completion a never-hibernated run reaches. It also pins the refusal edges:
// an unsafe id, an unparked id, and a lossy Thaw the Wake boundary catches.
func TestHibernationRoundTripByteIdentical(t *testing.T) {
	store, err := microagent.NewHibernationStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewHibernationStore: %v", err)
	}

	orig := &histAgent{id: "rt", turns: 5}
	orig.Step(context.Background(), nil) // took=1
	orig.Step(context.Background(), nil) // took=2 — real state to lose
	snap, err := orig.Freeze()
	if err != nil {
		t.Fatalf("Freeze: %v", err)
	}

	n, err := store.Park("rt", orig)
	if err != nil {
		t.Fatalf("Park: %v", err)
	}
	if n != len(snap) {
		t.Errorf("Park reported %d bytes, want %d", n, len(snap))
	}
	if !store.Parked("rt") {
		t.Fatal("Parked(rt) = false after Park, want true")
	}

	// Wake into a FRESH agent — the original goroutine + RAM are gone.
	woken := &histAgent{}
	if err := store.Wake("rt", woken); err != nil {
		t.Fatalf("Wake: %v", err)
	}
	if store.Parked("rt") {
		t.Error("frozen file still present after a successful Wake, want removed")
	}
	got, err := woken.Freeze()
	if err != nil {
		t.Fatalf("woken Freeze: %v", err)
	}
	if !bytes.Equal(got, snap) {
		t.Errorf("woken context not byte-identical:\n got %s\nwant %s", got, snap)
	}

	// Resume both the woken agent and a never-hibernated reference to done; the
	// two runs must end in the same context (no divergence from hibernation).
	ref := &histAgent{id: "rt", turns: 5}
	ref.Step(context.Background(), nil)
	ref.Step(context.Background(), nil)
	driveDone(t, woken)
	driveDone(t, ref)
	wb, _ := woken.Freeze()
	rb, _ := ref.Freeze()
	if !bytes.Equal(wb, rb) {
		t.Errorf("resumed hibernated run diverged from a never-hibernated run:\n hibernated %s\n reference  %s", wb, rb)
	}

	// Refusal edges.
	if err := store.Wake("never", &histAgent{}); !errors.Is(err, microagent.ErrNotHibernated) {
		t.Errorf("Wake(unparked) = %v, want ErrNotHibernated", err)
	}
	if _, err := store.Park("../escape", orig); !errors.Is(err, microagent.ErrUnsafeHibernateID) {
		t.Errorf("Park(unsafe id) = %v, want ErrUnsafeHibernateID", err)
	}
	if err := store.Wake("a/b", &histAgent{}); !errors.Is(err, microagent.ErrUnsafeHibernateID) {
		t.Errorf("Wake(unsafe id) = %v, want ErrUnsafeHibernateID", err)
	}

	// A lossy Thaw is refused at the wake boundary and leaves the frozen copy.
	if _, err := store.Park("lossy", &lossyAgent{n: 7}); err != nil {
		t.Fatalf("Park(lossy): %v", err)
	}
	if err := store.Wake("lossy", &lossyAgent{}); !errors.Is(err, microagent.ErrThawMismatch) {
		t.Errorf("Wake(lossy) = %v, want ErrThawMismatch", err)
	}
	if !store.Parked("lossy") {
		t.Error("a refused (lossy) Wake destroyed the frozen copy, want it preserved")
	}
}

// countAgent counts Thaw calls and, as the single-flight leader, blocks its first
// Thaw on a release channel so the wake-stampede test can hold every waker in flight
// before the one restore completes — then assert exactly one Thaw ran (issue #4034).
type countAgent struct {
	thaws   *int32
	arrived *sync.WaitGroup // Done'd by the leader when its restore begins
	release <-chan struct{} // the leader's Thaw blocks here until the test releases it
	state   []byte
}

func (a *countAgent) Step(context.Context, microagent.Gateway) (bool, error) { return true, nil }
func (a *countAgent) Freeze() ([]byte, error)                                { return append([]byte(nil), a.state...), nil }
func (a *countAgent) Thaw(b []byte) error {
	if atomic.AddInt32(a.thaws, 1) == 1 && a.arrived != nil {
		a.arrived.Done() // leader reached the single restore...
		<-a.release      // ...and holds it open while the stampede piles up behind it
	}
	a.state = append([]byte(nil), b...)
	return nil
}

// TestWakeCoalescesStampedeOnOneID is the #4034 acceptance witness: N goroutines
// barrier-released against the SAME hibernated id coalesce to exactly one Thaw whose
// result fans out to all N (no double-restore, file removed exactly once), instead of
// N racing ReadFile→Thaw→Remove on the one frozen file.
func TestWakeCoalescesStampedeOnOneID(t *testing.T) {
	store, err := microagent.NewHibernationStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewHibernationStore: %v", err)
	}
	if _, err := store.Park("hot", &countAgent{thaws: new(int32), state: []byte("resident-state")}); err != nil {
		t.Fatalf("Park: %v", err)
	}

	const N = 32
	var thaws, entered int32
	var arrived sync.WaitGroup
	arrived.Add(1)
	release := make(chan struct{})
	// One shared sink: an id maps to ONE hibernated agent, so every concurrent waker
	// targets that same agent — the leader's single Thaw restores it for all of them.
	sink := &countAgent{thaws: &thaws, arrived: &arrived, release: release}

	errs := make([]error, N)
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			atomic.AddInt32(&entered, 1)
			errs[i] = store.Wake("hot", sink)
		}(i)
	}

	// Hold the leader inside its Thaw (arrived) until every waker has entered Wake and
	// piled up behind the single inflight call, then release the one restore.
	arrived.Wait()
	for atomic.LoadInt32(&entered) < N {
		runtime.Gosched()
	}
	time.Sleep(25 * time.Millisecond) // let the last entrants reach the inflight join
	close(release)
	wg.Wait()

	if got := atomic.LoadInt32(&thaws); got != 1 {
		t.Errorf("Thaw ran %d times across the stampede, want exactly 1 (single-flight did not coalesce)", got)
	}
	for i, e := range errs {
		if e != nil {
			t.Errorf("waiter %d: Wake = %v, want nil (waiters share the leader's success)", i, e)
		}
	}
	if !bytes.Equal(sink.state, []byte("resident-state")) {
		t.Errorf("restored state = %q, want the parked bytes", sink.state)
	}
	if store.Parked("hot") {
		t.Error("frozen file still present after a coalesced wake, want removed exactly once")
	}

	// Distinct ids still wake concurrently (no global-lock regression): two ids parked
	// and woken in parallel both complete.
	for _, id := range []string{"a", "b"} {
		if _, err := store.Park(id, &countAgent{thaws: new(int32), state: []byte(id)}); err != nil {
			t.Fatalf("Park(%s): %v", id, err)
		}
	}
	var dwg sync.WaitGroup
	derrs := make([]error, 2)
	for i, id := range []string{"a", "b"} {
		dwg.Add(1)
		go func(i int, id string) {
			defer dwg.Done()
			derrs[i] = store.Wake(id, &countAgent{thaws: new(int32)})
		}(i, id)
	}
	dwg.Wait()
	for i, e := range derrs {
		if e != nil {
			t.Errorf("distinct-id wake %d: %v, want nil", i, e)
		}
	}
}

// panicAgent's Thaw panics — the single-flight leader's restore blows up so the test
// can prove a panicking Hibernable cannot wedge the id (#4034): Wake catches the panic
// and returns ErrWakePanicked to the leader AND every follower coalesced behind it,
// instead of deadlocking them on a done channel that never closes. When arrived/release
// are set the leader announces it reached Thaw and holds there until released, so the
// stampede piles up on the one inflight call before it panics.
type panicAgent struct {
	arrived *sync.WaitGroup // Done'd when the leader enters Thaw
	release <-chan struct{} // leader blocks here until the test releases the panic
	state   []byte
}

func (a *panicAgent) Step(context.Context, microagent.Gateway) (bool, error) { return true, nil }
func (a *panicAgent) Freeze() ([]byte, error)                                { return append([]byte(nil), a.state...), nil }
func (a *panicAgent) Thaw([]byte) error {
	if a.arrived != nil {
		a.arrived.Done()
		<-a.release
	}
	panic("panicAgent: boom in Thaw")
}

// TestWakePanicDoesNotWedgeID proves the #4034 single-flight is panic-safe. When the
// caller-supplied Hibernable panics inside the leader's Thaw, Wake catches it and
// returns ErrWakePanicked — to the leader AND to every follower coalesced behind it —
// instead of leaving the inflight entry and done channel dangling, which would deadlock
// every current and future waiter on that id. The frozen file survives the panic (no
// partial Remove), so the id stays wakeable by a healthy agent afterward.
func TestWakePanicDoesNotWedgeID(t *testing.T) {
	store, err := microagent.NewHibernationStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewHibernationStore: %v", err)
	}
	if _, err := store.Park("hot", &histAgent{id: "hot", turns: 3}); err != nil {
		t.Fatalf("Park: %v", err)
	}

	const N = 16
	var arrived sync.WaitGroup
	arrived.Add(1)
	release := make(chan struct{})
	// One shared sink whose Thaw panics: an id maps to ONE hibernated agent, so every
	// waker coalesces onto the leader's single (panicking) restore.
	sink := &panicAgent{arrived: &arrived, release: release}

	errs := make([]error, N)
	done := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = store.Wake("hot", sink)
		}(i)
	}
	go func() { wg.Wait(); close(done) }()

	// Hold the leader inside its panicking Thaw until the stampede has piled up behind
	// the one inflight call, then release it so the single restore panics with followers
	// already waiting on it.
	arrived.Wait()
	time.Sleep(25 * time.Millisecond) // let the followers reach the inflight join
	close(release)

	// A wedged id would hang every waiter forever — bound the wait and fail loudly.
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Wake stampede did not return after a leader panic — the id is wedged (deadlock)")
	}

	for i, e := range errs {
		if !errors.Is(e, microagent.ErrWakePanicked) {
			t.Errorf("waiter %d: Wake = %v, want ErrWakePanicked (panic fanned out to every waiter)", i, e)
		}
	}

	// The id is not wedged: the frozen file survived the panic (no partial Remove) and a
	// healthy agent can still wake it cleanly.
	if !store.Parked("hot") {
		t.Fatal("frozen file removed despite a panicking restore, want it preserved")
	}
	woken := &histAgent{}
	if err := store.Wake("hot", woken); err != nil {
		t.Fatalf("Wake after a prior panic = %v, want a clean restore (id must not be wedged)", err)
	}
	if store.Parked("hot") {
		t.Error("frozen file still present after the recovery Wake, want removed")
	}
}

// TestResidentCapWarmBand pins the #4035 two-watermark warm band: below the low-water
// mark WarmRefill fills up to the high-water cap (batch, not per-slot); above it
// WarmPark drains down to low-water (a warm reserve survives); a plain NewResidentCap
// has no band and is byte-identical (both folds return 0).
func TestResidentCapWarmBand(t *testing.T) {
	// No band: a plain cap reports no low-water and never asks to warm-refill/park.
	plain := microagent.NewResidentCap(4)
	if plain.LowWater() != 0 {
		t.Errorf("plain LowWater = %d, want 0 (no band)", plain.LowWater())
	}
	plain.Admit()
	if got := plain.WarmRefill(10); got != 0 {
		t.Errorf("plain WarmRefill = %d, want 0 (no band)", got)
	}
	if got := plain.WarmPark(10); got != 0 {
		t.Errorf("plain WarmPark = %d, want 0 (no band)", got)
	}

	// Warm band [low=2, high=6].
	c := microagent.NewResidentCapBand(2, 6)
	if c.Limit() != 6 || c.LowWater() != 2 {
		t.Fatalf("band Limit/LowWater = %d/%d, want 6/2", c.Limit(), c.LowWater())
	}

	// Empty and below low-water: refill up to the HIGH-water cap (to 6), not just to low.
	if got := c.WarmRefill(100); got != 6 {
		t.Errorf("WarmRefill at 0 resident = %d, want 6 (fill to high-water)", got)
	}
	if got := c.WarmRefill(3); got != 3 {
		t.Errorf("WarmRefill bounded by parked = %d, want 3", got)
	}

	// Fill to the low-water mark: at/above low, refill stops (hysteresis, not per-slot).
	c.Admit()
	c.Admit() // resident = 2 == low
	if got := c.WarmRefill(100); got != 0 {
		t.Errorf("WarmRefill at low-water = %d, want 0 (band satisfied)", got)
	}
	if got := c.WarmPark(100); got != 0 {
		t.Errorf("WarmPark at low-water = %d, want 0 (nothing above the reserve)", got)
	}

	// Climb above low-water, then drain: WarmPark sheds down to low (2), never below.
	c.Admit()
	c.Admit()
	c.Admit() // resident = 5
	if got := c.WarmPark(100); got != 3 {
		t.Errorf("WarmPark at resident 5 = %d, want 3 (drain to low-water 2)", got)
	}
	if got := c.WarmPark(1); got != 1 {
		t.Errorf("WarmPark bounded by idle = %d, want 1", got)
	}

	// Dropping below low-water re-arms refill up to the high-water cap.
	c.Release()
	c.Release()
	c.Release()
	c.Release() // resident = 1 < low = 2
	if got := c.WarmRefill(100); got != 5 {
		t.Errorf("WarmRefill below low-water = %d, want 5 (fill 1->6)", got)
	}

	// Clamps: low>high collapses to high; high<=0 selects DefaultResidentCap.
	if cc := microagent.NewResidentCapBand(9, 4); cc.LowWater() != 4 || cc.Limit() != 4 {
		t.Errorf("clamp low>high: LowWater/Limit = %d/%d, want 4/4", cc.LowWater(), cc.Limit())
	}
	if cc := microagent.NewResidentCapBand(3, 0); cc.Limit() != microagent.DefaultResidentCap {
		t.Errorf("high<=0 Limit = %d, want DefaultResidentCap %d", cc.Limit(), microagent.DefaultResidentCap)
	}
}

func driveDone(t *testing.T, m microagent.Microagent) {
	t.Helper()
	for i := 0; i < 1000; i++ {
		done, err := m.Step(context.Background(), nil)
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		if done {
			return
		}
	}
	t.Fatal("agent did not finish within 1000 steps")
}

// TestResidentCapBoundsConcurrency pins the ResidentCap counter semantics
// (#2012 scope 3): admits succeed up to the limit and then refuse, Release
// returns a slot, Peak records the high-water, and Release with no slot held is
// a programming error.
func TestResidentCapBoundsConcurrency(t *testing.T) {
	c := microagent.NewResidentCap(3)
	if c.Limit() != 3 {
		t.Fatalf("Limit = %d, want 3", c.Limit())
	}
	if !c.Admit() || !c.Admit() || !c.Admit() {
		t.Fatal("first 3 Admit calls should all succeed")
	}
	if c.Admit() {
		t.Fatal("4th Admit should refuse past the cap")
	}
	if c.Resident() != 3 || c.Peak() != 3 {
		t.Fatalf("resident=%d peak=%d, want 3/3", c.Resident(), c.Peak())
	}
	c.Release()
	if c.Resident() != 2 {
		t.Fatalf("resident=%d after Release, want 2", c.Resident())
	}
	if !c.Admit() {
		t.Fatal("Admit after Release should succeed")
	}
	if c.Peak() != 3 {
		t.Errorf("Peak = %d after churn, want the high-water 3", c.Peak())
	}
	if got := microagent.NewResidentCap(0).Limit(); got != microagent.DefaultResidentCap {
		t.Errorf("NewResidentCap(0).Limit() = %d, want DefaultResidentCap %d", got, microagent.DefaultResidentCap)
	}
	func() {
		defer func() {
			if recover() == nil {
				t.Error("Release with no slot held should panic")
			}
		}()
		microagent.NewResidentCap(1).Release()
	}()
}

// TestHibernationKeepsResidentSetBounded is the #2012 acceptance witness for
// "with R resident cap and N >> R enrolled, resident RSS stays ~O(R) and all
// agents still make progress." N enrolled step-agents are parked on disk and
// driven by more workers than resident slots; each cycle wakes an agent, takes
// one step, then re-parks it and frees the slot. The resident high-water never
// exceeds R, every agent completes, and every completed history is exactly the
// full linear run — proving no state was lost across dozens of park/wake cycles.
func TestHibernationKeepsResidentSetBounded(t *testing.T) {
	const N, R, workers, turns = 64, 8, 16, 6
	store, err := microagent.NewHibernationStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewHibernationStore: %v", err)
	}
	cap := microagent.NewResidentCap(R)

	ids := make([]string, N)
	for i := range ids {
		id := fmt.Sprintf("a-%02d", i)
		ids[i] = id
		// Enroll: freeze the fresh (turn 0) context to disk. No goroutine held.
		if _, err := store.Park(id, &histAgent{id: id, turns: turns}); err != nil {
			t.Fatalf("enroll Park(%s): %v", id, err)
		}
	}

	work := make(chan string, N)
	for _, id := range ids {
		work <- id
	}

	var remaining = int32(N)
	var failed int32
	var errMu sync.Mutex
	var firstErr error
	fail := func(format string, a ...any) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = fmt.Errorf(format, a...)
		}
		errMu.Unlock()
		atomic.StoreInt32(&failed, 1)
	}

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for atomic.LoadInt32(&remaining) > 0 && atomic.LoadInt32(&failed) == 0 {
				var id string
				select {
				case id = <-work:
				default:
					runtime.Gosched()
					continue
				}
				if !cap.Admit() { // the resident cap is the binding limit
					work <- id
					runtime.Gosched()
					continue
				}
				a := &histAgent{}
				if err := store.Wake(id, a); err != nil {
					fail("Wake(%s): %v", id, err)
					cap.Release()
					return
				}
				done, err := a.Step(context.Background(), nil)
				if err != nil {
					fail("Step(%s): %v", id, err)
					cap.Release()
					return
				}
				if done {
					if len(a.hist) != turns {
						fail("agent %s done with %d history entries, want %d", id, len(a.hist), turns)
					}
					for k := 0; k < len(a.hist) && k < turns; k++ {
						if want := fmt.Sprintf("%s:turn:%d", id, k+1); a.hist[k] != want {
							fail("agent %s hist[%d]=%q, want %q", id, k, a.hist[k], want)
						}
					}
					cap.Release()
					atomic.AddInt32(&remaining, -1)
					continue
				}
				if _, err := store.Park(id, a); err != nil {
					fail("re-Park(%s): %v", id, err)
					cap.Release()
					return
				}
				cap.Release()
				work <- id
			}
		}()
	}
	wg.Wait()

	if firstErr != nil {
		t.Fatalf("driver: %v", firstErr)
	}
	if r := atomic.LoadInt32(&remaining); r != 0 {
		t.Fatalf("%d/%d agents did not complete", r, N)
	}
	if cap.Peak() > R {
		t.Errorf("resident high-water %d exceeded cap R=%d — residency was NOT bounded", cap.Peak(), R)
	}
	if cap.Peak() < 2 {
		t.Errorf("resident high-water %d — expected real concurrency (N=%d, R=%d, workers=%d)", cap.Peak(), N, R, workers)
	}
	if cap.Resident() != 0 {
		t.Errorf("resident=%d after all agents done, want 0", cap.Resident())
	}
	for _, id := range ids {
		if store.Parked(id) {
			t.Errorf("agent %s still parked on disk after completion", id)
		}
	}
	t.Logf("N=%d enrolled, resident cap R=%d, peak resident=%d — all completed byte-identically across park/wake cycles", N, R, cap.Peak())
}
