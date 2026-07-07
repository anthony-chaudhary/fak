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
