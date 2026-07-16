package microagent_test

import (
	"bytes"
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

// thawCountAgent counts how many times Thaw is invoked on it. Freeze returns its state
// verbatim and Thaw stores the bytes back, so a Freeze after a Thaw is byte-identical (it
// passes HibernationStore.Wake's no-state-loss check). The counter is the #4035 witness: the
// warm re-admit path must run ZERO Thaws, the cold store path exactly one.
type thawCountAgent struct {
	thaws *int32
	state []byte
}

func (a *thawCountAgent) Step(context.Context, microagent.Gateway) (bool, error) { return true, nil }
func (a *thawCountAgent) Freeze() ([]byte, error) {
	return append([]byte(nil), a.state...), nil
}
func (a *thawCountAgent) Thaw(b []byte) error {
	atomic.AddInt32(a.thaws, 1)
	a.state = append([]byte(nil), b...)
	return nil
}

// TestWarmReserveZeroThawReAdmit is the #4035 Release-side acceptance witness: an agent
// Released WARM into the reserve is re-Admitted under the same id at ZERO Thaw — the reserve
// hands back the live agent, so the cold Freeze -> disk -> Thaw round-trip never runs. The
// disabled (cap 0) reserve refuses every Reserve, so the same id can only come back through the
// cold HibernationStore.Wake, which pays exactly one Thaw — the byte-identical "band off"
// baseline the warm hit removes.
func TestWarmReserveZeroThawReAdmit(t *testing.T) {
	var thaws int32
	live := &thawCountAgent{thaws: &thaws, state: []byte("live-state")}

	// Band ON: Reserve the live agent warm, then Take it back — zero Thaw, same pointer.
	r := microagent.NewWarmReserve(2)
	if r.Cap() != 2 {
		t.Fatalf("Cap = %d, want 2", r.Cap())
	}
	if !r.Reserve("w", live) {
		t.Fatal("Reserve into an empty cap-2 reserve should succeed")
	}
	if !r.Warm("w") || r.Len() != 1 {
		t.Fatalf("after Reserve: Warm=%v Len=%d, want true/1", r.Warm("w"), r.Len())
	}
	got, ok := r.Take("w")
	if !ok {
		t.Fatal("Take(w) missed, want a warm hit")
	}
	if got != microagent.Hibernable(live) {
		t.Error("Take(w) returned a different agent, want the same live pointer (no restore)")
	}
	if n := atomic.LoadInt32(&thaws); n != 0 {
		t.Errorf("warm re-admit ran %d Thaws, want 0 (the whole point of the warm band)", n)
	}
	if r.Warm("w") || r.Len() != 0 {
		t.Errorf("after Take: Warm=%v Len=%d, want false/0 (slot freed)", r.Warm("w"), r.Len())
	}
	// A second Take is a miss — nothing left to reuse, the caller falls through to cold.
	if _, ok := r.Take("w"); ok {
		t.Error("second Take(w) hit, want a miss after the agent was taken")
	}

	// Band OFF (cap 0): Reserve refuses and Take always misses (byte-identical "band off").
	off := microagent.NewWarmReserve(0)
	if off.Cap() != 0 {
		t.Errorf("NewWarmReserve(0).Cap() = %d, want 0 (disabled)", off.Cap())
	}
	if off.Reserve("w", &thawCountAgent{thaws: new(int32)}) {
		t.Error("Reserve into a cap-0 (disabled) reserve should refuse")
	}
	if _, ok := off.Take("w"); ok {
		t.Error("Take from a disabled reserve should always miss")
	}
	// The cold fallback the warm miss preserves pays exactly one Thaw — the tax the band removes.
	store, err := microagent.NewHibernationStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewHibernationStore: %v", err)
	}
	var coldThaws int32
	if _, err := store.Park("w", &thawCountAgent{thaws: &coldThaws, state: []byte("cold-state")}); err != nil {
		t.Fatalf("Park: %v", err)
	}
	if err := store.Wake("w", &thawCountAgent{thaws: &coldThaws}); err != nil {
		t.Fatalf("Wake: %v", err)
	}
	if n := atomic.LoadInt32(&coldThaws); n != 1 {
		t.Errorf("cold re-admit ran %d Thaws, want exactly 1 (the cold-start tax the warm band removes)", n)
	}
}

// TestWarmReserveCapBounded pins that the reserve never holds more than its cap (#4035): a
// Reserve past cap refuses (the caller cold-parks that agent instead), a duplicate id refuses,
// a nil agent refuses, and Len never exceeds Cap. Taking an id frees its slot for the next
// Reserve — this is the "never exceeding the preflight effective cap" half of the acceptance.
func TestWarmReserveCapBounded(t *testing.T) {
	r := microagent.NewWarmReserve(2)
	a1 := &thawCountAgent{thaws: new(int32), state: []byte("1")}
	a2 := &thawCountAgent{thaws: new(int32), state: []byte("2")}
	a3 := &thawCountAgent{thaws: new(int32), state: []byte("3")}

	if !r.Reserve("a", a1) || !r.Reserve("b", a2) {
		t.Fatal("first two Reserves under cap 2 should both succeed")
	}
	if r.Reserve("c", a3) {
		t.Error("third Reserve past cap 2 should refuse (caller cold-parks instead)")
	}
	if r.Len() != 2 || r.Len() > r.Cap() {
		t.Errorf("Len=%d Cap=%d, want Len 2 never exceeding cap", r.Len(), r.Cap())
	}
	// An id maps to one warm agent: a duplicate-id Reserve refuses.
	if r.Reserve("a", a3) {
		t.Error("duplicate-id Reserve should refuse (one warm agent per id)")
	}
	// A nil agent is refused (a nil warm hit would masquerade as a real one).
	if r.Reserve("d", nil) {
		t.Error("nil-agent Reserve should refuse")
	}
	// Taking one frees a slot so the next Reserve fits — still bounded by the cap.
	if _, ok := r.Take("a"); !ok {
		t.Fatal("Take(a) should hit")
	}
	if !r.Reserve("c", a3) {
		t.Error("Reserve after a Take freed a slot should now succeed")
	}
	if r.Len() != 2 || r.Len() > r.Cap() {
		t.Errorf("Len=%d Cap=%d after take+reserve, want 2 never exceeding cap", r.Len(), r.Cap())
	}
}

// TestWarmReserveDrain proves Drain sheds every held agent (the staleness/shutdown path) and
// leaves the reserve empty; draining an empty reserve returns nil.
func TestWarmReserveDrain(t *testing.T) {
	r := microagent.NewWarmReserve(4)
	if got := r.Drain(); got != nil {
		t.Errorf("Drain of an empty reserve = %v, want nil", got)
	}
	for _, id := range []string{"a", "b", "c"} {
		if !r.Reserve(id, &thawCountAgent{thaws: new(int32), state: []byte(id)}) {
			t.Fatalf("Reserve(%s) should succeed", id)
		}
	}
	shed := r.Drain()
	if len(shed) != 3 {
		t.Errorf("Drain returned %d agents, want 3", len(shed))
	}
	if r.Len() != 0 {
		t.Errorf("Len=%d after Drain, want 0", r.Len())
	}
	for _, id := range []string{"a", "b", "c"} {
		if r.Warm(id) {
			t.Errorf("%s still warm after Drain, want shed", id)
		}
	}
}

// TestWarmBandRoundTripPreservesState is the #4035 end-to-end witness that the warm band is a
// PURE FAST PATH over the cold hibernation round-trip: the same partially-run agent, re-admitted
// once through the warm reserve (0 Thaw) and once through the cold store (1 Thaw), resumes to the
// exact same completion a never-hibernated run reaches — so a warm hit is byte-identical to a
// cold miss, never a correctness dependency.
func TestWarmBandRoundTripPreservesState(t *testing.T) {
	store, err := microagent.NewHibernationStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewHibernationStore: %v", err)
	}
	reserve := microagent.NewWarmReserve(4)

	// Reference: a never-hibernated run of 5 turns.
	ref := &histAgent{id: "x", turns: 5}
	driveDone(t, ref)
	refBytes, _ := ref.Freeze()

	// Warm path: run 2 of 5 turns, Release WARM into the reserve, re-Admit via Take (0 Thaw),
	// then finish. Take hands back the live agent, so no Freeze/Thaw ran on this path at all.
	warm := &histAgent{id: "x", turns: 5}
	warm.Step(context.Background(), nil)
	warm.Step(context.Background(), nil)
	if !reserve.Reserve("x", warm) {
		t.Fatal("Reserve should hold the live agent under cap")
	}
	if reserve.Len() != 1 {
		t.Fatalf("reserve Len=%d after Reserve, want 1", reserve.Len())
	}
	took, ok := reserve.Take("x")
	if !ok {
		t.Fatal("Take should hit the warm agent (the 0-Thaw re-admit)")
	}
	driveDone(t, took)
	warmBytes, _ := took.Freeze()
	if !bytes.Equal(warmBytes, refBytes) {
		t.Errorf("warm-reuse resume diverged from reference:\n warm %s\n ref  %s", warmBytes, refBytes)
	}

	// Cold path: same 2-turn start, but Park + Wake (a real Thaw) before finishing.
	cold := &histAgent{id: "x", turns: 5}
	cold.Step(context.Background(), nil)
	cold.Step(context.Background(), nil)
	if _, err := store.Park("x", cold); err != nil {
		t.Fatalf("Park: %v", err)
	}
	woken := &histAgent{}
	if err := store.Wake("x", woken); err != nil {
		t.Fatalf("Wake: %v", err)
	}
	driveDone(t, woken)
	coldBytes, _ := woken.Freeze()
	if !bytes.Equal(coldBytes, refBytes) {
		t.Errorf("cold-wake resume diverged from reference:\n cold %s\n ref  %s", coldBytes, refBytes)
	}
	// The band is a pure fast path: warm and cold re-admit reach the identical completion.
	if !bytes.Equal(warmBytes, coldBytes) {
		t.Errorf("warm and cold re-admit diverged:\n warm %s\n cold %s", warmBytes, coldBytes)
	}
}

// TestWarmReserveConcurrent smoke-checks the mutex: many goroutines Reserve/Take/Warm/Len/Drain
// concurrently without racing, and Len never exceeds Cap. Run with -race for the real signal.
func TestWarmReserveConcurrent(t *testing.T) {
	r := microagent.NewWarmReserve(8)
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := string(rune('a' + i%8))
			r.Reserve(id, &thawCountAgent{thaws: new(int32), state: []byte(id)})
			_ = r.Warm(id)
			if n := r.Len(); n > r.Cap() {
				t.Errorf("Len=%d exceeded Cap=%d under concurrency", n, r.Cap())
			}
			r.Take(id)
			if i%16 == 0 {
				r.Drain()
			}
		}(i)
	}
	wg.Wait()
}
