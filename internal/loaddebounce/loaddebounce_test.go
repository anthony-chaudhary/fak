package loaddebounce

import (
	"testing"
	"time"
)

// fakeClock is a hand-cranked monotonic clock. The whole point of the primitive
// is that time is injected, so every test drives this clock explicitly and no
// test ever calls time.Sleep or reads the wall clock.
type fakeClock struct{ t time.Time }

func newFakeClock() *fakeClock { return &fakeClock{t: time.Unix(0, 0).UTC()} }

func (f *fakeClock) now() time.Time         { return f.t }
func (f *fakeClock) advance(d time.Duration) { f.t = f.t.Add(d) }

// TestDedupIdenticalSamplesEmitOnce is DoD witness #1: N identical consecutive
// samples emit exactly ONCE. After the first value settles and publishes, every
// later identical sample is a no-op.
func TestDedupIdenticalSamplesEmitOnce(t *testing.T) {
	clk := newFakeClock()
	var emitted []int
	p := NewPublisher[int](time.Millisecond, clk.now, func(v int) { emitted = append(emitted, v) })

	const n = 8
	for i := 0; i < n; i++ {
		p.Sample(7)                    // identical value every time
		clk.advance(2 * time.Millisecond) // let the debounce window elapse
		p.Flush()
	}

	if len(emitted) != 1 {
		t.Fatalf("dedup: %d identical samples emitted %d times %v, want exactly 1", n, len(emitted), emitted)
	}
	if emitted[0] != 7 {
		t.Fatalf("dedup: emitted %d, want 7", emitted[0])
	}
}

// TestDedupIdenticalBurstBeforeFlush is the same dedup guarantee when all the
// identical samples land inside one debounce window before any publish: it must
// still collapse to a single emission.
func TestDedupIdenticalBurstBeforeFlush(t *testing.T) {
	clk := newFakeClock()
	var emitted []int
	p := NewPublisher[int](time.Millisecond, clk.now, func(v int) { emitted = append(emitted, v) })

	for i := 0; i < 5; i++ {
		p.Sample(3)
		clk.advance(100 * time.Microsecond) // stays well inside the 1ms window
	}
	if len(emitted) != 0 {
		t.Fatalf("dedup burst: emitted before the window elapsed: %v", emitted)
	}
	clk.advance(2 * time.Millisecond)
	p.Flush()

	if len(emitted) != 1 || emitted[0] != 3 {
		t.Fatalf("dedup burst: emitted %v, want exactly [3]", emitted)
	}
}

// TestCoalesceBurstEmitsOnlyLatest is DoD witness #2: a burst of CHANGING values
// inside one reset-on-change window collapses to a single emission of the LATEST
// value. The intermediate values (1, 2) are never published.
func TestCoalesceBurstEmitsOnlyLatest(t *testing.T) {
	clk := newFakeClock()
	var emitted []int
	p := NewPublisher[int](time.Millisecond, clk.now, func(v int) { emitted = append(emitted, v) })

	// Three distinct values, each observed within the debounce window of the
	// previous one, so every Sample resets the deadline further out.
	p.Sample(1)
	clk.advance(200 * time.Microsecond)
	p.Sample(2)
	clk.advance(200 * time.Microsecond)
	p.Sample(3)

	if len(emitted) != 0 {
		t.Fatalf("coalesce: emitted mid-burst %v, want nothing until the window settles", emitted)
	}

	clk.advance(2 * time.Millisecond) // window finally elapses after the last change
	p.Flush()

	if len(emitted) != 1 {
		t.Fatalf("coalesce: burst of 3 changes emitted %d times %v, want exactly 1", len(emitted), emitted)
	}
	if emitted[0] != 3 {
		t.Fatalf("coalesce: emitted %d, want the LATEST value 3", emitted[0])
	}
}

// TestSettledChangesEachEmit proves the primitive is a debounce, not a drop: when
// distinct values are spaced FURTHER apart than the debounce window they each
// publish, in order.
func TestSettledChangesEachEmit(t *testing.T) {
	clk := newFakeClock()
	var emitted []int
	p := NewPublisher[int](time.Millisecond, clk.now, func(v int) { emitted = append(emitted, v) })

	for _, v := range []int{10, 11, 12} {
		p.Sample(v)
		clk.advance(2 * time.Millisecond) // each value settles before the next
		p.Flush()
	}

	if got := len(emitted); got != 3 {
		t.Fatalf("settled changes: emitted %d %v, want 3", got, emitted)
	}
	for i, want := range []int{10, 11, 12} {
		if emitted[i] != want {
			t.Fatalf("settled changes: emitted[%d]=%d, want %d (full %v)", i, emitted[i], want, emitted)
		}
	}
}

// TestNoNetChangeCancelsBurst covers the A->B->A collapse: a value that circles
// back to the last published value inside the window publishes nothing, matching
// the source watch channel coalescing to the latest before its dedup compare.
func TestNoNetChangeCancelsBurst(t *testing.T) {
	clk := newFakeClock()
	var emitted []int
	p := NewPublisher[int](time.Millisecond, clk.now, func(v int) { emitted = append(emitted, v) })

	// Publish A first so there is a published reference to circle back to.
	p.Sample(5)
	clk.advance(2 * time.Millisecond)
	p.Flush()
	if len(emitted) != 1 || emitted[0] != 5 {
		t.Fatalf("setup: want [5], got %v", emitted)
	}

	// A -> B -> A all inside one window: net change is zero, so nothing new.
	p.Sample(9) // B
	clk.advance(200 * time.Microsecond)
	p.Sample(5) // back to A (== last published)
	clk.advance(2 * time.Millisecond)
	p.Flush()

	if len(emitted) != 1 {
		t.Fatalf("no-net-change: emitted %v, want it to stay [5]", emitted)
	}
}

// TestDeadlineResetsOnEveryChange asserts the reset-on-every-change contract
// directly on the pure core: each distinct observation moves the deadline to
// now+debounce, and the value is due only after the FINAL change's window.
func TestDeadlineResetsOnEveryChange(t *testing.T) {
	base := time.Unix(0, 0).UTC()
	c := New[int](time.Millisecond)

	c.Observe(1, base)
	d1, armed := c.Deadline()
	if !armed || !d1.Equal(base.Add(time.Millisecond)) {
		t.Fatalf("after first observe: armed=%v deadline=%v, want %v", armed, d1, base.Add(time.Millisecond))
	}

	// A later distinct observation must push the deadline out (reset-on-change).
	t2 := base.Add(500 * time.Microsecond)
	c.Observe(2, t2)
	d2, _ := c.Deadline()
	if !d2.Equal(t2.Add(time.Millisecond)) {
		t.Fatalf("after reset: deadline=%v, want %v", d2, t2.Add(time.Millisecond))
	}

	// Not due at the ORIGINAL deadline because the reset pushed it out.
	if _, ok := c.Emit(d1); ok {
		t.Fatalf("emitted at the stale deadline %v; reset-on-change did not take", d1)
	}
	// Due at the new deadline, and it yields the latest value.
	v, ok := c.Emit(d2)
	if !ok || v != 2 {
		t.Fatalf("emit at %v = (%d,%v), want (2,true)", d2, v, ok)
	}
}
