package systembaseline

import (
	"testing"
	"time"
)

func TestCadenceControllerBoundsAndOverload(t *testing.T) {
	c := NewCadenceController(CadencePolicy{Minimum: 10 * time.Millisecond, Maximum: 100 * time.Millisecond, MaximumDutyPercent: 10})
	if got := c.Observe(time.Millisecond); got != 12500*time.Microsecond || c.Overloaded() {
		t.Fatalf("cheap got %v overload=%v", got, c.Overloaded())
	}
	if got := c.Observe(5 * time.Millisecond); got != 62500*time.Microsecond || c.Overloaded() {
		t.Fatalf("adapt got %v overload=%v", got, c.Overloaded())
	}
	if got := c.Observe(20 * time.Millisecond); got != 100*time.Millisecond || !c.Overloaded() {
		t.Fatalf("overload got %v overload=%v", got, c.Overloaded())
	}
}
func TestStableProcessCacheRejectsPIDReuse(t *testing.T) {
	var c StableProcessCache
	a := []ProcessSample{{PID: 7, StartID: 10, Image: "old"}}
	_, m := c.Apply(a)
	if m != 1 {
		t.Fatal(m)
	}
	b := []ProcessSample{{PID: 7, StartID: 10}}
	h, m := c.Apply(b)
	if h != 1 || m != 0 || b[0].Image != "old" {
		t.Fatalf("stable h=%d m=%d %#v", h, m, b)
	}
	reuse := []ProcessSample{{PID: 7, StartID: 11, Image: "new"}}
	h, m = c.Apply(reuse)
	if h != 0 || m != 1 || reuse[0].Image != "new" {
		t.Fatalf("reuse h=%d m=%d %#v", h, m, reuse)
	}
}
