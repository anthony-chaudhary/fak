package modelloadplan

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"
)

// fak-test:runtime fast est=1s
func TestExpertCacheLrfuClassLadderAndRetainedFrequency(t *testing.T) {
	c, err := NewExpertCacheLRFU(1, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range []struct {
		ids    []int
		tokens int64
	}{{[]int{0}, 128}, {[]int{0}, 0}, {[]int{0}, 0}, {[]int{1}, 128}} {
		if _, err := c.Plan(0, step.ids, step.tokens); err != nil {
			t.Fatal(err)
		}
	}
	zero, _ := c.Inspect(0, 0)
	if zero.CRF != 2.5 || zero.DecayedCRF != 1.25 || zero.Class != 1 {
		t.Fatalf("fixed half-life/class trace = %+v", zero)
	}
	got, err := c.Plan(0, []int{2}, 1)
	if err != nil || len(got.Fills) != 1 || got.Fills[0].Evicted == nil || *got.Fills[0].Evicted != (ExpertCacheKey{0, 1}) {
		t.Fatalf("age/2^class ladder chose wrong victim: %+v, %v", got, err)
	}
	// Both old experts have age128; class0's128 exceeds class1's64.
	if kept, _ := c.Inspect(0, 0); kept.Slot < 0 {
		t.Fatal("higher retained class evicted at equal age")
	}
	evicted, _ := c.Inspect(0, 1)
	if evicted.Slot != -1 || evicted.CRF != 1 {
		t.Fatalf("eviction erased historical frequency: %+v", evicted)
	}
	if _, err := c.Plan(0, []int{1}, 0); err != nil {
		t.Fatal(err)
	}
	reused, _ := c.Inspect(0, 1)
	want := float32(math.Exp2(-129.0/128.0)) + 1
	if reused.CRF != want || reused.CRF <= 1 {
		t.Fatalf("frequency did not survive eviction: %+v, want %v", reused, want)
	}
}

// fak-test:runtime fast est=1s
func TestExpertCacheLrfuProtectsSelectedAndSortsMisses(t *testing.T) {
	c, _ := NewExpertCacheLRFU(1, 4, 2)
	ids := []int{1, 0, 1}
	before := append([]int(nil), ids...)
	got, err := c.Plan(0, ids, 1)
	if err != nil || got.Hits != 0 || len(got.Fills) != 2 {
		t.Fatalf("initial plan: %+v, %v", got, err)
	}
	if !reflect.DeepEqual(ids, before) || got.Fills[0].Key.Expert != 0 || got.Fills[1].Key.Expert != 1 || got.Fills[0].Slot == got.Fills[1].Slot {
		t.Fatalf("miss order/protection/input mutation: %+v, %v", got, ids)
	}
	if one, _ := c.Inspect(0, 1); one.CRF != 1 {
		t.Fatalf("duplicate IDs counted repeatedly: %+v", one)
	}
	got, err = c.Plan(0, []int{2, 1}, 1)
	if err != nil || got.Hits != 1 || len(got.Fills) != 1 || got.Fills[0].Key.Expert != 2 || got.Fills[0].Evicted == nil || *got.Fills[0].Evicted != (ExpertCacheKey{0, 0}) {
		t.Fatalf("selected hit evicted: %+v, %v", got, err)
	}
	// Returned eviction identity is not internal storage.
	got.Fills[0].Evicted.Expert = 3
	got.Fills[0].Key.Expert = 3
	if two, _ := c.Inspect(0, 2); two.Slot < 0 {
		t.Fatal("returned plan aliases policy state")
	}
	if one, _ := c.Inspect(0, 1); one.Slot < 0 {
		t.Fatal("selected hit disappeared")
	}
}

// fak-test:runtime fast est=1s
func TestExpertCacheLrfuLayerIsolationAndLRUTies(t *testing.T) {
	c, _ := NewExpertCacheLRFU(2, 2, 1)
	if _, err := c.Plan(0, []int{0}, 64); err != nil {
		t.Fatal(err)
	}
	got, err := c.Plan(1, []int{0}, 32)
	if err != nil || len(got.Fills) != 1 || got.Fills[0].Evicted == nil || *got.Fills[0].Evicted != (ExpertCacheKey{0, 0}) {
		t.Fatalf("same expert index in another layer was protected: %+v, %v", got, err)
	}
	a, _ := c.Inspect(0, 0)
	b, _ := c.Inspect(1, 0)
	if a.LayerTime != 64 || b.LayerTime != 32 || a.LastUse != 0 || b.LastUse != 0 || a.CRF != 1 || b.CRF != 1 {
		t.Fatalf("layer clocks/history mixed: %+v %+v", a, b)
	}
	c, _ = NewExpertCacheLRFU(1, 3, 2)
	if _, err := c.Plan(0, []int{1, 0}, 0); err != nil {
		t.Fatal(err)
	}
	got, err = c.Plan(0, []int{2}, 0)
	if err != nil || len(got.Fills) != 1 || got.Fills[0].Evicted == nil || *got.Fills[0].Evicted != (ExpertCacheKey{0, 0}) {
		t.Fatalf("equal-score tie is not class LRU order: %+v, %v", got, err)
	}
}

// fak-test:runtime fast est=1s
func TestExpertCacheLrfuRefusalsAreAtomic(t *testing.T) {
	c, _ := NewExpertCacheLRFU(1, 3, 2)
	if _, err := c.Plan(0, []int{0, 1}, 1); err != nil {
		t.Fatal(err)
	}
	observe := func() []ExpertCacheLRFUEntry {
		out := make([]ExpertCacheLRFUEntry, 3)
		for i := range out {
			out[i], _ = c.Inspect(0, i)
		}
		return out
	}
	before := observe()
	for _, tc := range []struct {
		layer  int
		ids    []int
		tokens int64
		want   error
	}{{-1, nil, 0, ErrExpertCacheLRFUInput}, {1, nil, 0, ErrExpertCacheLRFUInput}, {0, []int{-1}, 0, ErrExpertCacheLRFUInput}, {0, []int{3}, 0, ErrExpertCacheLRFUInput}, {0, []int{0}, -1, ErrExpertCacheLRFUInput}, {0, []int{0, 1, 2}, 1, ErrExpertCacheLRFUCapacity}} {
		got, err := c.Plan(tc.layer, tc.ids, tc.tokens)
		if !errors.Is(err, tc.want) || !reflect.DeepEqual(got, ExpertCacheLRFUPlan{}) || !reflect.DeepEqual(before, observe()) {
			t.Fatalf("non-atomic refusal: %+v, %v", got, err)
		}
	}
	// Failed plans also leave the internal class order intact.
	got, err := c.Plan(0, []int{2}, 0)
	if err != nil || len(got.Fills) != 1 || got.Fills[0].Evicted == nil || *got.Fills[0].Evicted != (ExpertCacheKey{0, 0}) {
		t.Fatalf("refusal changed LRU order: %+v, %v", got, err)
	}
}

// fak-test:runtime fast est=1s
func TestExpertCacheLrfuGeometryAndClockBounds(t *testing.T) {
	for _, g := range [][3]int{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {-1, 1, 1}, {1, 1, 2}, {ExpertCacheLRFUMaxEntries, 2, 1}, {1, ExpertCacheLRFUMaxEntries + 1, 1}} {
		if c, err := NewExpertCacheLRFU(g[0], g[1], g[2]); c != nil || !errors.Is(err, ErrExpertCacheLRFUConfig) {
			t.Fatalf("geometry %v = %v, %v", g, c, err)
		}
	}
	c, _ := NewExpertCacheLRFU(1, 1, 1)
	if _, err := c.Plan(0, nil, math.MaxInt64); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Plan(0, nil, 1); !errors.Is(err, ErrExpertCacheLRFUClock) {
		t.Fatalf("clock overflow accepted: %v", err)
	}
	if _, err := c.Plan(0, []int{0}, 0); err != nil {
		t.Fatal(err)
	}
	e, _ := c.Inspect(0, 0)
	if e.CRF != 1 || e.LastUse != math.MaxInt64 || e.LayerTime != math.MaxInt64 {
		t.Fatalf("first use at maximum clock overflowed: %+v", e)
	}
	var nilPolicy *ExpertCacheLRFU
	if _, err := nilPolicy.Plan(0, nil, 0); !errors.Is(err, ErrExpertCacheLRFUInput) {
		t.Fatal(err)
	}
	if _, err := nilPolicy.Inspect(0, 0); !errors.Is(err, ErrExpertCacheLRFUInput) {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=1s
func TestExpertCacheLrfuSelectedClassHeadAndListIntegrity(t *testing.T) {
	c, _ := NewExpertCacheLRFU(2, 3, 2)
	for _, step := range []struct {
		layer  int
		ids    []int
		tokens int64
	}{
		{0, []int{0}, 0}, {0, []int{0}, 0}, // other layer's class1, age0
		{1, []int{0}, 128}, // this layer's class0, ages before its next hit
	} {
		if _, err := c.Plan(step.layer, step.ids, step.tokens); err != nil {
			t.Fatal(err)
		}
		assertExpertCacheLrfuLists(t, c)
	}
	got, err := c.Plan(1, []int{0, 1}, 0)
	if err != nil || got.Hits != 1 || len(got.Fills) != 1 || got.Fills[0].Evicted == nil || *got.Fills[0].Evicted != (ExpertCacheKey{0, 0}) {
		t.Fatalf("selected class0 head evicted on equal age instead of unselected class1: %+v, %v", got, err)
	}
	assertExpertCacheLrfuLists(t, c)
	// Exercise removal/movement across classes, with repeated
	// evictions and shared slot groups. The trace and token clocks are fixed.
	for i := 0; i < 1024; i++ {
		if _, err := c.Plan(i%2, []int{i % 3}, int64(i%5)); err != nil {
			t.Fatal(err)
		}
		assertExpertCacheLrfuLists(t, c)
	}
	c, _ = NewExpertCacheLRFU(1, 1, 1)
	classes := map[int]bool{}
	for uses := 1; uses <= 300; uses++ {
		if _, err := c.Plan(0, []int{0}, 0); err != nil {
			t.Fatal(err)
		}
		entry, _ := c.Inspect(0, 0)
		want := 0
		for threshold := 2; threshold <= uses && want < 7; threshold *= 2 {
			want++
		}
		if entry.Class != want || entry.CRF != float32(uses) {
			t.Fatalf("class boundary at %d: %+v, want class%d", uses, entry, want)
		}
		classes[entry.Class] = true
		assertExpertCacheLrfuLists(t, c)
	}
	if len(classes) != 8 {
		t.Fatalf("class coverage = %v", classes)
	}
}

// fak-test:runtime fast est=1s
func TestExpertCacheLrfuFloat32ClassBoundaries(t *testing.T) {
	// The expected classes include the donor's float-result rounding before
	// its integer cast. In particular, the value just below128 belongs to7,
	// not6: log2 is approximately6.999999914 and rounds to float32(7).
	for _, tc := range []struct {
		crf  float32
		want int
	}{
		{1, 0}, {math.Nextafter32(2, 0), 0}, {2, 1},
		{math.Nextafter32(4, 0), 1}, {4, 2},
		{math.Nextafter32(8, 0), 3}, {8, 3},
		{math.Nextafter32(64, 0), 6}, {64, 6},
		{math.Nextafter32(128, 0), 7}, {128, 7}, {256, 7},
	} {
		if got := expertCacheLRFUClass(tc.crf); got != tc.want {
			t.Errorf("CRF bits%08x: class%d, want%d", math.Float32bits(tc.crf), got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1s
func TestExpertCacheLrfuConcurrentPlanAndInspect(t *testing.T) {
	const workers, iterations, layers, experts = 8, 128, 2, 4
	c, err := NewExpertCacheLRFU(layers, experts, 3)
	if err != nil {
		t.Fatal(err)
	}
	start := make(chan struct{})
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			<-start
			layer := worker % layers
			for i := 0; i < iterations; i++ {
				expert := (worker + i) % experts
				if _, err := c.Plan(layer, []int{expert}, 1); err != nil {
					errs <- fmt.Errorf("worker%d plan: %w", worker, err)
					return
				}
				entry, err := c.Inspect(layer, expert)
				if err != nil || math.IsNaN(float64(entry.CRF)) || math.IsInf(float64(entry.CRF), 0) || entry.CRF < 1 || entry.LastUse < 0 || entry.LastUse > entry.LayerTime {
					errs <- fmt.Errorf("worker%d inspection: %+v, %v", worker, entry, err)
					return
				}
			}
		}(worker)
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	for layer := 0; layer < layers; layer++ {
		for expert := 0; expert < experts; expert++ {
			entry, err := c.Inspect(layer, expert)
			if err != nil || entry.LayerTime != workers/layers*iterations {
				t.Errorf("final layer%d expert%d clock: %+v, %v", layer, expert, entry, err)
			}
		}
	}
	// All workers have joined, so internal list inspection is uncontended.
	// Victim order is intentionally unspecified across goroutine schedules.
	assertExpertCacheLrfuLists(t, c)
}

func assertExpertCacheLrfuLists(t *testing.T, c *ExpertCacheLRFU) {
	t.Helper()
	seen := make([]bool, len(c.keys))
	for class := range c.head {
		previous := -1
		for slot := c.head[class]; slot >= 0; slot = c.next[slot] {
			if slot >= len(seen) || seen[slot] {
				t.Fatalf("duplicate/out-of-bounds class slot %d", slot)
			}
			seen[slot] = true
			if c.classes[slot] != class || c.prev[slot] != previous {
				t.Fatalf("class list cross-link at %d", slot)
			}
			if key := c.keys[slot]; key >= 0 && c.slots[key] != slot {
				t.Fatalf("resident reverse map disagrees at %d", slot)
			}
			previous = slot
		}
		if previous != c.tail[class] {
			t.Fatalf("class %d tail mismatch", class)
		}
	}
	for slot, present := range seen {
		if !present {
			t.Fatalf("slot %d missing from class lists", slot)
		}
	}
	for key, slot := range c.slots {
		if slot >= 0 && (slot >= len(c.keys) || c.keys[slot] != key) {
			t.Fatalf("expert %d reverse mapping invalid", key)
		}
	}
}
