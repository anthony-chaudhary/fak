package polymodel

import (
	"errors"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Residency pool.
// ---------------------------------------------------------------------------

func TestPoolAdmitWithinBudget(t *testing.T) {
	p := NewPool(100)
	for _, m := range []Model{
		{ID: "a", WeightBytes: 30},
		{ID: "b", WeightBytes: 30},
		{ID: "c", WeightBytes: 30},
	} {
		evicted, err := p.Admit(m)
		if err != nil {
			t.Fatalf("admit %s: %v", m.ID, err)
		}
		if len(evicted) != 0 {
			t.Fatalf("admit %s evicted %v under budget", m.ID, evicted)
		}
	}
	if p.Used() != 90 {
		t.Fatalf("used = %d, want 90", p.Used())
	}
	if p.Len() != 3 {
		t.Fatalf("len = %d, want 3", p.Len())
	}
}

func TestPoolLRUEvictsColdest(t *testing.T) {
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "a", WeightBytes: 40})
	mustAdmit(t, p, Model{ID: "b", WeightBytes: 40})
	// Touch a so b becomes the coldest; admitting c (40) must evict b, not a.
	p.Touch("a")
	evicted, err := p.Admit(Model{ID: "c", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit c: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "b" {
		t.Fatalf("evicted = %v, want [b] (the coldest after touching a)", evicted)
	}
	if !p.Has("a") || !p.Has("c") || p.Has("b") {
		t.Fatalf("residency = %v, want a,c not b", p.Resident())
	}
	if p.Used() > p.Budget() {
		t.Fatalf("used %d exceeds budget %d", p.Used(), p.Budget())
	}
}

// TestPoolLemonadeEvictionProtectsHeavyModel verifies that Lemonade reload-cost-weighted
// eviction scoring (idle_duration / (load_duration * weight_factor)) evicts a fast-loading
// model before an expensive-to-reload heavy model, even when the heavy model became idle
// earlier (longer idle duration).
func TestPoolLemonadeEvictionProtectsHeavyModel(t *testing.T) {
	p := NewPool(100)
	simTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	p.SetNow(func() time.Time { return simTime })

	// heavy: 10s reload duration, weight factor 1.0.
	heavy := Model{
		ID:           "heavy-14b",
		WeightBytes:  40,
		LoadDuration: 10 * time.Second,
		WeightFactor: 1.0,
	}
	// light: 500ms reload duration, weight factor 1.0.
	light := Model{
		ID:           "light-0.5b",
		WeightBytes:  40,
		LoadDuration: 500 * time.Millisecond,
		WeightFactor: 1.0,
	}

	// Admit heavy at t=0.
	mustAdmit(t, p, heavy)

	// Advance time by 100ms and admit light at t=100ms.
	simTime = simTime.Add(100 * time.Millisecond)
	mustAdmit(t, p, light)

	// Advance time by 100ms to t=200ms before triggering eviction.
	// At this point:
	// - heavy has been idle for 200ms.
	// - light has been idle for 100ms.
	// Under pure LRU, heavy would be evicted (longest idle: 200ms > 100ms).
	// Under Lemonade eviction scoring:
	// score(heavy) = 200ms / (10,000ms * 1.0) = 0.02
	// score(light) = 100ms / (500ms * 1.0)    = 0.20
	// score(light) is 10x higher -> light is the eviction victim.
	simTime = simTime.Add(100 * time.Millisecond)

	evicted, err := p.Admit(Model{ID: "incoming", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit incoming: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "light-0.5b" {
		t.Fatalf("evicted = %v, want [light-0.5b] (cheap to reload, protected heavy-14b)", evicted)
	}
	if !p.Has("heavy-14b") {
		t.Fatal("heavy-14b should remain resident in pool")
	}
	if p.Has("light-0.5b") {
		t.Fatal("light-0.5b should have been evicted")
	}
	if !p.Has("incoming") {
		t.Fatal("incoming should be resident in pool")
	}

	// Conversely, if heavy becomes genuinely cold (idle for a very long time),
	// its eviction score rises and it should eventually be evicted.
	// Advance by 1 hour (3600s). Touch incoming so incoming is fresh.
	simTime = simTime.Add(3600 * time.Second)
	p.Touch("incoming")
	// At t=3600.2s:
	// score(heavy)    = 3600.2s / 10s = 360.02
	// score(incoming) = 0s / 1s       = 0
	// Admitting a 40-byte model must now evict heavy-14b.
	evicted2, err := p.Admit(Model{ID: "newer", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit newer: %v", err)
	}
	if len(evicted2) != 1 || evicted2[0] != "heavy-14b" {
		t.Fatalf("evicted2 = %v, want [heavy-14b] after heavy became genuinely stale", evicted2)
	}
	if p.Has("heavy-14b") {
		t.Fatal("heavy-14b should have been evicted after 1h idle")
	}
}

// TestPoolLemonadeEvictionDefaultPreservesLRU confirms that when LoadDuration and
// WeightFactor are unset (zero values), default values (1s, 1.0) are applied and
// standard LRU recency ordering is strictly preserved.
func TestPoolLemonadeEvictionDefaultPreservesLRU(t *testing.T) {
	p := NewPool(100)
	// Unset LoadDuration and WeightFactor
	mustAdmit(t, p, Model{ID: "m1", WeightBytes: 40})
	mustAdmit(t, p, Model{ID: "m2", WeightBytes: 40})

	// Without touching, m1 was admitted first and is colder than m2.
	// Admitting m3 (40) must evict m1.
	evicted, err := p.Admit(Model{ID: "m3", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit m3: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "m1" {
		t.Fatalf("evicted = %v, want [m1] (oldest admitted with default scores)", evicted)
	}
	if p.Has("m1") || !p.Has("m2") || !p.Has("m3") {
		t.Fatalf("resident = %v, want m2, m3", p.Resident())
	}

	// Now touch m2, making m3 the colder one.
	p.Touch("m2")
	evicted2, err := p.Admit(Model{ID: "m4", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit m4: %v", err)
	}
	if len(evicted2) != 1 || evicted2[0] != "m3" {
		t.Fatalf("evicted2 = %v, want [m3] (colder after m2 touched)", evicted2)
	}
	if p.Has("m3") || !p.Has("m2") || !p.Has("m4") {
		t.Fatalf("resident = %v, want m2, m4", p.Resident())
	}
}

// TestPoolLemonadeWeightFactorProtection asserts that between two models with equal
// load durations and equal idle times, the one with higher WeightFactor receives a
// lower score and is protected from eviction.
func TestPoolLemonadeWeightFactorProtection(t *testing.T) {
	p := NewPool(100)
	simTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	p.SetNow(func() time.Time { return simTime })

	// Both models take 1s to load, but vip has WeightFactor=5.0 and standard has WeightFactor=1.0.
	vip := Model{ID: "vip", WeightBytes: 40, LoadDuration: 1 * time.Second, WeightFactor: 5.0}
	std := Model{ID: "std", WeightBytes: 40, LoadDuration: 1 * time.Second, WeightFactor: 1.0}

	mustAdmit(t, p, vip)
	mustAdmit(t, p, std)

	simTime = simTime.Add(1 * time.Second)
	// At t=1s:
	// score(vip) = 1s / (1s * 5.0) = 0.20
	// score(std) = 1s / (1s * 1.0) = 1.00
	// std has higher score -> std evicted, vip protected.
	evicted, err := p.Admit(Model{ID: "c", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit c: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "std" {
		t.Fatalf("evicted = %v, want [std] (vip protected by WeightFactor)", evicted)
	}
	if !p.Has("vip") || p.Has("std") {
		t.Fatalf("resident = %v, want vip surviving", p.Resident())
	}
}

// TestEvictionScoreHelper tests the pure EvictionScore function edge cases and formula.
func TestEvictionScoreHelper(t *testing.T) {
	const eps = 1e-9
	check := func(name string, idle, load time.Duration, weight, want float64) {
		got := EvictionScore(idle, load, weight)
		if got < want-eps || got > want+eps {
			t.Fatalf("%s: EvictionScore(%v, %v, %g) = %g, want %g", name, idle, load, weight, got, want)
		}
	}

	// Basic calculation
	check("1s idle / (1s load * 1.0 weight)", 1*time.Second, 1*time.Second, 1.0, 1.0)
	check("500ms idle / (2s load * 2.0 weight)", 500*time.Millisecond, 2*time.Second, 2.0, 0.125)
	check("2s idle / (500ms load * 1.0 weight)", 2*time.Second, 500*time.Millisecond, 1.0, 4.0)

	// Defaults: load <= 0 defaults to 1s, weight <= 0 defaults to 1.0
	check("zero load duration defaults to 1s", 500*time.Millisecond, 0, 1.0, 0.5)
	check("negative load duration defaults to 1s", 500*time.Millisecond, -1*time.Second, 1.0, 0.5)
	check("zero weight factor defaults to 1.0", 1*time.Second, 1*time.Second, 0, 1.0)
	check("negative weight factor defaults to 1.0", 1*time.Second, 1*time.Second, -2.5, 1.0)

	// Idle <= 0 clamps to 0
	check("zero idle gives 0 score", 0, 1*time.Second, 1.0, 0.0)
	check("negative idle clamps to 0", -500*time.Millisecond, 1*time.Second, 1.0, 0.0)
}

func TestPoolPinnedNeverEvicted(t *testing.T) {
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "draft", WeightBytes: 60, Pinned: true})
	mustAdmit(t, p, Model{ID: "x", WeightBytes: 30})
	// A 60-byte model fits only by dropping the pinned draft → refused, unchanged.
	used, ln := p.Used(), p.Len()
	_, err := p.Admit(Model{ID: "big", WeightBytes: 60})
	if !errors.Is(err, ErrPinnedNoRoom) {
		t.Fatalf("admit big: err = %v, want ErrPinnedNoRoom", err)
	}
	if p.Used() != used || p.Len() != ln {
		t.Fatalf("pool mutated on refused admit: used %d->%d len %d->%d", used, p.Used(), ln, p.Len())
	}
	// A 40-byte model fits by evicting unpinned x only; the pinned draft survives.
	evicted, err := p.Admit(Model{ID: "ok", WeightBytes: 40})
	if err != nil {
		t.Fatalf("admit ok: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "x" {
		t.Fatalf("evicted = %v, want [x]", evicted)
	}
	if !p.Has("draft") {
		t.Fatal("pinned draft was evicted")
	}
}

func TestPoolTooLargeRefusedUnchanged(t *testing.T) {
	p := NewPool(50)
	mustAdmit(t, p, Model{ID: "a", WeightBytes: 20})
	used := p.Used()
	_, err := p.Admit(Model{ID: "huge", WeightBytes: 60})
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("err = %v, want ErrTooLarge", err)
	}
	if p.Used() != used || p.Has("huge") {
		t.Fatal("pool mutated on too-large admit")
	}
}

func TestPoolReadmitIsTouch(t *testing.T) {
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "a", WeightBytes: 40})
	mustAdmit(t, p, Model{ID: "b", WeightBytes: 40})
	// Re-admit a (now the older of the two) → it becomes most-recent, so the next
	// eviction targets b. Used must not double-count a's bytes.
	if _, err := p.Admit(Model{ID: "a", WeightBytes: 40}); err != nil {
		t.Fatalf("re-admit a: %v", err)
	}
	if p.Used() != 80 {
		t.Fatalf("used = %d after re-admit, want 80 (no double count)", p.Used())
	}
	evicted, _ := p.Admit(Model{ID: "c", WeightBytes: 40})
	if len(evicted) != 1 || evicted[0] != "b" {
		t.Fatalf("evicted = %v, want [b] (re-admit refreshed a's recency)", evicted)
	}
}

// TestPoolBudgetNeverExceeded drives an arbitrary deterministic admit/touch
// sequence and asserts the core invariant after every step: Used() <= Budget().
func TestPoolBudgetNeverExceeded(t *testing.T) {
	p := NewPool(256)
	// A fixed pseudo-sequence (no rand import: reproducible by construction).
	sizes := []int64{50, 70, 90, 30, 110, 40, 200, 60, 20, 80, 130, 10, 170, 25, 95}
	for i, sz := range sizes {
		id := ModelID(string(rune('a' + i)))
		_, err := p.Admit(Model{ID: id, WeightBytes: sz})
		if err != nil && !errors.Is(err, ErrTooLarge) && !errors.Is(err, ErrPinnedNoRoom) {
			t.Fatalf("admit %s(%d): unexpected err %v", id, sz, err)
		}
		if p.Used() > p.Budget() {
			t.Fatalf("after admit %s(%d): used %d exceeds budget %d", id, sz, p.Used(), p.Budget())
		}
		if i%3 == 0 && p.Len() > 0 {
			p.Touch(p.Resident()[0])
		}
	}
}

// TestPoolResizeShrinkEvictsLRU shrinks the budget at runtime and asserts the coldest
// UNPINNED residents are paged out in LRU order until the resident set fits, returned in
// eviction order — the re-budget-under-pressure direction of the knob.
func TestPoolResizeShrinkEvictsLRU(t *testing.T) {
	p := NewPool(120)
	mustAdmit(t, p, Model{ID: "a", WeightBytes: 40})
	mustAdmit(t, p, Model{ID: "b", WeightBytes: 40})
	mustAdmit(t, p, Model{ID: "c", WeightBytes: 40})
	// Make a the hottest, then b; c stays coldest. Shrinking 120→50 must free 70 bytes:
	// evict c then b (coldest-first), leaving a (40 <= 50).
	p.Touch("b")
	p.Touch("a")
	evicted, err := p.Resize(50)
	if err != nil {
		t.Fatalf("resize: %v", err)
	}
	if len(evicted) != 2 || evicted[0] != "c" || evicted[1] != "b" {
		t.Fatalf("evicted = %v, want [c b] (coldest-first)", evicted)
	}
	if p.Budget() != 50 || p.Used() != 40 || !p.Has("a") || p.Has("b") || p.Has("c") {
		t.Fatalf("after shrink: budget=%d used=%d resident=%v, want 50/40/[a]", p.Budget(), p.Used(), p.Resident())
	}
}

// TestPoolResizeGrowEvictsNothing proves growing the budget — and any shrink that still
// fits the residents — evicts nothing and only re-sets the budget.
func TestPoolResizeGrowEvictsNothing(t *testing.T) {
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "a", WeightBytes: 30})
	mustAdmit(t, p, Model{ID: "b", WeightBytes: 30})
	if evicted, err := p.Resize(500); err != nil || len(evicted) != 0 { // grow
		t.Fatalf("grow: err=%v evicted=%v, want no eviction", err, evicted)
	}
	if p.Budget() != 500 || p.Len() != 2 {
		t.Fatalf("after grow: budget=%d len=%d, want 500/2", p.Budget(), p.Len())
	}
	if evicted, err := p.Resize(60); err != nil || len(evicted) != 0 { // shrink but >= used(60)
		t.Fatalf("shrink-to-fit: err=%v evicted=%v, want no eviction", err, evicted)
	}
	if p.Used() != 60 || p.Len() != 2 {
		t.Fatalf("shrink-to-fit dropped a resident: used=%d len=%d", p.Used(), p.Len())
	}
}

// TestPoolResizePinnedOverflowRefused shrinks below the pinned footprint: no eviction can
// make room (pinned are exempt), so Resize refuses with ErrPinnedNoRoom and leaves the pool
// byte-for-byte unchanged — the same all-or-nothing discipline as Admit.
func TestPoolResizePinnedOverflowRefused(t *testing.T) {
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "pin", WeightBytes: 60, Pinned: true})
	mustAdmit(t, p, Model{ID: "x", WeightBytes: 30})
	budget, used, ln := p.Budget(), p.Used(), p.Len()
	// Shrink to 50 < pinned 60: even evicting all unpinned (x=30) leaves 60 > 50 → refuse.
	if _, err := p.Resize(50); !errors.Is(err, ErrPinnedNoRoom) {
		t.Fatalf("resize below pinned footprint: err=%v, want ErrPinnedNoRoom", err)
	}
	if p.Budget() != budget || p.Used() != used || p.Len() != ln {
		t.Fatalf("refused resize mutated pool: budget %d->%d used %d->%d len %d->%d",
			budget, p.Budget(), used, p.Used(), ln, p.Len())
	}
	// Shrink to exactly the pinned footprint (60): evicts unpinned x, the pinned model stays.
	evicted, err := p.Resize(60)
	if err != nil {
		t.Fatalf("resize to pinned footprint: %v", err)
	}
	if len(evicted) != 1 || evicted[0] != "x" || !p.Has("pin") {
		t.Fatalf("evicted=%v resident=%v, want [x] with pin surviving", evicted, p.Resident())
	}
}

// TestPoolResizeInvariantHolds drives a grow/shrink/negative-clamp resize sequence and
// asserts used <= budget after every step, that a negative budget clamps to 0 (matching
// NewPool), and that the clamp evicts every unpinned resident.
func TestPoolResizeInvariantHolds(t *testing.T) {
	p := NewPool(300)
	for _, m := range []Model{
		{ID: "a", WeightBytes: 80},
		{ID: "b", WeightBytes: 80},
		{ID: "c", WeightBytes: 80},
	} {
		mustAdmit(t, p, m)
	}
	for _, nb := range []int64{300, 500, 160, 90, 400, -50} {
		if _, err := p.Resize(nb); err != nil {
			t.Fatalf("resize(%d): %v", nb, err)
		}
		want := nb
		if want < 0 {
			want = 0
		}
		if p.Budget() != want {
			t.Fatalf("budget = %d after resize(%d), want %d", p.Budget(), nb, want)
		}
		if p.Used() > p.Budget() {
			t.Fatalf("resize(%d): used %d exceeds budget %d", nb, p.Used(), p.Budget())
		}
	}
	if p.Len() != 0 || p.Used() != 0 {
		t.Fatalf("after clamp-to-0 resize: len=%d used=%d, want empty", p.Len(), p.Used())
	}
}

// ---------------------------------------------------------------------------
// Decode lane.
// ---------------------------------------------------------------------------

func TestNextDecoderPolicy(t *testing.T) {
	reqs := []Request{
		{Model: "low", Decode: 5, Priority: 1, Seq: 1},
		{Model: "hi-late", Decode: 5, Priority: 9, Seq: 7},
		{Model: "hi-early", Decode: 5, Priority: 9, Seq: 2},
		{Model: "done", Decode: 0, Priority: 99, Seq: 0},
	}
	// Highest priority, then lowest Seq → "hi-early". "done" has no decode work.
	if got := NextDecoder(reqs, nil); reqs[got].Model != "hi-early" {
		t.Fatalf("NextDecoder = %s, want hi-early", reqs[got].Model)
	}
	// Residency filter: only "low" is warm → it is chosen despite lower priority.
	p := NewPool(100)
	mustAdmit(t, p, Model{ID: "low", WeightBytes: 10})
	if got := NextDecoder(reqs, p); reqs[got].Model != "low" {
		t.Fatalf("NextDecoder(resident=low) = %s, want low", reqs[got].Model)
	}
	// Nothing eligible → -1.
	if got := NextDecoder([]Request{{Model: "x", Decode: 0}}, nil); got != -1 {
		t.Fatalf("NextDecoder(no work) = %d, want -1", got)
	}
}

func TestScheduleSerialDecodeAndConservation(t *testing.T) {
	reqs := []Request{
		{Model: "a", Prefill: 100, Decode: 5, Priority: 5, Seq: 1},
		{Model: "b", Prefill: 200, Decode: 3, Priority: 5, Seq: 2},
	}
	steps, st := Schedule(reqs, 2)

	if st.MaxConcurrentDecode != 1 {
		t.Fatalf("MaxConcurrentDecode = %d, want 1 (the serial-lane invariant)", st.MaxConcurrentDecode)
	}
	if st.PrefillTokens != 300 || st.DecodeTokens != 8 {
		t.Fatalf("tokens prefill=%d decode=%d, want 300/8", st.PrefillTokens, st.DecodeTokens)
	}

	// Every prefill emitted exactly once; decode tokens conserved per model.
	prefillCount := map[ModelID]int{}
	decodeTokens := map[ModelID]int{}
	for _, s := range steps {
		switch s.Phase {
		case Prefill:
			prefillCount[s.Model]++
		case Decode:
			if s.Tokens <= 0 || s.Tokens > 2 {
				t.Fatalf("decode step tokens=%d, want 1..quantum(2)", s.Tokens)
			}
			decodeTokens[s.Model] += s.Tokens
		}
	}
	if prefillCount["a"] != 1 || prefillCount["b"] != 1 {
		t.Fatalf("prefill counts = %v, want one each", prefillCount)
	}
	if decodeTokens["a"] != 5 || decodeTokens["b"] != 3 {
		t.Fatalf("decode tokens = %v, want a=5 b=3", decodeTokens)
	}

	// The lane interleaves: with quantum 2, a and b alternate while both have work,
	// so the decode sub-sequence is not all-a-then-all-b.
	var decodeOrder []ModelID
	for _, s := range steps {
		if s.Phase == Decode {
			decodeOrder = append(decodeOrder, s.Model)
		}
	}
	if len(decodeOrder) < 2 || decodeOrder[0] != "a" || decodeOrder[1] != "b" {
		t.Fatalf("decode order = %v, want interleaved starting a,b", decodeOrder)
	}
}

func TestScheduleEmptyAndDefaults(t *testing.T) {
	steps, st := Schedule(nil, 0)
	if len(steps) != 0 || st.DecodeSteps != 0 || st.MaxConcurrentDecode != 0 {
		t.Fatalf("empty schedule = %v / %+v", steps, st)
	}
	// quantum<=0 defaults to 1: a 3-token decode yields 3 single-token steps.
	steps, st = Schedule([]Request{{Model: "a", Decode: 3}}, 0)
	if st.DecodeSteps != 3 {
		t.Fatalf("DecodeSteps = %d, want 3 (quantum defaulted to 1)", st.DecodeSteps)
	}
	_ = steps
}

func TestDecodeBandwidthAccounting(t *testing.T) {
	steps := []Step{
		{Model: "big", Phase: Prefill, Tokens: 1000}, // prefill is NOT bandwidth-counted
		{Model: "big", Phase: Decode, Tokens: 4},
		{Model: "small", Phase: Decode, Tokens: 10},
		{Model: "unknown", Phase: Decode, Tokens: 100}, // missing weight → 0
	}
	weights := map[ModelID]int64{"big": 1_000_000, "small": 10_000}
	got := DecodeBandwidthBytes(steps, weights)
	want := int64(4)*1_000_000 + int64(10)*10_000
	if got != want {
		t.Fatalf("bandwidth = %d, want %d", got, want)
	}
}

// ---------------------------------------------------------------------------
// Cache-led MTP: speculative accept.
// ---------------------------------------------------------------------------

func TestAcceptGreedy(t *testing.T) {
	cases := []struct {
		name          string
		draft, target []int
		accepted, adv int
		keep, evict   int
	}{
		// All 3 accepted + a bonus 4th from the verify pass → advance 4, evict 0.
		{"all+bonus", []int{1, 2, 3}, []int{1, 2, 3, 4}, 3, 4, 3, 0},
		// All 3 accepted, no bonus position → advance 3, evict 0.
		{"all-no-bonus", []int{1, 2, 3}, []int{1, 2, 3}, 3, 3, 3, 0},
		// Diverge at index 1: keep 1, correct to target → advance 2, evict 2.
		{"partial", []int{1, 9, 9}, []int{1, 2, 3}, 1, 2, 1, 2},
		// Reject at index 0 → advance 1 (the correction), evict all 3.
		{"all-rejected", []int{9, 9, 9}, []int{1, 2, 3}, 0, 1, 0, 3},
		// No draft → a plain decode: advance 1, nothing to keep/evict.
		{"empty-draft", nil, []int{7}, 0, 1, 0, 0},
	}
	for _, c := range cases {
		r := AcceptGreedy(c.draft, c.target)
		if r.Accepted != c.accepted || r.Advance != c.adv || r.KeepKV != c.keep || r.EvictKV != c.evict {
			t.Fatalf("%s: got %+v, want accepted=%d advance=%d keep=%d evict=%d",
				c.name, r, c.accepted, c.adv, c.keep, c.evict)
		}
		// Invariant: every drafted position is either kept or evicted, never lost.
		if r.KeepKV+r.EvictKV != len(c.draft) {
			t.Fatalf("%s: keep+evict=%d != draft len %d", c.name, r.KeepKV+r.EvictKV, len(c.draft))
		}
	}
}

func TestAcceptTree(t *testing.T) {
	// A LINEAR chain must reduce exactly to AcceptGreedy.
	chain := SpecTree{Nodes: []TreeNode{
		{TargetArgmax: 1, Children: []int{1}}, // root: predicts 1
		{Token: 1, TargetArgmax: 2, Children: []int{2}},
		{Token: 2, TargetArgmax: 3, Children: []int{3}},
		{Token: 3, TargetArgmax: 4},
	}}
	tr := AcceptTree(chain)
	gr := AcceptGreedy([]int{1, 2, 3}, []int{1, 2, 3, 4})
	if tr.Advance != gr.Advance || tr.KeepKV != gr.KeepKV || tr.EvictKV != gr.EvictKV {
		t.Fatalf("chain tree %+v != AcceptGreedy %+v", tr, gr)
	}
	if len(tr.Path) != 3 || tr.Path[0] != 1 || tr.Path[2] != 3 {
		t.Fatalf("chain path = %v, want [1 2 3]", tr.Path)
	}

	// A BRANCH: the accepted path descends a non-first sibling at each level.
	branch := SpecTree{Nodes: []TreeNode{
		{TargetArgmax: 5, Children: []int{1, 2}}, // root predicts 5 → node 2 (Token 5)
		{Token: 9},                               // rejected sibling
		{Token: 5, TargetArgmax: 7, Children: []int{3, 4}},
		{Token: 1},                  // rejected sibling (Token 1 != 7)
		{Token: 7, TargetArgmax: 0}, // matches 7 → accepted; predicts 0, no children → stop
	}}
	br := AcceptTree(branch)
	if len(br.Path) != 2 || br.Path[0] != 2 || br.Path[1] != 4 {
		t.Fatalf("branch path = %v, want [2 4]", br.Path)
	}
	if br.Advance != 3 || br.KeepKV != 2 || br.EvictKV != 2 {
		t.Fatalf("branch result = %+v, want advance=3 keep=2 evict=2", br)
	}

	// ALL REJECTED at the root: nothing matches the target's argmax.
	none := SpecTree{Nodes: []TreeNode{
		{TargetArgmax: 100, Children: []int{1, 2}},
		{Token: 1}, {Token: 2},
	}}
	nr := AcceptTree(none)
	if len(nr.Path) != 0 || nr.Advance != 1 || nr.KeepKV != 0 || nr.EvictKV != 2 {
		t.Fatalf("all-rejected = %+v, want path=[] advance=1 keep=0 evict=2", nr)
	}

	// Invariant across all trees: KEEP + EVICT == number of speculative nodes.
	for name, tree := range map[string]SpecTree{"chain": chain, "branch": branch, "none": none} {
		r := AcceptTree(tree)
		if r.KeepKV+r.EvictKV != len(tree.Nodes)-1 {
			t.Fatalf("%s: keep+evict=%d != speculative nodes %d", name, r.KeepKV+r.EvictKV, len(tree.Nodes)-1)
		}
	}

	// Empty tree is a no-op.
	if e := AcceptTree(SpecTree{}); e.Advance != 0 || len(e.Path) != 0 {
		t.Fatalf("empty tree = %+v, want zero", e)
	}
}

func TestPickDrafterCheapestSameFamily(t *testing.T) {
	p := NewPool(1000)
	mustAdmit(t, p, Model{ID: "target", Family: "qwen", WeightBytes: 500})
	mustAdmit(t, p, Model{ID: "mid", Family: "qwen", WeightBytes: 200})
	mustAdmit(t, p, Model{ID: "tiny", Family: "qwen", WeightBytes: 50})
	mustAdmit(t, p, Model{ID: "alien", Family: "llama", WeightBytes: 10})

	// Cheapest same-family peer (not the target itself) → "tiny".
	if d := PickDrafter("target", p); d != "tiny" {
		t.Fatalf("PickDrafter(target) = %q, want tiny", d)
	}
	// A model with no same-family peer → no drafter.
	if d := PickDrafter("alien", p); d != "" {
		t.Fatalf("PickDrafter(alien) = %q, want \"\" (no same-family peer)", d)
	}
	// Unique family ("") → never drafts.
	mustAdmit(t, p, Model{ID: "solo", Family: "", WeightBytes: 5})
	if d := PickDrafter("solo", p); d != "" {
		t.Fatalf("PickDrafter(solo) = %q, want \"\" (empty family)", d)
	}
	// Non-resident active → no drafter.
	if d := PickDrafter("ghost", p); d != "" {
		t.Fatalf("PickDrafter(ghost) = %q, want \"\"", d)
	}
}

func TestCanShare(t *testing.T) {
	base := Model{ID: "base", Family: "qwen", PrefixDigest: "sha-AAA"}
	twin := Model{ID: "twin", Family: "qwen", PrefixDigest: "sha-AAA"} // same family + weights band
	fork := Model{ID: "fork", Family: "qwen", PrefixDigest: "sha-BBB"} // same family, DIFFERENT weights
	alien := Model{ID: "alien", Family: "llama", PrefixDigest: "sha-AAA"}
	bare := Model{ID: "bare", Family: "qwen"} // no declared shareable band

	cases := []struct {
		name string
		a, b Model
		want bool
	}{
		{"identical band shares", base, twin, true},
		{"self always shares", base, base, true},
		{"different weights do NOT share (KV would differ)", base, fork, false},
		{"different family does NOT share", base, alien, false},
		{"empty digest never shares", base, bare, false},
		{"empty digest never shares (reverse)", bare, twin, false},
	}
	for _, c := range cases {
		if got := CanShare(c.a, c.b); got != c.want {
			t.Fatalf("%s: CanShare(%s,%s)=%v, want %v", c.name, c.a.ID, c.b.ID, got, c.want)
		}
	}
}

func TestEffectiveTokensPerVerify(t *testing.T) {
	const eps = 1e-9
	check := func(k int, a, want float64) {
		if got := EffectiveTokensPerVerify(k, a); got < want-eps || got > want+eps {
			t.Fatalf("E(k=%d,a=%g) = %g, want %g", k, a, got, want)
		}
	}
	check(0, 0.9, 1)    // no draft → plain decode
	check(3, 0, 1)      // never accept → 1 real token/verify
	check(2, 1, 3)      // always accept → k+1
	check(1, 0.5, 1.5)  // 1 + 0.5
	check(2, 0.5, 1.75) // 1 + 0.5 + 0.25
	check(4, 1, 5)      // k+1

	// Monotone in acceptance and in draft length (the speedup levers).
	if EffectiveTokensPerVerify(3, 0.8) <= EffectiveTokensPerVerify(3, 0.4) {
		t.Fatal("E must increase with acceptance probability")
	}
	if EffectiveTokensPerVerify(8, 0.8) <= EffectiveTokensPerVerify(3, 0.8) {
		t.Fatal("E must increase with draft length")
	}
}

func TestEnabledDefaultsOff(t *testing.T) {
	t.Setenv(FlagEnv, "")
	if Enabled() {
		t.Fatal("poly-model lane must default OFF (not-yet-production feature)")
	}
	for _, on := range []string{"on", "ON", "1", "true", "yes"} {
		t.Setenv(FlagEnv, on)
		if !Enabled() {
			t.Fatalf("FAK_POLYMODEL=%q must enable", on)
		}
	}
	for _, off := range []string{"off", "0", "false", "no", "maybe"} {
		t.Setenv(FlagEnv, off)
		if Enabled() {
			t.Fatalf("FAK_POLYMODEL=%q must stay OFF", off)
		}
	}
}

func mustAdmit(t *testing.T, p *Pool, m Model) {
	t.Helper()
	if _, err := p.Admit(m); err != nil {
		t.Fatalf("admit %s: %v", m.ID, err)
	}
}
