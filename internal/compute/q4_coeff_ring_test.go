package compute

import "testing"

// Hermetic mirror of the Q4 decode coefficient ring-arena eviction policy implemented in
// internal/compute/cuda/quant_matmul.cuh (#raider-kv-sizing-footprint). The C++ bookkeeping is
// pure integer arithmetic over an absolute cursor / contiguous live window; this Go twin copies
// that algorithm 1:1 (same slot count, same front-pop rule, same dead-region reservation) so the
// POLICY is testable without a GPU. Keep the two in lockstep: any change to the .cuh ring
// (slot count, pop condition, reservation semantics) must be mirrored here.
//
// Invariants pinned below:
//  1. byte bound - the live window (alive + dead regions) never exceeds the arena cap;
//  2. no aliasing - no two live regions ever occupy overlapping physical bytes, even across
//     wrap and across middle-region kills (the contiguity reservation is what guarantees it);
//  3. kill-on-free - a freed weight stops being returned by lookup immediately, but its region
//     stays reserved until the ring laps it (a later mint may not land inside it early);
//  4. FIFO == LRU under the serve decode pattern - every tensor touched once per token in fixed
//     order: when the working set fits, no table re-mints within a cycle;
//  5. physical extent - every reserved region (alive or dead, tail-padding included) fits
//     contiguously inside the arena: offset plus length never crosses the cap, which is what
//     keeps each mint one contiguous kernel write rather than a modulo-wrapped range (#14119).

const q4CoeffMirrorSlots = 512 // mirrors kQ4CoeffSlots

type q4CoeffSlot struct {
	key     string
	out, in int
	length  int64
	alive   bool
	start   int64 // absolute ring position of this region's first byte
}

type q4CoeffRing struct {
	cap    int64
	slots  [q4CoeffMirrorSlots]q4CoeffSlot
	head   int
	n      int
	cursor int64
	span   int64
	mints  int64 // total mint launches (hit + miss), for the residency assertion
	hits   int64
}

func (r *q4CoeffRing) slotAt(k int) int { return (r.head + k) % q4CoeffMirrorSlots }

func (r *q4CoeffRing) popFront() {
	s := r.head
	r.span -= r.slots[s].length
	r.slots[s] = q4CoeffSlot{}
	r.head = (s + 1) % q4CoeffMirrorSlots
	r.n--
}

// evict mirrors q4_coeff_cache_evict: drop the live entry for a freed weight pointer; the region
// stays reserved until the ring laps it.
func (r *q4CoeffRing) evict(key string) {
	for k := 0; k < r.n; k++ {
		s := &r.slots[r.slotAt(k)]
		if !s.alive || s.key != key {
			continue
		}
		s.alive = false
		return
	}
}

// lookup mirrors the scan half of q4_coeffs_for: reports whether a live entry serves the key.
func (r *q4CoeffRing) lookup(key string, out, in int) bool {
	for k := 0; k < r.n; k++ {
		s := &r.slots[r.slotAt(k)]
		if s.alive && s.key == key && s.out == out && s.in == in {
			return true
		}
	}
	return false
}

// mint mirrors q4_coeffs_for minus the kernel launch: returns the physical offset of the table
// (existing hit or fresh region) and whether the lookup/mint succeeded.
func (r *q4CoeffRing) mint(key string, out, in int, bytes int64) (int64, bool) {
	r.mints++
	for k := 0; k < r.n; k++ {
		s := &r.slots[r.slotAt(k)]
		if s.alive && s.key == key && s.out == out && s.in == in {
			r.hits++
			return s.physStart(r), true
		}
	}
	if bytes > r.cap {
		return 0, false
	}
	offset := r.cursor % r.cap
	if bytes > r.cap-offset {
		padding := r.cap - offset
		for r.n > 0 && r.span > r.cap-padding {
			r.popFront()
		}
		if r.n > 0 {
			r.slots[r.slotAt(r.n-1)].length += padding
			r.span += padding
		}
		r.cursor += padding
	}
	for r.n == q4CoeffMirrorSlots || bytes > r.cap-r.span {
		r.popFront()
	}
	phys := r.cursor % r.cap
	if phys+bytes > r.cap {
		// mirrors the .cuh fail-closed physical bound: a drifted mint is refused instead of
		// returning an offset whose contiguous write would overrun the arena end
		return 0, false
	}
	s := &r.slots[r.slotAt(r.n)]
	*s = q4CoeffSlot{key: key, out: out, in: in, length: bytes, alive: true, start: r.cursor}
	r.n++
	r.span += bytes
	r.cursor += bytes
	return phys, true
}

// physStart resolves a slot's physical offset from its absolute start.
func (s *q4CoeffSlot) physStart(r *q4CoeffRing) int64 { return s.start % r.cap }

// liveRegions returns (key, physStart, length) for every alive region, oldest first.
func (r *q4CoeffRing) liveRegions() []struct {
	key  string
	phys int64
	len  int64
} {
	var out []struct {
		key  string
		phys int64
		len  int64
	}
	for k := 0; k < r.n; k++ {
		s := &r.slots[r.slotAt(k)]
		if !s.alive {
			continue
		}
		out = append(out, struct {
			key  string
			phys int64
			len  int64
		}{s.key, s.physStart(r), s.length})
	}
	return out
}

// checkInvariants asserts the byte bound, the contiguous physical bound, and the no-aliasing
// invariant over EVERY reserved region (alive + dead) on a physical byte map. Dead regions carry
// the tail-padding reservations, so they get the same physical-extent check as live tables: a
// region whose offset plus length crossed the cap would be a wrapped kernel write (#14119).
func (r *q4CoeffRing) checkInvariants(t *testing.T) {
	t.Helper()
	if r.span > r.cap {
		t.Fatalf("live window %d exceeds arena cap %d", r.span, r.cap)
	}
	used := make([]string, r.cap)
	for k := 0; k < r.n; k++ {
		s := &r.slots[r.slotAt(k)]
		phys := s.physStart(r)
		if phys < 0 || s.length > r.cap-phys {
			t.Fatalf("contiguous region %q at %d length %d crosses arena cap %d", s.key, phys, s.length, r.cap)
		}
		for i := int64(0); i < s.length; i++ {
			p := phys + i
			if used[p] != "" {
				t.Fatalf("aliasing: physical byte %d held by both %q and %q", p, used[p], s.key)
			}
			used[p] = s.key
		}
	}
}

// TestQ4CoeffRingDecodeCycleFIFOLikeLRU pins invariant 4: the serve decode pattern touches every
// resident Q4_K tensor exactly once per token in fixed layer order, so when the working set fits
// the arena, FIFO recycles nothing and no table re-mints within a cycle.
func TestQ4CoeffRingDecodeCycleFIFOLikeLRU(t *testing.T) {
	const cap = 100
	const layers = 4
	const perTable = 10 // 4 x 10 = 40 <= 100: working set fits
	r := &q4CoeffRing{cap: cap}
	mint := func(layer int) {
		phys, ok := r.mint("layer-"+ringItoa(layer), 1, 1, perTable)
		if !ok {
			t.Fatalf("layer %d: mint refused", layer)
		}
		if phys < 0 || phys >= cap {
			t.Fatalf("layer %d: physical offset %d outside arena", layer, phys)
		}
	}
	for token := 0; token < 10; token++ {
		for layer := 0; layer < layers; layer++ {
			mint(layer)
		}
	}
	if r.hits != 36 {
		t.Fatalf("hits = %d, want 36 (first-touch misses aside, every touch must be a cache hit)", r.hits)
	}
	if got := len(r.liveRegions()); got != layers {
		t.Fatalf("live regions = %d, want %d (working set fully resident)", got, layers)
	}
	r.checkInvariants(t)
}

// TestQ4CoeffRingOverflowRecyclesOldest pins the overflow path: when the working set exceeds the
// arena, the OLDEST region is recycled first (FIFO), the newest table lands where the recycled
// region used to be, and no live table aliases another.
func TestQ4CoeffRingOverflowRecyclesOldest(t *testing.T) {
	const cap = 30
	r := &q4CoeffRing{cap: cap}
	for _, key := range []string{"A", "B", "C"} {
		if _, ok := r.mint(key, 1, 1, 10); !ok {
			t.Fatalf("%s: mint refused", key)
		}
	}
	physD, ok := r.mint("D", 1, 1, 10)
	if !ok {
		t.Fatal("D: mint refused")
	}
	if physD != 0 {
		t.Fatalf("D landed at physical %d, want 0 (A's recycled region)", physD)
	}
	seen := map[string]bool{}
	for _, reg := range r.liveRegions() {
		seen[reg.key] = true
	}
	if seen["A"] || !seen["B"] || !seen["C"] || !seen["D"] {
		t.Fatalf("live set = %v, want {B C D} (A recycled oldest-first)", seen)
	}
	r.checkInvariants(t)
}

// TestQ4CoeffRingKillOnFreeReservesRegion pins invariant 3: freeing a weight removes its table
// from lookup immediately, but the region stays reserved until the ring laps it - a later mint
// may reuse the popped-front region, never a still-reserved dead one.
func TestQ4CoeffRingKillOnFreeReservesRegion(t *testing.T) {
	const cap = 30
	r := &q4CoeffRing{cap: cap}
	for _, key := range []string{"A", "B", "C"} {
		if _, ok := r.mint(key, 1, 1, 10); !ok {
			t.Fatalf("%s: mint refused", key)
		}
	}
	r.evict("B") // B occupies physical [10,20)
	if r.lookup("B", 1, 1) {
		t.Fatal("freed weight B still served from the cache")
	}
	// Mint D: span=30 -> pop A (front) -> span=20 -> D fits at physical 0 (A's lapsed region).
	physD, ok := r.mint("D", 1, 1, 10)
	if !ok {
		t.Fatal("D: mint refused")
	}
	if physD != 0 {
		t.Fatalf("D landed at physical %d, want 0 (A's lapsed region)", physD)
	}
	// Mint E: span=30 (dead B + C + D) -> pop dead B -> E lands at physical 10, B's lapsed region.
	physE, ok := r.mint("E", 1, 1, 10)
	if !ok {
		t.Fatal("E: mint refused")
	}
	if physE != 10 {
		t.Fatalf("E landed at physical %d, want 10 (B's lapsed region)", physE)
	}
	seen := map[string]bool{}
	for _, reg := range r.liveRegions() {
		seen[reg.key] = true
	}
	if seen["B"] || !seen["C"] || !seen["D"] || !seen["E"] {
		t.Fatalf("live set = %v, want {C D E}", seen)
	}
	r.checkInvariants(t)
}

// TestQ4CoeffRingWrapStress hammers the slot ring past its 512-slot boundary with a tight cap so
// the head wraps many times, asserting the byte bound and no-aliasing invariants after every
// single operation.
func TestQ4CoeffRingWrapStress(t *testing.T) {
	const cap = 4096 // 512 slots x 8 bytes: the slot ring saturates exactly
	r := &q4CoeffRing{cap: cap}
	next := 0
	for i := 0; i < 1500; i++ {
		if i%7 == 3 {
			r.evict("w-" + ringItoa(next-1)) // free the most recent weight sometimes
		}
		next++
		if _, ok := r.mint("w-"+ringItoa(next), i%5, i%3, 8); !ok {
			t.Fatalf("mint %d refused", i)
		}
		r.checkInvariants(t)
	}
}

func ringItoa(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

func TestQ4CoeffRingContiguousEnd(t *testing.T) {
	r := &q4CoeffRing{cap: 8}
	if offset, ok := r.mint("A", 1, 1, 6); !ok || offset != 0 {
		t.Fatalf("first mint = %d, %v", offset, ok)
	}
	if offset, ok := r.mint("B", 1, 1, 4); !ok || offset != 0 {
		t.Fatalf("cross-end mint = %d, %v; want contiguous offset zero", offset, ok)
	}
	if r.span != 4 || r.n != 1 || r.lookup("A", 1, 1) {
		t.Fatalf("padding/old owner retained after complete eviction: span=%d n=%d", r.span, r.n)
	}
	r.checkInvariants(t)
}

// TestQ4CoeffRingContiguousWrapCap256 pins the #14119 repro: with a 256-byte arena and two
// distinct 192-byte tables, the second mint must NOT return offset 192 - a 192-byte contiguous
// kernel write there would overrun the arena end. The wrap charges the tail to the oldest region
// and lands the second table at zero. Red on the pre-reservation bookkeeping, which returned 192.
func TestQ4CoeffRingContiguousWrapCap256(t *testing.T) {
	const cap = 256
	r := &q4CoeffRing{cap: cap}
	p1, ok := r.mint("T1", 1, 1, 192)
	if !ok || p1 != 0 {
		t.Fatalf("first table = (%d, %v); want (0, true)", p1, ok)
	}
	p2, ok := r.mint("T2", 1, 1, 192)
	if !ok {
		t.Fatal("second table: mint refused")
	}
	if p2 != 0 {
		t.Fatalf("second table landed at physical %d; a 192-byte contiguous write there overruns cap %d", p2, cap)
	}
	if r.lookup("T1", 1, 1) || !r.lookup("T2", 1, 1) {
		t.Fatal("wrap must recycle the oldest table and keep the newest live")
	}
	if r.span != 192 || r.n != 1 {
		t.Fatalf("window after wrap: span=%d n=%d; want span=192 n=1", r.span, r.n)
	}
	r.checkInvariants(t)
}
