package model

import (
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// paging_ring_test.go — the gate for pagedRing (paging_ring.go), the bounded per-weight resident
// cache (the "per-weight VRAM ring" of the #2726 offload step). Under the cpu-ref backend it pins
// the ring's contract: a ring GEMM is bit-equal to a resident GEMM on both a hit and a miss; the
// hit/pageIn/evict counters are exact; used()<=budget() always (the polymodel bound); an evicted
// weight pages back in on next use; and pinned/too-large admits behave per polymodel.

// ringResident computes the resident-MatMul reference for weight w [out,in] against xt — the same
// reference paging_test.go pins pagedKernel against, so "bit-equal to resident" means the identical
// thing for both primitives.
func ringResident(t *testing.T, be compute.Backend, out, in int, w []float32, xt compute.Tensor) []float32 {
	t.Helper()
	wRes := be.Upload(compute.NewF32(be, []int{out, in}, w), compute.F32)
	return be.Read(be.MatMul(wRes, xt))
}

func ringEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestPagedRingMatMulBitEqualResident: with a budget that fits every weight, each named weight's ring
// GEMM is byte-for-byte the resident GEMM; a repeated access is a HIT (no new page-in) and is still
// bit-equal. Proves the resident-reuse path is numerically identical to a fresh upload.
func TestPagedRingMatMulBitEqualResident(t *testing.T) {
	be := compute.Default()
	const out, in = 8, 16
	rng := rand.New(rand.NewSource(21))
	xt := be.Upload(compute.NewF32(be, []int{in}, pagingRandVec(rng, in)), compute.F32)

	names := []string{"experts.0", "experts.1", "experts.2"}
	weights := map[string][]float32{}
	refs := map[string][]float32{}
	for _, n := range names {
		w := pagingRandVec(rng, out*in)
		weights[n] = w
		refs[n] = ringResident(t, be, out, in, w, xt)
	}

	r := newPagedRing(be, 3*100) // fits all three
	for _, n := range names {
		got := r.matMul(n, []int{out, in}, weights[n], xt, 100, false)
		if !ringEqual(got, refs[n]) {
			t.Fatalf("%s: ring GEMM not bit-equal to resident on miss", n)
		}
	}
	if r.pageIn != 3 || r.hit != 0 || r.evict != 0 {
		t.Fatalf("after 3 distinct misses: pageIn=%d hit=%d evict=%d, want 3/0/0", r.pageIn, r.hit, r.evict)
	}
	if r.residentCount() != 3 {
		t.Fatalf("residentCount=%d, want 3 (all fit the budget)", r.residentCount())
	}
	// Re-access experts.0 → a HIT: no new page-in, handle reused, still bit-equal.
	got := r.matMul("experts.0", []int{out, in}, weights["experts.0"], xt, 100, false)
	if !ringEqual(got, refs["experts.0"]) {
		t.Fatalf("experts.0: ring GEMM not bit-equal to resident on HIT")
	}
	if r.pageIn != 3 || r.hit != 1 {
		t.Fatalf("after a hit: pageIn=%d hit=%d, want 3/1", r.pageIn, r.hit)
	}
}

// TestPagedRingEvictsColdestUnderBudget: with a budget that fits exactly ONE weight, cycling through
// two weights forces LRU eviction each switch; the resident footprint never exceeds the budget, an
// evicted weight pages back IN on its next use (not silently lost), and every result stays bit-equal.
func TestPagedRingEvictsColdestUnderBudget(t *testing.T) {
	be := compute.Default()
	const out, in = 4, 8
	rng := rand.New(rand.NewSource(22))
	xt := be.Upload(compute.NewF32(be, []int{in}, pagingRandVec(rng, in)), compute.F32)
	w1 := pagingRandVec(rng, out*in)
	w2 := pagingRandVec(rng, out*in)
	ref1 := ringResident(t, be, out, in, w1, xt)
	ref2 := ringResident(t, be, out, in, w2, xt)

	r := newPagedRing(be, 100) // fits exactly one weight of 100 bytes

	assertBudget := func() {
		if r.used() > r.budget() {
			t.Fatalf("used()=%d exceeds budget()=%d — the ring bound was violated", r.used(), r.budget())
		}
		if r.residentCount() > 1 {
			t.Fatalf("residentCount()=%d with a one-weight budget", r.residentCount())
		}
	}

	if got := r.matMul("w1", []int{out, in}, w1, xt, 100, false); !ringEqual(got, ref1) {
		t.Fatal("w1 miss not bit-equal to resident")
	}
	assertBudget()
	if !r.isResident("w1") || r.isResident("w2") {
		t.Fatal("after w1: expected only w1 resident")
	}
	// w2 does not fit alongside w1 → w1 (coldest unpinned) is evicted.
	if got := r.matMul("w2", []int{out, in}, w2, xt, 100, false); !ringEqual(got, ref2) {
		t.Fatal("w2 miss not bit-equal to resident")
	}
	assertBudget()
	if r.isResident("w1") || !r.isResident("w2") {
		t.Fatal("after w2: expected w1 evicted, w2 resident")
	}
	if r.pageIn != 2 || r.evict != 1 {
		t.Fatalf("after w1,w2: pageIn=%d evict=%d, want 2/1", r.pageIn, r.evict)
	}
	// Re-access w1 → it pages IN again (not cached across its eviction), evicting w2. Still bit-equal.
	if got := r.matMul("w1", []int{out, in}, w1, xt, 100, false); !ringEqual(got, ref1) {
		t.Fatal("w1 re-page-in not bit-equal to resident")
	}
	assertBudget()
	if r.pageIn != 3 || r.evict != 2 || r.hit != 0 {
		t.Fatalf("after re-accessing w1: pageIn=%d evict=%d hit=%d, want 3/2/0", r.pageIn, r.evict, r.hit)
	}
}

// TestPagedRingPinnedNeverEvicted: a pinned resident is never dropped to admit a newcomer — the
// newcomer's admit returns ErrPinnedNoRoom, so matMul returns nil (caller falls back) and the
// resident set is unchanged. Mirrors polymodel's pinned-exemption at per-weight granularity.
func TestPagedRingPinnedNeverEvicted(t *testing.T) {
	be := compute.Default()
	const out, in = 4, 8
	rng := rand.New(rand.NewSource(23))
	xt := be.Upload(compute.NewF32(be, []int{in}, pagingRandVec(rng, in)), compute.F32)
	w1 := pagingRandVec(rng, out*in)
	w2 := pagingRandVec(rng, out*in)
	ref1 := ringResident(t, be, out, in, w1, xt)

	r := newPagedRing(be, 100) // one-weight budget
	if got := r.matMul("pinned", []int{out, in}, w1, xt, 100, true); !ringEqual(got, ref1) {
		t.Fatal("pinned weight miss not bit-equal to resident")
	}
	// w2 cannot be admitted without dropping the pinned resident → nil, nothing changes.
	if got := r.matMul("w2", []int{out, in}, w2, xt, 100, false); got != nil {
		t.Fatalf("over-pinned-budget matMul returned %d floats, want nil (caller fallback)", len(got))
	}
	if !r.isResident("pinned") || r.isResident("w2") {
		t.Fatal("pinned weight must survive; w2 must not be admitted")
	}
	if r.pageIn != 1 || r.evict != 0 {
		t.Fatalf("pageIn=%d evict=%d, want 1/0 (pinned held, w2 refused)", r.pageIn, r.evict)
	}
}

// TestPagedRingWeightTooLargeReturnsNil: a weight whose byte size alone exceeds the budget cannot be
// made resident (polymodel.ErrTooLarge) — matMul returns nil and pages nothing, so the caller routes
// it to a per-op paged / host GEMM instead. The bounded ring never overcommits the device.
func TestPagedRingWeightTooLargeReturnsNil(t *testing.T) {
	be := compute.Default()
	const out, in = 4, 8
	rng := rand.New(rand.NewSource(24))
	xt := be.Upload(compute.NewF32(be, []int{in}, pagingRandVec(rng, in)), compute.F32)
	w := pagingRandVec(rng, out*in)

	r := newPagedRing(be, 50) // budget 50 < the weight's declared 100 bytes
	if got := r.matMul("huge", []int{out, in}, w, xt, 100, false); got != nil {
		t.Fatalf("too-large matMul returned %d floats, want nil", len(got))
	}
	if r.pageIn != 0 || r.residentCount() != 0 {
		t.Fatalf("pageIn=%d residentCount=%d, want 0/0 (nothing admitted)", r.pageIn, r.residentCount())
	}
}

// The Q8 witnesses below size the ring budget with q8ResidentBytes — the resident footprint of a
// Q8_0 weight the way the budget accounts it (one int8 code per weight plus one f32 scale per
// 32-wide block, the 1.125 B/weight of q8Tensor's doc), so used()<=budget() is meaningful. It moved
// to expert_ring_hal.go when the live HAL began sizing ring admissions with the same arithmetic
// (#5611); this is now one definition serving both.

// TestPagedRingMatMulQ8BitEqualResident is the Q8_0 twin of TestPagedRingMatMulBitEqualResident: with
// a budget that fits every Q8 weight, each named weight's ring GEMM is byte-for-byte a resident Q8
// GEMM on a MISS and again on a HIT, the hit/pageIn counters are exact, and used()<=budget(). Proves
// the ring's quantized page-in / resident-reuse path is numerically identical to a fresh resident Q8
// upload — the "quantized, not f32-only" scope of #3174 R1, at Q8 granularity.
func TestPagedRingMatMulQ8BitEqualResident(t *testing.T) {
	be := compute.Default()
	const out, in = 8, 64 // in must be a multiple of qBlk (32)
	rng := rand.New(rand.NewSource(21))
	xt := be.Upload(compute.NewF32(be, []int{in}, pagingRandVec(rng, in)), compute.F32)

	names := []string{"experts.0", "experts.1", "experts.2"}
	qts := map[string]*q8Tensor{}
	refs := map[string][]float32{}
	for _, n := range names {
		qt := quantizeQ8(pagingRandVec(rng, out*in), out, in)
		qts[n] = qt
		// Resident Q8 reference: the same Upload(NewQ8(...), Q8_0)+MatMul the ring performs on a miss.
		wRes := be.Upload(compute.NewQ8(be, []int{out, in}, qt.q, qt.d, qBlk), compute.Q8_0)
		refs[n] = be.Read(be.MatMul(wRes, xt))
	}

	var total int64
	for _, n := range names {
		total += q8ResidentBytes(qts[n])
	}
	r := newPagedRing(be, total) // fits all three Q8 weights

	for _, n := range names { // three distinct misses
		if got := r.matMulQ8(n, qts[n], xt, q8ResidentBytes(qts[n]), false); !ringEqual(got, refs[n]) {
			t.Fatalf("%s: Q8 ring GEMM not bit-equal to resident on miss", n)
		}
	}
	if r.pageIn != 3 || r.hit != 0 || r.evict != 0 {
		t.Fatalf("after 3 distinct Q8 misses: pageIn=%d hit=%d evict=%d, want 3/0/0", r.pageIn, r.hit, r.evict)
	}
	// Re-access experts.0 → a HIT: no new page-in, resident handle reused, still bit-equal.
	if got := r.matMulQ8("experts.0", qts["experts.0"], xt, q8ResidentBytes(qts["experts.0"]), false); !ringEqual(got, refs["experts.0"]) {
		t.Fatal("experts.0: Q8 ring GEMM not bit-equal to resident on HIT")
	}
	if r.pageIn != 3 || r.hit != 1 {
		t.Fatalf("after a Q8 hit: pageIn=%d hit=%d, want 3/1", r.pageIn, r.hit)
	}
	if r.used() > r.budget() {
		t.Fatalf("used()=%d exceeds budget()=%d — the ring bound was violated", r.used(), r.budget())
	}
}

// TestPagedRingQ8EvictsAndRepages: with a Q8 budget that fits only two of three experts, admitting the
// third evicts the coldest (LRU); the evicted expert is NOT lost — it pages back in on its next use,
// still bit-equal — and the resident footprint never exceeds the budget. The quantized form of the
// bounded-residency contract that distinguishes the ring from splitKernel's unbounded host set.
func TestPagedRingQ8EvictsAndRepages(t *testing.T) {
	be := compute.Default()
	const out, in = 8, 64
	rng := rand.New(rand.NewSource(7))
	xt := be.Upload(compute.NewF32(be, []int{in}, pagingRandVec(rng, in)), compute.F32)

	names := []string{"experts.0", "experts.1", "experts.2"}
	qts := map[string]*q8Tensor{}
	refs := map[string][]float32{}
	for _, n := range names {
		qt := quantizeQ8(pagingRandVec(rng, out*in), out, in)
		qts[n] = qt
		wRes := be.Upload(compute.NewQ8(be, []int{out, in}, qt.q, qt.d, qBlk), compute.Q8_0)
		refs[n] = be.Read(be.MatMul(wRes, xt))
	}

	// Budget fits exactly two Q8 weights; the three are equal-sized, so the third admit evicts one.
	r := newPagedRing(be, q8ResidentBytes(qts["experts.0"])+q8ResidentBytes(qts["experts.1"]))
	for _, n := range names {
		if got := r.matMulQ8(n, qts[n], xt, q8ResidentBytes(qts[n]), false); !ringEqual(got, refs[n]) {
			t.Fatalf("%s: Q8 miss not bit-equal to resident", n)
		}
	}
	if r.isResident("experts.0") {
		t.Fatal("experts.0 (LRU) should have been evicted to admit experts.2")
	}
	if r.evict != 1 || r.used() > r.budget() {
		t.Fatalf("3 Q8 admits into a 2-slot budget: evict=%d used=%d budget=%d, want evict 1 and used<=budget", r.evict, r.used(), r.budget())
	}
	// experts.0 pages back in on next use, still bit-equal (bounded residency, not silently lost).
	if got := r.matMulQ8("experts.0", qts["experts.0"], xt, q8ResidentBytes(qts["experts.0"]), false); !ringEqual(got, refs["experts.0"]) {
		t.Fatal("experts.0 Q8 re-page-in not bit-equal to resident")
	}
	if r.used() > r.budget() {
		t.Fatalf("used()=%d exceeds budget()=%d after repage", r.used(), r.budget())
	}
}

// TestPagedRingMatMulQ4KBitEqualResident: the Q4_K twin — a raw-super-block Q4_K weight's ring GEMM is
// byte-for-byte a resident Q4_K GEMM on a miss and a hit. Proves matMulQ4K stages the dequant-fused
// Q4_K path (the ~0.56 B/weight expert residency of #3174 R1) through the same bit-exact lifecycle,
// so all three dtypes (f32/Q8/Q4_K) share the one witnessed matMulStaged core.
func TestPagedRingMatMulQ4KBitEqualResident(t *testing.T) {
	be := compute.Default()
	const out, in = 8, 256 // in must be a multiple of qkK (the Q4_K super-block K)
	rng := rand.New(rand.NewSource(31))
	xt := be.Upload(compute.NewF32(be, []int{in}, pagingRandVec(rng, in)), compute.F32)

	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	blk := make([]byte, q4kBlockBytes)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			randQ4KBlock(rng, blk)
			copy(raw[(o*nblk+b)*q4kBlockBytes:], blk)
		}
	}
	qt := quantizeQ4KFromRaw(raw, out, in)
	wBytes := int64(len(qt.raw))

	wRes := be.Upload(compute.NewQ4K(be, []int{out, in}, qt.raw), compute.Q4_K)
	ref := be.Read(be.MatMul(wRes, xt))

	r := newPagedRing(be, wBytes) // fits exactly the one Q4_K weight
	if got := r.matMulQ4K("experts.0", qt, xt, wBytes, false); !ringEqual(got, ref) {
		t.Fatal("Q4_K ring GEMM not bit-equal to resident on miss")
	}
	if got := r.matMulQ4K("experts.0", qt, xt, wBytes, false); !ringEqual(got, ref) {
		t.Fatal("Q4_K ring GEMM not bit-equal to resident on hit")
	}
	if r.pageIn != 1 || r.hit != 1 || r.used() > r.budget() {
		t.Fatalf("Q4_K: pageIn=%d hit=%d used=%d budget=%d, want 1/1 and used<=budget", r.pageIn, r.hit, r.used(), r.budget())
	}
}
