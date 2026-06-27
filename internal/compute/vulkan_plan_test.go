package compute

import (
	"math"
	"testing"
)

// vulkan_plan_test.go — the host-witnessable rungs for the issue #10 Vulkan scaffold. None of
// these need an AMD Vulkan GPU: they pin (1) the sub-buffer suballocator's packing invariants
// (alignment, non-overlap, in-bounds, block-count reduction, Reset reuse), (2) the
// command-buffer planner's submit-count reduction + the BatchRecorder seam's both branches,
// (3) the Q8 GEMM tiled reference's bit-identity to the cpuref decode kernel and its
// window-invariance + the Approx-gate cosine to the f32 reference, and (4) the async-overlap
// pipeline's exact recurrence and bounds. The wall-clock "decode within 2× llama.cpp on an
// RX 7600 / ≥50 tok/s / ≤2× memory" acceptance is deferred to an AMD Vulkan bench node.

// ---- sub-buffer tensor allocation -----------------------------------------------

// interval is a [lo,hi) byte range used to assert no two live sub-ranges overlap in a block.
type interval struct{ lo, hi int64 }

// TestSubAllocatorAlignmentAndNonOverlap is the load-bearing allocator rung: every handed-out
// sub-range starts at an aligned offset, fits inside its block, and never overlaps another live
// range in the same block — the invariants a real sub-buffer binding depends on.
func TestSubAllocatorAlignmentAndNonOverlap(t *testing.T) {
	const align, blockSize = 256, 4096
	a := NewSubAllocator(align, blockSize)
	live := map[int][]interval{}
	for i := 0; i < 500; i++ {
		size := int64(1 + (i*37)%600) // 1..600 bytes, deterministic spread, all < blockSize
		sa := a.Alloc(size)
		if sa.Offset%align != 0 {
			t.Fatalf("alloc %d: offset %d not %d-aligned", i, sa.Offset, align)
		}
		if sa.Offset+sa.Size > blockSize { // in-bounds: no range escapes its block
			t.Fatalf("alloc %d: range [%d,%d) escapes block size %d", i, sa.Offset, sa.Offset+sa.Size, blockSize)
		}
		// in-bounds within its block and non-overlapping with that block's live ranges.
		for _, iv := range live[sa.Block] {
			if sa.Offset < iv.hi && iv.lo < sa.Offset+sa.Size {
				t.Fatalf("alloc %d: [%d,%d) overlaps [%d,%d) in block %d",
					i, sa.Offset, sa.Offset+sa.Size, iv.lo, iv.hi, sa.Block)
			}
		}
		live[sa.Block] = append(live[sa.Block], interval{sa.Offset, sa.Offset + sa.Size})
	}
	if a.Blocks() < 1 {
		t.Fatal("expected at least one backing block")
	}
}

// TestSubAllocatorPacksManyTensors witnesses the actual payoff: 2000 small tensors land in a
// handful of backing blocks instead of 2000 device allocations (the maxMemoryAllocationCount
// pressure the refactor removes), and the live working set never exceeds the reserved memory.
func TestSubAllocatorPacksManyTensors(t *testing.T) {
	const align, blockSize = 64, 1 << 20 // 1 MiB blocks
	a := NewSubAllocator(align, blockSize)
	const n, sz = 2000, 1024
	for i := 0; i < n; i++ {
		a.Alloc(sz)
	}
	// 2000×1024 = 2,048,000 bytes, just over two 1 MiB blocks: a 1000× cut in allocation count.
	if a.Blocks() != 2 {
		t.Fatalf("expected 2 backing blocks for %d×%d bytes, got %d", n, sz, a.Blocks())
	}
	if a.Blocks() >= n {
		t.Fatalf("packing must use far fewer blocks than tensors (%d vs %d)", a.Blocks(), n)
	}
	if a.PeakRequested() != n*sz {
		t.Fatalf("peak live = %d, want %d", a.PeakRequested(), n*sz)
	}
	if a.PeakRequested() > a.Reserved() {
		t.Fatalf("live working set %d exceeds reserved %d", a.PeakRequested(), a.Reserved())
	}
	if eff := a.PackingEfficiency(); eff <= 0.9 || eff > 1.0 {
		t.Fatalf("packing efficiency %.4f out of expected (0.9,1.0]", eff)
	}
}

// TestSubAllocatorResetReuses checks Reset rewinds for transient reuse without re-reserving:
// re-running the identical alloc pattern after Reset must not grow the block count or the
// reserved bytes (the per-forward-pass recycle the transient pool needs).
func TestSubAllocatorResetReuses(t *testing.T) {
	a := NewSubAllocator(64, 4096)
	pattern := []int64{100, 256, 1000, 64, 2048}
	for _, s := range pattern {
		a.Alloc(s)
	}
	blocks, reserved, peak := a.Blocks(), a.Reserved(), a.PeakRequested()
	a.Reset()
	if a.Live() != 0 {
		t.Fatalf("Reset must zero live, got %d", a.Live())
	}
	for _, s := range pattern {
		a.Alloc(s)
	}
	if a.Blocks() != blocks || a.Reserved() != reserved {
		t.Fatalf("Reset must reuse: blocks %d->%d reserved %d->%d", blocks, a.Blocks(), reserved, a.Reserved())
	}
	if a.PeakRequested() != peak {
		t.Fatalf("peak should be unchanged by an identical re-run: %d != %d", a.PeakRequested(), peak)
	}
}

// TestSubAllocatorOversize checks a request larger than a block gets its own exact-sized
// dedicated block at offset 0, and the allocator keeps serving normal-sized requests after it.
func TestSubAllocatorOversize(t *testing.T) {
	a := NewSubAllocator(64, 1024)
	big := a.Alloc(4096) // > blockSize
	if big.Offset != 0 {
		t.Fatalf("oversize alloc must start at offset 0, got %d", big.Offset)
	}
	if a.Reserved() < 4096 {
		t.Fatalf("oversize block must reserve >= 4096, got %d", a.Reserved())
	}
	small := a.Alloc(128) // must still be served (in a fresh block)
	if small.Offset+small.Size > a.Reserved() {
		t.Fatal("small alloc after oversize escaped reserved memory")
	}
	if small.Offset%64 != 0 {
		t.Fatalf("small alloc offset %d not aligned", small.Offset)
	}
}

// TestSubAllocatorAllocCountCap pins the safety property the sub-buffer refactor exists for: the
// device caps the NUMBER of live allocations (maxMemoryAllocationCount, ~4096 on desktop drivers,
// flagged in vulkan_shim.cpp), so a deep model that would need thousands of per-tensor
// allocations must collapse to a handful of packed backing blocks that fit under the cap.
func TestSubAllocatorAllocCountCap(t *testing.T) {
	const align, blockSize = 256, 1 << 20 // 1 MiB blocks
	const capLimit = 4096                 // a typical desktop maxMemoryAllocationCount
	a := NewSubAllocator(align, blockSize)
	const n, sz = 5000, 2048 // more tensors than the cap, each tiny
	for i := 0; i < n; i++ {
		a.Alloc(sz)
	}
	if a.PeakAllocations() != n {
		t.Fatalf("peak allocations = %d, want %d (the naive per-tensor count)", a.PeakAllocations(), n)
	}
	if a.PeakAllocations() <= capLimit {
		t.Fatalf("test premise broken: %d tensors must exceed the %d-allocation cap", a.PeakAllocations(), capLimit)
	}
	// The naive one-allocation-per-tensor path would blow the cap; the packer must fit under it.
	if !a.FitsAllocCount(capLimit) {
		t.Fatalf("packed %d backing blocks must fit under the %d-allocation cap", a.Blocks(), capLimit)
	}
	if a.Blocks() >= n {
		t.Fatalf("packing must use far fewer blocks (%d) than tensors (%d)", a.Blocks(), n)
	}
	// A cap below the packed block count must report unfit (no false safety); a non-positive cap
	// means "uncapped" and always fits.
	if a.FitsAllocCount(a.Blocks() - 1) {
		t.Fatalf("cap %d below block count %d must report unfit", a.Blocks()-1, a.Blocks())
	}
	if !a.FitsAllocCount(0) {
		t.Fatal("a non-positive cap means uncapped (always fits)")
	}
}

// TestSubAllocatorResetZerosLiveAllocCount checks Reset rewinds the live tensor count for the next
// pass while the peak (the residency high-water the device cap is checked against) and the packed
// block count persist — the per-pass recycle must not forget the cap-safety high-water.
func TestSubAllocatorResetZerosLiveAllocCount(t *testing.T) {
	a := NewSubAllocator(64, 4096)
	for _, s := range []int64{100, 256, 1000} {
		a.Alloc(s)
	}
	if a.Allocations() != 3 {
		t.Fatalf("live allocations = %d, want 3", a.Allocations())
	}
	blocks, peak := a.Blocks(), a.PeakAllocations()
	a.Reset()
	if a.Allocations() != 0 {
		t.Fatalf("Reset must zero live allocations, got %d", a.Allocations())
	}
	if a.PeakAllocations() != peak || a.Blocks() != blocks {
		t.Fatalf("Reset must preserve peak/blocks: peak %d->%d blocks %d->%d", peak, a.PeakAllocations(), blocks, a.Blocks())
	}
}

// ---- single command-buffer per forward pass -------------------------------------

// buildForwardPlan records a representative transformer forward pass: per layer a
// norm/q/k/v/attn/o/gate/up/down dispatch chain (all device-resident, no host read), then a
// single final argmax that reads the host. This is the op stream the command-buffer planner
// collapses to one submit.
func buildForwardPlan(layers int) *ForwardPlan {
	p := &ForwardPlan{}
	for l := 0; l < layers; l++ {
		for _, name := range []string{"rmsnorm", "q_proj", "k_proj", "v_proj", "attn", "o_proj", "ffn_gate", "ffn_up", "ffn_down"} {
			p.RecordDispatch(name)
		}
	}
	p.RecordHostRead("argmax")
	return p
}

// TestForwardPlanSubmitReduction is the command-buffer rung: a whole forward pass that reads the
// host exactly once (the final argmax) costs ONE batched submit vs one-per-op naively, so the
// single-command-buffer recording removes len-1 queue submits + fence stalls.
func TestForwardPlanSubmitReduction(t *testing.T) {
	const layers = 32
	p := buildForwardPlan(layers)
	wantNaive := layers*9 + 1
	if p.NaiveSubmits() != wantNaive {
		t.Fatalf("naive submits = %d, want %d", p.NaiveSubmits(), wantNaive)
	}
	if p.Submits() != 1 {
		t.Fatalf("a pass with one final host read must batch to 1 submit, got %d", p.Submits())
	}
	if p.SubmitReduction() != wantNaive-1 {
		t.Fatalf("submit reduction = %d, want %d", p.SubmitReduction(), wantNaive-1)
	}
}

// TestForwardPlanMidStreamReads pins the fence accounting at the boundaries: a mid-stream host
// read closes a command buffer, trailing un-read dispatches force a final flush, a pass ending
// in a host read needs no trailing flush, and an empty pass costs nothing.
func TestForwardPlanMidStreamReads(t *testing.T) {
	cases := []struct {
		ops  []DispatchOp
		want int
	}{
		{nil, 0}, // empty
		{[]DispatchOp{{Name: "a"}, {Name: "b"}}, 1},                                   // no host read -> one final flush
		{[]DispatchOp{{Name: "a"}, {Name: "r", HostRead: true}}, 1},                   // ends in a read -> exactly one
		{[]DispatchOp{{Name: "a"}, {Name: "r", HostRead: true}, {Name: "b"}}, 2},      // read then trailing -> two
		{[]DispatchOp{{Name: "r1", HostRead: true}, {Name: "r2", HostRead: true}}, 2}, // two reads back to back
	}
	for i, c := range cases {
		p := &ForwardPlan{}
		for _, op := range c.ops {
			p.Record(op)
		}
		if got := p.Submits(); got != c.want {
			t.Fatalf("case %d: Submits = %d, want %d", i, got, c.want)
		}
	}
}

// fakeBatchBE is a BatchRecorder built on the cpu-ref backend, so the batched branch of
// RecordForward can be exercised with no Vulkan device present (the analogue of fakeGraphBE).
type fakeBatchBE struct {
	*cpuBackend
	began, flushed int
}

func (f *fakeBatchBE) BeginBatch() { f.began++ }
func (f *fakeBatchBE) FlushBatch() { f.flushed++ }

// TestRecordForwardSeam exercises both branches of the BatchRecorder seam on a non-Vulkan
// build: a recorder wraps the body in exactly one BeginBatch/FlushBatch and reports batched;
// the plain cpu-ref (not a recorder) runs the body eagerly and reports false.
func TestRecordForwardSeam(t *testing.T) {
	be := &fakeBatchBE{cpuBackend: cpu()}
	ran := 0
	if !RecordForward(be, func() { ran++ }) {
		t.Fatal("a BatchRecorder must report batched=true")
	}
	if be.began != 1 || be.flushed != 1 || ran != 1 {
		t.Fatalf("batched path: began=%d flushed=%d ran=%d, want 1/1/1", be.began, be.flushed, ran)
	}

	ran = 0
	if RecordForward(cpu(), func() { ran++ }) {
		t.Fatal("cpu-ref is not a BatchRecorder; must report batched=false")
	}
	if ran != 1 {
		t.Fatalf("eager path must run body once, ran %d", ran)
	}
}

// ---- Q8 device-GEMM shader (host-exact tiled reference) -------------------------

// q8Weight quantizes a random f32 [out,in] weight to Q8_0 and returns its codes + per-block
// scales (the same buffers vulkan.go uploads and the shader binds).
func q8Weight(s *lcg, out, in int) (codes []int8, scales []float32) {
	w := randVec(s, out*in)
	q := QuantizeQ8(cpu(), []int{out, in}, w, 32)
	return q.buf.(*hostBuf).i8, q.Quant.Scale
}

// TestQ8TiledGEMMBitExactToQdot8 is the load-bearing Q8 rung: every output cell of the tiled
// device-shader reference equals the cpuref decode kernel qdot8scalar over the same per-block
// activation quantization — so the shader the device runs has a byte-exact host oracle.
func TestQ8TiledGEMMBitExactToQdot8(t *testing.T) {
	var s lcg = 4242
	const block = 32
	for _, d := range []struct{ out, in, P, win int }{
		{8, 32, 1, 0},     // one block, single window
		{16, 96, 3, 1},    // window of 1 block (forces the windowing loop every block)
		{7, 256, 5, 3},    // window not dividing the block count
		{32, 8960, 2, 64}, // a 1.5B FFN down_proj width, the shader's WIN_BLOCKS=64 window
	} {
		codes, scales := q8Weight(&s, d.out, d.in)
		X := randVec(&s, d.P*d.in)
		Y := Q8TiledGEMM(codes, scales, X, d.out, d.in, d.P, d.win)
		nblk := d.in / block
		for tk := 0; tk < d.P; tk++ {
			qx, dx := quantizeVecQ8(X[tk*d.in:tk*d.in+d.in], block)
			for o := 0; o < d.out; o++ {
				want := qdot8scalar(codes[o*d.in:o*d.in+d.in], scales[o*nblk:o*nblk+nblk], qx, dx, block)
				if math.Float32bits(Y[tk*d.out+o]) != math.Float32bits(want) {
					t.Fatalf("dims %v cell [%d,%d]: %v != qdot8scalar %v", d, tk, o, Y[tk*d.out+o], want)
				}
			}
		}
	}
}

// TestQ8TiledGEMMWindowInvariance witnesses the shader's input-tiling claim: any window size
// produces byte-identical output to the single-window pass, because acc accumulates in pure
// block order regardless of which blocks are staged together.
func TestQ8TiledGEMMWindowInvariance(t *testing.T) {
	var s lcg = 909
	out, in, P := 12, 512, 4 // 16 blocks
	codes, scales := q8Weight(&s, out, in)
	X := randVec(&s, P*in)
	full := Q8TiledGEMM(codes, scales, X, out, in, P, 0) // single window
	for _, win := range []int{1, 2, 3, 5, 7, 16, 64} {
		got := Q8TiledGEMM(codes, scales, X, out, in, P, win)
		for i := range full {
			if math.Float32bits(got[i]) != math.Float32bits(full[i]) {
				t.Fatalf("win=%d cell %d drift: %v != %v (windowing must not move a byte)", win, i, got[i], full[i])
			}
		}
	}
}

// TestQ8TiledGEMMApproxCosine holds the tiled Q8 GEMM to the Q8 lane's Approx contract: its
// logits track the f32 reference closely (cosine well above the device gate), so the shader is a
// faithful quantization of the matmul, not a divergent kernel.
func TestQ8TiledGEMMApproxCosine(t *testing.T) {
	var s lcg = 321
	out, in, P := 48, 256, 3
	w := randVec(&s, out*in)
	q := QuantizeQ8(cpu(), []int{out, in}, w, 32)
	codes, scales := q.buf.(*hostBuf).i8, q.Quant.Scale
	X := randVec(&s, P*in)
	got := Q8TiledGEMM(codes, scales, X, out, in, P, 0)
	// f32 reference: per (t,o) fdot of the dense weight row and the activation row.
	ref := make([]float32, P*out)
	for tk := 0; tk < P; tk++ {
		for o := 0; o < out; o++ {
			ref[tk*out+o] = fdot(w[o*in:o*in+in], X[tk*in:tk*in+in])
		}
	}
	if c := cosine(got, ref); c < 0.999 {
		t.Fatalf("Q8 tiled GEMM cosine to f32 reference = %.6f, want >= 0.999", c)
	}
}

// ---- async-compute pipeline structure -------------------------------------------

// TestAsyncOverlapBalancedPipeline pins the exact pipeline makespan for n equal stages
// (T=C=1): serial = 2n, overlapped = n+1, so the speedup rises toward 2 with depth — the
// structural ceiling of double-buffered async compute.
func TestAsyncOverlapBalancedPipeline(t *testing.T) {
	for _, n := range []int{1, 2, 4, 32} {
		stages := make([]OverlapStage, n)
		for i := range stages {
			stages[i] = OverlapStage{ComputeCost: 1, TransferCost: 1}
		}
		r := AsyncOverlap(stages)
		if r.Serial != float64(2*n) {
			t.Fatalf("n=%d serial = %v, want %v", n, r.Serial, 2*n)
		}
		if r.Overlapped != float64(n+1) {
			t.Fatalf("n=%d overlapped = %v, want %v", n, r.Overlapped, n+1)
		}
		if n == 1 && r.Overlapped != r.Serial {
			t.Fatalf("a single stage cannot overlap: %v != %v", r.Overlapped, r.Serial)
		}
		if n > 1 && (r.Speedup <= 1.0 || r.Speedup > 2.0) {
			t.Fatalf("n=%d speedup %.4f must be in (1,2]", n, r.Speedup)
		}
	}
}

// TestAsyncOverlapBoundsAndDegenerate checks the recurrence honors its bounds: overlapped never
// beats max(Σtransfer, Σcompute) or the head+tail, never exceeds serial, and degenerate inputs
// (no transfers to hide, or an empty pass) report a speedup of exactly 1.
func TestAsyncOverlapBoundsAndDegenerate(t *testing.T) {
	stages := []OverlapStage{
		{Name: "l0", ComputeCost: 3, TransferCost: 2},
		{Name: "l1", ComputeCost: 1, TransferCost: 5},
		{Name: "l2", ComputeCost: 4, TransferCost: 1},
	}
	r := AsyncOverlap(stages)
	var sumT, sumC float64
	for _, s := range stages {
		sumT += s.TransferCost
		sumC += s.ComputeCost
	}
	lower := math.Max(math.Max(sumT, sumC), stages[0].TransferCost+stages[len(stages)-1].ComputeCost)
	if r.Overlapped < lower {
		t.Fatalf("overlapped %v below lower bound %v", r.Overlapped, lower)
	}
	if r.Overlapped > r.Serial {
		t.Fatalf("overlapped %v exceeds serial %v", r.Overlapped, r.Serial)
	}

	// transfers free -> nothing to overlap; overlapped == serial == Σcompute, speedup 1.
	free := []OverlapStage{{ComputeCost: 2}, {ComputeCost: 3}, {ComputeCost: 1}}
	rf := AsyncOverlap(free)
	if rf.Overlapped != rf.Serial || rf.Serial != 6 || rf.Speedup != 1 {
		t.Fatalf("transfer-free: overlapped=%v serial=%v speedup=%v, want 6/6/1", rf.Overlapped, rf.Serial, rf.Speedup)
	}

	// empty pass -> zero makespan, speedup 1 (not a divide-by-zero).
	re := AsyncOverlap(nil)
	if re.Serial != 0 || re.Overlapped != 0 || re.Speedup != 1 {
		t.Fatalf("empty: serial=%v overlapped=%v speedup=%v, want 0/0/1", re.Serial, re.Overlapped, re.Speedup)
	}
}
