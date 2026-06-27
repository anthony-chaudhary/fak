package compute

// vulkan_plan.go — the host-tractable scaffold for issue #10 (B-002), "optimize the fak
// Vulkan backend." The Acceptance bullets that pin the win — "decode within 2× llama.cpp on
// an RX 7600", "throughput ≥ 50 tok/s", "memory ≤ 2× baseline" — are WALL-CLOCK and
// device-residency measurements on an AMD Vulkan GPU: they cannot be witnessed on a host
// without one, and are deferred to an AMD Vulkan bench node (see the report). This file ships
// the four structural pieces the perf work stands on, each the honest, hardware-independent
// form of one scope bullet, in the same shape prefill.go (issue #9) used for the CUDA lane —
// pure Go, always compiled, unit-tested, with the device-only half pinned behind a build tag:
//
//   1. SubAllocator / SubAlloc — "sub-buffer tensor allocation." Today vulkan.go's dalloc
//      issues one fvk_malloc (one VkBuffer + one VkDeviceMemory) PER tensor; a real device
//      caps the allocation count (maxMemoryAllocationCount, ~4096 on desktop drivers) and
//      charges fixed per-allocation overhead. This bump suballocator packs many tensors into a
//      few large backing blocks at aligned sub-offsets — pure byte arithmetic, so the packing
//      invariants (alignment, non-overlap, in-bounds, residency high-water) are testable here.
//
//   2. ForwardPlan / DispatchOp + BatchRecorder — "single command-buffer per forward pass."
//      The planner counts how many GPU SUBMISSIONS a pass costs naively (one per op) vs batched
//      (one command buffer, fenced only at host reads) — the queue-submit + fence-stall
//      reduction, countable with no device. BatchRecorder is the always-compiled HAL seam
//      (BeginBatch/FlushBatch) the vulkanBackend already exposes to realize it.
//
//   3. Q8TiledGEMM — "a Q8 device-GEMM shader." A host-exact reference of the device Q8 GEMM
//      shader (shaders/q8_matmul.comp): the same windowed, per-block combine order, so the
//      shader the device runs has a bit-exact oracle to match (bit-identical to the cpuref
//      decode kernel qdot8scalar, and window-invariant — the proof its input tiling lifts the
//      inDim limit without moving a byte).
//
//   4. AsyncOverlap / OverlapStage — "async-compute pipeline structure." The exact two-queue
//      producer/consumer recurrence (transfer queue stages the next layer's spilled weights
//      while the compute queue works the current one), returning the serial vs double-buffered
//      makespan. Costs are caller-supplied and device-measured; the model fabricates no time,
//      it computes only the STRUCTURE of the overlap.

// ---- sub-buffer tensor allocation -----------------------------------------------

// SubAlloc is one packed sub-range handed out by a SubAllocator: which backing device block it
// lives in, its aligned byte offset within that block, and the requested payload size. On a
// real device the block is a single fvk_malloc (one VkBuffer + VkDeviceMemory) and the
// (Block, Offset) pair binds a descriptor at a sub-range of it; here it is pure arithmetic.
type SubAlloc struct {
	Block  int   // index of the backing device allocation this sub-range lives in
	Offset int64 // aligned byte offset within the block
	Size   int64 // requested size in bytes (the un-padded payload)
}

// SubAllocator packs many tensor buffers into a few large device allocations at aligned
// offsets — the "sub-buffer tensor allocation" refactor for vulkan.go. Today dalloc issues one
// fvk_malloc per tensor; a real device caps the number of allocations
// (maxMemoryAllocationCount, ~4096 on desktop drivers) and charges fixed per-allocation
// overhead, so per-tensor allocation both wastes memory and risks the cap on a deep model. This
// bump suballocator hands out sub-ranges of fixed-size backing blocks, every offset rounded UP
// to `align` (the device's minStorageBufferOffsetAlignment ∨ nonCoherentAtomSize, so both a
// sub-buffer binding and a host-coherent flush are legal), opening a new block only when the
// current one cannot fit the aligned request. A request larger than a block gets its own
// exact-sized dedicated block. Reset() rewinds the cursors for per-forward-pass transient reuse
// WITHOUT freeing the blocks (the packed-sub-range analogue of vulkan.go's freeTransient pool).
// It holds no device handle — only byte arithmetic — so its packing invariants are
// unit-testable on any host.
type SubAllocator struct {
	align      int64
	blockSize  int64
	cur        int     // index of the block currently being filled (advances on overflow; Reset -> 0)
	caps       []int64 // capacity of each backing block (blockSize, or the size of a dedicated oversize block)
	used       []int64 // bump cursor (bytes consumed, including alignment padding) per block
	requested  int64   // sum of requested payload bytes currently outstanding (Reset -> 0)
	peakReq    int64   // high-water of `requested` — the working set the device must hold resident
	reserved   int64   // total device bytes reserved = Σ caps (monotonic; Reset does not free)
	allocs     int     // count of live (non-empty) sub-ranges handed out since the last Reset
	peakAllocs int     // high-water of `allocs` — the per-tensor device-allocation count the packer replaces
}

// NewSubAllocator makes a suballocator that packs into `blockSize`-byte backing blocks with
// every sub-offset aligned to `align`. Both must be ≥ 1 (an align of 1 means unaligned).
func NewSubAllocator(align, blockSize int64) *SubAllocator {
	if align < 1 {
		align = 1
	}
	if blockSize < 1 {
		blockSize = 1
	}
	return &SubAllocator{align: align, blockSize: blockSize}
}

// alignUp rounds x up to the next multiple of a (a ≥ 1).
func alignUp(x, a int64) int64 {
	if a <= 1 {
		return x
	}
	return ((x + a - 1) / a) * a
}

// Alloc hands out a sub-range of `size` bytes at an aligned offset, growing by a new backing
// block only when no current block can fit it. A non-positive size is a no-op zero range (it
// reserves nothing). The returned (Block, Offset) never overlaps a live range and always lies
// within its block's capacity — the invariants the test pins.
func (a *SubAllocator) Alloc(size int64) SubAlloc {
	if size <= 0 {
		return SubAlloc{Block: a.cur, Offset: 0, Size: 0}
	}
	for {
		if a.cur >= len(a.caps) {
			// No block left to try — open one. blockSize normally, or exactly `size` when the
			// request is larger than a block (a dedicated oversize allocation at offset 0).
			cap := a.blockSize
			if size > cap {
				cap = size
			}
			a.caps = append(a.caps, cap)
			a.used = append(a.used, 0)
			a.reserved += cap
		}
		off := alignUp(a.used[a.cur], a.align)
		if off+size <= a.caps[a.cur] {
			a.used[a.cur] = off + size
			a.requested += size
			if a.requested > a.peakReq {
				a.peakReq = a.requested
			}
			a.allocs++
			if a.allocs > a.peakAllocs {
				a.peakAllocs = a.allocs
			}
			return SubAlloc{Block: a.cur, Offset: off, Size: size}
		}
		a.cur++ // this block can't fit the aligned request; try (or open) the next one
	}
}

// Reset rewinds every block's cursor for transient reuse on the next forward pass, WITHOUT
// freeing the backing blocks: the blocks stay reserved (the residency high-water is kept), the
// outstanding-bytes counter returns to zero, and the next Alloc refills from block 0. This is
// the per-pass recycle the transient pool needs without paying re-allocation each step.
func (a *SubAllocator) Reset() {
	a.cur = 0
	for i := range a.used {
		a.used[i] = 0
	}
	a.requested = 0
	a.allocs = 0
}

// Blocks is the number of backing device allocations the packer has opened — the count that
// replaces one-fvk_malloc-per-tensor and must stay far under maxMemoryAllocationCount.
func (a *SubAllocator) Blocks() int { return len(a.caps) }

// Reserved is the total device bytes reserved across all backing blocks (Σ block capacities).
func (a *SubAllocator) Reserved() int64 { return a.reserved }

// Live is the outstanding requested payload bytes since the last Reset.
func (a *SubAllocator) Live() int64 { return a.requested }

// PeakRequested is the high-water of Live — the working set the device must hold resident, the
// honest residency number to compare against a memory budget (independent of block padding).
func (a *SubAllocator) PeakRequested() int64 { return a.peakReq }

// PackingEfficiency is PeakRequested / Reserved in [0,1]: how much of the reserved device
// memory is live payload (the rest is alignment padding + unfilled block tails). 0 when nothing
// has been reserved yet.
func (a *SubAllocator) PackingEfficiency() float64 { return ratio(a.peakReq, a.reserved) }

// Allocations is the number of live (non-empty) sub-ranges handed out since the last Reset — the
// per-pass tensor count packed into the backing blocks.
func (a *SubAllocator) Allocations() int { return a.allocs }

// PeakAllocations is the high-water of Allocations: the largest number of tensors held live in
// one pass, which is exactly how many device allocations the naive one-fvk_malloc-per-tensor path
// would hold at once. This is the count the packer collapses down to Blocks(), and the number
// that must stay under the device's maxMemoryAllocationCount.
func (a *SubAllocator) PeakAllocations() int { return a.peakAllocs }

// FitsAllocCount reports whether the packer's backing-block count — the real number of device
// allocations (one fvk_malloc per block) — stays within maxAllocs, a device's
// maxMemoryAllocationCount (~4096 on desktop drivers; the shim flags this exact cap in
// vulkan_shim.cpp). A non-positive maxAllocs means "no cap" (always true). This is the safety
// property the sub-buffer refactor exists to guarantee: a deep model whose PeakAllocations would
// blow the cap under per-tensor allocation still fits, because Blocks() ≪ PeakAllocations.
func (a *SubAllocator) FitsAllocCount(maxAllocs int) bool {
	return maxAllocs <= 0 || a.Blocks() <= maxAllocs
}

// ---- single command-buffer per forward pass -------------------------------------

// DispatchOp is one recorded compute dispatch in a forward pass: a stable op name and whether
// its result must be read back to the host THIS step (an Argmax/Read — a point the queue must
// be submitted and fenced so the host sees the bytes). It carries no buffers — it is the SHAPE
// of the op stream the command-buffer planner reasons over, not the data.
type DispatchOp struct {
	Name     string
	HostRead bool // result crosses device->host here, forcing a submit+fence at this op
}

// ForwardPlan records the op stream of one forward pass and answers the command-buffer question
// structurally. Today each vulkan.go op takes vulkanMu and (outside an explicit batch) the shim
// can submit+fence per dispatch; the "single command-buffer per forward pass" refactor records
// every dispatch into ONE command buffer and submits ONCE, fencing only where a result actually
// crosses to the host. Submits() returns the batched submission count, NaiveSubmits() the per-op
// baseline, and the gap is the queue-submit + fence-stall reduction the batching buys —
// countable with no device. The BatchRecorder seam below is the HAL method pair the
// vulkanBackend already exposes to realize it.
type ForwardPlan struct {
	ops []DispatchOp
}

// Record appends one dispatch to the plan.
func (p *ForwardPlan) Record(op DispatchOp) { p.ops = append(p.ops, op) }

// RecordDispatch appends an interior dispatch (no host read).
func (p *ForwardPlan) RecordDispatch(name string) { p.Record(DispatchOp{Name: name}) }

// RecordHostRead appends a dispatch whose result is read host-ward (a fence point).
func (p *ForwardPlan) RecordHostRead(name string) { p.Record(DispatchOp{Name: name, HostRead: true}) }

// Ops returns the recorded op stream.
func (p *ForwardPlan) Ops() []DispatchOp { return p.ops }

// Len is the number of recorded dispatches.
func (p *ForwardPlan) Len() int { return len(p.ops) }

// NaiveSubmits is the per-op baseline: one queue submit + host fence per recorded dispatch —
// the worst case the single-command-buffer refactor removes.
func (p *ForwardPlan) NaiveSubmits() int { return len(p.ops) }

// Submits is the batched submission count under one-command-buffer-per-pass recording: the
// command buffer accumulates dispatches and is submitted+fenced only at each host-read point,
// with one final flush if any trailing dispatches were recorded after the last host read. A
// pass with no ops costs 0 submits; a pass that reads the host only once (the common case — a
// final Argmax) costs exactly 1.
func (p *ForwardPlan) Submits() int {
	if len(p.ops) == 0 {
		return 0
	}
	submits := 0
	pending := false
	for _, op := range p.ops {
		pending = true
		if op.HostRead {
			submits++       // must submit+fence so the host can read this result
			pending = false // the command buffer closed at this fence
		}
	}
	if pending {
		submits++ // final flush for the trailing un-read dispatches
	}
	return submits
}

// SubmitReduction is NaiveSubmits − Submits: the number of queue submissions (and their host
// fence stalls) the single-command-buffer recording eliminates for this pass.
func (p *ForwardPlan) SubmitReduction() int { return p.NaiveSubmits() - p.Submits() }

// BatchRecorder is the OPTIONAL capability a backend implements to record a whole forward pass
// into a single command buffer and submit it once — the "single command-buffer per forward
// pass" seam at the HAL. The Vulkan backend (vulkan.go) already exposes BeginBatch/FlushBatch
// (fvk_batch_begin/fvk_batch_flush), so it satisfies this interface under -tags vulkan; on a
// non-Vulkan build no registered backend implements it, RecordForward runs the body eagerly,
// and this file still compiles and runs. Declaring the interface here, in the always-compiled
// file, is what makes the seam type-check with or without a Vulkan device — the same
// guarded-stub discipline prefill.go uses for the CUDA-graph capturer.
type BatchRecorder interface {
	Backend
	BeginBatch()
	FlushBatch()
}

// RecordForward runs body wrapped in one command-buffer batch when be supports it (a
// BatchRecorder), submitting the batch once at the end; otherwise it runs body directly. It
// reports whether the batched path was taken. This is the guarded wiring a forward loop calls
// so the same code path serves a Vulkan device (batched, one submit) and the CPU reference
// (eager) with no build tag at the call site — the BatchRecorder twin of CapturePrefillGraph.
func RecordForward(be Backend, body func()) (batched bool) {
	br, ok := be.(BatchRecorder)
	if !ok {
		body()
		return false
	}
	br.BeginBatch()
	body()
	br.FlushBatch()
	return true
}

// ---- Q8 device-GEMM shader (host-exact tiled reference) -------------------------

// Q8TiledGEMM computes Y[P,out] for the Q8_0 weight (int8 codes [out,in] + per-block f32 scales
// [out, in/32]) against the f32 activation panel X[P,in], in the EXACT windowed, per-block
// combine order of the device Q8 GEMM shader (shaders/q8_matmul.comp). For each row t it
// quantizes X[t,:] per 32-block (amax/127, round-half-away-from-zero — quantizeVecQ8), then for
// each output o accumulates, in block order, Σ_b (the block's s0..s3 4-lane int dot) · dW[o,b] ·
// dX[b], windowing the input in chunks of `winBlocks` whole blocks exactly as the shader stages
// SHARED_CAP floats at a time. Because `acc` already sums per-block contributions in block
// order, windowing changes only WHICH blocks are resident at once, not the combine order — so
// the result is independent of winBlocks (Q8TiledGEMM(...,k) == Q8TiledGEMM(...,full), witnessed
// in the test), which is exactly why the shader's input tiling lifts the inDim≤2048 limit
// without moving a byte. Per output it is bit-exact to the cpuref decode kernel qdot8scalar, so
// the device shader has a host-exact oracle: the Q8 lane's Approx gate (argmax-exact vs the f32
// reference) is witnessable here without a GPU.
//
// winBlocks ≤ 0 (or ≥ the block count) means "single window" — the whole input at once. codes,
// scales and X are the same row-major layout vulkan.go uploads (uploadQ8Locked) and the shader
// binds.
func Q8TiledGEMM(codes []int8, scales []float32, X []float32, out, in, P, winBlocks int) []float32 {
	const block = q8Block // 32, Q8_0
	if in%block != 0 {
		panic("compute: Q8TiledGEMM requires in divisible by 32 (Q8_0 block)")
	}
	nblk := in / block
	if winBlocks <= 0 || winBlocks > nblk {
		winBlocks = nblk
	}
	Y := make([]float32, P*out)
	for t := 0; t < P; t++ {
		qx, dx := quantizeVecQ8(X[t*in:t*in+in], block)
		for o := 0; o < out; o++ {
			wc := codes[o*in : o*in+in]
			ws := scales[o*nblk : o*nblk+nblk]
			var acc float32
			for wb := 0; wb < nblk; wb += winBlocks {
				win := winBlocks
				if wb+win > nblk {
					win = nblk - wb
				}
				for lb := 0; lb < win; lb++ {
					b := wb + lb
					cb := b * block
					// 4 int accumulators in the shader's s0..s3 layout, reduced (s0+s1)+(s2+s3).
					var s0, s1, s2, s3 int32
					for i := 0; i < block; i += 4 {
						s0 += int32(wc[cb+i]) * int32(qx[cb+i])
						s1 += int32(wc[cb+i+1]) * int32(qx[cb+i+1])
						s2 += int32(wc[cb+i+2]) * int32(qx[cb+i+2])
						s3 += int32(wc[cb+i+3]) * int32(qx[cb+i+3])
					}
					acc += float32((s0+s1)+(s2+s3)) * ws[b] * dx[b]
				}
			}
			Y[t*out+o] = acc
		}
	}
	return Y
}

// ---- async-compute pipeline structure -------------------------------------------

// OverlapStage is one pipeline stage's abstract costs: ComputeCost on the compute queue, and
// the TransferCost of host->device staging of THIS stage's spilled (budget-evicted,
// host-visible) weights that must land before its compute can run. Units are caller-supplied
// and device-measured (e.g. derived from the prefill roofline's bytes ÷ measured bandwidth and
// FLOPs ÷ measured throughput) — this file fabricates no throughput; it computes only the
// STRUCTURE of the overlap.
type OverlapStage struct {
	Name         string
	ComputeCost  float64
	TransferCost float64
}

// OverlapResult is the two makespans AsyncOverlap compares plus the achieved speedup.
type OverlapResult struct {
	Serial     float64 // single-queue baseline: every transfer then its compute, summed (no overlap)
	Overlapped float64 // double-buffered: transfer of stage i+1 runs while stage i computes
	Speedup    float64 // Serial / Overlapped (1.0 when nothing overlaps)
}

// AsyncOverlap models the async-compute pipeline the Vulkan backend wants when a residency
// budget spills cold weights host-visible (vulkan.go's dallocWeight / FAK_GPU_BUDGET_MB path):
// those weights must be staged to the device before their layer computes, and that staging can
// run on a dedicated TRANSFER queue concurrently with the COMPUTE queue working the previous
// layer (double buffering). It returns the serial makespan (transfer then compute, summed — the
// single-queue baseline) and the overlapped makespan from the exact two-queue producer/consumer
// recurrence: transfers run in order on the transfer queue, computes run in order on the compute
// queue, and compute_i waits for BOTH transfer_i and compute_{i-1}. The overlap hides every
// interior transfer behind a compute, so the win is the structural payoff of async compute —
// bounded entirely by the caller's device-measured costs, never a fabricated time. The
// overlapped makespan is always ≥ max(Σ transfers, Σ computes) and ≥ firstTransfer + lastCompute
// (the unavoidable pipeline head and tail), and ≤ the serial baseline.
func AsyncOverlap(stages []OverlapStage) OverlapResult {
	var serial float64
	for _, s := range stages {
		serial += s.TransferCost + s.ComputeCost
	}
	var transferFinish, computeFinish float64
	for _, s := range stages {
		transferFinish += s.TransferCost // transfer queue runs the stages in order
		start := transferFinish          // compute_i needs its own transfer done...
		if computeFinish > start {
			start = computeFinish // ...and the previous compute done (compute queue is serial)
		}
		computeFinish = start + s.ComputeCost
	}
	res := OverlapResult{Serial: serial, Overlapped: computeFinish, Speedup: 1}
	if computeFinish > 0 {
		res.Speedup = serial / computeFinish
	}
	return res
}
