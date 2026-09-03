//go:build cuda

package compute

import (
	"os"
	"sync/atomic"
	"unsafe"
)

// q8DeviceBlock is the Q8_0 per-block size the device narrow-at-H2D quant uses (llama.cpp block_q8_0
// = 32, the cpuref default). The resident weight carries it in QuantSpec.Block so the GEMM kernel
// reconstructs nblk = in/block; `in` must be divisible by it.
const q8DeviceBlock = 32

// cudaBudgetBytes resolves FAK_GPU_BUDGET_MB — the device-local weight budget in MiB — against this
// device's total VRAM. 0 / unset / invalid = unbounded (place every explicit weight allocation with
// cudaMalloc, the prior behavior); a positive value caps CUDA device-local weight residency; "auto"
// derives the cap from totalDeviceLocal (see resolveGPUBudgetBytes), failing open to unbounded when
// capacity is unknown. Weights past the cap go into cudaMallocManaged so the driver can page them on
// demand instead of losing the allocation race and hard-panicking.
func cudaBudgetBytes(totalDeviceLocal int64) int64 {
	return resolveGPUBudgetBytes(os.Getenv("FAK_GPU_BUDGET_MB"), totalDeviceLocal, totalDeviceLocal > 0)
}

var cudaDev *cudaBackend

// cudaBuf is a device-resident Buffer: a VRAM pointer + byte length. Op OUTPUTS (allocated
// via devTr) are ASYNC under #482 — enqueued on g_stream and NOT host-observable until a host
// fence (Read/Argmax) drains the stream — so each records the backend's fence generation at
// enqueue time and Ready() reports whether a later fence has bumped past it. Buffers that are
// synchronous on return (weights, whose Upload H2D is a blocking cudaMemcpy; KV views; the
// argmax scalar) carry be==nil and are always Ready.
type cudaBuf struct {
	ptr      unsafe.Pointer // device pointer (cudaMalloc); int8 codes for Q8_0, raw bytes for Q4_K
	n        int            // bytes at ptr
	class    MemoryClass    // allocation purpose retained for strict mutable-state validation
	device   int            // CUDA device this buffer is resident on (0 for the single-device path; set by the NCCL collective seam so a multi-GPU all-reduce knows each rank's home — #971)
	host     uintptr        // source host pointer if this came from a cached Upload (0 otherwise)
	hostKeep HostBuffer     // strong owner: prevents Go from recycling host while pointer-keyed cache entry is live
	hostDt   Dtype          // narrowed dtype this upload was cached under (so Free evicts the right key)
	hostLo   Layout         // layout this upload was cached under (ditto — same host buffer, two layouts)
	be       *cudaBackend   // non-nil => async op output; Ready() tracks be.fenceGen vs bornGen
	bornGen  uint64         // fence generation in which this async buffer was enqueued
	managed  bool           // ptr came from cudaMallocManaged, not pooled cudaMalloc
	// invalid is set when an in-place Qwen GDN operation reports a launch or
	// asynchronous execution failure. Such a buffer may be freed, but never read
	// or submitted again as a usable state/output.
	invalid uint32
	// budgetedWeightBytes/managedWeight account only explicit resident WEIGHT buffers (F16/Q8/Q4K
	// uploads). Generic F32 Upload is also used for per-token inputs, so it stays outside this budget.
	budgetedWeightBytes int64
	managedWeight       bool
	// scales is the SECOND VRAM buffer a resident Q8_0 weight carries (#485): the per-block(32)
	// f32 scales living beside the int8 codes in ptr. Q4_K keeps d/dmin/scales/codes packed in the
	// raw super-block bytes at ptr, so it leaves scales==nil. Freed alongside ptr in Free.
	scales  unsafe.Pointer
	scalesN int // bytes at scales (0 when there is no scale side-channel)
}

// residentBytes is the total VRAM the weight occupies — codes (ptr) plus any scale side-channel.
// The #485 VRAM witness reads it to prove a Q8_0/Q4_K weight stays narrow (≈ int8/int4 size, not
// the f32 size a dequant-to-f32 upload would have paid).
func (b *cudaBuf) residentBytes() int { return b.n + b.scalesN }

// invalidForSubmit is the single narrow validity check shared by every CUDA
// pointer consumer. It intentionally checks only the post-execution poison bit:
// callers retain their existing type, nil, residency, readiness, and panic/error
// contracts while gaining one invariant — a failed in-place GDN buffer can never
// cross the C ABI again.
func (b *cudaBuf) invalidForSubmit() bool {
	return b != nil && atomic.LoadUint32(&b.invalid) != 0
}

func (b *cudaBuf) invalidStateError(operand string) error {
	if !b.invalidForSubmit() {
		return nil
	}
	return &Qwen35GDNInvalidStateError{Operand: operand}
}

// cudaBufForSubmit preserves the historical type-assertion panic for non-CUDA
// tensors, but converts the poison bit into the typed invalid-state panic before
// any device pointer is submitted.
func (c *cudaBackend) cudaBufForSubmit(t Tensor) *cudaBuf {
	b := t.buf.(*cudaBuf)
	if err := b.invalidStateError("buffer"); err != nil {
		panic(err)
	}
	return b
}

// Ready reports whether the buffer's producing kernel has been fenced host-ward. An async op
// output is ready once a Read/Argmax has bumped the fence generation past the one it was
// enqueued in: the single g_stream is FIFO and a host fence drains all prior work, so one
// generation bump materializes every buffer enqueued before it. Synchronous buffers (be==nil)
// are ready on return. This is the bit the model loop reads to know the logits are still
// device-resident mid-step (#482) — it never gates device execution, which is stream-ordered.
func (b *cudaBuf) Ready() bool {
	if b == nil || b.invalidForSubmit() {
		return false
	}
	if b.be == nil {
		return true
	}
	return atomic.LoadUint64(&b.be.fenceGen) > b.bornGen
}

// uploadCache shares one VRAM copy per distinct host weight buffer across all sessions. Each cached
// cudaBuf retains its HostBuffer owner: the uintptr key alone is not a GC root, and allowing the
// source slice to die lets Go recycle its address for a different same-shaped weight, which would
// false-hit stale VRAM. A model's
// weights are zero-copy views into one blob (m.tensor(name) returns the SAME pointer every
// call), so without this each NewBackendSession re-uploaded the whole model — N sessions ×
// the full weight set, which exhausts VRAM in a multi-session bench. Only rank >= 2 tensors enter
// this pointer-keyed cache: rank-1 uploads are activations, norms, or biases and must copy their
// current bytes. Go may recycle a dead rank-1 slice at the same address for the next token; caching
// it would return the prior token's VRAM buffer and make decode deterministically degenerate.
//
// The key is (host pointer, narrowed dtype, layout), NOT the pointer alone: under #484 the SAME
// host weight may be uploaded as F32 and as F16, or as F16 in two layouts (RowMajor vs the
// ColMajor transpose-repack), and those are DISTINCT resident buffers. Keying on the pointer
// alone would alias them and hand back the wrong layout/dtype.
type ucKey struct {
	hp uintptr
	dt Dtype
	lo Layout
}

var uploadCache = map[ucKey]Tensor{}

type cudaBackend struct {
	name string
	tier string
	// totalMem is the device's total VRAM in bytes (totalGlobalMem from fcuda_init), KEPT so
	// the backend can satisfy DeviceCapacity — fak's one programmatic "does this fit on this
	// device?" number, which init() previously read into a local and threw away.
	totalMem int64
	// fenceGen counts host fences (Read/Argmax — the ONLY two). Each async op output records
	// the generation it was enqueued in (cudaBuf.bornGen); a fence bumps fenceGen, flipping
	// every buffer enqueued before it to Ready (#482). Read/written atomically: producers hold
	// cudaMu but Ready() readers (the model loop / the witness test) do not take the lock.
	fenceGen uint64
	// transient holds per-token op-output buffers (NOT weights or KV). Recycle() returns
	// them all to the C-side pool at a token boundary so steady-state decode stops paying
	// cudaMalloc per op. Guarded by cudaMu (every appender holds it).
	transient []*cudaBuf
	// Device-local residency budget (Stage-1 offload parity with Vulkan). budgetBytes caps
	// cudaMalloc-backed resident WEIGHT bytes; 0 = unbounded. dlUsed tracks bytes placed with
	// cudaMalloc while under the cap. When the next explicit weight would exceed the cap, it is
	// deliberately placed in managed memory (cudaMallocManaged) in upload order, so early/hot
	// layers stay device-local and the cold tail spills by choice instead of OOM. Guarded by
	// cudaMu (mutated only inside locked upload/free paths).
	budgetBytes int64
	dlUsed      int64
	managedN    int
	// immutableWeightUpload* are cumulative backend-authored counters for successful
	// immutable-weight H2D misses. They deliberately do not describe activations,
	// cached hits, refused calls, or current live residency. Guarded by cudaMu.
	immutableWeightUploadCalls         uint64
	immutableWeightUploadTransferBytes uint64
	immutableWeightUploadResidentBytes uint64
	// faultLatch is the session-scoped device-fault boundary (#6412). This backend owns the
	// process's single CUDA context (one g_stream, one cudaMu), so backend scope IS session
	// scope: once a launch/execution/allocation fault is observed here, gated entry points
	// refuse typed instead of computing on a suspect context. nil admits everything, so a
	// backend constructed without a latch (older tests) behaves exactly as before.
	faultLatch *DeviceFaultLatch
	// capturing tracks whether a CUDA graph capture is currently open on g_stream (#10716).
	// Under capture, parameter and constant memory uploads must be emitted unconditionally
	// into the stream so replayed graph executions do not read stale host-cached state.
	capturing bool
}

// cudaFaultReconstructBudget bounds how many context reconstructions a poisoned session may
// attempt before the latch declares it unrecoverable. Nothing in-package drives Reconstruct
// yet — the serving boundary owns teardown/rebuild — but the budget must be fixed at
// construction so a future driver cannot retry forever against a dead device.
const cudaFaultReconstructBudget = 3

// DeviceFaultLatch satisfies DeviceFaultReporter: it exposes the session latch so a serving
// path holding only a compute.Backend can gate on AdmitDevice and drive recovery.
func (c *cudaBackend) DeviceFaultLatch() *DeviceFaultLatch {
	return c.faultLatch
}

func (c *cudaBackend) CUDADebugResidencyBudget() (budgetBytes, dlUsed int64, managedN int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.budgetBytes, c.dlUsed, c.managedN
}

func (c *cudaBackend) CUDADebugSetResidencyBudget(budgetBytes int64) (oldBudgetBytes, oldDLUsed int64, oldManagedN int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	oldBudgetBytes, oldDLUsed, oldManagedN = c.budgetBytes, c.dlUsed, c.managedN
	c.budgetBytes, c.dlUsed, c.managedN = budgetBytes, 0, 0
	return oldBudgetBytes, oldDLUsed, oldManagedN
}

func (c *cudaBackend) CUDADebugRestoreResidencyBudget(budgetBytes, dlUsed int64, managedN int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	c.budgetBytes, c.dlUsed, c.managedN = budgetBytes, dlUsed, managedN
}

// CUDAImmutableWeightUploadSnapshot returns cumulative successful immutable-weight
// H2D operations. Matrix upload-cache hits are suppressed; deliberately uncached
// rank-one weight copies are counted individually. The concrete optional method keeps
// CUDA observability out of the common Backend interface; callers that need it
// type-assert this exact method.
// Subtracting snapshots describes a serialized campaign window; overlapping
// coalesced requests share this backend and cannot claim request-exclusive deltas.
func (c *cudaBackend) CUDAImmutableWeightUploadSnapshot() (calls, transferBytes, residentBytes uint64) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return c.immutableWeightUploadCalls, c.immutableWeightUploadTransferBytes, c.immutableWeightUploadResidentBytes
}

// accountImmutableWeightUpload runs only after the last H2D copy for one weight
// upload succeeds. Runtime/activation UploadClass calls return before these sites;
// immutable rank-one norms and biases still count even though pointer caching is
// deliberately restricted to matrices. Caller holds cudaMu.
func (c *cudaBackend) accountImmutableWeightUpload(transferBytes int, buf *cudaBuf) {
	if transferBytes <= 0 || buf == nil || buf.ptr == nil {
		return
	}
	c.immutableWeightUploadCalls++
	c.immutableWeightUploadTransferBytes += uint64(transferBytes)
	c.immutableWeightUploadResidentBytes += uint64(buf.residentBytes())
}

// Name returns the registry id of this backend ("cuda").
func (c *cudaBackend) Name() string { return c.name }

// SupportsRoutedExpertKQuant advertises native resident Q4_K MatMul plus F16 staging.
func (c *cudaBackend) SupportsRoutedExpertKQuant() bool { return true }
func (c *cudaBackend) Tier() string                     { return c.tier }
func (c *cudaBackend) Class() CorrectnessClass          { return Approx } // device GEMM != fdot order
func (c *cudaBackend) Caps() Caps {
	// Async (#482): ops enqueue on g_stream and return unready Buffers; the SOLE host fences
	// are Read and Argmax. DeviceMemory: resident tensors (incl. the KV cache) are not host-
	// addressable. GraphCompile (#483): the fixed per-token decode op stream is capturable into
	// a cudaGraph_t on g_stream and replayable as ONE cudaGraphLaunch (instead of N kernel
	// launches). It is advertised true exactly when that path is live (graphEnabled /
	// FAK_CUDA_GRAPH=1) so it stays consistent with GraphBegin's consent — a consumer that reads
	// false cleanly falls back to the synchronous per-op core (the cpu-ref/Metal default).
	// UploadDtype (#484/#485): Upload(t, F16) narrows weights to __half at H2D (with a ColMajor
	// transpose-repack) for tensor-core HGEMM; Upload(t, Q8_0) narrows an f32 weight to resident
	// int8 codes + f32 scales, and a Q4_K host weight uploads its raw super-block bytes verbatim —
	// MatMul/BatchedMatMul then run the native quantized device GEMMs (no dequant-to-f32), keeping
	// the weight narrow in VRAM. FusedAttn (#486): Attention lowers to ONE fused flash/online-softmax
	// kernel (k_flash_attention) — tiled over the KV window with a running max/sum so no scores[nPos]
	// row is materialized; the naive kernel is retained only as the microbench baseline.
	// CapacityProbe (capacity.go): the backend can REPORT its VRAM ceiling (DeviceMemory),
	// the report half of the hardware-capacity bridge. It is the one number this backend has
	// always held (totalGlobalMem) but used to discard.
	_, _, hostKnown := hostSystemMemory()
	// Collective is advertised true ONLY once a real NCCL communicator is up (fcuda_nccl_init
	// succeeded over >1 device, recorded in cudaNCCLWorld). Until then it stays false so a host
	// never picks the device collective path before it can actually all-reduce across GPUs — the
	// honesty line (#971): no multi-GPU claim until a device tensor reduces across 2 GPUs.
	return Caps{Async: true, DeviceMemory: true, GraphCompile: graphEnabled, UploadDtype: true, FusedAttn: true, CapacityProbe: true, HostCapacityProbe: hostKnown, Collective: atomic.LoadInt32(&cudaNCCLWorld) > 1}
}

func (c *cudaBackend) HostMemory() (total, free int64, known bool) {
	return hostSystemMemory()
}

func (c *cudaBackend) accountWeightPlacement(buf *cudaBuf, nbytes int) {
	if c.budgetBytes == 0 || buf == nil || buf.ptr == nil {
		return
	}
	if buf.managed {
		c.managedN++
		buf.managedWeight = true
		return
	}
	c.dlUsed += int64(nbytes)
	buf.budgetedWeightBytes = int64(nbytes)
}
