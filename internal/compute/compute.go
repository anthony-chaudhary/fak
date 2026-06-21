// Package compute is the hardware-abstraction seam (HAL) for the in-kernel forward
// pass. Today the model in internal/model calls matRows / parMatRows / qMatRows /
// qGemm8 / rmsnorm / applyRopeRow / softmaxInPlace / silu / argmaxF32 as concrete
// package-level functions, over bare []float32, gated for non-x86 by a //go:build
// fork and for Q8 by a `bool`. That bakes seven hardware assumptions into the math
// itself (see HARDWARE-PORTABILITY in docs/explainers): float32-only, host-pointer
// aliasing, x86 build-tag dispatch, synchronous return-by-value, goroutine-only
// parallelism, row-major layout, eager full-RAM weight residency.
//
// This package lifts all seven *in the type system* so that adding a GPU / XPU / NPU
// / dataflow / WASM backend is a new Backend registration, never an edit to the
// forward loop. Day-1 only one backend exists — a pure-Go, scalar, allocation-by-
// value CPU *reference* (cpuref.go) whose kernels reproduce the model's exact
// reduction order, so when the model adopts the seam its bit-identity rungs (R2
// max|Δ|=0, R14 d==0, the HF argmax oracle) survive byte-for-byte. The reference is
// deliberately stdlib-only and free of unsafe / asm / cgo / os.Getenv, so it is also
// the portable floor every other target degrades to (it compiles to wasm unchanged).
//
// What is NEUTRALIZED in the contract (even with only CPU implemented):
//   - dtype monoculture  -> Dtype is a first-class enum on every Tensor + QuantSpec;
//     a weight's Dtype selects the kernel, collapsing the f32/Q8 "forward pass exists
//     twice" duplication into one dispatch. fp8 / MX / int4 / asymmetric-NPU schemes
//     are new Dtype+QuantSpec values, not a third clone.
//   - host-pointer aliasing -> a Tensor holds NO host pointer; host addressability is
//     reachable only by type-asserting its Buffer to HostBuffer (implemented solely by
//     the CPU backend) or via Backend.Host(t)->(_,false on a device). A device tensor
//     cannot be silently reinterpreted as a host slice.
//   - x86 build-tag dispatch -> Register/Pick is a runtime registry; Tier() is each
//     backend's PRIVATE capability probe (CPUID on x86, driver query on a GPU). Build
//     tags gate which backends COMPILE IN; the registry picks which one RUNS.
//   - synchronous return-by-value -> Buffer.Ready() + Caps.Async let a device backend
//     enqueue and return an unready Buffer, fencing only inside Read/Argmax. Argmax is
//     a first-class scalar-reduction op so greedy decode never copies the full logits
//     vector host-ward every step.
//   - goroutine-only parallelism -> the interface exposes WHOLE ops (MatMul, Attention),
//     never "split these rows across workers", so a device is free to express its own
//     intra-kernel parallelism. The reference's fork-join stays private to the backend.
//   - row-major-only -> Layout is a descriptor on every Tensor; a tensor-core backend
//     declares Tiled/ColMajor and repacks at Upload without the loop seeing it.
//   - eager full-RAM residency -> WeightSource lets a backend stream/stage weights
//     instead of os.ReadFile-ing the whole blob; Upload(t, as) narrows dtype at H2D.
//
// What is NOT yet lifted is tracked honestly in the known-open ledger in the doc, not
// hidden: each open assumption is tagged with the named seam that will close it.
package compute

// ---- Dtype ----------------------------------------------------------------------

// Dtype is the storage/compute element format of a Tensor — a first-class value, so the
// model's tensorMeta.Dtype string (parsed then discarded at weights.go:50 today) becomes
// real dispatch instead of an implicit `float32`. A weight's Dtype selects the matmul
// kernel; that is what collapses the f32/Q8 twin into one path.
type Dtype uint8

const (
	F32  Dtype = iota // IEEE binary32 — the reference currency
	F16               // IEEE binary16 (device-native; widened at load today)
	BF16              // bfloat16
	Q8_0              // 8-bit, symmetric, per-block(32) scale — llama.cpp's block_q8_0
	I8                // generic int8 (scheme in QuantSpec)
	I4                // generic int4 / nibble-packed (scheme in QuantSpec)
	FP8               // 8-bit float (E4M3/E5M2; variant in QuantSpec)
)

// Bytes is the per-element storage width. For sub-byte formats (I4) it reports the
// rounded-up byte cost of a single code; real packing is described by QuantSpec.
func (d Dtype) Bytes() int {
	switch d {
	case F32:
		return 4
	case F16, BF16:
		return 2
	default:
		return 1
	}
}

func (d Dtype) String() string {
	switch d {
	case F32:
		return "f32"
	case F16:
		return "f16"
	case BF16:
		return "bf16"
	case Q8_0:
		return "q8_0"
	case I8:
		return "i8"
	case I4:
		return "i4"
	case FP8:
		return "fp8"
	default:
		return "dtype?"
	}
}

// Quantized reports whether the dtype needs a QuantSpec to be interpreted.
func (d Dtype) Quantized() bool { return d == Q8_0 || d == I8 || d == I4 || d == FP8 }

// ---- Layout ---------------------------------------------------------------------

// Layout is a physical-arrangement descriptor carried on every Tensor. The CPU
// reference honors only RowMajor; the field exists so a cuBLAS/tensor-core backend can
// declare ColMajor or a vendor Tiled format and repack at Upload time, without the
// forward loop ever indexing the bytes itself.
type Layout uint8

const (
	RowMajor Layout = iota
	ColMajor
	Tiled // backend-private blocked layout (shape semantics unchanged)
)

// ---- QuantSpec ------------------------------------------------------------------

// QuantSpec generalizes the hardcoded q8Tensor{d, qBlk==32}: it describes the
// quantization scheme richly enough for the formats accelerators actually use —
// asymmetric (zero-point), per-channel or per-block, sub-byte packing, calibrated
// static activation scales — so a new scheme is data, not a new code path. The CPU
// reference uses {Block:32, Bits:8, Symmetric:true} (exactly Q8_0).
type QuantSpec struct {
	Block     int       // group size sharing one scale (32 for Q8_0); 0 = per-tensor
	Axis      uint8     // 0=per-tensor 1=per-channel 2=per-block
	Bits      uint8     // 4 | 8
	Symmetric bool      // true => ZeroPoint unused
	Scale     []float32 // per-group scales (the q8Tensor.d analogue)
	ZeroPoint []int32   // per-group zero points; nil when Symmetric
	// StaticAct marks that activation scales are calibrated offline (an edge-NPU need)
	// rather than recomputed per token; the reference ignores it (always dynamic).
	StaticAct bool
}

// ---- Tensor / Buffer ------------------------------------------------------------

// Buffer is opaque, backend-owned storage: a host slice on the CPU reference, a VRAM
// allocation id on a GPU, a future on an async backend. The forward loop NEVER
// dereferences it — it touches data only through Backend methods. Ready() is true once
// the value is materialized (always true on a synchronous backend; false on an async
// device until its producing kernel has been fenced).
type Buffer interface {
	Ready() bool
}

// HostBuffer is the host-addressable Buffer, implemented ONLY by the CPU reference
// backend. Because the accessor lives here and not on Tensor, a device tensor (whose
// Buffer is not a HostBuffer) cannot be reinterpreted as a host slice by construction —
// this is the type-level fix for the unsafe.Slice host-pointer aliasing assumption. It
// deliberately exposes no unsafe.Pointer, so the contract stays wasm-clean.
type HostBuffer interface {
	Buffer
	F32() []float32 // valid when the owning Tensor's Dtype is a float format
	I8() []int8     // valid when the owning Tensor's Dtype is an int/quant format
}

// Tensor is the value type that replaces a bare []float32 at the seam. It carries its
// dtype/layout/shape and (for quantized dtypes) its QuantSpec, but holds NO host
// pointer — only an opaque Buffer and a reference to the backend that owns it. Two
// tensors are interchangeable across the loop regardless of where their bytes live.
type Tensor struct {
	Dtype  Dtype
	Layout Layout
	Shape  []int
	Quant  *QuantSpec // non-nil iff Dtype.Quantized()
	buf    Buffer
	be     Backend
}

// Buf exposes the opaque storage handle (e.g. so a backend can recover its own
// HostBuffer). It is intentionally the only door to the bytes, and only the owning
// backend knows how to open it.
func (t Tensor) Buf() Buffer      { return t.buf }
func (t Tensor) Backend() Backend { return t.be }
func (t Tensor) Ready() bool {
	if t.buf == nil {
		return false
	}
	return t.buf.Ready()
}

// Numel is the product of the shape (0-dim tensor => 1).
func (t Tensor) Numel() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// makeTensor is the backend-internal constructor (backends in this package use it; the
// public entry for host data is Backend.Upload / the cpuref host constructors).
func makeTensor(be Backend, dt Dtype, layout Layout, shape []int, q *QuantSpec, buf Buffer) Tensor {
	return Tensor{Dtype: dt, Layout: layout, Shape: shape, Quant: q, buf: buf, be: be}
}

// ---- Correctness class ----------------------------------------------------------

// CorrectnessClass is typed so the bit-identity contract cannot silently rot. Only a
// Reference backend may be held to the exact rungs (max|Δ|=0 R2/R14 and the HF argmax
// oracle); every Approx backend (the Q8 lane, and every future device) is held to the
// looser argmax-exact + logit-cosine gate, with a per-backend cosine threshold. The
// test harness asserts the class (see RequireReference), so it is mechanically
// impossible to expect bit-identity of a device, or to promote a device to reference.
type CorrectnessClass uint8

const (
	Reference CorrectnessClass = iota
	Approx
)

func (c CorrectnessClass) String() string {
	if c == Reference {
		return "reference"
	}
	return "approx"
}

// ---- Capabilities ---------------------------------------------------------------

// Caps advertises OPTIONAL capabilities a backend supports, so the core interface can
// assume none of them. A backend that fuses attention, compiles a whole graph (the
// seam dataflow/wafer chips need), or runs async opts in here; the forward loop
// type-asserts for a capability and falls back to the synchronous core when absent —
// which makes every backend combination correct-by-construction.
type Caps struct {
	Async        bool // methods enqueue and return unready Buffers; sole host fence is Read/Argmax
	FusedAttn    bool // can lower Attention to one flash/paged-attention kernel
	FusedFFN     bool // can fuse RMSNorm->gate/up->SwiGLU->down (an edge-NPU coarse menu)
	GraphCompile bool // can consume a recorded op-list as a static placeable graph
	UploadDtype  bool // Upload honors `as` (narrows weights at H2D); CPU reference ignores it
	DeviceMemory bool // resident tensors are NOT host-addressable (Host returns false)
	Collective   bool // implements CollectiveBackend (AllReduceSum/AllGather/ReduceScatter over device tensors — the tensor-parallel cross-rank seam)
}

// ---- KV store -------------------------------------------------------------------

// KVConfig is the minimal cache geometry the KVStore needs (mirrors the fields of
// model.Config the cache uses), passed explicitly so the backend holds no model state.
type KVConfig struct {
	NumLayers  int
	NumKVHeads int
	HeadDim    int
	RopeTheta  float64
}

// KVStore is the interface the kernel-owned attention cache lives behind — the one
// device-resident-state seam, shipped from day 1 rather than deferred (it is a GPU/NPU
// backend's single most important seam: device KV must not be a raw [][]float32). The
// CPU reference impl wraps the existing flat row-major slices verbatim, so Evict's
// single-rotation re-RoPE and Clone stay bit-exact and the kvmmu quarantine witness
// (evict == never-saw, max|Δ|=0) is untouched. KeysView/ValuesView hand back a Tensor
// (a host view on CPU, a device handle elsewhere) so attention never reaches into the
// raw storage.
type KVStore interface {
	// AppendKV records one position: kRaw is pre-RoPE K (kept so Evict can reposition a
	// survivor in a single rotation), kRoPE is post-RoPE K, v is the value row.
	AppendKV(layer int, kRaw, kRoPE, v Tensor, pos int)
	Len() int
	KeysView(layer int) Tensor   // flat [pos, nKV*hd] post-RoPE keys
	ValuesView(layer int) Tensor // flat [pos, nKV*hd] values
	Pos() []int                  // absolute position of each cached entry
	// Evict removes [from, from+n) from every layer and compacts survivors so the cache
	// is byte-for-byte what it would be had the span never been seen (the KV quarantine
	// primitive). Returns positions removed.
	Evict(from, n int) int
	Clone() KVStore // deep copy for prefix reuse (the vDSO payoff)
}

// ---- WeightSource ---------------------------------------------------------------

// WeightSource is how a backend obtains a weight by name, so weights need not be a
// single host-resident f32 blob. The CPU reference implements it as today's behavior
// (a view into the slurped blob); a GPU/NPU/wasm backend implements it as a stream or a
// pre-staged device-native buffer — lifting the eager-full-RAM-residency assumption at
// the type level. `want` lets the backend request a narrower dtype at materialization.
type WeightSource interface {
	Weight(name string, want Dtype) (Tensor, error)
}

// ---- Backend --------------------------------------------------------------------

// Backend is the small whole-op interface the forward loop targets. Shapes are implicit
// in the Tensors; attention attributes (causal/grp/scale) are PARAMETERS, never host
// loop bounds, so a vendor attention primitive consumes them directly. Every method on
// the CPU reference delegates to the model's exact arithmetic (see cpuref.go), so
// routing the loop through this interface adds only a method indirection, not a numeric
// change.
type Backend interface {
	Name() string            // stable id, e.g. "cpu-ref"
	Tier() string            // private capability probe result, e.g. "scalar","avx512","sm90"
	Class() CorrectnessClass // Reference or Approx (harness-enforced)
	Caps() Caps              // optional capabilities advertised

	// residency / dtype. CPU reference: Upload/Read are identity over the host slice.
	Upload(t Tensor, as Dtype) Tensor // host -> resident, optionally narrowing dtype
	Host(t Tensor) ([]float32, bool)  // host-addressable f32 view, or (nil,false) on a device
	Read(t Tensor) []float32          // resident -> host f32 (the only host fence on an async backend)
	Free(t Tensor)                    // release device storage (no-op on CPU)
	NewKV(cfg KVConfig) KVStore

	// primitives — dtype dispatched on the WEIGHT tensor where applicable.
	MatMul(w, x Tensor) Tensor               // y = x @ Wᵀ ; w [out,in], x [in] -> y [out]
	BatchedMatMul(w, X Tensor, P int) Tensor // prefill GEMM ; X [P,in] -> Y [P,out]
	RMSNorm(x, weight Tensor, eps float32) Tensor
	RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor // rotate each head; returns a new tensor
	SwiGLU(gate, up Tensor) Tensor                                 // silu(gate)*up
	AddInPlace(dst, src Tensor)                                    // residual: dst += src
	AddBias(dst, bias Tensor)                                      // dst += bias
	// Attention is one fused op: scores = (q·k)*scale over the causal window in kv,
	// softmax, ΣwV. causal/grp/scale are attributes a device lowers to flash/paged attn.
	Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor
	// Argmax is the scalar reduction so greedy decode never pulls the full logits host-ward.
	Argmax(logits Tensor) int
}

// ---- CollectiveBackend (the tensor-parallel cross-rank seam) ---------------------
//
// CollectiveBackend is the OPTIONAL cross-rank reduction interface a backend implements to
// participate in tensor-parallel serving — the device-tensor counterpart of the model
// package's Collective (which reduces host []float32). It is discovered the way the doc
// describes every optional capability: the forward loop type-asserts the Backend for it
// (and a host may cheaply pre-check Caps().Collective), falling back to single-device when
// it is absent. Adding NCCL/RDMA collectives is then a backend that implements these two
// methods — never an edit to the forward loop.
//
// Why this lives at the HAL and not only in model.Collective: a real multi-GPU all-reduce
// runs over DEVICE-resident tensors on one communicator, so the parts a collective combines
// carry their owning Backend. That lets the reference reject a cross-backend reduction (a
// CUDA tensor cannot be reduced against a host tensor) — a fail-closed contract the host
// []float32 seam structurally cannot express. The CPU reference (cpu-ref) implements this as
// the single-box, exact default: AllReduceSum adds in rank order (bit-identical to the model
// package's sumPartialsRankOrder), AllGather concatenates in rank order, and ReduceScatter —
// the dual of AllGather — scatters that same rank-order sum into equal per-rank shards, so by
// construction AllReduceSum ≡ AllGather∘ReduceScatter. Because the parts and the reduction
// order are FIXED, a real collective swapped in behind this interface is correct only if it
// reproduces these bytes — the same invariant the model-side gate pins.
//
// The single-rank case (len(parts)==1) is the IDENTITY for both methods: it returns the lone
// part's data unchanged. This is the HAL twin of model.ForwardTP(ranks=1)==Forward — the
// "bit-exact vs the single-device path" rung, witnessable without any multi-GPU hardware.
type CollectiveBackend interface {
	Backend
	// AllReduceSum returns a new tensor that is the element-wise sum of the equal-length
	// per-rank partial tensors, added in rank order (parts[0] then += parts[r] for r=1..).
	// The fixed order makes the result deterministic and the row-parallel gate exact. It
	// fails closed on no parts, ragged partials, a non-F32 part, an unready part, or a part
	// owned by a different backend (the cross-backend reduction a real communicator rejects).
	AllReduceSum(parts []Tensor) (Tensor, error)
	// AllGather returns a new tensor that is the rank-ordered concatenation of the per-rank
	// shards (parts[0]‖parts[1]‖…). Shard lengths may differ (that is the point — gather
	// uneven bands). Same fail-closed contract as AllReduceSum, minus the equal-length rule.
	AllGather(parts []Tensor) (Tensor, error)
	// ReduceScatter performs the rank-order AllReduceSum of the equal-length partials and then
	// SCATTERS the result into len(parts) equal contiguous shards, returning shard r (the r-th
	// 1/P band of the reduced vector) to rank r. It is the dual of AllGather and the third
	// canonical Megatron collective: AllReduceSum ≡ AllGather∘ReduceScatter, the identity the
	// reference pins. Sequence-parallel TP uses it in place of the post-block AllReduce so each
	// rank retains only its 1/P slice of the activation — the activation-memory lever that lets
	// long-context tensor parallelism scale near-linearly instead of replicating the full
	// activation on every rank. The reduced length must divide evenly by the rank count (real
	// NCCL requires sendcount % nranks == 0); a ragged or indivisible input fails closed. The
	// same fail-closed contract as AllReduceSum otherwise, and single rank is the identity (one
	// shard equal to the lone part).
	ReduceScatter(parts []Tensor) ([]Tensor, error)
	// AllToAll redistributes the P = len(parts) equal-length per-rank vectors by the block
	// TRANSPOSE the layout-changing collectives need: each rank's length-N vector is read as P
	// contiguous shards of size N/P, and output rank r receives, in rank order, shard r from
	// EVERY input rank (out[r] = parts[0]'s shard r ‖ parts[1]'s shard r ‖ …). It is the fourth
	// canonical Megatron collective and the one ReduceScatter/AllGather cannot express: it moves
	// a DIFFERENT shard to each peer instead of reducing or concatenating one — the primitive that
	// turns a sequence-sharded activation into a head-sharded one (and an MoE expert dispatch into
	// its combine). Two structural identities make it self-checking, the AllToAll analogue of
	// AllReduceSum ≡ AllGather∘ReduceScatter: it is an INVOLUTION (AllToAll∘AllToAll == identity,
	// the transpose of a transpose), and ReduceScatter is recoverable as an AllToAll followed by a
	// local per-rank elementwise reduce — both byte-for-byte. The per-rank length must divide by
	// the rank count (real NCCL AllToAll requires sendcount % nranks == 0); the same fail-closed
	// contract as ReduceScatter otherwise, and single rank is the identity.
	AllToAll(parts []Tensor) ([]Tensor, error)
}

// ---- Registry -------------------------------------------------------------------
//
// The registry replaces the //go:build amd64 fork as the dispatch mechanism: backends
// self-register in init(); the host picks one by name. The package deliberately does
// NOT read os.Getenv itself (it is empty on wasm and the package must stay portable) —
// the host resolves the name (from FAK_BACKEND on native, from a JS config on wasm) and
// passes it to Pick. Default() returns the Reference floor, which is always present.

var registry []Backend

// Register adds a backend to the registry. Called from a backend's init(). A second
// registration with the same Name replaces the first (so a test double can override).
func Register(b Backend) {
	for i, e := range registry {
		if e.Name() == b.Name() {
			registry[i] = b
			return
		}
	}
	registry = append(registry, b)
}

// Registered returns the names of all registered backends, Reference first.
func Registered() []string {
	out := make([]string, 0, len(registry))
	for _, b := range registry {
		out = append(out, b.Name())
	}
	return out
}

// Lookup returns the backend named `name` without falling back. Hosts that expose a
// user-facing backend selector should use this rather than Pick so a typo cannot
// silently run on the Reference floor and masquerade as an accelerator result.
func Lookup(name string) (Backend, bool) {
	for _, b := range registry {
		if b.Name() == name {
			return b, true
		}
	}
	return nil, false
}

// Pick returns the backend named `name`, or Default() if name is "" or unknown. This is
// the runtime, ISA-neutral analogue of resolveTier()/FAK_QKERNEL — generalized across
// the whole device boundary rather than just x86 SIMD tiers.
func Pick(name string) Backend {
	if name != "" {
		if b, ok := Lookup(name); ok {
			return b
		}
	}
	return Default()
}

// Default returns the first registered Reference backend, else the first registered
// backend, else nil. The Reference is the portable floor every target degrades to.
func Default() Backend {
	for _, b := range registry {
		if b.Class() == Reference {
			return b
		}
	}
	if len(registry) > 0 {
		return registry[0]
	}
	return nil
}

// RequireReference reports whether b may be subjected to the exact bit-identity rungs.
// The test harness calls this before any max|Δ|=0 / argmax-oracle assertion, so the
// bit-identity scoping is mechanically enforced: an Approx (device/Q8) backend is never
// held to bit-identity, and a device can never be silently treated as the reference.
func RequireReference(b Backend) bool { return b != nil && b.Class() == Reference }
