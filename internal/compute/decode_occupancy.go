package compute

// decode_occupancy.go — the host-tractable, decode-shaped occupancy + HBM-traffic witness
// for the KernelWiki Blackwell decode-kernel shortlist (docs/notes/2026-07-10-kernelwiki-study.md,
// "Noted, not filed — Blackwell decode-kernel shortlist"). That shortlist deliberately filed NO
// tickets: each wiki witness said "measure the fak-side occupancy/traffic gap FIRST," so this
// file lands that measurement scaffold. It answers, per decode kernel, the two questions a
// device profiler would — how full is the machine (occupancy), and how much HBM does the kernel
// move (traffic) — using EXACT counts, never a timer, the same honesty discipline as prefill.go
// (exact FLOPs/bytes, no fabricated throughput) and fusion_traffic.go (operand counts, host-identical).
//
// Why decode is its own shape, distinct from prefill.go. Prefill drives a P-token panel: every
// GEMM has grid ∝ P·out and the machine fills. DECODE emits ONE token: the flash-attention launch
// is `k_flash_attention<<<nH, 128, …>>>` (cuda_kernels.cu:701,758) — one block per QUERY HEAD, so
// the entire grid is nH blocks. On an A100 (108 SMs) a 9-head model (SmolLM2-135M) launches 9
// blocks onto 108 SMs: 99 SMs sit idle every decode step, and no per-SM occupancy tuning can fill
// them because there is no more grid to schedule. That grid-underfill is the load-bearing decode
// fact, and it is EXACT (a count of blocks vs SMs), not a measurement — so the file/no-file verdict
// for a candidate that targets per-SM occupancy (e.g. register→tmem migration) is decidable on a
// CUDA-less host. The ncu run on a real A100 (tools/dgx_decode_occupancy_ncu.sh) then CORROBORATES
// the achieved-occupancy / DRAM-throughput percentages; it does not change the structural verdict.
//
// What is EXACT here vs what the device confirms. The per-SM occupancy limiter (registers / shared
// memory / warps / blocks) is computed with the standard CUDA occupancy-calculator arithmetic over
// the published per-compute-capability limits (NVArch below) — exact given the launch shape and the
// kernel's register count. The one input this host cannot know is the compiler-assigned registers/
// thread; it is passed in (FlashDecodeLaunch takes it) and is exactly what ncu reports as
// `launch__registers_per_thread`. Crucially, the DEVICE-level verdict (grid underfill ⇒ idle SMs)
// is INDEPENDENT of that register count, so the headline decode gaps are decided without it.

// ---- NVIDIA per-compute-capability occupancy limits -----------------------------
//
// The per-SM ISA limits the occupancy calculator reads. These are architecture (compute-capability)
// constants from the CUDA C Programming Guide "Technical Specifications per Compute Capability"
// table — NOT board-level specs. The SM COUNT is a per-SKU board property (an A100 has 108 enabled
// SMs) and is deliberately NOT in this table: it is passed to Occupancy() as deviceSMs, the same way
// rocm_arch.go keeps the ISA wavefront width in its table but not the CU count. Registers/threads/
// warps/blocks-per-SM are identical across CC 8.0/9.0/10.0; only shared-memory-per-SM grew.

// NVCompute is an NVIDIA compute capability fak models a decode launch against.
type NVCompute uint8

const (
	// NVUnknown is the zero value: a compute capability with no occupancy row.
	NVUnknown NVCompute = iota
	// SM80 is GA100 / A100 (CC 8.0) — the decode MEASUREMENT target (the bridge's bench box).
	SM80
	// SM90 is GH100 / H100 (CC 9.0) — Hopper, the first with programmatic dependent launch (PDL).
	SM90
	// SM100 is GB100 / B200 (CC 10.0) — Blackwell, the shortlist's real target (tmem, nvfp4, CLC).
	SM100
)

// NVArch is one compute capability's per-SM occupancy limits: the four resource ceilings the
// occupancy calculator takes the min over, plus the allocation granularities that round a launch's
// demand up before it is charged. Every field is an ISA constant, host-known and device-independent.
type NVArch struct {
	Cap                  NVCompute
	Label                string // short id, e.g. "sm_80"
	RegsPerSM            int    // physical registers per SM (65536 on 8.0/9.0/10.0)
	MaxThreadsPerSM      int    // resident-thread ceiling (2048 on 8.0/9.0/10.0)
	MaxWarpsPerSM        int    // resident-warp ceiling (= MaxThreadsPerSM/WarpSize)
	MaxBlocksPerSM       int    // resident-block ceiling (32 on 8.0/9.0/10.0)
	SmemPerSM            int    // shared memory per SM in bytes (164 KiB on 8.0; 228 KiB on 9.0/10.0)
	WarpSize             int    // 32 on all NVIDIA
	RegAllocUnit         int    // per-warp register allocation granularity (256 on Volta+)
	WarpAllocGranularity int    // warps rounded to a multiple of this when charging registers (4)
	SmemAllocUnit        int    // shared-memory allocation granularity in bytes (128 on Volta+)
}

// nvArches is the occupancy-limit table, one row per modeled compute capability, in generation
// order. The values are the CUDA C Programming Guide per-compute-capability specs; sm_80 is the
// row the A100 decode measurement is graded against, sm_100 the row the shortlist ultimately targets.
var nvArches = []NVArch{
	{Cap: SM80, Label: "sm_80", RegsPerSM: 65536, MaxThreadsPerSM: 2048, MaxWarpsPerSM: 64,
		MaxBlocksPerSM: 32, SmemPerSM: 164 * 1024, WarpSize: 32, RegAllocUnit: 256,
		WarpAllocGranularity: 4, SmemAllocUnit: 128},
	{Cap: SM90, Label: "sm_90", RegsPerSM: 65536, MaxThreadsPerSM: 2048, MaxWarpsPerSM: 64,
		MaxBlocksPerSM: 32, SmemPerSM: 228 * 1024, WarpSize: 32, RegAllocUnit: 256,
		WarpAllocGranularity: 4, SmemAllocUnit: 128},
	{Cap: SM100, Label: "sm_100", RegsPerSM: 65536, MaxThreadsPerSM: 2048, MaxWarpsPerSM: 64,
		MaxBlocksPerSM: 32, SmemPerSM: 228 * 1024, WarpSize: 32, RegAllocUnit: 256,
		WarpAllocGranularity: 4, SmemAllocUnit: 128},
}

var nvByCap = func() map[NVCompute]NVArch {
	m := make(map[NVCompute]NVArch, len(nvArches))
	for _, a := range nvArches {
		m[a.Cap] = a
	}
	return m
}()

// LookupNVArch resolves a compute capability to its occupancy-limit row, or (zero, false) if fak
// models no limits for it — the fail-closed admission an occupancy calc uses so an unknown arch is
// never silently graded against the wrong ceilings (the analogue of LookupROCmArch).
func LookupNVArch(cap NVCompute) (NVArch, bool) {
	a, ok := nvByCap[cap]
	return a, ok
}

// KnownNVArches returns the occupancy-limit table in generation order.
func KnownNVArches() []NVArch {
	out := make([]NVArch, len(nvArches))
	copy(out, nvArches)
	return out
}

// A100SMs is the enabled-SM count of the shipping A100 (GA100 enables 108 of the die's 128 SMs).
// It is the board-level deviceSMs the decode measurement on the bridge's A100 is graded against —
// separated from NVArch because it is a SKU spec, not an ISA constant.
const A100SMs = 108

// ---- decode launch shapes -------------------------------------------------------

// DecodeLaunch is the launch geometry of one decode kernel: the block size, the compiler's
// registers/thread (the one device-measured input — ncu's launch__registers_per_thread), the
// dynamic shared memory per block, and the TOTAL grid size at the decode step. Every fak decode
// kernel reduces to one of these; the constructors below build the three the shortlist names.
type DecodeLaunch struct {
	Kernel            string // kernel id, e.g. "k_flash_attention"
	ThreadsPerBlock   int
	RegsPerThread     int // launch__registers_per_thread (compiler-assigned; passed in / measured)
	SmemBytesPerBlock int
	GridBlocks        int // blocks launched this decode step (the whole grid, not per-SM)
}

// flashDecodeThreads mirrors FLASH_THREADS in cuda_kernels.cu (the flash/DSA attention block size).
const flashDecodeThreads = 128

// FlashDecodeLaunch is the launch of k_flash_attention at a decode step (cuda_kernels.cu:758):
// `<<<nH, FLASH_THREADS, (hd+FLASH_THREADS)*4>>>` — one block per query head, so GridBlocks = nH.
// regsPerThread is the compiler's register count (ncu launch__registers_per_thread); the grid-
// underfill verdict does not depend on it, but the per-SM limiter does. This is the launch the
// tmem-accumulator (:719) and __ldcs (:726,742) candidates both target.
func FlashDecodeLaunch(g DecodeGeometry, regsPerThread int) DecodeLaunch {
	return DecodeLaunch{
		Kernel:            "k_flash_attention",
		ThreadsPerBlock:   flashDecodeThreads,
		RegsPerThread:     regsPerThread,
		SmemBytesPerBlock: (g.HeadDim + flashDecodeThreads) * 4, // query row + reduction row, f32
		GridBlocks:        g.NHeads,                             // one block per query head
	}
}

// Q8GemmDecodeLaunch is the launch of k_q8_gemm at a decode step (cuda_kernels.cu:455):
// `<<<dim3(out, P), 256>>>` with P=1 → GridBlocks = out (one block per output row). This is the
// decode GEMV the clc-decode-tile-scheduling (:455) candidate targets; its grid is LARGE (out is a
// projection width), so unlike flash attention it fills the machine — its gap is the ragged final
// wave (wave quantization), not global underfill.
func Q8GemmDecodeLaunch(out, regsPerThread int) DecodeLaunch {
	return DecodeLaunch{
		Kernel:            "k_q8_gemm",
		ThreadsPerBlock:   256,
		RegsPerThread:     regsPerThread,
		SmemBytesPerBlock: 0,
		GridBlocks:        out, // dim3(out, 1) at decode
	}
}

// AWQGemvDecodeLaunch is the launch of k_awq_gemv (cuda_kernels.cu:1083): `<<<out, 256>>>` — one
// block per output row. The persistent-kernel-work-stealing-tail-fix (:1084) candidate targets this
// one-block-per-row shape; like q8_gemm it fills when out is large and leaves a wave-quant tail.
func AWQGemvDecodeLaunch(out, regsPerThread int) DecodeLaunch {
	return DecodeLaunch{
		Kernel:            "k_awq_gemv",
		ThreadsPerBlock:   256,
		RegsPerThread:     regsPerThread,
		SmemBytesPerBlock: 0,
		GridBlocks:        out,
	}
}

// ---- occupancy ------------------------------------------------------------------

// DecodeOccupancy is the occupancy analysis of one decode launch on one arch+SM-count: the per-SM
// resident-block limit and which resource binds it, the per-SM theoretical occupancy that implies,
// and — the decode-specific part — how the whole grid lands on the device: how many SMs sit idle,
// the wave structure, and the achieved DEVICE-wide warp occupancy. The per-SM view is the classic
// occupancy calculator; the grid view is what makes decode's underfill visible.
type DecodeOccupancy struct {
	Arch          string
	Kernel        string
	DeviceSMs     int
	WarpsPerBlock int

	// per-SM occupancy (the occupancy-calculator view)
	BlocksPerSM    int     // resident blocks one SM can hold (min over the four limiters)
	BindingLimiter string  // "register" | "shared" | "warp" | "block" — which ceiling binds
	TheoreticalOcc float64 // BlocksPerSM*WarpsPerBlock / MaxWarpsPerSM (per-SM, assumes SM is filled)

	// grid view (the decode-specific part)
	GridBlocks     int     // blocks in the whole launch
	TotalSlots     int     // DeviceSMs * BlocksPerSM — resident blocks the whole device can hold
	IdleSMs        int     // SMs that get NO block this step (the decode-underfill headline; 0 if grid ≥ SMs)
	Waves          int     // ceil(GridBlocks / TotalSlots) — scheduling waves the grid needs
	WaveQuantWaste float64 // 1 − GridBlocks/(Waves*TotalSlots): the ragged-final-wave idle fraction
	DeviceOcc      float64 // achieved device-wide warp occupancy = resident warps / (DeviceSMs*MaxWarpsPerSM)
}

// blocksPerSM runs the standard occupancy-calculator min over the four per-SM resource ceilings and
// reports the binding one. Register demand is charged at warp granularity (regs/thread × warpSize,
// rounded up to RegAllocUnit, then warps-per-SM rounded down to WarpAllocGranularity); shared memory
// is rounded up to SmemAllocUnit. A zero demand for a resource means that resource does not bind
// (a register-free or smem-free kernel). Ties resolve in resource order register→shared→warp→block.
func (a NVArch) blocksPerSM(l DecodeLaunch) (n int, limiter string) {
	warpsPerBlock := ceilDivInt(l.ThreadsPerBlock, a.WarpSize)
	if warpsPerBlock < 1 {
		warpsPerBlock = 1
	}

	// warp ceiling and hard block ceiling (always present)
	blocksWarp := a.MaxWarpsPerSM / warpsPerBlock
	blocksBlock := a.MaxBlocksPerSM

	// register ceiling (absent when the compiler count is unknown/zero)
	blocksReg := maxInt32Blocks
	if l.RegsPerThread > 0 {
		regsPerWarp := roundUpInt(l.RegsPerThread*a.WarpSize, a.RegAllocUnit)
		warpsByReg := roundDownInt(a.RegsPerSM/regsPerWarp, a.WarpAllocGranularity)
		blocksReg = warpsByReg / warpsPerBlock
	}

	// shared-memory ceiling (absent when the kernel uses no dynamic smem)
	blocksSmem := maxInt32Blocks
	if l.SmemBytesPerBlock > 0 {
		per := roundUpInt(l.SmemBytesPerBlock, a.SmemAllocUnit)
		blocksSmem = a.SmemPerSM / per
	}

	// min with resource-order tie-break
	n, limiter = blocksReg, "register"
	if blocksSmem < n {
		n, limiter = blocksSmem, "shared"
	}
	if blocksWarp < n {
		n, limiter = blocksWarp, "warp"
	}
	if blocksBlock < n {
		n, limiter = blocksBlock, "block"
	}
	if n < 0 {
		n = 0
	}
	return n, limiter
}

// maxInt32Blocks stands in for "this resource does not bind" — larger than any real per-SM block
// count, so it never wins the min but keeps the arithmetic in plain ints.
const maxInt32Blocks = 1 << 20

// Occupancy computes the per-SM and grid-level occupancy of launch l on arch a with deviceSMs SMs.
// The per-SM half is the occupancy calculator; the grid half projects the whole launch onto the
// device — the step that exposes decode's idle SMs. deviceSMs ≤ 0 is treated as A100SMs (the
// measurement box) so a bare call still models the reference device.
func (a NVArch) Occupancy(l DecodeLaunch, deviceSMs int) DecodeOccupancy {
	if deviceSMs <= 0 {
		deviceSMs = A100SMs
	}
	bpsm, limiter := a.blocksPerSM(l)
	warpsPerBlock := ceilDivInt(l.ThreadsPerBlock, a.WarpSize)
	if warpsPerBlock < 1 {
		warpsPerBlock = 1
	}
	occ := DecodeOccupancy{
		Arch:           a.Label,
		Kernel:         l.Kernel,
		DeviceSMs:      deviceSMs,
		WarpsPerBlock:  warpsPerBlock,
		BlocksPerSM:    bpsm,
		BindingLimiter: limiter,
		TheoreticalOcc: ratio(int64(bpsm*warpsPerBlock), int64(a.MaxWarpsPerSM)),
		GridBlocks:     l.GridBlocks,
		TotalSlots:     deviceSMs * bpsm,
	}
	// idle SMs: the scheduler spreads blocks one-per-SM first, so a grid smaller than the SM count
	// leaves DeviceSMs−GridBlocks SMs with nothing to do this decode step. This is the underfill.
	if l.GridBlocks < deviceSMs {
		occ.IdleSMs = deviceSMs - l.GridBlocks
	}
	// wave structure + achieved device occupancy.
	if occ.TotalSlots > 0 {
		occ.Waves = ceilDivInt(l.GridBlocks, occ.TotalSlots)
		residentBlocks := l.GridBlocks
		if residentBlocks > occ.TotalSlots {
			residentBlocks = occ.TotalSlots // one wave's worth resident at a time
		}
		occ.DeviceOcc = ratio(int64(residentBlocks*warpsPerBlock), int64(deviceSMs*a.MaxWarpsPerSM))
		if occ.Waves > 0 {
			occ.WaveQuantWaste = 1 - ratio(int64(l.GridBlocks), int64(occ.Waves*occ.TotalSlots))
		}
	}
	return occ
}

// ---- decode HBM traffic ---------------------------------------------------------

// DecodeGeometry is the model shape at a decode step: the attention head layout, the KV-cache
// length the flash inner loop streams, and the projection widths that size the decode GEMV grids.
// P is fixed at 1 (decode emits one token) so it is not a field; KVLen is the flash loop bound nPos.
type DecodeGeometry struct {
	DModel   int // residual width
	NHeads   int // query heads (= flash GridBlocks)
	NKVHeads int // KV heads (GQA: ≤ NHeads) — the reuse-optimal K/V stream width
	HeadDim  int // per-head width
	DFF      int // FFN inner width (a decode GEMV out dim)
	Vocab    int // LM-head width (the largest decode GEMV out dim)
	KVLen    int // N: KV-cache length streamed by the flash inner loop this step
}

// DecodeTraffic is the HBM traffic of one decode attention layer: the bytes the kernel AS WRITTEN
// streams, the bytes a GQA-reuse-optimal kernel would stream, the wasted re-read between them, and
// the arithmetic intensity. Every field is an exact operand count (f32), host-identical. The
// waste term is what the __ldcs / one-block-per-KV-head reuse candidates target; the intensity is
// the memory-bound verdict (~0.5 FLOP/byte) that makes decode attention a streaming problem.
type DecodeTraffic struct {
	Streamed     int64   // HBM bytes k_flash_attention actually moves (K/V re-read per query head)
	ReuseOptimal int64   // HBM bytes if GQA siblings shared one K/V stream (per KV head)
	KVReuseWaste int64   // Streamed − ReuseOptimal — the redundant K/V re-read (≥ 0)
	KVStreamFrac float64 // K/V stream bytes / Streamed — how much of decode traffic is the KV stream
	Intensity    float64 // FLOPs / Streamed bytes — the roofline intensity (memory-bound ≈ 0.5)
}

// DecodeHBMTraffic models one decode attention layer at geometry g. Both the as-written and the
// reuse-optimal paths stream Q once (nH·hd) and write O once (nH·hd); they differ only in the K/V
// stream — k_flash_attention runs one independent block per QUERY head, so each of the nH blocks
// re-reads its KV head's K and V across all KVLen positions (nH·KVLen·hd each), whereas a kernel
// that shared the GQA group would read each KV head's K/V once (nKV·KVLen·hd each). At a real decode
// length (KVLen ≫ hd) the K/V stream dwarfs Q/O, so the reduction is ~(nH−nKV)/nH of the dominant
// term. FLOPs are the two matmuls (Q·Kᵀ and the ΣwV output) over the streamed keys: 4·nH·KVLen·hd,
// giving the ~0.5 FLOP/byte intensity that pins decode attention as memory-bound at every length.
func DecodeHBMTraffic(g DecodeGeometry) DecodeTraffic {
	nH := int64(g.NHeads)
	nKV := int64(g.NKVHeads)
	hd := int64(g.HeadDim)
	N := int64(g.KVLen)

	qo := 4 * (nH*hd + nH*hd)           // Q read + O write, f32, both paths
	kvStreamed := 4 * (2 * nH * N * hd) // K + V, re-read per query head (as written)
	kvOptimal := 4 * (2 * nKV * N * hd) // K + V, one stream per KV head (GQA-shared)

	streamed := qo + kvStreamed
	optimal := qo + kvOptimal
	waste := streamed - optimal
	if waste < 0 {
		waste = 0
	}
	flops := int64(4) * nH * N * hd // Q·Kᵀ (2·nH·N·hd) + ΣwV (2·nH·N·hd)
	return DecodeTraffic{
		Streamed:     streamed,
		ReuseOptimal: optimal,
		KVReuseWaste: waste,
		KVStreamFrac: ratio(kvStreamed, streamed),
		Intensity:    ratio(flops, streamed),
	}
}

// decodeMemoryBoundRidge is the roofline ridge point (FLOP/byte) below which a kernel is memory-
// bound. It is the A100's peak-FLOP/s ÷ peak-HBM-bytes/s: ~19.5 TFLOP/s f32 ÷ ~1.55 TB/s ≈ 12.6.
// Decode attention's ~0.5 FLOP/byte sits two orders below it — memory-bound with no ambiguity — so
// the classification is robust to the exact ridge; the constant is documented, not load-bearing.
const decodeMemoryBoundRidge = 12.6

// MemoryBound reports whether the decode traffic's intensity sits below the memory-bound ridge.
// This is the predicate the __ldcs / KV-reuse candidates gate on: a streaming-cache hint only helps
// a kernel that is memory-bound on a use-once stream, which DecodeHBMTraffic shows decode attention is.
func (t DecodeTraffic) MemoryBound() bool { return t.Intensity < decodeMemoryBoundRidge }

// ---- candidate gap verdicts -----------------------------------------------------

// DecodeKernelGap is one shortlist candidate's verdict against this witness: the seam it targets,
// whether an A100 decode occupancy/traffic run CAN witness it (false ⇒ a Blackwell/Hopper-only
// mechanism this A100 witness cannot measure — nvfp4, PDL, CLC try-cancel), whether the model then
// PREDICTS a real gap the candidate would close, the model quantity the verdict reads, and why.
// The file/no-file decision is Measurable && PredictedGap.
type DecodeKernelGap struct {
	Candidate    string
	Seam         string
	Measurable   bool
	PredictedGap bool
	KeysOn       string
	Rationale    string
}

// ShouldFile is the promotion rule the study's "file-after-measurement" gate encodes: promote a
// candidate to a ticket only when this witness can measure it AND the model shows a real gap.
func (g DecodeKernelGap) ShouldFile() bool { return g.Measurable && g.PredictedGap }

// wave-quant waste above this fraction is a real tail worth a persistent/CLC tile scheduler; below
// it the ragged final wave is negligible. 5% is a deliberately conservative floor.
const decodeTailGapFloor = 0.05

// DecodeGapReport grades all eight shortlist candidates against the occupancy + traffic model at
// the given arch/SM-count/geometry. flashRegs is the compiler's registers/thread for the flash
// kernel (ncu launch__registers_per_thread; the applied caller passes the measured value). The five
// A100-measurable candidates' PredictedGap is COMPUTED from the model (grid underfill, memory-bound
// intensity, wave-quant tail); the three Blackwell/Hopper-only mechanisms are marked Measurable=false
// with a fixed rationale — this witness cannot see them, and says so rather than guessing.
func DecodeGapReport(a NVArch, deviceSMs int, g DecodeGeometry, flashRegs int) []DecodeKernelGap {
	flash := a.Occupancy(FlashDecodeLaunch(g, flashRegs), deviceSMs)
	// the decode GEMV that fills the machine — use the FFN width as a representative large out dim.
	gemv := a.Occupancy(Q8GemmDecodeLaunch(g.DFF, flashRegs), deviceSMs)
	traffic := DecodeHBMTraffic(g)

	// headline underfill: flash decode grid = nH ≪ SMs ⇒ idle SMs. Exact, register-independent.
	underfill := flash.IdleSMs > 0
	// memory-bound on a use-once KV stream that dominates traffic.
	streamingGap := traffic.MemoryBound() && traffic.KVStreamFrac > 0.5
	// ragged final wave on the (otherwise machine-filling) decode GEMV.
	tailGap := gemv.WaveQuantWaste > decodeTailGapFloor

	return []DecodeKernelGap{
		{
			Candidate: "persistent-kernel-work-stealing-tail-fix", Seam: "cuda_kernels.cu:1084 (k_awq_gemv) / :758 (flash decode grid=nH)",
			Measurable: true, PredictedGap: underfill,
			KeysOn:    "IdleSMs",
			Rationale: "flash decode launches one block per query head (grid=nH); on A100 that leaves nH≪108 SMs, so most SMs idle every step. A persistent/work-stealing kernel that keeps all SMs busy across heads+positions closes exactly this underfill.",
		},
		{
			Candidate: "l1-cache-hints-decode", Seam: "cuda_kernels.cu:726,742 (K/V load loops)",
			Measurable: true, PredictedGap: streamingGap,
			KeysOn:    "Intensity(MemoryBound), KVStreamFrac",
			Rationale: "decode attention is memory-bound (~0.5 FLOP/byte) and the K/V stream is the dominant HBM traffic, read use-once; __ldcs streaming loads avoid polluting L1/L2 with that use-once stream, protecting the reused query row.",
		},
		{
			Candidate: "clc-decode-tile-scheduling", Seam: "cuda_kernels.cu:455 (k_q8_gemm)",
			Measurable: true, PredictedGap: tailGap,
			KeysOn:    "WaveQuantWaste",
			Rationale: "the P=1 decode GEMV fills the machine but its grid=out tiles into scheduling waves whose ragged final wave leaves the tail idle; a CLC/persistent tile scheduler reclaims that wave-quantization waste.",
		},
		{
			Candidate: "moe-launch-fusion-ladder", Seam: "internal/compute/fusion_traffic.go:141",
			Measurable: true, PredictedGap: underfill,
			KeysOn:    "IdleSMs (per-expert GEMV)",
			Rationale: "an MoE decode step routes a handful of tokens to each expert, so every expert GEMV is a tiny-grid launch that underfills the machine like flash decode; fusing the launch ladder amortizes the many underfilled launches. (Exercised only by an MoE model — the dense A100 bench witnesses the same underfill shape.)",
		},
		{
			Candidate: "tmem-accumulator-migration", Seam: "cuda_kernels.cu:719 (float acc[FLASH_ACC_MAX])",
			Measurable: true, PredictedGap: false, // decided below-comment: grid-bound, not per-SM-bound
			KeysOn:    "BindingLimiter, IdleSMs",
			Rationale: "migrating the acc[] register array to Blackwell tensor memory raises PER-SM occupancy, but decode is GRID-bound (IdleSMs>0): the device already runs only nH≪SMs blocks, so freeing registers adds no schedulable work. No device-level gap — do not file (also an sm_100-only mechanism).",
		},
		{
			Candidate: "clc-try-cancel-speculative", Seam: "internal/compute/discard_admit.go:44",
			Measurable: false, PredictedGap: false,
			KeysOn:    "—",
			Rationale: "host-side speculative admission + CLC try-cancel is a Hopper+ launch-control mechanism, not an intra-kernel decode-occupancy phenomenon; this occupancy/traffic witness cannot see it. Defer to a launch-control witness.",
		},
		{
			Candidate: "nvfp4-two-level-block-scale", Seam: "internal/ggufload/gguf_dequant.go:422",
			Measurable: false, PredictedGap: false,
			KeysOn:    "—",
			Rationale: "nvfp4 is an sm_100-only numeric format; the A100 (sm_80) bench has no nvfp4 path to profile. Measure two-level block-scale traffic on a Blackwell node.",
		},
		{
			Candidate: "pdl-moe-kernel-overlap", Seam: "cuda_kernels.cu:454 (k_q8_quant_act → k_q8_gemm)",
			Measurable: false, PredictedGap: false,
			KeysOn:    "—",
			Rationale: "programmatic dependent launch (PDL) overlaps consecutive kernels and is sm_90+; the gap it targets is inter-kernel launch latency, which an intra-kernel occupancy witness does not measure. Defer to a launch-latency witness on Hopper+.",
		},
	}
}

// ---- small int helpers (occupancy arithmetic) -----------------------------------

func ceilDivInt(a, b int) int {
	if b <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

func roundUpInt(v, unit int) int {
	if unit <= 0 {
		return v
	}
	return ((v + unit - 1) / unit) * unit
}

func roundDownInt(v, unit int) int {
	if unit <= 0 {
		return v
	}
	return (v / unit) * unit
}
