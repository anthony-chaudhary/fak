package model

import (
	"fmt"
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// SessionFromPrefix starts a session whose cache is a clone of an already-computed
// prefix, so only the suffix needs prefilling (real prefix reuse). For
// GLM-MoE-DSA the clone carries the DSA attention/index cache instead of the dense
// GQA K/V rows.
//
// It refuses, by name, an architecture whose session state is NOT the KVCache (#5548). A
// recompute session carries its prefix as a token history and leaves the cache empty, so
// this clone would hand back a session that has ingested nothing while its caller believes
// it holds the prefix — and then prefills only the suffix. Nothing downstream can catch
// that: an empty cache is a well-formed zero-length prefix at every consumer. The refusal
// is loud for the same reason requireGemma4Session's is: a wrong answer with no error is
// the failure this whole family of guards exists to prevent.
func (m *Model) SessionFromPrefix(prefix *KVCache) *Session {
	if !m.Cfg.KVPrefixReuseSupported() {
		panic("model: SessionFromPrefix is not available for architecture " + m.Cfg.archFamilyKey() +
			": its session state is the token history, not the KV cache, so a cache clone carries no prefix")
	}
	return &Session{M: m, Cache: prefix.Clone()}
}

// Session drives generation over a kernel-owned KV cache. Prefill ingests a prompt;
// Step decodes one token. Both share the exact per-token math the verified full
// forward pass uses, so cached decode is provably identical to full prefill.
// GLM-MoE-DSA uses a separate DSA attention/index cache carried inside KVCache.
//
// Naming: this is the token-decoder sense of "session" (a generator over a KV
// cache), NOT the drive-state session. The canonical "session" — run-state,
// budget, priority, pace — is internal/session.Table / session.State; the wire
// DTO of that drive state is gateway.SessionState. See the vocabulary worklist at
// docs/notes/VOCAB-DISAMBIGUATION-WORKLIST-2026-06-24.md.
// PrefixSnapshot is an independently owned inference prefix. For legacy/host
// sessions Cache is sufficient. Device hybrid sessions additionally carry attention
// KV and recurrent Qwen state; keeping all three in one owner prevents partial restores.
type PrefixSnapshot struct {
	owner      *Session
	epoch      uint64
	Cache      *KVCache
	halKV      compute.KVStore
	halLineage tokenLineage
	qwen35     *qwen35HALState
	Backend    compute.Backend
	Tokens     int
	// DenseGPULayers / GPULayers preserve layer placement across prefix snapshots.
	DenseGPULayers int
	GPULayers      int
	// Native MTP consumes the exact pre-final-norm residual history. It is part of
	// session state just as surely as KV: restoring one without the other can make
	// a rejected draft visible to the next proposal even though attention rolled back.
	captureTargetHidden bool
	targetHidden        [][]float32
	targetHiddenTokens  []int
}

// PrefixSnapshot captures a deep clone suitable for shared-prefix admission.
func (s *Session) PrefixSnapshot() (*PrefixSnapshot, error) {
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	if s == nil || s.Cache == nil {
		return nil, fmt.Errorf("model: cannot snapshot nil session cache")
	}
	tokens := s.Cache.Len()
	if s.Backend != nil && s.halKV != nil {
		// HAL attention KV is the physical position authority. Hybrid device paths
		// may keep the host model cache positionless while all prefix state lives in
		// halKV plus recurrent tensors.
		tokens = s.halKV.Len()
	}
	out := &PrefixSnapshot{owner: s, epoch: s.cacheGeometryEpoch, Cache: s.Cache.Clone(), halLineage: s.halLineage.clone(0), Backend: s.Backend, Tokens: tokens, DenseGPULayers: s.DenseGPULayers, GPULayers: s.GPULayers}
	s.targetHiddenMu.RLock()
	out.captureTargetHidden = s.captureTargetHidden
	out.targetHidden = cloneTargetHidden(s.targetHidden)
	out.targetHiddenTokens = append([]int(nil), s.targetHiddenTokens...)
	s.targetHiddenMu.RUnlock()
	if s.Backend == nil {
		return out, nil
	}
	if s.halKV == nil {
		return nil, fmt.Errorf("model: backend session has no device KV store")
	}
	out.halKV = s.halKV.Clone()
	var err error
	out.qwen35, err = cloneQwen35HALState(s.qwen35HAL, s.Backend)
	if err != nil {
		out.Close()
		return nil, err
	}
	return out, nil
}

// Clone makes a second independent owner for lookup; the cache retains the original.
func (p *PrefixSnapshot) Clone() (*PrefixSnapshot, error) {
	started := prefixProfileStart()
	defer func() { emitPrefixProfile(started, "device_clone", "complete", p, nil) }()
	if p == nil || p.Cache == nil {
		return nil, nil
	}
	out := &PrefixSnapshot{
		owner: p.owner, epoch: p.epoch, Cache: p.Cache.Clone(), halLineage: p.halLineage.clone(0), Backend: p.Backend, Tokens: p.Tokens,
		captureTargetHidden: p.captureTargetHidden,
		targetHidden:        cloneTargetHidden(p.targetHidden),
		targetHiddenTokens:  append([]int(nil), p.targetHiddenTokens...),
	}
	if p.Backend == nil {
		return out, nil
	}
	if p.halKV == nil {
		return nil, fmt.Errorf("model: device prefix snapshot has no KV store")
	}
	out.halKV = p.halKV.Clone()
	var err error
	out.qwen35, err = cloneQwen35HALState(p.qwen35, p.Backend)
	if err != nil {
		out.Close()
		return nil, err
	}
	return out, nil
}

// Restore installs this snapshot into a fresh backend session and transfers ownership.
func (p *PrefixSnapshot) Restore(s *Session) error {
	if p != nil && p.owner != nil {
		p.owner.cacheGeometryMu.RLock()
		defer p.owner.cacheGeometryMu.RUnlock()
	}
	if p != nil && p.owner != nil && p.epoch != p.owner.cacheGeometryEpoch {
		return fmt.Errorf("model: stale prefix snapshot after cache rebuild")
	}
	if p == nil || s == nil || p.Cache == nil {
		return fmt.Errorf("model: invalid prefix snapshot restore")
	}
	if p.Backend != s.Backend {
		return fmt.Errorf("model: prefix snapshot backend mismatch")
	}
	if s.Backend != nil {
		if p.halKV == nil {
			return fmt.Errorf("model: device prefix snapshot missing KV")
		}
		if s.halKV != nil {
			s.halKV.Free()
		}
		s.closeQwen35HALState()
		s.halKV, s.halLineage, s.qwen35HAL = p.halKV, p.halLineage, p.qwen35
		p.halKV, p.qwen35 = nil, nil
		p.halLineage = tokenLineage{}
	}
	if s.DenseGPULayers == 0 && p.DenseGPULayers != 0 {
		s.DenseGPULayers = p.DenseGPULayers
	}
	if s.GPULayers == 0 && p.GPULayers != 0 {
		s.GPULayers = p.GPULayers
	}
	s.Cache = p.Cache
	p.Cache = nil
	p.halLineage = tokenLineage{}
	s.targetHiddenMu.Lock()
	s.captureTargetHidden = p.captureTargetHidden
	s.targetHidden, s.targetHiddenTokens = p.targetHidden, p.targetHiddenTokens
	s.targetHiddenMu.Unlock()
	p.targetHidden, p.targetHiddenTokens = nil, nil
	return nil
}

// Close releases all device ownership held by a cache node or failed clone.
func (p *PrefixSnapshot) Close() {
	if p == nil {
		return
	}
	if p.halKV != nil {
		p.halKV.Free()
		p.halKV = nil
	}
	if p.qwen35 != nil {
		p.qwen35.free(p.Backend)
		p.qwen35 = nil
	}
	p.Cache = nil
	p.targetHidden, p.targetHiddenTokens = nil, nil
}

func cloneTargetHidden(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}

const q4kMLPOutputSlabReceiptSchema = "fak-q4k-gateup-slab/v1"

type q4kMLPOutputSlabStats struct {
	Calls          uint64
	Allocations    uint64
	Reuses         uint64
	HighWaterBytes uint64
}

type Session struct {
	cacheGeometryMu     sync.RWMutex
	cacheGeometryFailed bool
	cacheGeometryEpoch  uint64
	M                   *Model
	Cache               *KVCache
	// Backend is non-nil when this session is intentionally running through the
	// internal/compute HAL instead of the legacy direct []float32 path. The legacy
	// path stays the default until the full optimized prefill/batch path is adopted.
	Backend compute.Backend
	// DenseGPULayers bounds the contiguous layer band [0, DenseGPULayers) placed on the
	// device (Backend), with the remaining layers [DenseGPULayers, NumLayers) executing on
	// host CPU. GPULayers is supported as an alias. 0 (default) runs all layers on Backend.
	DenseGPULayers int
	GPULayers      int
	// Q4KGateUpOutputSlab explicitly enables the experimental session-owned Metal
	// gate/up output slab. The default remains off until the measured KEEP gate lands.
	Q4KGateUpOutputSlab bool
	halKV               compute.KVStore
	halLineage          tokenLineage
	// halW memoizes weights staged onto Backend so a device session uploads each weight
	// to VRAM exactly once, not once per token. (On cpu-ref, Upload is identity over the
	// zero-copy host view, so caching changes nothing and the bit-equality gate holds.)
	halW map[string]compute.Tensor
	// borrowedHALW marks halW keys whose handles belong to model-lifetime immutable
	// device residency. It preserves the hot plain-map lookup while making Close's
	// ownership decision O(1) per entry; unmarked entries remain session-owned.
	borrowedHALW map[string]struct{}
	// ExpertRingBytes bounds the DEVICE residency of ROUTED expert weights (`.mlp.experts.N.*`) at
	// this many bytes, staging them through a pagedRing (expert_ring_hal.go, #5611) instead of the
	// never-evicting halW memoizer above. It is what makes the ACTIVATED expert set a bounded object:
	// a MoE checkpoint activates ~3% of its experts per token, so a budget far below the expert bulk
	// still serves the working set, and the coldest expert is evicted rather than accumulated. Dense,
	// attention, router, lm_head and SHARED-expert weights are unaffected — they are activated every
	// token and keep permanent residency. 0 (the default) disables the ring entirely and leaves every
	// path byte-for-byte unchanged.
	ExpertRingBytes int64
	// ExpertPinBudget is how many routed experts the ring may hold PINNED — exempt from LRU eviction
	// — chosen as the hottest of the persisted cross-session usage prior and drifted between turns by
	// ExpertRingEndTurn (expert_ring_pins.go, #5613). It buys back the cold page-in a hot expert
	// otherwise pays after every eviction and every restart. It must stay well under the ring's own
	// capacity in experts: a pin-set as large as the ring leaves no victims and the ring degrades to
	// a memoizer that refuses new work. 0 (the default) is plain LRU.
	ExpertPinBudget int
	// ExpertUsagePath is where this session dumps its routed-expert usage histogram at each turn
	// boundary and warm-starts its pin-set from at the first staging. A missing file is the cold
	// first run, not an error — the run BUILDS the prior a later one starts from, which is why a
	// path alone (with no pin budget) is still useful. "" disables persistence entirely; a caller
	// folding several sessions' dumps sums them with SumExpertUsageHistograms and points here at the
	// result.
	ExpertUsagePath string
	// ExpertRingEvict selects how the ring ranks eviction victims among its UNPINNED residents
	// (expert_ring_policy.go, #5615). The zero value is LRU — polymodel's own choice, which the ring
	// has always inherited and which is a default rather than a finding. Promote the value-aware
	// candidate only from a measured verdict on this workload's own trace
	// (SelectExpertRingEvictPolicy); it is fixed for the ring's life, because switching mid-flight
	// would score one window under two policies.
	ExpertRingEvict ExpertRingEvictPolicy
	// ExpertPrefetch selects whether each MoE layer's activated set is staged into the ring up front
	// (the default, expert_ring_prefetch.go, #5614) or discovered one expert at a time as the GEMMs
	// reach for it. Inert without a ring, like every knob above it.
	ExpertPrefetch ExpertPrefetchMode
	// expertRing is the bounded routed-expert ring, built lazily on the first routed-expert staging
	// when ExpertRingBytes > 0 and freed by Close. nil on every session that never declared a budget.
	// When sharedRing is set it points at THAT ring instead, so every routed-expert path — demand,
	// prefetch, telemetry — reads a shared ring by exactly the rule it read a private one by.
	expertRing *pagedRing
	// sharedRing is the cross-agent routed-expert residency this session attached to (R7/#5618,
	// expert_ring_shared.go), nil for the per-conversation default. It is what makes Close DETACH
	// rather than free: the bytes belong to the (model, device) pair, not to this conversation.
	//
	// Only routed-expert WEIGHT residency is shared. KV cache, conversation state and halW stay
	// per-session — the safety property SharedExpertRing.Attach enforces by refusing a session over a
	// different *Model or Backend.
	sharedRing *SharedExpertRing
	// ringAgent is this session's identity in the shared ring's coalescing ledger; empty when private.
	ringAgent string
	// ringDepth is the reentrancy depth of the shared-ring lock span this session currently holds (see
	// Session.ringEnter). It needs no synchronization of its own: a Session is single-goroutine, so
	// only the goroutine running its forward can ever touch it.
	ringDepth int
	// expertPinErr retains a warm-start load failure until the next turn boundary reports it: a
	// corrupt usage dump must degrade the session to a cold start, not fail it, but must not vanish.
	expertPinErr error
	// halStep counts tokens run through the HAL (diagnostic / legacy warm-up counter).
	halStep int
	// halLogitsWarm gates CUDA-graph capture: it flips true only after one FULL
	// logits-producing forward has run UNCAPTURED, so every weight on the logits path
	// (model.norm, lm_head) and every transient size (incl. the vocab-wide logits buffer)
	// is already pooled. Capturing before that warm step is what crashed the serve (#932):
	// the first captured logits step hit a fresh cudaMalloc — illegal mid-capture. A
	// noLogits prefill step never touches the logits path, so counting steps (the old
	// halStep>=2 gate) did not guarantee it was warm; this flag does.
	halLogitsWarm bool
	// qwen35HAL owns the backend-resident convolution/recurrent state for every
	// Qwen3.5/3.6 linear-attention layer. It is separate from Cache.linear, which is
	// the legacy CPU/reference object's state and is never consulted by a HAL session.
	qwen35HAL *qwen35HALState
	// qwen35MetalStateIdentity is request-observation state only. Keeping it on
	// Session lets receipt-opted selector-off Metal control runs own an identity
	// without fabricating a HAL owner or entering prefix snapshots.
	qwen35MetalStateIdentity *qwen35MetalStateIdentityObservation
	// halClosed makes Close idempotent and prevents an operation failure from falling
	// through to the legacy CPU path on a later request. halFailure is the witnessed
	// backend error re-raised by any attempted reuse of that failed session.
	halClosed  bool
	halFailure error

	// glmDsaHeadNameLogged gates the one-time FAK_GLMDSA_DUMP head-resolution log (#996 LM-head probe).
	glmDsaHeadNameLogged bool

	// Quant selects the Q8_0 quantized forward path (quant_forward.go) for this session's
	// prefill and decode. The f32 path is the default and is left byte-for-byte unchanged;
	// set Quant only on a session whose Model has had Quantize() called. The KV cache it
	// builds is the same f32 object either way, so Evict/Clone and the proven KV rungs are
	// independent of this flag.
	Quant bool

	// Q4 selects the resident int4 (Q4_0-style) forward path (quant_q4.go) over the Q8/f32
	// paths. Set only on a session whose Model has had QuantizeQ4() called (the q4w resident
	// copy exists). When Q4 is set, decode routes weight matmuls through q4Kernel — int4
	// streams ~1.8× fewer bytes/token than Q8, raising the decode ceiling toward the
	// llama.cpp q4 bar (see QWEN36-NATIVE-PERF-PLAN-2026-06-19.md). Prefill still runs the
	// Q8 batched GEMM; int4 is a decode-path optimization for now.
	Q4 bool

	// Q4K selects the resident hybrid Q4_K path (quant_q4k*.go) for a model loaded via the
	// memory-lean ggufload.LoadModelQ4K loader: identity-normalized matmul weights run as
	// raw Q4_K blocks (~0.56 B/weight, the bandwidth bulk), the normalize-sensitive
	// projections as Q8_0, small tensors as f32. A resident-hybrid matKernel dispatches per
	// name, so this one flag moves both prefill and decode onto the resident mixed path.
	// This is the end-to-end-correct, low-memory route toward the q4_k_m decode bar.
	Q4K bool

	// F16 routes this session's matmul weight uploads through the device F16 GEMM
	// (compute.F16 — the fp16 tensor-core HGEMM, #484, floor cudaFP16CosineMin=0.997)
	// instead of Q8 (s.Quant) or Q4_K (s.Q4K). Unlike those it needs no resident
	// prequantized copy: the host f32 weight is narrowed to __half at H2D by the
	// UploadDtype-capable backend (Caps().UploadDtype), so the same manifest weight is
	// uploaded once as F16. This is the Session-level device-dtype select that lets an
	// fak-cuda-f16 bench engine exist (Lever 4 residual, epic #1476 C4). Inert on a
	// backend without UploadDtype (cpu-ref ignores the narrowing) — useHALF16Weights
	// gates on it, so a non-device session falls through to the f32 weightHAL path.
	F16 bool

	// GPTQ selects the resident AutoGPTQ/GPTQModel path loaded by LoadGPTQ. It routes
	// matmul weights through residentMatRows (GPTQ when present, f32 for small tensors)
	// using the shared per-token blockStep skeleton. It is opt-in so existing f32/Q8/Q4
	// sessions remain byte-for-byte on their prior paths.
	GPTQ bool

	// CPUOffloadExperts routes the MoE expert GEMMs (mlp.experts.* and mlp.shared_experts.*)
	// to host RAM while the dense projections + router + attention run on Backend — the
	// llama.cpp `--n-cpu-moe` hybrid. It is the path that lets a Q4_K model whose experts dwarf
	// VRAM (GLM-5.2 Q4_K_M ≈ 424 GB) serve at all: experts live in the 1007 GB host RAM, the
	// every-token dense FLOPs stay on the GPU. Only the GLM-DSA forward honors it today
	// (glmDsaMatKernel, moe_offload.go); with Backend nil it is a no-op (everything already on
	// host) and the forward stays byte-for-byte the resident path. See splitKernel.
	CPUOffloadExperts bool

	// ExpertSpillLayers GRADES CPUOffloadExperts (#5612): with the split on, spill only the FIRST N
	// MoE layers' expert GEMMs to host RAM and keep the rest device-resident — llama.cpp's
	// `--n-cpu-moe N`, over the model's real MoE layer ordinals (MoEExpertLayers). It is the dial
	// between the two endpoints the split kernel could express before: all experts host, or none.
	//
	// 0 (the default) or a value at/above the MoE layer count means UNGRADED — every expert weight
	// spills, byte-for-byte the pre-#5612 predicate — so the field is inert unless a plan sets it,
	// and CPUOffloadExperts alone still means what it always meant. Nothing spills when
	// CPUOffloadExperts is false, whatever this holds. Size it with Model.ResolveExpertSpillPlacement and install
	// it with Session.ApplyExpertSpillPlacement, which keeps it in agreement with ExpertRingBytes.
	ExpertSpillLayers int

	// spillOnHost memoizes the graded placement predicate (expertSpillOnHost). It is consulted per
	// GEMM and its construction walks every resident tensor name, so it is built once per session;
	// ApplyExpertSpillPlacement clears it so a re-planned session cannot keep running the old grade.
	spillOnHost func(string) bool

	// PrecisionPolicy enables dynamic whole-token precision. When set, Prefill/Step
	// speculatively run the Q8_0 path, inspect the returned distribution, and may roll the
	// KV cache back to recompute the same token/span in f32. It is additive: nil preserves
	// the fixed f32 / fixed Quant behavior exactly.
	PrecisionPolicy *DynamicPrecisionPolicy
	PrecisionStats  PrecisionStats

	// qScratch reuses the Q8 activation vector storage for serial quantized decode/head
	// GEMVs. Each qMatRows call consumes the vector before the next quantization overwrites
	// it, so this removes hot-path allocation without changing any Q8 arithmetic.
	qScratch q8Vec
	qDecode  *qDecodeBuf

	// Metal routes PREFILL's projection GEMMs through the Metal GPU backend
	// (metal_prefill.go, built only under -tags fakmetal) to reach llama.cpp-Metal prefill
	// parity on Apple Silicon — prefill is compute-bound, where the GPU's FLOP advantage is
	// decisive. Decode is untouched (it stays the bandwidth-bound CPU Q8 path, where Metal
	// barely helps). Only set Metal on a quantized model after metalgemm.Available() is true;
	// the same f32 KV cache is built either way, so KV semantics are unchanged.
	Metal bool

	// MetalQ4K routes the resident-Q4_K hybrid PREFILL's q4_k-majority projection/MLP GEMMs
	// through the Metal q4_k dequant-GEMM (internal/metalgemm/q4k.m, built only under -tags
	// fakmetal) instead of the CPU q4kGemm. Unlike Metal (above) this needs no f16 weight set —
	// the raw q4_k blocks stay resident on the GPU (the 27B q4_k_m fits 36 GB; f16 would not),
	// and the GPU's parallel dequant clears the CPU int8 ceiling (~23 GB/s → 125 GB/s steady).
	// Opt-in (FAK_METAL): it currently keeps the CPU q4kw copy resident too, so on a memory-tight
	// box the GPU upload double-counts — the loader change that drops the CPU copy is the
	// follow-up. The CPU path is byte-faithful, so logits are unchanged within the GPU
	// float-order band (TestMetalQ4KPrefillMatchesCPU). Decode is untouched (a lone GEMV is
	// occupancy-bound; the decode bar needs the one-command-buffer forward, a tracked follow-up).
	MetalQ4K bool

	// PhaseProfiler is an opt-in coarse wall-time profiler used by modelbench to split
	// Qwen3.6 prefill/decode into real execution phases. Nil keeps the hot path free of
	// time.Now calls.
	PhaseProfiler *PhaseProfiler

	// qwen35DecodeGraph is an opt-in per-session production-orchestrator trace.
	qwen35DecodeGraph *qwen35DecodeGraphRecorder

	// q4kExpertStats is opt-in readback for resident-Q4_K MoE decode. It records how many
	// routed experts the Q4_K session saw, and how many actually took the Metal Q6_K-down
	// fused path. The counters are session-local; generation already owns a Session serially.
	q4kExpertStats Q4KExpertStats

	// q4kHybridPrefillChunks is an internal execution marker for the resident Qwen
	// hybrid panel path. It advances only after a whole chunk has appended successfully;
	// the paired base records the position that chunk actually started at. Keeping this
	// separate from Cache.Len lets tests distinguish resident append from a numerically
	// correct token-loop fallback without adding an operator-facing control.
	q4kHybridPrefillChunks   int
	q4kHybridPrefillLastBase int
	// q4kMLPOutputSlab is the optional, session-local host readback backing for one grouped Q4_K
	// gate/up prefill result. Generation owns a Session serially, and each layer consumes gate/up
	// before the next layer overwrites it. It is retained only inside the P<=512, 68 MiB envelope
	// and released by Close; nil preserves the allocating path.
	q4kMLPOutputSlab      []float32
	q4kMLPOutputSlabStats q4kMLPOutputSlabStats

	// tap is an opt-in diagnostic dump hook for a single decode position. Nil on all
	// normal sessions; tests and FAK_HIDDEN_TAP use it to capture hidden-state probes.
	tap       *hiddenTap
	tapActive *hiddenTap

	// glmDsaSharedTopK carries the current token's most recent full-indexer
	// decision across IndexShare layers while tokenHiddenGLMDsa walks the block stack.
	glmDsaSharedTopK []int

	// gemma4Hist is the token history of a gemma4 Session (gemma4_session.go, #5495). The
	// dedicated gemma4 forward is cacheless, so this slice IS the session state: each
	// Prefill/Step appends to it and re-runs the forward over the whole prefix. It is empty
	// on every other architecture, and a gemma4 session leaves Cache untouched — the cached
	// per-layer-window KV path is #5496.
	gemma4Hist []int

	// decodeScores reuses one attention-score buffer across heads AND decode steps. A
	// single Session decodes serially and the per-step head loop is serial, so one buffer
	// (fully overwritten each head) is bit-identical to a fresh make per head. This removes
	// the per-head/per-step `make([]float32, context)` that otherwise made f32 decode
	// allocate O(n²) score bytes over an n-token generation — pure GC-pressure relief, no
	// arithmetic change (TestDecodeStepAllocationStaysBounded guards the bound).
	decodeScores []float32
	v4Expert     v4LiveExpertRuntime
	// closeOnce covers backend resources and the model-weight reference. Legacy sessions also
	// participate because retained no-copy Metal weights borrow model-owned backing.
	closeOnce        sync.Once
	modelWeightsHeld bool
	// mixedQKV is session-owned experimental dispatch state. It is initialized from the
	// explicit selector once per session and zeroed by Close; no mutable package-global owner exists.
	mixedQKV mixedQKVSession

	// targetHidden keeps the exact residual stream immediately before final_norm for
	// positions evaluated by the native f32 token path. It is deliberately session-owned:
	// MTP consumers must never reconstruct this state from logits or embeddings.
	captureTargetHidden bool
	targetHiddenMu      sync.RWMutex
	targetHidden        [][]float32
	targetHiddenTokens  []int
}

// NewSession starts a fresh generation session.
func (m *Model) NewSession() *Session {
	if !m.holdModelWeights() {
		panic("model: weights are closing or closed")
	}
	s := &Session{M: m, Cache: NewKVCache(m.Cfg), modelWeightsHeld: true}
	s.initMixedQKV()
	return s
}

func (s *Session) denseGPULayers() int {
	if s == nil {
		return 0
	}
	if s.DenseGPULayers != 0 && s.GPULayers != 0 && s.DenseGPULayers != s.GPULayers {
		panic(fmt.Sprintf("model: DenseGPULayers=%d and GPULayers=%d conflict", s.DenseGPULayers, s.GPULayers))
	}
	if s.DenseGPULayers != 0 {
		return s.DenseGPULayers
	}
	return s.GPULayers
}

func (s *Session) validateDenseGPULayers() (int, bool) {
	if s == nil || s.M == nil {
		return 0, false
	}
	n := s.denseGPULayers()
	cfg := s.M.Cfg
	if n < 0 || n > cfg.NumLayers {
		panic(fmt.Sprintf("model: DenseGPULayers=%d out of bounds for NumLayers=%d", n, cfg.NumLayers))
	}
	if n > 0 && s.Backend == nil {
		panic(fmt.Sprintf("model: DenseGPULayers=%d requires a non-nil Backend", n))
	}
	if s.Cache == nil {
		s.Cache = NewKVCache(cfg)
	}
	return n, n > 0 && n < cfg.NumLayers
}

func (s *Session) hostMatKernel() matKernel {
	m, cfg := s.M, s.M.Cfg
	if s.Q4 && m.q4w != nil {
		return sessionQ4Kernel{s: s}
	}
	if s.Q4K && m.q4kw != nil {
		return sessionQ4KKernel{s: s}
	}
	if s.Quant {
		if cfg.IsMoE() {
			return q8Kernel{m: m}
		}
		return sessionQ8Kernel{s: s}
	}
	if m.has("model.layers.0.self_attn.q_proj.weight") {
		return f32Kernel{m: m}
	}
	return residentKernel{m: m}
}

// token runs one position through all layers and projects to logits. It is
// tokenHidden (the shared prefill/decode compute) followed by the LM head; kept as
// the decode path (Step) where every step's logits are actually consumed.
// requirePreNorm panics if this session's model uses a non-PreNorm block topology
// on a code path (HAL / Metal / quant-batch) that is a SEAM-0 hand-copy still
// hardcoding the Llama PreNorm wiring. Non-PreNorm topologies run only on the
// topology-aware f32 blockStep / cacheless layer() paths today (MODEL-ARCH-SEAM
// SEAM-0 collapses the remaining copies); this turns a silent wrong result into a
// loud, honest boundary.
func (s *Session) requirePreNorm(path string) {
	if t := s.M.Cfg.BlockTopology; t != PreNorm {
		panic("model: " + path + " does not yet implement BlockTopology " + t.String() + " (only PreNorm); see MODEL-ARCH-SEAM SEAM-0")
	}
	if s.M.Cfg.hasLayerSpecificRopeTheta() {
		panic("model: " + path + " does not yet implement layer-specific RoPE theta; see MODEL-ARCH-SEAM Gemma3 O3")
	}
}

func (s *Session) tappedLogitsAt(pos int, logits []float32) []float32 {
	if tap := s.activeTap(); tap != nil && tap.wants(pos) {
		tap.dumpLogits(pos, logits)
	}
	return logits
}

func (s *Session) token(id, pos int) []float32 {
	s.validateDenseGPULayers()
	if s.Backend != nil {
		s.requirePreNorm("HAL decode")
		return s.tappedLogitsAt(pos, s.tokenHAL(id, pos))
	}
	if s.Q4 {
		return s.tappedLogitsAt(pos, s.headQ4(s.tokenHiddenQ(id, pos)))
	}
	if s.Q4K {
		// Resident Q4_K decode: block matmuls dispatch per name (raw q4_k majority + Q8
		// minority); the LM head is whichever resident format it loaded as, so headResident
		// picks q4k/q8/f32 rather than assuming Q8.
		return s.tappedLogitsAt(pos, s.headResident(s.tokenHiddenQ(id, pos)))
	}
	if s.GPTQ {
		return s.tappedLogitsAt(pos, s.headResident(s.tokenHiddenGPTQ(id, pos)))
	}
	if s.Quant {
		// GPU-resident decode forward (#67): run the whole token — forward + final norm + LM head —
		// in one Metal command buffer and return logits directly (metal_decode.go). Returns nil for a
		// hybrid/MoE model or when the resident path declines, so this is a cheap gate on the CPU path.
		if logits := s.metalDecodeLogitsQ8(id, pos); logits != nil {
			return s.tappedLogitsAt(pos, logits)
		}
		return s.tappedLogitsAt(pos, s.headQ(s.tokenHiddenQ(id, pos)))
	}
	return s.tappedLogitsAt(pos, s.head(s.tokenHidden(id, pos)))
}

func (s *Session) requireGLMDsaSession() {
	// #86 (partial): a compute.Backend is now PERMITTED — the GLM-MoE-DSA forward routes its
	// dense GEMMs (MoE/FFN, projections, head) through the backend (backendKernel) while the DSA
	// index-scoring + sparse-attention + KV stay host-resident (s.Cache.glm). Metal/PrecisionPolicy
	// are still unwired and fail closed.
	if s.Metal || s.PrecisionPolicy != nil {
		panic("model: GLM-MoE-DSA Session: Metal/PrecisionPolicy paths are unwired (CPU resident DSA cache; compute.Backend GEMM offload is allowed)")
	}
	if s.Cache.glm == nil {
		s.Cache.glm = newGLMDsaKVCache(s.M.Cfg)
	}
}

func (s *Session) glmDsaHead(xf []float32) []float32 {
	var out []float32
	if s.Backend != nil {
		// #86 (partial): the vocab projection (the largest single GEMM) runs on the backend.
		// lmHeadMatHAL resolves the resident head weight (untied q8 / f32) + uploads it.
		be := s.Backend
		xt := uploadHostF32Class(be, []int{s.M.Cfg.HiddenSize}, xf, compute.MemoryActivation, "glm-dsa-lm-head-activation")
		out = be.Read(be.MatMul(s.lmHeadMatHAL(), xt))
		be.Free(xt)
	} else if s.Quant {
		out = s.headQ(xf)
	} else {
		out = s.head(xf)
	}
	glmDsaDumpHead(s, out)
	return out
}

// glmDsaDumpHead localizes the GLM-5.2 LM-head bug (#996): the residual dump proved the final
// hidden VARIES per decode step yet the output tokens REPEAT, so the head maps varying residuals to
// the same few tokens. Under FAK_GLMDSA_DUMP it logs (once) which head weight resolved (tied
// embedding vs untied lm_head) + (per call) the top-3 logit ids/values — if a few ids dominate every
// step regardless of the residual, the head weight for those ids is anomalous.
func glmDsaDumpHead(s *Session, logits []float32) {
	if !glmDsaDumpOn || len(logits) == 0 {
		return
	}
	if !s.glmDsaHeadNameLogged {
		s.glmDsaHeadNameLogged = true
		hn := s.M.headName()
		_, tiedInManifest := s.M.q8w["lm_head.weight"]
		_, tiedQ4K := s.M.q4kw["lm_head.weight"]
		fmt.Fprintf(os.Stderr, "GLMDSA_HEAD resolved head=%q hasLMHeadF32=%v hasLMHeadQ8=%v hasLMHeadQ4K=%v tied=%v\n",
			hn, s.M.has("lm_head.weight"), tiedInManifest, tiedQ4K, !s.M.has("lm_head.weight") && !tiedInManifest && !tiedQ4K)
	}
	// top-3 logits
	var i0, i1, i2 int
	for i, v := range logits {
		switch {
		case v > logits[i0]:
			i2, i1, i0 = i1, i0, i
		case v > logits[i1]:
			i2, i1 = i1, i
		case v > logits[i2]:
			i2 = i
		}
	}
	fmt.Fprintf(os.Stderr, "GLMDSA_HEAD top3=[(%d,%.3f) (%d,%.3f) (%d,%.3f)]\n",
		i0, logits[i0], i1, logits[i1], i2, logits[i2])
}

// head applies the (tied) LM head to a post-final-norm hidden vector. Split out from
// token so prefill can run it ONCE: Prefill returns only the last position's logits,
// so computing the 49,152×576 head at every prefill position (its weight, the tied
// embedding, is the single largest tensor at 113 MB) and discarding all but the last
// is pure waste. Skipping it is bit-identical — the head feeds neither the KV cache
// nor any hidden state, only the returned logits — so R2/R3/R14 stay oracle-green.
func (s *Session) head(xf []float32) []float32 {
	t := s.phaseStart()
	logits := parMatRows(s.M.lmHead(), xf, s.M.Cfg.VocabSize, s.M.Cfg.HiddenSize)
	logitScaleInPlace(logits, s.M.Cfg) // Cohere 0.0625 / Gemma2 logit softcap; no-op for Llama
	s.phaseEnd("lm_head_f32", t)
	return logits
}

// rememberTargetHidden records the residual stream only after the target has
// evaluated pos. Re-evaluating a position replaces it and deterministically
// discards every later entry, matching a rewritten target history.
func (s *Session) rememberTargetHidden(pos, token int, hidden []float32) {
	if s == nil || !s.captureTargetHidden || pos < 0 {
		return
	}
	s.targetHiddenMu.Lock()
	defer s.targetHiddenMu.Unlock()
	if pos < len(s.targetHidden) {
		s.targetHidden = s.targetHidden[:pos]
		s.targetHiddenTokens = s.targetHiddenTokens[:pos]
	}
	for len(s.targetHidden) < pos {
		s.targetHidden = append(s.targetHidden, nil)
		s.targetHiddenTokens = append(s.targetHiddenTokens, -1)
	}
	s.targetHidden = append(s.targetHidden, append([]float32(nil), hidden...))
	s.targetHiddenTokens = append(s.targetHiddenTokens, token)
}

// TargetHiddenAt returns a defensive copy of the exact pre-final-norm hidden
// vector captured when committed position pos was evaluated. A stale entry past
// the current cache boundary is never exposed after speculative rollback.
func (s *Session) TargetHiddenAt(pos int) ([]float32, error) {
	if s == nil || !s.captureTargetHidden || s.Cache == nil || pos < 0 || pos >= s.Cache.Len() {
		return nil, fmt.Errorf("model: target hidden position %d is unavailable", pos)
	}
	s.targetHiddenMu.RLock()
	defer s.targetHiddenMu.RUnlock()
	if pos >= len(s.targetHidden) || pos >= len(s.targetHiddenTokens) || len(s.targetHidden[pos]) == 0 ||
		pos >= len(s.Cache.lineage.ids) || s.targetHiddenTokens[pos] < 0 ||
		uint64(s.targetHiddenTokens[pos]) > uint64(^uint32(0)) ||
		uint32(s.targetHiddenTokens[pos]) != s.Cache.lineage.ids[pos] {
		return nil, fmt.Errorf("model: target hidden position %d is unavailable", pos)
	}
	return append([]float32(nil), s.targetHidden[pos]...), nil
}

// TokenEmbedding returns a defensive copy from this session's target model
// embedding table.
func (s *Session) TokenEmbedding(token int) ([]float32, error) {
	if s == nil || s.M == nil || token < 0 || token >= s.M.Cfg.VocabSize {
		return nil, fmt.Errorf("model: token embedding id %d is out of range", token)
	}
	h := s.M.Cfg.HiddenSize
	if h <= 0 {
		return nil, fmt.Errorf("model: token embedding id %d is unavailable", token)
	}
	embed := s.M.embedRows()
	if token >= len(embed)/h {
		return nil, fmt.Errorf("model: token embedding id %d is unavailable", token)
	}
	start := token * h
	return append([]float32(nil), embed[start:start+h]...), nil
}

// tokenHidden runs one position (absolute index pos, embedding-looked-up hidden x)
// through all layers against the cache, appending this position's K/V, and returns
// the post-final-norm hidden vector (NOT yet projected to logits). Immediately before
// finalNorm it captures the exact target residual required by Qwen3.8 MTP; the head
// is applied by the caller.
func (s *Session) tokenHidden(id, pos int) (out []float32) {
	if s.Quant {
		return s.tokenHiddenQ(id, pos)
	}
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	tap := s.activeTap()
	if tap != nil && !tap.wants(pos) {
		tap = nil
	}
	prevTap := s.tapActive
	s.tapActive = tap
	defer func() { s.tapActive = prevTap }()

	embed := m.embedRows()
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)
	scaleEmbedInPlace(x, cfg) // Gemma sqrt(hidden); no-op for Llama

	for l := 0; l < cfg.NumLayers; l++ {
		cos, sin := ropeRowForLayer(cfg, l, pos)
		x = s.blockStep(l, pos, x, cos, sin, f32Kernel{m})
	}
	s.Cache.appendPosition(pos, id)
	if tap != nil {
		tap.writeMeta(cfg, H, pos)
	}
	s.rememberTargetHidden(pos, id, x)
	out = m.finalNorm(x)
	return out
}

// blockStep is the single-position decoder block: pre-attn norm, q/k/v, RoPE, cache
// append, causal GQA, output projection, then SwiGLU MLP. The mat kernel selects f32
// (tokenHidden) vs Q8 (tokenHiddenQ); both share THIS skeleton so the block orchestration
// — the level an architecture axis lives at — exists in exactly one place. Only the
// weight-matmul arithmetic differs by kernel; the RMSNorm, RoPE, GQA, residuals, and
// SwiGLU are the identical f32 math for both, so the f32 path stays bit-exact and the
// Q8 path stays within its own argmax/cosine gate.
func (s *Session) blockStep(l, qpos int, x, cos, sin []float32, mat matKernel) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	p := func(str string) string { return layerName(l, str) }

	// MLP / FFN. The FFN sub-layer is the one architecture axis dispatched here
	// (dense SwiGLU vs MoE). Dense (NumExperts==0) lowers to the same gate/up/down
	// SwiGLU through the same mat kernel; MoE changes only this residual delta and
	// never the attention/cache path above.
	mlpBody := func(xn []float32) []float32 {
		t := s.phaseStart()
		out := m.ffnForLayer(l).apply(m, l, mat.prep(xn), mat)
		s.phaseEnd("mlp_decode", t)
		return out
	}
	runBlock := func(attnBody sublayer) []float32 {
		attnNorm := m.attentionNorms(l)
		mlpNorm := attnNorm
		if cfg.BlockTopology == ParallelResidual {
			mlpNorm = m.parallelMLPNorms(l, attnNorm)
		} else {
			mlpNorm = m.mlpNorms(l)
		}
		composeBlockAtLayer(l, cfg.BlockTopology, x, attnNorm, mlpNorm, eps, cfg, attnBody, mlpBody)
		return x
	}
	if cfg.isLinearAttnLayer(l) {
		s.recordQwen35LayerGraph(l, true)
		if _, resident := mat.(sessionQ4KKernel); resident {
			if out, _, accepted, err := s.tryQwen35MetalDecodeBlock(l, x); accepted {
				if err != nil {
					panic(err)
				}
				invokeResidualHook(cfg, l, out)
				if tap := s.tapActive; tap != nil {
					tap.applySteer(l, qpos, out)
					tap.dumpLayer(l, layerKindLabel(cfg, l), out)
				}
				return out
			}
		}
		out := runBlock(func(xn []float32) []float32 {
			return s.linearAttnStep(l, xn, mat)
		})
		if tap := s.tapActive; tap != nil {
			tap.applySteer(l, qpos, out)
			tap.dumpLayer(l, layerKindLabel(cfg, l), out)
		}
		return out
	}

	s.recordQwen35LayerGraph(l, false)

	// attnBody runs attention on an already-normalized input and returns the raw
	// output-projection result (pre residual/post-norm). It appends THIS position's
	// K/V to the kernel-owned cache exactly as before — the cache writes are part of
	// attention and run once per block regardless of topology.
	attnBody := func(xn []float32) []float32 {
		t := s.phaseStart()
		xp := mat.prep(xn)
		qWidth := nH * hd
		qn, kn, vn := p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")
		var q, gate, kk, vv []float32
		if mixed, handled, err := s.tryMixedQKV(mat, qn, kn, vn, xp, xn, 2*qWidth, w, w); handled {
			if err != nil {
				// Submission transfers ownership to the mixed-QKV call. Retrying the established
				// path here would encode the same projections twice and violate command ownership.
				panic(err)
			}
			q, gate = splitPackedQueryGate(mixed[0], nH, hd)
			kk, vv = mixed[1], mixed[2]
		} else {
			if cfg.AttnOutputGate {
				qf := mat.mul(qn, xp, 2*qWidth, H)
				q, gate = splitPackedQueryGate(qf, nH, hd)
			} else {
				q = mat.mul(qn, xp, qWidth, H)
			}
			kk = mat.mul(kn, xp, w, H)
			vv = mat.mul(vn, xp, w, H)
		}
		s.phaseEnd("full_attn_qkv_proj", t)
		t = s.phaseStart()
		m.applyProjBias(l, q, kk, vv)
		// qk-norm AFTER projection, BEFORE RoPE; no-op for Llama.
		m.applyLayerQKNorm(l, q, kk)
		// RoPE q and k per head at this position, stashing the PRE-RoPE, post-qk-norm K
		// first so a later Evict can reposition this entry in a single rotation.
		if cfg.Alibi {
			s.Cache.Kraw[l] = append(s.Cache.Kraw[l], kk...)
		} else {
			s.ropeRowQK(l, q, kk, cos, sin)
		}
		// append this position's (post-RoPE) K/V to the kernel-owned cache
		s.Cache.K[l] = append(s.Cache.K[l], kk...)
		s.Cache.V[l] = append(s.Cache.V[l], vv...)
		s.phaseEnd("full_attn_qk_norm_rope", t)

		nPos := len(s.Cache.K[l]) / w
		// SWA read-time mask: query (the row just appended, at absolute position qpos)
		// attends only keys whose absolute position is >= qpos-W+1. lo=0 (full causal)
		// when W<0. Keyed off pos[] so it stays correct after an Evict compaction.
		lo := windowLoStep(s.Cache.pos, nPos, qpos, cfg.windowForLayer(l))
		attnOut := make([]float32, nH*hd)
		// One reused scores scratch for all heads this step (lo/nPos are head-independent);
		// grow() keeps amortized total allocation O(n) instead of the O(n²) a per-head make
		// would cost. Fully overwritten per head below, so reuse is bit-identical.
		s.decodeScores = grow(s.decodeScores, nPos-lo)
		t = s.phaseStart()
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[h*hd : (h+1)*hd]
			scores := s.decodeScores
			for j := lo; j < nPos; j++ {
				kh := s.Cache.K[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				scores[j-lo] = dot(qh, kh)*scale + cfg.alibiScoreBias(h, j, nPos)
			}
			softcapInPlace(scores, attnCap)
			m.softmaxAttentionScores(l, h, scores)
			if m.attnObs != nil { // #852: emit the post-softmax row (copy-out, math untouched)
				emitAttnRow(m.attnObs, l, qpos, h, lo, scores)
			}
			out := attnOut[h*hd : (h+1)*hd]
			for j := lo; j < nPos; j++ {
				vh := s.Cache.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				wj := scores[j-lo]
				saxpy(out, vh, wj)
			}
		}
		s.phaseEnd("full_attn_decode", t)
		if cfg.AttnOutputGate {
			t = s.phaseStart()
			for i := 0; i < qWidth; i++ {
				attnOut[i] *= sigmoidf(gate[i])
			}
			s.phaseEnd("full_attn_gate", t)
		}
		t = s.phaseStart()
		ao := mat.prep(attnOut)
		out := mat.mul(p("self_attn.o_proj.weight"), ao, H, nH*hd)
		m.addBiasIfPresent(out, p("self_attn.o_proj.bias"))
		s.phaseEnd("full_attn_o_proj", t)
		return out
	}
	out := runBlock(attnBody)
	if tap := s.tapActive; tap != nil {
		tap.applySteer(l, qpos, out)
		tap.dumpLayer(l, layerKindLabel(cfg, l), out)
	}
	return out
}

// ropeRowQK applies RoPE to one position's q (nH heads) and k (nKV heads) in place,
// stashing the PRE-RoPE k into layer l's Kraw FIRST so KVCache.Evict can reposition a
// survivor in a single rotation. This is the single-row rotate-and-stash that every
// per-position site funnels through (decode f32/Q8, multi-user decode, profiling),
// so a RoPE-convention change lands in one place rather than ~5 hand-copies.
func (s *Session) ropeRowQK(l int, q, k, cos, sin []float32) {
	// stash PRE-RoPE k first (rotation below mutates k in place)
	s.Cache.Kraw[l] = append(s.Cache.Kraw[l], k...)
	ropeRowQKInto(q, k, cos, sin, s.M.Cfg.HeadDim, s.M.Cfg.NumHeads, s.M.Cfg.NumKVHeads)
}

// ropeRowQKInto is the operand-only form: rotate q's nH heads and k's nKV heads in
// place at one position. The Kraw stash is intentionally NOT here — the f32/Q8 decode
// paths stash k BEFORE rotation (ropeRowQK orders that correctly), while a caller that
// stashes pre-RoPE k itself (e.g. a panel path that batched the stash) calls this
// directly after its own append.
func ropeRowQKInto(q, k, cos, sin []float32, hd, nH, nKV int) {
	for h := 0; h < nH; h++ {
		applyRopeRow(q[h*hd:(h+1)*hd], cos, sin)
	}
	for h := 0; h < nKV; h++ {
		applyRopeRow(k[h*hd:(h+1)*hd], cos, sin)
	}
}

// Prefill ingests a prompt and returns the logits of its LAST token (the
// distribution over the first generated token). Each token is placed at the next
// absolute position (Cache.Len()), so a prior Evict() compaction shifts these down.
func (s *Session) Prefill(ids []int) []float32 {
	if len(ids) == 0 {
		return nil
	}
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	s.validateDenseGPULayers()
	if result, used, err := s.tryQwen35SequencePrefill(ids, true); used {
		if err != nil {
			s.failBackendForward(-1, "sequence prefill", err)
		}
		logits := s.Backend.Read(result.Logits)
		s.retireRequestResources()
		return logits
	}
	// Coordinated expert-parallel serve (#4835): announce this forward to the follower ranks
	// and hold the group until it completes, so they replay it and reach the same per-layer
	// AllReduces. Inert (one nil field read) unless a coordinator is installed on rank 0.
	if rel := s.epAnnounce(epOpPrefill, ids); rel != nil {
		defer rel()
	}
	if s.M.Cfg.usesMLAMoELayout() {
		s.requireGLMDsaSession()
		return s.glmDsaHead(s.tokenLoopHidden(s.tokenHiddenGLMDsa, ids))
	}
	if s.M.Cfg.isMiniMaxSparseAttn() {
		// MiniMax-M3 MSA: the incremental cache path runs the lightning-indexer block
		// selection per position over the cached K/V (minimax_m3_session.go), so cached
		// decode/prefix-reuse agree with the cacheless Forward. It must precede the generic
		// MoE token-loop below, which would otherwise run dense GQA on the sparse layers.
		s.requireMiniMaxSession()
		return s.head(s.tokenLoopHidden(s.tokenHiddenMiniMax, ids))
	}
	if s.M.Cfg.isGemma4() {
		// #5495: the DEDICATED gemma4 forward, on whatever resident store the model loaded as
		// (residentMatRows dispatches f32/Q8_0/Q4_K/int4/k-quant/GPTQ by name). It must precede
		// every generic lane below — each of them assumes the scalar cfg.HeadDim, which is the
		// wrong shape on this arch's local layers.
		return s.prefillGemma4(ids)
	}
	if s.Backend != nil {
		// A device session executes the WHOLE forward through the HAL (matWeightHAL now
		// dispatches q8w/q4kw/f32), so the device path must take precedence over the CPU
		// resident-quant lanes below — else a Q4_K device session prefilled on the CPU cache
		// (s.Cache) while decoding from the empty device cache (s.halKV) → garbage (#949).
		s.requirePreNorm("HAL prefill")
		return s.prefillHAL(ids, true)
	}
	if s.Q4 {
		// Resident int4 prefill: the batched Q8 GEMM has no int4 twin yet, so prefill runs
		// the shared per-token blockStep with the int4 kernel. Slower than batched but uses
		// only the resident int4 weights (the lean q4-only mode freed the Q8_0 copy).
		return s.headQ4(s.tokenLoopHidden(s.tokenHiddenQ, ids))
	}
	if s.Q4K {
		// Resident Q4_K prefill (plan P1/P3). For a PreNorm standard-attention model (the
		// q4_k_m regime the plan targets), run the BATCHED q4 GEMM: q4_k_m majority via
		// q4kGemm, Q6_K / normalize-sensitive minority via the proven qGemm8, KV filled in
		// one pass — each weight super-block dequantized once and reused across all P prompt
		// tokens. Architectures the batched q4 lane does not yet cover (MoE / DenseMLP /
		// Alibi / Qwen35-hybrid / non-PreNorm / layer-specific RoPE theta) fall back to the
		// per-token blockStep, exactly as the Q8 token-loop fallback does. The LM head is
		// whichever resident format it loaded as (headResident).
		if !q8PrefillNeedsTokenLoop(s.M.Cfg) {
			return s.headResident(s.prefillBatchedQ4K(ids))
		}
		// Qwen3.5/3.6 hybrid (the q8PrefillNeedsTokenLoop case the generic batched-Q4K lane
		// refuses): batch each layer's projection/MLP GEMMs over the prompt panel while keeping
		// the GDN recurrence, the resident-Q4K twin of the q8Qwen35HybridPrefillOK gate. Closes
		// QWEN36-NATIVE-PERF-PLAN P3's per-token-fallback prefill wall.
		if logits, used := s.tryPrefillQwen35HybridQ4K(ids, true); used {
			return logits
		}
		return s.headResident(s.tokenLoopHidden(s.tokenHiddenQ, ids))
	}
	if s.GPTQ {
		return s.headResident(s.tokenLoopHidden(s.tokenHiddenGPTQ, ids))
	}
	if s.Metal {
		// The Qwen3.6 hybrid (Gated-DeltaNet) cannot run through the generic full-attention
		// Metal prefill, so route it to the hybrid twin — projection/MLP GEMMs batched on the
		// GPU, GDN recurrence on the CPU — instead of tripping requirePreNorm (#71).
		if metalQwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			return s.headQ(s.prefillBatchedMetalQwen35Hybrid(ids))
		}
		s.requirePreNorm("Metal prefill")
		// Prefill projections on the GPU; the head stays the cheap CPU single-token GEMV.
		return s.headQ(s.prefillBatchedMetal(ids))
	}
	if s.PrecisionPolicy != nil {
		return s.prefillDynamic(ids)
	}
	if s.Quant {
		if q8Qwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			return s.prefillQwen35HybridQ(ids)
		}
		if cfg := s.M.Cfg; q8PrefillNeedsTokenLoop(cfg) {
			return s.headQ(s.tokenLoopHidden(s.tokenHiddenQ, ids))
		}
		return s.headQ(s.prefillBatchedQ(ids))
	}
	if q8PrefillNeedsTokenLoop(s.M.Cfg) {
		// The batched f32 GEMM is one-weight-many-rows and still hardcodes the PreNorm
		// block copy and one shared RoPE table. MoE routes each token to its own top-k
		// experts, DenseMLP removes the up-projection, ALiBi replaces RoPE with score
		// bias, non-PreNorm topology changes the residual/norm graph, and Gemma3-style
		// per-layer RoPE theta changes the rotation by layer; these axes run through
		// blockStep here, where the FFN/topology/RoPE dispatch lives.
		return s.prefillTokenLoop(ids)
	}
	// PreNorm (default): batched + parallel, one GEMM over all P tokens instead of
	// GEMV-per-token. Bit-identical to the per-token tokenHidden loop
	// (TestPrefillBatchedMatchesSerial), so the cache it builds is exactly the proven
	// one and R2/R3/R14 stay exact.
	return s.head(s.prefillBatched(ids))
}

// PrefillNoLogits ingests a prompt exactly like Prefill but discards the final-token
// distribution. It is for teacher-forced context growth where the caller already knows the
// next input token and only needs KV state advanced.
func (s *Session) PrefillNoLogits(ids []int) {
	if len(ids) == 0 {
		return
	}
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	s.validateDenseGPULayers()
	if _, used, err := s.tryQwen35SequencePrefill(ids, false); used {
		if err != nil {
			s.failBackendForward(-1, "sequence prefill", err)
		}
		s.retireRequestResources()
		return
	}
	if s.M.Cfg.usesMLAMoELayout() {
		s.requireGLMDsaSession()
		for _, id := range ids {
			s.tokenHiddenGLMDsa(id, s.Cache.Len())
		}
		return
	}
	if s.M.Cfg.isMiniMaxSparseAttn() {
		s.requireMiniMaxSession()
		for _, id := range ids {
			s.tokenHiddenMiniMax(id, s.Cache.Len())
		}
		return
	}
	if s.M.Cfg.isGemma4() {
		// #5495. The dedicated gemma4 forward is cacheless, so advancing state IS appending to
		// the history — there is no K/V to fill and nothing to discard. A later Prefill/Step
		// recomputes over the full prefix, so this is exactly PrefillNoLogits's contract.
		s.gemma4Ingest(ids)
		return
	}
	if s.Backend != nil {
		s.requirePreNorm("HAL prefill")
		s.prefillHAL(ids, false)
		return
	}
	if s.Q4 {
		for _, id := range ids {
			s.tokenHiddenQ(id, s.Cache.Len())
		}
		return
	}
	if s.Q4K {
		if !q8PrefillNeedsTokenLoop(s.M.Cfg) {
			s.prefillBatchedQ4K(ids)
			return
		}
		if _, used := s.tryPrefillQwen35HybridQ4K(ids, false); used {
			return
		}
		for _, id := range ids {
			s.tokenHiddenQ(id, s.Cache.Len())
		}
		return
	}
	if s.GPTQ {
		for _, id := range ids {
			s.tokenHiddenGPTQ(id, s.Cache.Len())
		}
		return
	}
	if s.PrecisionPolicy != nil {
		s.Prefill(ids)
		return
	}
	if s.Metal {
		// Hybrid (Gated-DeltaNet) routes to the hybrid twin; see Prefill above (#71).
		if metalQwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			s.prefillBatchedMetalQwen35Hybrid(ids)
			return
		}
		s.requirePreNorm("Metal prefill")
		s.prefillBatchedMetal(ids)
		return
	}
	if s.Quant {
		if q8Qwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			s.prefillQwen35HybridQNoLogits(ids)
			return
		}
		if q8PrefillNeedsTokenLoop(s.M.Cfg) {
			for _, id := range ids {
				s.tokenHiddenQ(id, s.Cache.Len())
			}
			return
		}
		s.prefillBatchedQ(ids)
		return
	}
	if q8PrefillNeedsTokenLoop(s.M.Cfg) {
		for _, id := range ids {
			s.tokenHidden(id, s.Cache.Len())
		}
		return
	}
	s.prefillBatched(ids)
}

func q8PrefillNeedsTokenLoop(cfg Config) bool {
	return cfg.IsMoE() || cfg.DenseMLP || cfg.Alibi || cfg.IsQwen35Hybrid() || cfg.AttnOutputGate || cfg.BlockTopology != PreNorm || cfg.hasLayerSpecificRopeTheta()
}

// tokenLoopHidden feeds ids through step one at a time, each at the cache's CURRENT
// absolute length (step appends to the cache, so the position advances with it), and
// returns the last token's hidden row. Every resident-quant prefill lane funnels through
// this when its batched GEMM does not cover the architecture; they differ only in which
// per-token step they hand in and which head they run on the result.
func (s *Session) tokenLoopHidden(step func(id, pos int) []float32, ids []int) []float32 {
	var last []float32
	for _, id := range ids {
		last = step(id, s.Cache.Len())
	}
	return last
}

// prefillTokenLoop runs the per-token f32 forward (tokenHidden) over ids in cache order and
// returns the head logits of the final token — the non-batched prefill path taken by the
// arches q8PrefillNeedsTokenLoop selects out of the one-GEMM batched lane.
func (s *Session) prefillTokenLoop(ids []int) []float32 {
	return s.head(s.tokenLoopHidden(s.tokenHidden, ids))
}

// Step decodes one already-chosen token and returns the next-token logits. Quantized
// sessions reuse their logits buffer; consume or copy the returned slice before the next
// quantized Prefill/Step call on the same session.
func (s *Session) Step(id int) []float32 {
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	s.validateDenseGPULayers()
	// Coordinated expert-parallel serve (#4835) — see the note in Prefill.
	if rel := s.epAnnounce(epOpStep, []int{id}); rel != nil {
		defer rel()
	}
	if s.M.Cfg.usesMLAMoELayout() {
		s.requireGLMDsaSession()
		return s.glmDsaHead(s.tokenHiddenGLMDsa(id, s.Cache.Len()))
	}
	if s.M.Cfg.isMiniMaxSparseAttn() {
		s.requireMiniMaxSession()
		return s.head(s.tokenHiddenMiniMax(id, s.Cache.Len()))
	}
	if s.M.Cfg.isGemma4() {
		return s.stepGemma4(id) // #5495 — see Prefill
	}
	if s.Backend != nil {
		s.ensureOpenBackendSession()
		return s.token(id, s.halKV.Len())
	}
	if s.PrecisionPolicy != nil {
		return s.stepDynamic(id)
	}
	return s.token(id, s.Cache.Len())
}

// Generate greedily decodes n tokens after the prompt and returns their ids.
func (s *Session) Generate(prompt []int, n int) []int {
	logits := s.Prefill(prompt)
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		next := argmaxF32(logits)
		out = append(out, next)
		if s.M.Cfg.IsEOS(next) {
			break
		}
		logits = s.Step(next)
	}
	return out
}

func argmaxF32(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}
