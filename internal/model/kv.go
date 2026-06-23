package model

import (
	"math"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// KVCache is the first-class, kernel-OWNED attention state. This is the object the
// whole fusion is about: because it lives in the Go kernel's address space (not
// behind an HTTP boundary in a vLLM process), the context-MMU can EVICT a poisoned
// token span from it (so the model physically cannot attend to that span), and the
// vDSO can reuse a cached prefix — real operations on real K/V tensors, not
// metaphors. Each layer holds K and V as flat row-major [pos * (NumKVHeads*HeadDim)]
// slices; pos[] records each entry's absolute RoPE position so eviction can compact
// the cache and relabel subsequent positions to equal a run that never saw the span.
type KVCache struct {
	cfg    Config
	K      [][]float32 // [layer] -> flat, NumKVHeads*HeadDim per cached position (post-RoPE)
	Kraw   [][]float32 // [layer] -> the SAME entries pre-RoPE, so a span can be repositioned
	V      [][]float32
	pos    []int // absolute position of each cached entry (shared across layers)
	linear *linearAttnCache
	glm    *glmDsaKVCache
}

// NewKVCache allocates an empty cache for a model. Kraw (pre-RoPE K) is kept so that
// eviction can re-derive a survivor's post-RoPE K at its new position in a SINGLE
// rotation (bit-exact to a fresh prefill), rather than composing two rotations
// (which is mathematically equal but drifts ~1e-6 — enough to flip a greedy token).
func NewKVCache(cfg Config) *KVCache {
	return &KVCache{
		cfg:    cfg,
		K:      make([][]float32, cfg.NumLayers),
		Kraw:   make([][]float32, cfg.NumLayers),
		V:      make([][]float32, cfg.NumLayers),
		linear: newLinearAttnCache(cfg),
		glm:    newGLMDsaKVCache(cfg),
	}
}

// Len is the number of cached positions.
func (c *KVCache) Len() int { return len(c.pos) }

// kvStride is the per-position width of one layer's K (and V) row.
func (c *KVCache) kvStride() int { return c.cfg.NumKVHeads * c.cfg.HeadDim }

// Evict removes a contiguous span [from, from+n) of cached positions from EVERY
// layer and compacts the survivors so the cache is byte-for-byte what it would be
// if the span had NEVER been seen. This is the KV-level quarantine primitive.
//
// The subtlety the adversarial review caught: a survivor that came AFTER the evicted
// span had its K rotated by RoPE at its ORIGINAL absolute position; after compaction
// it sits at a lower position, so its cached K must be RE-ROTATED by the position
// delta. RoPE is linear in position (angle = pos·inv_freq), so re-rotating a K vector
// from old position p to new position p' is exactly applying RoPE of (p'-p). Without
// this, end-span eviction passes by luck but MIDDLE-span eviction is silently wrong.
// (V is not rotated, so it never needs fixing.) Returns positions removed.
func (c *KVCache) Evict(from, n int) int {
	if from < 0 || n <= 0 || from >= len(c.pos) {
		return 0
	}
	end := from + n
	if end > len(c.pos) {
		end = len(c.pos)
	}
	if c.cfg.isGLMMoeDsa() {
		return c.evictGLMDsa(from, end)
	}
	if c.linear != nil {
		panic("model: KVCache.Evict does not support Gated-DeltaNet recurrent state")
	}
	w := c.kvStride()
	hd, nKV := c.cfg.HeadDim, c.cfg.NumKVHeads
	for l := 0; l < c.cfg.NumLayers; l++ {
		c.K[l] = append(c.K[l][:from*w], c.K[l][end*w:]...)
		c.Kraw[l] = append(c.Kraw[l][:from*w], c.Kraw[l][end*w:]...)
		c.V[l] = append(c.V[l][:from*w], c.V[l][end*w:]...)
	}
	// c.pos still holds each survivor's ORIGINAL absolute position; its new position
	// is its new index i. Where they differ, re-derive K[i] from the PRE-RoPE Kraw[i]
	// in one rotation at position i — bit-exact to a prefill that never saw the span.
	c.pos = append(c.pos[:from], c.pos[end:]...)
	for i := range c.pos {
		if c.pos[i] != i {
			for l := 0; l < c.cfg.NumLayers; l++ {
				if c.cfg.Alibi {
					copy(c.K[l][i*w:(i+1)*w], c.Kraw[l][i*w:(i+1)*w])
					continue
				}
				cos, sin := ropeRowForLayer(c.cfg, l, i)
				for h := 0; h < nKV; h++ {
					dst := c.K[l][i*w+h*hd : i*w+(h+1)*hd]
					copy(dst, c.Kraw[l][i*w+h*hd:i*w+(h+1)*hd]) // raw (pre-RoPE)
					applyRopeRow(dst, cos, sin)                 // single rotation at new pos
				}
			}
		}
		c.pos[i] = i
	}
	return end - from
}

func (c *KVCache) evictGLMDsa(from, end int) int {
	if c.glm != nil {
		c.glm.evict(c.cfg, from, end)
	}
	c.pos = append(c.pos[:from], c.pos[end:]...)
	for i := range c.pos {
		if c.pos[i] != i && c.glm != nil {
			c.glm.rerotateSurvivor(c.cfg, i)
		}
		c.pos[i] = i
	}
	return end - from
}

// Clone deep-copies the cache so a computed prefix can be REUSED across sessions —
// the vDSO payoff: a shared system-prompt / tool-result prefix's K/V is computed once
// and spliced into the next session, skipping its prefill entirely. Because the copy
// is exact, the reusing session is bit-identical to one that prefilled the prefix.
func (c *KVCache) Clone() *KVCache {
	return c.CloneWithReserve(0)
}

// CloneWithReserve is Clone plus spare per-layer capacity for extra future positions.
// Fleet serving knows the planned decode/result tail length; reserving it here avoids
// re-copying the already-cloned prefix on the first append and on later growth steps.
func (c *KVCache) CloneWithReserve(extraPositions int) *KVCache {
	if extraPositions < 0 {
		extraPositions = 0
	}
	extraFloats := extraPositions * c.kvStride()
	n := &KVCache{
		cfg:    c.cfg,
		K:      make([][]float32, len(c.K)),
		Kraw:   make([][]float32, len(c.Kraw)),
		V:      make([][]float32, len(c.V)),
		pos:    cloneIntsWithReserve(c.pos, extraPositions),
		linear: c.linear.clone(),
	}
	if c.glm != nil {
		n.glm = c.glm.cloneWithReserve(c.cfg, extraPositions)
	}
	for l := range c.K {
		n.K[l] = cloneFloat32WithReserve(c.K[l], extraFloats)
		n.Kraw[l] = cloneFloat32WithReserve(c.Kraw[l], extraFloats)
		n.V[l] = cloneFloat32WithReserve(c.V[l], extraFloats)
	}
	return n
}

// Reserve grows this cache's spare capacity for extra future positions while preserving
// its exact current contents. It does not change Len().
func (c *KVCache) Reserve(extraPositions int) {
	if extraPositions <= 0 {
		return
	}
	extraFloats := extraPositions * c.kvStride()
	c.pos = reserveInts(c.pos, extraPositions)
	for l := range c.K {
		c.K[l] = reserveFloat32(c.K[l], extraFloats)
		c.Kraw[l] = reserveFloat32(c.Kraw[l], extraFloats)
		c.V[l] = reserveFloat32(c.V[l], extraFloats)
	}
	if c.glm != nil {
		c.glm.reserve(c.cfg, extraPositions)
	}
}

func cloneFloat32WithReserve(src []float32, extra int) []float32 {
	dst := make([]float32, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func cloneIntsWithReserve(src []int, extra int) []int {
	dst := make([]int, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func reserveFloat32(src []float32, extra int) []float32 {
	if cap(src) >= len(src)+extra {
		return src
	}
	dst := make([]float32, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func reserveInts(src []int, extra int) []int {
	if cap(src) >= len(src)+extra {
		return src
	}
	dst := make([]int, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func cloneFloat64WithReserve(src []float64, extra int) []float64 {
	dst := make([]float64, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

func reserveFloat64(src []float64, extra int) []float64 {
	if cap(src) >= len(src)+extra {
		return src
	}
	dst := make([]float64, len(src), len(src)+extra)
	copy(dst, src)
	return dst
}

// SessionFromPrefix starts a session whose cache is a clone of an already-computed
// prefix, so only the suffix needs prefilling (real prefix reuse). For
// GLM-MoE-DSA the clone carries the DSA attention/index cache instead of the dense
// GQA K/V rows.
func (m *Model) SessionFromPrefix(prefix *KVCache) *Session {
	return &Session{M: m, Cache: prefix.Clone()}
}

// Session drives generation over a kernel-owned KV cache. Prefill ingests a prompt;
// Step decodes one token. Both share the exact per-token math the verified full
// forward pass uses, so cached decode is provably identical to full prefill.
// GLM-MoE-DSA uses a separate DSA attention/index cache carried inside KVCache.
type Session struct {
	M     *Model
	Cache *KVCache
	// Backend is non-nil when this session is intentionally running through the
	// internal/compute HAL instead of the legacy direct []float32 path. The legacy
	// path stays the default until the full optimized prefill/batch path is adopted.
	Backend compute.Backend
	halKV   compute.KVStore
	// halW memoizes weights staged onto Backend so a device session uploads each weight
	// to VRAM exactly once, not once per token. (On cpu-ref, Upload is identity over the
	// zero-copy host view, so caching changes nothing and the bit-equality gate holds.)
	halW map[string]compute.Tensor
	// halStep counts tokens run through the HAL; the first two warm the device buffer pool
	// + weight cache before a graph-capturing backend starts replaying captured tokens.
	halStep int

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

	// CPUOffloadExperts routes the MoE expert GEMMs (mlp.experts.* and mlp.shared_experts.*)
	// to host RAM while the dense projections + router + attention run on Backend — the
	// llama.cpp `--n-cpu-moe` hybrid. It is the path that lets a Q4_K model whose experts dwarf
	// VRAM (GLM-5.2 Q4_K_M ≈ 424 GB) serve at all: experts live in the 1007 GB host RAM, the
	// every-token dense FLOPs stay on the GPU. Only the GLM-DSA forward honors it today
	// (glmDsaMatKernel, moe_offload.go); with Backend nil it is a no-op (everything already on
	// host) and the forward stays byte-for-byte the resident path. See splitKernel.
	CPUOffloadExperts bool

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

	// glmDsaSharedTopK carries the current token's most recent full-indexer
	// decision across IndexShare layers while tokenHiddenGLMDsa walks the block stack.
	glmDsaSharedTopK []int

	// decodeScores reuses one attention-score buffer across heads AND decode steps. A
	// single Session decodes serially and the per-step head loop is serial, so one buffer
	// (fully overwritten each head) is bit-identical to a fresh make per head. This removes
	// the per-head/per-step `make([]float32, context)` that otherwise made f32 decode
	// allocate O(n²) score bytes over an n-token generation — pure GC-pressure relief, no
	// arithmetic change (TestDecodeStepAllocationStaysBounded guards the bound).
	decodeScores []float32
}

// NewSession starts a fresh generation session.
func (m *Model) NewSession() *Session {
	return &Session{M: m, Cache: NewKVCache(m.Cfg)}
}

// invFreq is the single RoPE inverse-frequency builder. The layer parameter selects
// layer-specific RoPE bases for families such as Gemma3; Llama/Qwen leave the
// per-layer override empty and use one shared table. After the bare Llama inv_freq is
// built, Phi longrope may divide each dimension by its pinned factor and Llama3
// rope_scaling may rescale the frequencies. Defaults are identity, so the Llama path
// is bit-identical.
func invFreq(cfg Config, layer int) []float64 {
	rotaryDim := cfg.rotaryDim()
	half := rotaryDim / 2
	factor := ropeLongFactor(cfg) // nil unless longrope
	theta := cfg.ropeThetaForLayer(layer)
	// HF: inv_freq[j] = 1 / theta^((2j)/dim) where dim is the rotated width. For full
	// rotary dim==head_dim; for Qwen3.5/Qwen3-Next partial rotary the HF default-rope init
	// uses dim = int(head_dim*partial_rotary_factor) (the rotary_dim), not head_dim.
	denom := cfg.HeadDim
	if cfg.IsQwen35Hybrid() || cfg.isMiniMax() {
		// MiniMax-M3 (like Qwen3.5/Qwen3-Next) initializes inv_freq over the rotary_dim
		// (= int(head_dim*partial_rotary_factor)), not the full head_dim: HF computes
		// dim = int(head_dim*partial_rotary_factor) and inv_freq = 1/theta^(2j/dim).
		denom = rotaryDim
	}
	inv := make([]float64, half)
	for j := 0; j < half; j++ {
		inv[j] = 1.0 / math.Pow(theta, float64(2*j)/float64(denom))
		if factor != nil && j < len(factor) && factor[j] != 0 {
			inv[j] /= factor[j]
		}
	}
	applyRopeScaling(cfg, inv)
	return inv
}

func (c Config) rotaryDim() int {
	if c.PartialRotaryFactor <= 0 || c.PartialRotaryFactor >= 1 {
		return c.HeadDim
	}
	dim := int(float64(c.HeadDim) * c.PartialRotaryFactor)
	if dim < 2 {
		dim = 2
	}
	if dim > c.HeadDim {
		dim = c.HeadDim
	}
	if dim%2 != 0 {
		dim--
	}
	return dim
}

type ropeInvKey struct {
	headDim   int
	rotaryDim int
	thetaBits uint64
	layer     int
	// scaling axis: distinct scaling params must not collide in the cache. For Llama
	// (scaling=="") these are zero/empty, so the key reduces to the original triple.
	scaling                       string
	factorBits, lowBits, highBits uint64
	origCtx                       int
	betaFastBits, betaSlowBits    uint64
	truncate                      bool
	truncateSet                   bool
	longropeFactorHash            uint64
}

var ropeInvCache sync.Map // map[ropeInvKey][]float64

func cachedInvFreq(cfg Config, layer int) []float64 {
	rp, _ := cfg.defaultRopeParameters()
	factor := cfg.RopeFactor
	if factor == 0 {
		factor = rp.Factor
	}
	origCtx := cfg.RopeOrigContext
	if origCtx == 0 {
		origCtx = rp.OriginalMaxPositionEmbeddings
	}
	truncate := false
	truncateSet := false
	if rp.Truncate != nil {
		truncate = *rp.Truncate
		truncateSet = true
	}
	key := ropeInvKey{
		headDim:            cfg.HeadDim,
		rotaryDim:          cfg.rotaryDim(),
		thetaBits:          math.Float64bits(cfg.ropeThetaForLayer(layer)),
		layer:              layer,
		scaling:            cfg.RopeScaling,
		factorBits:         math.Float64bits(factor),
		lowBits:            math.Float64bits(cfg.RopeLowFreqFactor),
		highBits:           math.Float64bits(cfg.RopeHighFreqFactor),
		origCtx:            origCtx,
		betaFastBits:       math.Float64bits(rp.BetaFast),
		betaSlowBits:       math.Float64bits(rp.BetaSlow),
		truncate:           truncate,
		truncateSet:        truncateSet,
		longropeFactorHash: ropeFactorHash(cfg),
	}
	if inv, ok := ropeInvCache.Load(key); ok {
		return inv.([]float64)
	}
	inv := invFreq(cfg, layer)
	actual, _ := ropeInvCache.LoadOrStore(key, inv)
	return actual.([]float64)
}

func (c Config) ropeThetaForLayer(layer int) float64 {
	if layer >= 0 && layer < len(c.RopeThetaPerLayer) && c.RopeThetaPerLayer[layer] != 0 {
		return c.RopeThetaPerLayer[layer]
	}
	return c.RopeTheta
}

// ropeFactorHash folds the PINNED longrope rescale vector into the inv-freq cache
// key so two configs that share head_dim+theta but differ in their (pinned) factor
// vector never collide on a cached table. 0 for the non-longrope path, so Llama's
// cache key is unchanged. Because the factor is pinned per session, this hash is
// constant for a session's lifetime — the same property Evict's re-rotation relies
// on to draw the identical inv_freq as the prefill.
func ropeFactorHash(cfg Config) uint64 {
	f := ropeLongFactor(cfg)
	if f == nil {
		return 0
	}
	// FNV-1a over the factor bits.
	h := uint64(1469598103934665603)
	for _, v := range f {
		b := math.Float64bits(v)
		for s := 0; s < 64; s += 8 {
			h ^= (b >> uint(s)) & 0xff
			h *= 1099511628211
		}
	}
	return h
}

// ropeRow returns cos/sin (length head_dim/2) for absolute position p.
func ropeRow(cfg Config, p int) (cos, sin []float32) {
	return ropeRowForLayer(cfg, 0, p)
}

func ropeRowForLayer(cfg Config, layer, p int) (cos, sin []float32) {
	return ropeRowFromInvScaled(cachedInvFreq(cfg, layer), p, cfg.ropeAttentionFactor())
}

func ropeRowFromInv(inv []float64, p int) (cos, sin []float32) {
	return ropeRowFromInvScaled(inv, p, 1)
}

func ropeRowFromInvScaled(inv []float64, p int, scale float64) (cos, sin []float32) {
	cos = make([]float32, len(inv))
	sin = make([]float32, len(inv))
	ropeRowInto(cos, sin, inv, p)
	if scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range cos {
			cos[i] *= s
			sin[i] *= s
		}
	}
	return
}

func ropeRowInto(cos, sin []float32, inv []float64, p int) {
	for j := range inv {
		a := float64(p) * inv[j]
		cos[j] = float32(math.Cos(a))
		sin[j] = float32(math.Sin(a))
	}
}

func applyRopeRow(hv, cos, sin []float32) {
	half := len(cos)
	for j := 0; j < half; j++ {
		a, b := hv[j], hv[j+half]
		// The explicit float32() conversions pin each product to f32 precision, blocking
		// the Go compiler's OPTIONAL fusion of a*c±b*s into an FMA. Without them the same
		// rotation fuses at one inlined call site (prefill) but not another (Evict
		// reposition) on arm64, so a repositioned K drifts 1 ULP from the prefill K and the
		// bit-exact KV rungs fail on Apple Silicon while passing on amd64 (which never
		// fuses). Pinning to the unfused value makes the rotation deterministic on every
		// architecture and call site — the "bit-identical kernel" guarantee the cache,
		// Clone, and Evict reposition all depend on.
		hv[j] = float32(a*cos[j]) - float32(b*sin[j])
		hv[j+half] = float32(b*cos[j]) + float32(a*sin[j])
	}
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

func (s *Session) token(id, pos int) []float32 {
	if s.Backend != nil {
		s.requirePreNorm("HAL decode")
		return s.tokenHAL(id, pos)
	}
	if s.Q4 {
		return s.headQ4(s.tokenHiddenQ(id, pos))
	}
	if s.Q4K {
		// Resident Q4_K decode: block matmuls dispatch per name (raw q4_k majority + Q8
		// minority); the LM head is whichever resident format it loaded as, so headResident
		// picks q4k/q8/f32 rather than assuming Q8.
		return s.headResident(s.tokenHiddenQ(id, pos))
	}
	if s.Quant {
		return s.headQ(s.tokenHiddenQ(id, pos))
	}
	return s.head(s.tokenHidden(id, pos))
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
	if s.Backend != nil {
		// #86 (partial): the vocab projection (the largest single GEMM) runs on the backend.
		// lmHeadMatHAL resolves the resident head weight (untied q8 / f32) + uploads it.
		be := s.Backend
		xt := be.Upload(compute.NewF32(be, []int{s.M.Cfg.HiddenSize}, xf), compute.F32)
		return be.Read(be.MatMul(s.lmHeadMatHAL(), xt))
	}
	if s.Quant {
		return s.headQ(xf)
	}
	return s.head(xf)
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

// tokenHidden runs one position (absolute index pos, embedding-looked-up hidden x)
// through all layers against the cache, appending this position's K/V, and returns
// the post-final-norm hidden vector (NOT yet projected to logits). This is the single
// shared code path for prefill and decode; the head is applied by the caller.
func (s *Session) tokenHidden(id, pos int) []float32 {
	if s.Quant {
		return s.tokenHiddenQ(id, pos)
	}
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize

	embed := m.embedRows()
	x := append([]float32(nil), embed[id*H:(id+1)*H]...)
	scaleEmbedInPlace(x, cfg) // Gemma sqrt(hidden); no-op for Llama

	for l := 0; l < cfg.NumLayers; l++ {
		cos, sin := ropeRowForLayer(cfg, l, pos)
		x = s.blockStep(l, pos, x, cos, sin, f32Kernel{m})
	}
	s.Cache.pos = append(s.Cache.pos, pos)
	return m.finalNorm(x)
}

// matKernel selects which arithmetic a single-position block's weight matmuls run
// through, so the f32 decode block (tokenHidden) and its Q8 twin (tokenHiddenQ) share
// ONE block skeleton — blockStep — instead of two hand-copied transcriptions. prep
// converts a (just-normed) activation into the kernel's preferred operand form ONCE,
// and mul applies a named weight to that prepared operand. The f32 kernel's prep is
// the identity, so its instruction stream is byte-for-byte today's parMatRows path;
// the Q8 kernel's prep quantizes the activation a single time and the q/k/v (and
// gate/up) matmuls reuse it, exactly as the hand-copied tokenHiddenQ did — so neither
// reduction order moves and the bit-equality / Q8 gates both hold.
type matKernel interface {
	// prep turns a normed activation vector into the operand the kernel multiplies
	// against weights (an f32 slice for f32, a quantized vector for Q8). It is called
	// once per distinct activation; the returned handle feeds every mul that shares it.
	prep(x []float32) any
	// mul applies the named weight (out×in, row-major) to a prepared activation handle.
	mul(name string, x any, out, in int) []float32
}

// f32Kernel runs the single-position block in f32: prep is the identity and every
// matmul routes through parMatRows (the row-parallel, in-order fdot GEMV), preserving
// the exact reduction order the bit-exact f32 rungs (R2/R14) certify.
type f32Kernel struct{ m *Model }

func (k f32Kernel) prep(x []float32) any { return x }
func (k f32Kernel) mul(name string, x any, out, in int) []float32 {
	return parMatRows(k.m.tensor(name), x.([]float32), out, in)
}

// q8Kernel runs the single-position block in Q8_0 with an independent q8Vec for each
// prep and qMatRows against the prebuilt Q8_0 weight. The fresh operand allocation is
// intentional: it remains safe for nested callers such as MoE experts that need an
// earlier prepared operand after a later prep.
type q8Kernel struct{ m *Model }

func (k q8Kernel) prep(x []float32) any { return quantizeVecQ8(x) }
func (k q8Kernel) mul(name string, x any, out, in int) []float32 {
	return qMatRows(k.m.q8(name), x.(q8Vec))
}

// sessionQ8Kernel is the allocation-light Q8 fallback kernel for single-position block
// paths where prepared operands are consumed before the next prep. It reuses the same
// session-owned activation quant buffer as headQ and the fast PreNorm decode path.
type sessionQ8Kernel struct{ s *Session }

func (k sessionQ8Kernel) prep(x []float32) any { return k.s.quantizeVecQ8(x) }
func (k sessionQ8Kernel) mul(name string, x any, out, in int) []float32 {
	return qMatRows(k.s.M.q8(name), x.(q8Vec))
}

// residentKernel is the cacheless Forward kernel: prefer the exact f32 tensor when it
// exists, otherwise read the quant-on-load Q8_0 resident copy. That keeps ordinary
// Forward bit-identical while allowing lean models to run without re-inflating dropped
// projection weights.
type residentKernel struct{ m *Model }

func (k residentKernel) prep(x []float32) any { return x }
func (k residentKernel) mul(name string, x any, out, in int) []float32 {
	return k.m.residentMatRows(name, x.([]float32), out, in)
}

// backendKernel routes matKernel GEMMs through a compute.Backend (the GPU pure kernels) instead of
// the host residentMatRows. It is the GLM-MoE-DSA forward's device path (#86, partial): the heavy
// dense GEMMs — MoE/FFN experts + router, and the attention projections — run on the backend, while
// the DSA index-scoring + sparse-attention glue and the KV cache stay host-resident. The activation
// stays host-side (prep returns x); mul uploads the resident weight (q8 -> k_q8_gemm, pure; f32 ->
// the backend's f32 GEMM) and the activation, runs be.MatMul, and reads the result back to host, so
// it is a drop-in for residentMatRows in the host data flow. The device↔host copy per GEMM keeps the
// glue simple (correctness-first); a fully device-resident GLM-DSA forward is the next slice.
type backendKernel struct{ s *Session }

func (k backendKernel) prep(x []float32) any { return x }
func (k backendKernel) mul(name string, x any, out, in int) []float32 {
	s := k.s
	be := s.Backend
	xf := x.([]float32)
	if len(xf) != in {
		panic("model: backendKernel " + name + " activation length mismatch")
	}
	wt := s.glmDsaWeightHAL(name, out, in)
	xt := be.Upload(compute.NewF32(be, []int{in}, xf), compute.F32)
	y := be.Read(be.MatMul(wt, xt))
	if len(y) != out {
		panic("model: backendKernel " + name + " result length mismatch")
	}
	return y
}

// dsaSparseKernel is the OPTIONAL device path for GLM-MoE-DSA's sparse attention compute
// (the per-head softmax(scale·q·k)·V over the host-selected keys). glmDsaAttendCached
// type-asserts its matKernel for it and falls back to the host loop when absent — so only a
// backendKernel whose compute.Backend implements compute.DSASparseBackend (the cpu-ref, and
// the cuda backend via k_dsa_sparse_attend) takes the device path; residentKernel and a
// backend without the capability leave the attention math byte-for-byte host-resident. The
// KEY SELECTION (index scores + top-k) is computed host-side and handed in via the gathered
// selK/selV, so the device's only divergence is the f32 reduction order over the same keys.
type dsaSparseKernel interface {
	// sparseAttend runs the device sparse attention over nSel gathered selected keys/values
	// for one query position. q is [nH*qkHead]; selK [nSel,nH*qkHead]; selV [nSel,nH*vHead].
	// It returns the [nH*vHead] attnConcat and ok=false (caller falls back to host) when the
	// backend does not advertise compute.DSASparseBackend.
	sparseAttend(q, selK, selV []float32, nSel, nH, qkHead, vHead int, scale float32) ([]float32, bool)
}

// sparseAttend routes GLM-DSA's sparse attention to the compute.Backend when it advertises
// DSASparseBackend: it uploads the query + the host-gathered selected K/V rows, runs the
// device op (q8/f32 -> k_dsa_sparse_attend on the cuda backend, the pure GPU kernel), and
// reads the [nH*vHead] result back. A backend without the capability returns ok=false so the
// caller keeps the host loop. Like backendKernel.mul this copies host↔device per call
// (correctness-first); a fully device-resident DSA cache is the next slice.
func (k backendKernel) sparseAttend(q, selK, selV []float32, nSel, nH, qkHead, vHead int, scale float32) ([]float32, bool) {
	be := k.s.Backend
	sb, ok := be.(compute.DSASparseBackend)
	if !ok {
		return nil, false
	}
	qt := be.Upload(compute.NewF32(be, []int{nH * qkHead}, q), compute.F32)
	kt := be.Upload(compute.NewF32(be, []int{nSel * nH * qkHead}, selK), compute.F32)
	vt := be.Upload(compute.NewF32(be, []int{nSel * nH * vHead}, selV), compute.F32)
	out := be.Read(sb.DSASparseAttend(qt, kt, vt, nSel, nH, qkHead, vHead, scale))
	if len(out) != nH*vHead {
		panic("model: backendKernel sparseAttend result length mismatch")
	}
	return out, true
}

// glmDsaWeightHAL uploads a resident GLM projection weight to the backend (cached in halW), mirroring
// residentMatRows's dtype dispatch on the upload side: a q8-resident weight uploads codes+scales
// (k_q8_gemm, pure — needs cuda.go uploadQ8Resident), an f32 weight uploads f32 (backend f32 GEMM).
func (s *Session) glmDsaWeightHAL(name string, out, in int) compute.Tensor {
	if qt, ok := s.M.q8w[name]; ok {
		return s.weightHALQ8(name, qt)
	}
	if s.M.has(name) {
		return s.weightHAL(name)
	}
	panic("model: glmDsaWeightHAL missing resident weight " + name + " (q4_k device GLM-DSA upload is a follow-up)")
}

// residentMatRows reads a projection by name from either the ordinary f32 store
// or the memory-lean Q8 store. It is used by architecture-specific reference
// paths such as GLM-MoE-DSA, where the cacheless/session math is still CPU f32
// but large projection weights may have been quantized at load and dropped from
// raw f32 residency.
func (m *Model) residentMatRows(name string, x []float32, out, in int) []float32 {
	if m.has(name) {
		return matRows(m.tensor(name), x, out, in)
	}
	if qt := m.q8w[name]; qt != nil {
		if qt.out != out || qt.in != in {
			panic("model: q8 tensor shape mismatch: " + name)
		}
		return qMatRows(qt, quantizeVecQ8(x))
	}
	if qt := m.q4w[name]; qt != nil {
		if qt.out != out || qt.in != in {
			panic("model: int4 tensor shape mismatch: " + name)
		}
		return q4MatRows(qt, x)
	}
	if qt := m.q4kw[name]; qt != nil {
		if qt.out != out || qt.in != in {
			panic("model: resident Q4_K tensor shape mismatch: " + name)
		}
		return q4kMatRows(qt, x)
	}
	panic("model: missing tensor " + name)
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
		composeBlock(cfg.BlockTopology, x, attnNorm, mlpNorm, eps, cfg, attnBody, mlpBody)
		return x
	}
	if cfg.isLinearAttnLayer(l) {
		return runBlock(func(xn []float32) []float32 {
			return s.linearAttnStep(l, xn, mat)
		})
	}

	// attnBody runs attention on an already-normalized input and returns the raw
	// output-projection result (pre residual/post-norm). It appends THIS position's
	// K/V to the kernel-owned cache exactly as before — the cache writes are part of
	// attention and run once per block regardless of topology.
	attnBody := func(xn []float32) []float32 {
		t := s.phaseStart()
		xp := mat.prep(xn)
		qWidth := nH * hd
		var q []float32
		var gate []float32
		if cfg.AttnOutputGate {
			qf := mat.mul(p("self_attn.q_proj.weight"), xp, 2*qWidth, H)
			q = make([]float32, qWidth)
			gate = make([]float32, qWidth)
			for h := 0; h < nH; h++ {
				copy(q[h*hd:(h+1)*hd], qf[h*2*hd:h*2*hd+hd])
				copy(gate[h*hd:(h+1)*hd], qf[h*2*hd+hd:h*2*hd+2*hd])
			}
		} else {
			q = mat.mul(p("self_attn.q_proj.weight"), xp, qWidth, H)
		}
		kk := mat.mul(p("self_attn.k_proj.weight"), xp, w, H)
		vv := mat.mul(p("self_attn.v_proj.weight"), xp, w, H)
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
			out := attnOut[h*hd : (h+1)*hd]
			for j := lo; j < nPos; j++ {
				vh := s.Cache.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
				wj := scores[j-lo]
				for d := 0; d < hd; d++ {
					out[d] += wj * vh[d]
				}
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
	return runBlock(attnBody)
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
	if s.M.Cfg.isGLMMoeDsa() {
		s.requireGLMDsaSession()
		var last []float32
		for _, id := range ids {
			last = s.tokenHiddenGLMDsa(id, s.Cache.Len())
		}
		return s.glmDsaHead(last)
	}
	if s.Q4 {
		// Resident int4 prefill: the batched Q8 GEMM has no int4 twin yet, so prefill runs
		// the shared per-token blockStep with the int4 kernel. Slower than batched but uses
		// only the resident int4 weights (the lean q4-only mode freed the Q8_0 copy).
		var last []float32
		for _, id := range ids {
			last = s.tokenHiddenQ(id, s.Cache.Len())
		}
		return s.headQ4(last)
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
		if q4kQwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			return s.prefillQwen35HybridQ4K(ids)
		}
		var last []float32
		for _, id := range ids {
			last = s.tokenHiddenQ(id, s.Cache.Len())
		}
		return s.headResident(last)
	}
	if s.Backend != nil {
		s.requirePreNorm("HAL prefill")
		return s.prefillHAL(ids, true)
	}
	if s.Metal {
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
			var last []float32
			for _, id := range ids {
				last = s.tokenHiddenQ(id, s.Cache.Len())
			}
			return s.headQ(last)
		}
		return s.headQ(s.prefillBatchedQ(ids))
	}
	if s.M.Cfg.IsMoE() || s.M.Cfg.DenseMLP || s.M.Cfg.Alibi || s.M.Cfg.IsQwen35Hybrid() || s.M.Cfg.AttnOutputGate || s.M.Cfg.BlockTopology != PreNorm || s.M.Cfg.hasLayerSpecificRopeTheta() {
		// The batched f32 GEMM is one-weight-many-rows and still hardcodes the PreNorm
		// block copy and one shared RoPE table. MoE routes each token to its own top-k
		// experts, DenseMLP removes the up-projection, ALiBi replaces RoPE with score
		// bias, non-PreNorm topology changes the residual/norm graph, and Gemma3-style
		// per-layer RoPE theta changes the rotation by layer; these axes run through
		// blockStep here, where the FFN/topology/RoPE dispatch lives.
		var last []float32
		for _, id := range ids {
			last = s.tokenHidden(id, s.Cache.Len())
		}
		return s.head(last)
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
	if s.M.Cfg.isGLMMoeDsa() {
		s.requireGLMDsaSession()
		for _, id := range ids {
			s.tokenHiddenGLMDsa(id, s.Cache.Len())
		}
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
		if q4kQwen35HybridPrefillOK(s.M.Cfg, len(ids)) && s.Cache.Len() == 0 {
			s.prefillQwen35HybridQ4KNoLogits(ids)
			return
		}
		for _, id := range ids {
			s.tokenHiddenQ(id, s.Cache.Len())
		}
		return
	}
	if s.Backend != nil {
		s.requirePreNorm("HAL prefill")
		s.prefillHAL(ids, false)
		return
	}
	if s.PrecisionPolicy != nil {
		s.Prefill(ids)
		return
	}
	if s.Metal {
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
	if s.M.Cfg.IsMoE() || s.M.Cfg.DenseMLP || s.M.Cfg.Alibi || s.M.Cfg.IsQwen35Hybrid() || s.M.Cfg.AttnOutputGate || s.M.Cfg.BlockTopology != PreNorm || s.M.Cfg.hasLayerSpecificRopeTheta() {
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

// Step decodes one already-chosen token and returns the next-token logits. Quantized
// sessions reuse their logits buffer; consume or copy the returned slice before the next
// quantized Prefill/Step call on the same session.
func (s *Session) Step(id int) []float32 {
	if s.M.Cfg.isGLMMoeDsa() {
		s.requireGLMDsaSession()
		return s.glmDsaHead(s.tokenHiddenGLMDsa(id, s.Cache.Len()))
	}
	if s.Backend != nil {
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
