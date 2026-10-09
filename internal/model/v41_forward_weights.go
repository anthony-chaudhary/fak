package model

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"math"
)

// v41ProjScratch is the per-forward REUSED materialization target for the two
// grouped output projections (attn.wo_a.weight / attn.wo_b.weight), which the
// forward consumes as whole f32 blocks through V41GroupedOutputProjection. It
// exists to bound the in-kernel warmup's resident set (fak#13288).
//
// Before this scratch, v41Layer read wo_a/wo_b through v41ProjF32, which
// allocates a FRESH whole-tensor f32 block per layer per forward
// (residentF32Mat's make([]float32, out*in)). At the published V4.1 geometry
// that is ~416 MiB/layer (wo_a [1024, 65536] + wo_b [5120, 8192]); across the
// 40-layer stack it is ~16.25 GiB of transient that is charged to no ledger and
// does not appear in the serve memory plan. Go's GC does not reclaim it
// promptly while the streamed-expert tier holds ~43 GiB live, so the warmup's
// RSS grew monotonically to 56.57 GiB against a 43.896 GiB staging-host-charge
// and the kernel OOM-killed the process.
//
// The forward is strictly SEQUENTIAL across layers and calls
// V41GroupedOutputProjection read-only, so the layer-l block is dead before
// layer l+1 begins. Reusing ONE buffer per forward therefore bounds the
// wo_a/wo_b term to a single layer's worth (~416 MiB) instead of the accumulated
// 40-layer churn, with byte-identical arithmetic (the same f32 values land in
// the same layout).
type v41ProjScratch struct {
	woA              []float32
	woB              []float32
	expertGateUp     v41ExpertGateUpFunc
	expertDown       v41ExpertDownFunc
	denseProjection  v41DenseProjectionFunc
	groupedOutput    v41GroupedOutputFunc
	mhcProjection    v41MHCProjectionFunc
	queryNorm        v41QueryNormFunc
	kvNorm           v41KVNormFunc
	ffnNorm          v41FFNNormFunc
	sharedActivation v41SharedActivationFunc

	// exp1/exp3/exp2 are the REUSED materialization targets for one routed
	// expert's three projections (ffn.experts.<e>.w1/w3/w2.weight). They were
	// the dominant remaining uncharged warmup term after #13288: expertWeightF32
	// allocated a FRESH make([]float32, out*in) for EACH faulted projection
	// (~135 MiB/expert at the published V4.1 geometry), and the MoE loop faults
	// three projections per pick, many picks per token, across the 40-layer
	// stack. The MoE loop is strictly SEQUENTIAL and v41SwiGLU consumes the
	// three blocks read-only, so one triple reused across every pick/token/layer
	// bounds the term to one expert's worth instead of the accumulated churn
	// that still drove the warmup RSS ~14 GiB above the plan bound and thrashed
	// the host at trunk 732956939 (#13288).
	exp1 []float32
	exp3 []float32
	exp2 []float32

	// mhc is the REUSED target for mhc.mixes.weight, the last whole-tensor f32
	// materializer in the forward. It is read once per layer at the top of
	// v41Layer and consumed read-only by the mHC split; the layer loop is
	// sequential, so one buffer bounds it to a single layer's worth.
	mhc []float32

	// layerExperts is a LAYER-SCOPED f32 cache of routed-expert projections,
	// keyed by tensor name and bounded by expertLayerCacheBytes. It is reset at the
	// top of every layer (the layer is the reuse unit) and serves repeated reads of
	// the same (expert, projection) within one layer from RAM instead of re-faulting
	// the checkpoint tier (#13296). The MoE loop is sequential and consumes each
	// block read-only, and the store order is per-forward, so no aliasing escapes.
	//
	// layerExpertFIFO is the eviction queue in LEAST-RECENTLY-USED order: a hit
	// (v41LayerCacheGet) refreshes the key's position, so it is an LRU queue, not
	// an insertion-order FIFO (#13294). The prefill's interleaved routed-expert
	// reads would otherwise evict a hot key before the next pass re-reads it.
	layerExperts    map[string][]float32
	layerExpertFIFO []string

	// expertLayerCacheBytes bounds layerExperts. Zero disables the cache (the
	// default path, byte-identical to the pre-#13296 stream).
	expertLayerCacheBytes int64
	layerExpertBytes      int64
}

// V41ExpertLayerCacheDefaultBytes is the FALLBACK bound for the layer-scoped
// routed-expert f32 cache when the tier declares no resident budget. It is sized
// to hold ONE position's distinct routed set at the published V4.1 geometry --
// topK (6) triples, each triple w1+w3+w2 = 3 * 2304 * 5120 * 4 = 135 MiB, so
// ~810 MiB -- which guarantees a repeated token is served entirely from RAM while
// keeping the cache a bounded, layer-scoped transient (reset every layer) rather
// than the accumulated f32 churn #13288 removed. A model with no checkpoint tier
// leaves the cache OFF entirely.
//
// It is EXPORTED so the serve-side memory plan (fak#13300, cmd/fak) can charge
// the SAME bound the runtime allocates: a plan that charged a different cache
// size would reintroduce exactly the plan-vs-runtime disagreement #13288 was.
const V41ExpertLayerCacheDefaultBytes int64 = 810 << 20

// v41ExpertLayerCacheDefaultBytes is the package-internal alias the forward reads;
// it is the one declaration (the exported constant), never a second copy.
const v41ExpertLayerCacheDefaultBytes int64 = V41ExpertLayerCacheDefaultBytes

// v41LayerExpertCacheBudget sizes the layer-scoped routed-expert f32 cache for
// one forward. It is a bounded constant sized to one position's distinct routed
// set at the published geometry (~810 MiB), independent of the tier's resident
// budget: that budget is denominated in QUANTIZED strides (~1/7 the f32 footprint
// at these shapes), so binding an f32 cache to it would introduce a multi-GiB
// uncharged f32 term -- exactly the #13288 failure mode. A model with no
// checkpoint tier reports 0 and leaves the cache OFF, preserving the historical
// byte-for-byte stream.
func (m *Model) v41LayerExpertCacheBudget() int64 {
	if m == nil || m.expertCheckpoint == nil {
		return 0
	}
	return v41ExpertLayerCacheDefaultBytes
}

// V41TransientResidentBytes reports the exact high-water bytes the V4.1 native
// forward holds SIMULTANEOUSLY in its forward-scoped scratch (v41ProjScratch)
// and its layer-scoped routed-expert cache, derived from the loaded config
// geometry. It is a read-only sizing value for the serve memory plan (#13300):
// the buffers below were previously charged to no ledger, so the warmup's RSS
// exceeded the declared plan and the kernel OOM-killed the process (#13288).
//
// The estimated buffers coexist at the layer's peak, so they are summed:
//
//   - woA  = attn.wo_a.weight f32 block: OLoraRank * NumHeads * HeadDim elems
//   - woB  = attn.wo_b.weight f32 block: HiddenSize * (OLoraRank*OGroups) elems
//   - exp1/exp3 = routed expert w1/w3 f32 blocks: MoEIntermediateSize * HiddenSize each
//   - exp2 = routed expert w2 f32 block: HiddenSize * MoEIntermediateSize
//   - mhc  = mhc.mixes.weight f32 block: v41MHCMixWidth * 4*HiddenSize elems
//     (the published flattened four-stream form; the reduced fixture's
//     24 x HiddenSize form is smaller, so charging the flattened width
//     is conservative on both)
//   - layerExperts = the #13296 layer cache, bounded by
//     v41ExpertLayerCacheBudget(): the 810 MiB default when a checkpoint
//     tier is attached, else 0 (cache off).
//
// A non-V4.1 model (no MoE intermediate size, no hidden size, or a
// non-`deepseek41` family) returns 0, so every other serve is byte-for-byte
// unchanged. All arithmetic is overflow-checked: any axis whose product or sum
// would overflow int64 returns math.MaxInt64, so an unrepresentable estimate
// can only refuse MORE, never wrap to a falsely-fitting small number.
func (m *Model) V41TransientResidentBytes() int64 {
	if m == nil {
		return 0
	}
	return m.Cfg.V41TransientResidentBytes(m.v41LayerExpertCacheBudget())
}

// V41TransientResidentBytes returns the config-derived high-water bytes a V4.1
// native forward holds simultaneously in its scratch and (optionally) its
// layer-scoped routed-expert cache. cacheBudget is the layer cache bound in
// bytes (0 = cache off; the #13296 default when a checkpoint tier is attached).
//
// It is exposed on Config, not only on *Model, so the serve memory plan can
// charge the transient from the GGUF header BEFORE the model loads — the same
// plan-vs-runtime disagreement (a charge the plan never saw) is the #13288
// failure class. A non-V4.1 config, or one without a hidden/MoE-intermediate
// geometry, returns 0 so every other serve is byte-for-byte unchanged. All
// arithmetic saturates: an unrepresentable geometry returns math.MaxInt64, so
// it can only refuse MORE, never wrap to a falsely-fitting number.
func (c Config) V41TransientResidentBytes(cacheBudget int64) int64 {
	H := c.HiddenSize
	I := c.MoEIntermediateSize
	if H <= 0 || I <= 0 {
		// Not a V4.1-shaped (MoE, hidden-sized) config: no V4.1 forward runs,
		// so there is no transient to charge. Fail open, byte-for-byte.
		return 0
	}
	if !c.IsDeepSeekV41() {
		return 0
	}
	const maxInt64 = int64(^uint64(0) >> 1)

	// woA: OLoraRank * NumHeads * HeadDim elems.
	woA := v41SatMul(maxInt64, int64(c.OLoraRank), int64(c.NumHeads), int64(c.HeadDim))
	// woB: HiddenSize * (OLoraRank * OGroups) elems.
	oDim := v41SatMul(maxInt64, int64(c.OLoraRank), int64(c.OGroups))
	woB := v41SatMul(maxInt64, int64(H), oDim)
	// exp1 + exp3: two routed-expert w1/w3 blocks of I*H elems each.
	expW13 := v41SatMul(maxInt64, 2, int64(I), int64(H))
	// exp2: one routed-expert w2 block of H*I elems.
	expW2 := v41SatMul(maxInt64, int64(H), int64(I))
	// mhc: v41MHCMixWidth * 4H elems (the published flattened four-stream
	// projection; the reduced fixture's 24 x H form is smaller, so charging the
	// flattened width is conservative on both).
	mhc := v41SatMul(maxInt64, int64(v41MHCMixWidth), 4, int64(H))

	elems := v41SatAdd(maxInt64, woA, woB, expW13, expW2, mhc)
	total := v41SatMul(maxInt64, elems, 4) // f32 bytes
	if cacheBudget > 0 {
		total = v41SatAdd(maxInt64, total, cacheBudget)
	}
	return total
}

// v41SatMul returns the saturating product of its operands, clamping to bound
// on any overflow so an unrepresentable geometry can only refuse MORE.
func v41SatMul(bound int64, vals ...int64) int64 {
	acc := int64(1)
	for _, v := range vals {
		if v == 0 {
			return 0
		}
		if acc > 0 && v > 0 && acc > bound/v {
			return bound
		}
		acc *= v
	}
	return acc
}

// v41SatAdd returns the saturating sum of its operands, clamping to bound on any
// overflow. A negative operand is a geometry bug and clamps to bound as well.
func v41SatAdd(bound int64, vals ...int64) int64 {
	acc := int64(0)
	for _, v := range vals {
		if v < 0 || acc > bound-v {
			return bound
		}
		acc += v
	}
	return acc
}

// v41LayerCacheGet returns a retained f32 block for name, or nil on a miss.
//
// A HIT refreshes the key's recency (it is moved to the most-recently-used end
// of layerExpertFIFO) so v41LayerCachePut's eviction becomes LRU rather than
// FIFO. This is the #13294 frontier lever: the prefill's routed-expert reads are
// interleaved across the token dimension, so the key re-read last pass is the
// one FIFO had evicted first. Re-touching on the read that just used it keeps
// the hot set resident, which is exactly the "raise the bounded resident hit
// fraction" the first-token throughput seam names. The map lookup and promotion
// are observation-only w.r.t. arithmetic: the RETURNED block is the same bytes.
func (s *v41ProjScratch) v41LayerCacheGet(name string) ([]float32, bool) {
	if s == nil || s.layerExperts == nil {
		return nil, false
	}
	w, ok := s.layerExperts[name]
	if !ok {
		return nil, false
	}
	s.v41LayerCacheTouch(name)
	return w, true
}

// v41LayerCacheTouch moves name to the most-recently-used end of the eviction
// queue. It is O(len) over the layer's resident keys (bounded by the cache
// budget / block size, i.e. small) and is called on a cache hit and on the
// insertion path, so the queue always reflects recency, never bare insertion
// order. A name absent from the queue is a no-op (defensive: the map and queue
// are only ever mutated together under the sequential MoE loop).
func (s *v41ProjScratch) v41LayerCacheTouch(name string) {
	if s == nil || len(s.layerExpertFIFO) == 0 {
		return
	}
	// The common case is a touch of a key already at the MRU end (the
	// immediately preceding insert); skip the rebuild then.
	if s.layerExpertFIFO[len(s.layerExpertFIFO)-1] == name {
		return
	}
	for i, v := range s.layerExpertFIFO {
		if v == name {
			s.layerExpertFIFO = append(s.layerExpertFIFO[:i], s.layerExpertFIFO[i+1:]...)
			s.layerExpertFIFO = append(s.layerExpertFIFO, name)
			return
		}
	}
}

// v41LayerCachePut retains name's f32 block, evicting the LEAST-RECENTLY-USED
// entry until the declared byte budget admits it (a hit refreshes recency
// through v41LayerCacheGet, so a re-read key is never evicted ahead of a colder
// one). A block larger than the whole budget is not cached (the caller still
// holds its own copy).
func (s *v41ProjScratch) v41LayerCachePut(name string, w []float32) {
	if s == nil || s.expertLayerCacheBytes <= 0 || len(w) == 0 {
		return
	}
	sz := int64(len(w)) * 4
	if sz > s.expertLayerCacheBytes {
		return
	}
	if s.layerExperts == nil {
		s.layerExperts = map[string][]float32{}
	}
	if _, dup := s.layerExperts[name]; dup {
		// Already resident: refresh recency, do not double-count bytes.
		s.v41LayerCacheTouch(name)
		return
	}
	for s.layerExpertBytes+sz > s.expertLayerCacheBytes && len(s.layerExpertFIFO) > 0 {
		old := s.layerExpertFIFO[0]
		s.layerExpertFIFO = s.layerExpertFIFO[1:]
		if prev, ok := s.layerExperts[old]; ok {
			s.layerExpertBytes -= int64(len(prev)) * 4
			delete(s.layerExperts, old)
		}
	}
	s.layerExperts[name] = w
	s.layerExpertFIFO = append(s.layerExpertFIFO, name)
	s.layerExpertBytes += sz
}

// v41LayerCacheReset drops every retained block. Called at the top of a layer so
// the cache's working set is exactly one layer's routed experts.
func (s *v41ProjScratch) v41LayerCacheReset() {
	if s == nil {
		return
	}
	s.layerExperts = nil
	s.layerExpertFIFO = nil
	s.layerExpertBytes = 0
}

// v41ProjF32Into is the caller-buffer twin of v41ProjF32: it resolves a named
// V4.1 per-layer projection and writes its f32 block into dst, growing dst when
// needed, rather than allocating a fresh whole-tensor block on every call. The
// f32-manifest path COPIES the manifest view into dst (a manifest view aliases
// the model's own m.raw, so it must never be returned as a reusable buffer); the
// quant-store arms dequantize into dst with the exact same loops and block order
// residentF32Mat uses, so the output is byte-identical.
//
// dst is returned with len exactly out*in. A weight absent from every store
// fails closed with the same typed ErrV41ForwardStage naming the tensor that
// #13276 requires (never a panic through residentF32Mat's m.tensor read).
func (m *Model) v41ProjF32Into(l int, leaf string, dst []float32) ([]float32, error) {
	name := layerName(l, leaf)
	grow := func(n int) []float32 {
		if cap(dst) < n {
			return make([]float32, n)
		}
		return dst[:n]
	}
	if m.has(name) {
		// COPY the manifest view into dst, never return it aliasing. m.tensor
		// resolves to an unsafe.Slice over the model's own m.raw backing store
		// (weights.go manifestTensor), so returning it zero-copy would let a
		// later layer's quant dequant (a different store for the same leaf, e.g.
		// layer 0 wo_a = F32 and layer 1 wo_a = Q2_K) reuse the buffer via grow()
		// and OVERWRITE the model's resident weights. The copy is byte-identical
		// to the previous read; only the aliasing is removed.
		view := m.tensor(name)
		return append(grow(len(view))[:0], view...), nil
	}
	if qt := m.kqw[name]; qt != nil {
		qt.ensureRawCPU("V4.1 grouped output projection read")
		bb := qt.kind.blockBytes()
		rowBytes := qt.rowBytes()
		w := grow(qt.out * qt.in)
		for b := 0; b < qt.out; b++ {
			row := qt.raw[b*rowBytes : (b+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				kQuantDequantSuperBlock(w[b*qt.in+j*qt.kind.blockWeights():], row[j*bb:(j+1)*bb], qt.kind)
			}
		}
		return w, nil
	}
	if qt := m.q2w[name]; qt != nil {
		src := dequantQ2Tensor(qt)
		return append(grow(len(src))[:0], src...), nil
	}
	if qt := m.q8w[name]; qt != nil {
		src := dequantQ8Tensor(qt)
		return append(grow(len(src))[:0], src...), nil
	}
	if qt := m.q4kw[name]; qt != nil {
		raw, err := qt.materializeRaw()
		if err != nil {
			return nil, v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
		}
		rowBytes := qt.nblk * q4kBlockBytes
		w := grow(qt.out * qt.in)
		for o := 0; o < qt.out; o++ {
			row := raw[o*rowBytes : (o+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				q4kDequantSuperBlock(w[o*qt.in+j*qkK:], row[j*q4kBlockBytes:(j+1)*q4kBlockBytes])
			}
		}
		return w, nil
	}
	if qt := m.q4w[name]; qt != nil {
		w := grow(qt.out * qt.in)
		for o := 0; o < qt.out; o++ {
			for b := 0; b < qt.nblk; b++ {
				dequantQ4Block(w[o*qt.in+b*qBlk4:], qt.d[o*qt.nblk+b], qt.q[o*qt.nblk*(qBlk4/2)+b*(qBlk4/2):])
			}
		}
		return w, nil
	}
	if qt := m.gptqw[name]; qt != nil {
		w := grow(qt.out * qt.in)
		for o := 0; o < qt.out; o++ {
			gptqDequantRow(w[o*qt.in:], qt, o)
		}
		return w, nil
	}
	return nil, v41StageErr(v41StageAttention, l,
		fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
}

// v41MHCWeightLayout reports how an admitted mhc.mixes.weight is laid out:
// flat=true for the published flattened-four-stream projection (24 x 4H logical,
// or the artifact's 4H x 24 storage transpose), flat=false for the reduced
// fixture's legacy 24 x H single-stream matmul. ok=false means the weight is
// absent or holds neither admitted geometry, so the caller fails closed rather
// than reading a wrong sub-matrix.
func (m *Model) v41MHCWeightLayout(l int) (flat, transposed, ok bool) {
	out, in, present := m.residentShape(layerName(l, "mhc.mixes.weight"))
	if !present {
		return false, false, false
	}
	switch {
	case out == v41MHCMixWidth && in == 4*m.Cfg.HiddenSize:
		return true, false, true // logical [24, 4H]
	case out == 4*m.Cfg.HiddenSize && in == v41MHCMixWidth:
		return true, true, true // stored artifact transpose [4H, 24]
	case out == v41MHCMixWidth && in == m.Cfg.HiddenSize:
		return false, false, true // reduced legacy [24, H]
	default:
		return false, false, false
	}
}

// v41MHCMixF32 reads a layer's mhc.mixes.weight as a full f32 block, resolved
// residency-completely. The reduced fixture carries it in the f32 manifest, and
// that path returns the manifest view unchanged (byte-identical to the pre-#13062
// read); a real quantized serve keeps the Q2_K block in a resident store instead,
// so the f32 manifest view would panic in m.tensor. Reading the resident store
// here is what makes the flattened-four-stream forward reachable for the pinned
// artifact. The returned buffer is in the STORE'S native row-major layout, so a
// caller must consult v41MHCWeightLayout to interpret [24,4H] vs the stored
// [4H,24] transpose. A weight absent from every store fails closed with a typed
// error naming the tensor rather than panicking.
func (m *Model) v41MHCMixF32(l int) ([]float32, error) {
	name := layerName(l, "mhc.mixes.weight")
	if w, ok := m.residentF32Mat(name); ok {
		return w, nil
	}
	return nil, v41StageErr(v41StageMHC, l,
		fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
}

// v41MHCMixF32Into is the caller-buffer twin of v41MHCMixF32: it resolves the
// layer's mhc.mixes.weight residency-completely and writes the f32 block into
// dst, growing dst when needed, rather than materializing a fresh whole-tensor
// block per layer. The layer loop is sequential and the mHC split consumes the
// block read-only, so a single reused buffer bounds this term to one layer's
// worth (#13288 companion seam).
//
// It COPIES on the f32-manifest path for the same reason v41ProjF32Into does: a
// manifest tensor is an unsafe.Slice over the model's own m.raw, and returning it
// as the reusable buffer would alias resident weight memory. A weight absent from
// every store fails closed with the same typed ErrV41ForwardStage naming the
// tensor.
func (m *Model) v41MHCMixF32Into(l int, dst []float32) ([]float32, error) {
	name := layerName(l, "mhc.mixes.weight")
	if w, ok := m.residentF32Mat(name); ok {
		return append(dst[:0], w...), nil
	}
	return nil, v41StageErr(v41StageMHC, l,
		fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
}

// residentF32Mat resolves a named V4.1 matmul weight to a full f32 block from
// whichever store actually carries it: the f32 manifest first (returned
// zero-copy, byte-identical), then the resident quant stores
// (kqw/q2w/q8w/q4kw/q4w/gptqw), dequantizing the store's native row-major layout.
// It reports false when the weight is resident in no store, so a caller can fail
// closed with a typed error naming the tensor instead of panicking through
// m.tensor. This is the read-side twin of residentShape: a quantized serve keeps
// the weight in a store with no manifest entry, so a manifest-only read would
// panic on a weight that admission (residentShape) already admitted.
func (m *Model) residentF32Mat(name string) ([]float32, bool) {
	if m.has(name) {
		return m.tensor(name), true
	}
	if qt := m.kqw[name]; qt != nil {
		qt.ensureRawCPU("mHC mix read")
		bb := qt.kind.blockBytes()
		rowBytes := qt.rowBytes()
		w := make([]float32, qt.out*qt.in)
		for b := 0; b < qt.out; b++ {
			row := qt.raw[b*rowBytes : (b+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				kQuantDequantSuperBlock(w[b*qt.in+j*qt.kind.blockWeights():], row[j*bb:(j+1)*bb], qt.kind)
			}
		}
		return w, true
	}
	if qt := m.q2w[name]; qt != nil {
		return dequantQ2Tensor(qt), true
	}
	if qt := m.q8w[name]; qt != nil {
		return dequantQ8Tensor(qt), true
	}
	if qt := m.q4kw[name]; qt != nil {
		raw, err := qt.materializeRaw()
		if err != nil {
			return nil, false
		}
		rowBytes := qt.nblk * q4kBlockBytes
		w := make([]float32, qt.out*qt.in)
		for o := 0; o < qt.out; o++ {
			row := raw[o*rowBytes : (o+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				q4kDequantSuperBlock(w[o*qt.in+j*qkK:], row[j*q4kBlockBytes:(j+1)*q4kBlockBytes])
			}
		}
		return w, true
	}
	if qt := m.q4w[name]; qt != nil {
		w := make([]float32, qt.out*qt.in)
		for o := 0; o < qt.out; o++ {
			for b := 0; b < qt.nblk; b++ {
				dequantQ4Block(w[o*qt.in+b*qBlk4:], qt.d[o*qt.nblk+b], qt.q[o*qt.nblk*(qBlk4/2)+b*(qBlk4/2):])
			}
		}
		return w, true
	}
	if qt := m.gptqw[name]; qt != nil {
		w := make([]float32, qt.out*qt.in)
		for o := 0; o < qt.out; o++ {
			gptqDequantRow(w[o*qt.in:], qt, o)
		}
		return w, true
	}
	return nil, false
}

// v41ProjF32 reads a per-layer V4.1 projection weight (an attention projection or
// a shared-expert matmul leaf) as a full f32 block, resolved residency-completely
// through the SAME store dispatch v41MHCMixF32 uses: the f32 manifest first, then
// the resident quant stores (kqw/q2w/q8w/q4kw/q4w/gptqw). The reduced fixture
// carries these tensors in the f32 manifest and that path returns the manifest
// view unchanged (byte-identical to the pre-#13276 read); a real quantized serve
// (the pinned Q2_K artifact) keeps them in a resident k-quant store, so the
// manifest-only m.tensor read would panic "model: missing tensor ...".
//
// A projection absent from every store fails closed with a typed
// ErrV41ForwardStage naming the tensor rather than panicking, preserving the
// #13276 contract: the physical strix3 serve must reach a NAMED refusal, never a
// silent mis-shape.
func (m *Model) v41ProjF32(l int, leaf string) ([]float32, error) {
	name := layerName(l, leaf)
	w, ok := m.residentF32Mat(name)
	if !ok {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	return w, nil
}

// v41ProjMatRows is the streaming-safe twin of v41ProjF32 for the per-layer
// linear projections the V4.1 forward applies through matRows. v41ProjF32
// resolves the weight by expanding the WHOLE tensor to f32 (residentF32Mat's
// make([]float32, out*in) at :857/878/888/897), an allocation ~13x the
// compressed Q2_K raw that is charged to no ledger and does not appear in the
// serve memory plan (#2150). On the streamed arm that uncharged per-layer f32
// expansion is transient against the simultaneously-resident streamed-expert
// tier cache, so the in-kernel warmup's resident set grows monotonically until
// the kernel OOM-kills the process.
//
// This helper instead reads the weight in its resident store form through
// residentMatRows (internal/model/kernel.go:263), whose k-quant arm reuses a
// single block-sized scratch buffer (quant_kquant.go:448) instead of
// materializing the tensor. It resolves residency-completely and fails closed
// with the same typed ErrV41ForwardStage the #13276 contract requires when the
// weight is in no store, so a physical serve still reaches a NAMED refusal
// rather than a panic.
//
// The f32-manifest path is byte-identical to v41ProjF32+matRows:
// residentMatRowsBase dispatches m.has(name) to matRows(m.tensor(name), ...),
// exactly the pre-#2150 read.
func (m *Model) v41ProjMatRows(l int, leaf string, x []float32, out, in int) ([]float32, error) {
	return m.v41ProjMatRowsWithProjection(l, leaf, x, out, in, nil)
}

func (m *Model) v41ProjMatRowsWithProjection(l int, leaf string, x []float32, out, in int, project v41DenseProjectionFunc) ([]float32, error) {
	return m.v41ProjectionRows(l, leaf, x, out, in, 1, project)
}

// hasResidentWeight reports whether a named matmul weight is resident in ANY
// store residentMatRows can read (the f32 manifest, the q8w/q4w/q4kw/kqw/q2w/
// GPTQ stores), mirroring residentF32Mat's dispatch without materializing any
// full f32 block. It is the presence predicate that lets v41ProjMatRows fail
// closed with a typed error instead of tripping residentMatRowsBase's panic.
func (m *Model) hasResidentWeight(name string) bool {
	if m.has(name) {
		return true
	}
	if m.kqw != nil {
		if _, ok := m.kqw[name]; ok {
			return true
		}
	}
	if m.q2w != nil {
		if _, ok := m.q2w[name]; ok {
			return true
		}
	}
	if m.q8w != nil {
		if _, ok := m.q8w[name]; ok {
			return true
		}
	}
	if m.q4kw != nil {
		if _, ok := m.q4kw[name]; ok {
			return true
		}
	}
	if m.q4w != nil {
		if _, ok := m.q4w[name]; ok {
			return true
		}
	}
	if m.gptqw != nil {
		if _, ok := m.gptqw[name]; ok {
			return true
		}
	}
	return false
}

// v41ExpertF32 reads one routed-expert projection (ffn.experts.<e>.w1/w3/w2.weight)
// as a full f32 block, resolved residency-completely AND tier-completely. A routed
// expert is deliberately absent from every resident store on the streamed arm --
// the R5 checkpoint tier exists to fault exactly one expert's stride out of a fused
// checkpoint slab only when it is routed -- so a manifest-only m.tensor read panics
// on a tensor the admission (v41AdmitShape, which already falls through to
// m.expertCheckpoint.Has) admitted. Resolution order mirrors
// Session.resolveExpertWeight: the resident stores WIN (byte-identical to the
// pre-tier path), then the checkpoint tier is faulted and its raw bytes dequantized
// to f32 for the reduced all-CPU forward. A projection present in neither the
// resident stores nor the tier fails closed with a typed ErrV41ForwardStage naming
// the tensor rather than panicking, preserving the #13276/#13278 contract that the
// physical serve reaches a NAMED refusal and never a silent mis-shape.
func (m *Model) v41ExpertF32(l int, leaf string) ([]float32, error) {
	name := layerName(l, leaf)
	if w, ok := m.residentF32Mat(name); ok {
		return w, nil
	}
	if m.expertCheckpoint.Has(name) {
		ew, err := m.expertCheckpoint.fault(name)
		if err != nil {
			// Double-wrap: ErrV41ForwardStage keeps the #13276/#13278 named-refusal
			// contract while %w on the inner error keeps a typed tier refusal (the
			// #13280 budget guard) reachable through errors.Is, so a bounded-residency
			// refusal is legible and never masked as a generic missing-tensor stage.
			return nil, v41StageErr(v41StageMoE, l,
				fmt.Errorf("%w: tensor %s: %w", ErrV41ForwardStage, name, err))
		}
		w, err := expertWeightF32(ew)
		if err != nil {
			return nil, v41StageErr(v41StageMoE, l,
				fmt.Errorf("%w: tensor %s: %v", ErrV41ForwardStage, name, err))
		}
		if w != nil {
			return w, nil
		}
	}
	return nil, v41StageErr(v41StageMoE, l,
		fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
}

// expertWeightF32 dequantizes one faulted checkpoint-tier projection to a full f32
// row-major [out,in] block. It accepts exactly the representations the tier stages
// (a Q4_K tensor or a k-quant super-block tensor) and rebuilds the same layout
// residentF32Mat produces for a resident twin, so a tier-served expert computes
// byte-identically to a resident one. A weight in no recognized representation
// returns (nil, nil), which the caller surfaces as the named refusal; a staged
// representation whose checkpoint range cannot be faulted returns the read error
// so the caller can name it rather than compute on a silent zero buffer.
//
// Both branches MATERIALIZE before reading: a tier-faulted Q4_K tensor is staged
// lazily (q4kTensor.lazy) and a k-quant tensor's raw may still be checkpoint-backed,
// so slicing qt.raw straight away would read a zero-length buffer and miscompute.
// This mirrors the materialize-then-dequant order residentF32Mat uses for the
// resident twins (q4kw via q4kTensor.materializeRaw, kqw via ensureRawCPU).
func expertWeightF32(w expertWeight) ([]float32, error) {
	if w.q4 != nil {
		qt := w.q4
		raw, err := qt.materializeRaw()
		if err != nil {
			return nil, err
		}
		rowBytes := qt.nblk * q4kBlockBytes
		out := make([]float32, qt.out*qt.in)
		for o := 0; o < qt.out; o++ {
			row := raw[o*rowBytes : (o+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				q4kDequantSuperBlock(out[o*qt.in+j*qkK:], row[j*q4kBlockBytes:(j+1)*q4kBlockBytes])
			}
		}
		return out, nil
	}
	if w.kq != nil {
		qt := w.kq
		qt.ensureRawCPU("routed-expert tier read")
		bb := qt.kind.blockBytes()
		rowBytes := qt.rowBytes()
		out := make([]float32, qt.out*qt.in)
		for b := 0; b < qt.out; b++ {
			row := qt.raw[b*rowBytes : (b+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				kQuantDequantSuperBlock(out[b*qt.in+j*qt.kind.blockWeights():], row[j*bb:(j+1)*bb], qt.kind)
			}
		}
		return out, nil
	}
	return nil, nil
}

// expertWeightF32Into is the caller-buffer twin of expertWeightF32: it
// dequantizes one faulted checkpoint-tier projection into dst, growing dst when
// needed, rather than allocating a fresh whole-tensor f32 block on every read.
// The dequant loops and block order are the exact ones expertWeightF32 uses, so
// the output is byte-identical; the ONLY difference is where the f32 block
// lives. dst is returned with len exactly out*in.
//
// This is the bounded-allocation form the in-kernel warmup needs (#13288): the
// MoE loop is strictly sequential and consumes each projection read-only, so a
// single reused buffer per projection slot bounds the streamed-experts' per-fault
// f32 materialization to one expert's worth instead of the fresh ~135 MiB block
// expertWeightF32 allocated per fault. A nil/absent representation still returns
// (nil, nil), which the caller surfaces as the named refusal.
func expertWeightF32Into(w expertWeight, dst []float32) ([]float32, error) {
	grow := func(n int) []float32 {
		if cap(dst) < n {
			return make([]float32, n)
		}
		return dst[:n]
	}
	if w.q4 != nil {
		qt := w.q4
		raw, err := qt.materializeRaw()
		if err != nil {
			return nil, err
		}
		rowBytes := qt.nblk * q4kBlockBytes
		out := grow(qt.out * qt.in)
		for o := 0; o < qt.out; o++ {
			row := raw[o*rowBytes : (o+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				q4kDequantSuperBlock(out[o*qt.in+j*qkK:], row[j*q4kBlockBytes:(j+1)*q4kBlockBytes])
			}
		}
		return out, nil
	}
	if w.kq != nil {
		qt := w.kq
		qt.ensureRawCPU("routed-expert tier read")
		bb := qt.kind.blockBytes()
		rowBytes := qt.rowBytes()
		out := grow(qt.out * qt.in)
		for b := 0; b < qt.out; b++ {
			row := qt.raw[b*rowBytes : (b+1)*rowBytes]
			for j := 0; j < qt.nblk; j++ {
				kQuantDequantSuperBlock(out[b*qt.in+j*qt.kind.blockWeights():], row[j*bb:(j+1)*bb], qt.kind)
			}
		}
		return out, nil
	}
	return nil, nil
}

// v41ExpertTripleInto resolves one routed expert's three SwiGLU projections
// (w1/w3/w2) into reusable f32 blocks, serving repeated reads of the SAME
// (expert, projection) within a layer from a layer-scoped f32 cache instead of
// re-faulting the R5 checkpoint tier (#13296).
//
// The pre-fix prefill faulted the tier per (token, pick): with top-6 routing over
// a 30-token prefill that is up to 30*6*3 = 540 tier reads at a layer, many of
// them the SAME expert projection the token dimension had already read. The
// tier's bounded host cache absorbs those repeats ONLY while the interleaved
// working set fits its bound; on the published artifact the routed union (~126
// GiB in f32-many strides) exceeds the resident bound (~43 GiB), so the per-token
// order thrashes it and every position re-reads the same slab (measured at bound
// < union: tier Reads scales linearly with token count, Hits stay 0). Retaining
// the materialized f32 triple for the ONE layer in flight makes the second and
// later reads of a repeated expert a RAM hit at ANY tier bound, so the layer's
// tier Reads are bounded by its distinct routed set, independent of token count.
//
// The cache is layer-scoped by construction (reset at the top of v41Layer), and
// the three blocks are handed back for a read-only v41SwiGLU, so no aliasing
// escapes the layer. On a cache miss the read is the historical
// v41ExpertF32Into (resident stores first, then the tier), so a model with no
// tier and no cache budget keeps the pre-#13296 stream byte-for-byte. The values
// handed to v41SwiGLU are byte-identical either way.
//
// Attribution (#13294 DoD item 1): a cache hit is a routed-expert read resolved
// from residency, so it notes a ResidentHit exactly as a resident-store hit does
// (v41ExpertF32Into). Every read the contraction issues is therefore accounted
// for as either a tier fault or a residency hit, and the phase's
// ResidentHitFraction reflects the #13296 cache's real contribution.
//
// retain decides whether the materialized triple is COPIED back into the
// layer-scoped cache. It exists because retention is load-bearing only on the
// TOKEN-MAJOR stream, where a repeated expert across the token dimension is
// re-read and the cache serves it. The EXPERT-MAJOR grouped contraction (#13304,
// v41ContractRoutedGrouped) materializes each expert's triple exactly ONCE for
// the whole panel and never re-reads it -- the already-landed #13294 witness
// (v41_prefill_expert_union_test.go) states the layer cache "records zero hits on
// this path" -- so retaining a triple there is a dead f32 copy of the entire
// activated working set on the prefill first-token path. A grouped caller passes
// retain=false. The read side is unchanged: a cache GET still happens either way
// and still notes its ResidentHit, so the decode/step callers (retain=true) keep
// the historical byte-for-byte behavior. A false retain is fail-SAFE: if a future
// caller ever re-read the same expert within one layer, the miss path re-resolves
// the identical bytes (v41ExpertF32Into), so the only consequence is a missed
// optimization, never a wrong value.
func (m *Model) v41ExpertTripleInto(l int, stem string, scratch *v41ProjScratch, retain bool) (w1, w3, w2 []float32, err error) {
	leaves := [3]string{".w1.weight", ".w3.weight", ".w2.weight"}
	for i, leaf := range leaves {
		name := layerName(l, stem+leaf)
		if w, ok := scratch.v41LayerCacheGet(name); ok {
			// A layer-cache hit is a routed-expert read served from residency (it
			// performs zero tier IO and zero dequant), so it counts in the SAME
			// ledger bucket as a resident-store hit (#13294 DoD item 1). Without
			// this note the read lands in NEITHER bucket and the phase's
			// ResidentHitFraction denominator drops every cache hit, understating
			// the residency the #13296 cache provides -- exactly the number the
			// first-token throughput frontier is steered by.
			m.v41NoteExpertResidentHit()
			switch i {
			case 0:
				w1 = w
			case 1:
				w3 = w
			case 2:
				w2 = w
			}
			continue
		}
		var dst []float32
		switch i {
		case 0:
			dst = scratch.exp1
		case 1:
			dst = scratch.exp3
		case 2:
			dst = scratch.exp2
		}
		w, rerr := m.v41ExpertF32Into(l, stem+leaf, dst)
		if rerr != nil {
			return nil, nil, nil, v41StageErr(v41StageMoE, l, rerr)
		}
		switch i {
		case 0:
			scratch.exp1 = w
			w1 = w
		case 1:
			scratch.exp3 = w
			w3 = w
		case 2:
			scratch.exp2 = w
			w2 = w
		}
	}
	// Retain copies of the materialized triple in the layer cache so the next
	// token that routes this expert is a RAM hit. Copies are required because the
	// scratch exp1/exp3/exp2 buffers are reused for the NEXT expert; the caller's
	// w1/w3/w2 keep pointing at the live scratch, never at the retained copies.
	// The expert-major grouped caller passes retain=false: it materializes each
	// expert once and never re-reads it, so the copy would be dead (see the
	// function doc comment).
	if retain {
		m.v41CacheExpertTriple(l, stem, scratch, w1, w3, w2)
		v41NoteExpertTripleRetention()
	}
	return w1, w3, w2, nil
}

// hostExpertDown resolves ONLY the routed expert's down projection (w2) into the
// reusable scratch buffer for the device gate/up seam (#13358): the gate and up
// projections ran on the backend, so materializing them on the host is exactly
// the uncharged f32 expansion this seam removes. It reuses the same
// residency-complete, fail-closed resolution the full triple uses
// (v41ExpertF32Into: resident stores first, then the R5 checkpoint tier), so a
// projection present in neither still refuses with the same typed
// ErrV41ForwardStage naming the tensor.
//
// #13516: the resolved w2 is RETAINED in the layer-scoped cache exactly as the
// host triple arm retains its triple (v41ExpertTripleInto -> v41CacheExpertTriple).
// Without this retention the device-handled grouped arm re-faulted the SAME
// expert's down projection once per (token, slot) row: the host arm materializes
// one triple per expert group, but this seam resolved w2 per row and never put
// it back, so under the streamed/checkpoint regime (tier hit_fraction=0.00) each
// pick paid a fresh checkpoint fault -- the 2.1x prefill regression #13516. The
// cached copy is a COPY because scratch.exp2 is reused for the next expert; the
// caller's returned slice keeps pointing at the live scratch, never the retained
// copy, so no aliasing escapes the layer.
func (m *Model) hostExpertDown(l int, stem string, scratch *v41ProjScratch) ([]float32, error) {
	name := layerName(l, stem+".w2.weight")
	if w, ok := scratch.v41LayerCacheGet(name); ok {
		m.v41NoteExpertResidentHit()
		return w, nil
	}
	w, rerr := m.v41ExpertF32Into(l, stem+".w2.weight", scratch.exp2)
	if rerr != nil {
		return nil, v41StageErr(v41StageMoE, l, rerr)
	}
	scratch.exp2 = w
	// Retain a copy for the rest of the layer so the next row of the SAME expert
	// group is a RAM hit: one fault per expert group, not one per pick.
	m.v41CacheExpertTriple(l, stem, scratch, nil, nil, w)
	return w, nil
}

// v41CacheExpertTriple retains copies of one expert's three f32 projections in
// the layer-scoped cache (see v41ExpertTripleInto). Copies are required because
// the scratch buffers are reused for the next expert; a cached slice returned
// from a later get must not alias live scratch.
func (m *Model) v41CacheExpertTriple(l int, stem string, scratch *v41ProjScratch, w1, w3, w2 []float32) {
	for i, w := range [][]float32{w1, w3, w2} {
		if len(w) == 0 {
			continue
		}
		leaf := [3]string{".w1.weight", ".w3.weight", ".w2.weight"}[i]
		name := layerName(l, stem+leaf)
		if _, ok := scratch.v41LayerCacheGet(name); ok {
			continue
		}
		// Cache hits must still refresh recency when retention is disabled.
		// On a miss, avoid a copy that v41LayerCachePut would discard.
		if scratch == nil || scratch.expertLayerCacheBytes <= 0 {
			continue
		}
		cp := make([]float32, len(w))
		copy(cp, w)
		scratch.v41LayerCachePut(name, cp)
	}
}

// v41ExpertF32Into is the caller-buffer twin of v41ExpertF32: it resolves one
// routed-expert projection residency-completely (resident stores first, then the
// R5 checkpoint tier) and writes the f32 block into dst instead of allocating a
// fresh one. It preserves every contract of v41ExpertF32 -- the resident stores
// WIN byte-for-byte, a tier fault dequantizes byte-identically, and a projection
// present in neither fails closed with the same typed ErrV41ForwardStage naming
// the tensor (never a panic).
//
// The resident path COPIES into dst rather than returning the store's slice: a
// resident f32-manifest tensor resolves to an unsafe.Slice over the model's own
// backing memory (weights.go manifestTensor), so returning it as a reusable
// buffer would let a later read (a different store for the same leaf) dequant
// over it and silently overwrite model weights -- the exact aliasing bug the
// adversarial review of #13288 found and 0480cf446 fixed. Copying removes the
// alias with byte-identical values.
//
// Phase attribution (#13294 DoD item 1): each resolution is also attributed to
// the phase the session entry point declared -- a resident-store read counts a
// ResidentHit (zero tier IO, zero dequant), a tier fault counts
// Faults/FaultedBytes/DequantBytes. The tier arm's FaultedBytes is the entry
// stride the tier moved, reported here as len(ew.q4.raw)/len(ew.kq.raw) AFTER
// materialization -- the same byte count ExpertCheckpointStats.BytesRead
// records for the read -- and DequantBytes is len(out)*4, the f32 block the
// dequant materialized (not a copy: the dequant BLK-COUNT math itself is
// untouched). All notes are inert no-ops unless a phase was set.
func (m *Model) v41ExpertF32Into(l int, leaf string, dst []float32) ([]float32, error) {
	name := layerName(l, leaf)
	if w, ok := m.residentF32Mat(name); ok {
		m.v41NoteExpertResidentHit()
		return append(dst[:0], w...), nil
	}
	if m.expertCheckpoint.Has(name) {
		// #13299: time the fault door (the tier range read) separately from the
		// f32 materialization below, so the ledger can attribute the physical
		// 492 s first token (fak#13294) to disk wait vs dequant. Inert when no
		// clock is installed (v41NowNanos == 0).
		doorOpen := m.v41NowNanos()
		ew, err := m.expertCheckpoint.fault(name)
		if err != nil {
			return nil, v41StageErr(v41StageMoE, l,
				fmt.Errorf("%w: tensor %s: %w", ErrV41ForwardStage, name, err))
		}
		if doorOpen != 0 {
			m.v41NoteExpertFaultDoorNanos(m.v41NowNanos() - doorOpen)
		}
		faultedBytes := faultedRawBytes(ew)
		dequantOpen := m.v41NowNanos()
		w, err := expertWeightF32Into(ew, dst)
		if err != nil {
			return nil, v41StageErr(v41StageMoE, l,
				fmt.Errorf("%w: tensor %s: %v", ErrV41ForwardStage, name, err))
		}
		if dequantOpen != 0 {
			m.v41NoteExpertDequantNanos(m.v41NowNanos() - dequantOpen)
		}
		if w != nil {
			m.v41NoteExpertTierFault(faultedBytes, int64(len(w))*4)
			return w, nil
		}
	}
	return nil, v41StageErr(v41StageMoE, l,
		fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
}

// faultedRawBytes reports the raw CHECKPOINT byte stride one faulted
// projection's weight carries after materialization -- the tier's per-fault IO
// unit (FaultedBytes here, BytesRead on ExpertCheckpointStats). It counts only
// the staged representations (q4 / k-quant); anything else reports 0, which a
// successful dequant never produces because expertWeightF32Into returns nil
// for it.
func faultedRawBytes(w expertWeight) int64 {
	if w.q4 != nil {
		return int64(len(w.q4.raw))
	}
	if w.kq != nil {
		return int64(len(w.kq.raw))
	}
	return 0
}

// v41MHCProjectFull executes the published flattened-four-stream mHC mix
// projection: the four width-H streams are laid end to end into xflat (4H), one
// shared RMS scale 1/sqrt(mean(xflat^2)+eps) is computed over the whole flattened
// residual, and the 24 mix coefficients are the linear projection of xflat scaled
// by that rsqrt. It mirrors the pinned reference oracle
// (v4_flash_oracle_test.go oracleV4FlashMHCProjection). transposed selects the
// admitted storage orientation: false reads the logical row-major [24, 4H]
// (mixes[m] = sum_i w[m*4H+i]*xflat[i]); true reads the artifact's stored
// input-major [4H, 24] transpose (mixes[m] = sum_i w[i*24+m]*xflat[i]).
//
// It validates its own inputs so a wrong-width or wrong-stream-count residual
// cannot masquerade as the flattened geometry and be silently mis-projected.
func v41MHCProjectFull(wMix []float32, streams [][]float32, H int, eps float32, transposed bool) ([]float32, error) {
	flatWidth := 4 * H
	if len(streams) != 4 || H <= 0 {
		return nil, fmt.Errorf("model: V41 full mHC projection wants 4 streams of width %d, got %d", H, len(streams))
	}
	for i, s := range streams {
		if len(s) != H {
			return nil, fmt.Errorf("model: V41 full mHC projection stream %d width %d, want %d", i, len(s), H)
		}
	}
	if len(wMix) != v41MHCMixWidth*flatWidth {
		return nil, fmt.Errorf("model: V41 full mHC projection weight has %d values, want %d", len(wMix), v41MHCMixWidth*flatWidth)
	}
	xflat := make([]float32, 0, flatWidth)
	for _, s := range streams {
		xflat = append(xflat, s...)
	}
	var ss float32
	for _, v := range xflat {
		ss += v * v
	}
	rsqrt := float32(1 / math.Sqrt(float64(ss/float32(len(xflat))+eps)))
	mixes := make([]float32, v41MHCMixWidth)
	for m := 0; m < v41MHCMixWidth; m++ {
		var s float32
		if transposed {
			for i := 0; i < flatWidth; i++ {
				s += wMix[i*v41MHCMixWidth+m] * xflat[i]
			}
		} else {
			row := wMix[m*flatWidth : (m+1)*flatWidth]
			for i := 0; i < flatWidth; i++ {
				s += row[i] * xflat[i]
			}
		}
		mixes[m] = s * rsqrt
	}
	return mixes, nil
}

type v41DenseProjectionOutcome uint8

const (
	v41ProjectionDeclined v41DenseProjectionOutcome = iota
	v41ProjectionHandled
	v41ProjectionError
)

type v41DenseProjectionFunc func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error)

type V41ProjectionOperationError struct {
	Layer int
	Leaf  string
	Stage string
	Cause error
}

func (e *V41ProjectionOperationError) Error() string {
	return fmt.Sprintf("model: selected V4.1 projection %s failed at layer %d: %v", e.Leaf, e.Layer, e.Cause)
}
func (e *V41ProjectionOperationError) Unwrap() error                     { return e.Cause }
func (e *V41ProjectionOperationError) SelectedProjectionOperation() bool { return true }

var errV41ProjectionResult = fmt.Errorf("%w: invalid selected V4.1 projection result", ErrV41ForwardStage)

func v41CompressorProjectionLeaf(leaf string) bool {
	return leaf == "attn.compressor.wkv.weight" || leaf == "attn.compressor.wgate.weight"
}

func v41ProjectionOperationErr(l int, leaf string, cause error) error {
	if cause == nil {
		cause = errV41ProjectionResult
	}
	stage := v41StageAttention
	if v41CompressorProjectionLeaf(leaf) {
		stage = v41StageCompress
	}
	if v41IndexerProjectionLeaf(leaf) {
		stage = v41StageIndexer
	}
	if l == -1 {
		stage = v41StageHead
	}
	if strings.HasPrefix(leaf, "ffn.") || leaf == "ffn_norm.weight" {
		stage = v41StageMoE
	}
	return &V41ProjectionOperationError{Layer: l, Leaf: leaf, Stage: string(stage), Cause: v41StageErr(stage, l, cause)}
}

func (m *Model) v41ProjectionRows(l int, leaf string, panel []float32, out, in, rows int, project v41DenseProjectionFunc) ([]float32, error) {
	name := layerName(l, leaf)
	stage := v41StageAttention
	compressor := v41CompressorProjectionLeaf(leaf)
	indexer := v41IndexerProjectionLeaf(leaf)
	if indexer {
		stage = v41StageIndexer
	}
	if compressor {
		stage = v41StageCompress
	}
	head := l == -1 && leaf == m.headName()
	if head {
		name = leaf
		stage = v41StageHead
	}
	inputN, inputOK := checkedMulInt(rows, in)
	outputN, outputOK := checkedMulInt(rows, out)
	_, weightOK := checkedMulInt(out, in)
	if rows <= 0 || out <= 0 || in <= 0 || !inputOK || !outputOK || !weightOK || len(panel) != inputN {
		return nil, v41StageErr(stage, l, fmt.Errorf("%w: invalid projection dimensions for %s", ErrV41ForwardStage, name))
	}
	wr, wc, present := m.residentShape(name)
	if head {
		wr, wc, present = m.v41HeadWeightShape(name)
	}
	if !present || wr != out || wc != in {
		return nil, v41StageErr(stage, l, fmt.Errorf("%w: invalid resident projection shape for %s", ErrV41ForwardStage, name))
	}
	if project != nil {
		y, outcome, cause := project(l, leaf, panel, out, in, rows)
		switch outcome {
		case v41ProjectionHandled:
			if cause != nil {
				return nil, v41ProjectionOperationErr(l, leaf, cause)
			}
			if len(y) != outputN {
				return nil, v41ProjectionOperationErr(l, leaf, errV41ProjectionResult)
			}
			return y, nil
		case v41ProjectionError:
			return nil, v41ProjectionOperationErr(l, leaf, cause)
		case v41ProjectionDeclined:
		default:
			return nil, v41ProjectionOperationErr(l, leaf, errV41ProjectionResult)
		}
	}
	opened := m.v41NowNanos()
	succeeded := 0
	defer func() {
		if head {
			m.v41NoteHeadProjection(0, 1, 0, succeeded, 0, 0, opened)
		} else if indexer {
			m.v41NoteIndexerProjection(0, 1, 0, succeeded, 0, 0, opened)
		} else if compressor {
			m.v41NoteCompressorProjection(0, 1, 0, succeeded, 0, 0, opened)
		} else {
			m.v41NoteDenseProjection(0, 1, 0, succeeded, 0, 0, opened)
		}
	}()
	var y []float32
	if head && m.v41PackedHead(name) != nil {
		var err error
		y, err = m.v41PackedHeadRows(name, panel, out, in, rows)
		if err != nil {
			return nil, v41StageErr(v41StageHead, -1, err)
		}
	} else if rows == 1 {
		y = m.residentMatRows(name, panel, out, in)
	} else if m.prism != nil && m.prism.weightWidth[name] != 0 {
		y = make([]float32, outputN)
		for row := 0; row < rows; row++ {
			copy(y[row*out:(row+1)*out], m.residentMatRows(name, panel[row*in:(row+1)*in], out, in))
		}
	} else {
		y = m.residentMatMulBatch(name, panel, out, in, rows)
	}
	succeeded = rows
	return y, nil
}

func (s *Session) v41DenseProjectionFunc() v41DenseProjectionFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	return func(l int, leaf string, panel []float32, out, in, rows int) (result []float32, outcome v41DenseProjectionOutcome, cause error) {
		head := l == -1 && leaf == s.M.headName()
		compressor := v41CompressorProjectionLeaf(leaf)
		indexer := v41IndexerProjectionLeaf(leaf)
		guarded := compressor || indexer
		if !head {
			switch leaf {
			case "indexer.wq_b.weight", "indexer.wk.weight", "indexer.weights_proj.weight", "attn.compressor.wkv.weight", "attn.compressor.wgate.weight", "attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight", "ffn.gate.weight", "ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight", "ffn.shared_experts.w2.weight":
			default:
				return nil, v41ProjectionDeclined, nil
			}
		}
		name := layerName(l, leaf)
		if head {
			name = leaf
			n, ok := checkedMulInt(rows, in)
			_, wok := checkedMulInt(out, in)
			_, yok := checkedMulInt(rows, out)
			wr, wc, present := s.M.v41HeadWeightShape(name)
			if rows <= 0 || out <= 0 || in <= 0 || !ok || !wok || !yok || len(panel) != n || !present || wr != out || wc != in {
				return nil, v41ProjectionError, s.v41HeadCloseFailure("payload", errV41ProjectionResult)
			}
			for _, v := range panel {
				if !finite32(v) {
					return nil, v41ProjectionError, s.v41HeadCloseFailure("payload", errV41ProjectionResult)
				}
			}
		}
		if guarded {
			n, ok := checkedMulInt(rows, in)
			_, wok := checkedMulInt(out, in)
			_, yok := checkedMulInt(rows, out)
			wr, wc, present := s.M.residentShape(name)
			if rows <= 0 || out <= 0 || in <= 0 || !ok || !wok || !yok || len(panel) != n || !present || wr != out || wc != in {
				return nil, v41ProjectionError, s.v41DenseLeafCloseFailure(l, leaf, "payload", errV41ProjectionResult)
			}
			for _, v := range panel {
				if !finite32(v) {
					return nil, v41ProjectionError, s.v41DenseLeafCloseFailure(l, leaf, "payload", errV41ProjectionResult)
				}
			}
		}
		var stage func() compute.Tensor
		dtype := compute.F32
		switch {
		case s.M.has(name):
			stage = func() compute.Tensor { return s.weightHAL(name) }
		case s.M.q8w[name] != nil:
			dtype = compute.Q8_0
			if head || guarded {
				if in%qBlk != 0 {
					return nil, v41ProjectionDeclined, nil
				}
			}
			stage = func() compute.Tensor { return s.weightHALQ8(name, s.M.q8w[name]) }
		case s.M.q4kw[name] != nil:
			dtype = compute.Q4_K
			if head || guarded {
				q := s.M.q4kw[name]
				if in%qkK != 0 || q.nblk != in/qkK {
					return nil, v41ProjectionDeclined, nil
				}
			}
			stage = func() compute.Tensor { return s.weightHALQ4K(name, s.M.q4kw[name]) }
		case s.M.kqw[name] != nil:
			qt := s.M.kqw[name]
			desc, ok := LookupQuantDescriptor(qt.kind)
			if !ok || !desc.SupportsHAL() || (!head && qt.kind == kindQ6K && !s.useHALKQuantWeight(qt)) {
				return nil, v41ProjectionDeclined, nil
			}
			dtype = desc.Dtype()
			if (head || guarded) && (desc.BlockWeights() <= 0 || desc.BlockBytes() <= 0 || in%desc.BlockWeights() != 0 || qt.nblk != in/desc.BlockWeights()) {
				return nil, v41ProjectionDeclined, nil
			}
			stage = func() compute.Tensor { return s.weightHALKQuant(name, qt) }

		case head && s.M.v41PackedHead(name) != nil:
			q := s.M.v41PackedHead(name)
			if q.prism != nil || in%qkK != 0 {
				return nil, v41ProjectionDeclined, nil
			}
			switch q.format {
			case packedEmbeddingQ2K:
				dtype = compute.Q2_K
			case packedEmbeddingQ4K:
				dtype = compute.Q4_K
			default:
				return nil, v41ProjectionDeclined, nil
			}
		default:
			return nil, v41ProjectionDeclined, nil
		}
		if !compute.BackendSupportsDeviceWeightDtype(s.Backend, dtype) || (dtype != compute.F32 && !s.Backend.Caps().UploadDtype) {
			return nil, v41ProjectionDeclined, nil
		}
		switch dtype {
		case compute.Q3_K:
			if native, ok := s.Backend.(interface{ SupportsQ3KMatMul() bool }); ok && !native.SupportsQ3KMatMul() {
				return nil, v41ProjectionDeclined, nil
			}
		case compute.Q5_K:
			if native, ok := s.Backend.(interface{ SupportsQ5KMatMul() bool }); ok && !native.SupportsQ5KMatMul() {
				return nil, v41ProjectionDeclined, nil
			}
		case compute.Q6_K:
			if native, ok := s.Backend.(interface{ SupportsQ6KMatMul() bool }); ok && !native.SupportsQ6KMatMul() {
				return nil, v41ProjectionDeclined, nil
			}
		}
		if head || guarded {
			stage = func() compute.Tensor { return s.v41HeadWeightHAL(name, dtype, out, in) }
		}
		opened := s.M.v41NowNanos()
		successfulRows := 0
		var uploaded, readback int64
		defer func() {
			if head {
				s.M.v41NoteHeadProjection(1, 0, successfulRows, 0, uploaded, readback, opened)
			} else if indexer {
				s.M.v41NoteIndexerProjection(1, 0, successfulRows, 0, uploaded, readback, opened)
			} else if compressor {
				s.M.v41NoteCompressorProjection(1, 0, successfulRows, 0, uploaded, readback, opened)
			} else {
				s.M.v41NoteDenseProjection(1, 0, successfulRows, 0, uploaded, readback, opened)
			}
		}()
		opStage := "weight upload"
		closeFailure := func(err error) error {
			if guarded {
				return s.v41DenseLeafCloseFailure(l, leaf, opStage, err)
			}
			if !head {
				return err
			}
			return s.v41HeadCloseFailure(opStage, err)
		}
		defer func() {
			if r := recover(); r != nil {
				if payload, ok := r.(v41HeadPayloadError); ok {
					if !guarded {
						opStage = "payload"
					}
					result, outcome, cause = nil, v41ProjectionError, closeFailure(payload.cause)
					return
				}
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						panic(r)
					}
					var backend *compute.BackendError
					if errors.As(err, &backend) {
						result, outcome, cause = nil, v41ProjectionError, closeFailure(err)
						return
					}
				}
				panic(r)
			}
		}()
		run := func() ([]float32, error) {
			weight := stage()
			if guarded && dtype == compute.Q4_K {
				q := s.M.q4kw[name]
				if q4kHostCopyReleasable(s, q, weight.Ready()) {
					q.raw = nil
				}
			}
			input := panel
			if s.M.prism != nil && s.M.prism.weightWidth[name] != 0 {
				input = make([]float32, len(panel))
				for row := 0; row < rows; row++ {
					copy(input[row*in:(row+1)*in], s.M.prismProjectInput(name, panel[row*in:(row+1)*in]))
				}
			}
			if head || guarded {
				for _, v := range input {
					if !finite32(v) {
						opStage = "payload"
						return nil, errV41ProjectionResult
					}
				}
			}
			shape := []int{in}
			if rows > 1 {
				shape = []int{rows, in}
			}
			opStage = "activation upload"
			x := s.uploadHostF32(shape, input, compute.MemoryActivation, "V4.1 dense projection activation")
			defer s.Backend.Free(x)
			uploaded = int64(len(input)) * 4
			opStage = "matmul"
			var y compute.Tensor
			if rows == 1 {
				y = s.Backend.MatMul(weight, x)
			} else {
				y = s.Backend.BatchedMatMul(weight, x, rows)
			}
			defer s.Backend.Free(y)
			opStage = "readback"
			values := s.Backend.Read(y)
			readback = int64(len(values)) * 4
			want, ok := checkedMulInt(rows, out)
			if !ok || len(values) != want {
				return nil, errV41ProjectionResult
			}
			if head || guarded {
				for _, v := range values {
					if !finite32(v) {
						return nil, errV41ProjectionResult
					}
				}
			}
			opStage = "lora"
			for row := 0; row < rows; row++ {
				s.M.loraApply(name, panel[row*in:(row+1)*in], values[row*out:(row+1)*out])
			}
			if head || guarded {
				for _, v := range values {
					if !finite32(v) {
						return nil, errV41ProjectionResult
					}
				}
			}
			return values, nil
		}
		var err error
		result, err = run()
		if err != nil {
			return nil, v41ProjectionError, closeFailure(err)
		}
		successfulRows = rows
		return result, v41ProjectionHandled, nil
	}
}

func (s *Session) v41DenseLeafCloseFailure(layer int, leaf, stage string, cause error) error {
	if !v41IndexerProjectionLeaf(leaf) {
		return s.v41CompressorCloseFailure(layer, stage, cause)
	}
	closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-indexer-projection", Layer: layer, Stage: stage, Cause: cause}
	s.halFailure = closed
	s.Close()
	return closed
}

func (s *Session) v41CompressorCloseFailure(layer int, stage string, cause error) error {
	closed := &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-compressor-projection", Layer: layer, Stage: stage, Cause: cause}
	s.halFailure = closed
	s.Close()
	return closed
}
