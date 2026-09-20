package model

// v41_forward.go assembles the DeepSeek V4.1 text forward behind new entry
// points (issue #12901, leaf of parent #12640). It composes the already-landed
// V4.1 leaves — the projected-input shared attention state (v41_attention_state.go),
// the mHC coefficient split/mix (v41_mhc.go), the routed+shared router
// (v41_router.go), the sparse sink attention contraction + grouped output
// projection (v41_sparse_attention.go) — into one ordered pass: embedding ->
// per-layer [mHC + attention + MoE] -> final norm -> head logits.
//
// Scope (gold-plating boundary from #12901): TEXT-forward assembly only. No GPU
// qualification, no GGUF/checkpoint loading, no streaming, no vision, no DSpark.
// Engram packed-row retrieval is wired as of #13007 (v41_forward_engram.go):
// a declared in-range Engram layer is admitted only when the model carries a
// packed-row stage and the three mixing tensors, and the stage is injected at
// the START of the layer per the reference schedule. That is fail-closed: a
// config declaring an in-range Engram layer WITHOUT a wired row source is
// refused at admission by v41EngramForwardAdmitted with an error wrapping
// ErrV41NativeUnsupported, so the assembly never emits non-Engram logits for a
// model it did not fully run. The same holds
// for a declared shared-KV source layer (v41KVSourceForwardAdmitted) and a
// declared compressor/indexer layer (v41CompressIndexForwardAdmitted): the
// reduced assembly projects a per-layer attn.wkv.weight, executes neither
// compression stage, and never reuses a shared KV/index source, so an in-range
// declaration is refused rather than silently run against a per-layer cache. The
// blocked-candidate selection the lightning indexer reads is likewise not executed
// (v41CandidateSourceForwardAdmitted), so an in-range candidate source is refused
// rather than silently run as unblocked attention. The
// mHC hyperconnection multiplicity is likewise fixed at the published four-stream
// layout (v41HCMultForwardAdmitted), so a declared hc_mult other than 4 is refused
// rather than silently run as four streams. The assembled
// pass is exercised by a REDUCED model
// against an independent scalar oracle (v41_forward_test.go); it does not itself
// qualify the official checkpoint for generation.
//
// Fail-closed design. Every stage is gated by v41ForwardAdmitted before any math
// runs, and forwardV41 re-checks each stage as it goes, returning a typed
// *V41ForwardError. A weightless V4.1 *Model (the probe in
// v41_failclosed_test.go) has no manifest, so admission fails on the embedding
// stage with an error that WRAPS ErrV41NativeUnsupported — keeping
// errors.Is(err, ErrV41NativeUnsupported) true for that fence test — while a
// fully-populated reduced model proceeds. A loaded model missing exactly one
// stage fails with an error that wraps ErrV41ForwardStage. forwardV41 NEVER
// returns logits when a stage is missing.

import (
	"errors"
	"fmt"
	"math"
)

// ErrV41ForwardStage reports that a required V4.1 forward stage could not be
// assembled (absent weight, inconsistent shape, or missing stream/state). It is
// the closed class every stage-specific V41ForwardError wraps.
var ErrV41ForwardStage = errors.New("model: DeepSeek V4.1 forward stage is not implemented")

// v41ForwardStage names the ordered assembly stage a failure occurred at.
type v41ForwardStage string

const (
	v41StageEmbedding v41ForwardStage = "embedding"
	v41StageLayer     v41ForwardStage = "layer"
	v41StageMHC       v41ForwardStage = "mhc"
	v41StageAttention v41ForwardStage = "attention"
	v41StageMoE       v41ForwardStage = "moe"
	v41StageEngram    v41ForwardStage = "engram"
	v41StageFinalNorm v41ForwardStage = "final_norm"
	v41StageHead      v41ForwardStage = "head"
	v41StageCompress  v41ForwardStage = "compress"
	v41StageIndexer   v41ForwardStage = "indexer"
)

// v41StageCompress and v41StageIndexer are the stage names for the CED/CSA2
// compressor and the lightning indexer. As of #13006 the reduced text forward
// EXECUTES both stages: the compressor pools a compressed layer's projected KV
// rows (v41CompressedRows), and the lightning indexer scores a declared index
// source against those compressed keys and selects rows (v41IndexRows). A config
// declaring an in-range compressor or indexer layer without its weights is still
// refused at admission by v41CompressIndexForwardAdmitted; a malformed schedule
// fails closed there too.

// V41ForwardError is the typed fail-closed error for one V4.1 assembly stage.
// Layer is the zero-based decoder layer the stage belongs to, or -1 for a
// model-global stage (embedding, final norm, head).
type V41ForwardError struct {
	Stage v41ForwardStage
	Layer int
	Err   error
}

func (e *V41ForwardError) Error() string {
	if e == nil {
		return "model: nil V4.1 forward error"
	}
	where := ""
	if e.Layer >= 0 {
		where = fmt.Sprintf(" layer=%d", e.Layer)
	}
	if e.Err != nil {
		return fmt.Sprintf("model: V4.1 forward stage %s%s: %v", e.Stage, where, e.Err)
	}
	return fmt.Sprintf("model: V4.1 forward stage %s%s failed", e.Stage, where)
}

// Unwrap exposes the wrapped cause so errors.Is reaches ErrV41ForwardStage or,
// for the weightless probe, ErrV41NativeUnsupported.
func (e *V41ForwardError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

func v41StageErr(stage v41ForwardStage, layer int, err error) error {
	if err == nil {
		err = ErrV41ForwardStage
	}
	return &V41ForwardError{Stage: stage, Layer: layer, Err: err}
}

// v41ForwardState is the session-owned continuation state for the reduced V4.1
// text forward. It carries committed token history and one bounded temporal
// attention state per decoder layer. attn is the per-forward registry through
// which source layers publish completed rows and selections to later readers.
// Session.Step still recomputes full history; these states are seeded for later
// incremental leaves and do not activate incremental arithmetic.
type v41ForwardState struct {
	history []int
	attn    *V41AttentionState
	layers  []*V41AttentionState

	// expertGateUp is the OPTIONAL device gate/up callback the session installs
	// when its backend can run a routed expert's gate/up projections plus SwiGLU
	// (the shared q4kExpertInputHAL operation from TICKET-05, bound in
	// Session.v41State). When non-nil the MoE loop offers each pick to it first:
	// on a handled result the I-wide fused intermediate comes from the device and
	// only the existing host down contraction runs over it, so the gate/up f32
	// weights are never materialized on the host. A nil callback (Model.Forward,
	// a non-device session, or a backend the shared operation declines) preserves
	// the historical host triple byte-for-byte.
	expertGateUp v41ExpertGateUpFunc
}

// v41ExpertGateUpOutcome is the closed result vocabulary of one v41ExpertGateUpFunc
// call. handled-success returns the I-wide fused intermediate for the existing host
// down contraction; declined leaves the pick to the historical host gate/up/down
// triple; handled-error is a SELECTED execution failure and must surface, never be
// swallowed as a decline.
type v41ExpertGateUpOutcome uint8

const (
	// v41GateUpDeclined: the shared device operation did not admit this expert, so
	// the caller falls through to the host triple byte-for-byte.
	v41GateUpDeclined v41ExpertGateUpOutcome = iota
	// v41GateUpHandled: gate MatMul, up MatMul and SwiGLU all ran on the backend
	// and the returned slice is the I-wide fused intermediate.
	v41GateUpHandled
	// v41GateUpError: a selected device execution failed; it must remain visible.
	v41GateUpError
)

// v41ExpertGateUpFunc is the optional per-pick device gate/up operation. It
// receives the routed expert's layer stem and the normalized input, and reports
// one of the closed v41ExpertGateUpOutcome values. On v41GateUpHandled it returns
// the I-wide fused intermediate (gate ⊙ silu, then SwiGLU) produced on the device
// backend, sized to the expert intermediate width.
type v41ExpertGateUpFunc func(layer int, stem string, xn []float32) ([]float32, v41ExpertGateUpOutcome, error)

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
	woA []float32
	woB []float32

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

func (st *v41ForwardState) attentionState(headDim, ratioCap int) (*V41AttentionState, error) {
	if st.attn != nil {
		return st.attn, nil
	}
	state, err := NewV41AttentionState(headDim, ratioCap)
	if err != nil {
		return nil, err
	}
	st.attn = state
	return state, nil
}

func (st *v41ForwardState) layerState(layer int) *V41AttentionState {
	if st == nil || layer < 0 || layer >= len(st.layers) {
		return nil
	}
	return st.layers[layer]
}

func (st *v41ForwardState) setLayerState(layer, count int, state *V41AttentionState) {
	if len(st.layers) != count {
		st.layers = make([]*V41AttentionState, count)
	}
	st.layers[layer] = state
}

func (st *v41ForwardState) retainedCopyCount() int {
	if st == nil {
		return 0
	}
	total := 0
	for _, layer := range st.layers {
		if layer != nil {
			total += layer.retainedCopies
		}
	}
	return total
}

// ---- reduced V4.1 geometry helpers -----------------------------------------

// v41KVLoraRankReduced is the reduced-model KV latent width. The published V4.1
// KV latent (v41KVLoraRank = 512) is a checkpoint constant the text Config does
// not carry; the reduced assembly deliberately uses HeadDim so the reduced test
// fixture stays tiny and self-consistent. It is not a claim about the official
// checkpoint geometry.
func v41KVLoraRankReduced(cfg Config) int { return cfg.HeadDim }

// v41ForwardGeometry reports whether cfg declares the published full V4.1
// geometry (true) or the reduced test fixture (false). It FAILS CLOSED: when the
// parsed DeepSeekV41.Attention envelope is populated (a published/parsed config)
// and the config is not a coherently narrowed reduced fixture, the full geometry
// is REQUIRED and any mismatch is a typed ErrV41ForwardStage -- the assembly
// never silently falls back to the reduced stand-in. A lone-axis drift (a mutated
// head width while the decoder stack stays at the published envelope) is refused.
//
// The discriminator is the Attention envelope's HeadDim plus the flat layer
// count: a reduced fixture narrows both the decoder stack and the head width
// (NumLayers 40 -> 1, HeadDim 512 -> 32). A config with no DeepSeekV41 metadata or
// an empty envelope is classified by its own flat HeadDim: 512 (v41KVLoraRank)
// means full, anything else (the reduced fixture's 32) means reduced.
func v41ForwardGeometry(cfg Config) (bool, error) {
	m := cfg.DeepSeekV41
	if m == nil || m.Attention.HeadDim == 0 {
		return cfg.HeadDim == v41KVLoraRank, nil
	}
	attn := m.Attention
	// A parsed/published config derives its Attention envelope from the SAME flat
	// geometry, so a genuine published config always agrees with its envelope on
	// every decoder axis. Two situations can disagree:
	//
	//   * The reduced fixture reuses the retained published metadata pointer but
	//     narrows the flat decoder stack (NumLayers 40 -> 1) AND the head width
	//     (HeadDim 512 -> 32) -- a deliberate, coherent narrowing. That config is a
	//     fixture and is classified by its flat head width.
	//   * A corrupted full config drifts a single axis (typically the head width)
	//     while leaving the rest of the decoder stack at the published envelope.
	//     That is NOT a deliberate reduction and FAILS CLOSED: a full-intended
	//     config must never silently run the reduced stand-in geometry.
	//
	// The two are distinguished by whether the flat decoder stack itself was
	// narrowed below the envelope. Requiring BOTH a non-published head width and a
	// narrowed layer count keeps the reduced fixture admitted while refusing a
	// lone-axis drift.
	if attn.HeadDim != cfg.HeadDim {
		reducedFixture := cfg.HeadDim != v41KVLoraRank && cfg.NumLayers < attn.NumLayers
		if !reducedFixture {
			return false, v41StageErr(v41StageAttention, -1,
				fmt.Errorf("%w: published V4.1 attention envelope declares head_dim=%d but config has %d and the decoder stack is not a narrowed fixture (layers %d vs %d)",
					ErrV41ForwardStage, attn.HeadDim, cfg.HeadDim, cfg.NumLayers, attn.NumLayers))
		}
		return cfg.HeadDim == v41KVLoraRank, nil
	}
	// Authoritative (self-consistent) envelope: the full published geometry is
	// REQUIRED. Any mismatch on a decoder axis is a typed fail-closed error -- the
	// assembly never falls back to the reduced stand-in.
	type axis struct {
		name      string
		got, want int
	}
	for _, a := range []axis{
		{"head_dim", cfg.HeadDim, v41KVLoraRank},
		{"num_hidden_layers", cfg.NumLayers, attn.NumLayers},
		{"hidden_size", cfg.HiddenSize, attn.HiddenSize},
		{"num_attention_heads", cfg.NumHeads, attn.NumHeads},
		{"num_key_value_heads", cfg.NumKVHeads, attn.NumKVHeads},
	} {
		if a.got != a.want {
			return false, v41StageErr(v41StageAttention, -1,
				fmt.Errorf("%w: published V4.1 attention envelope declares %s=%d but config has %d", ErrV41ForwardStage, a.name, a.want, a.got))
		}
	}
	return true, nil
}

// v41ForwardKVLatentRank resolves the attn.wkv output width for an admitted
// config: the published v41KVLoraRank (512) on the full path, the tiny HeadDim
// stand-in on the reduced fixture. It returns v41ForwardGeometry's error rather
// than defaulting to the reduced width, so a config whose published envelope is
// inconsistent can never be admitted against a reduced stand-in shape.
func v41ForwardKVLatentRank(cfg Config) (int, error) {
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return 0, err
	}
	if full {
		return v41KVLoraRank, nil
	}
	return v41KVLoraRankReduced(cfg), nil
}

// v41MHCMixWidth is the mHC coefficient-vector width for hc=4: (2+hc)*hc.
const v41MHCMixWidth = 24

// v41RouterConfigFullGeometry is the real-path routed+shared geometry. It routes
// through v41RouterConfigFromConfig so the forward only admits the published
// 384/top-6/1-shared/1.5-scale envelope; any other axis is refused here rather
// than silently running a different MoE geometry. The router admission error is
// wrapped into the typed ErrV41ForwardStage so a config missing (or carrying a
// non-published) full-geometry axis fails closed at the forward stage boundary.
func v41RouterConfigFullGeometry(cfg Config) (v41RouterConfig, error) {
	rc, err := v41RouterConfigFromConfig(cfg)
	if err != nil {
		return v41RouterConfig{}, v41StageErr(v41StageMoE, -1,
			fmt.Errorf("%w: published router envelope rejected: %w", ErrV41ForwardStage, err))
	}
	return rc, nil
}

// ---- admission -------------------------------------------------------------

// v41EngramForwardAdmitted fails closed when the config declares an Engram layer
// that lies WITHIN the model's decoder stack but the model's Engram stage is not
// fully wired for it. As of #13007 the reduced text assembly DOES execute
// packaged-row Engram retrieval (v41_forward_engram.go): a declared in-range
// Engram layer is admitted only when it has a wired row source and the three
// mixing tensors (engram_kv.weight, engram_q_norm.weight, engram_k_norm.weight).
// Otherwise the assembly would emit logits for a model it did not fully run, so
// the layer is refused with an error wrapping ErrV41NativeUnsupported — the same
// native-forward-unavailable class the #12967 weightless fence uses, because the
// Engram stage cannot execute without its packed-row source. Declared Engram
// layers outside [0,NumLayers) are unreachable by this forward and stay admitted,
// which keeps the reduced oracle fixture (NumLayers=1, EngramLayerIDs [1,14])
// runnable.
//
// Method receiver (not a bare Config) is required because admission must see
// whether THIS model carries the packed-row stage; v41ForwardAdmitted calls it
// after the embedding/manifest checks so a weightless model still fails at the
// embedding fence with ErrV41NativeUnsupported first.
func (m *Model) v41EngramForwardAdmitted() error {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		return nil
	}
	for _, layer := range d41.EngramLayerIDs {
		if layer < 0 || layer >= m.Cfg.NumLayers {
			continue
		}
		stage := m.v41EngramStageFor()
		if stage == nil || stage.cacheIndex(layer) < 0 {
			return v41StageErr(v41StageEngram, layer,
				fmt.Errorf("%w: layer %d declares Engram but no packed-row source is wired", ErrV41NativeUnsupported, layer))
		}
		cols := stage.columns
		H := m.Cfg.HiddenSize
		if err := m.v41AdmitShape(layerName(layer, "engram_kv.weight"), v41StageEngram, layer, cols*stage.headDim, (stage.hc+1)*H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "engram_q_norm.weight"), v41StageEngram, layer, stage.hc*H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "engram_k_norm.weight"), v41StageEngram, layer, stage.hc*H); err != nil {
			return err
		}
	}
	return nil
}

// v41CompressIndexForwardAdmitted fails closed when the config declares a
// CED/CSA2 compressor regime or a lightning-indexer source that lies WITHIN the
// model's decoder stack. The reduced text assembly executes neither stage (their
// packed-row / compressed-stream inputs cannot be materialized weight-free — see
// the scope note at the top of this file), so silently dropping a declared
// in-range compressor/indexer layer would emit reduced logits for a model the
// assembly never ran. Declarations that only touch out-of-range layers stay
// admitted, which keeps the reduced oracle fixture runnable: it derives from the
// published 40-layer config but narrows NumLayers to 1, so every CompressRatios
// entry above index 0 and every index source ({2,8,...}) is unreachable. Executing
// the real compressor/indexer stages is #13006's remaining integration work; until
// then this is the fail-closed boundary.
//
// Two malformed-schedule arms are also refused here: a negative ratio (invalid
// geometry, never a compressed layer) and a schedule shorter than the decoder
// stack (an uncovered layer with no declared regime). Both must fail closed, not
// fall through to the uncompressed regime.
func (m *Model) v41CompressIndexForwardAdmitted() error {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		return nil
	}
	cfg := m.Cfg
	// Every layer in the model's decoder stack must declare a regime. A schedule
	// shorter than the stack leaves the uncovered layers with no declared
	// compression regime, which must fail closed rather than being silently read as
	// ratio 0.
	if len(d41.CompressRatios) < cfg.NumLayers {
		return v41StageErr(v41StageCompress, len(d41.CompressRatios),
			fmt.Errorf("%w: compression schedule declares %d ratios but the model has %d layers", ErrV41ForwardStage, len(d41.CompressRatios), cfg.NumLayers))
	}
	H := cfg.HiddenSize
	for layer := 0; layer < cfg.NumLayers; layer++ {
		// Ratio 0 and 1 are the uncompressed regimes. A ratio > 1 declares a
		// compressed layer whose compressor must be wired; a negative ratio is
		// malformed geometry that must fail closed rather than being silently
		// treated as uncompressed.
		ratio := d41.CompressRatios[layer]
		switch {
		case ratio < 0:
			return v41StageErr(v41StageCompress, layer,
				fmt.Errorf("%w: layer %d declares malformed compressor ratio %d", ErrV41ForwardStage, layer, ratio))
		case ratio > 1:
			width := v41CompressorWidth(cfg)
			if err := m.v41AdmitShape(layerName(layer, "attn.compressor.wkv.weight"), v41StageCompress, layer, width, H); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(layer, "attn.compressor.wgate.weight"), v41StageCompress, layer, width, H); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(layer, "attn.compressor.norm.weight"), v41StageCompress, layer, width); err != nil {
				return err
			}
		}
	}
	indexHeads := cfg.IndexNHeads
	indexDim := cfg.IndexHeadDim
	for _, layer := range d41.IndexSourceLayerIDs {
		if layer < 0 || layer >= cfg.NumLayers {
			continue
		}
		if indexHeads <= 0 || indexDim <= 0 {
			return v41StageErr(v41StageIndexer, layer,
				fmt.Errorf("%w: layer %d declares a lightning-indexer source but the indexer geometry (nHeads=%d headDim=%d) is not declared", ErrV41ForwardStage, layer, indexHeads, indexDim))
		}
		wq := indexHeads * indexDim
		if err := m.v41AdmitShape(layerName(layer, "indexer.wq_b.weight"), v41StageIndexer, layer, wq, cfg.QLoraRank); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "indexer.wk.weight"), v41StageIndexer, layer, indexDim, v41CompressorWidth(cfg)); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "indexer.k_norm.weight"), v41StageIndexer, layer, indexDim); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "indexer.weights_proj.weight"), v41StageIndexer, layer, indexHeads, H); err != nil {
			return err
		}
	}
	return nil
}

// v41CompressorWidth is the compressor latent width. The published V4.1
// compressor pools to the KV latent width; the reduced text assembly does not
// carry that checkpoint constant on the text Config, so it pools at HeadDim,
// matching v41KVLoraRankReduced. It is not a claim about the official geometry.
func v41CompressorWidth(cfg Config) int { return cfg.HeadDim }

// v41CompressedRows pools the per-position projected KV rows of one layer
// through the CED/CSA2 compressor. It returns the emitted compressed rows in
// causal order. A non-compressed regime returns the input rows unchanged. It
// fails closed on any malformed geometry rather than emitting a partial stream.
func (m *Model) v41CompressedRows(l int, ratio int, kvRows [][]float32, inputs [][]float32) ([][]float32, error) {
	if ratio <= 1 {
		return kvRows, nil
	}
	cfg := m.Cfg
	width := v41CompressorWidth(cfg)
	H := cfg.HiddenSize
	wkv := m.tensor(layerName(l, "attn.compressor.wkv.weight"))
	wgate := m.tensor(layerName(l, "attn.compressor.wgate.weight"))
	normWeight := m.tensor(layerName(l, "attn.compressor.norm.weight"))
	eps := float32(cfg.RMSNormEps)
	pool, err := NewV41CompressorPool(ratio, width)
	if err != nil {
		return nil, v41StageErr(v41StageCompress, l, err)
	}
	var out [][]float32
	for pos := range inputs {
		var in []float32
		if pos < len(inputs) {
			in = inputs[pos]
		}
		kv := matRows(wkv, in, width, H)
		score := matRows(wgate, in, width, H)
		pooled, emitted, err := pool.PushNormalized(pos, kv, score, normWeight, eps)
		if err != nil {
			return nil, v41StageErr(v41StageCompress, l, err)
		}
		if emitted {
			out = append(out, pooled)
		}
	}
	return out, nil
}

// v41IndexRows scores a projected index query against the compressed keys and
// returns the selected compressed row IDs for one layer. It returns nil when the
// layer declares no index source. Candidate blocks are selected when the layer is
// the declared candidate source, so the selection reads a blocked pool rather
// than the full compressed set.
func (m *Model) v41IndexRows(l int, qLat []float32, hidden []float32, keys [][]float32) ([]int32, error) {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		return nil, nil
	}
	isSource := false
	for _, src := range d41.IndexSourceLayerIDs {
		if src == l {
			isSource = true
			break
		}
	}
	if !isSource {
		return nil, nil
	}
	cfg := m.Cfg
	nHeads, headDim := cfg.IndexNHeads, cfg.IndexHeadDim
	if nHeads <= 0 || headDim <= 0 {
		return nil, v41StageErr(v41StageIndexer, l,
			fmt.Errorf("%w: indexer geometry nHeads=%d headDim=%d is not declared", ErrV41ForwardStage, nHeads, headDim))
	}
	wqB := m.tensor(layerName(l, "indexer.wq_b.weight"))
	wk := m.tensor(layerName(l, "indexer.wk.weight"))
	kNorm := m.tensor(layerName(l, "indexer.k_norm.weight"))
	wProj := m.tensor(layerName(l, "indexer.weights_proj.weight"))
	compressLen := len(keys)
	q := matRows(wqB, qLat, nHeads*headDim, cfg.QLoraRank)
	flatKeys := make([]float32, 0, compressLen*headDim)
	for _, row := range keys {
		projected := matRows(wk, row, headDim, len(row))
		if len(kNorm) == headDim {
			projected = rmsnormCfg(projected, kNorm, float32(cfg.RMSNormEps), cfg)
		}
		flatKeys = append(flatKeys, projected...)
	}
	weights := matRows(wProj, hidden, nHeads, cfg.HiddenSize)
	for h := 0; h < nHeads; h++ {
		weights[h] *= cfg.attnScale() * float32(1.0/math.Sqrt(float64(nHeads)))
	}
	topKBlocks, blockSize := 0, 0
	if d41.CandidateSourceLayerID == l {
		topKBlocks, blockSize = d41.CandidateTopKBlocks, d41.CandidateBlockSize
	}
	pub, err := NewV41IndexerPublication(l, q, flatKeys, weights, nHeads, headDim, compressLen, topKBlocks, blockSize, cfg.IndexTopK, 0)
	if err != nil {
		return nil, v41StageErr(v41StageIndexer, l, err)
	}
	return pub.Rows(), nil
}

// v41KVSourceForwardAdmitted fails closed when the config declares a shared-KV
// source layer that lies WITHIN the model's decoder stack but whose shared state
// the reduced assembly cannot actually publish. As of #12896 the assembly DOES
// execute the shared KV/index schedule (v41_attention.go): a declared source
// pools its projected KV input through the CED/CSA2 compressor and publishes the
// rows, and a later reader resolves them. A source is therefore admitted only
// when it declares a compressed regime (ratio > 1, so the compressor runs) and
// carries the compressor tensors its pooling reads; a source at ratio <= 1 has
// no pooled stream to publish, so it is refused rather than silently running a
// per-layer KV cache where the official model reuses a source layer's state.
// Declarations that only touch out-of-range layers stay admitted, which keeps the
// reduced oracle fixture runnable: it derives from the published 40-layer config
// but narrows NumLayers to 1, so every kv source ({2,8,14,20}) is unreachable.
func (m *Model) v41KVSourceForwardAdmitted() error {
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil {
		return nil
	}
	cfg := m.Cfg
	for _, layer := range d41.KVSourceLayerIDs {
		if layer < 0 || layer >= cfg.NumLayers {
			continue
		}
		if !v41AttentionRatioImplemented(v41CompressRatioAt(cfg, layer)) {
			return v41StageErr(v41StageCompress, layer,
				fmt.Errorf("%w: layer %d declares a shared-KV source but its compress ratio %d is not an implemented variant", ErrV41ForwardStage, layer, v41CompressRatioAt(cfg, layer)))
		}
		if v41CompressRatioAt(cfg, layer) <= 1 {
			return v41StageErr(v41StageAttention, layer,
				fmt.Errorf("%w: layer %d declares a shared-KV source but has no compressed regime to publish", ErrV41ForwardStage, layer))
		}
		H := cfg.HiddenSize
		width := v41CompressorWidth(cfg)
		if err := m.v41AdmitShape(layerName(layer, "attn.compressor.wkv.weight"), v41StageCompress, layer, width, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "attn.compressor.wgate.weight"), v41StageCompress, layer, width, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(layer, "attn.compressor.norm.weight"), v41StageCompress, layer, width); err != nil {
			return err
		}
	}
	return nil
}

// v41CandidateSourceForwardAdmitted fails closed when the config declares a
// candidate-source layer that lies WITHIN the model's decoder stack. The reduced
// text assembly runs its own full per-layer attention contraction and never
// executes the CED/CSA2 blocked-candidate selection the lightning indexer reads
// (the reference's candidate_source_layer_id / candidate_topk_blocks /
// candidate_block_size schedule), so silently running an in-range candidate source
// would emit logits from an unblocked attention path where the official model
// selects a sparse candidate pool first. Declarations that only touch out-of-range
// layers stay admitted, which keeps the reduced oracle fixture runnable: it derives
// from the published 40-layer config but narrows NumLayers to 1, so the candidate
// source (20) is unreachable. Executing the real candidate selection is the
// CED/CSA2 attention leaf's remaining work; until then this is the fail-closed
// boundary.
func v41CandidateSourceForwardAdmitted(cfg Config) error {
	m := cfg.DeepSeekV41
	if m == nil {
		return nil
	}
	layer := m.CandidateSourceLayerID
	if layer >= 0 && layer < cfg.NumLayers {
		return v41StageErr(v41StageIndexer, layer,
			fmt.Errorf("%w: layer %d declares a candidate source but the reduced forward does not execute the CED/CSA2 blocked-candidate selection", ErrV41ForwardStage, layer))
	}
	return nil
}

// v41HCMultForwardAdmitted fails closed when the config declares an mHC
// hyperconnection multiplicity other than 4. The reduced text assembly hardcodes
// the four-stream geometry (v41MHCSplit called with hc=4, four identical stand-in
// streams, and the width-24 mix projection v41MHCMixWidth), so it executes only
// the published hc_mult=4 layout. A config declaring a different multiplicity
// would otherwise run the four-stream hyperconnection for a model the assembly
// never ran -- a silent omission contrary to the fail-closed invariant. The
// published artifact and every reduced fixture declare hc_mult=4, so the reduced
// oracle forward stays admitted; executing a non-4 multiplicity is the mHC
// geometry leaf's remaining work.
func v41HCMultForwardAdmitted(cfg Config) error {
	m := cfg.DeepSeekV41
	if m == nil {
		return nil
	}
	if m.HCMult != 4 {
		return v41StageErr(v41StageMHC, -1,
			fmt.Errorf("%w: config declares mHC multiplicity %d but the reduced forward only executes the four-stream hc_mult=4 geometry", ErrV41ForwardStage, m.HCMult))
	}
	return nil
}

// v41ForwardAdmitted returns nil only when this is an admitted V4.1 config with
// every required stage's weights present and shape-consistent. It is the gate
// both Model.Forward and Session.Prefill/Step run before the assembly. A
// weightless model fails at the embedding stage with an error wrapping
// ErrV41NativeUnsupported (the #12967 fence contract); a loaded model missing a
// stage fails with an error wrapping ErrV41ForwardStage.
func (m *Model) v41ForwardAdmitted() error {
	if m == nil {
		return v41StageErr(v41StageEmbedding, -1, fmt.Errorf("%w: nil model", ErrV41NativeUnsupported))
	}
	if !m.Cfg.IsDeepSeekV41() {
		return v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: config is not DeepSeek V4.1", ErrV41ForwardStage))
	}
	if len(m.manifest) == 0 {
		return v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: reduced weights absent", ErrV41NativeUnsupported))
	}
	cfg := m.Cfg
	// Geometry discrimination fails closed: a parsed/published config whose flat
	// axes disagree with its Attention envelope is refused here rather than
	// silently assembled against the reduced stand-in geometry.
	if _, err := v41ForwardGeometry(cfg); err != nil {
		return err
	}
	// kvLatentRank is resolved through the same fail-closed discriminator, so an
	// inconsistent published envelope can never be admitted at the reduced shape.
	kvLatentRank, err := v41ForwardKVLatentRank(cfg)
	if err != nil {
		return err
	}
	if err := m.v41AdmitShape("model.embed_tokens.weight", v41StageEmbedding, -1, cfg.VocabSize, cfg.HiddenSize); err != nil {
		return err
	}
	if err := m.v41AdmitShape("model.norm.weight", v41StageFinalNorm, -1, cfg.HiddenSize); err != nil {
		return err
	}
	if !m.hasWeight("lm_head.weight") && !m.hasWeight("model.embed_tokens.weight") {
		return v41StageErr(v41StageHead, -1,
			fmt.Errorf("%w: no lm_head.weight and no tied embedding", ErrV41ForwardStage))
	}
	if err := m.v41AdmitShape("lm_head.weight", v41StageHead, -1, cfg.VocabSize, cfg.HiddenSize); err != nil {
		return err
	}
	if _, err := v41RouterConfigFullGeometry(cfg); err != nil {
		return err
	}
	if err := m.v41EngramForwardAdmitted(); err != nil {
		return err
	}
	if err := m.v41CompressIndexForwardAdmitted(); err != nil {
		return err
	}
	if err := m.v41KVSourceForwardAdmitted(); err != nil {
		return err
	}
	if err := v41CandidateSourceForwardAdmitted(cfg); err != nil {
		return err
	}
	if err := v41HCMultForwardAdmitted(cfg); err != nil {
		return err
	}
	if err := v41AttentionGeometryForwardAdmitted(cfg); err != nil {
		return err
	}
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	qHeadDim := nH * hd
	oDim := cfg.OLoraRank * cfg.OGroups
	I := cfg.MoEIntermediateSize
	// full selects the artifact geometry over the reduced fixture. The V4.1 Q/KV
	// latent norms are an artifact-only stage, so their admission is gated on it.
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return err
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if err := m.v41AdmitShape(layerName(l, "attn_norm.weight"), v41StageLayer, l, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn_norm.weight"), v41StageLayer, l, H); err != nil {
			return err
		}
		if err := m.v41AdmitMHC(l); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wq_a.weight"), v41StageAttention, l, cfg.QLoraRank, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wq_b.weight"), v41StageAttention, l, qHeadDim, cfg.QLoraRank); err != nil {
			return err
		}
		// Q/KV latent norms (reference Attention: self.q_norm = RMSNorm(q_lora_rank)
		// and self.kv_norm = RMSNorm(head_dim)). They are admitted and applied ONLY
		// on the full path: the reduced fixture's pre-#13009 arithmetic has no such
		// stage and must stay byte-identical, so a reduced config neither requires
		// nor reads these leaves. See v41Layer's application site (#13290).
		if full {
			if err := m.v41AdmitShape(layerName(l, "attn.wq_a_norm.weight"), v41StageAttention, l, cfg.QLoraRank); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(l, "attn.kv_norm.weight"), v41StageAttention, l, kvLatentRank); err != nil {
				return err
			}
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wkv.weight"), v41StageAttention, l, kvLatentRank, H); err != nil {
			return err
		}
		// wo_a is the leaf's group-major [Groups, OLoRARank, HeadsPerGroup*HeadDim]
		// tensor, so its total is OLoRARank*NumHeads*HeadDim; wo_b is
		// [H, Groups*OLoRARank]. The artifact stores the group-major form while
		// the reduced fixture declares the equivalent flat form, so admit either
		// declaration (see v41AdmitGroupedWoA; the #13264 seam).
		if err := m.v41AdmitGroupedWoA(l); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.wo_b.weight"), v41StageAttention, l, H, oDim); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "attn.sink"), v41StageAttention, l, nH); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.gate.weight"), v41StageMoE, l, cfg.NumExperts, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.gate.e_score_correction_bias"), v41StageMoE, l, cfg.NumExperts); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.shared_experts.w1.weight"), v41StageMoE, l, I, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.shared_experts.w3.weight"), v41StageMoE, l, I, H); err != nil {
			return err
		}
		if err := m.v41AdmitShape(layerName(l, "ffn.shared_experts.w2.weight"), v41StageMoE, l, H, I); err != nil {
			return err
		}
		for e := 0; e < cfg.NumExperts; e++ {
			stem := "ffn.experts." + itoa(e)
			if err := m.v41AdmitShape(layerName(l, stem+".w1.weight"), v41StageMoE, l, I, H); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(l, stem+".w3.weight"), v41StageMoE, l, I, H); err != nil {
				return err
			}
			if err := m.v41AdmitShape(layerName(l, stem+".w2.weight"), v41StageMoE, l, H, I); err != nil {
				return err
			}
		}
	}
	_ = hd
	return nil
}

// v41AdmitMHC admits a layer's mHC coefficient block in either the reduced
// fixture's legacy [mixWidth, H] geometry or the published artifact's flattened
// four-stream geometry. The reference (inference/model.py mHC) projects the
// width-4H flattened residual through hc_attn_fn; the staged vcruz Q2_K artifact
// stores that projection as [4H, 24] (input-major), while the forward consumes it
// logically as [24, 4H] (coefficient-major). Both the logical [mixWidth, 4H]
// orientation and the stored [4H, mixWidth] transpose are admitted here so a real
// artifact load reaches the forward instead of refusing at admission; every other
// shape (including the reduced [mixWidth, H]) still falls through to the named
// two-axis shape guard and fails closed. The base/scale vectors are unchanged.
func (m *Model) v41AdmitMHC(l int) error {
	H := m.Cfg.HiddenSize
	name := layerName(l, "mhc.mixes.weight")
	out, in, ok := m.residentShape(name)
	if !ok {
		return v41StageErr(v41StageMHC, l, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	switch {
	case out == v41MHCMixWidth && (in == H || in == 4*H):
		// logical [mixWidth, in]
	case in == v41MHCMixWidth && out == 4*H:
		// stored artifact transpose [4H, mixWidth]
	default:
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: tensor %s shape [%d %d], want [%d %d], [%d %d] or [%d %d]",
				ErrV41ForwardStage, name, out, in, v41MHCMixWidth, H, v41MHCMixWidth, 4*H, 4*H, v41MHCMixWidth))
	}
	if err := m.v41AdmitShape(layerName(l, "mhc.base"), v41StageMHC, l, v41MHCMixWidth); err != nil {
		return err
	}
	if err := m.v41AdmitShape(layerName(l, "mhc.scale"), v41StageMHC, l, 3); err != nil {
		return err
	}
	return nil
}

// v41AdmitGroupedWoA admits a layer's attn.wo_a.weight in either declaration of
// the same weight. The forward consumes it through V41GroupedOutputProjection,
// which reads the group-major [Groups, OLoRARank, HeadsPerGroup*HeadDim] tensor
// the published artifact's attn_output_a.weight stores. The reduced fixture
// declares the equivalent flat [OLoraRank, NumHeads*HeadDim] two-axis form.
// The two forms carry the SAME element count
// (OGroups*OLoraRank*HeadsPerGroup*HeadDim == OLoraRank*NumHeads*HeadDim) but
// assign different numbers to the two axes, so a single v41AdmitShape row cannot
// admit both: demanding the flat form refused the real artifact by name before
// the grouped projection ever ran (the #13264 seam), and demanding the grouped
// form would refuse the reduced fixture.
//
// Both declarations are admitted here. Any other shape - including a grouped
// form whose axes are individually consistent but whose total disagrees, and the
// artifact's [OGroups*OLoraRank, HeadsPerGroup*HeadDim] with a non-divisible
// head count - falls through to the named two-axis shape guard and fails closed,
// so the no-silent-mis-shape property is retained. Presence is resolved
// residency-completely by residentShape, so a resident-store wo_a is admitted at
// its real geometry on either arm.
func (m *Model) v41AdmitGroupedWoA(l int) error {
	name := layerName(l, "attn.wo_a.weight")
	cfg := m.Cfg
	out, in, ok := m.residentShape(name)
	if !ok {
		return v41StageErr(v41StageAttention, l, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	flatRows, flatCols := cfg.OLoraRank, cfg.NumHeads*cfg.HeadDim
	groupedRows := cfg.OGroups * cfg.OLoraRank
	groupedHeadsPerGroup := 0
	if cfg.OGroups > 0 && cfg.NumHeads%cfg.OGroups == 0 {
		groupedHeadsPerGroup = cfg.NumHeads / cfg.OGroups
	}
	groupedCols := groupedHeadsPerGroup * cfg.HeadDim
	switch {
	case out == flatRows && in == flatCols:
		return nil
	case groupedHeadsPerGroup > 0 && out == groupedRows && in == groupedCols:
		return nil
	default:
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: tensor %s shape [%d %d], want [%d %d] (flat) or [%d %d] (grouped)",
				ErrV41ForwardStage, name, out, in, flatRows, flatCols, groupedRows, groupedCols))
	}
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
	name := layerName(l, leaf)
	if !m.hasResidentWeight(name) {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	return m.residentMatRows(name, x, out, in), nil
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
func (m *Model) v41ExpertTripleInto(l int, stem string, scratch *v41ProjScratch) (w1, w3, w2 []float32, err error) {
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
	m.v41CacheExpertTriple(l, stem, scratch, w1, w3, w2)
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

// v41AdmitShape asserts a named tensor is present with the expected shape.
// Presence + shape are resolved residency-completely (residentShape), so a
// weight that a quantized serve keeps in a resident store (kqw/q4kw/q8w/...)
// rather than the f32 manifest is admitted at its real [out, in] geometry
// instead of being refused as "missing". The fail-closed property is retained:
// a tensor absent from every store still refuses by name, and a resident tensor
// whose geometry disagrees with the expected shape still refuses.
func (m *Model) v41AdmitShape(name string, stage v41ForwardStage, layer int, want ...int) error {
	if len(want) == 2 {
		out, in, ok := m.residentShape(name)
		if !ok {
			// A routed expert the R5 streamed-experts tier holds is by design
			// ABSENT from every resident store (manifest/q8w/q4w/q4kw/kqw/q2w/
			// gptqw), because the whole point of the tier is to fault one expert's
			// stride out of a fused checkpoint slab only when it is routed. So a
			// resident-store miss falls through to the tier's index, a
			// presence-only check (no IO, no fault), before refusing by name.
			//
			// The descriptor geometry is deliberately NOT re-checked here: the tier
			// entry carries the fused tensor's declared rows/cols, but the
			// per-expert shape is the loader's already-validated [out,in] declaration
			// (FusedExpertTensor.Rows/Cols), and the forward's own read of the
			// projection is the authority on shape use. A tier that carries the name
			// at a disagreeing geometry is therefore admitted at presence, and any
			// real disagreement surfaces at the forward's shape use rather than
			// being silently accepted as a different tensor.
			if m.expertCheckpoint.Has(name) {
				return nil
			}
			return v41StageErr(stage, layer, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
		}
		if out != want[0] || in != want[1] {
			return v41StageErr(stage, layer,
				fmt.Errorf("%w: tensor %s shape [%d %d], want %v", ErrV41ForwardStage, name, out, in, want))
		}
		return nil
	}
	meta, ok := m.manifest[name]
	if !ok {
		return v41StageErr(stage, layer, fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, name))
	}
	if len(meta.Shape) != len(want) {
		return v41StageErr(stage, layer,
			fmt.Errorf("%w: tensor %s rank %d, want %d", ErrV41ForwardStage, name, len(meta.Shape), len(want)))
	}
	for i := range want {
		if meta.Shape[i] != want[i] {
			return v41StageErr(stage, layer,
				fmt.Errorf("%w: tensor %s shape %v, want %v", ErrV41ForwardStage, name, meta.Shape, want))
		}
	}
	return nil
}

// ---- the ordered assembly --------------------------------------------------

// forwardV41 runs the reduced V4.1 text forward over ids (or the state's full
// token history when st is non-nil) and returns per-position hidden states and
// logits. Every stage is checked; a missing stage returns a typed error and no
// logits. It is package-private: the public entry points are Model.Forward and
// Session.Prefill/Step.
func (m *Model) forwardV41(ids []int, st *v41ForwardState) (act *Activations, err error) {
	if err := m.v41ForwardAdmitted(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	// Model.Forward (st == nil) is a pure full-prefill over ids. A session folds
	// ids into its history and recomputes the whole history so Step is consistent
	// with a single longer Forward.
	seq := ids
	runState := st
	committed := false
	if st != nil {
		historyLen := len(st.history)
		st.history = append(st.history, ids...)
		seq = st.history
		// The step-local run state carries the session-owned device gate/up callback
		// (#13358) so the MoE loop offers each pick to the device seam. It is not
		// step-local continuation data and is never written back below.
		runState = &v41ForwardState{history: seq, expertGateUp: st.expertGateUp}
		defer func() {
			if !committed {
				st.history = st.history[:historyLen]
			}
		}()
	}
	if len(seq) == 0 {
		if st != nil {
			committed = true
		}
		return &Activations{}, nil
	}
	for _, id := range seq {
		if id < 0 || id >= cfg.VocabSize {
			return nil, v41StageErr(v41StageEmbedding, -1,
				fmt.Errorf("%w: token id %d out of range [0,%d)", ErrV41ForwardStage, id, cfg.VocabSize))
		}
	}

	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	eps := float32(cfg.RMSNormEps)

	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, err
	}

	// ---- embedding ----
	embed := m.tensor("model.embed_tokens.weight")
	if len(embed) < cfg.VocabSize*H {
		return nil, v41StageErr(v41StageEmbedding, -1,
			fmt.Errorf("%w: embedding table has %d values, want %d", ErrV41ForwardStage, len(embed), cfg.VocabSize*H))
	}
	x := make([][]float32, len(seq))
	for t, id := range seq {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}

	// Persistent mHC streams, one four-stream set per position. On the full path
	// stream 0 carries the live hidden state and streams 1..3 are the reference's
	// persistent residual streams initialized to zero -- DISTINCT from stream 0,
	// not the reduced stand-in's four identical copies. Both paths carry the same
	// [][][]float32 shape; only the initialization and mixing differ, so the
	// reduced arithmetic is byte-identical to the pre-#13009 assembly.
	streams := make([][][]float32, len(seq))
	for t := range streams {
		set := make([][]float32, 4)
		set[0] = x[t]
		for h := 1; h < 4; h++ {
			set[h] = make([]float32, H)
		}
		streams[t] = set
	}

	hcIters := 1
	hcEps := float32(1e-6)
	if cfg.DeepSeekV41 != nil {
		if cfg.DeepSeekV41.HCSinkhornIters > 0 {
			hcIters = cfg.DeepSeekV41.HCSinkhornIters
		}
		if cfg.DeepSeekV41.HCEps > 0 {
			hcEps = float32(cfg.DeepSeekV41.HCEps)
		}
	}
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return nil, err
	}

	act = &Activations{Seq: len(seq), Hidden: [][]float32{flatten(x)}}
	// One scratch for the whole forward: the grouped output projections are read
	// as whole f32 blocks but the layer loop is sequential, so a single reused
	// buffer bounds the wo_a/wo_b term to one layer's worth instead of the
	// 40-layer accumulated churn that OOM-killed the warmup (#13288).
	scratch := &v41ProjScratch{}
	for l := 0; l < cfg.NumLayers; l++ {
		if err := m.v41Layer(l, seq, x, streams, full, hd, nH, H, eps, hcIters, hcEps, routeCfg, runState, scratch); err != nil {
			return nil, err
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	act.Logits = make([][]float32, len(seq))
	for t := 0; t < len(seq); t++ {
		logits, err := m.v41Head(x[t])
		if err != nil {
			return nil, err
		}
		act.Logits[t] = logits
	}
	if st != nil {
		st.layers = runState.layers
		// Completed cross-layer publications are step-local and can be released.
		st.attn = nil
		committed = true
	}
	return act, nil
}

// v41Layer applies one V4.1 decoder layer to x in place, updating the persistent
// mHC streams[t] for each position. tokens carries the ids for the positions in
// x so a declared Engram layer can hash them. The reduced path reproduces the
// pre-#13009 arithmetic exactly (four identical stand-in streams derived from the
// normalized input, and stream 0 of the post-mix written back); the full path
// reads and writes all four DISTINCT persistent streams.
func (m *Model) v41Layer(l int, tokens []int, x [][]float32, streams [][][]float32, full bool, hd, nH, H int, eps float32, hcIters int, hcEps float32, routeCfg v41RouterConfig, st *v41ForwardState, scratch *v41ProjScratch) error {
	cfg := m.Cfg
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}

	// Engram injection happens at the START of the layer, into the residual,
	// before attention and before attn_norm (ds41_graph_before_attention).
	if cfg.DeepSeekV41 != nil {
		for _, eng := range cfg.DeepSeekV41.EngramLayerIDs {
			if eng == l {
				if err := m.v41EngramInject(l, x, tokens, eps); err != nil {
					return err
				}
				break
			}
		}
	}

	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	ffnNorm := m.tensor(layerName(l, "ffn_norm.weight"))
	// mhc.mixes.weight is the last whole-tensor f32 materializer in the forward.
	// It is read once per layer and consumed read-only, so it reads into the
	// forward-scoped scratch rather than allocating a fresh block per layer
	// (#13288 companion seam). Values are byte-identical.
	wMix, err := m.v41MHCMixF32Into(l, scratch.mhc)
	if err != nil {
		return err
	}
	scratch.mhc = wMix
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	// Attention + shared-expert projections are read residency-completely: a
	// quantized serve keeps them in a resident k-quant store (isQuantWeight admits
	// every .attn.wq_a/.attn.wq_b/.attn.wkv/.attn.wo_a/.attn.wo_b and the shared
	// ffn.shared_experts leaves), so a manifest-only m.tensor read panics on a
	// weight admission (residentShape) already admitted (#13276). Each read fails
	// closed with a typed error naming the tensor when absent from every store.
	// The two grouped output projections (attn.wo_a / attn.wo_b) are consumed by
	// V41GroupedOutputProjection as whole f32 blocks, so the streaming matRows
	// twin does not apply. They are read INTO a forward-scoped reused scratch
	// instead of a fresh whole-tensor allocation per layer: the f32 values and
	// layout are byte-identical to the previous v41ProjF32 read, but the peak
	// resident contribution is bounded to one layer instead of the accumulated
	// 40-layer churn that OOM-killed the warmup (#13288).
	woA, err := m.v41ProjF32Into(l, "attn.wo_a.weight", scratch.woA)
	if err != nil {
		return err
	}
	scratch.woA = woA
	woB, err := m.v41ProjF32Into(l, "attn.wo_b.weight", scratch.woB)
	if err != nil {
		return err
	}
	scratch.woB = woB
	// The remaining per-layer projections are applied only through matRows, so
	// they read through the streaming-safe v41ProjMatRows at their use site
	// instead of materializing a whole f32 block here (#2150). Fail closed NOW,
	// with the same typed #13276 refusal, if any is absent from every store, so
	// the refuse happens before any layer arithmetic rather than mid-forward.
	for _, leaf := range []string{
		"attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
		"ffn.gate.weight",
		"ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight",
		"ffn.shared_experts.w2.weight",
	} {
		if !m.hasResidentWeight(layerName(l, leaf)) {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, layerName(l, leaf)))
		}
	}
	// attn.sink is a 1-D per-head vector (not an isQuantWeight matmul leaf), so it
	// stays on the f32 manifest.
	sink := m.tensor(layerName(l, "attn.sink"))
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))

	seq := len(x)
	// ---- mHC coefficient split (one split per layer) ----
	//
	// The mHC mix projection geometry is per-layer (it is a per-layer weight), so
	// resolve it once. The published artifact's hc_attn_fn is the flattened
	// four-stream projection (logical 24 x 4H, stored [4H, 24]); the reference
	// (inference/model.py mHC; v4_flash_oracle_test.go:168) computes it over the
	// four width-H streams laid end to end with a single shared flatten-RMS. The
	// reduced fixture uses the legacy 24 x H single-stream matmul. Both are
	// executed here; only the reduced path's arithmetic is held byte-identical to
	// pre-#13009.
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	hcByPos := make([]v41MHCMix, seq)
	preByPos := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		// The reduced path projects the normalized input through the legacy
		// single-stream [24, H] block; the full path projects the four DISTINCT
		// persistent streams through the flattened 4H residual with one shared RMS.
		// xn is retained for the reduced pre-collapse stand-in below either way.
		xn := rmsnormCfg(x[t], attnNorm, eps, cfg)
		var mixes []float32
		if mhcFlat {
			var err error
			if mixes, err = v41MHCProjectFull(wMix, streams[t], H, eps, mhcTransposed); err != nil {
				return v41StageErr(v41StageMHC, l, err)
			}
		} else {
			mixes = matRows(wMix, xn, v41MHCMixWidth, H)
		}
		mix, err := v41MHCSplit(mixes, mixScale, mixBase, 4, hcIters, hcEps)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		hcByPos[t] = mix
		// The mHC pre-collapse always reads a four-stream set collapsed by the
		// learned pre coefficients. The reduced path reconstructs the reference's
		// stand-in (four identical copies of the normalized input, byte-identical
		// to pre-#13009) rather than reading the persistent set, so its numerics
		// are unchanged. The full path reads the four DISTINCT persistent streams
		// carried into this layer (stream 0 = live hidden, 1..3 = residual).
		streams4 := streams[t]
		if !full {
			streams4 = [][]float32{xn, xn, xn, xn}
		}
		collapsed, err := v41MHCPre(streams4, mix.pre)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		preByPos[t] = collapsed
	}

	// ---- attention: projected q/kv per position, then sparse sink per position ----
	qHeads := make([][]float32, seq) // [t][nH*hd], rotated
	kvRows := make([][]float32, seq) // [t][hd], rotated (single KV head)
	qLatRows := make([][]float32, seq)
	// RoPE geometry, fail-closed. The reference rotates only the LAST
	// qk_rope_head_dim components of each head (the leading qk_nope_head_dim pass
	// through byte-identical) using the interleaved adjacent-pair convention. A
	// head width that cannot carry that slice (a non-positive, odd, or
	// head-wider rope dim) is refused with a typed error rather than silently
	// rotating a wrong sub-vector.
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	// Batched query projections across the prefill panel: residentMatMulBatch
	// reads each attn.wq_a / attn.wq_b weight row ONCE and reuses it across all
	// seq prompt tokens, instead of the per-token GEMV loop re-streaming every
	// weight seq times (#13301). The per-token q_norm and RoPE glue below stays
	// scalar, so this is byte-for-byte the per-token v41ProjMatRows path (the
	// adapter's contract); seq==1 (decode) stays entirely on that scalar path.
	preFlat := make([]float32, seq*H)
	for t := 0; t < seq; t++ {
		copy(preFlat[t*H:(t+1)*H], preByPos[t])
	}
	qLatPanel, err := m.v41ProjPanel(l, "attn.wq_a.weight", preFlat, cfg.QLoraRank, H, seq)
	if err != nil {
		return err
	}
	for t := 0; t < seq; t++ {
		qLat := qLatPanel[t*cfg.QLoraRank : (t+1)*cfg.QLoraRank]
		if full {
			qNorm := m.tensor(layerName(l, "attn.wq_a_norm.weight"))
			if len(qNorm) != cfg.QLoraRank {
				return v41StageErr(v41StageAttention, l,
					fmt.Errorf("%w: q-lora norm has %d values, want %d", ErrV41ForwardStage, len(qNorm), cfg.QLoraRank))
			}
			// Reference: qr = self.q_norm(self.wq_a(x)). The RMSNorm over the
			// q-lora latent keeps its magnitude O(1) before the wq_b projection;
			// omitting it lets ~1e18 activations reach the 512-term attention
			// accumulation and overflow it to +Inf at HeadDim=512 (#13290).
			// Artifact-only: the reduced fixture carries no q_norm leaf and its
			// pre-#13009 arithmetic is unchanged. Written back into the panel row
			// so the batched wq_b panel reads the normed latents.
			copy(qLat, rmsnormCfg(qLat, qNorm, eps, cfg))
		}
	}
	qPanel, err := m.v41ProjPanel(l, "attn.wq_b.weight", qLatPanel, nH*hd, cfg.QLoraRank, seq)
	if err != nil {
		return err
	}
	for t := 0; t < seq; t++ {
		c := preByPos[t]
		q := qPanel[t*nH*hd : (t+1)*nH*hd]
		qLat := qLatPanel[t*cfg.QLoraRank : (t+1)*cfg.QLoraRank]
		// KV latent seam. The full path admits and projects attn.wkv at the
		// published latent rank (v41KVLoraRank = 512); the attention contraction
		// below consumes a per-position row of width hd (head_dim), so the full
		// projection is taken at the published latent rank and sliced to the first
		// hd columns for the sink contraction. On a valid full config
		// hd == v41KVLoraRank == 512, so the slice is a no-op; an hd wider than the
		// latent rank is refused rather than silently mis-read. The reduced fixture
		// keeps its HeadDim-wide projection byte-for-byte.
		var kv []float32
		if full {
			if hd > v41KVLoraRank {
				return v41StageErr(v41StageAttention, l,
					fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
			}
			kvFull, err := m.v41ProjMatRows(l, "attn.wkv.weight", c, v41KVLoraRank, H)
			if err != nil {
				return err
			}
			kv = kvFull[:hd]
		} else {
			kv, err = m.v41ProjMatRows(l, "attn.wkv.weight", c, hd, H)
			if err != nil {
				return err
			}
		}
		// Reference: kv = self.kv_norm(self.wkv(x)) then RoPE on the rope tail
		// (_window_kv). The RMSNorm over the KV vector keeps its magnitude O(1);
		// omitting it lets an unbounded projection reach the 512-term attention
		// accumulation and overflow it to +Inf at HeadDim=512 (#13290). Applied to
		// the full head_dim before the rope tail; artifact-only, so the reduced
		// fixture's pre-#13009 arithmetic is unchanged.
		if full {
			kvNorm := m.tensor(layerName(l, "attn.kv_norm.weight"))
			if len(kvNorm) != v41KVLoraRank {
				return v41StageErr(v41StageAttention, l,
					fmt.Errorf("%w: kv norm has %d values, want %d", ErrV41ForwardStage, len(kvNorm), v41KVLoraRank))
			}
			kv = rmsnormCfg(kv, kvNorm, eps, cfg)
		}
		cos, sin := v41RopeTableForLayer(cfg, l, t)
		for h := 0; h < nH; h++ {
			applyRopeTailInterleaved(q[h*hd:(h+1)*hd], cos, sin, ropeDim)
		}
		applyRopeTailInterleaved(kv, cos, sin, ropeDim)
		qHeads[t] = q
		kvRows[t] = kv
		qLatRows[t] = qLat
	}

	// ---- CED/CSA2 compressor + lightning indexer stages (#13006, #12896) ----
	//
	// A layer declaring CompressRatios[l] > 1 pools its per-position projected KV
	// rows through the CED/CSA2 compressor (v41CompressedRows), and a layer
	// declaring an in-range index source scores its projected index query against
	// those compressed keys and selects rows (v41IndexRows). Both stages execute
	// here and fail closed with a typed *V41ForwardError on malformed geometry; a
	// config that declares an in-range stage without its weights is refused at
	// admission, not here.
	//
	// #12896 maps the sink contraction onto the reference's compressed KV cache:
	// a compressed layer contracts the COMPRESSED stream directly (block-causal
	// visibility, one pooled key/value row per group) instead of expanding the
	// pooled latent back onto causal positions. A declared shared-KV source layer
	// publishes its compressed rows into the session state, and a later reader
	// layer resolves its KV stream from that published source.
	plan, err := v41AttentionPlanFor(cfg, l, m.v41AttentionRolesCached())
	if err != nil {
		return err
	}
	var compressedKV [][]float32
	if plan.Ratio > 1 {
		compressed, err := m.v41CompressedRows(l, plan.Ratio, kvRows, preByPos)
		if err != nil {
			return err
		}
		compressedKV = compressed
	}
	// A reader layer whose KV source precedes it consumes the source's published
	// compressed stream; a source layer publishes its own for later readers.
	sharedKV := compressedKV
	if plan.Role == V41AttentionRoleReader && plan.KVSourceLayer >= 0 && st != nil {
		attn, err := st.attentionState(hd, 8)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		if rows, ok := attn.KVSourceRows(plan.KVSourceLayer); ok && len(rows) > 0 {
			sharedKV = rows
		}
	}
	// The layer's own lightning-index selection for the newest position. A layer
	// that is not an index source computes none (nil).
	localIdx, err := m.v41IndexRows(l, qLatRows[seq-1], preByPos[seq-1], compressedKV)
	if err != nil {
		return err
	}
	// The per-position index list the compressed contraction consumes, flattened
	// [seq][TopKWidth]. A layer that is itself the index source uses its local
	// selection replicated across the causal positions it published it for; a
	// reader layer reuses its source's published top-k selection; a layer with
	// neither passes no list and the contraction falls back to the causal mask.
	indexList, err := m.v41AttentionIndexList(plan, st, localIdx, hd, len(sharedKV), seq)
	if err != nil {
		return err
	}

	// V41AttentionState is the session-owned validation anchor for the projected
	// KV window. It runs no projection and never changes the arithmetic; it fails
	// closed on malformed geometry before the sink contraction reads the rows.
	//
	// Each layer gets independent temporal ownership. Completed source rows stay
	// in the per-forward registry; retained session state contains only the
	// configured window tail and incomplete compressor group.
	if st != nil {
		layerState, err := NewV41AttentionState(hd, 8)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		if err := layerState.seedTemporal(kvRows, plan.Ratio, cfg.windowForLayer(l)); err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		st.setLayerState(l, cfg.NumLayers, layerState)

		registry, err := st.attentionState(hd, 8)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		// A declared source layer publishes its compressed rows and index keys so
		// later readers can resolve them within this same forward pass (the
		// V41AttentionState source-then-consumer ordering).
		updates := m.v41AttentionSourceUpdates(plan, compressedKV, qLatRows)
		if len(updates) > 0 {
			if err := registry.publishUpdates(updates); err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
		}
		// An index source publishes its own per-position top-k selection so a
		// later reader layer can reuse it without recomputing the scoring path.
		// The selection is the source's local index list, one row per published
		// query position.
		if indexSourceAt(cfg.DeepSeekV41, l) && indexList != nil && plan.TopKWidth > 0 {
			rows := make([][]int32, seq)
			for t := range rows {
				rows[t] = append([]int32(nil), indexList[t*plan.TopKWidth:(t+1)*plan.TopKWidth]...)
			}
			if err := registry.PublishTopK(plan.Ratio, rows); err != nil {
				return v41StageErr(v41StageIndexer, l, err)
			}
		}
	}

	scale := cfg.attnScale()
	attnOut := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		// A compressed/shared layer contracts the COMPRESSED KV stream directly:
		// block-causal visibility over pooled group rows, with the lightning
		// indexer's row selection when the layer published one. A per-layer layer
		// keeps the exact per-position causal sink contraction.
		if plan.Ratio > 1 || (plan.Role == V41AttentionRoleReader && len(sharedKV) > 0 && len(sharedKV) < seq) {
			opt := V41AttentionSharedKVOptions{
				Layer: l, Ratio: maxInt(plan.Ratio, 1), Groups: len(sharedKV),
				HeadDim: hd, Heads: nH, Softmax: scale, Sink: sink,
			}
			if indexList != nil && len(indexList) >= (t+1)*plan.topKWidth() {
				// The index source publishes one selection row per query position.
				opt.Idx = indexList[t*plan.topKWidth() : (t+1)*plan.topKWidth()]
				opt.IndexTopK = plan.topKWidth()
				opt.TopK = plan.topKWidth()
			}
			o, err := V41AttentionCompressedForward(qHeads[t], sharedKV, opt)
			if err != nil {
				return err
			}
			projected, err := V41GroupedOutputProjection(o, woA, woB, 1, 1, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
			if err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
			attnOut[t] = projected
			continue
		}
		// #13303: the plain layer's visible keys are its CONFIGURED causal
		// window, not the unconditioned full prefix. A positive
		// Config.Window[l] restricts the row set to the trailing w causal keys
		// max(0,t-w+1)..t; the -1 sentinel (and a nil/short Window, which
		// windowForLayer defaults to -1) keeps the historical full-causal
		// prefix 0..t byte-for-byte. The window changes only WHICH ordered rows
		// are contracted; the score/softmax/value math is untouched.
		window := cfg.windowForLayer(l)
		keys := v41PlainWindowKeys(t, window)
		idx := v41PlainWindowIndexList(keys)
		rows := len(keys)
		flatKV := make([]float32, 0, rows*hd)
		for _, k := range keys {
			flatKV = append(flatKV, kvRows[k]...)
		}
		o, err := V41SparseAttentionSink(qHeads[t], flatKV, sink, idx, V41SparseAttentionSinkOptions{
			B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: rows + 1, N: rows, Softmax: scale,
		})
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		projected, err := V41GroupedOutputProjection(o, woA, woB, 1, 1, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		attnOut[t] = projected
	}

	// ---- MoE: router + shared expert + routed experts ----
	//
	// The router runs once per position and its picks are retained for the
	// contraction loop below (byte-identical to the per-token router it replaces:
	// the same v41ProjMatRows over the same rmsnormCfg(x[t]) and the same
	// v41Route). The layer-scoped expert cache is reset here so its working set is
	// exactly this layer's routed experts (#13296).
	scratch.v41LayerCacheReset()
	if scratch.expertLayerCacheBytes == 0 {
		if v41TestLayerCacheBudgetOverride > 0 {
			scratch.expertLayerCacheBytes = v41TestLayerCacheBudgetOverride
		} else {
			scratch.expertLayerCacheBytes = m.v41LayerExpertCacheBudget()
		}
	}
	perTokenPicks := make([][]routePick, seq)
	for t := 0; t < seq; t++ {
		xn := rmsnormCfg(x[t], ffnNorm, eps, cfg)
		routerLogits, err := m.v41ProjMatRows(l, "ffn.gate.weight", xn, cfg.NumExperts, H)
		if err != nil {
			return err
		}
		picks, err := v41Route(routerLogits, gateBias, routeCfg)
		if err != nil {
			return v41StageErr(v41StageMoE, l, err)
		}
		perTokenPicks[t] = picks
	}

	// ---- routed-expert contraction ----
	//
	// #13304: a multi-token prefill panel is contracted EXPERT-MAJOR when its
	// routed union exceeds the layer cache. The grouped path materializes one
	// expert triple, contracts every (token, slot) row assigned to it, releases
	// it, then replays each token's weighted sum in original slot order -- so a
	// panel whose alternating routes re-read the same experts faults each distinct
	// projection once, and peak retained expert materialization stays one triple.
	// When the union fits the layer cache the two paths are equivalent (the cache
	// served the repeats either way); the single-token path keeps the historical
	// token-major stream byte-for-byte.
	routedByToken := make([][]float32, seq)
	if seq > 1 && !v41ForceTokenMajor {
		if err := m.v41ContractRoutedGrouped(l, x, perTokenPicks, scratch, ffnNorm, eps, cfg, routedByToken); err != nil {
			return err
		}
	} else {
		for t := 0; t < seq; t++ {
			xn := rmsnormCfg(x[t], ffnNorm, eps, cfg)
			routed := make([]float32, H)
			for _, pick := range perTokenPicks[t] {
				stem := "ffn.experts." + itoa(pick.expert)
				// #13358: offer the pick to the session's optional device gate/up
				// seam first. A handled result returns the I-wide fused intermediate
				// from the backend, so the existing host down contraction below runs
				// over it WITHOUT ever materializing the gate/up f32 weights. A
				// decline (or no callback at all) keeps the historical host triple
				// byte-for-byte. A selected device failure must surface.
				if st != nil && st.expertGateUp != nil {
					h, outcome, gerr := st.expertGateUp(l, stem, xn)
					switch outcome {
					case v41GateUpError:
						return v41StageErr(v41StageMoE, l, gerr)
					case v41GateUpHandled:
						w2, err := m.hostExpertDown(l, stem, scratch)
						if err != nil {
							return err
						}
						contractOpen := m.v41NowNanos()
						y := matRows(w2, h, H, cfg.MoEIntermediateSize)
						if contractOpen != 0 {
							m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
						} else {
							m.v41NoteExpertContraction()
						}
						for i := range routed {
							routed[i] += pick.weight * y[i]
						}
						continue
					}
				}
				w1, w3, w2, err := m.v41ExpertTripleInto(l, stem, scratch)
				if err != nil {
					return err
				}
				// #13299: time the routed-expert contraction (the SwiGLU the pick
				// applies) separately from the fault/dequant that produced its
				// weights, so the ledger can attribute the 492 s first token
				// (fak#13294) to scalar contraction vs tier IO. Inert with no clock.
				contractOpen := m.v41NowNanos()
				y := v41SwiGLU(w1, w3, w2, xn, cfg.MoEIntermediateSize, H, cfg)
				if contractOpen != 0 {
					m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
				} else {
					m.v41NoteExpertContraction()
				}
				for i := range routed {
					routed[i] += pick.weight * y[i]
				}
			}
			routedByToken[t] = routed
		}
	}

	for t := 0; t < seq; t++ {
		xn := rmsnormCfg(x[t], ffnNorm, eps, cfg)
		routed := routedByToken[t]
		shared, err := m.v41SharedExpertSwiGLU(l, xn, cfg)
		if err != nil {
			return err
		}
		moe, err := v41SharedExpertAdd(routed, shared, routeCfg)
		if err != nil {
			return v41StageErr(v41StageMoE, l, err)
		}

		// delta = attention + MoE, then the mHC post-mix back into four streams.
		delta := make([]float32, H)
		for i := 0; i < H; i++ {
			delta[i] = attnOut[t][i] + moe[i]
		}
		// The post-mix always reads a four-stream residual set. The reduced path
		// reconstructs the stand-in (four identical copies of the collapsed pre
		// vector) so its arithmetic is unchanged; the full path mixes the four
		// distinct persistent streams.
		residual := [][]float32{preByPos[t], preByPos[t], preByPos[t], preByPos[t]}
		if full {
			residual = streams[t]
		}
		next, err := v41MHCPost(delta, residual, hcByPos[t].post, hcByPos[t].comb)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		if full {
			// Write ALL FOUR post-mix streams back into the persistent set so the
			// next layer reads the updated state; stream 0 is the live hidden.
			for h := 0; h < 4; h++ {
				copy(streams[t][h], next[h])
			}
			copy(x[t], next[0])
		} else {
			// Reduced path: only stream 0 is propagated, exactly as before.
			copy(x[t], next[0])
		}
	}
	return nil
}

// v41SwiGLU is the standard SwiGLU expert/sub-layer: down(silu(w1 x) * w3 x).
// v41SharedExpertSwiGLU is the streaming-safe shared-expert SwiGLU: it applies the
// three shared-expert leaves through v41ProjMatRows (resident store form, bounded
// block scratch) instead of materializing each whole f32 weight, so the shared
// expert adds no uncharged per-layer f32 expansion (#2150). Numerically identical
// to v41SwiGLU over the same weights on the f32-manifest path.
func (m *Model) v41SharedExpertSwiGLU(l int, xn []float32, cfg Config) ([]float32, error) {
	I, H := cfg.MoEIntermediateSize, cfg.HiddenSize
	h1, err := m.v41ProjMatRows(l, "ffn.shared_experts.w1.weight", xn, I, H)
	if err != nil {
		return nil, err
	}
	h3, err := m.v41ProjMatRows(l, "ffn.shared_experts.w3.weight", xn, I, H)
	if err != nil {
		return nil, err
	}
	h := make([]float32, I)
	for i := 0; i < I; i++ {
		h[i] = act(h1[i], cfg) * h3[i]
	}
	y, err := m.v41ProjMatRows(l, "ffn.shared_experts.w2.weight", h, H, I)
	if err != nil {
		return nil, err
	}
	return y, nil
}

func v41SwiGLU(w1, w3, w2, xn []float32, I, H int, cfg Config) []float32 {
	h1 := matRows(w1, xn, I, H)
	h3 := matRows(w3, xn, I, H)
	h := make([]float32, I)
	for i := 0; i < I; i++ {
		h[i] = act(h1[i], cfg) * h3[i]
	}
	return matRows(w2, h, H, I)
}

// v41Head runs the final norm and LM head for one hidden vector. The final norm
// itself is applied inside m.logitsFromHidden (forward.go), which is the single
// shared tail Model.Forward and the parallel twins also end at; this wrapper only
// asserts the final-norm weight is present and the hidden width is admitted, so a
// missing norm weight fails closed before the shared tail runs.
func (m *Model) v41Head(x []float32) ([]float32, error) {
	if !m.has("model.norm.weight") {
		return nil, v41StageErr(v41StageFinalNorm, -1,
			fmt.Errorf("%w: missing model.norm.weight", ErrV41ForwardStage))
	}
	if len(x) != m.Cfg.HiddenSize {
		return nil, v41StageErr(v41StageFinalNorm, -1,
			fmt.Errorf("%w: final norm width %d, want %d", ErrV41ForwardStage, len(x), m.Cfg.HiddenSize))
	}
	if !m.hasWeight("lm_head.weight") && !m.hasWeight("model.embed_tokens.weight") {
		return nil, v41StageErr(v41StageHead, -1,
			fmt.Errorf("%w: no lm_head.weight and no tied embedding", ErrV41ForwardStage))
	}
	return m.logitsFromHidden(x), nil
}

// ---- session entry points --------------------------------------------------

// prefillV41 is the DeepSeek V4.1 branch of Session.Prefill. It fails closed
// (panics) with a typed error before any math when a stage is missing, matching
// the requirePreNorm convention but preserving the #12967 weightless-probe
// errors.Is(err, ErrV41NativeUnsupported) contract.
//
// Phase attribution (#13294 DoD item 1): the whole pass runs under
// V41PhasePrefill and one token is noted PER ID — the prompt delta this call
// was ASKED to ingest — only after the forward succeeded, so a fail-closed
// refusal never inflates the phase's denominator. The defer restores the phase
// to the inert default, so anything the next forward path runs notes nothing.
//
// Prefill first-token feasibility (#13294 follow-on): when the model declares a
// routed-expert fault bandwidth AND a prior pass has measured a per-token fault
// volume, the pass projects its first-token latency BEFORE entering the forward
// and fails closed with the typed ErrV41PrefillLatency refusal when it cannot
// clear the session's watchdog window. On the default (no declared bandwidth, or
// no measurement yet) this is inert and the pass runs byte-for-byte as before —
// the physical `fed6a6a37` rung's 0.1 tok/s prefill is exactly the case this
// turns from a 492 s wedge into a named, pre-emptive "no".
func (s *Session) prefillV41(ids []int) []float32 {
	if err := s.M.v41ForwardAdmitted(); err != nil {
		panic(err)
	}
	if err := s.M.v41PrefillFirstTokenAdmitted(len(ids)); err != nil {
		panic(err)
	}
	s.M.v41SetExpertFaultPhase(V41PhasePrefill)
	defer s.M.v41SetExpertFaultPhase(V41PhaseUnknown)
	act, err := s.M.forwardV41(ids, s.v41State())
	if err != nil {
		panic(err)
	}
	s.M.v41NoteExpertFaultToken(len(ids))
	return lastLogits(act)
}

// stepV41 is the DeepSeek V4.1 branch of Session.Step. The assembly is
// cacheless: Step folds the token into the committed history and recomputes the
// whole history, so its logits are exactly a longer Forward's last-position
// logits (the #12901 prefill/step consistency property).
//
// Phase attribution (#13294): the step runs under V41PhaseDecode and notes ONE
// token — the Step call, i.e. the served decode token the tok/s gate counts —
// so the decode ledger's fault cost honestly includes the whole-history
// recompute this cacheless assembly pays per step. That is exactly the number
// the physical 5 tok/s target needs to attribute.
func (s *Session) stepV41(id int) []float32 {
	if err := s.M.v41ForwardAdmitted(); err != nil {
		panic(err)
	}
	s.M.v41SetExpertFaultPhase(V41PhaseDecode)
	defer s.M.v41SetExpertFaultPhase(V41PhaseUnknown)
	act, err := s.M.forwardV41([]int{id}, s.v41State())
	if err != nil {
		panic(err)
	}
	s.M.v41NoteExpertFaultToken(1)
	return lastLogits(act)
}

// v41State lazily installs the session's V4.1 continuation state. When the
// session's backend can run a routed expert's gate/up projections plus SwiGLU on
// device (the shared q4kExpertInputHAL operation from TICKET-05), the state also
// binds the optional device gate/up callback so the MoE loop can keep the gate/up
// f32 weights off the host and feed the existing host down contraction. A
// non-device session or a backend the shared operation declines leaves the
// callback nil, preserving the historical host triple byte-for-byte.
func (s *Session) v41State() *v41ForwardState {
	if s.v41Forward == nil {
		s.v41Forward = &v41ForwardState{
			expertGateUp: s.v41ExpertGateUpFunc(),
		}
	}
	return s.v41Forward
}

// v41ExpertGateUpFunc binds the shared device gate/up operation (#13357) onto the
// V4.1 session backend, or returns nil when the session has no device backend
// that could execute it. It resolves the two gate/up projection names from the
// routed expert stem exactly as the host triple does, runs the shared
// q4kExpertInputHAL (which itself admits only bias-free SiLU experts whose gate
// and up weights have a device representation the ACTUAL backend can serve), and
// maps its (out, ok) result onto the closed outcome vocabulary. The helper's own
// admission is the gate: a non-device backend, a GELU expert, a biased
// projection, or an unservable dtype returns ok=false here as v41GateUpDeclined,
// never a panic and never a silent host fallback.
func (s *Session) v41ExpertGateUpFunc() v41ExpertGateUpFunc {
	if s == nil || s.Backend == nil || s.M == nil || !s.Backend.Caps().DeviceMemory {
		return nil
	}
	cfg := s.M.Cfg
	return func(layer int, stem string, xn []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		gateName := layerName(layer, stem+".w1.weight")
		upName := layerName(layer, stem+".w3.weight")
		out, ok := q4kExpertInputHAL(s, gateName, upName, xn, cfg.MoEIntermediateSize, cfg.HiddenSize)
		if !ok {
			return nil, v41GateUpDeclined, nil
		}
		return out, v41GateUpHandled, nil
	}
}

func lastLogits(act *Activations) []float32 {
	if act == nil || len(act.Logits) == 0 {
		return nil
	}
	return act.Logits[len(act.Logits)-1]
}
