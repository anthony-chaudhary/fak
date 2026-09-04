package model

import (
	"fmt"
	"os"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// graphBackend is an OPTIONAL capability: a device backend that can capture one token's op
// stream into a CUDA graph and replay it as a single launch. The HAL probes for it; cpu-ref
// does not implement it, so the Reference path is untouched (and stays bit-identical).
type graphBackend interface {
	GraphBegin() bool
	GraphEndLaunch()
	// GraphAbort ends and DISCARDS an open capture without launching it. The HAL calls it
	// only on the panic path (an unforeseen mid-capture cudaMalloc, or a KV append past the
	// preallocated capacity) so the shared capture stream is not left in capture mode — which
	// would fail every subsequent op and cascade the whole serve. The session whose capture
	// was aborted is left inconsistent (its KV host bookkeeping advanced without the captured
	// device writes executing) and must be dropped by the caller's decode-boundary recovery.
	GraphAbort()
}

// NewBackendSession starts a session whose per-token path runs through the
// internal/compute HAL. The legacy optimized path remains NewSession(); this entry
// point is the adoption gate for proving a backend can execute the model loop without
// touching the direct []float32 implementation.
func (m *Model) NewBackendSession(be compute.Backend) *Session {
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		panic(err)
	}
	return s
}

// NewBackendSessionChecked is the fail-closed constructor for a compute-HAL
// session. It returns a typed architecture/backend refusal before any weight
// upload or generic operator runs. NewBackendSession preserves the established
// panic-on-invalid-construction API by wrapping this checked entry point.
func (m *Model) NewBackendSessionChecked(be compute.Backend) (*Session, error) {
	if be == nil {
		be = compute.Default()
	}
	if be == nil {
		panic("model: no compute backend registered")
	}
	if err := m.ValidateBackendForwardPath(be); err != nil {
		return nil, err
	}
	kv := newHALKVStore(be, m.Cfg)
	if kv == nil {
		panic("model: compute backend " + be.Name() + " does not provide KVStore")
	}
	if !m.holdModelWeights() {
		if free, ok := kv.(interface{ Free() }); ok {
			free.Free()
		}
		panic("model: weights are closing or closed")
	}
	// A reusable captured graph is bound to one session's buffer addresses; reset it so
	// this session captures fresh (no-op on cpu-ref / when graphs are off).
	if gr, ok := be.(interface{ GraphReset() }); ok {
		gr.GraphReset()
	}
	s := &Session{
		M: m, Cache: NewKVCache(m.Cfg), Backend: be, halKV: kv,
		halW:             make(map[string]compute.Tensor),
		borrowedHALW:     make(map[string]struct{}),
		modelWeightsHeld: true,
	}
	s.initMixedQKV()
	if m.Cfg.IsQwen35Hybrid() {
		// ValidateBackendForwardPath already proved this exact structural capability and
		// path identity before KV or weight allocation.
		s.initQwen35HALState(be.(Qwen35GDNBackend))
	}
	return s, nil
}

// Close releases session-owned device state and its model-weight lifetime reference.
func (s *Session) Close() {
	if s == nil {
		return
	}
	s.closeOnce.Do(func() {
		if stats := s.q4kMLPOutputSlabStats; stats.Calls != 0 {
			fmt.Fprintf(os.Stderr, "{\"schema\":%q,\"engine\":\"fak-native\",\"backend\":\"metal\",\"quant\":\"q4_k\",\"calls\":%d,\"allocations\":%d,\"reuses\":%d,\"high_water_bytes\":%d}\n",
				q4kMLPOutputSlabReceiptSchema, stats.Calls, stats.Allocations, stats.Reuses, stats.HighWaterBytes)
		}
		// The grouped-Q4_K readback slab is session-owned even on the legacy (Backend=nil) path.
		s.q4kMLPOutputSlab = nil
		s.mixedQKV = mixedQKVSession{}
		// Sequence auxiliary state can be owned by a native capability even when
		// Backend is nil, so its teardown is outside the compute-HAL branch.
		s.closeQwen35HALState()
		if s.Backend != nil {
			s.halClosed = true
			if gr, ok := s.Backend.(interface{ GraphReset() }); ok {
				gr.GraphReset()
			}
			if b, ok := s.Backend.(batchBackend); ok {
				b.FlushBatch()
			}
			if s.halW != nil {
				for name, t := range s.halW {
					if _, borrowed := s.borrowedHALW[name]; !borrowed {
						s.Backend.Free(t)
					}
					delete(s.halW, name)
					delete(s.borrowedHALW, name)
				}
			}
			// Routed-expert weights served by the bounded ring are NOT in halW (that is the point), so they
			// need their own teardown or their device handles would outlive the session. A SHARED ring
			// (R7/#5618) is the exception and the reason the branch exists: its residency belongs to the
			// (model, device) pair, so this conversation ending must DETACH and leave every byte in place for
			// the agents still using it. Freeing here would page out a peer's working set — and Free a handle
			// it is about to multiply against.
			if s.sharedRing != nil {
				s.sharedRing.Detach(s)
			} else if s.expertRing != nil {
				s.expertRing.freeAll()
				s.expertRing = nil
			}
			if kv, ok := s.halKV.(interface{ Free() }); ok {
				kv.Free()
			}
			if s.v4Expert != nil {
				_ = s.v4Expert.Close()
				s.v4Expert = nil
			}
			if r, ok := s.Backend.(interface{ Recycle() }); ok {
				r.Recycle()
			}
			if t, ok := s.Backend.(interface{ Trim() }); ok {
				t.Trim()
			}
			s.halKV = nil
			s.halW = nil
			s.borrowedHALW = nil
		}
		if s.modelWeightsHeld {
			s.modelWeightsHeld = false
			s.M.releaseWeightSession()
		}
	})
}

type classedUploadBackend interface {
	UploadClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor
}

func uploadHostF32Class(be compute.Backend, shape []int, data []float32, class compute.MemoryClass, site string) compute.Tensor {
	src := compute.NewF32(compute.Default(), append([]int(nil), shape...), data)
	if b, ok := be.(classedUploadBackend); ok {
		return b.UploadClass(src, compute.F32, class, site)
	}
	return be.Upload(src, compute.F32)
}

func (s *Session) uploadHostF32(shape []int, data []float32, class compute.MemoryClass, site string) compute.Tensor {
	return uploadHostF32Class(s.Backend, shape, data, class, site)
}

func (s *Session) cachedImmutableWeight(sessionKey, residencyKey string, stage func() compute.Tensor) compute.Tensor {
	if s.halW != nil {
		if t, ok := s.halW[sessionKey]; ok {
			return t // already resident on the backend; never re-upload an immutable weight
		}
	}
	if resident := s.M.immutableDeviceWeights(s.Backend); resident != nil {
		weight := resident.getOrStage(residencyKey, stage)
		// Keep a session-local borrowed-handle memo so steady decode retains the original
		// plain-map fast path. Close distinguishes borrowed model residents from private
		// routed-expert fallbacks before freeing.
		if s.halW != nil {
			s.halW[sessionKey] = weight
			if s.borrowedHALW == nil {
				s.borrowedHALW = make(map[string]struct{})
			}
			s.borrowedHALW[sessionKey] = struct{}{}
		}
		return weight
	}
	t := stage()
	if s.halW != nil {
		s.halW[sessionKey] = t
	}
	return t
}

func (s *Session) weightHAL(name string) compute.Tensor {
	stage := func() compute.Tensor {
		meta, ok := s.M.manifest[name]
		if !ok {
			panic("model: missing tensor " + name)
		}
		return s.uploadHostF32(meta.Shape, s.M.tensor(name), compute.MemoryWeights, "hal-weight "+name)
	}
	return s.cachedImmutableWeight(name, "f32:"+name, stage)
}

func (s *Session) useHALQ8Weights() bool {
	return s.Quant && s.M != nil && s.M.q8w != nil && s.Backend != nil && s.Backend.Caps().UploadDtype
}

// useHALQ4KWeights reports whether this session stages its eligible matmul weights as RAW
// resident Q4_K (the dequant-fused device GEMM, #485/#949) instead of Q8 — true for a Q4_K
// model (s.Q4K, m.q4kw populated) on a device backend that consumes quantized uploads. The
// normalize-sensitive minority (q/k_proj, Q6_K) stays in q8w and the q8 branch serves it; only
// the q4kw-resident majority (FFN gate/up/down, v/o_proj, lm_head) routes through Q4_K — the
// same split glmDsaWeightHAL uses for GLM-DSA, now reachable for a plain dense arch on CUDA.
func (s *Session) useHALQ4KWeights() bool {
	return s.Q4K && s.M != nil && s.M.q4kw != nil && s.Backend != nil && s.Backend.Caps().UploadDtype
}

// useHALF16Weights reports whether this session narrows its matmul weight uploads to device F16
// (the #484 fp16 HGEMM) at H2D instead of Q8/Q4_K — true for an F16-tagged session (s.F16) on a
// device backend that honors the upload dtype (Caps().UploadDtype). Unlike Q8/Q4_K it needs no
// resident prequantized map (m.q8w/m.q4kw): F16 is derived from the same host f32 weight the f32
// weightHAL uploads, just narrowed to __half at the copy. cpu-ref reports UploadDtype=false, so a
// reference session falls through to the f32 path and the Reference stays bit-identical.
func (s *Session) useHALF16Weights() bool {
	return s.F16 && s.M != nil && s.Backend != nil && s.Backend.Caps().UploadDtype
}

// useHALKQuantWeights reports whether this session stages its resident dense k-quant weights
// (Q5_K/Q6_K in m.kqw) directly onto a device backend that consumes quantized uploads (#9352).
func (s *Session) useHALKQuantWeights() bool {
	return s.M != nil && s.M.kqw != nil && s.Backend != nil && s.Backend.Caps().UploadDtype
}

var halQ8BatchLayers = envIntMin("FAK_HAL_Q8_BATCH_LAYERS", 0, 2)

// weightHALStaged caches one resident quantized weight on the backend under key,
// uploading it as dtype exactly once. On a cache hit the cached tensor is returned
// without ever calling mk; only a miss builds the src via mk (which carries the
// per-format construction + its own nil-tensor guard) and uploads it. Shared by
// weightHALQ8 / weightHALQ4K.
func (s *Session) weightHALStaged(key string, mk func() compute.Tensor, dtype compute.Dtype) compute.Tensor {
	stage := func() compute.Tensor { return s.Backend.Upload(mk(), dtype) }
	return s.cachedImmutableWeight(key, key, stage)
}

// requireTensorPresent panics with the uniform "missing weight" message the
// weightHALQ8/weightHALQ4K staged builders both raise when their source tensor is
// nil. kind names the tensor family (e.g. "Q8", "Q4_K") in the message.
func requireTensorPresent(missing bool, kind, name string) {
	if missing {
		panic("model: missing " + kind + " tensor " + name)
	}
}

func (s *Session) weightHALQ8(name string, qt *q8Tensor) compute.Tensor {
	return s.weightHALStagedBounded("q8:"+name, name, func() compute.Tensor {
		requireTensorPresent(qt == nil, "Q8", name)
		return compute.NewQ8(compute.Default(), []int{qt.out, qt.in}, qt.q, qt.d, qBlk)
	}, compute.Q8_0, q8ResidentBytes(qt))
}

// weightHALQ4K stages a resident Q4_K weight (raw GGUF super-block bytes) onto the backend, the
// Q4_K twin of weightHALQ8. The cuda backend copies the raw super-blocks resident and serves them
// with k_q4k_gemm (the dequant-fused tile, #485); the cpu-ref backend dequants them in its Q4_K
// MatMul. Cached in halW so a device session uploads each weight to VRAM exactly once. This is what
// lets the GLM-DSA forward run its dense projections from a memory-lean Q4_K model on the device —
// the Q4_K majority of a 753B GLM-5.2 on the GPU, with only ~0.56 B/weight resident.
func (s *Session) weightHALQ4K(name string, qt *q4kTensor) compute.Tensor {
	return s.weightHALStagedBounded("q4k:"+name, name, func() compute.Tensor {
		requireTensorPresent(qt == nil, "Q4_K", name)
		raw, err := qt.materializeRaw()
		if err != nil {
			panic("model: lazy Q4_K read " + name + ": " + err.Error())
		}
		return compute.NewQ4K(compute.Default(), []int{qt.out, qt.in}, raw)
	}, compute.Q4_K, q4kResidentBytes(qt))
}

// weightHALKQuant stages verbatim Q5_K/Q6_K GGUF bytes. CUDA dequantizes in the
// GEMV tile, so residency remains at checkpoint size and warm calls reuse the same allocation.
func (s *Session) weightHALKQuant(name string, qt *kQuantTensor) compute.Tensor {
	if s.Backend == nil || qt == nil {
		panic("model: weightHALKQuant requires backend and tensor: " + name)
	}
	var dt compute.Dtype
	var host func() compute.Tensor
	switch qt.kind {
	case kindQ5K:
		dt = compute.Q5_K
		host = func() compute.Tensor { return compute.NewQ5K(compute.Default(), []int{qt.out, qt.in}, qt.raw) }
	case kindQ6K:
		dt = compute.Q6_K
		host = func() compute.Tensor { return compute.NewQ6K(compute.Default(), []int{qt.out, qt.in}, qt.raw) }
	default:
		panic("model: unsupported resident expert k-quant: " + qt.kind.String())
	}
	return s.weightHALStagedBounded("kquant-raw:"+name, name, host, dt, kQuantResidentBytes(qt))
}

// supportsRoutedExpertKQuant is the explicit optional capability used both by
// dispatch and execution. Keeping the predicate in one place prevents an earlier
// host fast path from silently intercepting a backend that can keep expert weights
// resident.
func (s *Session) supportsRoutedExpertKQuant() bool {
	if s == nil || s.Backend == nil {
		return false
	}
	routed, ok := s.Backend.(interface{ SupportsRoutedExpertKQuant() bool })
	return ok && routed.SupportsRoutedExpertKQuant()
}

// expertWeight is one resolved routed-expert projection: its canonical tensor name plus whichever
// tier can serve it — a resident quantized representation the model carries (q4 or kq), or the
// R5/#5616 checkpoint descriptor (ck) that says how to fault it out of the fused slab. Exactly one
// of the three is non-nil.
//
// ck carries a DESCRIPTOR, not bytes, and that distinction is the rung: a checkpoint-served weight
// is resolved, keyed, sized and held without any IO, and its stride is read only when the bounded
// device ring actually misses it.
type expertWeight struct {
	name string
	q4   *q4kTensor
	kq   *kQuantTensor
	ck   *checkpointStaging
}

// halKey is the dtype-prefixed staging key this weight lands under — the same key weightHALQ4K /
// weightHALKQuant use for halW and for the routed-expert ring, so a hold names exactly the resident
// the staging created. A checkpoint-served weight reports the key its own tier derived, which is
// built to the same rule, so a resident and a faulted copy of one projection are one ring entry.
func (w expertWeight) halKey() string {
	if w.ck != nil {
		return w.ck.key
	}
	if w.q4 != nil {
		return "q4k:" + w.name
	}
	return "kquant-raw:" + w.name
}

// resolveExpertWeight resolves one routed-expert projection to whichever tier is actually reachable:
// the resident stores first, then — only when both are empty for this name — the R5/#5616 checkpoint
// tier. The resident stores WIN, so a fully-resident checkpoint never touches the tier and this is
// the pre-R5 path byte-for-byte; a model with no tier answers exactly as it did before.
//
// Resolution reads NOTHING in every case. That is what lets both consumers decide an expert is
// reachable before staging any of it, and what keeps a checkpoint expert already resident in the
// ring free of checkpoint IO.
//
// It is the single resolution both consumers share — the demand path (expertSwiGLUHAL, below) and
// the R3 activated-set prefetch (expert_ring_prefetch.go) — because a prefetch that resolved
// weights by a different rule than the GEMM would stage residents the GEMM then missed.
func (s *Session) resolveExpertWeight(name string) (expertWeight, bool) {
	if s == nil || s.M == nil {
		return expertWeight{}, false
	}
	if qt := s.M.q4kw[name]; qt != nil {
		return expertWeight{name: name, q4: qt}, true
	}
	if qt := s.M.kqw[name]; qt != nil {
		return expertWeight{name: name, kq: qt}, true
	}
	ck, ok := s.M.expertCheckpoint.staging(name)
	if !ok {
		return expertWeight{}, false
	}
	return expertWeight{name: name, ck: ck}, true
}

// expertSwiGLUHAL keeps routed expert gate/up/down projections and the SwiGLU
// activation on Backend. It admits only projections with an honest resident Q4_K
// or one-time-staged F16 Q5_K/Q6_K representation.
func (s *Session) expertSwiGLUHAL(gateName, upName, downName string, x []float32) ([]float32, bool) {
	if s == nil || s.halW == nil || !s.supportsRoutedExpertKQuant() {
		return nil, false
	}
	weights := make([]expertWeight, 3)
	for i, name := range []string{gateName, upName, downName} {
		var found bool
		weights[i], found = s.resolveExpertWeight(name)
		if !found {
			return nil, false
		}
	}

	// Raw k-quant residency is already included in the model plan; no expanded F16 copy is created.
	// A checkpoint-served projection goes through the SAME bounded staging as a resident one — same
	// key, same dtype, same byte accounting — so the ring cannot tell the two apart and a hit costs
	// no checkpoint read.
	resident := func(w expertWeight) compute.Tensor {
		switch {
		case w.q4 != nil:
			return s.weightHALQ4K(w.name, w.q4)
		case w.kq != nil:
			return s.weightHALKQuant(w.name, w.kq)
		default:
			return s.weightHALStagedBounded(w.ck.key, w.name, w.ck.mk, w.ck.dt, w.ck.bytes)
		}
	}
	// One expert is THREE weights used together, so under a bounded routed-expert ring
	// (Session.ExpertRingBytes, #5611) staging `up` could evict `gate` and Free a handle the GEMMs
	// below still need. Hold each weight for the rest of this expert's computation and release it on
	// return; without a ring every hold is a no-op and this is byte-for-byte the previous path.
	//
	// Under a SHARED ring (R7/#5618) staging and holding must be ONE span, not two statements: a peer
	// agent's stage landing between them could evict the handle just returned and Free it — a
	// use-after-free the per-session ring could not produce, because nothing else could touch it. The
	// span is reentrant, so weightHALStagedBounded's own span nests inside this one, and it is a
	// no-op under the per-session default.
	staged := make([]compute.Tensor, len(weights))
	for i, w := range weights {
		r := s.routedExpertRing(w.name)
		if r == nil {
			staged[i] = resident(w)
			continue
		}
		done := s.ringEnter(r)
		staged[i] = resident(w)
		r.hold(w.halKey())
		done()
		defer func(key string) {
			done := s.ringEnter(r)
			r.release(key)
			done()
		}(w.halKey())
	}
	gateW, upW, downW := staged[0], staged[1], staged[2]

	hostX := compute.NewF32(compute.Default(), []int{len(x)}, append([]float32(nil), x...))
	dx := s.Backend.Upload(hostX, compute.F32)
	defer s.Backend.Free(dx)
	gate := s.Backend.MatMul(gateW, dx)
	defer s.Backend.Free(gate)
	up := s.Backend.MatMul(upW, dx)
	defer s.Backend.Free(up)
	act := s.Backend.SwiGLU(gate, up)
	defer s.Backend.Free(act)
	out := s.Backend.MatMul(downW, act)
	defer s.Backend.Free(out)
	return s.Backend.Read(out), true
}

// weightHALF16 stages the host f32 weight `name` onto the backend narrowed to device F16, the
// F16 twin of weightHALQ8/weightHALQ4K. Unlike those there is no resident prequantized source: the
// same manifest f32 tensor the f32 weightHAL uploads is handed to Upload with compute.F16, and an
// UploadDtype-capable backend narrows it to __half at H2D (#484). Cached under the f16 key so a
// device session uploads each weight to VRAM exactly once, like the Q8/Q4_K staged builders.
func (s *Session) weightHALF16(name string) compute.Tensor {
	return s.weightHALStaged("f16:"+name, func() compute.Tensor {
		meta, ok := s.M.manifest[name]
		if !ok {
			panic("model: missing tensor " + name)
		}
		return compute.NewF32(compute.Default(), append([]int(nil), meta.Shape...), s.M.tensor(name))
	}, compute.F16)
}

func (s *Session) matWeightHAL(name string) compute.Tensor {
	if s.useHALQ8Weights() {
		if qt, ok := s.M.q8w[name]; ok {
			return s.weightHALQ8(name, qt)
		}
	}
	if s.useHALQ4KWeights() {
		if qt, ok := s.M.q4kw[name]; ok {
			return s.weightHALQ4K(name, qt)
		}
	}
	if s.useHALKQuantWeights() {
		if qt, ok := s.M.kqw[name]; ok && qt != nil {
			return s.weightHALKQuant(name, qt)
		}
	}

	if s.useHALF16Weights() {
		return s.weightHALF16(name)
	}
	return s.weightHAL(name)
}

func (s *Session) lmHeadHAL() compute.Tensor {
	if s.M.has("lm_head.weight") {
		return s.weightHAL("lm_head.weight")
	}
	return s.weightHAL("model.embed_tokens.weight")
}

func (s *Session) lmHeadMatHAL() compute.Tensor {
	if s.useHALQ8Weights() {
		name := s.M.headName()
		if qt, ok := s.M.q8w[name]; ok {
			return s.weightHALQ8(name, qt)
		}
	}
	if s.useHALQ4KWeights() {
		name := s.M.q4kHeadName()
		if qt, ok := s.M.q4kw[name]; ok {
			return s.weightHALQ4K(name, qt)
		}
	}
	if s.useHALKQuantWeights() {
		name := "lm_head.weight"
		if _, ok := s.M.kqw[name]; !ok {
			name = "model.embed_tokens.weight"
		}
		if qt, ok := s.M.kqw[name]; ok && qt != nil {
			return s.weightHALKQuant(name, qt)
		}
	}
	if s.useHALF16Weights() {
		name := "lm_head.weight"
		if !s.M.has(name) {
			name = "model.embed_tokens.weight"
		}
		return s.weightHALF16(name)
	}
	return s.lmHeadHAL()
}

type batchBackend interface {
	BeginBatch()
	FlushBatch()
}

type ropeInPlaceBackend interface {
	RoPEInPlace(x compute.Tensor, pos, nHeads, headDim int, theta float64) compute.Tensor
}

type kvRoPEAppender interface {
	AppendKVRoPE(layer int, kRaw, val compute.Tensor, pos, nHeads, headDim int, theta float64)
}

type matMulAddBackend interface {
	MatMulAddInPlace(dst, w, x compute.Tensor)
}

type matMulArgmaxBackend interface {
	MatMulArgmax(w, x compute.Tensor) int
}

type rmsNormMatMulArgmaxBackend interface {
	RMSNormMatMulArgmax(w, x, normWeight compute.Tensor, eps float32) int
}

type rmsNormMatMulBackend interface {
	RMSNormMatMul(w, x, normWeight compute.Tensor, eps float32) compute.Tensor
}

type swigluMatMulAddBackend interface {
	SwiGLUMatMulAddInPlace(dst, w, gate, up compute.Tensor)
}

type matMul2Backend interface {
	MatMul2(w0, w1, x compute.Tensor) (compute.Tensor, compute.Tensor)
}

type matMul3Backend interface {
	MatMul3(wq, wk, wv, x compute.Tensor) (compute.Tensor, compute.Tensor, compute.Tensor)
}

type rmsNormMatMul2Backend interface {
	RMSNormMatMul2(w0, w1, x, normWeight compute.Tensor, eps float32) (compute.Tensor, compute.Tensor)
}

type rmsNormMatMul3Backend interface {
	RMSNormMatMul3(wq, wk, wv, x, normWeight compute.Tensor, eps float32) (compute.Tensor, compute.Tensor, compute.Tensor)
}

type weightBufferCapBackend interface {
	MaxWeightBufferBytes() int64
}

func deviceEmbeddingTableFits(be compute.Backend, shape []int) bool {
	capper, ok := be.(weightBufferCapBackend)
	if !ok || capper.MaxWeightBufferBytes() <= 0 {
		return true
	}
	elements := int64(1)
	for _, dimension := range shape {
		if dimension <= 0 || elements > (1<<63-1)/int64(dimension) {
			return false
		}
		elements *= int64(dimension)
	}
	return elements <= capper.MaxWeightBufferBytes()/int64(compute.F32.Bytes())
}

type embeddingRowBackend interface {
	EmbeddingRow(table compute.Tensor, row int) compute.Tensor
}

type halOutputMode uint8

const (
	halNoLogits halOutputMode = iota
	halFullLogits
	halArgmax
)

// tokenHALOutput is the f32 decode/prefill step expressed through compute.Backend whole-op
// calls. With cpu-ref it must be byte-identical to tokenHidden+head; with a future Approx
// backend it is held to that backend's argmax/cosine gate, never the exact rungs. The output
// mode lets prompt ingestion skip discarded logits and greedy decode use a device argmax.
func (s *Session) tokenHALOutput(id, pos int, mode halOutputMode) (compute.Tensor, int) {
	s.ensureOpenBackendSession()
	finishLineage := s.beginHALTokenLineageWrite([]int{id})
	defer finishLineage()
	be := s.Backend
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	scale := cfg.attnScale()

	var x compute.Tensor
	var embedTable compute.Tensor
	embedder, useDeviceEmbed := be.(embeddingRowBackend)
	if meta, ok := m.manifest["model.embed_tokens.weight"]; ok && !deviceEmbeddingTableFits(be, meta.Shape) {
		useDeviceEmbed = false
	}
	if useDeviceEmbed {
		embedTable = s.weightHAL("model.embed_tokens.weight")
	} else {
		x = s.uploadHostF32([]int{H}, append([]float32(nil), m.embedRows()[id*H:(id+1)*H]...), compute.MemoryActivation, "hal-token-input")
	}
	var batch batchBackend
	if b, ok := be.(batchBackend); ok {
		batch = b
		batch.BeginBatch()
		defer batch.FlushBatch()
	}
	if useDeviceEmbed {
		x = embedder.EmbeddingRow(embedTable, id)
	}
	rope := func(x compute.Tensor, pos, nHeads, headDim int, theta float64) compute.Tensor {
		if r, ok := be.(ropeInPlaceBackend); ok {
			return r.RoPEInPlace(x, pos, nHeads, headDim, theta)
		}
		return be.RoPE(x, pos, nHeads, headDim, theta)
	}
	addMatMul := func(dst, w, x compute.Tensor) {
		if fused, ok := be.(matMulAddBackend); ok && w.Dtype == compute.F32 {
			fused.MatMulAddInPlace(dst, w, x)
			return
		}
		y := be.MatMul(w, x)
		be.AddInPlace(dst, y)
	}
	useQ8Weights := s.useHALQ8Weights()

	// CUDA-graph fast path: capture this token's whole op stream and replay it as ONE launch
	// — the only way past the proven ~12 tok/s op-per-call WSL floor. Pin the goroutine so the
	// open capture sees a single consistent stream. The x upload above and the logits Read
	// below stay OUTSIDE the captured region.
	//
	// We capture ONLY a logits-producing step (the steady decode topology the kept exec graph
	// reuses via cudaGraphExecUpdate) and ONLY after one such step has already run UNCAPTURED
	// (halLogitsWarm). That uncaptured step pools every weight (model.norm, lm_head) and every
	// transient size (incl. the vocab-wide logits buffer); without it the first captured logits
	// step hit a fresh cudaMalloc mid-capture and crashed the serve (#932). A noLogits prefill
	// step is left uncaptured: its topology differs from the decode graph, and it never warms
	// the logits path, so it must not be the step that instantiates the reused exec.
	gr, canGraph := be.(graphBackend)
	// Qwen hybrid decode is graph-replay safe when device-side partial RoPE and
	// query-gate split backends are available with unscaled RoPE and pre-warmed state
	// (s.qwen35HybridGraphSafe()). Unsafe hybrid configurations gracefully fall back
	// to eager execution.
	gpuLayers, isSplit := s.validateDenseGPULayers()
	if cfg.IsQwen35Hybrid() && !s.qwen35HybridGraphSafe() {
		if resetter, ok := be.(interface{ GraphReset() }); ok {
			resetter.GraphReset()
		}
		s.halLogitsWarm = false
	}
	canGraph = canGraph && !isSplit && (!cfg.IsQwen35Hybrid() || s.qwen35HybridGraphSafe())
	capturing := false
	if canGraph && mode != halNoLogits && s.halLogitsWarm {
		runtime.LockOSThread()
		if gr.GraphBegin() {
			capturing = true
		} else {
			runtime.UnlockOSThread()
		}
	}
	// Panic-safety: if anything below unwinds (e.g. an unforeseen mid-capture cudaMalloc, or a
	// KV append past the preallocated capacity) while a capture is still open, end+discard it so
	// the shared stream is not left in capture mode and the PROCESS survives for the next
	// request. finishGraph clears `capturing` on the normal path, making this a no-op then.
	defer func() {
		if capturing {
			gr.GraphAbort()
			runtime.UnlockOSThread()
			capturing = false
		}
	}()
	finishGraph := func() {
		if capturing {
			gr.GraphEndLaunch() // end capture, instantiate, launch, fence
			runtime.UnlockOSThread()
			capturing = false
		}
		be.Free(x) // per-token input/residual buffer; weights and KV are owned elsewhere.
		s.halStep++
		if mode != halNoLogits {
			// A full logits path just ran; its weights + transient sizes are now pooled, so the
			// next logits step is safe to capture.
			s.halLogitsWarm = true
		}
	}

	targetLayers := cfg.NumLayers
	if isSplit {
		targetLayers = gpuLayers
	}

	for l := 0; l < targetLayers; l++ {
		p := func(str string) string { return layerName(l, str) }
		if cfg.IsQwen35Hybrid() {
			if cfg.isLinearAttnLayer(l) {
				s.qwen35LinearHAL(l, x, eps)
			} else {
				s.qwen35FullAttentionHAL(l, pos, x, eps, scale, grp)
			}
		} else {
			var q, kRaw, v compute.Tensor
			inputNorm := s.normWeightHAL(p("input_layernorm.weight"))
			if fused, ok := be.(rmsNormMatMul3Backend); ok && !cfg.AttentionBias {
				q, kRaw, v = fused.RMSNormMatMul3(
					s.matWeightHAL(p("self_attn.q_proj.weight")),
					s.matWeightHAL(p("self_attn.k_proj.weight")),
					s.matWeightHAL(p("self_attn.v_proj.weight")),
					x,
					inputNorm,
					eps,
				)
			} else {
				xn := be.RMSNorm(x, inputNorm, eps)
				if fused, ok := be.(matMul3Backend); ok && !cfg.AttentionBias {
					q, kRaw, v = fused.MatMul3(
						s.matWeightHAL(p("self_attn.q_proj.weight")),
						s.matWeightHAL(p("self_attn.k_proj.weight")),
						s.matWeightHAL(p("self_attn.v_proj.weight")),
						xn,
					)
				} else {
					q = be.MatMul(s.matWeightHAL(p("self_attn.q_proj.weight")), xn)
					kRaw = be.MatMul(s.matWeightHAL(p("self_attn.k_proj.weight")), xn)
					v = be.MatMul(s.matWeightHAL(p("self_attn.v_proj.weight")), xn)
				}
			}
			if cfg.AttentionBias {
				be.AddBias(q, s.weightHAL(p("self_attn.q_proj.bias")))
				be.AddBias(kRaw, s.weightHAL(p("self_attn.k_proj.bias")))
				be.AddBias(v, s.weightHAL(p("self_attn.v_proj.bias")))
			}
			q = rope(q, pos, nH, hd, cfg.RopeTheta)
			if appender, ok := s.halKV.(kvRoPEAppender); ok {
				appender.AppendKVRoPE(l, kRaw, v, pos, nKV, hd, cfg.RopeTheta)
			} else {
				k := be.RoPE(kRaw, pos, nKV, hd, cfg.RopeTheta)
				s.halKV.AppendKV(l, kRaw, k, v, pos)
			}

			attnOut := be.Attention(q, s.halKV, l, true, grp, scale)
			addMatMul(x, s.matWeightHAL(p("self_attn.o_proj.weight")), attnOut)
		}

		postAttnNorm := s.normWeightHAL(p("post_attention_layernorm.weight"))
		if cfg.IsDeepSeekV4() {
			if err := s.applyV4ExpertHAL(l, id, x, postAttnNorm, eps); err != nil {
				panic(err)
			}
		} else {
			var g, u compute.Tensor
			if fused, ok := be.(rmsNormMatMul2Backend); ok {
				g, u = fused.RMSNormMatMul2(
					s.matWeightHAL(p("mlp.gate_proj.weight")),
					s.matWeightHAL(p("mlp.up_proj.weight")),
					x,
					postAttnNorm,
					eps,
				)
			} else {
				xn2 := be.RMSNorm(x, postAttnNorm, eps)
				if fused, ok := be.(matMul2Backend); ok {
					g, u = fused.MatMul2(
						s.matWeightHAL(p("mlp.gate_proj.weight")),
						s.matWeightHAL(p("mlp.up_proj.weight")),
						xn2,
					)
				} else {
					g = be.MatMul(s.matWeightHAL(p("mlp.gate_proj.weight")), xn2)
					u = be.MatMul(s.matWeightHAL(p("mlp.up_proj.weight")), xn2)
				}
			}
			if fused, ok := be.(swigluMatMulAddBackend); ok {
				fused.SwiGLUMatMulAddInPlace(x, s.matWeightHAL(p("mlp.down_proj.weight")), g, u)
			} else {
				ff := be.SwiGLU(g, u)
				addMatMul(x, s.matWeightHAL(p("mlp.down_proj.weight")), ff)
			}
		}
		if useQ8Weights && batch != nil && halQ8BatchLayers > 0 && (l+1)%halQ8BatchLayers == 0 && l+1 < cfg.NumLayers {
			batch.FlushBatch()
			batch.BeginBatch()
		}
	}

	if isSplit {
		if batch != nil {
			batch.FlushBatch()
		}
		xHost := be.Read(x)
		be.Free(x)
		s.halStep++
		if mode != halNoLogits {
			s.halLogitsWarm = true
		}

		var xf []float32
		if cfg.usesMLAMoELayout() {
			needLast := mode != halNoLogits
			out, err := s.decodeBandGLMDsa(id, xHost, gpuLayers, cfg.NumLayers, pos, false, needLast)
			if err != nil {
				panic(err)
			}
			if !needLast {
				s.Cache.appendPosition(pos, id)
				return compute.Tensor{}, 0
			}
			xf = out
		} else {
			tap := s.activeTap()
			if tap != nil && !tap.wants(pos) {
				tap = nil
			}
			prevTap := s.tapActive
			s.tapActive = tap
			defer func() { s.tapActive = prevTap }()

			for l := gpuLayers; l < cfg.NumLayers; l++ {
				cos, sin := ropeRowForLayer(cfg, l, pos)
				xHost = s.blockStep(l, pos, xHost, cos, sin, s.hostMatKernel())
			}
			s.Cache.appendPosition(pos, id)
			if tap != nil {
				tap.writeMeta(cfg, H, pos)
			}
			s.rememberTargetHidden(pos, id, xHost)
			if mode == halNoLogits {
				return compute.Tensor{}, 0
			}
			xf = s.M.finalNorm(xHost)
		}

		logits := s.headResident(xf)
		if mode == halArgmax {
			return compute.Tensor{}, argmaxF32(logits)
		}
		return s.uploadHostF32([]int{len(logits)}, logits, compute.MemoryActivation, "hal-split-logits"), 0
	}

	return s.halFinalLogits(x, mode, capturing, useQ8Weights, finishGraph)
}

// halFinalLogits runs the post-layer final RMSNorm + lm_head and finishes the CUDA-graph
// capture, returning the same (logits, nextToken) contract as the inline tail it replaces:
// halNoLogits short-circuits; halArgmax prefers a fused norm+matmul+argmax (then a fused
// matmul+argmax, then host Argmax); the logits modes prefer a fused norm+matmul. The fused
// argmax paths stay gated on !capturing and (for the F32 matmul) an F32 head, and every
// fused-norm path stays gated on !useQ8Weights, exactly as before.
func (s *Session) halFinalLogits(x compute.Tensor, mode halOutputMode, capturing, useQ8Weights bool, finishGraph func()) (compute.Tensor, int) {
	be := s.Backend
	eps := float32(s.M.Cfg.RMSNormEps)
	if mode == halNoLogits {
		finishGraph()
		return compute.Tensor{}, 0
	}
	finalNorm := s.normWeightHAL("model.norm.weight")
	if mode == halArgmax {
		if fused, ok := be.(rmsNormMatMulArgmaxBackend); ok && !capturing && !useQ8Weights {
			next := fused.RMSNormMatMulArgmax(s.lmHeadHAL(), x, finalNorm, eps)
			finishGraph()
			return compute.Tensor{}, next
		}
	}
	if mode != halArgmax {
		if fused, ok := be.(rmsNormMatMulBackend); ok && !useQ8Weights {
			logits := fused.RMSNormMatMul(s.lmHeadHAL(), x, finalNorm, eps)
			finishGraph()
			return logits, 0
		}
	}
	hidden := be.RMSNorm(x, finalNorm, eps)
	head := s.lmHeadMatHAL()
	if mode == halArgmax {
		if fused, ok := be.(matMulArgmaxBackend); ok && !capturing && head.Dtype == compute.F32 {
			next := fused.MatMulArgmax(head, hidden)
			finishGraph()
			return compute.Tensor{}, next
		}
		logits := be.MatMul(head, hidden)
		finishGraph()
		next := be.Argmax(logits)
		return compute.Tensor{}, next
	}
	logits := be.MatMul(head, hidden)
	finishGraph()
	return logits, 0
}

// tokenHAL preserves the public Step/Prefill contract: return the full host logits.
func (s *Session) tokenHAL(id, pos int) []float32 {
	be := s.Backend
	logits, _ := s.tokenHALOutput(id, pos, halFullLogits)
	out := be.Read(logits)
	if out == nil {
		panic(fmt.Sprintf("model: compute backend %s returned unreadable logits", be.Name()))
	}
	s.recycleHALToken()
	return out
}

// tokenHALNoLogits advances backend KV state for a token whose distribution is discarded.
func (s *Session) tokenHALNoLogits(id, pos int) {
	s.tokenHALOutput(id, pos, halNoLogits)
	s.recycleHALToken()
}

// tokenHALArgmax keeps greedy decode on the backend: full logits stay device-resident and
// only the winning token id crosses the host boundary.
func (s *Session) tokenHALArgmax(id, pos int) int {
	_, next := s.tokenHALOutput(id, pos, halArgmax)
	s.recycleHALToken()
	return next
}

func (s *Session) recycleHALToken() {
	// Token boundary: a device backend recycles this token's transient op buffers (the KV
	// cache has already copied what it keeps; weights are cached separately). No-op on
	// cpu-ref. This is what keeps steady-state decode off the per-op cudaMalloc path.
	if r, ok := s.Backend.(interface{ Recycle() }); ok {
		r.Recycle()
	}
}

func (s *Session) prefillHAL(ids []int, wantLogits bool) []float32 {
	s.ensureOpenBackendSession()
	if len(ids) == 0 {
		return nil
	}
	last := len(ids) - 1
	for i, id := range ids {
		if i == last && wantLogits {
			return s.tokenHAL(id, s.halKV.Len())
		}
		s.tokenHALNoLogits(id, s.halKV.Len())
	}
	return nil
}
