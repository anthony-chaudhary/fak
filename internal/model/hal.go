package model

import (
	"fmt"
	"os"
	"runtime"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// graphBackend is an OPTIONAL capability: a device backend that can capture one token's op
// stream into a CUDA graph and replay it as a single launch. The HAL probes for it; cpu-ref
// does not implement it, so the Reference path is untouched (and stays bit-identical).
type graphBackend interface {
	GraphBegin() bool
	GraphEndLaunch()
}

// NewBackendSession starts a session whose per-token path runs through the
// internal/compute HAL. The legacy optimized path remains NewSession(); this entry
// point is the adoption gate for proving a backend can execute the model loop without
// touching the direct []float32 implementation.
func (m *Model) NewBackendSession(be compute.Backend) *Session {
	if be == nil {
		be = compute.Default()
	}
	if be == nil {
		panic("model: no compute backend registered")
	}
	kv := be.NewKV(compute.KVConfig{
		NumLayers:  m.Cfg.NumLayers,
		NumKVHeads: m.Cfg.NumKVHeads,
		HeadDim:    m.Cfg.HeadDim,
		RopeTheta:  m.Cfg.RopeTheta,
	})
	if kv == nil {
		panic("model: compute backend " + be.Name() + " does not provide KVStore")
	}
	// A reusable captured graph is bound to one session's buffer addresses; reset it so
	// this session captures fresh (no-op on cpu-ref / when graphs are off).
	if gr, ok := be.(interface{ GraphReset() }); ok {
		gr.GraphReset()
	}
	return &Session{M: m, Cache: NewKVCache(m.Cfg), Backend: be, halKV: kv, halW: make(map[string]compute.Tensor)}
}

// Close releases device-resident HAL state owned by this session. Legacy sessions have
// no external residency, so Close is a no-op for them.
func (s *Session) Close() {
	if s == nil || s.Backend == nil {
		return
	}
	if b, ok := s.Backend.(batchBackend); ok {
		b.FlushBatch()
	}
	if s.halW != nil {
		for name, t := range s.halW {
			s.Backend.Free(t)
			delete(s.halW, name)
		}
	}
	if kv, ok := s.halKV.(interface{ Free() }); ok {
		kv.Free()
	}
	if r, ok := s.Backend.(interface{ Recycle() }); ok {
		r.Recycle()
	}
	if t, ok := s.Backend.(interface{ Trim() }); ok {
		t.Trim()
	}
	s.halKV = nil
	s.halW = nil
}

func (s *Session) hostF32(shape []int, data []float32) compute.Tensor {
	src := compute.NewF32(compute.Default(), append([]int(nil), shape...), data)
	return s.Backend.Upload(src, compute.F32)
}

func (s *Session) weightHAL(name string) compute.Tensor {
	if s.halW != nil {
		if t, ok := s.halW[name]; ok {
			return t // already resident on the backend; never re-upload an immutable weight
		}
	}
	meta, ok := s.M.manifest[name]
	if !ok {
		panic("model: missing tensor " + name)
	}
	t := s.hostF32(meta.Shape, s.M.tensor(name))
	if s.halW != nil {
		s.halW[name] = t
	}
	return t
}

func (s *Session) useHALQ8Weights() bool {
	return s.Quant && s.M != nil && s.M.q8w != nil && s.Backend != nil && s.Backend.Caps().UploadDtype
}

var halQ8BatchLayers = initHALQ8BatchLayers()

func initHALQ8BatchLayers() int {
	if s := os.Getenv("FAK_HAL_Q8_BATCH_LAYERS"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 0 {
			return n
		}
	}
	return 2
}

func (s *Session) weightHALQ8(name string, qt *q8Tensor) compute.Tensor {
	key := "q8:" + name
	if s.halW != nil {
		if t, ok := s.halW[key]; ok {
			return t
		}
	}
	if qt == nil {
		panic("model: missing Q8 tensor " + name)
	}
	src := compute.NewQ8(compute.Default(), []int{qt.out, qt.in}, qt.q, qt.d, qBlk)
	t := s.Backend.Upload(src, compute.Q8_0)
	if s.halW != nil {
		s.halW[key] = t
	}
	return t
}

// weightHALQ4K stages a resident Q4_K weight (raw GGUF super-block bytes) onto the backend, the
// Q4_K twin of weightHALQ8. The cuda backend copies the raw super-blocks resident and serves them
// with k_q4k_gemm (the dequant-fused tile, #485); the cpu-ref backend dequants them in its Q4_K
// MatMul. Cached in halW so a device session uploads each weight to VRAM exactly once. This is what
// lets the GLM-DSA forward run its dense projections from a memory-lean Q4_K model on the device —
// the Q4_K majority of a 753B GLM-5.2 on the GPU, with only ~0.56 B/weight resident.
func (s *Session) weightHALQ4K(name string, qt *q4kTensor) compute.Tensor {
	key := "q4k:" + name
	if s.halW != nil {
		if t, ok := s.halW[key]; ok {
			return t
		}
	}
	if qt == nil {
		panic("model: missing Q4_K tensor " + name)
	}
	src := compute.NewQ4K(compute.Default(), []int{qt.out, qt.in}, qt.raw)
	t := s.Backend.Upload(src, compute.Q4_K)
	if s.halW != nil {
		s.halW[key] = t
	}
	return t
}

func (s *Session) matWeightHAL(name string) compute.Tensor {
	if s.useHALQ8Weights() {
		if qt, ok := s.M.q8w[name]; ok {
			return s.weightHALQ8(name, qt)
		}
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
	if useDeviceEmbed {
		embedTable = s.weightHAL("model.embed_tokens.weight")
	} else {
		x = s.hostF32([]int{H}, append([]float32(nil), m.embedRows()[id*H:(id+1)*H]...))
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

	// CUDA-graph fast path: after warm-up (so the buffer pool + weight cache are populated
	// and no cudaMalloc happens during capture), capture this token's whole op stream and
	// replay it as ONE launch — the only way past the proven ~12 tok/s op-per-call WSL
	// floor. Pin the goroutine so the open capture sees a single consistent stream. The x
	// upload above and the logits Read below stay OUTSIDE the captured region.
	gr, canGraph := be.(graphBackend)
	capturing := false
	if canGraph && s.halStep >= 2 {
		runtime.LockOSThread()
		if gr.GraphBegin() {
			capturing = true
		} else {
			runtime.UnlockOSThread()
		}
	}
	finishGraph := func() {
		if capturing {
			gr.GraphEndLaunch() // end capture, instantiate, launch, fence
			runtime.UnlockOSThread()
			capturing = false
		}
		be.Free(x) // per-token input/residual buffer; weights and KV are owned elsewhere.
		s.halStep++
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := func(str string) string { return layerName(l, str) }
		var q, kRaw, v compute.Tensor
		inputNorm := s.weightHAL(p("input_layernorm.weight"))
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

		var g, u compute.Tensor
		postAttnNorm := s.weightHAL(p("post_attention_layernorm.weight"))
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
		if useQ8Weights && batch != nil && halQ8BatchLayers > 0 && (l+1)%halQ8BatchLayers == 0 && l+1 < cfg.NumLayers {
			batch.FlushBatch()
			batch.BeginBatch()
		}
	}

	if mode == halNoLogits {
		finishGraph()
		return compute.Tensor{}, 0
	}
	finalNorm := s.weightHAL("model.norm.weight")
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

