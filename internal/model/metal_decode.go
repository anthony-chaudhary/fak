//go:build darwin && cgo && fakmetal

package model

// metal_decode.go — wires the dense Q8 model's decode through the GPU-resident decode forward
// (internal/metalgemm/decode.m, issue #67). Where tokenHiddenQ's fast path runs each token's seven
// projections as separate CPU Q8 GEMVs (and metal_q8_on.go would route them as separate per-call
// GPU GEMVs, each launch-bound), this runs the WHOLE token — projections, RMSNorm, RoPE, GQA
// attention over the cache, SwiGLU — in ONE Metal command buffer with the activation resident on
// the GPU. The per-token submit/sync is paid once instead of ~7*nLayers times, the lever the
// MAC-QWEN36 perf diagnosis named. Dense Qwen2.5-family only (no GDN hybrid, no MoE); the resident
// path declines for anything else so the caller falls back to the proven CPU Q8 decode.

import (
	"fmt"
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var (
	metalDecMu    sync.Mutex
	metalDecReady = map[*Model]bool{} // models whose decode topology is registered + uploaded

	// metalDecodeEnvOnce reads FAK_METAL_DECODE once: when set, the resident decode forward runs
	// even without s.Metal, so a run can use CPU prefill (no f16 upload) + GPU-resident Q8 decode —
	// the isolation that keeps a 7B's footprint to one Q8 weight copy on the GPU.
	metalDecodeEnvOnce sync.Once
	metalDecodeEnv     bool
)

func metalDecodeEnabled(s *Session) bool {
	metalDecodeEnvOnce.Do(func() { metalDecodeEnv = os.Getenv("FAK_METAL_DECODE") != "" })
	return s.Metal || metalDecodeEnv
}

// metalDecodeConfig registers this dense model with the resident decode forward once: it uploads
// each layer's seven Q8 projections (q8.m table) + the f16 RMSNorm/bias vectors (gW table) and
// records the geometry. Returns false (so the caller falls back to CPU) for a hybrid/MoE model or
// if any upload is declined. Cheap, off the hot path.
func (m *Model) metalDecodeConfig() bool {
	metalDecMu.Lock()
	defer metalDecMu.Unlock()
	if metalDecReady[m] {
		return true
	}
	cfg := m.Cfg
	if cfg.IsQwen35Hybrid() || cfg.IsMoE() || cfg.QKNorm {
		return false // resident decode forward v0 is the dense, no-QK-norm architecture only
	}
	if m.q8layers == nil {
		return false // Q8 layer cache not built (Quantize not run / not a Q8 session)
	}
	up := func(qt *q8Tensor) int {
		w := metalgemm.UploadQ8(qt.q, qt.d, qt.out, qt.in)
		if w == nil {
			return -1
		}
		return w.ID()
	}
	scale := cfg.attnScale()
	metalgemm.DecodeConfig(cfg.NumLayers, cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads,
		cfg.IntermediateSize, float32(cfg.RMSNormEps), float32(cfg.RopeTheta), scale, cfg.AttentionBias)
	for l := 0; l < cfg.NumLayers; l++ {
		ql := m.q8Layer(l)
		qid, kid, vid, oid := up(ql.qProj), up(ql.kProj), up(ql.vProj), up(ql.oProj)
		gid, uid, did := up(ql.gateProj), up(ql.upProj), up(ql.downProj)
		if qid < 0 || kid < 0 || vid < 0 || oid < 0 || gid < 0 || uid < 0 || did < 0 {
			return false
		}
		inN := metalgemm.UploadVec(ql.inputNorm)
		postN := metalgemm.UploadVec(ql.postNorm)
		qb, kb, vb := -1, -1, -1
		if cfg.AttentionBias {
			qb = metalgemm.UploadVec(ql.qBias)
			kb = metalgemm.UploadVec(ql.kBias)
			vb = metalgemm.UploadVec(ql.vBias)
		}
		metalgemm.DecodeLayer(l, qid, kid, vid, oid, gid, uid, did, inN, postN, qb, kb, vb)
	}
	// Register the final RMSNorm + the Q8 LM head so the resident forward also runs them on the GPU
	// and returns logits directly (no CPU head). Declines (CPU head) if the head upload fails.
	finalNorm := metalgemm.UploadVec(m.tensor("model.norm.weight"))
	headWid := up(m.q8Head())
	if finalNorm < 0 || headWid < 0 {
		return false
	}
	metalgemm.DecodeHead(finalNorm, headWid, cfg.VocabSize)
	metalDecReady[m] = true
	if os.Getenv("FAK_QPROFILE") != "" || os.Getenv("FAK_METAL_DECODE") != "" {
		fmt.Fprintf(os.Stderr, "[metal-decode] GPU-resident Q8 decode forward engaged: %d layers, H=%d nH=%d nKV=%d I=%d bias=%v\n",
			cfg.NumLayers, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize, cfg.AttentionBias)
	}
	return true
}

// metalDecodeLogitsQ8 runs one decode token through the GPU-resident Q8 forward — projections,
// RMSNorm, RoPE, GQA attention over the cache, SwiGLU, the final norm AND the LM head — and returns
// the vocab logits directly. It mirrors token()'s fast Q8 path: gather the per-layer post-RoPE K/V
// context, run the token, append the new K/V row to the cache, advance pos. Returns nil (so the
// caller runs the CPU path) when the resident forward is unavailable or declines for this model.
func (s *Session) metalDecodeLogitsQ8(id, pos int) []float32 {
	m, cfg := s.M, s.M.Cfg
	if !metalDecodeEnabled(s) || !s.Quant || !metalgemm.Available() || !m.metalDecodeConfig() {
		return nil
	}
	H, hd := cfg.HiddenSize, cfg.HeadDim
	w := cfg.NumKVHeads * hd
	L := pos // cache length == the new token's absolute position

	var Kctx, Vctx []float32
	if L > 0 {
		Kctx = make([]float32, cfg.NumLayers*L*w)
		Vctx = make([]float32, cfg.NumLayers*L*w)
		for l := 0; l < cfg.NumLayers; l++ {
			if len(s.Cache.K[l]) < L*w || len(s.Cache.V[l]) < L*w {
				return nil // cache shorter than expected — defer to the CPU path
			}
			copy(Kctx[l*L*w:(l+1)*L*w], s.Cache.K[l][:L*w])
			copy(Vctx[l*L*w:(l+1)*L*w], s.Cache.V[l][:L*w])
		}
	}

	embed := m.embedRows()
	x := make([]float32, H)
	copy(x, embed[id*H:(id+1)*H])
	scaleEmbedInPlace(x, cfg) // Gemma; no-op for Llama/Qwen

	_, newKpost, newV, logits, ok := metalgemm.DecodeStep(x, Kctx, Vctx, L, cfg.NumLayers, w, H, cfg.VocabSize)
	if !ok || logits == nil {
		return nil
	}
	// Append the new token's post-RoPE K and V to the cache (matches the fast Q8 path, which keeps
	// K/V but not Kraw during decode). pos advances by one.
	for l := 0; l < cfg.NumLayers; l++ {
		s.Cache.K[l] = append(s.Cache.K[l], newKpost[l*w:(l+1)*w]...)
		s.Cache.V[l] = append(s.Cache.V[l], newV[l*w:(l+1)*w]...)
	}
	s.Cache.pos = append(s.Cache.pos, pos)
	logitScaleInPlace(logits, cfg) // Cohere/Gemma2; no-op for Llama/Qwen
	return logits
}
