package model

// kimik31_dsa.go — the Kimi-K3.1 Gated DSA layer forward, host oracle
// (ticket 1611-kimi-k31 leaf L6): sparse MLA over paged KV with the
// top-2048 lightning indexer.
//
// K3.1's 24 full-attention layers are "Gated DSA": an absorbed-form MLA
// (the cache holds the compressed latent, K/V projections are folded into
// the query side) whose visible cells are picked per token by a learned
// lightning indexer that keeps its own per-cell key cache alongside the
// attention cache. This file pins the STATE, the canonical tensor names,
// the param struct with fail-closed validation, and the deterministic
// token-level top-k; the forward itself lives in kimik31_dsa_forward.go.
// It extends the L5 conventions (kimik31_kda.go): closed sentinels,
// fail-closed Validate naming the offending field, exported Step/PrefillSeq
// plus unexported head helpers, plain-slice weights so the recurrence stays
// testable without a loaded checkpoint.
//
// REFERENCE (recurrence pinned against concordant sources, fetched 2026-11-07):
//   - llama.cpp master src/models/kimi-k31.cpp build_dsa_layer (lines ~501-646)
//     + graph() (kq_scale_mla = 1/sqrt(n_embd_head_k_mla), prev_sel plumbing,
//     is_indexer_full default all-full at load_arch_hparams line 53):
//       qr   = rmsnorm(wq_a * x, q_a_norm)          // the q latent
//       Qcur = wq_b * qr                            // [NH x (nope+rope)]
//       lat  = wkv_a_mqa * x                        // [kv_lora + rope]
//       kv_c = rmsnorm(lat[:kv_lora], kv_a_norm); k_pe = lat[kv_lora:]
//       NO RoPE anywhere: "mla_use_nope is asserted at conversion time"
//       (line 532) — the rope halves are plain dot products.
//       cache row := concat(kv_c, k_pe); the indexer key
//       ik   = LayerNorm(indexer_attn_k * x, k_norm.w, k_norm.b, f_norm_eps)
//       (TRUE mean-subtracting LayerNorm, not RMS; f_norm_eps = 1e-5,
//       load_arch_hparams line 47) is cached one row per cell, cell-aligned
//       with the attention cache ("the indexer cache sits next to the
//       attention cache and shares its cells", lines 277-282).
//       iq   = indexer_attn_q_b * qr                // [idx_heads x idx_dim]
//       iw   = scale(indexer_proj * x, 1/sqrt(idx_dim*idx_heads))
//       (pre-scaled "so the huge score tensor is not scaled afterwards",
//       line 571)
//       score[t] = sum_h iw[h] * ReLU(sum_d iq[h,d]*ik[t,d])
//       (the non-fused path, lines 593-604: per-head kq -> relu -> *iw ->
//       sum_rows; future cells carry -Inf via the additive causal mask)
//       sel  = top-min(topK, n_cells) of score, then build_top_k_mask:
//       fill -Inf, set selected rows to 0, ADD the causal mask (llama-graph.cpp
//       build_top_k_mask ~line 3487) — visibility = selected AND causally
//       valid.
//       attention (build_attn_mha, llama-graph.cpp ~line 3033, v_mla=wv_b):
//       per head h, q_abs = wk_b[h]^T * q_nope_h (the K absorption, where
//       q_nope_h is head h's raw nope half of Qcur — line 547),
//       s_t = kq_scale * (q_abs . kv_c[t] + q_pe[h] . k_pe[t]),
//       kq_scale = 1/sqrt(nope+rope) applied INSIDE the softmax, F32
//       accumulation, max-subtracted; out_h = sum_t w_t * (wv_b[h] * lat[t]).
//       gate: out *= sigmoid(wqkv_gate * x) when present (TENSOR_NOT_REQUIRED,
//       line 148; the official checkpoint ships self_attn.g_proj on every DSA
//       layer — measured from the HF safetensors index), then wo * out.
//   - ggml/src/ggml-cpu/ops.cpp ggml_compute_forward_top_k_f32 (~line 8777):
//     std::partial_sort by score descending; tie order UNspecified and the
//     result is consumed purely as a SET (build_top_k_mask scatters it into a
//     mask). This oracle therefore ranks by (score desc, INDEX ASC) — a
//     documented deterministic tie-break; any tie-respecting reference agrees
//     on the visible SET whenever tied scores straddle the top-k boundary only
//     in a measure-zero way, and the fixtures pin the exact rule.
//   - SGLang kimi-k3.1-day-0 python/sglang/srt/models/kimi_k3.py
//     (KimiK3MLAAttention, lines ~2544-2702): same gate placement
//     (x * sigmoid(g_proj(x)) before o_proj, TP-local), and the combined
//     kv_b_proj split layout: unflatten(0, (-1, nope+v)).split([nope, v]) —
//     head h owns slab rows [h*(nope+v), (h+1)*(nope+v)), first nope rows K,
//     rest V (post-load hook ~line 3881). llama.cpp stores the same interleave
//     (wkv_b {kv_lora, NH*(nope+v)} line 139; split wk_b/wv_b per-head blocks
//     lines 143-144). The L2 map (internal/ggufload/kimik31_tensors.go) accepts
//     EITHER spelling; this oracle consumes both through WKb/WVb.
//
// PAGING SEAM (leaf L8): the state is flat cell arrays — Latent[nCells x
// (kv_lora+rope)] and IndexK[nCells x idx_dim], one row per attention cell,
// exactly the two caches llama.cpp keeps side by side. L8 maps cells -> pages
// through cachelayout without touching this recurrence; Append is the only
// mutation point.
//
// DEFERRED (later leaves, no surface invented here): the shared-indexer layer
// variant (is_indexer_full == 0 reuses the previous DSA layer's selection —
// K3.1 ships all-full, 5 indexer tensors x 24 DSA layers measured in the
// official index; L8 drives such layers through
// ForwardKimiK31DSAStepWithSelection), quantized weight kernels, and the
// ClassifyForwardPath -> ForwardKimiK31 dispatch (L8 retires the refusal).
// Activation parity vs a runnable reference dump (PF-11) stays gated on a
// staged K3.1 artifact, as the study records; this leaf pins the recurrence
// contract with closed-form fixtures + an independent float64 transcription.

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
)

// KimiK31DSALayerState holds the two cell-aligned caches of one Gated DSA
// layer: the absorbed-MLA latent (normalized compressed part || raw PE part,
// one row per attention cell) and the lightning-indexer keys (one row per
// cell). Flat slices are the paging seam for L8.
type KimiK31DSALayerState struct {
	LatentDim int       // kv_lora_rank + qk_rope (576 for K3.1)
	IdxDim    int       // indexer key dim (128 for K3.1)
	Latent    []float32 // [nCells x LatentDim]
	IndexK    []float32 // [nCells x IdxDim]
}

// NewKimiK31DSALayerState creates a fresh DSA layer state with room for
// capacity cells.
func NewKimiK31DSALayerState(capacity, latentDim, idxDim int) *KimiK31DSALayerState {
	if capacity < 0 {
		capacity = 0
	}
	return &KimiK31DSALayerState{
		LatentDim: latentDim,
		IdxDim:    idxDim,
		Latent:    make([]float32, 0, capacity*latentDim),
		IndexK:    make([]float32, 0, capacity*idxDim),
	}
}

// Append adds one cell: the latent row (concat(rmsnorm(kv_cmpr), k_pe)) and
// the indexer key row (LayerNorm'd). Rows must match the state geometry.
func (s *KimiK31DSALayerState) Append(latent, indexK []float32) {
	if len(latent) != s.LatentDim || len(indexK) != s.IdxDim {
		panic(fmt.Sprintf("model: kimik31 dsa state append: row lengths (%d,%d) != dims (%d,%d)",
			len(latent), len(indexK), s.LatentDim, s.IdxDim))
	}
	s.Latent = append(s.Latent, latent...)
	s.IndexK = append(s.IndexK, indexK...)
}

// Len is the number of cached cells.
func (s *KimiK31DSALayerState) Len() int { return len(s.Latent) / s.LatentDim }

// Reset drops all cells (the backing arrays keep their capacity).
func (s *KimiK31DSALayerState) Reset() {
	s.Latent = s.Latent[:0]
	s.IndexK = s.IndexK[:0]
}

// Clone returns a deep copy of the state.
func (s *KimiK31DSALayerState) Clone() *KimiK31DSALayerState {
	if s == nil {
		return nil
	}
	return &KimiK31DSALayerState{
		LatentDim: s.LatentDim,
		IdxDim:    s.IdxDim,
		Latent:    append([]float32(nil), s.Latent...),
		IndexK:    append([]float32(nil), s.IndexK...),
	}
}

// ByteSize is the live byte count of the two cell arrays.
func (s *KimiK31DSALayerState) ByteSize() int64 {
	return int64(len(s.Latent)+len(s.IndexK)) * 4
}

// KimiK31DSAWeightNames pins the canonical tensor names (the L2 map,
// internal/ggufload/kimik31_tensors.go) one Gated DSA layer's forward
// consumes. L8 resolves these through Model.tensor; the oracle itself takes
// plain slices so the recurrence stays testable without a loaded checkpoint.
// KvBCombined and the KBProj/VBProj pair are alternative spellings of the
// SAME absorbed K/V up-projection (see the file header for the interleave).
type KimiK31DSAWeightNames struct {
	QaProj, QaNorm, QbProj string
	KvAProj, KvANorm       string
	KvBCombined            string
	KBProj, VBProj         string
	OGate                  string // optional (TENSOR_NOT_REQUIRED)
	OProj                  string
	IdxQb, IdxK, IdxProj   string
	IdxKNormW, IdxKNormB   string
}

// KimiK31DSAWeightNamesFor returns the canonical per-layer DSA tensor names
// for decoder layer l.
func KimiK31DSAWeightNamesFor(layer int) KimiK31DSAWeightNames {
	p := "model.layers." + strconv.Itoa(layer) + ".self_attn."
	return KimiK31DSAWeightNames{
		QaProj:      p + "q_a_proj.weight",
		QaNorm:      p + "q_a_layernorm.weight",
		QbProj:      p + "q_b_proj.weight",
		KvAProj:     p + "kv_a_proj_with_mqa.weight",
		KvANorm:     p + "kv_a_layernorm.weight",
		KvBCombined: p + "kv_b_proj.weight",
		KBProj:      p + "k_b_proj.weight",
		VBProj:      p + "v_b_proj.weight",
		OGate:       p + "o_gate.weight",
		OProj:       p + "o_proj.weight",
		IdxQb:       p + "indexer.wq_b.weight",
		IdxK:        p + "indexer.wk.weight",
		IdxProj:     p + "indexer.weights_proj.weight",
		IdxKNormW:   p + "indexer.k_norm.weight",
		IdxKNormB:   p + "indexer.k_norm.bias",
	}
}

// KimiK31DSAParams holds the per-layer weights and scalars for one K3.1
// Gated DSA layer. Geometry fields mirror the state (NumHeads 96, QLoRA 1536,
// KVLoRA 512, QkNope 128, QkRope 64, VDim 128, Hidden 7168, IdxHeads 32,
// IdxDim 128, TopK 2048 for K3.1); the oracle works at any consistent
// geometry so the recurrence stays testable on tiny tensors. Weights are in
// HF row-major orientation [out x in]: y[o] = sum_i W[o*in+i]*x[i].
type KimiK31DSAParams struct {
	NumHeads, QLoRA, KVLoRA, QkNope, QkRope, VDim, Hidden int
	IdxHeads, IdxDim, TopK                                int
	RMSEps, IdxEps                                        float32

	WQa       []float32 // [QLoRA x Hidden]
	WQaNorm   []float32 // [QLoRA]
	WQb       []float32 // [NumHeads*(QkNope+QkRope) x QLoRA]
	WKvA      []float32 // [(KVLoRA+QkRope) x Hidden]
	WKvANorm  []float32 // [KVLoRA]
	WKb       []float32 // [NumHeads*QkNope x KVLoRA]
	WVb       []float32 // [NumHeads*VDim x KVLoRA]
	WO        []float32 // [Hidden x NumHeads*VDim]
	WGate     []float32 // [NumHeads*VDim x Hidden]; nil/empty = no output gate
	WIdxQ     []float32 // [IdxHeads*IdxDim x QLoRA]
	WIdxK     []float32 // [IdxDim x Hidden]
	WIdxProj  []float32 // [IdxHeads x Hidden]
	IdxKNormW []float32 // [IdxDim]
	IdxKNormB []float32 // [IdxDim]
}

// ErrKimiK31DSAParamShape is the closed sentinel for fail-closed DSA param
// shape/domain validation failures.
var ErrKimiK31DSAParamShape = errors.New("model: kimi-k31 DSA param shape mismatch")

// ValidateKimiK31DSAParams fails closed on any malformed params, naming the
// offending field. Call it once when building params from tensors (L8); the
// forward assumes validated params and only guards impossible internal
// inconsistencies.
func ValidateKimiK31DSAParams(p *KimiK31DSAParams) error {
	if p == nil {
		return fmt.Errorf("%w: nil params", ErrKimiK31DSAParamShape)
	}
	if p.NumHeads <= 0 || p.QLoRA <= 0 || p.KVLoRA <= 0 || p.QkNope <= 0 ||
		p.QkRope <= 0 || p.VDim <= 0 || p.Hidden <= 0 ||
		p.IdxHeads <= 0 || p.IdxDim <= 0 || p.TopK <= 0 {
		return fmt.Errorf("%w: bad geometry (heads=%d qlora=%d kvlora=%d nope=%d rope=%d vdim=%d hidden=%d idxheads=%d idxdim=%d topk=%d)",
			ErrKimiK31DSAParamShape, p.NumHeads, p.QLoRA, p.KVLoRA, p.QkNope,
			p.QkRope, p.VDim, p.Hidden, p.IdxHeads, p.IdxDim, p.TopK)
	}
	if p.RMSEps <= 0 {
		return fmt.Errorf("%w: RMSEps %v must be > 0", ErrKimiK31DSAParamShape, p.RMSEps)
	}
	if p.IdxEps <= 0 {
		return fmt.Errorf("%w: IdxEps %v must be > 0", ErrKimiK31DSAParamShape, p.IdxEps)
	}
	headK := p.QkNope + p.QkRope
	if len(p.WQa) != p.QLoRA*p.Hidden {
		return fmt.Errorf("%w: WQa len %d != qlora*hidden %d", ErrKimiK31DSAParamShape, len(p.WQa), p.QLoRA*p.Hidden)
	}
	if len(p.WQaNorm) != p.QLoRA {
		return fmt.Errorf("%w: WQaNorm len %d != qlora %d", ErrKimiK31DSAParamShape, len(p.WQaNorm), p.QLoRA)
	}
	if len(p.WQb) != p.NumHeads*headK*p.QLoRA {
		return fmt.Errorf("%w: WQb len %d != numHeads*headK*qlora %d", ErrKimiK31DSAParamShape, len(p.WQb), p.NumHeads*headK*p.QLoRA)
	}
	if len(p.WKvA) != (p.KVLoRA+p.QkRope)*p.Hidden {
		return fmt.Errorf("%w: WKvA len %d != (kvlora+rope)*hidden %d", ErrKimiK31DSAParamShape, len(p.WKvA), (p.KVLoRA+p.QkRope)*p.Hidden)
	}
	if len(p.WKvANorm) != p.KVLoRA {
		return fmt.Errorf("%w: WKvANorm len %d != kvlora %d", ErrKimiK31DSAParamShape, len(p.WKvANorm), p.KVLoRA)
	}
	if len(p.WKb) != p.NumHeads*p.QkNope*p.KVLoRA {
		return fmt.Errorf("%w: WKb len %d != numHeads*nope*kvlora %d", ErrKimiK31DSAParamShape, len(p.WKb), p.NumHeads*p.QkNope*p.KVLoRA)
	}
	if len(p.WVb) != p.NumHeads*p.VDim*p.KVLoRA {
		return fmt.Errorf("%w: WVb len %d != numHeads*vdim*kvlora %d", ErrKimiK31DSAParamShape, len(p.WVb), p.NumHeads*p.VDim*p.KVLoRA)
	}
	if len(p.WO) != p.Hidden*p.NumHeads*p.VDim {
		return fmt.Errorf("%w: WO len %d != hidden*numHeads*vdim %d", ErrKimiK31DSAParamShape, len(p.WO), p.Hidden*p.NumHeads*p.VDim)
	}
	if len(p.WIdxQ) != p.IdxHeads*p.IdxDim*p.QLoRA {
		return fmt.Errorf("%w: WIdxQ len %d != idxHeads*idxdim*qlora %d", ErrKimiK31DSAParamShape, len(p.WIdxQ), p.IdxHeads*p.IdxDim*p.QLoRA)
	}
	if len(p.WIdxK) != p.IdxDim*p.Hidden {
		return fmt.Errorf("%w: WIdxK len %d != idxdim*hidden %d", ErrKimiK31DSAParamShape, len(p.WIdxK), p.IdxDim*p.Hidden)
	}
	if len(p.WIdxProj) != p.IdxHeads*p.Hidden {
		return fmt.Errorf("%w: WIdxProj len %d != idxHeads*hidden %d", ErrKimiK31DSAParamShape, len(p.WIdxProj), p.IdxHeads*p.Hidden)
	}
	if len(p.IdxKNormW) != p.IdxDim {
		return fmt.Errorf("%w: IdxKNormW len %d != idxdim %d", ErrKimiK31DSAParamShape, len(p.IdxKNormW), p.IdxDim)
	}
	if len(p.IdxKNormB) != p.IdxDim {
		return fmt.Errorf("%w: IdxKNormB len %d != idxdim %d", ErrKimiK31DSAParamShape, len(p.IdxKNormB), p.IdxDim)
	}
	// The output gate is optional (TENSOR_NOT_REQUIRED upstream); when
	// present it must be full-rank over the attention output.
	if p.WGate != nil && len(p.WGate) != p.NumHeads*p.VDim*p.Hidden {
		return fmt.Errorf("%w: WGate len %d != numHeads*vDim*hidden %d",
			ErrKimiK31DSAParamShape, len(p.WGate), p.NumHeads*p.VDim*p.Hidden)
	}
	return nil
}

// SelectKimiK31DSATopK ranks ALL cells by (score descending, index ascending)
// and returns the top min(topK, len(scores)) cell indices sorted ascending.
//
// Tie-break note: ggml_top_k (ggml_compute_forward_top_k_f32) partial-sorts by
// score only, leaving ties in unspecified order, and its output is consumed
// purely as a set (build_top_k_mask scatters the indices into an additive
// mask). Ranking ties by lower index first is a documented deterministic
// refinement that changes the visible set only when equal scores straddle
// the top-k boundary; the closed-form tests pin this exact rule. -Inf scores
// rank last (they enter the selection only when fewer than topK finite cells
// exist, mirroring ggml_top_k filling up to k), and the caller's causal
// intersection re-excludes them — see ForwardKimiK31DSAStep.
func SelectKimiK31DSATopK(scores []float32, topK int) []int {
	n := len(scores)
	if n == 0 || topK <= 0 {
		return nil
	}
	k := topK
	if k > n {
		k = n
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		sa, sb := scores[idx[a]], scores[idx[b]]
		if sa != sb { // NaN compares unequal to everything: it sorts to the end.
			return sa > sb
		}
		return idx[a] < idx[b]
	})
	sel := make([]int, k)
	copy(sel, idx[:k])
	sort.Ints(sel)
	return sel
}
