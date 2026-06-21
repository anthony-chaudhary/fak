package model

import "math"

// kvLayout is the attention-variant seam over the kernel-owned KV cache: it owns
// the per-position bytes a given architecture caches, and the rule for turning
// those bytes back into the per-head (K, V) that causal attention scores against.
//
// Why this exists (issue #25 / MODEL-ARCH-SEAM §3): DeepSeek V2/V3 MLA is the one
// family that changes the cache *data model*, not just the forward pass. Instead of
// caching full per-head K and V (width NumKVHeads*HeadDim each), it caches a single
// low-rank LATENT vector c_KV per position (plus one shared decoupled RoPE key) and
// RECONSTRUCTS K/V from it at read time. The cache therefore needs an interface so
// it can hold EITHER standard per-head K/V (Llama, the default) OR the MLA latent.
//
// The contract is read-side and naive by design: reconstructKV hands attention the
// SAME (k, v) per head that a standard cache would have stored. The standard layout's
// reconstruction is the identity (it already stores per-head K/V); the MLA layout's
// reconstruction up-projects the latent. "Naive" (issue's hard pin) means MLA
// decompresses the latent to full K/V and then runs ordinary attention — which makes
// the path hand-checkable against a tiny reference, the whole point of pinning it.
//
// This seam is deliberately additive: it does NOT rewire the proven f32 hot path in
// blockStep (that stays the verbatim standard per-head loop, gated bit-identical by
// TestRefactorMatchesSerial). The standard kvLayout is a parallel description of that
// same layout, proven byte-identical to the inline loop by TestStandardLayoutNoOp,
// so the abstraction can host a second data model without forking the Llama path.
type kvLayout interface {
	// name identifies the layout for diagnostics.
	name() string

	// cacheStride is the per-position width (in float32s) this layout stores per
	// layer. For the standard layout that is NumKVHeads*HeadDim (one K row; V is the
	// same width and kept alongside). For MLA it is the latent width KVLatentDim plus
	// the shared decoupled-RoPE key width — far smaller than NumKVHeads*HeadDim, which
	// is the ~10-50x per-position shrink MLA buys. It takes the Model because the MLA
	// geometry lives on the Model (the HF-mirrored Config carries no MLA fields).
	cacheStride(m *Model) int

	// reconstructKV returns this layer's per-head, post-RoPE K and per-head V for ONE
	// cached position, given that position's stored row and its absolute position
	// (needed for the decoupled-RoPE rotation under MLA). Both returned slices are
	// laid out [head][HeadDim] flat, width NumKVHeads*HeadDim — exactly what causal
	// GQA attention scores a query head against. The standard layout returns its
	// stored K/V verbatim; the MLA layout up-projects the latent.
	reconstructKV(m *Model, layer int, row []float32, pos int) (k, v []float32)
}

// standardKVLayout is the default Llama/Qwen per-head layout: the cache stores
// post-RoPE K and V directly at width NumKVHeads*HeadDim, so reconstruction is the
// identity. It is a faithful description of the layout blockStep already builds;
// reconstructKV simply hands back the stored rows.
type standardKVLayout struct{}

func (standardKVLayout) name() string { return "standard" }

func (standardKVLayout) cacheStride(m *Model) int { return m.Cfg.NumKVHeads * m.Cfg.HeadDim }

// reconstructKV for the standard layout: the row passed in is the already-stored
// post-RoPE K concatenated with V (width 2*NumKVHeads*HeadDim), split back out. The
// identity reconstruction — no projection, no rotation — which is why the standard
// path stays bit-identical to the inline blockStep attention.
func (standardKVLayout) reconstructKV(m *Model, _ int, row []float32, _ int) (k, v []float32) {
	w := m.Cfg.NumKVHeads * m.Cfg.HeadDim
	return row[:w], row[w : 2*w]
}

// MLAConfig holds the low-rank MLA projection geometry. It is intentionally separate
// from the (HF-mirrored) Config so the standard Config stays untouched and the MLA
// fixture is fully self-contained — no DeepSeek download, no new Config fields leaking
// into the Llama path. A non-nil Model.MLA selects the MLA layout for that model.
//
// Geometry (DeepSeek V2/V3 naming):
//   - KVLatentDim   : width of the cached compressed latent c_KV (DeepSeek ~512).
//   - RopeDim       : width of the shared decoupled RoPE key k_R (DeepSeek ~64).
//   - DownKV[h*K+..] : W_DKV, projects hidden (H) -> latent (KVLatentDim).
//   - UpK / UpV      : W_UK / W_UV, project latent (KVLatentDim) -> NumKVHeads*HeadDim.
//   - DownR          : W_KR, projects hidden (H) -> RopeDim (the pre-RoPE decoupled key).
//
// The cached row per position is [c_KV (KVLatentDim) | k_R_raw (RopeDim)]: the latent
// is copied verbatim on eviction (O2: zero re-derivation), and only the small k_R is
// re-rotated at a new position. Storing k_R PRE-RoPE mirrors Kraw in the standard
// cache, so a single rotation at read time is bit-exact to a fresh prefill.
type MLAConfig struct {
	KVLatentDim int
	RopeDim     int

	DownKV []float32 // [KVLatentDim, H]
	UpK    []float32 // [NumKVHeads*HeadDim, KVLatentDim]
	UpV    []float32 // [NumKVHeads*HeadDim, KVLatentDim]
	DownR  []float32 // [RopeDim, H]
}

// mlaKVLayout reconstructs per-head K/V naively: up-project the cached latent to full
// K and V, then rotate the decoupled RoPE key into the front RopeDim lanes of every
// head's K. The query side concatenates a matching RoPE part (built in mlaProject), so
// scoring a query head against this reconstructed K is ordinary dot-product attention.
type mlaKVLayout struct{}

func (mlaKVLayout) name() string { return "mla" }

// cacheStride: the cached row is the latent c_KV followed by the pre-RoPE decoupled
// key k_R_raw. This is what shrinks ~10-50x vs NumKVHeads*HeadDim per position.
func (mlaKVLayout) cacheStride(m *Model) int {
	return m.MLA.KVLatentDim + m.MLA.RopeDim
}

// reconstructKV is the NAIVE MLA read path: c_KV @ W_UK / W_UV gives full per-head
// K/V; then the shared k_R = RoPE(k_R_raw, pos) is broadcast into the front RopeDim
// lanes of every head's K. V is returned verbatim from the up-projection (V is not
// rotated). The returned K width is NumKVHeads*HeadDim with the convention that each
// head's first RopeDim lanes carry the decoupled RoPE key and the remaining lanes
// carry the content key — the same split the query uses.
func (mlaKVLayout) reconstructKV(m *Model, _ int, row []float32, pos int) (k, v []float32) {
	cfg := m.Cfg
	mla := m.MLA
	nKV, hd := cfg.NumKVHeads, cfg.HeadDim
	w := nKV * hd

	latent := row[:mla.KVLatentDim]
	rRaw := row[mla.KVLatentDim : mla.KVLatentDim+mla.RopeDim]

	// Up-project the latent to full per-head K and V (the "decompress" of
	// decompress-then-attend).
	k = mlaMatRows(mla.UpK, latent, w, mla.KVLatentDim)
	v = mlaMatRows(mla.UpV, latent, w, mla.KVLatentDim)

	// Rotate the shared decoupled RoPE key at this absolute position and broadcast it
	// into the front RopeDim lanes of every head's K. RopeDim is even (it is rotated as
	// RopeDim/2 pairs); the rotation is one applyRopeRow at `pos`, bit-exact to prefill.
	kR := append([]float32(nil), rRaw...)
	cos, sin := ropeRowFromInv(mlaRopeInv(mla.RopeDim, cfg.RopeTheta), pos)
	applyRopeRow(kR, cos, sin)
	for h := 0; h < nKV; h++ {
		copy(k[h*hd:h*hd+mla.RopeDim], kR)
	}
	return k, v
}

// mlaProject builds, for one position's (normed) hidden vector, the per-position MLA
// cache row to append: [c_KV (down-projected latent) | k_R_raw (pre-RoPE decoupled
// key)]. It is the WRITE side that pairs with mlaKVLayout.reconstructKV — the latent
// is stored verbatim (copied unchanged on eviction; zero re-derivation, O2) and the
// small k_R is stored pre-RoPE so a single rotation at read/reposition time is
// bit-exact to a fresh prefill (mirroring how the standard cache keeps Kraw).
//
// The matching query is built separately (the test's buildMLAQuery): the query for
// head h is [ RoPE(q_R, pos) (RopeDim) | q_C_h (HeadDim-RopeDim) ], so a full dot of
// query and reconstructed key is q_R·k_R + q_C·k_C — ordinary attention over the
// reconstructed K, which is exactly what makes the naive form checkable.
func (m *Model) mlaProject(xn []float32, _ int) (cacheRow []float32) {
	mla := m.MLA
	cKV := mlaMatRows(mla.DownKV, xn, mla.KVLatentDim, m.Cfg.HiddenSize)
	rRaw := mlaMatRows(mla.DownR, xn, mla.RopeDim, m.Cfg.HiddenSize)
	cacheRow = make([]float32, 0, mla.KVLatentDim+mla.RopeDim)
	cacheRow = append(cacheRow, cKV...)
	cacheRow = append(cacheRow, rRaw...)
	return cacheRow
}

// mlaRopeInv builds RoPE inverse frequencies for the decoupled-RoPE width RopeDim,
// independent of HeadDim. It mirrors invFreq but over RopeDim so the small shared key
// k_R rotates correctly regardless of the content head width.
func mlaRopeInv(ropeDim int, theta float64) []float64 {
	half := ropeDim / 2
	inv := make([]float64, half)
	for j := 0; j < half; j++ {
		inv[j] = 1.0 / math.Pow(theta, float64(2*j)/float64(ropeDim))
	}
	return inv
}

// modelLayout selects the kvLayout for a model: MLA when MLA geometry is present,
// otherwise the default standard per-head layout. Llama/Qwen models have MLA==nil and
// therefore keep the standard layout, unchanged.
func (m *Model) modelLayout() kvLayout {
	if m.MLA != nil {
		return mlaKVLayout{}
	}
	return standardKVLayout{}
}

// attendOne runs causal GQA attention for a SINGLE query position against a set of
// cached rows, reconstructing each position's per-head K/V through the given layout.
// It is the read path both layouts share: standard hands back stored K/V, MLA
// up-projects the latent. q is the full per-head query [NumHeads*HeadDim]; rows[j] is
// position j's cached row; positions[j] is its absolute RoPE position. The result is
// the attention output [NumHeads*HeadDim] — the input to o_proj.
//
// This is the naive, checkable form: per query head h, score against every cached
// position's reconstructed K head (h/GroupSize), softmax, weighted-sum the V head.
func attendOne(m *Model, layout kvLayout, layer int, q []float32, rows [][]float32, positions []int) []float32 {
	cfg := m.Cfg
	nH, hd := cfg.NumHeads, cfg.HeadDim
	grp := cfg.GroupSize()
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	nPos := len(rows)

	// Reconstruct every cached position's per-head K/V once (naive: full decompress).
	ks := make([][]float32, nPos)
	vs := make([][]float32, nPos)
	for j := 0; j < nPos; j++ {
		ks[j], vs[j] = layout.reconstructKV(m, layer, rows[j], positions[j])
	}

	out := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := q[h*hd : (h+1)*hd]
		scores := make([]float32, nPos)
		for j := 0; j < nPos; j++ {
			kh := ks[j][kvh*hd : (kvh+1)*hd]
			scores[j] = dot(qh, kh) * scale
		}
		softmaxInPlace(scores)
		oh := out[h*hd : (h+1)*hd]
		for j := 0; j < nPos; j++ {
			vh := vs[j][kvh*hd : (kvh+1)*hd]
			wj := scores[j]
			for d := 0; d < hd; d++ {
				oh[d] += wj * vh[d]
			}
		}
	}
	return out
}

// mlaMatRows is the serial, in-order matrix-row product used by the MLA layout
// reconstruction. It reduces with the plain `dot` (single accumulator, ascending i)
// so the naive MLA path is deterministic and the hand reference in the test reduces
// over indices in the IDENTICAL order — the property that makes the naive form
// bit-checkable. (The standard layout never calls this: its reconstruction is the
// identity, so it stays byte-for-byte the inline blockStep K/V.)
func mlaMatRows(w, x []float32, out, in int) []float32 {
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		y[o] = dot(w[o*in:o*in+in], x)
	}
	return y
}
