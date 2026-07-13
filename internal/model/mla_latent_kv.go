package model

import "fmt"

// mla_latent_kv.go — the latent-resident MLA KV cache (issue #4364, epic #4352 colibri).
//
// DeepSeek V2/V3/V4 (and GLM-5.2) Multi-head Latent Attention does not cache full
// per-head K and V. Per token per layer it needs only the compressed low-rank latent
// c_KV (KVLatentDim) plus ONE shared decoupled-RoPE key k_R (RopeDim), and it
// RECONSTRUCTS per-head K/V from that pair at read time. This file makes that residency
// a first-class data structure: LatentMLAKVCache stores exactly the [c_KV | k_R_raw] row
// mlaProject writes — never the materialized per-head K/V — and hands attention the same
// reconstructed (K, V) a naive per-head cache would have stored, via the production MLA
// read path (mlaKVLayout.reconstructKV). Nothing here re-implements the projection math;
// it reuses the write helper (mlaProject) and the read helper (reconstructKV) landed for
// #4356's absorbed decode, so the resident cache cannot drift from the standard MLA path.
//
// The memory-first win, at DeepSeek-V3 geometry (kv_lora_rank d_c=512, rope d_h^R=64 vs a
// full-MHA baseline n_h=128, d_h=128): resident is d_c + d_h^R = 576 f32/token/layer, the
// materialized per-head K/V baseline is 2*n_h*d_h = 32768, i.e. 32768/576 ~= 56.9x — the
// "~57x smaller than materialized per-head K/V" this cache exists to buy. The golden test
// (mla_latent_kv_test.go) pins reconstruct-from-latent == the reference materialized K/V
// bit-exactly on a small synthetic case; CompressionRatio() reports the shrink numerically.

// LatentMLAKVCache is the memory-first MLA KV cache: per stored token it holds ONLY the
// compressed MLA latent c_KV (width KVLatentDim) and one shared pre-RoPE decoupled key
// k_R_raw (width RopeDim) — the exact [c_KV | k_R_raw] row mlaProject produces — plus that
// token's absolute RoPE position (needed to re-rotate k_R at read). Per-head K and V are
// never stored; they are reconstructed on demand by ReconstructKV. It is a plain, pure-Go
// value type: no locks, no weights of its own, just a borrowed *Model for the geometry and
// projection weights the reconstruction reads.
type LatentMLAKVCache struct {
	m    *Model      // borrowed: carries Cfg (head geometry + RoPE theta) and MLA (projection weights)
	rows [][]float32 // rows[t] = [c_KV (KVLatentDim) | k_R_raw (RopeDim)] — the ONLY per-token bytes kept
	pos  []int       // pos[t] = absolute RoPE position of token t
}

// NewLatentMLAKVCache builds an empty latent-resident cache bound to an MLA model. It
// requires m.MLA != nil: a non-MLA (Llama/Qwen) model has no latent to be resident on and
// keeps the standard per-head kvLayout, so binding one here is a programming error. Callers
// gate exactly as modelLayout() does (m.MLA != nil).
func NewLatentMLAKVCache(m *Model) *LatentMLAKVCache {
	if m == nil || m.MLA == nil {
		panic("model: NewLatentMLAKVCache requires an MLA model (m.MLA != nil)")
	}
	return &LatentMLAKVCache{m: m}
}

// rowWidth is the resident per-token row width: latent + decoupled-key.
func (c *LatentMLAKVCache) rowWidth() int { return c.m.MLA.KVLatentDim + c.m.MLA.RopeDim }

// AppendHidden compresses one token's (normed) hidden state into its MLA latent and
// pre-RoPE decoupled key and stores that compressed row, returning the new token index.
// It is the write side: it calls the same mlaProject the standard MLA path uses, so a
// token stored here is byte-identical to one the production MLA cache would hold.
func (c *LatentMLAKVCache) AppendHidden(xn []float32, pos int) int {
	return c.AppendRow(c.m.mlaProject(xn, pos), pos)
}

// AppendRow stores an already-compressed MLA cache row [c_KV | k_R_raw] at absolute
// position pos — e.g. a row copied verbatim on eviction (the O2 "zero re-derivation"
// property MLAConfig documents: the latent is portable, only k_R re-rotates at read). It
// copies the row so the caller may reuse its buffer, and refuses a wrong-width row.
func (c *LatentMLAKVCache) AppendRow(row []float32, pos int) int {
	if w := c.rowWidth(); len(row) != w {
		panic(fmt.Sprintf("model: LatentMLAKVCache row width %d, want %d (KVLatentDim+RopeDim)", len(row), w))
	}
	t := len(c.rows)
	c.rows = append(c.rows, append([]float32(nil), row...))
	c.pos = append(c.pos, pos)
	return t
}

// Len is the number of tokens resident in the cache.
func (c *LatentMLAKVCache) Len() int { return len(c.rows) }

// Position returns token t's absolute RoPE position.
func (c *LatentMLAKVCache) Position(t int) int { return c.pos[t] }

// Row returns a copy of token t's stored compressed row [c_KV | k_R_raw]. It is a copy so
// the caller cannot mutate resident state; the same rows feed attendOne / attendOneAbsorbed
// directly (their rows[j] IS this row), which is why the resident form needs no separate
// decompress-to-attend adapter.
func (c *LatentMLAKVCache) Row(t int) []float32 {
	return append([]float32(nil), c.rows[t]...)
}

// Rows returns copies of all resident compressed rows and their positions, in token order
// — the exact (rows, positions) pair attendOne / attendOneAbsorbed consume. This is how the
// latent-resident cache plugs into the existing MLA read paths without materializing K/V.
func (c *LatentMLAKVCache) Rows() (rows [][]float32, positions []int) {
	rows = make([][]float32, len(c.rows))
	for t := range c.rows {
		rows[t] = append([]float32(nil), c.rows[t]...)
	}
	positions = append([]int(nil), c.pos...)
	return rows, positions
}

// ReconstructKV rebuilds token t's per-head, post-RoPE K and per-head V from its stored
// compressed row — the read side of the latent-resident pair. Both slices are [head][HeadDim]
// flat, width NumKVHeads*HeadDim, exactly what causal GQA attention scores against. It
// delegates to the production MLA read path (mlaKVLayout.reconstructKV), so the reconstruction
// is identical to the standard MLA cache's, not a second implementation that could drift.
func (c *LatentMLAKVCache) ReconstructKV(t int) (k, v []float32) {
	return mlaKVLayout{}.reconstructKV(c.m, 0, c.rows[t], c.pos[t])
}

// ResidentFloatsPerToken is the f32 count this cache actually stores per token per layer:
// the latent width plus the shared decoupled-key width (KVLatentDim + RopeDim).
func (c *LatentMLAKVCache) ResidentFloatsPerToken() int { return c.rowWidth() }

// MaterializedFloatsPerToken is the f32 count a naive per-head K/V cache would store per
// token per layer instead: full K plus full V, each NumKVHeads*HeadDim wide. This is the
// baseline the latent-resident cache shrinks away from.
func (c *LatentMLAKVCache) MaterializedFloatsPerToken() int {
	return 2 * c.m.Cfg.NumKVHeads * c.m.Cfg.HeadDim
}

// ResidentFloats is the total resident f32 footprint across every stored token.
func (c *LatentMLAKVCache) ResidentFloats() int { return c.Len() * c.ResidentFloatsPerToken() }

// MaterializedFloats is the total f32 footprint the same tokens would cost as materialized
// per-head K/V — the memory this cache does NOT pay.
func (c *LatentMLAKVCache) MaterializedFloats() int { return c.Len() * c.MaterializedFloatsPerToken() }

// CompressionRatio is materialized / resident per-token footprint — the memory-first shrink
// the latent-resident cache buys (~57× at DeepSeek geometry, see the file header). It is a
// pure function of the geometry (independent of how many tokens are resident) and is > 1 for
// any genuinely compressed latent.
func (c *LatentMLAKVCache) CompressionRatio() float64 {
	return float64(c.MaterializedFloatsPerToken()) / float64(c.ResidentFloatsPerToken())
}
