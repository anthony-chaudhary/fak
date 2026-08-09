// Package kvbudget is a pure, GPU-free calculator for the KV-cache VRAM budget
// of concurrent GLM-5.2 (glm_moe_dsa) decode streams — the laptop-composable
// half of issue #3080 ("perf(serve): KV paging + context-budget tuning for
// concurrent GLM-5.2 streams within 206 GiB free VRAM").
//
// The paged-KV-vs-contiguous A/B and the served aggregate tok/s are GPU-gated
// and stay owed by a GPU-node worker (see the triage doc §4). The KV-bytes/token
// sizing math, however, is a closed-form calculation with no hardware in the
// loop, so this package delivers it: given a cache geometry, a context length,
// a KV quant, and a free-VRAM budget, it returns the KV GiB per stream and the
// max concurrent streams that fit.
//
// # Source of truth
//
// Every number here reproduces the landed triage doc
//
//	docs/notes/GLM52-GPU-SERVER-LANEB-KV-BUDGET-TRIAGE-2026-07-06.md
//
// which computes the {ctx, KV GiB/stream, max streams that fit} columns from
// GLM-5.2's MLA + DeepSeek-Sparse-Attention (DSA) cache shape. Per its §3.1,
// the resident KV state per token per layer is NOT a full per-head K/V — MLA
// caches a compressed latent (KVLoraRank) plus one decoupled rotary key
// (QKRopeHeadDim, MQA-shared), and DSA adds a separate indexer key
// (IndexHeadDim) per index layer:
//
//	KV_elems/token = layers      × (KVLoraRank + QKRopeHeadDim)  // MLA latent + rope key
//	               + idx_layers  × IndexHeadDim                  // DSA indexer key (≤ layers)
//	KV_bytes/token = KV_elems/token × bytes_per_elem             // F16 KV ⇒ 2
//
// # Provenance of the numbers
//
// The Shape field names/meaning mirror the real config the loader reads —
// model.Config.{NumLayers, KVLoraRank, QKRopeHeadDim, IndexHeadDim} in
// internal/model/config.go. This package intentionally does NOT import that
// package (it stays pure and standalone so it builds on a laptop with no model
// deps); instead GLM52DSA below carries the DeepSeek-lineage ESTIMATE values
// the triage doc §3.2 pins as the source of truth until Lane F pins the real
// GGUF header. Those values are ESTIMATES, not measurements — if the resident
// GGUF caches a decompressed per-head KV, a larger KVLoraRank, a different layer
// count, or a KV precision other than F16, the cells move proportionally (the
// method stands, the numbers move). Swap Shape/Quant to re-derive.
package kvbudget

import (
	"fmt"
	"math"
	"strings"
)

// GiB is 1024^3 — the binary gibibyte the VRAM budget and cache sizes are
// measured in (matching the triage doc's `÷ 1024³`).
const GiB = 1 << 30

// AttnKind selects which attention architecture's KV-cache element formula a
// Shape sizes — the general branch ktransformers' kv_cache_calculator makes over
// the loaded model header (kv_cache_calculator.py:86-121@0c2912a). The zero
// value is MLA (GLM52DSA and the triage-doc reproduction leave Kind unset), so
// an existing MLA-only Shape keeps its meaning bit-for-bit.
type AttnKind int

const (
	// MLA (DeepSeek-V2/V3, GLM-5.2) caches a compressed KV latent (KVLoraRank)
	// plus one decoupled rope key (QKRopeHeadDim) per layer, and — when the
	// header declares a DeepSeek-NSA / GLM-DSA lightning indexer — an extra
	// IndexHeadDim key over IndexLayers.
	MLA AttnKind = iota
	// MHA (Llama, Qwen, and every standard multi-head / grouped-query model)
	// caches a full per-head K and V: NumKVHeads × (HeadDim + VHeadDim) per
	// layer.
	MHA
)

// Shape is a KV-cache geometry: the per-token, per-layer element counts an
// attention architecture caches, independent of KV precision (precision is a
// Quant, below). Kind selects the element formula — MLA (+ optional NSA/DSA
// indexer) or MHA — so one Shape sizes an arbitrary served model, not just
// GLM-5.2. The fields mirror model.Config in internal/model/config.go, and
// model.Config.KVCacheShape populates a Shape straight from a loaded header
// instead of the GLM52DSA estimate (#5242).
type Shape struct {
	// Kind selects the per-token element formula (MLA vs MHA). Its zero value is
	// MLA, so a Shape that sets only the MLA fields (e.g. GLM52DSA) is unchanged.
	Kind AttnKind
	// Layers is the number of decoder layers that cache an MLA latent + rope
	// key per token (model.Config.NumLayers).
	Layers int
	// KVLoraRank is the compressed MLA KV latent width, post kv_a_layernorm
	// (model.Config.KVLoraRank). This — not a materialized per-head K/V — is
	// what makes hundreds of streams fit.
	KVLoraRank int
	// QKRopeHeadDim is the decoupled rotary key width, MQA-shared (single) per
	// layer (model.Config.QKRopeHeadDim).
	QKRopeHeadDim int
	// IndexLayers is how many layers carry their own DSA indexer key. It is an
	// upper bound at Layers; GLM-DSA shares one indexer across a group of
	// layers (glmDsaIndexerIsShared), so the true count is ≤ Layers.
	IndexLayers int
	// IndexHeadDim is the DSA indexer key width per token per index layer
	// (model.Config.IndexHeadDim).
	IndexHeadDim int
	// NumKVHeads, HeadDim, and VHeadDim are the MHA (Kind==MHA) geometry: the
	// number of key/value heads and their key and value widths
	// (model.Config.{NumKVHeads, HeadDim, VHeadDim}). A square-head model leaves
	// VHeadDim==HeadDim. All zero for an MLA Shape.
	NumKVHeads int
	HeadDim    int
	VHeadDim   int
	// PerLayer is the OPTIONAL per-layer refinement of the uniform geometry above
	// — a sliding-window cap and, for MHA, differing head widths on the layers
	// that have them (#5498). Nil (the zero value, and what every Shape built
	// today carries) means uniform: every layer attends over the whole context at
	// the scalar geometry, and every figure below is bit-for-bit what it was
	// before this field existed. It is a POINTER so Shape stays comparable.
	PerLayer *LayerProfile
}

// GLM52DSA is the DeepSeek-lineage ESTIMATE the triage doc §3.2 pins as the
// source of truth for GLM-5.2 (glm_moe_dsa) until Lane F pins the GGUF header.
// IndexLayers is the §3.3 upper bound (every layer indexes); the real count is
// smaller because the indexer is shared.
var GLM52DSA = Shape{
	Layers:        92,  // ESTIMATE — ceiling doc "~92-layer"; GGUF pins it
	KVLoraRank:    512, // ESTIMATE — DeepSeek-MLA lineage default
	QKRopeHeadDim: 64,  // ESTIMATE — DeepSeek-MLA lineage default
	IndexLayers:   92,  // upper bound (≤ Layers); the indexer is shared
	IndexHeadDim:  128, // ESTIMATE — DeepSeek-V3.2-DSA lineage default
}

// Quant is a KV-cache element precision: an effective bytes-per-element, so a
// non-integer quant (q4 ≈ 0.5 B/elem) is representable. It is the §3.4 lever
// (`--cache-type-k/v`) that trades KV footprint against accuracy in the A/B.
type Quant struct {
	Name         string
	BytesPerElem float64
}

// The KV quants the triage doc §3.4 names as levers. F16 is the llama.cpp
// default and the precision every cell in the doc's table is computed at.
var (
	F16  = Quant{Name: "f16", BytesPerElem: 2}  // default; the doc's table precision
	Q8_0 = Quant{Name: "q8_0", BytesPerElem: 1} // ~2× the fit at a quality cost
	Q4   = Quant{Name: "q4", BytesPerElem: 0.5} // ~4× the fit at a larger quality cost
)

// MLAElemsPerToken is the MLA latent + decoupled rope key cached per token,
// summed over layers: Layers × (KVLoraRank + QKRopeHeadDim). This is the
// dominant, best-characterized KV term.
func (s Shape) MLAElemsPerToken() int { return s.Layers * (s.KVLoraRank + s.QKRopeHeadDim) }

// IndexElemsPerToken is the DSA indexer key cached per token, summed over the
// index layers: IndexLayers × IndexHeadDim. Doc §3.3 treats this as an upper
// bound (some layers share the indexer).
func (s Shape) IndexElemsPerToken() int { return s.IndexLayers * s.IndexHeadDim }

// MHAElemsPerToken is the full per-head K+V cached per token for standard
// multi-head / grouped-query attention, summed over layers:
// Layers × NumKVHeads × (HeadDim + VHeadDim) — ktransformers' MHA branch
// (kv_cache_calculator.py:121@0c2912a). Zero for an MLA Shape.
func (s Shape) MHAElemsPerToken() int {
	return s.Layers * s.NumKVHeads * (s.HeadDim + s.VHeadDim)
}

// KVElemsPerToken is the total KV elements cached per token, branched on the
// Shape's attention arch: for MHA the per-head K+V sum; for MLA the compressed
// latent + rope key + DSA indexer key. This is the general branch over
// attention arch — the same math sizes an MLA or an MHA header (#5242). An
// MLA Shape (Kind unset) reduces to the prior MLA+DSA formula exactly.
func (s Shape) KVElemsPerToken() int {
	if s.Kind == MHA {
		return s.MHAElemsPerToken()
	}
	return s.MLAElemsPerToken() + s.IndexElemsPerToken()
}

// MLABytesPerToken is the MLA-only KV footprint per token at the given quant
// (the "MLA only" column of the doc's table).
func (s Shape) MLABytesPerToken(q Quant) float64 {
	return float64(s.MLAElemsPerToken()) * q.BytesPerElem
}

// KVBytesPerToken is the full MLA+DSA KV footprint per token at the given quant.
func (s Shape) KVBytesPerToken(q Quant) float64 {
	return float64(s.KVElemsPerToken()) * q.BytesPerElem
}

// KVGiBPerStream is the full KV cache GiB one stream of ctx tokens holds at the
// given quant: ctx × KV_bytes/token ÷ 1024³ (doc §3.3) for a uniform Shape, and
// the per-layer sum bounded at min(window, ctx) for one that declares a
// PerLayer profile (#5498). This is the exact value; reportGiB rounds it to the
// doc's tabulated precision.
func (s Shape) KVGiBPerStream(ctx int, q Quant) float64 {
	return s.KVBytesPerStream(ctx, q) / GiB
}

// MLAGiBPerStream is the MLA-only cache GiB per stream (the doc's "MLA only"
// column) — what would fit if DSA's separate index cache were not counted.
func (s Shape) MLAGiBPerStream(ctx int, q Quant) float64 {
	return s.MLABytesPerStream(ctx, q) / GiB
}

// MaxStreams is how many streams of a given per-stream KV footprint fit in a
// VRAM budget: floor(budget / per-stream). Pure integer fit — the headroom
// reservation is applied by the caller choosing the budget (raw 206 vs usable
// 165). Returns 0 for a non-positive per-stream footprint.
func MaxStreams(budgetGiB, perStreamGiB float64) int {
	if perStreamGiB <= 0 {
		return 0
	}
	return int(math.Floor(budgetGiB / perStreamGiB))
}

// Budgets and headroom the triage doc §3.3 sizes against.
const (
	// FreeVRAMGiB is the ~206 GiB left after the 433.82 GiB GLM-5.2 weights on
	// the 8-GPU datacenter server — the "raw" budget (all of it to KV).
	FreeVRAMGiB = 206.0
	// HeadroomFactor reserves ~20% of free VRAM for activations, CUDA-graph
	// pools, the batch working set, and paging fragmentation (doc §3.3).
	HeadroomFactor = 0.8
	// UsableVRAMGiB is FreeVRAMGiB × HeadroomFactor rounded to the doc's
	// figure: round(206 × 0.8) = round(164.8) = 165 (asserted in the test).
	UsableVRAMGiB = 165.0
)

// reportGiB rounds a per-stream GiB figure to the 3-decimal precision the
// triage doc §3.3 tabulates. Max-streams is sized against this REPORTED figure
// (not the raw float), which is why 4k F16 lands at 417 raw — one above the
// exact-bytes floor of 416. That off-by-one is the doc's "~" tilde: sizing a
// serve against a per-stream number reported to 3 decimals, not the full
// rational. The other cells are insensitive to the rounding.
func reportGiB(x float64) float64 { return math.Round(x*1000) / 1000 }

// Row is one line of the KV-budget table: a context length with its per-stream
// KV footprint (combined and MLA-only, at the doc's reported precision) and the
// max concurrent streams that fit under the raw and usable VRAM budgets.
type Row struct {
	Ctx              int
	KVGiBPerStream   float64 // combined MLA + DSA index, reported (3-dp)
	MLAGiBPerStream  float64 // MLA latent + rope key only, reported (3-dp)
	MaxStreamsRaw    int     // floor(FreeVRAMGiB   / KVGiBPerStream)
	MaxStreamsUsable int     // floor(UsableVRAMGiB / KVGiBPerStream)
}

// FitRow computes one budget Row for a context length at a given quant, sizing
// against FreeVRAMGiB (raw) and UsableVRAMGiB (usable). Per-stream figures are
// reported at the doc's 3-decimal precision, and max-streams is floor(budget /
// reported-per-stream) — the exact method the triage doc §3.3 uses.
func (s Shape) FitRow(ctx int, q Quant) Row {
	kv := reportGiB(s.KVGiBPerStream(ctx, q))
	mla := reportGiB(s.MLAGiBPerStream(ctx, q))
	return Row{
		Ctx:              ctx,
		KVGiBPerStream:   kv,
		MLAGiBPerStream:  mla,
		MaxStreamsRaw:    MaxStreams(FreeVRAMGiB, kv),
		MaxStreamsUsable: MaxStreams(UsableVRAMGiB, kv),
	}
}

// DocCtxLengths are the three context sizes the triage doc §3.3 tabulates.
var DocCtxLengths = []int{4096, 8192, 16384}

// Table computes the KV-budget rows for the given shape/quant over the given
// context lengths.
func (s Shape) Table(q Quant, ctxLengths []int) []Row {
	rows := make([]Row, 0, len(ctxLengths))
	for _, ctx := range ctxLengths {
		rows = append(rows, s.FitRow(ctx, q))
	}
	return rows
}

// DocTable reproduces the triage doc §3.3 table exactly: GLM52DSA at F16 over
// DocCtxLengths (4k/8k/16k). Its values are asserted cell-for-cell against the
// landed doc in the test.
func DocTable() []Row { return GLM52DSA.Table(F16, DocCtxLengths) }

// MarkdownTable renders rows as the GitHub-flavored Markdown table the triage
// doc §3.3 uses (columns: ctx, combined KV GiB/stream, MLA-only KV GiB/stream,
// max streams raw @FreeVRAMGiB, max streams usable @UsableVRAMGiB). Pure
// string-building — no I/O — so a caller decides whether to write it anywhere.
func MarkdownTable(rows []Row) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf(
		"| ctx | KV GiB/stream (MLA+idx) | KV GiB/stream (MLA only) | max streams @%.0f GiB raw | max streams @~%.0f GiB usable |\n",
		FreeVRAMGiB, UsableVRAMGiB))
	b.WriteString("|--:|--:|--:|--:|--:|\n")
	for _, r := range rows {
		b.WriteString(fmt.Sprintf("| %d | %.3f | %.3f | %d | %d |\n",
			r.Ctx, r.KVGiBPerStream, r.MLAGiBPerStream, r.MaxStreamsRaw, r.MaxStreamsUsable))
	}
	return b.String()
}
