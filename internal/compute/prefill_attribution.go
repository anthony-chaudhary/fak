package compute

import (
	"fmt"
	"math"
	"strings"
)

// prefill_attribution.go — the SW-side stage attribution for long-prompt prefill on the
// inkernel CUDA route (ticket prefill-longprompt-scaling-p2048-20261108, Gap 3 of the
// matched-BF16 gaps). The matched-BF16 h2h measured P128/P512/P2048 = 508/850/632 tok/s
// cold on one H200: P2048 is SLOWER than P512 while every reference engine rises. This
// file answers the ticket's profiling step WITHOUT a GPU, the way prefill.go's
// PrefillCostModel does for dense models: it counts the EXACT arithmetic and operand
// bytes of every stage of the qwen35-hybrid device sequence prefill at a given panel
// height P, reuses StageCost/RooflineSeconds for the time LOWER BOUND at caller-supplied
// device peaks, and exposes the P-scaling exponent per stage so the superlinear term is
// identified structurally — which stage's work grows faster than P.
//
// Path under account (all panelized: one launch spans the whole P-token panel, no Go
// per-position loop anywhere on this path):
//
//	entry      internal/model/qwen35_hal.go tryQwen35SequencePrefill
//	device loop internal/compute/cuda_qwen35_sequence.go Qwen35SequencePrefill (per-layer loop)
//	GEMMs      internal/compute/cuda_kernels.go qwen35SequenceMatMulLocked (cuBLAS panel GEMM)
//	KV append  internal/compute/cuda_kernels.go qwen35SequenceAppendKVLocked (batched over P)
//	attention  internal/compute/cuda/attention.cuh k_qwen38_causal_attention_panel_hd256
//	GDN scan   internal/compute/cuda/qwen35_gdn.cuh fcuda_qwen35_gdn_sequence_f32
//
// The ONLY stage whose work is superlinear in P is the causal attention over the growing
// context on the full-attention layers: its FLOPs and its logical K/V re-read bytes both
// grow as P(P+1)/2. Every other stage is O(P) with a P-independent dispatch count. That
// is the attribution; the test pins it as a closed contract.

// Qwen38HybridGeometry is the Qwen3.8-27B shape the inkernel CUDA prefill reasons over:
// the qwen35-family hybrid of FULL ATTENTION layers (every FullAttentionInterval-th
// block) and GATED DELTANET linear-attention layers, plus the shared SwiGLU FFN. The
// constants are the authoritative tensor shapes of the pinned BF16 checkpoint
// (docs/benchmarks/oss-baselines/rtp-test-h200/raw/20261107T130517Z-matched-bf16/
// convert.log) cross-checked against the C ABI validator in
// cuda/attention.cuh fak_qwen35_causal_attention_panel_f32 (nH==24, nKV==4, hd==256) and
// model.Config.linearAttnDims (convDim = 2*nK*kHd + nV*vHd = 10240).
type Qwen38HybridGeometry struct {
	Hidden                int // residual width D = 5120
	Vocab                 int // token vocabulary = 248320
	NLayers               int // transformer blocks = 64 (the blk.64 nextn MTP block is speculative-decoding side-channel, off in the h2h envelope)
	FullAttentionInterval int // layer l is full attention when (l+1)%interval == 0; 4 -> 16 full-attn + 48 GDN layers
	NHeads                int // full-attn query heads = 24
	NKVHeads              int // full-attn KV heads = 4 (GQA ratio 6)
	HeadDim               int // full-attn head dim = 256
	GDNKeyHeads           int // GDN key heads nK = 16
	GDNValueHeads         int // GDN value heads nV = 48
	GDNKeyHeadDim         int // GDN key head dim kHd = 128
	GDNValueHeadDim       int // GDN value head dim vHd = 128
	FFNInner              int // SwiGLU inner width = 17408
	WeightElemBytes       int // weight storage bytes/element = 2 (BF16 checkpoint)
}

// Qwen38_27BGeometry is the pinned checkpoint's geometry (see the type's doc for sources).
var Qwen38_27BGeometry = Qwen38HybridGeometry{
	Hidden: 5120, Vocab: 248320, NLayers: 64, FullAttentionInterval: 4,
	NHeads: 24, NKVHeads: 4, HeadDim: 256,
	GDNKeyHeads: 16, GDNValueHeads: 48, GDNKeyHeadDim: 128, GDNValueHeadDim: 128,
	FFNInner: 17408, WeightElemBytes: 2,
}

// FullAttentionLayers is the count of full-attention blocks (NLayers/interval; the model
// family guarantees the alignment — 64/4 = 16).
func (g Qwen38HybridGeometry) FullAttentionLayers() int { return g.NLayers / g.FullAttentionInterval }

// GDNLayers is the count of Gated-DeltaNet blocks (64 - 16 = 48).
func (g Qwen38HybridGeometry) GDNLayers() int { return g.NLayers - g.FullAttentionLayers() }

// GDNStateBytes is the recurrent state one GDN layer carries: nK matrices of
// kHd x vHd f32 = 16*128*128*4 = 1 MiB per layer. It is CONSTANT in P (the scan updates
// it in place); the scan keeps it L2-resident, so HBM traffic is ~state size, not P times
// state size — the logical state bytes are charged once, not per position.
func (g Qwen38HybridGeometry) GDNStateBytes() int64 {
	return int64(g.GDNKeyHeads) * int64(g.GDNKeyHeadDim) * int64(g.GDNValueHeads) * int64(g.GDNValueHeadDim) * 4
}

// causalKeys is the number of (query,key) pairs a causal panel of P rows attends: the
// i-th query sees i+1 keys, summed over i = P(P+1)/2. This is THE quantity that makes
// attention superlinear; every O(P) stage below never sees it.
func causalKeys(P int) int64 { return int64(P) * (int64(P) + 1) / 2 }

// AttributionStage is one stage of the panel prefill with its exact work counts, the
// modeled FIXED launch count (independent of P — the panelization property), its
// structural P-scaling class, and a source pointer. It embeds StageCost so the
// roofline seams (RooflineSeconds, Bound, totalBytes) apply unchanged.
type AttributionStage struct {
	StageCost
	Launches int64  // modeled launch sites at this P; equal at every P
	Scaling  string // "O(P)" or "O(P^2)" — the structural classification
	Note     string // kernel / call-site pointer
}

// gemmStage builds the analytic cost of one Y[P,out] = X[P,in] @ Wᵀ panel GEMM in the
// hybrid geometry's weight dtype: 2·P·out·in flops, the resident weight stream, and the
// f32 activation read (X) + write (Y). Same shape as PrefillGeometry.gemmCost, kept
// local because the hybrid widths (qg, convDim, ...) live on this geometry.
func gemmStage(name string, out, in, P, weightElemBytes int) StageCost {
	flops := int64(2) * int64(P) * int64(out) * int64(in)
	wb := int64(out) * int64(in) * int64(weightElemBytes)
	ab := int64(4) * int64(P) * (int64(in) + int64(out))
	return StageCost{Name: name, FLOPs: flops, WeightBytes: wb, ActBytes: ab, Intensity: ratio(flops, wb+ab)}
}

// LongPromptStageTable returns the per-stage analytic cost of a COLD (prefix = 0)
// P-token panel prefill on the inkernel CUDA route, aggregated across the layers each
// stage appears in, in forward order. Counts are exact (a count of the arithmetic and
// the operands the stage must touch); no timer runs and no throughput is fabricated —
// the same honesty discipline as PrefillCostModel. The caller supplies device peaks to
// turn the counts into roofline seconds via StageCost.RooflineSeconds.
func LongPromptStageTable(g Qwen38HybridGeometry, P int) []AttributionStage {
	if P <= 0 {
		return nil
	}
	D := int64(g.Hidden)
	nFA, nGDN := int64(g.FullAttentionLayers()), int64(g.GDNLayers())
	ck := causalKeys(P)
	qOut := int64(g.NHeads) * int64(g.HeadDim)                                            // 6144: q half of the fused qg projection
	kvOut := int64(g.NKVHeads) * int64(g.HeadDim)                                         // 1024 per K or V
	convDim := int64(2*g.GDNKeyHeads*g.GDNKeyHeadDim + g.GDNValueHeads*g.GDNValueHeadDim) // 10240
	valDim := int64(g.GDNValueHeads) * int64(g.GDNValueHeadDim)                           // 6144: z gate + GDN out-proj width

	out := make([]AttributionStage, 0, 13)
	add := func(sc StageCost, launches int64, scaling, note string) {
		out = append(out, AttributionStage{StageCost: sc, Launches: launches, Scaling: scaling, Note: note})
	}
	// scale multiplies a PER-LAYER stage's counts by the layer count it appears in
	// (the same fold PrefillCostModel does via scaleStage): the builders below produce
	// one layer's work, and the panel runs it on every matching block.
	scale := func(sc StageCost, n int64) StageCost {
		sc.FLOPs *= n
		sc.WeightBytes *= n
		sc.ActBytes *= n
		return sc
	}

	// Embedding gather: P rows out of the [vocab x D] table (BF16 read, f32 panel write).
	add(StageCost{Name: "embedding_gather", FLOPs: 0,
		WeightBytes: 0, ActBytes: int64(P) * D * (int64(g.WeightElemBytes) + 4),
		Intensity: 0}, 1, "O(P)", "cuda_qwen35_sequence.go qwen35SequenceEmbeddingLocked")

	// ---- full-attention layers (x nFA) -------------------------------------
	// Fused q+gate projection [P,D] -> [P, 2*qOut]; SplitQG separates the halves.
	add(scale(gemmStage("attn_qg_proj", 2*g.NHeads*g.HeadDim, g.Hidden, P, g.WeightElemBytes), nFA),
		nFA, "O(P)", "attn_q.weight {5120,12288}; qwen35SequenceMatMulLocked + qwen35SequenceSplitQGLocked")
	// Separate K and V projections [P,D] -> [P,kvOut] each.
	add(scale(gemmStage("attn_kv_proj", 2*g.NKVHeads*g.HeadDim, g.Hidden, P, g.WeightElemBytes), nFA),
		2*nFA, "O(P)", "attn_k/attn_v.weight {5120,1024}")
	// THE superlinear stage. One SIMT panel launch per layer (split-KV takes nsplits=1
	// on a cold panel: the gate keys on CACHED prefix >= 1024, which is 0 here). Each
	// (row,head) CTA serially scans its causal key range with a barrier reduction per
	// key and re-reads its KV-head slice of every key it attends, so BOTH the FLOPs and
	// the logical K/V bytes grow as P(P+1)/2. Intrinsic FLOPs stay small (see the test:
	// <2% of total at P2048) — the term is byte amplification + serialization, not raw
	// FLOPs. HBM traffic after L2 dedup is below the logical count but still
	// superlinear; the HW timestamp fold measures the realized share.
	attnFLOPs := int64(4) * int64(g.NHeads) * int64(g.HeadDim) * ck // QK^T + PV, 2 flops/MAC
	attnBytes := int64(2)*int64(g.NHeads)*ck*int64(g.HeadDim)*4 +   // K + V re-stream, logical
		int64(2)*int64(P)*int64(g.NHeads)*int64(g.HeadDim)*4 // Q read + Out write
	add(scale(StageCost{Name: "causal_attention_panel", FLOPs: attnFLOPs, WeightBytes: 0,
		ActBytes: attnBytes, Intensity: ratio(attnFLOPs, attnBytes)}, nFA),
		nFA, "O(P^2)", "attention.cuh k_qwen38_causal_attention_panel_hd256 (SIMT arm; MMA arm default-OFF pending P-05 receipt)")
	// Elementwise sigmoid gate on the attention output: attn ⊙ sigmoid(gate), P*qOut wide.
	gateElems := int64(P) * qOut
	add(scale(StageCost{Name: "attn_gate_sigmoid_mul", FLOPs: 2 * gateElems, WeightBytes: 0,
		ActBytes: 3 * gateElems * 4, Intensity: ratio(2*gateElems, 3*gateElems*4)}, nFA),
		nFA, "O(P)", "fcuda_sigmoid_mul_f32")
	// Output projection [P,qOut] -> [P,D] (weight {6144,5120}).
	add(scale(gemmStage("attn_o_proj", g.Hidden, g.NHeads*g.HeadDim, P, g.WeightElemBytes), nFA),
		nFA, "O(P)", "attn_output.weight {6144,5120}")
	// Batched K/V append into the paged store: P rows of K+V per layer, one launch.
	kvAppendBytes := int64(P) * 2 * kvOut * 4
	add(scale(StageCost{Name: "kv_append", FLOPs: 0, WeightBytes: 0, ActBytes: kvAppendBytes, Intensity: 0}, nFA),
		nFA, "O(P)", "qwen35SequenceAppendKVLocked (batched over the panel)")

	// ---- GDN layers (x nGDN) -------------------------------------------------
	// Fused in-projections: qkv [P,D]->[P,convDim], z [P,D]->[P,valDim], b/a [P,D]->[P,nV] each.
	gdnInOut := convDim + valDim + 2*int64(g.GDNValueHeads)
	add(scale(gemmStage("gdn_in_proj", int(gdnInOut), g.Hidden, P, g.WeightElemBytes), nGDN),
		4*nGDN, "O(P)", "linear_attn.in_proj_qkv/z/b/a; qwen35SequenceMatMulLocked")
	// The fused device scan (ONE launch per layer): depthwise conv1d, qk norms, the
	// recurrent delta-rule scan over the panel, gated RMSNorm, and the out projection.
	// Recurrence cost per head-token ~ 6*kHd*vHd flops (decay-gated state, S^T k matvec,
	// outer-product update) -> O(P) with a large-but-P-independent constant. State bytes
	// are charged once (L2-resident during the scan; see GDNStateBytes).
	recFLOPs := int64(P) * int64(g.GDNKeyHeads) * 6 * int64(g.GDNKeyHeadDim) * int64(g.GDNValueHeadDim)
	outProj := int64(2) * int64(P) * D * valDim
	convFLOPs := int64(P) * convDim * 8 // depthwise convKernel=4: ~2 flops/tap x 4 taps
	normFLOPs := int64(P) * valDim * 4
	gdnScanFLOPs := recFLOPs + outProj + convFLOPs + normFLOPs
	scanBytesPerLayer := int64(P) * (convDim + valDim + valDim + D) * 4
	gdnScan := scale(StageCost{Name: "gdn_fused_scan", FLOPs: gdnScanFLOPs,
		WeightBytes: valDim * D * int64(g.WeightElemBytes), ActBytes: scanBytesPerLayer}, nGDN)
	// Recurrent state charged once per layer (constant in P; see GDNStateBytes), added
	// AFTER the per-layer scaling so it is not multiplied by the layer count twice.
	gdnScan.ActBytes += nGDN * g.GDNStateBytes()
	gdnScan.Intensity = ratio(gdnScan.FLOPs, gdnScan.totalBytes())
	add(gdnScan, nGDN, "O(P)", "qwen35_gdn.cuh fcuda_qwen35_gdn_sequence_f32 (chunked scan, O(P))")

	// ---- shared per-layer services (x NLayers) ------------------------------
	// SwiGLU FFN: gate+up [P,D]->[P,FFNInner] each, down [P,FFNInner]->[P,D].
	nAll := int64(g.NLayers)
	add(scale(gemmStage("ffn_gate_up", 2*g.FFNInner, g.Hidden, P, g.WeightElemBytes), nAll),
		2*nAll, "O(P)", "ffn_gate/ffn_up.weight {5120,17408}")
	add(scale(gemmStage("ffn_down", g.Hidden, g.FFNInner, P, g.WeightElemBytes), nAll),
		nAll, "O(P)", "ffn_down.weight {17408,5120}")
	// Elementwise layer services: RMSNorms (input/post/q/k), RoPE, residual adds,
	// swiglu, splitQG — all O(P) elementwise passes; traffic modeled as read+write of
	// the panels they touch. The modeled launch count is the enumerated call-site set
	// in the per-layer loop (cuda_qwen35_sequence.go), independent of P.
	servicesElems := int64(P) * (int64(g.NLayers)*3*D + nFA*(3*qOut+3*kvOut) + int64(g.NLayers)*(3*int64(g.FFNInner)))
	add(StageCost{Name: "elementwise_layer_services", FLOPs: 2 * servicesElems, WeightBytes: 0,
		ActBytes: 3 * servicesElems * 4, Intensity: ratio(2*servicesElems, 3*servicesElems*4)},
		16*9+48*4, "O(P)", "RMSNorm/RoPE/residual/swiglu/splitQG call sites, cuda_qwen35_sequence.go per-layer loop")

	// LM head over the LAST row only (prefill emits one next-token distribution; the
	// h2h envelope does not request all logits).
	lmFLOPs := int64(2) * int64(g.Vocab) * D
	lmWB := int64(g.Vocab) * D * int64(g.WeightElemBytes)
	lmAB := int64(4) * (D + int64(g.Vocab))
	add(StageCost{Name: "lm_head_last_row", FLOPs: lmFLOPs, WeightBytes: lmWB, ActBytes: lmAB,
		Intensity: ratio(lmFLOPs, lmWB+lmAB)}, 1, "O(1)", "output.weight {5120,248320}, last row only")

	return out
}

// ScalingExponent returns the P-scaling exponent of a stage between two panel heights:
// log(work(p2)/work(p1)) / log(p2/p1) on the stage's work proxy (FLOPs, falling back to
// bytes when FLOPs is zero). A linear stage reads ~1.0, the causal-attention stage ~2.0;
// anything between is where the measurement must arbitrate. Guarded against the
// degenerate inputs (p<=1, zero work) with 0.
func ScalingExponent(s1, s2 AttributionStage, p1, p2 int) float64 {
	if p1 <= 1 || p2 <= p1 {
		return 0
	}
	w1, w2 := s1.workProxy(), s2.workProxy()
	if w1 <= 0 || w2 <= 0 {
		return 0
	}
	return math.Log(float64(w2)/float64(w1)) / math.Log(float64(p2)/float64(p1))
}

// workProxy is the quantity a stage's wall time tracks: FLOPs when the stage computes,
// bytes otherwise (pure memory moves like embedding gather and KV append).
func (s AttributionStage) workProxy() int64 {
	if s.FLOPs > 0 {
		return s.FLOPs
	}
	return s.totalBytes()
}

// LongPromptPanelReport renders the P-panel stage attribution as a bounded text table:
// one row per stage with its scaling class, the fixed launch count, the roofline LOWER
// bound seconds (and share of the total) at each panel height, and the scaling exponent
// between the smallest and largest panel. It is the [SW-VERIFIED] attribution artifact
// the ticket's witness names: exact counts + caller-supplied device peaks, no timer.
func LongPromptPanelReport(g Qwen38HybridGeometry, promptLens []int, peakFLOPs, peakBytesPerSec float64) string {
	if len(promptLens) < 2 {
		return "panel needs >= 2 prompt lengths\n"
	}
	tables := make([][]AttributionStage, len(promptLens))
	totals := make([]float64, len(promptLens))
	for i, p := range promptLens {
		tables[i] = LongPromptStageTable(g, p)
		for _, s := range tables[i] {
			totals[i] += s.RooflineSeconds(peakFLOPs, peakBytesPerSec)
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Qwen3.8-27B hybrid prefill stage attribution (cold panel, inkernel CUDA route)\n")
	fmt.Fprintf(&b, "peaks: %.0f TFLOP/s compute, %.2f TB/s memory (caller-supplied; roofline LOWER bounds)\n",
		peakFLOPs/1e12, peakBytesPerSec/1e12)
	header := fmt.Sprintf("%-28s %-7s %-9s", "stage", "scaling", "launches")
	for _, p := range promptLens {
		header += fmt.Sprintf(" %16d", p) // ms/share columns, one per panel height
	}
	header += "  exponent"
	fmt.Fprintln(&b, header)
	for i, s := range tables[0] {
		row := fmt.Sprintf("%-28s %-7s %-9d", s.Name, s.Scaling, s.Launches)
		for j := range promptLens {
			sec := tables[j][i].RooflineSeconds(peakFLOPs, peakBytesPerSec)
			share := 0.0
			if totals[j] > 0 {
				share = 100 * sec / totals[j]
			}
			row += fmt.Sprintf(" %8.1fms/%4.1f%%", sec*1e3, share)
		}
		exp := ScalingExponent(tables[0][i], tables[len(promptLens)-1][i], promptLens[0], promptLens[len(promptLens)-1])
		row += fmt.Sprintf("  %.2f", exp)
		fmt.Fprintln(&b, row)
	}
	for j := range promptLens {
		var flops, bytes int64
		for _, s := range tables[j] {
			flops += s.FLOPs
			bytes += s.totalBytes()
		}
		fmt.Fprintf(&b, "%-28s %-7s TOTAL: %.1f ms roofline floor, %.2f TFLOP, %.2f GB moved\n",
			"total", "-", totals[j]*1e3, float64(flops)/1e12, float64(bytes)/1e9)
	}
	return b.String()
}
