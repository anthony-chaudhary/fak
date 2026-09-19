package ggufload

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// deepseek41_attention_preflight.go — the converter-aware, METADATA-ONLY attention
// descriptor preflight for the DeepSeek-V4.1-Flash ("deepseek41") family.
//
// WHY: every native V4.1 bring-up rung so far discovered a projection/shape problem
// only when later EXECUTION demanded the tensor (the #13245/#13254/#13262/#13264/
// #13266/#13276/#13278 chain). The header already carries the full tensor directory,
// so the same class of "the file does not carry the attention operand the forward
// reads, or carries it at the wrong geometry" can be aggregated OFF THE HEADER —
// before any payload byte is read and before a multi-minute load or a physical
// strix3 serve is scheduled. This is the deepseek41 child of the umbrella tensor
// preflight (#10519); it covers the attention descriptors only.
//
// CONTRACT: this is a pure classifier over the parsed File. It reads the tensor
// directory (names + dims) and the already-derived model.Config, never a payload
// byte, and never mutates anything. It returns a bounded, stable, layer/name-sorted
// list of diagnostics; an empty list means every checked descriptor is compatible.

// ds41AttentionDiag is one aggregated attention-descriptor diagnostic: the canonical
// tensor name the failing layer must carry, and a one-line reason (missing, or the
// expected/actual geometry).
type ds41AttentionDiag struct {
	Canonical string
	Reason    string
}

// ds41CanonicalLayerName composes the canonical HF per-layer tensor name the native
// V4.1 forward reads for layer l. It mirrors internal/model's layerName (which is
// unexported and must not be imported) so the descriptor check names tensors with
// the SAME spelling the forward's admission does.
func ds41CanonicalLayerName(l int, suffix string) string {
	return fmt.Sprintf("model.layers.%d.%s", l, suffix)
}

// deepseek41AttentionDescriptorDiagnostics aggregates per-layer V4.1 attention
// descriptor failures across every configured layer, or nil when arch is not
// deepseek41 or every descriptor is compatible.
//
// The check is deliberately self-consistent with the native forward's admission
// contract (internal/model/v41_forward.go v41ForwardAdmitted) and uses the SAME
// canonical names and derived config axes, so a descriptor that this preflight
// accepts is one admission can also resolve:
//
//	attn.wq_a.weight      [QLoraRank, H]        (q_lora projection)
//	attn.wq_b.weight      [nH*hd, QLoraRank]    (per-head query up-projection)
//	attn.wq_a_norm.weight [QLoraRank]           (full path only)
//	attn.wkv.weight       [KVLoraRank, H]       (compressed-KV projection)
//	attn.kv_norm.weight   [KVLoraRank]          (full path only)
//	attn.wo_a.weight      [OLoraRank, nH*hd] OR [OGroups*OLoraRank, (nH/OGroups)*hd]
//	attn.wo_b.weight      [H, OGroups*OLoraRank]
//	attn.sink             [nH]
//
// "full path only" mirrors the forward: the reduced fixture has no latent-norm
// stage, so its q/kv norms are neither required nor read. When the file DECLARES
// the norms anywhere (a real full artifact does) they are required on every layer,
// so a half-converted checkpoint is refused rather than admitted at a wrong width.
func deepseek41AttentionDescriptorDiagnostics(s *WeightSource, cfg model.Config) []ds41AttentionDiag {
	if s == nil || s.File == nil || !archIsDeepSeek41(cfg.ModelType) {
		return nil
	}
	layers := cfg.NumLayers
	if layers <= 0 {
		return nil
	}
	H, nH, hd := cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim
	if H <= 0 || nH <= 0 || hd <= 0 {
		// The geometry guards that make the descriptor expectations meaningful are
		// already enforced by File.Config/applyDeepSeek41Config; without them the
		// byte estimate is the correct next rung, not a descriptor refusal.
		return nil
	}

	// Canonical tensor name -> parsed GGUF dims (outer-first, matching the model's
	// shape convention). Built once from the directory; no payload is touched.
	shapeByCanonical := make(map[string][]int, len(s.File.Tensors))
	for _, ti := range s.File.Tensors {
		canon, ok := CanonicalTensorNameArch(ti.Name, cfg.ModelType)
		if !ok {
			continue
		}
		if _, dup := shapeByCanonical[canon]; dup {
			continue
		}
		if shape, err := modelShapeFromGGUFDims(ti.Name, ti.Dims); err == nil {
			shapeByCanonical[canon] = shape
		}
	}
	hasCanonical := func(name string) bool {
		_, ok := shapeByCanonical[name]
		return ok
	}

	// The latent norms are the full forward's optional stage: require them exactly
	// when the checkpoint declares them (so a reduced fixture stays admitted),
	// mirroring v41ForwardAdmitted's `full` switch decided by the same presence.
	requireLatentNorms := false
	for l := 0; l < layers && !requireLatentNorms; l++ {
		if hasCanonical(ds41CanonicalLayerName(l, "attn.wq_a_norm.weight")) || hasCanonical(ds41CanonicalLayerName(l, "attn.kv_norm.weight")) {
			requireLatentNorms = true
		}
	}

	var diags []ds41AttentionDiag
	add := func(canonical string, reason string) {
		diags = append(diags, ds41AttentionDiag{Canonical: canonical, Reason: reason})
	}
	check := func(canonical string, want ...int) {
		got, ok := shapeByCanonical[canonical]
		if !ok {
			add(canonical, "missing tensor")
			return
		}
		if !shapeEqualInts(got, want) {
			add(canonical, fmt.Sprintf("shape %v, want %v", got, want))
		}
	}

	kvLatentRank := cfg.KVLoraRank
	if kvLatentRank <= 0 {
		// The real artifact derives the rank from blk.<L>.attn_kv_a_norm.weight; a
		// header that carries neither the key nor the norm cannot be checked against
		// a rank, so the tombstones below name the missing norm and the probe stops
		// fabricating a width. ds41KVLoraRankFromTensors is the same derivation the
		// config path uses.
		kvLatentRank = ds41KVLoraRankFromTensors(s.File, layers)
	}

	oDim := 0
	if cfg.OGroups > 0 && cfg.OLoraRank > 0 {
		oDim = cfg.OGroups * cfg.OLoraRank
	}

	for l := 0; l < layers; l++ {
		check(ds41CanonicalLayerName(l, "attn.wq_a.weight"), cfg.QLoraRank, H)
		check(ds41CanonicalLayerName(l, "attn.wq_b.weight"), nH*hd, cfg.QLoraRank)
		if requireLatentNorms {
			check(ds41CanonicalLayerName(l, "attn.wq_a_norm.weight"), cfg.QLoraRank)
			check(ds41CanonicalLayerName(l, "attn.kv_norm.weight"), kvLatentRank)
		}
		check(ds41CanonicalLayerName(l, "attn.wkv.weight"), kvLatentRank, H)
		checkDeepSeek41GroupedWoA(shapeByCanonical, cfg, l, add)
		if oDim > 0 {
			check(ds41CanonicalLayerName(l, "attn.wo_b.weight"), H, oDim)
		} else {
			// Without the grouped output axes there is no derivable wo_b width; the
			// presence check still catches an outright missing operand.
			if !hasCanonical(ds41CanonicalLayerName(l, "attn.wo_b.weight")) {
				add(ds41CanonicalLayerName(l, "attn.wo_b.weight"), "missing tensor")
			}
		}
		check(ds41CanonicalLayerName(l, "attn.sink"), nH)
	}

	if len(diags) == 0 {
		return nil
	}
	// Stable, layer/name-sorted report: the canonical name embeds the layer index,
	// and the same key is used for the tie-break so the order is total and
	// input-order-independent.
	sort.SliceStable(diags, func(i, j int) bool {
		if diags[i].Canonical != diags[j].Canonical {
			return diags[i].Canonical < diags[j].Canonical
		}
		return diags[i].Reason < diags[j].Reason
	})
	return diags
}

// checkDeepSeek41GroupedWoA admits the grouped output projection in either
// declaration of the same weight, matching v41AdmitGroupedWoA: the artifact's
// group-major [OGroups*OLoraRank, (nH/OGroups)*hd] tensor (the value the real GGUF
// carries) or the reduced fixture's equivalent flat [OLoraRank, nH*hd] form. Any
// other shape — including a group-major total that disagrees with the flat total —
// is named, so the no-silent-mis-shape property is retained.
func checkDeepSeek41GroupedWoA(shapeByCanonical map[string][]int, cfg model.Config, l int, add func(string, string)) {
	name := ds41CanonicalLayerName(l, "attn.wo_a.weight")
	got, ok := shapeByCanonical[name]
	if !ok {
		add(name, "missing tensor")
		return
	}
	flat := []int{cfg.OLoraRank, cfg.NumHeads * cfg.HeadDim}
	groupedHeads := 0
	if cfg.OGroups > 0 && cfg.NumHeads%cfg.OGroups == 0 {
		groupedHeads = cfg.NumHeads / cfg.OGroups
	}
	switch {
	case shapeEqualInts(got, flat):
		return
	case cfg.OGroups > 0 && cfg.OLoraRank > 0 && groupedHeads > 0 &&
		shapeEqualInts(got, []int{cfg.OGroups * cfg.OLoraRank, groupedHeads * cfg.HeadDim}):
		return
	default:
		add(name, fmt.Sprintf("shape %v, want %v (flat) or [%d %d] (grouped)",
			got, flat, cfg.OGroups*cfg.OLoraRank, groupedHeads*cfg.HeadDim))
	}
}

// shapeEqualInts reports whether two shape vectors are element-wise equal.
func shapeEqualInts(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// renderDeepSeek41AttentionDiagnostics folds an aggregated diagnostic list into the
// single-line ModelPreflight.Reason, bounded to a fixed count so a systematically
// broken checkpoint cannot produce an unbounded refusal string.
const deepseek41AttentionReasonMax = 12

func renderDeepSeek41AttentionDiagnostics(diags []ds41AttentionDiag) string {
	var b strings.Builder
	b.WriteString("deepseek41 attention descriptors: ")
	for i, d := range diags {
		if i == deepseek41AttentionReasonMax {
			fmt.Fprintf(&b, "; (+%d more)", len(diags)-i)
			break
		}
		if i > 0 {
			b.WriteString("; ")
		}
		fmt.Fprintf(&b, "%s %s", d.Canonical, d.Reason)
	}
	return b.String()
}
