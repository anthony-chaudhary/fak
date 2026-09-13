package model

import (
	"fmt"
	"strings"
)

// v41_quant_sensitivity.go â€” the per-tensor-class MINIMUM-PRECISION (floor) map
// for the DeepSeek V4.1 Flash tensor layout, and the fail-closed re-quant
// admission guard that composes beside v4quant_admit.go's precision-family gate
// (issue #12932, parent track #12640).
//
// v4quant_admit.go answers "is this tensor's declared precision the one its class
// was trained at?" (FP4 experts must be FP4, not FP8, and so on). It does NOT
// answer the re-quantization question: "may this already-4-bit tensor be crushed
// further to IQ2_XXS / Q2_K without destroying an architecturally sensitive
// group?" The official V4.1-Flash checkpoint is already natively MXFP4/MXFP8, so
// a re-quant that stacks 2-bit error on top of 4-bit error is the unsafe
// operating point the parent track names. This file is the metadata-only guard
// that refuses it BEFORE any bytes move.
//
// Two pieces ship together, exactly as the issue describes:
//
//  1. THE DATA â€” v41SensitivityFloors: one min-safe floor per sensitivity group,
//     keyed to V4TensorClass where a V4 class exists and to a small set of
//     v41-only markers (Engram hash tables, Engram gate rows, hyper-connection
//     combine matrices, ViT tower) that the existing classifier does not name.
//  2. THE GUARD â€” admitRequantTensor: a pure per-tensor decision over a
//     checkpoint tensor index (name + dtype + shape), returning a typed
//     fail-closed refusal naming the tensor, class, floor, and proposed format.
//
// This mirrors v4quant_admit.go's posture: no weight I/O, no GEMM, no format
// decoder (the IQ2_XXS/IQ3_XXS/Q2_K dequant implementations are NOT here), no
// GPU/ROCm/AVX work, no whole-model quality claim. Unclassifiable tensor or
// unknown format tag refuses closed; there is no default-admit.

// V41RequantFormat is the closed set of re-quantization operating points a
// proposal may name. It is deliberately separate from V4Precision (which
// describes a checkpoint's ACTUAL precision family): a re-quant proposal is a
// TARGET format a conversion would write, and the guard ranks those targets by
// effective bits against a class floor. An unknown tag is never admitted.
type V41RequantFormat string

const (
	V41FormatFP8     V41RequantFormat = "FP8"     // 8-bit (E4M3/E5M2) â€” the checkpoint's non-expert family
	V41FormatQ8_0    V41RequantFormat = "Q8_0"    // ~8.5 bpw legacy
	V41FormatQ6_K    V41RequantFormat = "Q6_K"    // ~6.6 bpw
	V41FormatQ5_K    V41RequantFormat = "Q5_K"    // ~5.5 bpw
	V41FormatQ4_K    V41RequantFormat = "Q4_K"    // ~4.5 bpw
	V41FormatFP4     V41RequantFormat = "FP4"     // 4-bit (NVFP4/E2M1) â€” the checkpoint's expert family
	V41FormatIQ3_XXS V41RequantFormat = "IQ3_XXS" // ~3.06 bpw
	V41FormatQ3_K    V41RequantFormat = "Q3_K"    // ~3.4 bpw
	V41FormatIQ2_XXS V41RequantFormat = "IQ2_XXS" // ~2.06 bpw
	V41FormatQ2_K    V41RequantFormat = "Q2_K"    // ~2.6 bpw
)

// v41RequantFormatBits is the closed set's effective-bits table. The values are
// the canonical bpw of each ggml/gguf format used to RANK a proposal against a
// floor; they are a deterministic ordering key, not a measured compression
// ratio. An unknown tag has no entry and therefore no rank (fail closed).
func v41RequantFormatBits(f V41RequantFormat) (float64, bool) {
	switch f {
	case V41FormatQ8_0:
		return 8.5, true
	case V41FormatFP8:
		return 8.0, true
	case V41FormatQ6_K:
		return 6.5625, true
	case V41FormatQ5_K:
		return 5.5, true
	case V41FormatQ4_K:
		return 4.5, true
	case V41FormatFP4:
		return 4.0, true
	case V41FormatQ3_K:
		return 3.4375, true
	case V41FormatIQ3_XXS:
		return 3.0625, true
	case V41FormatQ2_K:
		return 2.5625, true
	case V41FormatIQ2_XXS:
		return 2.0625, true
	}
	return 0, false
}

// v41SensitiveGroup is the fine-grained sensitivity grouping the floor map keys
// on. It is a SUPERSET of V4TensorClass: every V4 class maps to a group, and the
// groups the existing classifier folds together (or does not name at all) are
// split out here â€” Engram hash tables and gate rows, hyper-connection combine
// matrices, and the ViT tower.
type v41SensitiveGroup string

const (
	v41GroupEngramTable v41SensitiveGroup = "engram_table" // *.engram.*.table  (n-gram memory written to residual)
	v41GroupEngramGate  v41SensitiveGroup = "engram_gate"  // *.engram.*.gate*  (learned write gate)
	v41GroupHyperConn   v41SensitiveGroup = "hyper_conn"   // *.hc.* / *.hyper* (Sinkhorn residual combine)
	v41GroupVision      v41SensitiveGroup = "vision_tower" // model.visual.*    (ViT tower)
)

// v41Floor is one row of the floor map: the min-safe format and its rationale.
type v41Floor struct {
	Format V41RequantFormat
	Why    string
}

// v41SensitivityFloors is THE DATA: the min-safe re-quant floor for the
// fine-grained groups the V4 classifier does not name. A proposal BELOW its
// group's floor is refused; a proposal at or above the floor is admissible from
// the floor's point of view (the existing precision gate still governs whether
// the declared dtype is legal for the class).
var v41SensitivityFloors = map[v41SensitiveGroup]v41Floor{
	v41GroupEngramTable: {V41FormatFP8, "Engram n-gram memory is written to the residual via a learned gate; low-bit corruption degrades long-context recall"},
	v41GroupEngramGate:  {V41FormatFP8, "learned Engram write gate; a low-bit gate mis-gates memory writes"},
	v41GroupHyperConn:   {V41FormatFP8, "Sinkhorn-normalized 4-way residual combine; low-bit breaks normalization and logit flow"},
	v41GroupVision:      {V41FormatFP8, "ViT tower sensitivity is outside the safe low-bit set; FP8 floor if retained"},
}

// v41ClassFloors maps every V4TensorClass to its min-safe re-quant floor.
func v41ClassFloors(class V4TensorClass) (v41Floor, bool) {
	switch class {
	case V4ClassRoutedExpert:
		// up/gate tolerate ~3-bit; down-proj is separated out by the
		// suffix-sensitive form in v41FloorFor. Default (up/gate) floor: IQ3_XXS.
		return v41Floor{V41FormatIQ3_XXS, "routed-expert up/gate tolerate ~3-bit; never IQ2_XXS"}, true
	case V4ClassIndexerQK:
		return v41Floor{V41FormatFP4, "indexer QK path: best-512 selection corrupts under low-bit, misrouting attention"}, true
	case V4ClassSharedExpert:
		return v41Floor{V41FormatFP8, "always-active shared expert; low-bit degrades every token"}, true
	case V4ClassAttention:
		return v41Floor{V41FormatFP8, "attention projections incl. MLA latents; hybrid-architecture KLD is sensitive"}, true
	case V4ClassRouter:
		return v41Floor{V41FormatFP8, "router/gate weight: routing flips cost more than weight error"}, true
	case V4ClassDenseFFN:
		return v41Floor{V41FormatIQ3_XXS, "dense-FFN floor; never IQ2_XXS"}, true
	case V4ClassHead:
		return v41Floor{V41FormatFP8, "LM head; never low-bit"}, true
	case V4ClassEmbedding:
		return v41Floor{V41FormatFP8, "token embedding; never low-bit"}, true
	case V4ClassNorm:
		return v41Floor{V41FormatFP8, "norms/layernorms and small biases; never low-bit"}, true
	case V4ClassMTP:
		return v41Floor{V41FormatFP8, "retained MTP/DSpark draft head floor; low-bit collapses self-speculation acceptance"}, true
	case V4ClassVision:
		return v41Floor{V41FormatFP8, "ViT tower floor if retained; otherwise SKIP before the floor is consulted"}, true
	}
	return v41Floor{}, false
}

// classifyV41SensitivityGroup places a canonical HF safetensors tensor name in a
// sensitivity group. It first peels the v41-only markers (Engram, hyper-conn,
// vision) BEFORE falling through to classifyV4Tensor, so e.g. an Engram table
// that the V4 classifier would leave unplaced becomes a keyed group. Returns
// ok=false for a name it cannot place â€” the caller turns that into a fail-closed
// refusal.
func classifyV41SensitivityGroup(name string) (v41SensitiveGroup, V4TensorClass, bool) {
	lower := strings.ToLower(name)

	// Vision tower â€” the loader's exact prefix (mirrors classifyV4Tensor).
	if strings.HasPrefix(name, "model.visual.") {
		return v41GroupVision, V4ClassVision, true
	}
	// Engram group: layer tensors carry "engram" in the path. Split the learned
	// write gate from the n-gram hash table (the ~203 GB share).
	if strings.Contains(lower, "engram") {
		if strings.Contains(lower, "gate") {
			return v41GroupEngramGate, V4ClassRoutedExpert, true
		}
		return v41GroupEngramTable, V4ClassRoutedExpert, true
	}
	// Hyper-connection combine matrices (Sinkhorn residual mix).
	if strings.Contains(lower, ".hc.") || strings.Contains(lower, "hyper_conn") || strings.Contains(lower, "hyperconnection") {
		return v41GroupHyperConn, V4ClassAttention, true
	}

	class, ok := classifyV4Tensor(name)
	if !ok {
		return "", "", false
	}
	return v41SensitiveGroup(class), class, true
}

// v41FloorFor resolves the min-safe floor for a tensor name. Engram tables and
// their gate share the engram-class floor; routed-expert down-proj is floored
// strictly higher (never below trained FP4) than up/gate. Returns ok=false for
// an unplaceable name.
func v41FloorFor(name string) (v41SensitiveGroup, V4TensorClass, v41Floor, bool) {
	group, class, ok := classifyV41SensitivityGroup(name)
	if !ok {
		return "", "", v41Floor{}, false
	}
	// The dedicated v41-only groups (Engram tables/gate, hyper-connection, ViT)
	// carry their own floor and MUST NOT be dragged into the routed-expert
	// class special-case below — an Engram table is ~203 GB of n-gram memory,
	// not a routed expert.
	if fl, ok := v41SensitivityFloors[group]; ok {
		return group, class, fl, true
	}
	// Routed-expert down-projection is the strictest expert floor: never below
	// the trained FP4. Everything else in the expert class floors at IQ3_XXS.
	if class == V4ClassRoutedExpert {
		lower := strings.ToLower(name)
		if strings.Contains(lower, "down") {
			return group, class, v41Floor{V41FormatFP4, "routed-expert down-proj: KLD blowup at low bits; never re-quant below trained FP4"}, true
		}
		return group, class, v41Floor{V41FormatIQ3_XXS, "routed-expert up/gate tolerate ~3-bit; never IQ2_XXS"}, true
	}
	fl, ok := v41ClassFloors(class)
	if !ok {
		return group, class, v41Floor{}, false
	}
	return group, class, fl, true
}

// V41SensitivityFloor reports the min-safe re-quant floor for a sensitivity
// group. It FAILS CLOSED on an unplaced group: ok=false means the caller must
// refuse rather than assume a default.
func V41SensitivityFloor(group v41SensitiveGroup) (V41RequantFormat, string, bool) {
	if fl, ok := v41SensitivityFloors[group]; ok {
		return fl.Format, fl.Why, true
	}
	// A group with no explicit v41 floor falls back to the V4 class floor, which
	// is total over classes. Anything else refuses closed.
	if fl, ok := v41ClassFloors(V4TensorClass(group)); ok {
		return fl.Format, fl.Why, true
	}
	return "", "", false
}

// UnsupportedRequantError is the typed, fail-closed refusal the re-quant
// admission guard returns. It is the re-quant sibling of
// UnsupportedFP4TensorError: it names the tensor, its class/group, the floor
// that binds it, and the proposed format that fell below the floor.
type UnsupportedRequantError struct {
	Tensor   string
	Class    V4TensorClass
	Group    v41SensitiveGroup
	Floor    V41RequantFormat
	Proposed V41RequantFormat
	Why      string
}

func (e *UnsupportedRequantError) Error() string {
	group := string(e.Group)
	if group == "" {
		group = "<unclassified>"
	}
	return "model: DeepSeek V4.1 re-quant admission refused tensor " + e.Tensor +
		" (group " + group + ", proposed " + string(e.Proposed) + "): " + e.Why +
		". The class floor is " + string(e.Floor) +
		"; re-quantizing below it could crush an architecturally sensitive group, so the" +
		" guard fails closed (issue #12932); see docs/deepseek/v4-fp4-quant-support-plan.md."
}

// V41RequantVerdict is the per-tensor re-quant admission result.
type V41RequantVerdict struct {
	Name        string            `json:"name"`
	Group       v41SensitiveGroup `json:"group,omitempty"`
	Class       V4TensorClass     `json:"class,omitempty"`
	Floor       V41RequantFormat  `json:"floor,omitempty"`
	Proposed    V41RequantFormat  `json:"proposed"`
	Disposition V4Disposition     `json:"disposition"`
	Reason      string            `json:"reason,omitempty"`
}

// admitRequantTensor is the pure per-tensor kernel: classify the tensor into a
// sensitivity group, resolve the group floor, rank the proposed format against
// it, and decide ADMIT / SKIP / REFUSE. It fails closed on:
//   - a tensor name it cannot place,
//   - a proposed format tag outside the closed set,
//   - a proposed format below the class floor,
//   - a group with no declared floor.
//
// The ViT tower is SKIPped exactly as the load path drops it (its dtype is never
// asserted); every other group is floored.
func admitRequantTensor(t V4TensorMeta, proposed V41RequantFormat) (V41RequantVerdict, *UnsupportedRequantError) {
	proposedBits, knownFormat := v41RequantFormatBits(proposed)
	if !knownFormat {
		why := "unknown re-quant format tag " + v4Quote(string(proposed))
		err := &UnsupportedRequantError{Tensor: t.Name, Proposed: proposed, Why: why}
		return V41RequantVerdict{Name: t.Name, Proposed: proposed, Disposition: V4Refuse, Reason: why}, err
	}
	group, class, ok := classifyV41SensitivityGroup(t.Name)
	if !ok {
		why := "unrecognized tensor class for re-quant admission"
		err := &UnsupportedRequantError{Tensor: t.Name, Proposed: proposed, Why: why}
		return V41RequantVerdict{Name: t.Name, Proposed: proposed, Disposition: V4Refuse, Reason: why}, err
	}
	if class == V4ClassVision {
		reason := "multimodal vision tower dropped (text forward never reads it)"
		return V41RequantVerdict{Name: t.Name, Group: group, Class: class, Proposed: proposed, Disposition: V4Skip, Reason: reason}, nil
	}
	_, _, fl, ok := v41FloorFor(t.Name)
	if !ok {
		why := "no declared re-quant floor for group " + string(group)
		err := &UnsupportedRequantError{Tensor: t.Name, Class: class, Group: group, Proposed: proposed, Why: why}
		return V41RequantVerdict{Name: t.Name, Group: group, Class: class, Proposed: proposed, Disposition: V4Refuse, Reason: why}, err
	}
	floorBits, _ := v41RequantFormatBits(fl.Format)
	verdict := V41RequantVerdict{Name: t.Name, Group: group, Class: class, Floor: fl.Format, Proposed: proposed}
	if proposedBits < floorBits {
		why := fmt.Sprintf("proposed %s (%.4f bpw) is below the %s floor (%.4f bpw)", proposed, proposedBits, fl.Format, floorBits)
		err := &UnsupportedRequantError{Tensor: t.Name, Class: class, Group: group, Floor: fl.Format, Proposed: proposed, Why: why}
		verdict.Disposition = V4Refuse
		verdict.Reason = why
		return verdict, err
	}
	verdict.Disposition = V4Admit
	return verdict, nil
}
