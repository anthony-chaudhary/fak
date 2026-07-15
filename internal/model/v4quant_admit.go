package model

import (
	"fmt"
	"sort"
	"strings"
)

// v4quant_admit.go — the per-tensor precision ADMISSION gate for DeepSeek V4's
// mixed-precision checkpoint (issue #3019, parent epic #3006; plan doc
// docs/deepseek/v4-fp4-quant-support-plan.md).
//
// V4 is not "another FP8 model": its routed MoE expert weights and its attention
// lightning-indexer QK path are FP4 (NVFP4/E2M1, trained with FP4-QAT), most other
// parameters are FP8, and a few (norms, embeddings, biases) stay high precision.
// This file is the plan's "first landable witness": a metadata-only detector that
// classifies every tensor into a V4 tensor class, asserts the precision that class
// is expected to carry, and — the property fak did NOT have before — FAILS CLOSED on
// an unrecognized FP4 tensor class instead of silently mis-loading it.
//
// Nothing here reads weights, runs a GEMM, or decodes an FP4 nibble: admission is
// pure bookkeeping over a checkpoint's tensor INDEX (names + dtypes + shapes), the
// exact thing the acceptance asks a Go fixture to parse "without downloading full
// weights". The tensor names mirror the canonical HuggingFace safetensors convention
// the real loader already routes (isQuantWeight, safetensors_quant.go), so the class
// map is one description of that same layout — TestV4AdmitConsistentWithQuantGate
// pins the agreement rather than letting this drift into a second guess.

// V4Precision is the closed set of precisions a V4 tensor may carry. FP4 and FP8 are
// the two the checkpoint actually mixes; V4High covers tensors kept in bf16/f16/f32
// (norms, embeddings, small biases); V4PrecisionUnknown is the fail-closed sentinel a
// caller must never admit.
type V4Precision string

const (
	// V4FP4 — 4-bit (NVFP4/E2M1) weights: routed MoE experts and the indexer QK path.
	V4FP4 V4Precision = "FP4"
	// V4FP8 — 8-bit (E4M3/E5M2) weights: attention projections, shared experts, router,
	// dense FFN, and (when untied) the LM head.
	V4FP8 V4Precision = "FP8"
	// V4High — kept at bf16/f16/f32: norms, layernorms, embeddings, correction bias.
	V4High V4Precision = "HIGH"
	// V4PrecisionUnknown — an unplaceable dtype. Never admissible; forces fail-closed.
	V4PrecisionUnknown V4Precision = "UNKNOWN"
)

// v4DtypePrecision maps a checkpoint dtype tag to the precision family the admission
// table reasons about. It recognizes the HF safetensors FP8 tags, the FP4 tags a V4
// NVFP4 checkpoint carries (or the logical F4_E2M1 the metadata fixture uses), and the
// high-precision float tags. Anything else is V4PrecisionUnknown so an unforeseen tag
// fails closed rather than being waved through as some default.
func v4DtypePrecision(dtype string) V4Precision {
	switch strings.ToUpper(strings.TrimSpace(dtype)) {
	case "F8_E4M3", "F8_E5M2", "FP8", "E4M3", "E5M2":
		return V4FP8
	case "F4_E2M1", "E2M1", "FP4", "NVFP4", "MXFP4":
		return V4FP4
	case "BF16", "F16", "F32", "F64", "FP16", "FP32":
		return V4High
	}
	return V4PrecisionUnknown
}

// V4TensorClass partitions a V4 checkpoint's tensors for the admission table
// (docs/deepseek/v4-fp4-quant-support-plan.md, "Per-tensor precision admission
// table"). The plan enumerates eight classes; V4ClassDenseFFN and V4ClassEmbedding
// are added so a FULL checkpoint (which has dense-MLP layers and an embedding matrix,
// both FP8/HIGH) classifies completely instead of tripping the fail-closed path on a
// perfectly legitimate non-FP4 weight, and V4ClassVision mirrors the loader's dropped
// multimodal tower. The FP4-bearing classes remain exactly the two the plan names:
// routed experts and the indexer QK path.
type V4TensorClass string

const (
	V4ClassRoutedExpert V4TensorClass = "routed_expert" // .mlp.experts.<e>.{gate,up,down}_proj.weight → FP4
	V4ClassIndexerQK    V4TensorClass = "indexer_qk"    // .self_attn.indexer.{wq_b,wk,weights_proj}   → FP4
	V4ClassSharedExpert V4TensorClass = "shared_expert" // .mlp.shared_experts.*.weight                → FP8
	V4ClassAttention    V4TensorClass = "attention"     // q/k/v/o + MLA q_a/q_b/kv_a_proj_with_mqa/kv_b→ FP8
	V4ClassRouter       V4TensorClass = "router"        // .mlp.gate.weight                            → FP8
	V4ClassDenseFFN     V4TensorClass = "dense_ffn"     // dense-layer .mlp.{gate,up,down}_proj.weight  → FP8
	V4ClassHead         V4TensorClass = "head"          // lm_head.weight (untied FP8, or tied HIGH)   → FP8/HIGH
	V4ClassEmbedding    V4TensorClass = "embedding"     // *.embed_tokens.weight                       → HIGH
	V4ClassNorm         V4TensorClass = "norm"          // *_norm, *_layernorm, indexer.k_norm, biases → HIGH/FP8
	V4ClassMTP          V4TensorClass = "mtp"           // mtp.* speculative-decode head (dropped)     → SKIP
	V4ClassVision       V4TensorClass = "vision"        // model.visual.* multimodal tower (dropped)   → SKIP
)

// v4ExpectedPrecisions is the closed allow-set of precisions a class may carry. A
// declared precision outside its class's set is a fail-closed refusal — that is what
// catches the "treat V4 as just another FP8 model" mistake (a routed expert that came
// in FP8) as well as an FP4 tensor landing in a class that must not be FP4. Classes
// the plan marks "FP8 / kept f32" carry both FP8 and HIGH in their set.
func v4ExpectedPrecisions(class V4TensorClass) []V4Precision {
	switch class {
	case V4ClassRoutedExpert, V4ClassIndexerQK:
		return []V4Precision{V4FP4}
	case V4ClassSharedExpert, V4ClassAttention, V4ClassRouter, V4ClassDenseFFN:
		return []V4Precision{V4FP8}
	case V4ClassHead:
		return []V4Precision{V4FP8, V4High}
	case V4ClassNorm:
		return []V4Precision{V4High, V4FP8}
	case V4ClassEmbedding:
		return []V4Precision{V4High}
	case V4ClassMTP:
		// A RETAINED MTP/draft head (RetainMTP) is a LOADED tensor, and it is FLOORED to a
		// non-FP4 minimum: a draft head admitted at FP4/int4 collapses self-speculation
		// acceptance, so speculation would decode no faster (often slower) than a plain
		// forward. FP8 is the floor; HIGH is fine above it. When RetainMTP is clear the head
		// is SKIPPED before this allow-set is ever consulted, so this floor only bites a
		// retained head (#4353). The floor is set by DRAFT-ACCEPTANCE, not GEMV-cosine: the
		// measured acceptance numbers that justify FP8-as-floor are DGX-gated (real GLM-5.2
		// self-spec decode, parent epic #3006); the floor itself is the composable code here.
		return []V4Precision{V4FP8, V4High}
	case V4ClassVision:
		// Vision tower is always dropped at load; its dtype is never asserted.
		return nil
	}
	return nil
}

// classifyV4Tensor maps a canonical HF safetensors tensor name to its V4 class. The
// ordering is significant: the most specific markers (MTP module, the indexer QK
// path, routed vs shared experts) are tested before the generic attention / dense-FFN
// / norm fall-throughs, so e.g. an indexer projection is never mistaken for a plain
// attention weight. Returns ok=false for a name it cannot place — the caller turns
// that into a fail-closed refusal.
func classifyV4Tensor(name string) (V4TensorClass, bool) {
	switch {
	// Tensors the loader drops at load time (skipLoadTensor / quantSourceTensorName in
	// safetensors.go / safetensors_quant.go): the multimodal vision tower (always) and
	// the MTP / speculative-decoding head (unless RetainMTP). Mirror the loader's EXACT
	// prefixes so the admission gate SKIPs precisely what the loader drops — a canonical
	// mtp.* / model.visual.* tensor must never trip the fail-closed path. The dot-prefixed
	// hand-rolled markers a first cut used (".eh_proj.", ".mtp.", "nextn", …) missed the
	// loader's top-level "mtp." prefix and are replaced by it. TestV4AdmitAgreesWithLoaderDrop
	// pins this against skipLoadTensor itself.
	case strings.HasPrefix(name, "model.visual."):
		return V4ClassVision, true
	case strings.HasPrefix(name, "mtp."):
		return V4ClassMTP, true

	// Attention lightning-indexer QK path (FP4). Its own k_norm stays high precision,
	// so peel that off into the norm class before claiming the indexer QK class.
	case strings.Contains(name, ".self_attn.indexer."):
		if strings.Contains(name, "_norm") {
			return V4ClassNorm, true
		}
		return V4ClassIndexerQK, true

	// Routed MoE experts (FP4) — checked before shared_experts and dense FFN.
	case strings.Contains(name, ".mlp.experts."), strings.Contains(name, ".ffn.experts."):
		return V4ClassRoutedExpert, true
	case strings.Contains(name, ".mlp.shared_experts."), strings.Contains(name, ".ffn.shared_experts."):
		return V4ClassSharedExpert, true

	// MoE router / gate weight (FP8). The e_score_correction_bias is a small f32
	// vector handled by the norm/keep bucket below, not here.
	case strings.HasSuffix(name, ".mlp.gate.weight"), strings.HasSuffix(name, ".ffn.gate.weight"):
		return V4ClassRouter, true

	// Dense-layer FFN (the first n_dense_layers) — non-expert gate/up/down (FP8).
	case strings.HasSuffix(name, ".mlp.gate_proj.weight"),
		strings.HasSuffix(name, ".mlp.up_proj.weight"),
		strings.HasSuffix(name, ".mlp.down_proj.weight"):
		return V4ClassDenseFFN, true

	// Attention projections including MLA latents (FP8).
	case strings.HasSuffix(name, ".self_attn.q_proj.weight"),
		strings.HasSuffix(name, ".self_attn.k_proj.weight"),
		strings.HasSuffix(name, ".self_attn.v_proj.weight"),
		strings.HasSuffix(name, ".self_attn.o_proj.weight"),
		strings.HasSuffix(name, ".self_attn.q_a_proj.weight"),
		strings.HasSuffix(name, ".self_attn.q_b_proj.weight"),
		strings.HasSuffix(name, ".self_attn.kv_a_proj_with_mqa.weight"),
		strings.HasSuffix(name, ".self_attn.kv_b_proj.weight"):
		return V4ClassAttention, true

	// LM head (untied FP8, or tied to the embedding at high precision).
	case name == "lm_head.weight", strings.HasSuffix(name, ".lm_head.weight"):
		return V4ClassHead, true

	// Token embedding matrix — kept high precision.
	case strings.HasSuffix(name, "embed_tokens.weight"):
		return V4ClassEmbedding, true

	// Norms, layernorms, and small bias vectors — kept high precision (or FP8).
	case strings.HasSuffix(name, "_norm.weight"),
		strings.HasSuffix(name, "_layernorm.weight"),
		strings.HasSuffix(name, ".norm.weight"),
		strings.HasSuffix(name, "e_score_correction_bias"),
		strings.HasSuffix(name, ".bias"):
		return V4ClassNorm, true
	}
	return "", false
}

// V4TensorMeta is one row of a checkpoint's tensor index: enough to admit against,
// and nothing more. It is exactly what a safetensors header or an HF tensor-index
// JSON carries per tensor — no data pointer, no bytes.
type V4TensorMeta struct {
	Name  string `json:"name"`
	Dtype string `json:"dtype"`
	Shape []int  `json:"shape,omitempty"`
}

// V4Disposition is the closed outcome the admission gate returns for one tensor.
type V4Disposition string

const (
	// V4Admit — recognized class whose declared dtype is within the class's allow-set.
	V4Admit V4Disposition = "ADMIT"
	// V4Skip — recognized but intentionally not loaded (the MTP/nextn module).
	V4Skip V4Disposition = "SKIP"
	// V4Refuse — fail closed: an unknown class, an unknown dtype, or a precision the
	// tensor's class does not permit.
	V4Refuse V4Disposition = "REFUSE"
)

// V4TensorVerdict is the per-tensor admission result.
type V4TensorVerdict struct {
	Name        string        `json:"name"`
	Class       V4TensorClass `json:"class,omitempty"`
	Precision   V4Precision   `json:"precision"`
	Disposition V4Disposition `json:"disposition"`
	Reason      string        `json:"reason,omitempty"`
}

// UnsupportedFP4TensorError is the typed, fail-closed refusal the V4 quant admission
// gate returns when it meets a tensor it cannot safely place: an FP4-typed tensor
// whose name classifies into no known FP4-bearing class (the headline "unrecognized
// FP4 tensor class" the #3019 acceptance requires), a tensor whose declared precision
// is outside its class's allow-set, or a tensor carrying a dtype tag the loader does
// not recognize at all. It is the FP4 sibling of arch_support.go's UnsupportedArchError
// (a named LOAD-time refusal, not a mid-request panic) but keeps its own message so a
// caller can tell a quant-admission refusal from an unsupported-forward refusal.
type UnsupportedFP4TensorError struct {
	// Tensor is the offending tensor name.
	Tensor string
	// Dtype is the precision tag the checkpoint declared for it.
	Dtype string
	// Class is the class it was placed in, or "" if it could not be classified.
	Class V4TensorClass
	// Why is the specific fail-closed reason (unrecognized class / precision mismatch /
	// unknown dtype).
	Why string
}

func (e *UnsupportedFP4TensorError) Error() string {
	class := string(e.Class)
	if class == "" {
		class = "<unclassified>"
	}
	return "model: DeepSeek V4 quant admission refused tensor " + e.Tensor +
		" (dtype " + e.Dtype + ", class " + class + "): " + e.Why +
		". Admitting it could silently mis-load a mixed-FP4/FP8 checkpoint, so the loader" +
		" fails closed (issue #3019); see docs/deepseek/v4-fp4-quant-support-plan.md."
}

// admitV4Tensor is the pure per-tensor kernel: classify, resolve the declared
// precision, and decide ADMIT / SKIP / REFUSE. A REFUSE verdict carries a non-nil
// *UnsupportedFP4TensorError so a caller can both inspect the verdict and return the
// typed error.
func admitV4Tensor(t V4TensorMeta) (V4TensorVerdict, *UnsupportedFP4TensorError) {
	prec := v4DtypePrecision(t.Dtype)
	class, ok := classifyV4Tensor(t.Name)
	if !ok {
		why := "unrecognized tensor class"
		if prec == V4FP4 {
			// The exact property the acceptance names: an FP4 tensor we cannot place.
			why = "unrecognized FP4 tensor class"
		}
		err := &UnsupportedFP4TensorError{Tensor: t.Name, Dtype: t.Dtype, Why: why}
		return V4TensorVerdict{Name: t.Name, Precision: prec, Disposition: V4Refuse, Reason: why}, err
	}
	// The vision tower is ALWAYS dropped. The MTP head is dropped by default too, but when
	// RetainMTP retains it for self-speculation it becomes a LOADED tensor that must clear
	// the non-FP4 precision FLOOR below (#4353) instead of being skipped — a retained draft
	// head is only worth keeping if it was not quantized into speculation-collapsing FP4.
	if class == V4ClassVision || (class == V4ClassMTP && !RetainMTP) {
		reason := "MTP speculative-decode head dropped by default (retain via RetainMTP)"
		if class == V4ClassVision {
			reason = "multimodal vision tower dropped (text forward never reads it)"
		}
		return V4TensorVerdict{Name: t.Name, Class: class, Precision: prec, Disposition: V4Skip,
			Reason: reason}, nil
	}
	if prec == V4PrecisionUnknown {
		why := "unknown dtype tag " + v4Quote(t.Dtype)
		err := &UnsupportedFP4TensorError{Tensor: t.Name, Dtype: t.Dtype, Class: class, Why: why}
		return V4TensorVerdict{Name: t.Name, Class: class, Precision: prec, Disposition: V4Refuse, Reason: why}, err
	}
	allowed := v4ExpectedPrecisions(class)
	if !v4PrecisionAllowed(prec, allowed) {
		why := fmt.Sprintf("class %s expects %s, got %s", class, v4JoinPrecisions(allowed), prec)
		err := &UnsupportedFP4TensorError{Tensor: t.Name, Dtype: t.Dtype, Class: class, Why: why}
		return V4TensorVerdict{Name: t.Name, Class: class, Precision: prec, Disposition: V4Refuse, Reason: why}, err
	}
	return V4TensorVerdict{Name: t.Name, Class: class, Precision: prec, Disposition: V4Admit}, nil
}

func v4PrecisionAllowed(p V4Precision, allowed []V4Precision) bool {
	for _, a := range allowed {
		if a == p {
			return true
		}
	}
	return false
}

func v4JoinPrecisions(ps []V4Precision) string {
	if len(ps) == 0 {
		return "(none)"
	}
	parts := make([]string, len(ps))
	for i, p := range ps {
		parts[i] = string(p)
	}
	return strings.Join(parts, "|")
}

// v4Quote quotes a dtype tag for an error message without pulling the stdlib strconv
// into this file's import set for a single use.
func v4Quote(s string) string { return "\"" + s + "\"" }

// V4AdmitReport is the whole-checkpoint admission result: the per-class tensor counts,
// the per-class precisions actually observed, the skipped (MTP) tensors, and — when
// the checkpoint fails closed — the first refusing verdict. It is JSON-serializable so
// a follow-on ticket (the FP4 dequant/GEMM work) can consume the admitted class map as
// its worklist.
type V4AdmitReport struct {
	Model        string                     `json:"model,omitempty"`
	TotalTensors int                        `json:"total_tensors"`
	Admitted     int                        `json:"admitted"`
	Skipped      int                        `json:"skipped"`
	ClassCounts  map[V4TensorClass]int      `json:"class_counts"`
	ClassPrec    map[V4TensorClass][]string `json:"class_precisions"`
	OK           bool                       `json:"ok"`
	Refusal      *V4TensorVerdict           `json:"refusal,omitempty"`
}

// AdmitV4Checkpoint runs the admission gate over an entire tensor index and FAILS
// CLOSED on the first tensor it cannot place: it returns the partial report plus a
// typed *UnsupportedFP4TensorError the moment any tensor refuses, rather than
// admitting the rest of a checkpoint whose precision contract is already broken. On a
// fully-admissible index it returns a report with OK=true and a nil error.
//
// This is the acceptance witness for #3019: hand it the metadata parsed from a V4
// tensor index (no weights) and it either produces the per-class admission map or
// names exactly which tensor made the checkpoint unloadable and why.
func AdmitV4Checkpoint(model string, tensors []V4TensorMeta) (V4AdmitReport, error) {
	rep := V4AdmitReport{
		Model:        model,
		TotalTensors: len(tensors),
		ClassCounts:  map[V4TensorClass]int{},
		ClassPrec:    map[V4TensorClass][]string{},
		OK:           true,
	}
	seenPrec := map[V4TensorClass]map[V4Precision]bool{}
	for _, t := range tensors {
		v, err := admitV4Tensor(t)
		if err != nil {
			verdict := v
			rep.OK = false
			rep.Refusal = &verdict
			return rep, err
		}
		switch v.Disposition {
		case V4Skip:
			rep.Skipped++
			rep.ClassCounts[v.Class]++
		case V4Admit:
			rep.Admitted++
			rep.ClassCounts[v.Class]++
			if seenPrec[v.Class] == nil {
				seenPrec[v.Class] = map[V4Precision]bool{}
			}
			seenPrec[v.Class][v.Precision] = true
		}
	}
	for class, precs := range seenPrec {
		list := make([]string, 0, len(precs))
		for p := range precs {
			list = append(list, string(p))
		}
		sort.Strings(list)
		rep.ClassPrec[class] = list
	}
	return rep, nil
}
