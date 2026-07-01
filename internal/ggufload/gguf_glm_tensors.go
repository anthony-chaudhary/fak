package ggufload

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// gguf_glm_tensors.go — the glm_moe_dsa (GLM-5.2) per-layer GGUF tensor-name map, the
// Pillar-1 "tensor names" slice of the native-753B track (docs/notes/
// native-753b-track-staged-plan.md). It maps the 1:1 GLM-specific GGUF tensor suffixes to
// the canonical HF names internal/model's native glm_dsa forward already consumes
// (self_attn.q_a_proj / kv_a_proj_with_mqa / kv_b_proj …, the indexer wq_b/wk/weights_proj,
// the router mlp.gate.weight + its e_score_correction_bias, and mlp.shared_experts.*).
//
// VALIDATED KEY SPELLINGS (2026-06-24). The GGUF-side spellings were re-pinned against the real
// community GLM-5.2-Q4_K_M GGUF on the lab GPU server (general.architecture "glm-dsa", llama.cpp
// LLM_ARCH_GLM_DSA — fak normalizes that to "glm_moe_dsa", see canonicalGGUFArch). The MLA + MoE
// + shared-expert names matched the guessed deepseek2.* convention; the DSA-indexer names did
// NOT — the real file uses an "indexer." sub-namespace (indexer.attn_q_b / indexer.attn_k /
// indexer.k_norm / indexer.proj), so the earlier best-guess attn_indexer_* names were corrected.
//
// MLA KV-b SPLIT (handled by the 2→1 merge, NOT this map): the real file splits the KV-b
// up-projection into SEPARATE blk.<L>.attn_k_b + blk.<L>.attn_v_b tensors, whereas fak's forward
// consumes ONE combined self_attn.kv_b_proj.weight. Those two names are deliberately left unmapped
// here so the loader's merge pre-pass (glmMoeDsaSplitKVB + mergeGLMMoeDsaKVB in this file, called
// from gguf_weightsource.go before CanonicalTensorNameArch) combines them — transposing attn_k_b
// per head (DeepSeek MLA "weight absorption": k_b is stored transposed, v_b is not) and laying out
// [k_nope rows, then v rows] per head. Verified against llama.cpp's convert split + fak's forward.
//
// NOT mapped here (by design): the batched ROUTED experts ffn_gate_exps / ffn_up_exps /
// ffn_down_exps. Each is a single [E,…] blob that must split into E per-expert canonical
// tensors (mlp.experts.<e>.{gate,up,down}_proj.weight) — a 1→E expansion CanonicalTensorNameArch
// (one name in, one name out) structurally cannot express. The loader-side expert splitter
// (glmMoeDsaBatchedExpert) handles that before CanonicalTensorNameArch is consulted.

// glmMoeDsaGGUFSuffix is the provisional GGUF-side spelling of each 1:1 glm_moe_dsa per-layer
// tensor (the part after "blk.<L>."). Grouped so the high-confidence deepseek2-convention
// names and the best-guess indexer names are visibly separate. RE-PIN THE INDEXER BLOCK
// against a real GGUF header before treating the GGUF load as validated.
const (
	// MLA latent attention (deepseek2 convention).
	glmGGUFAttnQADown  = "attn_q_a.weight"       // q_a_proj   (down-projection to q_lora_rank)
	glmGGUFAttnQADownB = "attn_q_a.bias"         // q_a_proj.bias (optional)
	glmGGUFAttnQANorm  = "attn_q_a_norm.weight"  // q_a_layernorm (the RMSNorm on the q latent)
	glmGGUFAttnQBUp    = "attn_q_b.weight"       // q_b_proj   (up-projection to heads)
	glmGGUFAttnKVAMQA  = "attn_kv_a_mqa.weight"  // kv_a_proj_with_mqa
	glmGGUFAttnKVAMQAB = "attn_kv_a_mqa.bias"    // kv_a_proj_with_mqa.bias (optional)
	glmGGUFAttnKVANorm = "attn_kv_a_norm.weight" // kv_a_layernorm
	glmGGUFAttnKVB     = "attn_kv_b.weight"      // kv_b_proj
	glmGGUFAttnOutputB = "attn_output.bias"      // o_proj.bias (the .weight is the base map's)

	// MoE router (deepseek2 convention): ffn_gate_inp is the router gate matmul; exp_probs_b
	// is the per-expert score-correction bias added to the router logits before top-k.
	glmGGUFRouter     = "ffn_gate_inp.weight" // mlp.gate.weight
	glmGGUFRouterBias = "exp_probs_b.bias"    // mlp.gate.e_score_correction_bias

	// Shared experts (deepseek2 convention): the always-on expert run beside the routed ones.
	glmGGUFSharedGate = "ffn_gate_shexp.weight" // mlp.shared_experts.gate_proj.weight
	glmGGUFSharedUp   = "ffn_up_shexp.weight"   // mlp.shared_experts.up_proj.weight
	glmGGUFSharedDown = "ffn_down_shexp.weight" // mlp.shared_experts.down_proj.weight

	// DSA learned indexer — VALIDATED 2026-06-24 against the real GLM-5.2 (glm-dsa) Q4_K_M
	// GGUF on the lab GPU server (llama.cpp LLM_ARCH_GLM_DSA). The real spellings live under
	// an "indexer." sub-namespace and use "proj" (not "weights"); these replace the earlier
	// best-guess attn_indexer_* names, which the real file did NOT use.
	glmGGUFIndexerWQB     = "indexer.attn_q_b.weight" // indexer.wq_b
	glmGGUFIndexerWK      = "indexer.attn_k.weight"   // indexer.wk
	glmGGUFIndexerKNorm   = "indexer.k_norm.weight"   // indexer.k_norm.weight (RMSNorm on the index key)
	glmGGUFIndexerKNormB  = "indexer.k_norm.bias"     // indexer.k_norm.bias
	glmGGUFIndexerWeights = "indexer.proj.weight"     // indexer.weights_proj
)

// glmMoeDsaCanonicalSuffix maps a glm_moe_dsa per-layer GGUF tensor suffix (after "blk.<L>.")
// to the canonical HF suffix (after "model.layers.<L>.") the native glm_dsa forward reads.
// Returns ok=false for any suffix that is not GLM-specific so CanonicalTensorNameArch falls
// through to the shared base map (attn_norm, ffn_norm, attn_output.weight, and the leading
// dense layers' ffn_gate/up/down), and for the batched routed experts (intentionally
// unmapped — see the file header).
func glmMoeDsaCanonicalSuffix(suffix string) (string, bool) {
	mapped, ok := map[string]string{
		glmGGUFAttnQADown:  "self_attn.q_a_proj.weight",
		glmGGUFAttnQADownB: "self_attn.q_a_proj.bias",
		glmGGUFAttnQANorm:  "self_attn.q_a_layernorm.weight",
		glmGGUFAttnQBUp:    "self_attn.q_b_proj.weight",
		glmGGUFAttnKVAMQA:  "self_attn.kv_a_proj_with_mqa.weight",
		glmGGUFAttnKVAMQAB: "self_attn.kv_a_proj_with_mqa.bias",
		glmGGUFAttnKVANorm: "self_attn.kv_a_layernorm.weight",
		glmGGUFAttnKVB:     "self_attn.kv_b_proj.weight",
		glmGGUFAttnOutputB: "self_attn.o_proj.bias",

		glmGGUFRouter:     "mlp.gate.weight",
		glmGGUFRouterBias: "mlp.gate.e_score_correction_bias",

		glmGGUFSharedGate: "mlp.shared_experts.gate_proj.weight",
		glmGGUFSharedUp:   "mlp.shared_experts.up_proj.weight",
		glmGGUFSharedDown: "mlp.shared_experts.down_proj.weight",

		glmGGUFIndexerWQB:     "self_attn.indexer.wq_b.weight",
		glmGGUFIndexerWK:      "self_attn.indexer.wk.weight",
		glmGGUFIndexerKNorm:   "self_attn.indexer.k_norm.weight",
		glmGGUFIndexerKNormB:  "self_attn.indexer.k_norm.bias",
		glmGGUFIndexerWeights: "self_attn.indexer.weights_proj.weight",
	}[suffix]
	return mapped, ok
}

// Batched routed-expert GGUF tensors (deepseek2 convention): one [E,…] blob per layer that the
// loader must SPLIT 1→E. CanonicalTensorNameArch leaves these unmapped on purpose — a single name
// cannot become E per-expert names — so the split lives here + the loader (gguf_weightsource.go).
const (
	glmGGUFExpertsGate = "ffn_gate_exps.weight" // -> mlp.experts.<e>.gate_proj.weight
	glmGGUFExpertsUp   = "ffn_up_exps.weight"   // -> mlp.experts.<e>.up_proj.weight
	glmGGUFExpertsDown = "ffn_down_exps.weight" // -> mlp.experts.<e>.down_proj.weight
)

func archUsesGGUFBatchedMoEExperts(arch string) bool {
	switch arch {
	case "glm_moe_dsa", "qwen3moe":
		return true
	}
	return false
}

// glmMoeDsaSkipGGUFTensor reports whether a glm_moe_dsa GGUF tensor is part of the Multi-Token-
// Prediction (MTP / "nextn") speculative-decoding head or a multimodal vision tower — neither read
// by the text causal-LM forward. llama.cpp likewise ignores these ("model has unused tensor
// blk.<L>.nextn.*"); fak skips them at load so a real GLM-5.2 checkpoint does not fail on a tensor
// the forward never consumes (mirrors the safetensors path's skipLoadTensor mtp/visual drop).
func glmMoeDsaSkipGGUFTensor(name string) bool {
	// MTP head: the GGUF spells it "blk.<L>.nextn.*" (eh_proj/enorm/hnorm/shared_head_norm/...).
	if strings.Contains(name, ".nextn.") {
		return true
	}
	// Multimodal vision tower (when present): llama.cpp's "v.*" / "mm." conversion namespace.
	if strings.HasPrefix(name, "v.") || strings.HasPrefix(name, "mm.") {
		return true
	}
	return false
}

// glmMoeDsaSkipGGUFTensorForType reports whether a tensor should be dropped at load for the
// given model type: true only for a glm_moe_dsa file whose tensor is an MTP/vision tensor the
// text forward never reads (glmMoeDsaSkipGGUFTensor). It is the shared guard the loader and the
// byte-accounting estimators use so the "glm_moe_dsa" family check stays in one place.
func glmMoeDsaSkipGGUFTensorForType(modelType, name string) bool {
	return modelType == "glm_moe_dsa" && glmMoeDsaSkipGGUFTensor(name)
}

// glmMoeDsaBatchedExpert reports whether a glm_moe_dsa GGUF tensor name is a batched routed-expert
// blob and, if so, returns its layer index and the per-expert canonical projection name
// (gate_proj/up_proj/down_proj). These are the tensors the loader splits into E per-expert 2-D
// tensors (mlp.experts.<e>.<proj>.weight) — the form internal/model's MoE forward consumes
// (moe.go expertName). Detected from the GGUF name (not the canonical map) so the split happens
// BEFORE CanonicalTensorNameArch, which deliberately returns no mapping for these.
// parseGLMBlkLayerSuffix splits a "blk.<layer>.<suffix>" GGUF tensor name into its
// decoder layer index and the remaining suffix (everything after the layer's dot).
// It is the shared front half of the glm_moe_dsa name classifiers — each then matches
// the returned suffix against its own constant set. ok=false when the name is not a
// blk.<int>.* tensor, so a caller's classification falls through unchanged.
func parseGLMBlkLayerSuffix(name string) (layer int, suffix string, ok bool) {
	if !strings.HasPrefix(name, "blk.") {
		return 0, "", false
	}
	rest := strings.TrimPrefix(name, "blk.")
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return 0, "", false
	}
	l, err := strconv.Atoi(rest[:dot])
	if err != nil {
		return 0, "", false
	}
	return l, rest[dot+1:], true
}

func glmMoeDsaBatchedExpert(name string) (layer int, proj string, ok bool) {
	l, suffix, ok := parseGLMBlkLayerSuffix(name)
	if !ok {
		return 0, "", false
	}
	switch suffix {
	case glmGGUFExpertsGate:
		return l, "gate_proj", true
	case glmGGUFExpertsUp:
		return l, "up_proj", true
	case glmGGUFExpertsDown:
		return l, "down_proj", true
	}
	return 0, "", false
}

// splitGLMMoeDsaExperts expands a batched routed-expert tensor — already dequantized to f32, model
// shape [E, out, in] (the GGUF stores [in, out, E]; modelShapeFromGGUFDims reverses it) — into E
// per-expert canonical 2-D tensors model.layers.<layer>.mlp.experts.<e>.<proj>.weight of shape
// [out, in]. Expert e is the contiguous block data[e*out*in : (e+1)*out*in], so the split is a pure
// row-major reslice along the leading expert axis (bit-equal to manual slicing). Each expert's slice
// is copied into its own backing array so a later quantize/normalize cannot alias across experts.
// parseGLMMoeDsaExpertShape validates a batched glm_moe_dsa routed-expert shape and returns
// its [E,out,in] dims. It is the shared prologue of the f32 split (splitGLMMoeDsaExperts) and
// the raw k-quant split (splitGLMMoeDsaExpertsKQuantRaw); each caller maps the error onto its
// own return arity. The dim/error messages are byte-identical to the inlined checks it replaces.
func parseGLMMoeDsaExpertShape(shape []int) (e, out, in int, err error) {
	if len(shape) != 3 {
		return 0, 0, 0, fmt.Errorf("gguf: glm_moe_dsa batched expert tensor must be 3-D [E,out,in], got shape %v", shape)
	}
	e, out, in = shape[0], shape[1], shape[2]
	if e <= 0 || out <= 0 || in <= 0 {
		return 0, 0, 0, fmt.Errorf("gguf: glm_moe_dsa batched expert tensor has non-positive dim in [%d,%d,%d]", e, out, in)
	}
	return e, out, in, nil
}

func splitGLMMoeDsaExperts(layer int, proj string, shape []int, data []float32) ([]model.NamedTensorF32, error) {
	e, out, in, err := parseGLMMoeDsaExpertShape(shape)
	if err != nil {
		return nil, err
	}
	per := out * in
	if len(data) != e*per {
		return nil, fmt.Errorf("gguf: glm_moe_dsa expert blob [%d,%d,%d] has %d values, want %d", e, out, in, len(data), e*per)
	}
	tensors := make([]model.NamedTensorF32, e)
	for x := 0; x < e; x++ {
		seg := make([]float32, per)
		copy(seg, data[x*per:(x+1)*per])
		tensors[x] = model.NamedTensorF32{
			Name:  fmt.Sprintf("model.layers.%d.mlp.experts.%d.%s.weight", layer, x, proj),
			Shape: []int{out, in},
			Data:  seg,
		}
	}
	return tensors, nil
}

// NamedResidentQ4K is one per-expert tensor split as RAW Q4_K bytes (no dequant): the name, the
// model [out,in] shape, and the contiguous Q4_K super-block bytes for that expert.
type NamedResidentQ4K struct {
	Name  string
	Shape []int
	Raw   []byte
}

// q4kSuperBlockBytes / q4kSuperBlockWeights mirror internal/model/quant_q4k.go: a Q4_K super-block
// is 144 bytes per 256 weights. The batched expert blob is E contiguous experts, each out*in
// weights, so expert e's raw bytes are a clean byte slice IFF out*in is a multiple of 256 (which
// Q4_K requires of any quantized tensor). This lets the routed experts — the 417 GB bulk of GLM-5.2
// — load RESIDENT (raw bytes copied straight to host/VRAM, dequant fused into the GEMM) with NO
// f32 round-trip, the load-time lever that turns a ~100-min lean load into an I/O-bound one.
const (
	q4kSuperBlockBytes   = 144
	q4kSuperBlockWeights = 256
)

// splitGLMMoeDsaExpertsQ4KRaw expands a batched Q4_K routed-expert tensor (model shape [E,out,in])
// into E per-expert resident-Q4_K byte slices WITHOUT dequantizing. It is the raw-byte twin of
// splitGLMMoeDsaExperts; ok=false (no error) means the dims are not block-aligned for a clean raw
// split, so the caller falls back to the f32 split. Each expert's bytes are copied into their own
// backing array so a later resident-store cannot alias across experts.
func splitGLMMoeDsaExpertsQ4KRaw(layer int, proj string, shape []int, raw []byte) ([]NamedResidentQ4K, bool, error) {
	return splitGLMMoeDsaExpertsKQuantRaw(layer, proj, shape, raw, q4kSuperBlockBytes)
}

// splitGLMMoeDsaExpertsKQuantRaw is the byte-only legacy wrapper for 256-weight super-block formats
// (Q4_K/Q5_K/Q6_K/IQ3_XXS/IQ4_XS). Q8_0 callers must use splitGLMMoeDsaExpertsRawQuant with a
// 32-weight block geometry.
func splitGLMMoeDsaExpertsKQuantRaw(layer int, proj string, shape []int, raw []byte, blockBytes int) ([]NamedResidentQ4K, bool, error) {
	return splitGLMMoeDsaExpertsRawQuant(layer, proj, shape, raw, q4kSuperBlockWeights, blockBytes)
}

// splitGLMMoeDsaExpertsRawQuant expands a batched raw-quant routed-expert tensor (model shape
// [E,out,in]) into E per-expert resident byte slices WITHOUT dequantizing. ok=false (no error)
// means the row reduction dim is not block-aligned for a clean raw split, so the caller
// dequant-splits to f32.
func splitGLMMoeDsaExpertsRawQuant(layer int, proj string, shape []int, raw []byte, blockWeights, blockBytes int) ([]NamedResidentQ4K, bool, error) {
	e, out, in, err := parseGLMMoeDsaExpertShape(shape)
	if err != nil {
		return nil, false, err
	}
	if blockWeights <= 0 || blockBytes <= 0 {
		return nil, false, fmt.Errorf("gguf: glm_moe_dsa raw expert split got invalid block geometry weights=%d bytes=%d", blockWeights, blockBytes)
	}
	// Gate on the REDUCTION dim (in), not out*in: a resident raw-quant row must be a whole
	// number of quant blocks, since the GEMV dequantizes blocks ALONG each row. out*in alignment
	// alone can pass while in alignment fails and would then mis-block every row.
	if in%blockWeights != 0 {
		return nil, false, nil // not row-aligned -> caller dequant-splits to f32 instead
	}
	per := out * in
	perBytes := (per / blockWeights) * blockBytes
	if len(raw) != e*perBytes {
		return nil, false, fmt.Errorf("gguf: glm_moe_dsa raw-quant expert blob [%d,%d,%d] has %d raw bytes, want %d (blockWeights=%d blockBytes=%d)", e, out, in, len(raw), e*perBytes, blockWeights, blockBytes)
	}
	tensors := make([]NamedResidentQ4K, e)
	for x := 0; x < e; x++ {
		seg := make([]byte, perBytes)
		copy(seg, raw[x*perBytes:(x+1)*perBytes])
		tensors[x] = NamedResidentQ4K{
			Name:  fmt.Sprintf("model.layers.%d.mlp.experts.%d.%s.weight", layer, x, proj),
			Shape: []int{out, in},
			Raw:   seg,
		}
	}
	return tensors, true, nil
}

// The MLA KV-b up-projection is split in the GGUF (llama.cpp LLM_ARCH_GLM_DSA, inherited from
// DeepSeek2's convert: kv_b is view()'d [n_head, qk_nope+v_head, kv_lora], split into k_b/v_b,
// and k_b is TRANSPOSED) into these two per-layer tensors, which fak's forward consumes as ONE
// combined self_attn.kv_b_proj.weight. glmMoeDsaSplitKVB combines them — see mergeGLMMoeDsaKVB.
const (
	glmGGUFAttnKB = "attn_k_b.weight" // [n_head, kv_lora, qk_nope] — TRANSPOSED at convert time
	glmGGUFAttnVB = "attn_v_b.weight" // [n_head, v_head, kv_lora]   — NOT transposed
)

// glmMoeDsaSplitKVB reports whether a glm_moe_dsa GGUF tensor name is one half of the split MLA
// KV-b up-projection and, if so, returns its layer index and which half ("k" or "v"). These two
// names are deliberately left out of glmMoeDsaCanonicalSuffix so the loader's 2->1 merge pre-pass
// (mergeGLMMoeDsaKVB) handles them BEFORE CanonicalTensorNameArch — the same shape the batched
// expert splitter (glmMoeDsaBatchedExpert) uses, but combining two tensors instead of splitting one.
func glmMoeDsaSplitKVB(name string) (layer int, half string, ok bool) {
	l, suffix, ok := parseGLMBlkLayerSuffix(name)
	if !ok {
		return 0, "", false
	}
	switch suffix {
	case glmGGUFAttnKB:
		return l, "k", true
	case glmGGUFAttnVB:
		return l, "v", true
	}
	return 0, "", false
}

// glmKVBHalf buffers one dequantized half (attn_k_b or attn_v_b) of a layer's MLA KV-b projection
// until its partner arrives, so the 2->1 merge works regardless of tensor stream order.
type glmKVBHalf struct {
	kShape, vShape []int
	kData, vData   []float32
	haveK, haveV   bool
}

// bufferGLMKVBHalf records one half ("k"/"v") of layer L's split KV-b and, once BOTH halves are
// present, merges them (mergeGLMMoeDsaKVB), clears the buffer entry, and returns the combined
// kv_b_proj tensor with ready=true. While only one half is seen it returns ready=false. Shared by
// the quantized (QuantModelProfile) and f32 (F32Tensors) loader paths so they merge identically.
func bufferGLMKVBHalf(buf map[int]glmKVBHalf, layer int, half string, shape []int, data []float32) (model.NamedTensorF32, bool, error) {
	h := buf[layer]
	switch half {
	case "k":
		if h.haveK {
			return model.NamedTensorF32{}, false, fmt.Errorf("gguf: glm_moe_dsa duplicate attn_k_b for layer %d", layer)
		}
		h.kShape, h.kData, h.haveK = shape, data, true
	case "v":
		if h.haveV {
			return model.NamedTensorF32{}, false, fmt.Errorf("gguf: glm_moe_dsa duplicate attn_v_b for layer %d", layer)
		}
		h.vShape, h.vData, h.haveV = shape, data, true
	default:
		return model.NamedTensorF32{}, false, fmt.Errorf("gguf: glm_moe_dsa kv_b unknown half %q (layer %d)", half, layer)
	}
	if h.haveK && h.haveV {
		delete(buf, layer)
		merged, err := mergeGLMMoeDsaKVB(layer, h.kShape, h.kData, h.vShape, h.vData)
		return merged, true, err
	}
	buf[layer] = h
	return model.NamedTensorF32{}, false, nil
}

// glmKVBUnpaired returns an error naming any layer left with only one KV-b half after the tensor
// stream is exhausted — a malformed GGUF that would otherwise silently drop a layer's kv_b_proj.
func glmKVBUnpaired(buf map[int]glmKVBHalf) error {
	for layer, h := range buf {
		missing := "attn_v_b"
		if !h.haveK {
			missing = "attn_k_b"
		}
		return fmt.Errorf("gguf: glm_moe_dsa layer %d has only one KV-b half (missing %s)", layer, missing)
	}
	return nil
}

// mergeGLMMoeDsaKVB combines the split MLA tensors attn_k_b + attn_v_b (both already dequantized
// to f32, model-order row-major) into the single canonical self_attn.kv_b_proj.weight the native
// glm_dsa forward reads. Layout (verified against llama.cpp's convert kv_b split + fak's forward,
// internal/model/glm_dsa.go:58-76; numeric round-trip diff 0.0):
//
//	attn_k_b model shape [nH, kvLora, qkNope]  (k_b was TRANSPOSED at convert time)
//	attn_v_b model shape [nH, vHead,  kvLora]  (v_b was NOT transposed)
//	kv_b_proj target     [nH*(qkNope+vHead), kvLora]  row-major; per head h the qkNope k_nope rows
//	                     come first, then the vHead v rows, each row dotted against the kvLora latent.
//
// So the k part needs a per-head TRANSPOSE [kvLora,qkNope]->[qkNope,kvLora]; the v part is a
// straight copy. Fails loud on any shape mismatch — a wrong merge would silently corrupt the model.
func mergeGLMMoeDsaKVB(layer int, kShape []int, kData []float32, vShape []int, vData []float32) (model.NamedTensorF32, error) {
	if len(kShape) != 3 || len(vShape) != 3 {
		return model.NamedTensorF32{}, fmt.Errorf("gguf: glm_moe_dsa kv_b merge expects 3-D attn_k_b/attn_v_b, got k=%v v=%v (layer %d)", kShape, vShape, layer)
	}
	nH, kvLoraK, qkNope := kShape[0], kShape[1], kShape[2]
	nHv, vHead, kvLoraV := vShape[0], vShape[1], vShape[2]
	if nH != nHv {
		return model.NamedTensorF32{}, fmt.Errorf("gguf: glm_moe_dsa kv_b merge head mismatch attn_k_b nH=%d vs attn_v_b nH=%d (layer %d)", nH, nHv, layer)
	}
	if kvLoraK != kvLoraV {
		return model.NamedTensorF32{}, fmt.Errorf("gguf: glm_moe_dsa kv_b merge kv_lora mismatch attn_k_b=%d vs attn_v_b=%d (layer %d)", kvLoraK, kvLoraV, layer)
	}
	kvLora := kvLoraK
	if nH <= 0 || kvLora <= 0 || qkNope <= 0 || vHead <= 0 {
		return model.NamedTensorF32{}, fmt.Errorf("gguf: glm_moe_dsa kv_b merge non-positive dim nH=%d kvLora=%d qkNope=%d vHead=%d (layer %d)", nH, kvLora, qkNope, vHead, layer)
	}
	if len(kData) != nH*kvLora*qkNope {
		return model.NamedTensorF32{}, fmt.Errorf("gguf: glm_moe_dsa attn_k_b has %d values, want %d (layer %d)", len(kData), nH*kvLora*qkNope, layer)
	}
	if len(vData) != nH*vHead*kvLora {
		return model.NamedTensorF32{}, fmt.Errorf("gguf: glm_moe_dsa attn_v_b has %d values, want %d (layer %d)", len(vData), nH*vHead*kvLora, layer)
	}
	perHead := qkNope + vHead
	out := make([]float32, nH*perHead*kvLora)
	for h := 0; h < nH; h++ {
		base := h * perHead
		kHead := kData[h*kvLora*qkNope:] // [kvLora, qkNope] for this head
		vHeadData := vData[h*vHead*kvLora:]
		// k part: TRANSPOSE [kvLora,qkNope] -> rows [qkNope, kvLora]
		for o := 0; o < qkNope; o++ {
			dst := out[(base+o)*kvLora:]
			for i := 0; i < kvLora; i++ {
				dst[i] = kHead[i*qkNope+o]
			}
		}
		// v part: straight copy [vHead,kvLora] -> rows [vHead, kvLora]
		for o := 0; o < vHead; o++ {
			copy(out[(base+qkNope+o)*kvLora:(base+qkNope+o+1)*kvLora], vHeadData[o*kvLora:(o+1)*kvLora])
		}
	}
	return model.NamedTensorF32{
		Name:  fmt.Sprintf("model.layers.%d.self_attn.kv_b_proj.weight", layer),
		Shape: []int{nH * perHead, kvLora},
		Data:  out,
	}, nil
}
