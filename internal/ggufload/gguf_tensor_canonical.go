package ggufload

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func normalizeCanonicalTensorData(name string, data []float32, cfg model.Config) ([]float32, error) {
	if cfg.IsQwen35Hybrid() {
		if out, handled, err := normalizeQwen35OrdinaryNormTensor(name, data, cfg); handled || err != nil {
			return out, err
		}
		if out, handled, err := normalizeQwen35LinearTensor(name, data, cfg); handled || err != nil {
			return out, err
		}
	}
	switch {
	case strings.HasSuffix(name, ".self_attn.q_proj.weight"):
		if cfg.IsQwen35Hybrid() && cfg.AttnOutputGate {
			return unpermuteQwen35GatedQTensor(name, data, cfg.NumHeads, cfg.HeadDim, cfg.HiddenSize)
		}
		if ggufArchStoresHFRotaryLayout(cfg.ModelType) {
			return data, nil
		}
		return unpermuteRotaryTensor(name, data, cfg.NumHeads, cfg.HeadDim, cfg.HiddenSize)
	case strings.HasSuffix(name, ".self_attn.k_proj.weight"):
		if ggufArchStoresHFRotaryLayout(cfg.ModelType) {
			return data, nil
		}
		return unpermuteRotaryTensor(name, data, cfg.NumKVHeads, cfg.HeadDim, cfg.HiddenSize)
	default:
		return data, nil
	}
}

func normalizeQwen35OrdinaryNormTensor(name string, src []float32, cfg model.Config) ([]float32, bool, error) {
	want := 0
	switch {
	case name == "model.norm.weight":
		want = cfg.HiddenSize
	case strings.HasSuffix(name, ".input_layernorm.weight"),
		strings.HasSuffix(name, ".post_attention_layernorm.weight"):
		want = cfg.HiddenSize
	case strings.HasSuffix(name, ".self_attn.q_norm.weight"),
		strings.HasSuffix(name, ".self_attn.k_norm.weight"):
		want = cfg.HeadDim
	default:
		return nil, false, nil
	}
	if want > 0 && len(src) != want {
		return nil, true, fmt.Errorf("gguf: tensor %s has %d values, qwen35 norm wants %d", name, len(src), want)
	}
	return subtractOneFromTensor(src), true, nil
}

func subtractOneFromTensor(src []float32) []float32 {
	dst := make([]float32, len(src))
	for i, v := range src {
		dst[i] = v - 1
	}
	return dst
}

func normalizeQwen35LinearTensor(name string, src []float32, cfg model.Config) ([]float32, bool, error) {
	nK := cfg.LinearNumKeyHeads
	nV := cfg.LinearNumValueHeads
	kHd := cfg.LinearKeyHeadDim
	vHd := cfg.LinearValueHeadDim
	if nK <= 0 || nV <= 0 || kHd <= 0 || vHd <= 0 {
		return nil, false, nil
	}
	if nV%nK != 0 {
		return nil, true, fmt.Errorf("gguf: qwen35 linear heads are not divisible: value=%d key=%d", nV, nK)
	}
	if nV == nK {
		return nil, false, nil
	}
	if kHd != vHd {
		return nil, true, fmt.Errorf("gguf: qwen35 linear key/value head dims differ: key=%d value=%d", kHd, vHd)
	}
	keyDim := nK * kHd
	switch {
	case strings.HasSuffix(name, ".linear_attn.in_proj_qkv.weight"),
		strings.HasSuffix(name, ".self_attn.qkv_proj.weight"):
		out, err := reorderQwen35LinearQKVRows(name, src, keyDim, nK, nV, vHd, cfg.HiddenSize)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.conv1d.weight"):
		out, err := reorderQwen35LinearQKVRows(name, src, keyDim, nK, nV, vHd, cfg.LinearConvKernelDim)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.in_proj_z.weight"),
		strings.HasSuffix(name, ".self_attn.q_gate_proj.weight"):
		out, err := reorderQwen35InterleavedValueRows(name, src, nK, nV, vHd, cfg.HiddenSize)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.in_proj_a.weight"),
		strings.HasSuffix(name, ".linear_attn.in_proj_b.weight"):
		out, err := reorderQwen35InterleavedValueRows(name, src, nK, nV, 1, cfg.HiddenSize)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.out_proj.weight"):
		out, err := reorderQwen35InterleavedValueCols(name, src, cfg.HiddenSize, nK, nV, vHd)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.A_log"),
		strings.HasSuffix(name, ".linear_attn.dt_bias"):
		out, err := reorderQwen35InterleavedValueVector(name, src, nK, nV)
		return out, true, err
	case strings.HasSuffix(name, ".linear_attn.norm.weight"):
		if len(src) != vHd {
			return nil, true, fmt.Errorf("gguf: tensor %s has %d values, qwen35 linear norm wants %d", name, len(src), vHd)
		}
		return src, true, nil
	default:
		return nil, false, nil
	}
}

func reorderQwen35LinearQKVRows(name string, src []float32, keyDim, nK, nV, headDim, rowWidth int) ([]float32, error) {
	valDim := nV * headDim
	want := (2*keyDim + valDim) * rowWidth
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, qwen35 qkv shape wants %d", name, len(src), want)
	}
	dst := append([]float32(nil), src...)
	vOff := 2 * keyDim * rowWidth
	v, err := reorderQwen35InterleavedValueRows(name, src[vOff:], nK, nV, headDim, rowWidth)
	if err != nil {
		return nil, err
	}
	copy(dst[vOff:], v)
	return dst, nil
}

func reorderQwen35InterleavedValueRows(name string, src []float32, nK, nV, headSpan, rowWidth int) ([]float32, error) {
	if nK <= 0 || nV <= 0 || headSpan <= 0 || rowWidth <= 0 || nV%nK != 0 {
		return nil, fmt.Errorf("gguf: tensor %s has invalid qwen35 value layout nK=%d nV=%d span=%d width=%d", name, nK, nV, headSpan, rowWidth)
	}
	want := nV * headSpan * rowWidth
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, qwen35 value rows want %d", name, len(src), want)
	}
	ratio := nV / nK
	dst := make([]float32, len(src))
	rowBlock := headSpan * rowWidth
	for k := 0; k < nK; k++ {
		for r := 0; r < ratio; r++ {
			dstHead := k*ratio + r
			srcHead := r*nK + k
			copy(dst[dstHead*rowBlock:(dstHead+1)*rowBlock], src[srcHead*rowBlock:(srcHead+1)*rowBlock])
		}
	}
	return dst, nil
}

func reorderQwen35InterleavedValueCols(name string, src []float32, rows, nK, nV, headDim int) ([]float32, error) {
	if rows <= 0 || nK <= 0 || nV <= 0 || headDim <= 0 || nV%nK != 0 {
		return nil, fmt.Errorf("gguf: tensor %s has invalid qwen35 value column layout rows=%d nK=%d nV=%d headDim=%d", name, rows, nK, nV, headDim)
	}
	cols := nV * headDim
	want := rows * cols
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, qwen35 value columns want %d", name, len(src), want)
	}
	ratio := nV / nK
	dst := make([]float32, len(src))
	for row := 0; row < rows; row++ {
		rowOff := row * cols
		for k := 0; k < nK; k++ {
			for r := 0; r < ratio; r++ {
				dstHead := k*ratio + r
				srcHead := r*nK + k
				copy(dst[rowOff+dstHead*headDim:rowOff+(dstHead+1)*headDim], src[rowOff+srcHead*headDim:rowOff+(srcHead+1)*headDim])
			}
		}
	}
	return dst, nil
}

func reorderQwen35InterleavedValueVector(name string, src []float32, nK, nV int) ([]float32, error) {
	if len(src) != nV {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, qwen35 value vector wants %d", name, len(src), nV)
	}
	return reorderQwen35InterleavedValueRows(name, src, nK, nV, 1, 1)
}

// ggufArchStoresHFRotaryLayout reports whether a GGUF of this architecture stores its
// q/k projection weights already in the HF "rotate_half" (NEOX) RoPE layout — i.e.
// convert_hf_to_gguf.py did NOT permute them on the way out. fak's forward pass always
// applies the rotate_half convention (forward.go), so for these models the q/k weights
// must be consumed exactly as stored: running them through unpermuteRotaryTensor scrambles
// every head's rotary pairs and yields incoherent output.
//
// Only the llama-family NORM-rope architectures (llama — which also carries Mistral/Mixtral
// exports — baichuan, command-r, …) are permuted by the converter and therefore still need
// the unpermute. Anything not on this NEOX allow-list keeps the historical unpermute, so no
// currently-covered architecture (the "llama" rotary test, the qwen35 hybrid test) regresses.
func ggufArchStoresHFRotaryLayout(arch string) bool {
	switch arch {
	case "qwen2", "qwen2moe", "qwen3", "qwen3moe",
		"gemma", "gemma2", "gemma3", "gemma4", "gemma4-assistant",
		"phi2", "phi3",
		"stablelm", "gptneox", "gpt2", "starcoder2",
		"falcon", "mpt", "olmo2", "gptoss":
		return true
	}
	return false
}

func unpermuteRotaryTensor(name string, src []float32, heads, headDim, in int) ([]float32, error) {
	if heads <= 0 || headDim <= 0 || in <= 0 {
		return nil, fmt.Errorf("gguf: tensor %s has invalid rotary shape heads=%d head_dim=%d in=%d", name, heads, headDim, in)
	}
	if headDim%2 != 0 {
		return nil, fmt.Errorf("gguf: tensor %s head_dim %d is not even", name, headDim)
	}
	if heads > math.MaxInt/headDim || heads*headDim > math.MaxInt/in {
		return nil, fmt.Errorf("gguf: tensor %s rotary shape overflows int", name)
	}
	want := heads * headDim * in
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, rotary shape wants %d", name, len(src), want)
	}
	dst := make([]float32, len(src))
	half := headDim / 2
	for h := 0; h < heads; h++ {
		for j := 0; j < half; j++ {
			for p := 0; p < 2; p++ {
				for c := 0; c < in; c++ {
					dst[((h*2+p)*half+j)*in+c] = src[((h*half+j)*2+p)*in+c]
				}
			}
		}
	}
	return dst, nil
}

func unpermuteQwen35GatedQTensor(name string, src []float32, heads, headDim, in int) ([]float32, error) {
	if heads <= 0 || headDim <= 0 || in <= 0 {
		return nil, fmt.Errorf("gguf: tensor %s has invalid gated rotary shape heads=%d head_dim=%d in=%d", name, heads, headDim, in)
	}
	if headDim%2 != 0 {
		return nil, fmt.Errorf("gguf: tensor %s head_dim %d is not even", name, headDim)
	}
	if heads > math.MaxInt/(2*headDim) || heads*2*headDim > math.MaxInt/in {
		return nil, fmt.Errorf("gguf: tensor %s gated rotary shape overflows int", name)
	}
	want := heads * 2 * headDim * in
	if len(src) != want {
		return nil, fmt.Errorf("gguf: tensor %s has %d values, gated rotary shape wants %d", name, len(src), want)
	}
	dst := make([]float32, len(src))
	half := headDim / 2
	for h := 0; h < heads; h++ {
		srcHead := h * 2 * headDim
		dstHead := h * 2 * headDim
		for j := 0; j < half; j++ {
			for p := 0; p < 2; p++ {
				for c := 0; c < in; c++ {
					dst[(dstHead+p*half+j)*in+c] = src[(srcHead+j*2+p)*in+c]
				}
			}
		}
		copy(dst[(dstHead+headDim)*in:(dstHead+2*headDim)*in], src[(srcHead+headDim)*in:(srcHead+2*headDim)*in])
	}
	return dst, nil
}

// CanonicalTensorName maps a GGUF tensor name to fak's canonical HF-Llama name with
// the Llama-family norm convention (the historical, arch-blind behavior). Loaders that
// know the file architecture call CanonicalTensorNameArch instead, so a family whose
// norm tensors carry different roles (Gemma's sandwich norm) maps correctly.
func CanonicalTensorName(name string) (string, bool) {
	return CanonicalTensorNameArch(name, "")
}

// archIsGemma reports whether arch is a Gemma family whose GGUF carries sandwich-norm
// tensors (a distinct pre-feedforward norm plus post-attention and post-feedforward
// norms) rather than the single Llama post-attention norm. For these, blk.*.ffn_norm is
// the PRE-feedforward norm — not the post-attention norm it is for Llama — so the
// canonical mapping must branch on the family or two tensors collide on one name. Gemma
// v1 is intentionally excluded: it has no post-feedforward norm and its ffn_norm is the
// pre-MLP norm in the Llama role, so it keeps the default mapping.
func archIsGemma(arch string) bool {
	switch arch {
	case "gemma2", "gemma3", "gemma4", "gemma4-assistant":
		return true
	}
	return false
}

func archIsGemma4(arch string) bool {
	return arch == "gemma4" || arch == "gemma4-assistant"
}

// CanonicalTensorNameArch maps a GGUF tensor name to fak's canonical HF name, honoring
// arch-specific tensor roles. arch=="" preserves the Llama-family mapping.
func CanonicalTensorNameArch(name, arch string) (string, bool) {
	switch name {
	case "token_embd.weight":
		return "model.embed_tokens.weight", true
	case "output_norm.weight":
		return "model.norm.weight", true
	case "output.weight":
		return "lm_head.weight", true
	case "rope_freqs.weight":
		// Gemma4's global (full-attention) layers carry a single shared rope frequency-
		// factor vector (proportional/NTK rope). It is a model-global tensor, read by the
		// forward when rotating the long-context layers.
		return "model.rope_freqs.weight", true
	}
	if !strings.HasPrefix(name, "blk.") {
		return "", false
	}
	rest := strings.TrimPrefix(name, "blk.")
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return "", false
	}
	layer := rest[:dot]
	if _, err := strconv.Atoi(layer); err != nil {
		return "", false
	}
	suffix := rest[dot+1:]
	// glm_moe_dsa (GLM-5.2: MLA attention + DeepSeek-style MoE + a learned DSA indexer)
	// carries per-layer tensors no Llama/Gemma family has — the MLA latent projections,
	// the router + its score-correction bias, the shared experts, and the DSA indexer.
	// Map those FIRST; anything not GLM-specific (attn_norm, ffn_norm, attn_output, and
	// the leading-dense layers' ffn_gate/up/down) falls through to the shared base map
	// below. The batched ROUTED experts (ffn_*_exps, one [E,…] blob per layer) are
	// deliberately NOT mapped here — a single GGUF name cannot become E per-expert
	// canonical tensors, so they stay an explicit "no canonical mapping" until the
	// loader-side expert splitter lands (then a glm_moe_dsa GGUF loads end to end).
	if arch == "glm_moe_dsa" {
		if mapped, ok := glmMoeDsaCanonicalSuffix(suffix); ok {
			return "model.layers." + layer + "." + mapped, true
		}
	}
	// Gemma sandwich norm: ffn_norm is the PRE-feedforward norm and post_ffw_norm the
	// POST-feedforward norm, distinct from the post-attention norm. The Llama default
	// keeps ffn_norm == post_attention_layernorm (the single pre-MLP norm).
	ffnNormCanon := "post_attention_layernorm.weight"
	if archIsGemma(arch) {
		ffnNormCanon = "pre_feedforward_layernorm.weight"
	}
	mapped, ok := map[string]string{
		"attn_norm.weight":           "input_layernorm.weight",
		"ffn_norm.weight":            ffnNormCanon,
		"post_attention_norm.weight": "post_attention_layernorm.weight",
		"post_ffw_norm.weight":       "post_feedforward_layernorm.weight",
		"layer_output_scale.weight":  "layer_output_scale.weight",
		"attn_q.weight":              "self_attn.q_proj.weight",
		"attn_k.weight":              "self_attn.k_proj.weight",
		"attn_v.weight":              "self_attn.v_proj.weight",
		"attn_qkv.weight":            "self_attn.qkv_proj.weight",
		"attn_gate.weight":           "self_attn.q_gate_proj.weight",
		"attn_output.weight":         "self_attn.o_proj.weight",
		"attn_q.bias":                "self_attn.q_proj.bias",
		"attn_k.bias":                "self_attn.k_proj.bias",
		"attn_v.bias":                "self_attn.v_proj.bias",
		"attn_q_norm.weight":         "self_attn.q_norm.weight",
		"attn_k_norm.weight":         "self_attn.k_norm.weight",
		"ffn_gate.weight":            "mlp.gate_proj.weight",
		"ffn_up.weight":              "mlp.up_proj.weight",
		"ffn_down.weight":            "mlp.down_proj.weight",
		"ssm_a":                      "linear_attn.A_log",
		"ssm_alpha.weight":           "linear_attn.in_proj_a.weight",
		"ssm_beta.weight":            "linear_attn.in_proj_b.weight",
		"ssm_conv1d.weight":          "linear_attn.conv1d.weight",
		"ssm_dt.bias":                "linear_attn.dt_bias",
		"ssm_norm.weight":            "linear_attn.norm.weight",
		"ssm_out.weight":             "linear_attn.out_proj.weight",
	}[suffix]
	if !ok {
		return "", false
	}
	return "model.layers." + layer + "." + mapped, true
}

func modelShapeFromGGUFDims(name string, dims []uint64) ([]int, error) {
	if len(dims) == 0 {
		return nil, fmt.Errorf("gguf: tensor %s has no dimensions", name)
	}
	shape := make([]int, len(dims))
	for i := range dims {
		d := dims[len(dims)-1-i]
		if d == 0 || d > uint64(math.MaxInt) {
			return nil, fmt.Errorf("gguf: tensor %s dimension %d overflows int", name, d)
		}
		shape[i] = int(d)
	}
	return shape, nil
}
