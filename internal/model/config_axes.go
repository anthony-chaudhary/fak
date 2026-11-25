package model

import (
	"fmt"
	"math"
	"strings"
)

func (c *Config) deriveConfigAxes(h configJSONHints) error {
	if err := c.deriveHeadAndLoopAxes(h); err != nil {
		return err
	}
	if c.HiddenActivation == "" {
		c.HiddenActivation = h.HiddenActivation
	}
	if c.HiddenAct == "" {
		c.HiddenAct = h.HiddenAct
	}
	family := c.archFamilyKey()

	c.deriveKVHeadAndRopeAxes(h, family)
	c.deriveBiasNormActivationAxes(h, family)
	if err := c.deriveBlockTopologyAxis(h, family); err != nil {
		return err
	}
	c.deriveGemmaAndQKNormAxes(h, family)
	c.deriveMoEAxes(h, family)
	c.deriveAltSpellingAxes(h)
	c.deriveLayerNormFamilyAxes(h, family)
	return nil
}

// deriveAltSpellingAxes folds alternative key spellings onto the canonical MoE
// axes when the canonical key is absent. Kimi K3.1's text config names the MoE
// knobs with its own spellings (num_experts_per_token, num_shared_experts,
// num_expert_group, moe_renormalize, moe_router_activation_func) and puts the
// image placeholder under media_placeholder_token_id; the expert count rides
// the existing NumExpertsAlt hint in deriveMoEAxes. Every family that ships a
// canonical key sets the canonical field directly, so the canonical-absent
// gates keep DeepSeek / Mixtral / Qwen-MoE decodes byte-identical.
func (c *Config) deriveAltSpellingAxes(h configJSONHints) {
	if c.NumExpertsPerTok == 0 && h.NumExpertsPerTokAlt != nil {
		c.NumExpertsPerTok = *h.NumExpertsPerTokAlt
	}
	if c.NSharedExperts == 0 && h.NSharedExpertsAlt != nil {
		c.NSharedExperts = *h.NSharedExpertsAlt
	}
	if c.NGroup == 0 && h.NGroupAlt != nil {
		c.NGroup = *h.NGroupAlt
	}
	// norm_topk_prob presence is tracked by the hint: an explicit canonical
	// false must not be overwritten by the alt spelling, and an absent canonical
	// key leaves the zero value free to take the alt.
	if h.NormTopKProb == nil && h.MoERenormalizeAlt != nil {
		c.NormTopKProb = *h.MoERenormalizeAlt
	}
	if c.ScoringFunc == "" && h.MoERouterActivationAlt != "" {
		c.ScoringFunc = h.MoERouterActivationAlt
	}
	if c.ImageTokenID == 0 && h.MediaPlaceholderTokenIDAlt != nil {
		c.ImageTokenID = *h.MediaPlaceholderTokenIDAlt
	}
}

// deriveHeadAndLoopAxes resolves HeadDim and NumLoops, then refuses an
// unsupported Nanbeige variant.
func (c *Config) deriveHeadAndLoopAxes(h configJSONHints) error {
	// MLA (DeepSeek-V2/V3, GLM-5.2) heads are NOT hidden_size/num_heads wide. HF's
	// DeepseekV2Config declares no head_dim at all; DeepseekV2Attention builds
	// q_head_dim = qk_nope_head_dim + qk_rope_head_dim and cuts every per-head width from
	// that. On DeepSeek-V3 the two numbers are unrelated: hidden_size/num_heads is
	// 7168/128 = 56 while q_head_dim is 128+64 = 192. Letting the generic fallback below
	// win would size the rotary table at HeadDim/2 = 28 frequencies while the MLA rotary
	// (glmDsaApplyInterleavedRoPE) indexes qk_rope_head_dim/2 = 32 of them — a slice-bounds
	// panic on the first token. This branch fires ONLY when head_dim is absent and both MLA
	// widths are present, so a config that states head_dim (fak's own exports, and every
	// GGUF, which carries it as attention.key_length) is untouched, and a non-MLA config
	// never reaches it. Witness: TestDeepSeekMLADerivedHeadDimCoversRopeSlice.
	if c.HeadDim == 0 && c.QKNopeHeadDim > 0 && c.QKRopeHeadDim > 0 {
		c.HeadDim = c.QKNopeHeadDim + c.QKRopeHeadDim
	}
	// Nanbeige4.2 explicitly fixes head_dim = 128 and must NOT derive hidden_size/num_heads (e.g. 3072/48=64).
	if !c.isNanbeige() && c.HeadDim == 0 && c.HiddenSize != 0 && c.NumHeads != 0 {
		c.HeadDim = c.HiddenSize / c.NumHeads
	}
	if c.NumLoops == 0 && h.LoopsAlt != nil {
		c.NumLoops = *h.LoopsAlt
	}
	if c.NumLoops == 0 && h.RecurrentLoopsAlt != nil {
		c.NumLoops = *h.RecurrentLoopsAlt
	}
	if c.isNanbeige() {
		return c.validateNanbeigeVariant(h)
	}
	return nil
}

func (c *Config) validateNanbeigeVariant(h configJSONHints) error {
	if c.NumLoops == 0 {
		c.NumLoops = 2
	}
	sharedCache := c.SharedCache || (h.SharedCache != nil && *h.SharedCache)
	if sharedCache {
		return &UnsupportedNanbeigeVariantError{
			Reason:      "shared_cache=true is not supported",
			NumLayers:   c.NumLayers,
			NumLoops:    c.NumLoops,
			HeadDim:     c.HeadDim,
			SharedCache: true,
		}
	}
	if h.LoopShareKV != nil && *h.LoopShareKV {
		return &UnsupportedNanbeigeVariantError{
			Reason:    "loop_share_kv=true is not supported",
			NumLayers: c.NumLayers,
			NumLoops:  c.NumLoops,
			HeadDim:   c.HeadDim,
		}
	}
	if h.SkipLoopFinalNorm != nil && *h.SkipLoopFinalNorm {
		return &UnsupportedNanbeigeVariantError{
			Reason:    "skip_loop_final_norm=true is not supported",
			NumLayers: c.NumLayers,
			NumLoops:  c.NumLoops,
			HeadDim:   c.HeadDim,
		}
	}
	if c.NumLayers != 22 {
		return &UnsupportedNanbeigeVariantError{
			Reason:      fmt.Sprintf("num_hidden_layers must be 22, got %d", c.NumLayers),
			NumLayers:   c.NumLayers,
			NumLoops:    c.NumLoops,
			HeadDim:     c.HeadDim,
			SharedCache: sharedCache,
		}
	}
	if c.NumLoops != 2 {
		return &UnsupportedNanbeigeVariantError{
			Reason:      fmt.Sprintf("num_loops must be 2, got %d", c.NumLoops),
			NumLayers:   c.NumLayers,
			NumLoops:    c.NumLoops,
			HeadDim:     c.HeadDim,
			SharedCache: sharedCache,
		}
	}
	if c.HeadDim != 128 {
		return &UnsupportedNanbeigeVariantError{
			Reason:      fmt.Sprintf("head_dim must be 128, got %d", c.HeadDim),
			NumLayers:   c.NumLayers,
			NumLoops:    c.NumLoops,
			HeadDim:     c.HeadDim,
			SharedCache: sharedCache,
		}
	}
	return nil
}

func (c *Config) deriveKVHeadAndRopeAxes(h configJSONHints, family string) {
	if c.RMSNormEps == 0 && h.LayerNormEps != nil {
		c.RMSNormEps = *h.LayerNormEps
	}
	if c.NumKVHeads == 0 {
		switch {
		case h.MultiQuery != nil && *h.MultiQuery:
			c.NumKVHeads = 1
		case h.NumKVHeadsAlt != nil && *h.NumKVHeadsAlt > 0 && *h.NumKVHeadsAlt <= c.NumHeads:
			c.NumKVHeads = *h.NumKVHeadsAlt
		}
	}
	if c.NumKVHeads == 0 {
		c.NumKVHeads = c.NumHeads
	}
	if c.IntermediateSize == 0 && strings.Contains(family, "falcon") && c.HiddenSize > 0 {
		c.IntermediateSize = 4 * c.HiddenSize
	}
	c.promoteLongRope("linear")
	if c.promoteLongRope("llama3") {
		c.RopeLowFreqFactor = c.LongRope.LowFreqFactor
		c.RopeHighFreqFactor = c.LongRope.HighFreqFactor
	}
	// yarn arrives in EITHER HF shape. The newer checkpoints nest it under
	// rope_parameters (handled just below); the long-context Qwen line — including the
	// Qwen3.6 hybrid backbone Bonsai is built on — still ships it under the CLASSIC
	// rope_scaling key, which decodes into LongRope. Only the llama3 kind was promoted
	// off LongRope above, so a classic-key yarn config left RopeScaling=="" and
	// applyRopeScaling returned a BARE inv_freq: at a 262K context the rotation was
	// unscaled and long-offset positions diverged silently. Promote it here, mirroring
	// the llama3 promotion, so both HF shapes reach the same yarn path (#4874).
	if c.promoteLongRope("yarn") && c.RopeTheta == 0 {
		c.RopeTheta = c.LongRope.RopeTheta
	}
	if c.RopeScaling == "" {
		if rp, ok := c.RopeParameters["default"]; ok && (rp.kind() == "yarn" || rp.kind() == "linear") {
			c.RopeScaling = rp.kind()
			c.RopeFactor = rp.Factor
			c.RopeOrigContext = rp.OriginalMaxPositionEmbeddings
			if c.RopeTheta == 0 {
				c.RopeTheta = rp.RopeTheta
			}
		}
	}
	if c.PartialRotaryFactor == 0 {
		if rp, ok := c.RopeParameters["default"]; ok && rp.PartialRotaryFactor != 0 {
			c.PartialRotaryFactor = rp.PartialRotaryFactor
		}
	}
	if c.RopeTheta == 0 {
		if rp, ok := c.RopeParameters["default"]; ok && rp.RopeTheta != 0 {
			c.RopeTheta = rp.RopeTheta
		}
	}
}

func (c *Config) deriveBiasNormActivationAxes(h configJSONHints, family string) {
	if h.AttentionBias == nil && strings.Contains(family, "qwen2") {
		// Qwen2/Qwen2.5 checkpoints historically omitted attention_bias while still
		// carrying q/k/v projection bias tensors. Newer Qwen3.5/Qwen3.6 hybrid configs
		// explicitly set attention_bias=false, so only apply this legacy default when
		// the key is absent.
		c.AttentionBias = true
	}
	if c.IsQwen35Hybrid() && h.NormGain1p == nil {
		// Qwen3.5 / Qwen3-Next ordinary RMSNorms are the (1+weight) "+1 gain" form (weights
		// init to zero); the gated DeltaNet norm (plain weight) is handled in linearAttnSeq.
		c.NormGain1p = true
	}
	if h.UseQKNorm != nil && h.QKNorm == nil {
		c.QKNorm = *h.UseQKNorm
	}
	if act := strings.ToLower(c.activationName()); act == "gelu_pytorch_tanh" && h.ActGeluTanh == nil {
		c.ActGeluTanh = true
	} else if act == "gelu" && h.ActGeluErf == nil {
		c.ActGeluErf = true
	}
}

func (c *Config) deriveBlockTopologyAxis(h configJSONHints, family string) error {
	if h.BlockTopology != "" {
		topo, ok := parseBlockTopology(h.BlockTopology)
		if !ok {
			return fmt.Errorf("block_topology: unknown %q", h.BlockTopology)
		}
		c.BlockTopology = topo
		return nil
	}
	switch {
	case strings.Contains(family, "gemma2") || strings.Contains(family, "gemma3"):
		c.BlockTopology = SandwichNorm
	case strings.Contains(family, "olmo2"):
		c.BlockTopology = PostNorm
	case strings.Contains(family, "gptneox") || strings.Contains(family, "cohere") || (strings.Contains(family, "falcon") && c.ParallelAttention):
		c.BlockTopology = ParallelResidual
	}
	return nil
}

func (c *Config) deriveGemmaAndQKNormAxes(h configJSONHints, family string) {
	if strings.Contains(family, "gemma") {
		if h.NormGain1p == nil {
			c.NormGain1p = true
		}
		if h.ActGeluTanh == nil {
			c.ActGeluTanh = true
		}
		if h.EmbedScale == nil && c.EmbedScale == 0 && c.HiddenSize > 0 {
			c.EmbedScale = math.Sqrt(float64(c.HiddenSize))
		}
	}
	c.deriveLayerAttentionAxes(family, h.SlidingWindow, h.UseSlidingWindow)
	if strings.Contains(family, "olmo2") || strings.Contains(family, "cohere2") || strings.Contains(family, "qwen3") || strings.Contains(family, "gemma3") || strings.Contains(family, "minimax") {
		if h.QKNorm == nil && h.UseQKNorm == nil {
			// MiniMax-M3 layers carry per-head q_norm/k_norm; the other families
			// above are the existing qk-norm checkpoints. A non-qk-norm MiniMax
			// export can still pin qk_norm=false explicitly to opt out.
			c.QKNorm = true
		}
	}
}

func (c *Config) deriveMoEAxes(h configJSONHints, family string) {
	// Ornith / Qwen3.5-MoE serialize the expert count under "num_experts" and the
	// shared-expert width under "shared_expert_intermediate_size". Fold the alts into
	// the canonical fields only when the canonical key was absent (the alt pointer is
	// nil unless the JSON literally carried that key), so without this an Ornith MoE
	// config resolves NumExperts==0, IsMoE() returns false, and the 35B/397B silently
	// load as a dense model. num_local_experts / shared_intermediate_size families are
	// unaffected (their alt pointers stay nil).
	if c.NumExperts == 0 && h.NumExpertsAlt != nil && *h.NumExpertsAlt > 0 {
		c.NumExperts = *h.NumExpertsAlt
	}
	if c.NumExperts == 0 && h.NumRoutedExpertsAlt != nil && *h.NumRoutedExpertsAlt > 0 {
		c.NumExperts = *h.NumRoutedExpertsAlt
	}
	if c.SharedIntermediateSize == 0 && h.SharedExpertIntermediateAlt != nil && *h.SharedExpertIntermediateAlt > 0 {
		c.SharedIntermediateSize = *h.SharedExpertIntermediateAlt
	}
	if strings.Contains(family, "mixtral") && c.NumExperts > 0 && h.NormTopKProb == nil {
		// HF Mixtral does not serialize norm_topk_prob, but MixtralSparseMoeBlock
		// normalizes the selected top-k router weights before the expert weighted sum.
		c.NormTopKProb = true
	}
	if (c.ModelType == "qwen3_5_moe" || c.ModelType == "qwen3_5_moe_text") && c.NumExperts > 0 && h.NormTopKProb == nil {
		// Qwen3.5-MoE always renormalizes selected router weights; published Ornith configs omit this field.
		c.NormTopKProb = true
	}
	if strings.Contains(family, "minimax") && h.NormGain1p == nil {
		// MiniMax-M3 uses Gemma-style RMSNorm: every norm — the input/post
		// layernorms, the per-head qk-norm, AND the lightning-indexer q/k
		// norms — scales by (1+weight) (use_gemma_norm in the HF config; the
		// modeling MiniMaxM3VLRMSNorm computes `output * (1.0 + weight)`).
		// Default it on for the family unless an export pins it off explicitly.
		c.NormGain1p = true
	}
	if strings.Contains(family, "minimax") && c.NumExperts > 0 {
		// MiniMax-M3's MoE router (MiniMaxM3VLTopKRouter) ALWAYS renormalizes the
		// top-k sigmoid weights by their sum (`top_k_weights /= sum`); the config
		// carries no norm_topk_prob knob, so default it on for the family.
		if !c.NormTopKProb {
			c.NormTopKProb = true
		}
		// Every M3 MoE layer has exactly one always-on shared expert
		// (MiniMaxM3VLSparseMoeBlock.shared_experts); the config records its width
		// as shared_intermediate_size but no n_shared_experts count. Default to 1.
		if c.NSharedExperts == 0 {
			c.NSharedExperts = 1
		}
	}
}

func (c *Config) deriveLayerNormFamilyAxes(h configJSONHints, family string) {
	if strings.Contains(family, "cohere") && h.LogitScale == nil && c.LogitScale == 0 {
		c.LogitScale = 0.0625
	}
	if strings.Contains(family, "cohere") && h.LayerNorm == nil {
		c.LayerNorm = true
	}
	if strings.Contains(family, "cohere") {
		// Cohere's optional qk-norm is a CohereLayerNorm with a TUPLE hidden size
		// (num_heads, head_dim): one mean-subtracting reduction per head over head_dim,
		// scaled by that head's own weight row. Same parameter LENGTH as OLMo2's flat
		// projection-width q_norm, so the shape cannot be inferred at the call site.
		c.QKNormPerHeadWeight = true
	}
	if strings.Contains(family, "gptneox") {
		if h.LayerNorm == nil {
			c.LayerNorm = true
		}
		if h.DenseMLP == nil {
			c.DenseMLP = true
		}
	}
	if strings.Contains(family, "falcon") {
		if h.LayerNorm == nil {
			c.LayerNorm = true
		}
		if h.DenseMLP == nil {
			c.DenseMLP = true
		}
	}
	if strings.Contains(family, "mpt") {
		if h.LayerNorm == nil {
			c.LayerNorm = true
		}
		if h.DenseMLP == nil {
			c.DenseMLP = true
		}
		if h.ActGeluErf == nil {
			c.ActGeluErf = true
		}
		if h.Alibi == nil {
			c.Alibi = true
		}
	}
	if strings.Contains(family, "stablelm") && h.LayerNorm == nil {
		c.LayerNorm = true
	}
}

// gemmaSlidingWindowPattern is the published local/global cadence PERIOD for a Gemma
// family — layer l is sliding unless (l+1)%period == 0 — or 0 for a family that has no
// such cadence.
//
// Gemma3 publishes the cadence in config.json (layer_types, or sliding_window_pattern
// defaulting to 6: five local layers then one global). Gemma2 publishes NEITHER: its
// config.json carries only `sliding_window`, and the alternation lives in the modeling
// code as `is_sliding = not bool(layer_idx % 2)` (Gemma2DecoderLayer.__init__) — even
// layers windowed, ODD layers full causal — which newer transformers lowers to exactly
// the same rule at period 2 (`"sliding_attention" if bool((i+1) % 2) else
// "full_attention"`). Before this defaulted, a Gemma2 checkpoint left LayerTypes empty,
// so the Window loop below stamped `sliding_window` onto EVERY layer and silently
// clipped the context of the half that must attend the whole prefix. That is invisible
// until the prompt outgrows the window, which is why only a numeric witness caught it:
// TestGemmaCPUNumericOracle/gemma2 (family_gemma_cpu_oracle_test.go).
//
// Only the window cadence changes for Gemma2. deriveRopeThetaPerLayer now also runs for
// it, but Gemma2 sets no rope_local_base_freq, so the sliding layers get a 0 entry and
// ropeThetaForLayer falls back to the single published rope_theta — the same base on
// every layer, as HF does, and hasLayerSpecificRopeTheta stays false.
func gemmaSlidingWindowPattern(family string) int {
	switch {
	case strings.Contains(family, "gemma3"):
		return 6
	case strings.Contains(family, "gemma2"):
		return 2
	}
	return 0
}

// familySlidingWindowPattern extends the Gemma cadence default above to every family
// whose HF config class SYNTHESIZES layer_types when config.json omits it. Gemma is not
// the only one: gpt-oss (GptOssForCausalLM) alternates with PERIOD 2, because
// GptOssConfig.__init__ builds
// `["sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(n)]` —
// even layers windowed, odd layers full, which is exactly the `(l+1)%pattern == 0 =>
// full_attention` rule deriveLayerAttentionAxes applies at period 2.
//
// Without this, a gpt-oss config that carries `sliding_window` but no `layer_types` left
// LayerTypes empty, so the Window loop stamped the sliding window onto EVERY layer and
// silently clipped the context of the half of the stack that must attend the whole
// prefix — the identical defect Gemma2 had, and identically invisible until the prompt
// outgrows the window. The published 20B/120B config.json does ship layer_types, so the
// checkpoint path was unaffected and only configs that omit it (GGUF-derived, minimal, or
// re-exported) were wrong. Witness: TestGPTOSSCPUNumericOracle/derived_cadence in
// family_gptoss_cpu_oracle_test.go.
func familySlidingWindowPattern(family string) int {
	if pattern := gemmaSlidingWindowPattern(family); pattern > 0 {
		return pattern
	}
	if strings.Contains(family, "gptoss") {
		return 2
	}
	return 0
}

func (c *Config) deriveLayerAttentionAxes(family string, slidingWindow *int, useSlidingWindow *bool) {
	if c.NumLayers <= 0 {
		return
	}
	if defaultPattern := familySlidingWindowPattern(family); defaultPattern > 0 && len(c.LayerTypes) == 0 {
		pattern := c.SlidingWindowPattern
		if pattern == 0 {
			pattern = defaultPattern
		}
		c.LayerTypes = make([]string, c.NumLayers)
		for l := range c.LayerTypes {
			if pattern > 0 && (l+1)%pattern == 0 {
				c.LayerTypes[l] = "full_attention"
			} else {
				c.LayerTypes[l] = "sliding_attention"
			}
		}
	}
	if len(c.Window) == 0 && slidingWindow != nil && *slidingWindow > 0 && (useSlidingWindow == nil || *useSlidingWindow) {
		c.Window = make([]int, c.NumLayers)
		for l := range c.Window {
			if c.layerType(l) == "full_attention" {
				c.Window[l] = -1
			} else {
				c.Window[l] = *slidingWindow
			}
		}
	}
	if len(c.RopeThetaPerLayer) == 0 {
		c.deriveRopeThetaPerLayer(family)
	}
}

func (c *Config) deriveRopeThetaPerLayer(family string) {
	if c.NumLayers <= 0 || len(c.LayerTypes) == 0 {
		return
	}
	fullTheta := c.RopeTheta
	localTheta := c.RopeLocalBaseFreq
	if rp, ok := c.RopeParameters["full_attention"]; ok && rp.RopeTheta != 0 {
		fullTheta = rp.RopeTheta
	}
	if rp, ok := c.RopeParameters["sliding_attention"]; ok && rp.RopeTheta != 0 {
		localTheta = rp.RopeTheta
	}
	if fullTheta == 0 && strings.Contains(family, "gemma3") {
		fullTheta = 1000000
	}
	if localTheta == 0 && strings.Contains(family, "gemma3") {
		localTheta = 10000
	}
	if fullTheta == 0 && localTheta == 0 {
		return
	}
	c.RopeThetaPerLayer = make([]float64, c.NumLayers)
	for l := range c.RopeThetaPerLayer {
		switch c.layerType(l) {
		case "sliding_attention":
			c.RopeThetaPerLayer[l] = localTheta
		case "full_attention":
			c.RopeThetaPerLayer[l] = fullTheta
		}
	}
}
