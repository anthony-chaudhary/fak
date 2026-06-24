package model

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// Config mirrors the subset of the HF model config the forward pass needs. It is
// read verbatim from the exported config.json — never hardcoded — so swapping the
// target model is a re-export, not a code edit.
//
// Stage-2 of the model-arch seam (MODEL-ARCH-SEAM.md §6, §2b class-1) extends this
// struct with the MECHANICAL architecture axes — scalar/elementwise edits that never
// change WHICH reductions happen or their order. EVERY new field defaults to the
// Llama behavior (off / identity), so an existing Llama checkpoint takes the identical
// instruction stream and R2/R14 stay max|Δ|=0 by construction (the Llama no-op gate,
// TestArchLlamaNoOp). The fields are grouped: the Llama-13 base first, then the
// additive Stage-2 axes.
type Config struct {
	HiddenSize        int               `json:"hidden_size"`
	NumLayers         int               `json:"num_hidden_layers"`
	NumHeads          int               `json:"num_attention_heads"`
	NumKVHeads        int               `json:"num_key_value_heads"`
	HeadDim           int               `json:"head_dim"`
	IntermediateSize  int               `json:"intermediate_size"`
	VocabSize         int               `json:"vocab_size"`
	RMSNormEps        float64           `json:"rms_norm_eps"`
	RopeTheta         float64           `json:"rope_theta"`
	TieWordEmbeddings bool              `json:"tie_word_embeddings"`
	AttentionBias     bool              `json:"attention_bias"`
	ModelType         string            `json:"model_type"`
	Architectures     []string          `json:"architectures,omitempty"`
	LayerTypes        []string          `json:"layer_types,omitempty"`
	HiddenAct         string            `json:"hidden_act,omitempty"`
	HiddenActivation  string            `json:"hidden_activation,omitempty"`
	TensorAliases     map[string]string `json:"tensor_aliases,omitempty"`

	// EOSTokenID is the legacy scalar EOS id. EOSTokenIDs is the Llama-3.x form, where
	// config.json emits eos_token_id as a LIST (e.g. [128001,128008,128009]); the custom
	// UnmarshalJSON below accepts scalar-or-list and populates both, so an int loader
	// and a set-membership stop check both work. When EOSTokenIDs is non-empty it is the
	// authoritative set; EOSTokenID keeps the first id for back-compat callers.
	EOSTokenID  int   `json:"-"`
	EOSTokenIDs []int `json:"-"`

	// ---- Stage-2 mechanical arch axes (all default = Llama no-op) -------------------

	// RopeScaling selects the inv_freq rescale applied in invFreq(). "" / "none" (default)
	// returns the bare Llama inv_freq bit-for-bit; "llama3" applies the piecewise
	// low/high-frequency-wavelength rescale that Llama-3.1/3.2/3.3 ship. The params below
	// are only read when RopeScaling=="llama3".
	RopeScaling        string  `json:"rope_scaling_type"`
	RopeFactor         float64 `json:"rope_scaling_factor"`
	RopeLowFreqFactor  float64 `json:"rope_scaling_low_freq_factor"`
	RopeHighFreqFactor float64 `json:"rope_scaling_high_freq_factor"`
	RopeOrigContext    int     `json:"rope_scaling_original_max_position_embeddings"`

	// QKNorm gates a per-head RMSNorm on q and k AFTER projection, BEFORE RoPE (Qwen3 /
	// OLMo2 / Gemma3 / Cohere2). Off (default) = no-op. The per-head norm weights are the
	// tensors self_attn.{q,k}_norm.weight; QKNormEps defaults to RMSNormEps when zero.
	QKNorm    bool    `json:"qk_norm"`
	QKNormEps float64 `json:"qk_norm_eps"`

	// NormGain1p makes RMSNorm read (1+w) instead of w (Gemma's "+1" gain centering).
	// false (default) = plain Llama weight.
	NormGain1p bool `json:"norm_gain_1p"`

	// LayerNorm selects mean-subtracting LayerNorm instead of RMSNorm for decoder/final
	// normalization (Cohere). false (default) = RMSNorm.
	LayerNorm bool `json:"layer_norm,omitempty"`

	// ActGeluTanh selects the tanh-approx GELU activation in the SwiGLU MLP (Gemma's
	// GeGLU) instead of SiLU. false (default) = SiLU.
	ActGeluTanh bool `json:"act_gelu_tanh"`

	// ActGeluErf selects exact GELU (erf form) instead of SiLU. false (default) = SiLU.
	ActGeluErf bool `json:"act_gelu_erf,omitempty"`

	// AttnSoftcap / LogitSoftcap are Gemma2 tanh soft-caps. 0 (default) = off. A non-zero
	// cap c maps z -> c*tanh(z/c) (applied to attention scores pre-softmax, and to final
	// logits, respectively).
	AttnSoftcap  float64 `json:"attn_logit_softcapping"`
	LogitSoftcap float64 `json:"final_logit_softcapping"`

	// EmbedScale multiplies the embedding row at lookup (Gemma uses sqrt(hidden)). 0 or 1
	// (default) = no scaling.
	EmbedScale float64 `json:"embed_scale"`

	// LogitScale multiplies the final logits (Cohere uses 0.0625). 0 or 1 (default) = no
	// scaling.
	LogitScale float64 `json:"logit_scale"`

	// ParallelAttention carries Falcon's parallel attention+MLP block hint. It maps to
	// ParallelResidual when true; false/omitted leaves other families unchanged.
	ParallelAttention bool `json:"parallel_attn,omitempty"`

	// Alibi selects additive per-head attention score bias instead of RoPE (MPT).
	// AlibiBiasMax defaults to 8 when zero, matching HF MPT.
	Alibi        bool    `json:"alibi,omitempty"`
	AlibiBiasMax float64 `json:"alibi_bias_max,omitempty"`

	// QueryPreAttnScalar overrides the per-head attention scale denominator. When non-zero
	// the scale is 1/sqrt(QueryPreAttnScalar) (Gemma) instead of the default 1/sqrt(HeadDim).
	QueryPreAttnScalar int `json:"query_pre_attn_scalar"`

	// Window is the per-layer sliding-window attention (SWA) bound: layer l attends
	// only to the most recent Window[l] absolute positions (inclusive of the query),
	// i.e. a query at absolute position p sees keys whose position is >= p-Window[l]+1.
	// A value of -1 (and the empty/short-slice default) means FULL causal attention.
	Window []int `json:"sliding_window_per_layer,omitempty"`

	// SlidingWindowPattern is Gemma3's local/global attention cadence: layers whose
	// 1-based index is divisible by the pattern are full-attention, all others are
	// sliding-attention. Zero means no inferred cadence unless a family default supplies one.
	SlidingWindowPattern int `json:"sliding_window_pattern,omitempty"`

	// RopeThetaPerLayer overrides RopeTheta for a layer. Empty/zero entries fall back to
	// RopeTheta, preserving the Llama shared-theta path. Gemma3 uses this for local vs
	// global attention layers, whose RoPE bases differ.
	RopeThetaPerLayer []float64 `json:"rope_theta_per_layer,omitempty"`

	// PartialRotaryFactor rotates only the leading fraction of each attention head
	// (GPT-NeoX). 0 or 1 means full-head RoPE, matching the Llama default.
	PartialRotaryFactor float64 `json:"partial_rotary_factor,omitempty"`

	// MaxPositionEmbeddings is the model's full context window. Longrope uses this to
	// pin its short-vs-long factor selection for the whole session.
	MaxPositionEmbeddings int `json:"max_position_embeddings"`

	// LongRope carries the nested rope_scaling object used by Phi longrope checkpoints.
	// It intentionally does not reuse RopeScaling, which is the flat string field used by
	// the Llama-3 export path above.
	LongRope          *RopeScaling   `json:"rope_scaling"`
	RopeParameters    RopeParameters `json:"rope_parameters,omitempty"`
	RopeLocalBaseFreq float64        `json:"rope_local_base_freq,omitempty"`

	// ---- Gemma4 heterogeneous per-layer attention geometry --------------------------
	//
	// Gemma4 interleaves local (sliding) and global (full) attention layers with
	// DIFFERENT head_dim and kv-head counts per layer: local layers use a small head_dim
	// with several kv heads; global layers use a large head_dim with a single kv head
	// whose projection also serves as V (no separate v_proj tensor). These per-layer
	// slices override the scalar HeadDim/NumKVHeads inside the dedicated gemma4 forward;
	// empty (the default) preserves the uniform Llama geometry on every other path.
	HeadDimPerLayer    []int `json:"head_dim_per_layer,omitempty"`
	NumKVHeadsPerLayer []int `json:"num_kv_heads_per_layer,omitempty"`
	RopeDimPerLayer    []int `json:"rope_dim_per_layer,omitempty"`

	// SuppressTokens are vocab ids forced to -inf at the final-logit stage (Gemma 4
	// masks its image/audio placeholder tokens, a known checkpoint issue). Empty = no-op.
	SuppressTokens []int `json:"suppress_tokens,omitempty"`

	// MoE (Mixture-of-Experts) FFN axis. KV-orthogonal: these fields restructure
	// only the FFN sub-layer (router -> top-k experts -> weighted sum), never the
	// attention/KV path. Llama/dense default is NumExperts==0.
	NumExperts          int     `json:"num_local_experts"`
	NumExpertsPerTok    int     `json:"num_experts_per_tok"`
	NormTopKProb        bool    `json:"norm_topk_prob"`
	MoEIntermediateSize int     `json:"moe_intermediate_size,omitempty"`
	NSharedExperts      int     `json:"n_shared_experts,omitempty"`
	FirstKDenseReplace  int     `json:"first_k_dense_replace,omitempty"`
	MoELayerFreq        int     `json:"moe_layer_freq,omitempty"`
	NGroup              int     `json:"n_group,omitempty"`
	TopKGroup           int     `json:"topk_group,omitempty"`
	RoutedScalingFactor float64 `json:"routed_scaling_factor,omitempty"`

	// Qwen3.5 / Qwen3-Next hybrid Gated-DeltaNet linear-attention axis. When LayerTypes
	// marks a layer "linear_attention", that layer is a recurrent state-space token mixer
	// (qwen35.go) instead of attention; "full_attention" layers use the standard GQA path
	// with the AttnOutputGate sigmoid gate. All zero/false for non-hybrid models.
	LinearConvKernelDim   int  `json:"linear_conv_kernel_dim,omitempty"`
	LinearKeyHeadDim      int  `json:"linear_key_head_dim,omitempty"`
	LinearNumKeyHeads     int  `json:"linear_num_key_heads,omitempty"`
	LinearValueHeadDim    int  `json:"linear_value_head_dim,omitempty"`
	LinearNumValueHeads   int  `json:"linear_num_value_heads,omitempty"`
	AttnOutputGate        bool `json:"attn_output_gate,omitempty"`
	FullAttentionInterval int  `json:"full_attention_interval,omitempty"`

	// DeepSeek V2/V3 MLA metadata. These fields are exported and audited so a real
	// DeepSeek artifact is not mistaken for the standard q/k/v attention path. The
	// current runtime still requires explicit MLA projection wiring before these become
	// executable support.
	QLoraRank     int      `json:"q_lora_rank,omitempty"`
	KVLoraRank    int      `json:"kv_lora_rank,omitempty"`
	QKNopeHeadDim int      `json:"qk_nope_head_dim,omitempty"`
	QKRopeHeadDim int      `json:"qk_rope_head_dim,omitempty"`
	VHeadDim      int      `json:"v_head_dim,omitempty"`
	IndexNHeads   int      `json:"index_n_heads,omitempty"`
	IndexHeadDim  int      `json:"index_head_dim,omitempty"`
	IndexTopK     int      `json:"index_topk,omitempty"`
	IndexerTypes  []string `json:"indexer_types,omitempty"`

	// MiniMax-M3 "MiniMax Sparse Attention" (MSA) metadata. MSA keeps a GQA backbone
	// on the real uncompressed K/V (NOT MLA latent compression), but a lightning
	// indexer scores every key, max-pools those scores into blocks of IndexBlockSize
	// keys, and for each query attends only to the union of the top-IndexTopKBlocks
	// scored blocks and the always-on IndexLocalBlocks most-recent blocks (block-level
	// causality). A "minimax_m3_sparse" entry in LayerTypes marks an MSA layer;
	// "full_attention" layers run dense causal GQA. All zero = no MSA (Llama default).
	// These mirror HF's index_block_size / index_topk_blocks / index_local_blocks.
	IndexBlockSize   int `json:"index_block_size,omitempty"`
	IndexTopKBlocks  int `json:"index_topk_blocks,omitempty"`
	IndexLocalBlocks int `json:"index_local_blocks,omitempty"`

	// MiniMax-M3 SwiGLU-OAI gated expert activation. The OAI gate clamps the gate to
	// SwigluLimit and the up branch to ±SwigluLimit, then out = (up+1)*(gate*sigmoid(
	// gate*SwigluAlpha)). Zero SwigluLimit means "no clamp"; SwigluAlpha falls back to
	// the gpt-oss/OAI default 1.702 when zero. Both zero (default) = the plain SiLU
	// SwiGLU every other family uses. SharedIntermediateSize is the always-on shared
	// expert's FFN width (defaults to IntermediateSize when zero).
	SwigluAlpha            float64 `json:"swiglu_alpha,omitempty"`
	SwigluLimit            float64 `json:"swiglu_limit,omitempty"`
	SharedIntermediateSize int     `json:"shared_intermediate_size,omitempty"`

	// DenseIntermediateSize is the FFN width of MiniMax-M3's first-k DENSE layers
	// (the real checkpoint runs dense OAI MLPs on the layers whose moe_layer_freq
	// entry is 0 before switching to routed MoE). The dense layer's mlp.{gate,up,
	// down}_proj live at this width, NOT IntermediateSize (the routed-expert width).
	// Defaults to IntermediateSize when zero, so a fully-MoE config is unaffected.
	DenseIntermediateSize int `json:"dense_intermediate_size,omitempty"`

	// DenseMLP selects GPT-NeoX's dense activation MLP:
	// hidden -> dense_h_to_4h -> GELU -> dense_4h_to_h. False keeps the Llama SwiGLU.
	DenseMLP bool `json:"dense_mlp,omitempty"`

	// BlockTopology selects the decoder block's norm-placement / residual wiring
	// (arch.go). The zero value is PreNorm (Llama), so every existing export —
	// which never sets this field — keeps the current byte-identical path. Derived
	// from arch at load (e.g. OLMo2 -> PostNorm, Gemma2 -> SandwichNorm,
	// GPTNeoX/Cohere -> ParallelResidual); not a verbatim config.json key today.
	BlockTopology BlockTopology `json:"-"`
}

// RopeScaling mirrors config.json's nested rope_scaling block for longrope checkpoints.
// Only the longrope type is interpreted; other nested types leave the plain/flat RoPE
// path in force.
type RopeScaling struct {
	Type                string  `json:"type"`
	RopeType            string  `json:"rope_type"`
	Factor              float64 `json:"factor"`
	AttentionFactor     float64 `json:"attention_factor"`
	LowFreqFactor       float64 `json:"low_freq_factor"`
	HighFreqFactor      float64 `json:"high_freq_factor"`
	RopeTheta           float64 `json:"rope_theta"`
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`
	BetaFast            float64 `json:"beta_fast"`
	BetaSlow            float64 `json:"beta_slow"`
	MScale              float64 `json:"mscale"`
	MScaleAllDim        float64 `json:"mscale_all_dim"`
	Truncate            *bool   `json:"truncate"`
	// ShortFactor / LongFactor are per-(head_dim/2) rescale vectors. Phi divides
	// inv_freq[j] by the selected factor[j]; which vector is selected is pinned at
	// session start to the model's max-context regime (see ropeLongFactor).
	ShortFactor []float64 `json:"short_factor"`
	LongFactor  []float64 `json:"long_factor"`
	// OriginalMaxPositionEmbeddings is the pre-extension context length. The
	// short-vs-long selection and the attention temperature both key off
	// max_position_embeddings vs this value.
	OriginalMaxPositionEmbeddings int `json:"original_max_position_embeddings"`
}

// RopeParameters accepts both HF shapes seen in the wild:
//   - Gemma3-style maps keyed by layer type: {"full_attention": {...}, ...}
//   - flat default objects: {"rope_theta": 10000, "rope_type": "default"}
type RopeParameters map[string]RopeScaling

func (rp *RopeParameters) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	out := make(RopeParameters)
	flat := false
	for k, v := range raw {
		var r RopeScaling
		if err := json.Unmarshal(v, &r); err != nil {
			flat = true
			break
		}
		out[k] = r
	}
	if flat {
		var r RopeScaling
		if err := json.Unmarshal(b, &r); err != nil {
			return err
		}
		out["default"] = r
	}
	*rp = out
	return nil
}

func (r *RopeScaling) kind() string {
	if r == nil {
		return ""
	}
	if r.Type != "" {
		return r.Type
	}
	return r.RopeType
}

// eosToken is the scalar-or-list shape of HF's eos_token_id field. config.json emits
// it as a bare int (older models) or a list (Llama-3.x), so we accept both.
type eosToken struct {
	ids []int
}

func (e *eosToken) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '[' {
		return json.Unmarshal(b, &e.ids)
	}
	var one int
	if err := json.Unmarshal(b, &one); err != nil {
		return err
	}
	e.ids = []int{one}
	return nil
}

// configAlias avoids infinite recursion when Config.UnmarshalJSON delegates to the
// struct decoder.
type configAlias Config

type configJSONHints struct {
	BlockTopology     string   `json:"block_topology"`
	AttentionBias     *bool    `json:"attention_bias"`
	UseQKNorm         *bool    `json:"use_qk_norm"`
	QKNorm            *bool    `json:"qk_norm"`
	NormGain1p        *bool    `json:"norm_gain_1p"`
	LayerNorm         *bool    `json:"layer_norm"`
	ActGeluTanh       *bool    `json:"act_gelu_tanh"`
	ActGeluErf        *bool    `json:"act_gelu_erf"`
	DenseMLP          *bool    `json:"dense_mlp"`
	EmbedScale        *float64 `json:"embed_scale"`
	LogitScale        *float64 `json:"logit_scale"`
	ParallelAttention *bool    `json:"parallel_attn"`
	LayerNormEps      *float64 `json:"layer_norm_epsilon"`
	MultiQuery        *bool    `json:"multi_query"`
	NumKVHeadsAlt     *int     `json:"num_kv_heads"`
	Alibi             *bool    `json:"alibi"`
	SlidingWindow     *int     `json:"sliding_window"`
	Window            []int    `json:"sliding_window_per_layer"`
	HiddenAct         string   `json:"hidden_act"`
	HiddenActivation  string   `json:"hidden_activation"`
}

// UnmarshalJSON decodes config.json, then folds the scalar-or-list eos_token_id into
// both EOSTokenID (first) and EOSTokenIDs (full set). The rope-scaling params live in
// HF under a nested rope_scaling object; the flat json tags above are what
// export_oracle.py flattens them to, so a re-export carries them with zero code change.
func (c *Config) UnmarshalJSON(b []byte) error {
	aux := struct {
		*configAlias
		EOS eosToken `json:"eos_token_id"`
	}{configAlias: (*configAlias)(c)}
	// Multimodal wrappers (Qwen3.5 "Qwen3_5ForConditionalGeneration") nest the language-
	// model config under "text_config"; the top level holds only architectures/model_type
	// and the vision config. Decode the nested LM config first so dims/layer_types/rope
	// populate, then overlay the top level — which carries no LM dims, so JSON's
	// absent-field semantics leave the nested values intact.
	var probe struct {
		TextConfig json.RawMessage `json:"text_config"`
	}
	_ = json.Unmarshal(b, &probe)
	lm := b
	if len(probe.TextConfig) > 0 && string(probe.TextConfig) != "null" {
		if err := json.Unmarshal(probe.TextConfig, &aux); err != nil {
			return err
		}
		lm = probe.TextConfig
	}
	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}
	var hints configJSONHints
	if err := json.Unmarshal(lm, &hints); err != nil {
		return err
	}
	c.EOSTokenIDs = aux.EOS.ids
	if len(c.EOSTokenIDs) > 0 {
		c.EOSTokenID = c.EOSTokenIDs[0]
	}
	return c.deriveConfigAxes(hints)
}

func (c *Config) deriveConfigAxes(h configJSONHints) error {
	if c.HeadDim == 0 && c.HiddenSize != 0 && c.NumHeads != 0 {
		c.HeadDim = c.HiddenSize / c.NumHeads
	}
	if c.HiddenActivation == "" {
		c.HiddenActivation = h.HiddenActivation
	}
	if c.HiddenAct == "" {
		c.HiddenAct = h.HiddenAct
	}
	family := c.archFamilyKey()

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
	if c.RopeScaling == "" && c.LongRope != nil && c.LongRope.kind() == "llama3" {
		c.RopeScaling = "llama3"
		c.RopeFactor = c.LongRope.Factor
		c.RopeLowFreqFactor = c.LongRope.LowFreqFactor
		c.RopeHighFreqFactor = c.LongRope.HighFreqFactor
		c.RopeOrigContext = c.LongRope.OriginalMaxPositionEmbeddings
	}
	if c.RopeScaling == "" {
		if rp, ok := c.RopeParameters["default"]; ok && rp.kind() == "yarn" {
			c.RopeScaling = "yarn"
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

	if h.BlockTopology != "" {
		topo, ok := parseBlockTopology(h.BlockTopology)
		if !ok {
			return fmt.Errorf("block_topology: unknown %q", h.BlockTopology)
		}
		c.BlockTopology = topo
	} else {
		switch {
		case strings.Contains(family, "gemma2") || strings.Contains(family, "gemma3"):
			c.BlockTopology = SandwichNorm
		case strings.Contains(family, "olmo2"):
			c.BlockTopology = PostNorm
		case strings.Contains(family, "gptneox") || strings.Contains(family, "cohere") || (strings.Contains(family, "falcon") && c.ParallelAttention):
			c.BlockTopology = ParallelResidual
		}
	}

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
	c.deriveLayerAttentionAxes(family, h.SlidingWindow)
	if strings.Contains(family, "olmo2") || strings.Contains(family, "cohere2") || strings.Contains(family, "qwen3") || strings.Contains(family, "gemma3") || strings.Contains(family, "minimax") {
		if h.QKNorm == nil && h.UseQKNorm == nil {
			// MiniMax-M3 layers carry per-head q_norm/k_norm; the other families
			// above are the existing qk-norm checkpoints. A non-qk-norm MiniMax
			// export can still pin qk_norm=false explicitly to opt out.
			c.QKNorm = true
		}
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
	if strings.Contains(family, "cohere") && h.LogitScale == nil && c.LogitScale == 0 {
		c.LogitScale = 0.0625
	}
	if strings.Contains(family, "cohere") && h.LayerNorm == nil {
		c.LayerNorm = true
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
	return nil
}

func (c *Config) deriveLayerAttentionAxes(family string, slidingWindow *int) {
	if c.NumLayers <= 0 {
		return
	}
	if len(c.LayerTypes) == 0 && strings.Contains(family, "gemma3") {
		pattern := c.SlidingWindowPattern
		if pattern == 0 {
			pattern = 6
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
	if len(c.Window) == 0 && slidingWindow != nil && *slidingWindow > 0 {
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

func (c Config) layerType(layer int) string {
	if layer < 0 || layer >= len(c.LayerTypes) {
		return ""
	}
	return c.LayerTypes[layer]
}

func (c Config) activationName() string {
	if c.HiddenActivation != "" {
		return c.HiddenActivation
	}
	return c.HiddenAct
}

func (c Config) archFamilyKey() string {
	var b strings.Builder
	b.WriteString(c.ModelType)
	for _, arch := range c.Architectures {
		b.WriteByte(' ')
		b.WriteString(arch)
	}
	key := strings.ToLower(b.String())
	r := strings.NewReplacer("_", "", "-", "", " ", "")
	return r.Replace(key)
}

func (c Config) isGPTNeoX() bool {
	return strings.Contains(c.archFamilyKey(), "gptneox")
}

func (c Config) isGPTOSS() bool {
	return strings.Contains(c.archFamilyKey(), "gptoss")
}

// isGLM reports a GLM-family model (zai-org GLM lineage: glm, glm4, chatglm,
// glm_moe, glm_moe_dsa). The family key lowercases model_type + architectures
// with separators stripped, so "glm_moe_dsa" -> "glmmoedsa". No other family in
// the top-10 support matrix contains "glm", so the substring is unambiguous.
// Used to gate GLM-specific load behavior (mtp/vision tensor skip); the dense
// attention + generic MoE FFN paths are family-agnostic and already cover the
// GLM MoE FFN. The GLM-MoE-DSA cacheless path is handled by the DSA-specific
// MLA/indexer branch; reusable KV/index cache support remains a separate gate.
func (c Config) isGLM() bool {
	return strings.Contains(c.archFamilyKey(), "glm")
}

// isGLMMoeDsa reports the GLM-5.2 architecture specifically: model_type
// "glm_moe_dsa" — a MoE model with Dynamic Sparse Attention (a learned,
// content-dependent indexer) plus IndexShare (one indexer reused across every
// four sparse-attention layers) and an MTP head. The "dsa" token in the family
// key is the reliable signal that the attention path is the sparse variant, not
// dense GQA. Cacheless Forward and Session Prefill/Step have tiny-oracle
// witnesses for the GLM DSA path; eviction/invalidation for reusable DSA index
// cache entries remains a separate gate.
func (c Config) isGLMMoeDsa() bool {
	return c.isGLM() && strings.Contains(c.archFamilyKey(), "dsa")
}

// isMiniMax reports a MiniMax-family model (model_type / architectures such as
// "minimax_m3", "minimax_m2", "MiniMaxM3ForCausalLM"). The family key lowercases
// model_type + architectures with separators stripped, so "minimax_m3" ->
// "minimaxm3". No other family in the support matrix contains "minimax", so the
// substring is unambiguous. Used to gate MiniMax-specific load behavior (the
// multimodal vision tower + MTP head tensor skip) and the MSA sparse-attention axis.
func (c Config) isMiniMax() bool {
	return strings.Contains(c.archFamilyKey(), "minimax")
}

// isMiniMaxSparseAttn reports the MiniMax-M3 architecture specifically: a MiniMax
// model whose layers select between dense "full_attention" and block-sparse
// "minimax_m3_sparse" MSA layers. The "m3"/"sparse" signal plus a "minimax_m3_sparse"
// LayerTypes entry distinguishes M3's MSA path from the earlier MiniMax (M1 lightning
// / M2 full) attention. MSA selection math is witnessed by msa_index*.go; the full
// wired forward (lightning indexer projections, qk-norm, partial RoPE, SwiGLU-OAI MoE)
// plus a real-checkpoint oracle remain a separate gate.
func (c Config) isMiniMaxSparseAttn() bool {
	if !c.isMiniMax() {
		return false
	}
	if strings.Contains(c.archFamilyKey(), "m3") {
		return true
	}
	for _, t := range c.LayerTypes {
		if t == "minimax_m3_sparse" {
			return true
		}
	}
	return false
}

// isMSALayer reports whether layer l runs MiniMax-M3 block-sparse attention (its
// LayerTypes entry is "minimax_m3_sparse") rather than dense causal GQA
// ("full_attention" or unset). False for every non-MiniMax model, so the standard
// attention path is unchanged.
func (c Config) isMSALayer(l int) bool {
	return c.isMiniMax() && c.layerType(l) == "minimax_m3_sparse"
}

func parseBlockTopology(s string) (BlockTopology, bool) {
	key := strings.NewReplacer("_", "", "-", "", " ", "").Replace(strings.ToLower(s))
	switch key {
	case "", "pre", "prenorm":
		return PreNorm, true
	case "post", "postnorm":
		return PostNorm, true
	case "sandwich", "sandwichnorm":
		return SandwichNorm, true
	case "parallel", "parallelresidual":
		return ParallelResidual, true
	default:
		return PreNorm, false
	}
}

// isEOS reports whether id is a stop token. The list (when present) is authoritative;
// otherwise the scalar EOSTokenID is used. EOSTokenID==-1 with an empty list is the
// "never early-stop" convention used by fixed-length tool decode.
func (c Config) IsEOS(id int) bool {
	if len(c.EOSTokenIDs) == 0 {
		return id == c.EOSTokenID
	}
	for _, eos := range c.EOSTokenIDs {
		if id == eos {
			return true
		}
	}
	return false
}

// isLongrope reports whether this config drives the Phi longrope RoPE variant.
func (c Config) isLongrope() bool { return c.LongRope != nil && c.LongRope.kind() == "longrope" }

// IsMoE reports whether the FFN sub-layer is a Mixture-of-Experts (router +
// per-expert SwiGLU + weighted sum) rather than a single dense SwiGLU FFN.
// Dense (NumExperts==0) is the Llama default and stays bit-identical.
func (c Config) IsMoE() bool { return c.NumExperts > 0 }

// GroupSize is how many query heads share one KV head (GQA). For SmolLM2-135M:
// 9 query heads / 3 kv heads = 3.
func (c Config) GroupSize() int { return c.NumHeads / c.NumKVHeads }

// windowForLayer returns the sliding-window bound for layer l, or -1 (full causal
// attention) when no window is configured for that layer. The default — a nil/short
// Window slice — yields -1 for every layer, so the score loops reduce EXACTLY to the
// pre-SWA full-causal path (bit-identical for non-SWA models).
func (c Config) windowForLayer(l int) int {
	if l < 0 || l >= len(c.Window) {
		return -1
	}
	return c.Window[l]
}

func (c Config) hasLayerSpecificRopeTheta() bool {
	for l := 0; l < c.NumLayers && l < len(c.RopeThetaPerLayer); l++ {
		if c.RopeThetaPerLayer[l] != 0 && c.RopeThetaPerLayer[l] != c.RopeTheta {
			return true
		}
	}
	return false
}
