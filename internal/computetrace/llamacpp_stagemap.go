package computetrace

// llamacpp_stagemap.go -- the per-GGUF-arch table that maps a llama.cpp
// reference's cb() tensor names onto fak activation stages.
//
// Each bring-up diffs fak against a llama.cpp reference build whose graph names
// its intermediates with cb(cur, "<name>", il). The names differ per arch, so
// the converter takes the table by GGUF arch instead of hard-coding one model.
// Every arch keeps the same fak stage label where the semantics agree
// ("residual_streams" is the layer output, "attention_projected" the attention
// block output, "moe_sum" the full FFN output), so a per-layer profile reads the
// same columns across models.

import (
	"errors"
	"math"
	"slices"
	"sort"
	"strings"
)

// Registered reference stage-map keys.
const (
	LlamaCppArchDeepSeek41    = "deepseek41"
	LlamaCppArchQwen4Exp      = "qwen4exp"
	LlamaCppArchMiniMaxM3Core = "minimax-m3-core"
	LlamaCppArchQwen35        = "qwen35"
)

// ErrUnknownStageMap is returned for an arch with no registered stage map.
var ErrUnknownStageMap = errors.New("computetrace: no llama.cpp stage map for arch")

// LlamaCppStage maps one llama.cpp tensor name onto a fak activation stage.
type LlamaCppStage struct {
	// Stage is the fak activation stage label (Event.Phase).
	Stage string `json:"stage"`
	// WidthDims is how many leading ne dims form one token row; every
	// remaining dim counts tokens.
	WidthDims int `json:"width_dims"`
	// RowRank, when nonzero, requires later axes to be singleton so head,
	// group or cache dimensions cannot silently become token rows.
	RowRank int `json:"row_rank,omitempty"`
	// RoPETail is the trailing row span the reference captured after RoPE
	// while fak samples before it; sample indices there are excluded.
	RoPETail int `json:"rope_tail,omitempty"`
	// TailAligned marks a stage the reference computes only for output rows on
	// the last layer and after the stack (llama.cpp gathers inp_out_ids there).
	// When such a stage holds fewer rows than the prompt, its rows are the
	// trailing tokens and are numbered from the end.
	TailAligned bool `json:"tail_aligned,omitempty"`
	// SupersededBy names another base whose presence in the same layer means
	// this name holds a different tensor there and is skipped.
	SupersededBy string `json:"superseded_by,omitempty"`
	// Note records a capture condition the stage depends on.
	Note string `json:"note,omitempty"`
}

// LlamaCppStageMap is the stage table for one GGUF arch.
type LlamaCppStageMap struct {
	Arch string `json:"arch"`
	// Reference names the llama.cpp build whose cb() names the table keys.
	Reference string `json:"reference"`
	// Layered keys "<base>" of per-layer tensors named "<base>-<layer>".
	Layered map[string]LlamaCppStage `json:"layered"`
	// Pre keys bare tensor names emitted before the layer stack; they are
	// numbered layer -1.
	Pre map[string]LlamaCppStage `json:"pre,omitempty"`
	// Post keys bare tensor names emitted after the layer stack; they are
	// numbered one past the last converted layer.
	Post map[string]LlamaCppStage `json:"post,omitempty"`
	// Unmapped lists reference names deliberately left out, with the reason,
	// so a port map can tell "not converted on purpose" from "forgotten".
	Unmapped map[string]string `json:"unmapped,omitempty"`
}

// llamaCppPostLayer is the provisional layer of a Post stage until the
// converter knows the last layer.
const llamaCppPostLayer = math.MaxInt32

func (m LlamaCppStageMap) lookup(name string) (LlamaCppStage, int, bool) {
	if s, ok := m.Pre[name]; ok {
		return s, -1, true
	}
	if s, ok := m.Post[name]; ok {
		return s, llamaCppPostLayer, true
	}
	base, layer, ok := splitLlamaCppName(name)
	if !ok {
		return LlamaCppStage{}, 0, false
	}
	s, ok := m.Layered[base]
	return s, layer, ok
}

// Names returns every reference name the map converts or deliberately skips,
// sorted.
func (m LlamaCppStageMap) Names() []string {
	var out []string
	for _, t := range []map[string]LlamaCppStage{m.Layered, m.Pre, m.Post} {
		for n := range t {
			out = append(out, n)
		}
	}
	for n := range m.Unmapped {
		out = append(out, n)
	}
	sort.Strings(out)
	return slices.Compact(out)
}

// finish numbers Post stages one past maxLayer, and re-numbers the TailAligned
// rows of the last layer and the Post stages as the trailing prompt tokens when
// they cover fewer tokens than the prompt.
func (m LlamaCppStageMap) finish(a *Artifact, maxLayer int, rows map[llamaCppRow]int) {
	lastTail, postTail := map[string]bool{}, map[string]bool{}
	for _, s := range m.Layered {
		if s.TailAligned {
			lastTail[s.Stage] = true
		}
	}
	for _, s := range m.Post {
		if s.TailAligned {
			postTail[s.Stage] = true
		}
	}
	prompt := len(a.InputTokens)
	for i := range a.Events {
		e := &a.Events[i]
		post := e.Layer == llamaCppPostLayer
		if (post && postTail[e.Phase]) || (e.Layer == maxLayer && lastTail[e.Phase]) {
			if n := rows[llamaCppRow{stage: e.Phase, layer: e.Layer}]; n < prompt {
				e.Token += prompt - n
			}
		}
		if post {
			e.Layer = maxLayer + 1
		}
	}
}

var llamaCppStageMaps = map[string]LlamaCppStageMap{
	LlamaCppArchMiniMaxM3Core: {
		Arch:      LlamaCppArchMiniMaxM3Core,
		Reference: "llama.cpp@970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc + MiniMax-M3 patch sha256:666bf3ccf76cdeec9b0e6b40b248440cd6fd234e9c43fb10130e3ec3230de289 src/models/minimax-m3.cpp; core row stages only, source-derived schema",
		Layered: map[string]LlamaCppStage{
			"ffn_moe_out": {Stage: "moe_routed", WidthDims: 1, RowRank: 2, TailAligned: true},
			"ffn_shexp":   {Stage: "shared_expert", WidthDims: 1, RowRank: 2, TailAligned: true},
			"ffn_out":     {Stage: "moe_sum", WidthDims: 1, RowRank: 2, TailAligned: true, Note: "full FFN output, including leading dense layers"},
			"l_out":       {Stage: "residual_streams", WidthDims: 1, RowRank: 2, TailAligned: true},
		},
		Post: map[string]LlamaCppStage{
			"result_norm":   {Stage: "final_norm", WidthDims: 1, RowRank: 2, TailAligned: true},
			"result_output": {Stage: "logits", WidthDims: 1, RowRank: 2, TailAligned: true},
		},
		Unmapped: map[string]string{
			"model.input_embed": "outside the core FFN/residual/output subset",
			"attn_norm":         "outside the core FFN/residual/output subset",
			"Qcur_normed":       "multi-head query axes require a separate stage contract",
			"Kcur_normed":       "multi-head key axes require a separate stage contract",
			"Qcur":              "post-RoPE query, not a pre-RoPE activation alias",
			"Kcur":              "post-RoPE key, not a pre-RoPE activation alias",
			"Vcur":              "multi-head value axes require a separate stage contract",
			"attn_out":          "dense helper callback is absent after the sparse output projection; no uniform projected-attention alias",
			"kqv_out":           "sparse attention before output projection, not attention_projected",
			"ffn_inp":           "attention residual addition; retain raw src0 relation, do not alias its sum to attention_projected",
			"ffn_norm":          "outside the core FFN/residual/output subset",
			"msa_fattn":         "grouped attention output before flattening and output projection",
			"msa_bs":            "indexer cache-block scores; axes are not fixed per-token activation rows",
			"msa_bsf":           "indexer cache-block scores; axes are not fixed per-token activation rows",
			"msa_block_mask":    "indexer block/head mask, not a per-token activation row",
			"msa_mask4":         "attention mask with cache/head axes, not a per-token activation row",
		},
	},
	LlamaCppArchDeepSeek41: {
		Arch:      LlamaCppArchDeepSeek41,
		Reference: "vcruz305/llama.cpp@5210c7c5ed61dddaee6ed476623abf4b63093d16",
		Layered: map[string]LlamaCppStage{
			"qr":       {Stage: "q_latent", WidthDims: 1},
			"kv":       {Stage: "kv_latent", WidthDims: 1, RoPETail: 64},
			"attn_out": {Stage: "attention_projected", WidthDims: 1},
			"ffn_out":  {Stage: "moe_sum", WidthDims: 1},
			"l_last":   {Stage: "residual_streams", WidthDims: 2},
		},
	},
	LlamaCppArchQwen4Exp: {
		Arch:      LlamaCppArchQwen4Exp,
		Reference: "Nathanw1014/llama.cpp@dff60048744f99cb0af68d02be2314456b3269dc src/models/qwen4exp.cpp, src/models/delta-net-base.cpp",
		Pre: map[string]LlamaCppStage{
			"model.input_embed": {Stage: "embedding", WidthDims: 1},
			"hc_init":           {Stage: "residual_streams", WidthDims: 2},
		},
		Layered: map[string]LlamaCppStage{
			"ple_embd":        {Stage: "ple_embedding", WidthDims: 1, Note: "a graph input leaf under the default host gather; dumped only with LLAMA_PLE_HOST_GATHER=0"},
			"ple_gate":        {Stage: "ple_gate", WidthDims: 2},
			"ple_gated_value": {Stage: "ple_gated_value", WidthDims: 2},
			"ple_conv_out":    {Stage: "ple_conv_out", WidthDims: 2},

			"linear_attn_qkv_mixed": {Stage: "gdn_qkv_mixed", WidthDims: 1},
			"z":                     {Stage: "gdn_z", WidthDims: 1},
			"final_output":          {Stage: "gdn_core_normed", WidthDims: 1},
			"linear_attn_out":       {Stage: "attention_projected", WidthDims: 1},

			"Qcur_full":     {Stage: "attn_q_full", WidthDims: 1},
			"Qcur_normed":   {Stage: "attn_q_normed", WidthDims: 2},
			"Kcur_normed":   {Stage: "attn_k_normed", WidthDims: 2},
			"gate_reshaped": {Stage: "attn_gate", WidthDims: 1},
			"attn_pregate":  {Stage: "attn_core", WidthDims: 1},
			"gate_sigmoid":  {Stage: "attn_gate_sigmoid", WidthDims: 1},
			"attn_gated":    {Stage: "attn_gated", WidthDims: 1},
			"attn_output":   {Stage: "attention_projected", WidthDims: 1, SupersededBy: "linear_attn_out"},

			"ffn_moe_out":                {Stage: "moe_routed", WidthDims: 1, TailAligned: true},
			"ffn_shexp":                  {Stage: "shared_expert", WidthDims: 1, TailAligned: true},
			"shared_expert_gate_sigmoid": {Stage: "shared_expert_gate", WidthDims: 1, TailAligned: true},
			"ffn_shexp_gated":            {Stage: "shared_expert_gated", WidthDims: 1, TailAligned: true},
			"ffn_out":                    {Stage: "moe_sum", WidthDims: 1, TailAligned: true},
			"l_last":                     {Stage: "residual_streams", WidthDims: 2, TailAligned: true},
		},
		Post: map[string]LlamaCppStage{
			"result_norm":   {Stage: "final_norm", WidthDims: 1, TailAligned: true},
			"result_output": {Stage: "logits", WidthDims: 1, TailAligned: true},
		},
		Unmapped: map[string]string{
			"hc_norm":    "emitted twice per layer (attention and ffn mix); the name does not say which",
			"hc_gate":    "emitted twice per layer (attention and ffn mix); the name does not say which",
			"hc_mixed":   "emitted twice per layer (attention and ffn mix); the name does not say which",
			"hc_inject":  "emitted twice per layer (attention and ffn mix); the name does not say which",
			"hc_combine": "emitted twice per layer (attention and ffn combine); the name does not say which",
			"Qcur":       "emitted before and after RoPE under one name",
			"Kcur":       "emitted before and after RoPE under one name",
			"Vcur":       "emitted twice under one name",

			"Qcur_reshaped": "non-contiguous view of Qcur_full; Qcur_normed carries the stage",
			"kqv_out":       "QSA-path intermediate of attn_core",

			"indexer_k_raw":        "indexer cache-block state, not a per-token row",
			"indexer_k_pooled":     "indexer cache-block state, not a per-token row",
			"indexer_k":            "indexer cache-block state, not a per-token row",
			"indexer_q":            "post-RoPE multi-head indexer query",
			"indexer_score":        "width is the cache block count, not fixed per token",
			"indexer_score_tokens": "width is the KV cell count, not fixed per token",
			"indexer_top_k":        "i32 selection, not an f32 activation",

			"beta":             "GDN gate internal",
			"beta_sigmoid":     "GDN gate internal",
			"alpha":            "GDN gate internal",
			"a_softplus":       "GDN gate internal",
			"gate":             "GDN gate internal",
			"state_predelta":   "GDN recurrent state, not a per-token row",
			"new_state":        "GDN recurrent state (delta-net-base), not a per-token row",
			"conv_state_at":    "GDN conv state, not a per-token row",
			"conv_output_raw":  "GDN conv internal",
			"conv_output_silu": "GDN conv internal",
			"q_conv":           "GDN conv split",
			"k_conv":           "GDN conv split",
			"v_conv":           "GDN conv split",
			"q_conv_predelta":  "GDN conv split",
			"k_conv_predelta":  "GDN conv split",
			"v_conv_predelta":  "GDN conv split",

			"shared_expert_gate": "pre-sigmoid logit; shared_expert_gate_sigmoid carries the stage",

			"h_nextn":      "MTP draft input; the pinned GGUF has no MTP layers",
			"mtp_tok_embd": "MTP draft head; the pinned GGUF has no MTP layers",
			"mtp_hnorm":    "MTP draft head; the pinned GGUF has no MTP layers",
			"mtp_enorm":    "MTP draft head; the pinned GGUF has no MTP layers",
			"mtp_eh_proj":  "MTP draft head; the pinned GGUF has no MTP layers",
			"mtp_ffn_out":  "MTP draft head; the pinned GGUF has no MTP layers",
		},
	},

	// qwen35: Qwen3.8-27B at ggml-org/llama.cpp@6c84c7d5. 64 decoder layers:
	// gated full attention where (il+1)%full_attention_interval==0 with interval
	// 4 (il%4==3), Gated-DeltaNet elsewhere; dense SiLU FFN, no MoE. The
	// optional cvec applied to the layer output is absent in the pinned GGUF,
	// so l_out is the plain per-layer residual stream. The MTP draft head
	// (block 65) builds only under LLM_GRAPH_TYPE_DECODER_MTP and never runs in
	// the main pass, so its names are unmapped. CPU prefill takes the chunked
	// GDN path (the fused op probe fails without a GPU backend), so chunked
	// internals appear in dumps but are unmapped by design.
	LlamaCppArchQwen35: {
		Arch:      LlamaCppArchQwen35,
		Reference: "ggml-org/llama.cpp@6c84c7d5d8833c6e0df69628f75a0f599797934e src/models/qwen35.cpp, src/models/delta-net-base.cpp; 64 layers, gated full attention at il%4==3 else GDN, dense SiLU FFN, cvec absent in the pinned GGUF; CPU prefill runs the chunked GDN path, whose chunked internals appear in dumps but are unmapped by design",
		Pre: map[string]LlamaCppStage{
			"model.input_embed": {Stage: "embedding", WidthDims: 1, RowRank: 2},
		},
		Layered: map[string]LlamaCppStage{
			"attn_norm":      {Stage: "attn_norm", WidthDims: 1, RowRank: 2},
			"attn_residual":  {Stage: "attn_residual", WidthDims: 1, RowRank: 2},
			"attn_post_norm": {Stage: "attn_post_norm", WidthDims: 1, RowRank: 2},
			"ffn_out":        {Stage: "moe_sum", WidthDims: 1, RowRank: 2, Note: "qwen35 FFN is dense SiLU, not MoE; moe_sum kept for cross-arch consistency as the full FFN output"},
			"l_out":          {Stage: "residual_streams", WidthDims: 1, RowRank: 2, Note: "layer output after cvec (identity when cvec is absent); the per-layer residual stream"},

			"linear_attn_qkv_mixed": {Stage: "gdn_qkv_mixed", WidthDims: 1},
			"z":                     {Stage: "gdn_z", WidthDims: 1},
			"final_output":          {Stage: "gdn_core_normed", WidthDims: 1},
			"linear_attn_out":       {Stage: "attention_projected", WidthDims: 1},

			"Qcur_full":     {Stage: "attn_q_full", WidthDims: 1},
			"Qcur_normed":   {Stage: "attn_q_normed", WidthDims: 2, RowRank: 3},
			"Kcur_normed":   {Stage: "attn_k_normed", WidthDims: 2, RowRank: 3},
			"gate_reshaped": {Stage: "attn_gate", WidthDims: 1},
			"attn_pregate":  {Stage: "attn_core", WidthDims: 1},
			"gate_sigmoid":  {Stage: "attn_gate_sigmoid", WidthDims: 1},
			"attn_gated":    {Stage: "attn_gated", WidthDims: 1},
			"attn_output":   {Stage: "attention_projected", WidthDims: 1, SupersededBy: "linear_attn_out"},
		},
		Post: map[string]LlamaCppStage{
			"result_norm":   {Stage: "final_norm", WidthDims: 1, RowRank: 2, TailAligned: true},
			"result_output": {Stage: "logits", WidthDims: 1, RowRank: 2, TailAligned: true},
		},
		Unmapped: map[string]string{
			"h_nextn":  "NextN/MTP draft-head input; R0 scope excludes the MTP draft head",
			"post_ffn": "FFN residual add; numerically identical to l_out when cvec is absent, and l_out carries residual_streams",

			"Qcur_reshaped": "head-split view of Qcur_full; non-contiguous, no stable per-token row",
			"Kcur":          "emitted both pre-QK-norm and post-RoPE under one name; RoPE span is section-dependent",
			"Vcur":          "emitted both pre-reshape and post-reshape under one name",
			"Qcur":          "post-RoPE query; no stable pre-RoPE alias at this pin",

			"beta":         "GDN gate tensor with head-axis layout, not a per-token row",
			"beta_sigmoid": "GDN gate tensor with head-axis layout, not a per-token row",
			"alpha":        "GDN gate tensor with head-axis layout, not a per-token row",
			"a_softplus":   "GDN gate intermediate",
			"gate":         "GDN gate intermediate",

			"state_predelta": "GDN recurrent-state snapshot, not a per-token row",
			"new_state":      "GDN recurrent-state update, not a per-token row",
			"output_state":   "GDN recurrent-state snapshot, not a per-token row",

			"conv_states":          "GDN conv rolling state",
			"conv_states_reshaped": "GDN conv rolling state view",
			"conv_state_last":      "GDN conv rolling state view",
			"conv_state_update":    "GDN conv rolling state view",
			"qkv_mixed_transposed": "transposed view of linear_attn_qkv_mixed",
			"conv_input":           "concat of conv state prefix and qkv rows; includes state, not a pure per-token activation",
			"conv_output_raw":      "GDN short-conv internal",
			"conv_output_silu":     "GDN short-conv internal",
			"q_conv":               "view splitting the conv output",
			"k_conv":               "view splitting the conv output",
			"v_conv":               "view splitting the conv output",
			"q_conv_predelta":      "post-l2norm/repeat conv split, head-axis layout",
			"k_conv_predelta":      "post-l2norm/repeat conv split, head-axis layout",
			"v_conv_predelta":      "post-l2norm/repeat conv split, head-axis layout",

			"q_in":         "chunked-GDN input after scale; head/chunk axes, not fixed per-token rows",
			"k_in":         "chunked-GDN input after scale; head/chunk axes, not fixed per-token rows",
			"v_in":         "chunked-GDN input after scale; head/chunk axes, not fixed per-token rows",
			"b_in":         "chunked-GDN input after scale; head/chunk axes, not fixed per-token rows",
			"g_in":         "chunked-GDN input after scale; head/chunk axes, not fixed per-token rows",
			"v_b":          "chunked-GDN solve intermediate",
			"k_b":          "chunked-GDN solve intermediate",
			"k_beta_g_exp": "chunked-GDN solve intermediate",
			"k_cumdecay":   "chunked-GDN solve intermediate",
			"g_cs":         "chunked-GDN solve intermediate",

			"decay_mask":              "chunk x chunk matrix, not a per-token row",
			"kq":                      "chunk x chunk matrix, not a per-token row",
			"attn":                    "chunk x chunk matrix, not a per-token row",
			"dnet_add_ch_lhs":         "chunk x chunk matrix, not a per-token row",
			"attn_pre_solve":          "chunk x chunk matrix, not a per-token row",
			"dnet_add_ch_attn_solved": "chunk x chunk matrix, not a per-token row",
			"g_last":                  "chunk-boundary decay/state-update term",
			"g_last_exp_t":            "chunk-boundary decay/state-update term",
			"g_diff":                  "chunk-boundary decay/state-update term",
			"key_gdiff":               "chunk-boundary decay/state-update term",
			"key_gdiff_t":             "chunk-boundary decay/state-update term",
			"dnet_add_ch_state":       "per-chunk recurrent-state snapshot (one emission per chunk iteration)",
			"v_prime":                 "per-chunk value-path slice",
			"v_t_new":                 "per-chunk value-path slice",
			"v_attn":                  "per-chunk value-path slice",
			"attn_inter":              "per-chunk value-path slice",
			"dnet_add_ch_attn_out":    "per-chunk value-path slice",
			"dnet_add_ar_state":       "autoregressive GDN state update; decode-only path",

			"mtp_tok_embd":         "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_hnorm":            "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_enorm":            "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_concat":           "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_eh_proj":          "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_attn_norm":        "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_Qcur_full":        "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_Qcur_normed":      "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_gate":             "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_Kcur_normed":      "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_Vcur":             "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_attn_pregate":     "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_attn_out":         "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_attn_residual":    "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_attn_post_norm":   "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_ffn_out":          "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_post_ffn":         "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
			"mtp_shared_head_norm": "MTP draft head (graph_mtp / LLM_GRAPH_TYPE_DECODER_MTP); not executed in the main-pass R0 capture",
		},
	},
}

// LlamaCppStageMapFor returns the stage map registered for a GGUF arch.
func LlamaCppStageMapFor(arch string) (LlamaCppStageMap, bool) {
	m, ok := llamaCppStageMaps[strings.TrimSpace(arch)]
	return m, ok
}

// LlamaCppStageMapArches lists the registered GGUF arches, sorted.
func LlamaCppStageMapArches() []string {
	out := make([]string, 0, len(llamaCppStageMaps))
	for a := range llamaCppStageMaps {
		out = append(out, a)
	}
	sort.Strings(out)
	return out
}
