package ggufload

// Qwen3.8-Flash-Next ("qwen4exp") GGUF hyperparameters and tensor schema. Key
// reads, validation and the tensor set follow Nathanw1014/llama.cpp@dff60048744f99cb0af68d02be2314456b3269dc
// src/models/qwen4exp.cpp load_arch_hparams/load_arch_tensors and the converter
// conversion/qwen4exp.py at the same revision (MIT License, Copyright (c) 2023-2026
// The ggml authors).

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	qwen4ExpArch = "qwen4exp"

	q4xKeyHCCount        = "hyper_connection.count"
	q4xKeyHCLowRank      = "hyper_connection.low_rank"
	q4xKeyIndexHeads     = "attention.indexer.head_count"
	q4xKeyIndexKeyLen    = "attention.indexer.key_length"
	q4xKeyIndexTopK      = "attention.indexer.top_k"
	q4xKeyCompressRatios = "attention.compress_ratios"
	q4xKeyRopeSections   = "rope.dimension_sections"
	q4xKeyRecurrent      = "attention.recurrent_layers"
	q4xKeySharedFFN      = "expert_shared_feed_forward_length"
	q4xKeySSMInner       = "ssm.inner_size"
	q4xKeyPLELayers      = "ple.layers"
	q4xKeyPLENgram       = "ple.ngram_size"
	q4xKeyPLEHeadsPerNg  = "ple.heads_per_ngram"
	q4xKeyPLEConv        = "ple.conv_kernel"
	q4xKeyPLEEOS         = "ple.eos_token_id"
	q4xKeyPLEImage       = "ple.image_token_id"
	q4xKeyPLEEmbedPerLyr = "embedding_length_per_layer_input"
	q4xKeyPLEMultipliers = "ple.layer_multipliers"
	q4xKeyPLEHeadOffsets = "ple.head_offsets"
	q4xKeyPLEHeadVocab   = "ple.head_vocab_sizes"

	// dff6004 llama-hparams.h bounds the PLE hash geometry.
	q4xMaxPLENgram = 8
	q4xMaxPLEHeads = 64
)

func archIsQwen4Exp(arch string) bool { return arch == qwen4ExpArch }

// applyQwen4ExpConfig reads every qwen4exp key the reference loader reads and
// refuses the values it refuses. p is the file's own "<arch>." prefix.
func applyQwen4ExpConfig(f *File, p string, cfg *model.Config) error {
	x := &model.Qwen4ExpHParams{}
	req := func(key string) (int, error) {
		v, ok := f.Uint64(p + key)
		if !ok {
			return 0, fmt.Errorf("gguf: qwen4exp missing %s%s", p, key)
		}
		if v == 0 || v > math.MaxInt32 {
			return 0, fmt.Errorf("gguf: qwen4exp %s%s must be in (0, 2^31), got %d", p, key, v)
		}
		return int(v), nil
	}
	var err error

	// A NextN key with no head tensor is a mainline file whose converter dropped the
	// draft head: the reference loads every block as trunk.
	if n := cfg.NumNextNPredictLayers; n != 0 {
		if f.hasTensor(fmt.Sprintf("blk.%d.nextn.eh_proj.weight", cfg.NumLayers)) {
			return fmt.Errorf("gguf: qwen4exp %snextn_predict_layers=%d; the MTP draft head is not supported", p, n)
		}
		cfg.NumLayers += n
		cfg.NumNextNPredictLayers = 0
		cfg.LayerTypes = nil
	}
	applyMoEExpertCounts(f, p, cfg)
	if cfg.NumExperts == 0 || cfg.NumExpertsPerTok == 0 {
		return fmt.Errorf("gguf: qwen4exp needs expert_count and expert_used_count")
	}
	if cfg.MoEIntermediateSize == 0 {
		cfg.MoEIntermediateSize = cfg.IntermediateSize / cfg.NumExpertsPerTok
	}
	x.SharedExpertFFN = cfg.IntermediateSize
	if f.hasKey(p + q4xKeySharedFFN) {
		if x.SharedExpertFFN, err = req(q4xKeySharedFFN); err != nil {
			return err
		}
	}
	if cfg.MoEIntermediateSize <= 0 || x.SharedExpertFFN <= 0 {
		return fmt.Errorf("gguf: qwen4exp expert and shared-expert widths must be positive")
	}
	cfg.SharedIntermediateSize = x.SharedExpertFFN
	cfg.NSharedExperts = 1
	cfg.NormTopKProb = true

	if x.RopeSections, err = q4xIntArray(f, p+q4xKeyRopeSections, 4); err != nil {
		return err
	}
	if cfg.LinearConvKernelDim == 0 || cfg.LinearKeyHeadDim == 0 || cfg.LinearNumKeyHeads == 0 || cfg.LinearNumValueHeads == 0 {
		return fmt.Errorf("gguf: qwen4exp needs non-zero ssm.conv_kernel, ssm.state_size, ssm.group_count and ssm.time_step_rank")
	}
	if _, err := req(q4xKeySSMInner); err != nil {
		return err
	}

	if x.HCCount, err = req(q4xKeyHCCount); err != nil {
		return err
	}
	if x.HCCount <= 1 {
		return fmt.Errorf("gguf: qwen4exp %s%s must be greater than one, got %d", p, q4xKeyHCCount, x.HCCount)
	}
	if x.HCLowRank, err = req(q4xKeyHCLowRank); err != nil {
		return err
	}
	cfg.HCMult = x.HCCount

	if x.IndexNHeads, err = req(q4xKeyIndexHeads); err != nil {
		return err
	}
	if x.IndexHeadDim, err = req(q4xKeyIndexKeyLen); err != nil {
		return err
	}
	if x.IndexTopK, err = req(q4xKeyIndexTopK); err != nil {
		return err
	}
	if f.hasKey(p + q4xKeyCompressRatios) {
		if x.CompressRatios, err = q4xIntArray(f, p+q4xKeyCompressRatios, cfg.NumLayers); err != nil {
			return err
		}
	}

	if f.hasKey(p + q4xKeyRecurrent) {
		rec, ok := f.BoolArray(p + q4xKeyRecurrent)
		if !ok || len(rec) != cfg.NumLayers {
			return fmt.Errorf("gguf: qwen4exp %s%s must be a %d-entry bool array", p, q4xKeyRecurrent, cfg.NumLayers)
		}
		cfg.LayerTypes = make([]string, cfg.NumLayers)
		for l, r := range rec {
			cfg.LayerTypes[l] = "full_attention"
			if r {
				cfg.LayerTypes[l] = "linear_attention"
			}
		}
	} else if len(cfg.LayerTypes) != cfg.NumLayers {
		interval := uint64(4)
		if v, ok := f.Uint64(p + "full_attention_interval"); ok {
			interval = v
		}
		if interval == 0 || interval > math.MaxInt32 {
			return fmt.Errorf("gguf: qwen4exp %sfull_attention_interval must be greater than zero, got %d", p, interval)
		}
		cfg.FullAttentionInterval = int(interval)
		cfg.LayerTypes = make([]string, cfg.NumLayers)
		for l := range cfg.LayerTypes {
			cfg.LayerTypes[l] = "linear_attention"
			if (l+1)%cfg.FullAttentionInterval == 0 {
				cfg.LayerTypes[l] = "full_attention"
			}
		}
	}

	if layers, _ := f.IntArray(p + q4xKeyPLELayers); len(layers) > 0 {
		ple, err := readQwen4ExpPLE(f, p, cfg.NumLayers, req)
		if err != nil {
			return err
		}
		x.PLE = ple
	}
	cfg.Qwen4Exp = x
	return nil
}

func readQwen4ExpPLE(f *File, p string, nLayers int, req func(string) (int, error)) (*model.Qwen4ExpPLE, error) {
	layers, _ := f.IntArray(p + q4xKeyPLELayers)
	if len(layers) != 1 {
		return nil, fmt.Errorf("gguf: qwen4exp %s%s lists %d layers, but only one PLE layer is supported", p, q4xKeyPLELayers, len(layers))
	}
	if layers[0] >= nLayers {
		return nil, fmt.Errorf("gguf: qwen4exp %s%s layer %d outside [0,%d)", p, q4xKeyPLELayers, layers[0], nLayers)
	}
	ple := &model.Qwen4ExpPLE{Layers: layers}
	var err error
	if ple.NgramSize, err = req(q4xKeyPLENgram); err != nil {
		return nil, err
	}
	if ple.HeadsPerNgram, err = req(q4xKeyPLEHeadsPerNg); err != nil {
		return nil, err
	}
	if ple.ConvKernel, err = req(q4xKeyPLEConv); err != nil {
		return nil, err
	}
	eos, ok := f.Uint64(p + q4xKeyPLEEOS)
	if !ok || eos > math.MaxInt32 {
		return nil, fmt.Errorf("gguf: qwen4exp missing %s%s", p, q4xKeyPLEEOS)
	}
	ple.EOSTokenID = int(eos)
	ple.ImageTokenID = ple.EOSTokenID
	if img, ok := f.Uint64(p + q4xKeyPLEImage); ok && img <= math.MaxInt32 {
		ple.ImageTokenID = int(img)
	}
	if ple.EmbedPerLayer, err = req(q4xKeyPLEEmbedPerLyr); err != nil {
		return nil, err
	}
	if ple.NgramSize < 2 || ple.NgramSize > q4xMaxPLENgram {
		return nil, fmt.Errorf("gguf: qwen4exp %s%s=%d outside [2,%d]", p, q4xKeyPLENgram, ple.NgramSize, q4xMaxPLENgram)
	}
	ple.NumHeads = (ple.NgramSize - 1) * ple.HeadsPerNgram
	if ple.NumHeads > q4xMaxPLEHeads {
		return nil, fmt.Errorf("gguf: qwen4exp PLE has %d hash heads, more than %d", ple.NumHeads, q4xMaxPLEHeads)
	}
	if ple.LayerMultipliers, err = q4xUint64Array(f, p+q4xKeyPLEMultipliers, ple.NgramSize); err != nil {
		return nil, err
	}
	if ple.HeadOffsets, err = q4xUint64Array(f, p+q4xKeyPLEHeadOffsets, ple.NumHeads); err != nil {
		return nil, err
	}
	if ple.HeadVocabSizes, err = q4xUint64Array(f, p+q4xKeyPLEHeadVocab, ple.NumHeads); err != nil {
		return nil, err
	}
	for h := 0; h < ple.NumHeads; h++ {
		if ple.HeadOffsets[h]+ple.HeadVocabSizes[h] > math.MaxInt32 {
			return nil, fmt.Errorf("gguf: qwen4exp PLE head %d range does not fit the int32 row index", h)
		}
	}
	return ple, nil
}

// q4xIntArray reads an integer array of at least n entries (a short array would
// leave the zero tail the reference refuses).
func q4xIntArray(f *File, key string, n int) ([]int, error) {
	v, ok := f.IntArray(key)
	if !ok || len(v) < n {
		return nil, fmt.Errorf("gguf: qwen4exp %s must be an integer array of at least %d entries", key, n)
	}
	return v, nil
}

func q4xUint64Array(f *File, key string, n int) ([]uint64, error) {
	v := uint64ArrayOrNil(f, key)
	if len(v) < n {
		return nil, fmt.Errorf("gguf: qwen4exp %s must be an integer array of at least %d entries", key, n)
	}
	return v[:n], nil
}

func init() {
	archTensorSchemas[qwen4ExpArch] = qwen4ExpTensorSchema
	archHFBindings[qwen4ExpArch] = qwen4ExpHFBindings
}

// qwen4ExpTensorSchema lists every tensor dff6004's load_arch_tensors creates for
// a target-only (no NextN) file, with its GGUF dims.
func qwen4ExpTensorSchema(cfg model.Config, f *File) []TensorRequest {
	x := cfg.Qwen4Exp
	if x == nil {
		return nil
	}
	nEmbd := uint64(cfg.HiddenSize)
	nVocab := uint64(cfg.VocabSize)
	hc := uint64(x.HCCount)
	hcDim := hc * nEmbd
	hcLR := uint64(x.HCLowRank)
	nExp := uint64(cfg.NumExperts)
	ffExp := uint64(cfg.MoEIntermediateSize)
	ffSh := uint64(x.SharedExpertFFN)
	headK := uint64(cfg.HeadDim)
	nHead := uint64(cfg.NumHeads)
	kvGQA := headK * uint64(cfg.NumKVHeads)
	keyDim := uint64(cfg.LinearKeyHeadDim * cfg.LinearNumKeyHeads)
	valDim := uint64(cfg.LinearValueHeadDim * cfg.LinearNumValueHeads)
	nV := uint64(cfg.LinearNumValueHeads)
	idx := uint64(x.IndexHeadDim)

	reqs := []TensorRequest{
		{Name: "token_embd.weight", Dims: []uint64{nEmbd, nVocab}},
		{Name: "output_hc_norm.weight", Dims: []uint64{hcDim}},
		{Name: "output_hc_down.weight", Dims: []uint64{hcDim, hcLR}},
		{Name: "output_hc_up.weight", Dims: []uint64{hcLR, hcDim}},
		{Name: "output.weight", Dims: []uint64{nEmbd, nVocab}, Optional: true},
	}
	if x.PLE != nil {
		dims := []uint64{uint64(x.PLE.EmbedPerLayer)}
		if t, ok := tensorByName(f, "per_layer_token_embd.weight"); ok && len(t.Dims) == 2 {
			dims = append(dims, t.Dims[1])
		}
		reqs = append(reqs, TensorRequest{Name: "per_layer_token_embd.weight", Dims: dims})
	}
	for l := 0; l < cfg.NumLayers; l++ {
		b := func(s string) string { return fmt.Sprintf("blk.%d.%s", l, s) }
		add := func(s string, dims ...uint64) { reqs = append(reqs, TensorRequest{Name: b(s), Dims: dims}) }
		for _, m := range []string{"attn", "ffn"} {
			add("hc_"+m+"_norm.weight", hcDim)
			add("hc_"+m+"_down.weight", hcDim, hcLR)
			add("hc_"+m+"_up.weight", hcLR, hcDim)
			add("hc_"+m+"_inject.weight", hcDim, hc)
		}
		if cfg.LayerTypes[l] == "full_attention" {
			if f.hasTensor(b("attn_qkv.weight")) {
				add("attn_qkv.weight", nEmbd, headK*nHead*2+2*kvGQA)
			} else {
				add("attn_q.weight", nEmbd, headK*nHead*2)
				add("attn_k.weight", nEmbd, kvGQA)
				add("attn_v.weight", nEmbd, kvGQA)
			}
			add("attn_output.weight", headK*nHead, nEmbd)
			add("attn_q_norm.weight", headK)
			add("attn_k_norm.weight", headK)
			add("indexer.q_proj.weight", nEmbd, uint64(x.IndexNHeads)*idx)
			add("indexer.k_proj.weight", nEmbd, idx)
			add("indexer.q_norm.weight", idx)
			add("indexer.k_norm.weight", idx)
		} else {
			add("attn_qkv.weight", nEmbd, keyDim*2+valDim)
			add("attn_gate.weight", nEmbd, valDim)
			add("ssm_conv1d.weight", uint64(cfg.LinearConvKernelDim), keyDim*2+valDim)
			add("ssm_dt.bias", nV)
			add("ssm_a", nV)
			add("ssm_beta.weight", nEmbd, nV)
			add("ssm_alpha.weight", nEmbd, nV)
			add("ssm_norm.weight", uint64(cfg.LinearValueHeadDim))
			add("ssm_out.weight", valDim, nEmbd)
		}
		if x.PLE != nil && x.PLE.Layers[0] == l {
			add("ple_key.weight", nEmbd, hcDim)
			add("ple_value.weight", nEmbd, nEmbd)
			add("ple_norm_key.weight", hcDim)
			add("ple_norm_query.weight", hcDim)
			add("ple_norm_conv.weight", hcDim)
			add("ple_conv1d.weight", uint64(x.PLE.ConvKernel), hcDim)
		}
		add("ffn_gate_inp.weight", nEmbd, nExp)
		add("ffn_down_exps.weight", ffExp, nEmbd, nExp)
		if f.hasTensor(b("ffn_gate_up_exps.weight")) {
			add("ffn_gate_up_exps.weight", nEmbd, 2*ffExp, nExp)
		} else {
			add("ffn_gate_exps.weight", nEmbd, ffExp, nExp)
			add("ffn_up_exps.weight", nEmbd, ffExp, nExp)
		}
		add("ffn_gate_inp_shexp.weight", nEmbd)
		add("ffn_gate_shexp.weight", nEmbd, ffSh)
		add("ffn_up_shexp.weight", nEmbd, ffSh)
		add("ffn_down_shexp.weight", ffSh, nEmbd)
	}
	return reqs
}

func tensorByName(f *File, name string) (TensorInfo, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

// qwen4ExpHFBindings ties each qwen4exp key to the f5d0827 HF text_config field
// the converter (conversion/qwen4exp.py, Qwen3.5 base) writes it from.
var qwen4ExpHFBindings = []hfBinding{
	{Key: "block_count", HF: "num_hidden_layers"},
	{Key: "context_length", HF: "max_position_embeddings"},
	{Key: "embedding_length", HF: "hidden_size"},
	{Key: "attention.head_count", HF: "num_attention_heads"},
	{Key: "attention.head_count_kv", HF: "num_key_value_heads"},
	{Key: "attention.key_length", HF: "head_dim"},
	{Key: "attention.value_length", HF: "head_dim"},
	{Key: "attention.layer_norm_rms_epsilon", HF: "rms_norm_eps"},
	{Key: "rope.freq_base", HF: "rope_parameters.rope_theta"},
	{Key: "rope.dimension_count", HF: "head_dim * partial_rotary_factor", Derive: q4xDeriveRopeDim},
	{Key: "rope.dimension_sections", HF: "rope_parameters.mrope_section + [0]", Derive: q4xDeriveRopeSections},
	{Key: "expert_count", HF: "num_experts"},
	{Key: "expert_used_count", HF: "num_experts_per_tok"},
	{Key: "expert_feed_forward_length", HF: "moe_intermediate_size"},
	{Key: "expert_shared_feed_forward_length", HF: "shared_expert_intermediate_size"},
	{Key: "ssm.conv_kernel", HF: "linear_conv_kernel_dim"},
	{Key: "ssm.state_size", HF: "linear_key_head_dim"},
	{Key: "ssm.group_count", HF: "linear_num_key_heads"},
	{Key: "ssm.time_step_rank", HF: "linear_num_value_heads"},
	{Key: "ssm.inner_size", HF: "linear_num_value_heads * linear_value_head_dim", Derive: q4xDeriveSSMInner},
	{Key: "full_attention_interval", HF: "full_attention_interval"},
	{Key: "hyper_connection.count", HF: "hc_count"},
	{Key: "hyper_connection.low_rank", HF: "hc_lowrank"},
	{Key: "attention.indexer.head_count", HF: "indexer_n_heads"},
	{Key: "attention.indexer.key_length", HF: "indexer_head_dim"},
	{Key: "attention.indexer.top_k", HF: "indexer_budget"},
	{Key: "attention.compress_ratios", HF: "layer_types x indexer_compress_ratio", Derive: q4xDeriveCompressRatios},
	{Key: "ple.layers", HF: "ple_layer_ids - 1", Derive: q4xDerivePLELayers},
	{Key: "ple.ngram_size", HF: "ngram_size"},
	{Key: "ple.heads_per_ngram", HF: "heads_per_ngram"},
	{Key: "ple.conv_kernel", HF: "ple_conv_kernel_size"},
	{Key: "ple.eos_token_id", HF: "eos_token_id", Derive: q4xDeriveEOS},
	{Key: "ple.image_token_id", HF: "image_token_id"},
	{Key: "embedding_length_per_layer_input", HF: "ple_embed_dim / ((ngram_size-1)*heads_per_ngram)", Derive: q4xDerivePLERowDim},
}

func hfNum(hf map[string]any, path string) (float64, bool) {
	v, ok := hfLookup(hf, path)
	if !ok {
		return 0, false
	}
	n, ok := v.(float64)
	return n, ok
}

func q4xDeriveRopeDim(hf map[string]any) (any, bool) {
	hd, ok1 := hfNum(hf, "head_dim")
	pr, ok2 := hfNum(hf, "partial_rotary_factor")
	if !ok2 {
		pr, ok2 = hfNum(hf, "rope_parameters.partial_rotary_factor")
	}
	return hd * pr, ok1 && ok2
}

func q4xDeriveRopeSections(hf map[string]any) (any, bool) {
	v, ok := hfLookup(hf, "rope_parameters.mrope_section")
	s, isArr := v.([]any)
	if !ok || !isArr {
		return nil, false
	}
	out := append([]any(nil), s...)
	for len(out) < 4 {
		out = append(out, float64(0))
	}
	return out, true
}

func q4xDeriveSSMInner(hf map[string]any) (any, bool) {
	n, ok1 := hfNum(hf, "linear_num_value_heads")
	d, ok2 := hfNum(hf, "linear_value_head_dim")
	return n * d, ok1 && ok2
}

func q4xDeriveCompressRatios(hf map[string]any) (any, bool) {
	lt, ok1 := hfLookup(hf, "layer_types")
	ratio, ok2 := hfNum(hf, "indexer_compress_ratio")
	types, isArr := lt.([]any)
	if !ok1 || !ok2 || !isArr {
		return nil, false
	}
	out := make([]any, len(types))
	for i, t := range types {
		out[i] = float64(0)
		if t == "full_attention" {
			out[i] = ratio
		}
	}
	return out, true
}

func q4xDerivePLELayers(hf map[string]any) (any, bool) {
	v, ok := hfLookup(hf, "ple_layer_ids")
	ids, isArr := v.([]any)
	if !ok || !isArr {
		return nil, false
	}
	out := make([]any, len(ids))
	for i, id := range ids {
		n, ok := id.(float64)
		if !ok {
			return nil, false
		}
		out[i] = n - 1
	}
	return out, true
}

func q4xDeriveEOS(hf map[string]any) (any, bool) {
	v, ok := hfLookup(hf, "eos_token_id")
	if a, isArr := v.([]any); ok && isArr && len(a) > 0 {
		return a[len(a)-1], true
	}
	return v, ok
}

func q4xDerivePLERowDim(hf map[string]any) (any, bool) {
	dim, ok1 := hfNum(hf, "ple_embed_dim")
	ng, ok2 := hfNum(hf, "ngram_size")
	hp, ok3 := hfNum(hf, "heads_per_ngram")
	if !ok1 || !ok2 || !ok3 || ng < 2 || hp <= 0 {
		return nil, false
	}
	return dim / ((ng - 1) * hp), true
}

// qwen4ExpCanonicalSuffix maps the qwen4exp-only per-layer tensors. The router,
// shared expert and indexer take their HF names (Qwen/Qwen3.8-Flash-Next@f5d08274;
// the converter splits HF index_qk_proj into q/k). Hyper-connection and PLE
// tensors keep their dff6004 stage names: no HF name for them is pinned yet.
func qwen4ExpCanonicalSuffix(suffix string) (string, bool) {
	switch suffix {
	case "ffn_gate_inp.weight":
		return "mlp.gate.weight", true
	case "ffn_gate_inp_shexp.weight":
		return "mlp.shared_expert_gate.weight", true
	case "ffn_gate_shexp.weight":
		return "mlp.shared_expert.gate_proj.weight", true
	case "ffn_up_shexp.weight":
		return "mlp.shared_expert.up_proj.weight", true
	case "ffn_down_shexp.weight":
		return "mlp.shared_expert.down_proj.weight", true
	case "indexer.q_proj.weight":
		return "self_attn.indexer.q_proj.weight", true
	case "indexer.k_proj.weight":
		return "self_attn.indexer.k_proj.weight", true
	case "indexer.q_norm.weight":
		return "self_attn.indexer.q_layernorm.weight", true
	case "indexer.k_norm.weight":
		return "self_attn.indexer.k_layernorm.weight", true
	}
	for _, m := range []string{"attn", "ffn"} {
		for _, part := range []string{"norm", "down", "up", "inject"} {
			if suffix == "hc_"+m+"_"+part+".weight" {
				return "hc_" + m + "." + part + ".weight", true
			}
		}
	}
	for _, part := range []string{"key", "value", "norm_key", "norm_query", "norm_conv", "conv1d"} {
		if suffix == "ple_"+part+".weight" {
			return "ple." + part + ".weight", true
		}
	}
	return "", false
}

var qwen4ExpGlobalTensorNames = map[string]string{
	"output_hc_norm.weight":       "model.output_hc.norm.weight",
	"output_hc_down.weight":       "model.output_hc.down.weight",
	"output_hc_up.weight":         "model.output_hc.up.weight",
	"per_layer_token_embd.weight": "model.ple_embedding.weight",
}
