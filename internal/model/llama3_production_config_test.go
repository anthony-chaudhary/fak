package model

import (
	"encoding/json"
	"strings"
	"testing"
)

// llama3_production_config_test.go — the WEIGHT-FREE structural witness for #298
// ("Production Llama 3.x Checkpoints [A-004]"): prove the loader derives the real
// Llama-3.1 8B-Instruct and 70B config shapes correctly — "not just the tiny oracle" —
// and that the 70B GQA geometry shards cleanly across multi-GPU rank counts.
//
// What this covers and what it deliberately does NOT:
//
//   - COVERED here (deterministic, no 16GB/140GB weights, no GPU): the config-derivation
//     path (UnmarshalJSON -> deriveConfigAxes — HeadDim/NumKVHeads derivation, nested
//     rope_scaling flattening, the scalar-or-list eos_token_id) over the PRODUCTION
//     config magnitudes, plus the tensor-parallel shard-plan derivation for the 70B's
//     8 KV-head groups across 1/2/4/8 devices (NewTPPlan, the GPU-free half of
//     "Multi-GPU sharding for 70B").
//   - NOT covered here (host/weight-gated, can't be faked): byte-for-byte forward parity
//     on the real checkpoints, argmax-exactness on a real run, and the within-2x-llama.cpp
//     throughput bar. Those stay in the weight-gated oracle witness
//     (TestOptionalLlama3OracleCoversScalingAndEOSList) and the GPU benchmark lane.
//
// The config JSON below mirrors the canonical published
// meta-llama/Llama-3.1-{8B-Instruct,70B} config.json fields; head_dim is omitted so the
// hidden/heads derivation (4096/32 and 8192/64 == 128) is exercised, not asserted from
// input.

// decodeProdConfig unmarshals a production HF config.json into a Config, exercising the
// real UnmarshalJSON -> deriveConfigAxes path. It is the structural analog of loading a
// checkpoint's config without its tensors.
func decodeProdConfig(t *testing.T, js string) Config {
	t.Helper()
	var cfg Config
	if err := json.Unmarshal([]byte(js), &cfg); err != nil {
		t.Fatalf("unmarshal production config: %v", err)
	}
	return cfg
}

// assertTilesDim checks that a plan's shards tile [0,Dim) contiguously and completely —
// the defining property of a correct multi-GPU partition (no KV-head group dropped, none
// double-owned).
func assertTilesDim(t *testing.T, p TPPlan, dim int) {
	t.Helper()
	if p.Dim != dim {
		t.Fatalf("plan Dim = %d, want %d", p.Dim, dim)
	}
	if err := p.Validate(); err != nil {
		t.Fatalf("plan does not validate: %v", err)
	}
	if p.Shards[0].Lo != 0 {
		t.Fatalf("first shard Lo = %d, want 0", p.Shards[0].Lo)
	}
	if last := p.Shards[len(p.Shards)-1]; last.Hi != dim {
		t.Fatalf("last shard Hi = %d, want %d", last.Hi, dim)
	}
	covered := 0
	for i, s := range p.Shards {
		if i > 0 && s.Lo != p.Shards[i-1].Hi {
			t.Fatalf("shard %d Lo = %d, want previous Hi %d (gap/overlap)", i, s.Lo, p.Shards[i-1].Hi)
		}
		covered += s.Width()
	}
	if covered != dim {
		t.Fatalf("shards cover %d indices, want %d (whole dimension)", covered, dim)
	}
}

// TestLlama31_8BInstructConfigDerivesProductionAxes proves the loader maps a real
// Llama-3.1-8B-Instruct config.json to the correct architecture axes — the structural
// "Load Llama 3.1 8B Instruct" rung of #298, at config granularity. The production
// magnitudes (4096 hidden, 128256 vocab) are asserted directly so this is provably NOT
// the 135M SmolLM2 tiny oracle (576 hidden / 49152 vocab).
func TestLlama31_8BInstructConfigDerivesProductionAxes(t *testing.T) {
	cfg := decodeProdConfig(t, `{
		"architectures": ["LlamaForCausalLM"],
		"model_type": "llama",
		"hidden_size": 4096,
		"num_hidden_layers": 32,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size": 14336,
		"vocab_size": 128256,
		"max_position_embeddings": 131072,
		"rms_norm_eps": 1e-05,
		"rope_theta": 500000.0,
		"bos_token_id": 128000,
		"eos_token_id": [128001, 128008, 128009],
		"tie_word_embeddings": false,
		"rope_scaling": {
			"factor": 8.0,
			"low_freq_factor": 1.0,
			"high_freq_factor": 4.0,
			"original_max_position_embeddings": 8192,
			"rope_type": "llama3"
		}
	}`)

	if !strings.Contains(cfg.archFamilyKey(), "llama") {
		t.Fatalf("family = %q, want llama", cfg.archFamilyKey())
	}
	// Core decoder geometry (the "all tensors" shape: every projection size is fixed by
	// these axes).
	if cfg.NumLayers != 32 || cfg.HiddenSize != 4096 || cfg.NumHeads != 32 || cfg.IntermediateSize != 14336 {
		t.Fatalf("8B geometry = layers:%d hidden:%d heads:%d inter:%d, want 32/4096/32/14336",
			cfg.NumLayers, cfg.HiddenSize, cfg.NumHeads, cfg.IntermediateSize)
	}
	if cfg.VocabSize != 128256 {
		t.Fatalf("8B vocab = %d, want 128256 (the Llama-3.x tokenizer)", cfg.VocabSize)
	}
	// GQA: 8 KV heads, 32 query heads -> group size 4. HeadDim derived 4096/32 = 128.
	if cfg.NumKVHeads != 8 {
		t.Fatalf("8B num_key_value_heads = %d, want 8 (GQA)", cfg.NumKVHeads)
	}
	if cfg.NumHeads%cfg.NumKVHeads != 0 {
		t.Fatalf("8B GQA broken: %d query heads not a multiple of %d KV heads", cfg.NumHeads, cfg.NumKVHeads)
	}
	if cfg.HeadDim != 128 {
		t.Fatalf("8B derived head_dim = %d, want 128 (4096/32)", cfg.HeadDim)
	}
	// Llama-3 long-context RoPE rescale, flattened from the nested rope_scaling object.
	if cfg.RopeTheta != 500000 {
		t.Fatalf("8B rope_theta = %v, want 500000", cfg.RopeTheta)
	}
	if cfg.RopeScaling != "llama3" || cfg.RopeFactor != 8 || cfg.RopeOrigContext != 8192 ||
		cfg.RopeLowFreqFactor != 1 || cfg.RopeHighFreqFactor != 4 {
		t.Fatalf("8B llama3 rope = type:%q factor:%v orig:%d low:%v high:%v, want llama3/8/8192/1/4",
			cfg.RopeScaling, cfg.RopeFactor, cfg.RopeOrigContext, cfg.RopeLowFreqFactor, cfg.RopeHighFreqFactor)
	}
	// Llama-3.x emits eos_token_id as a LIST; the Instruct turn terminator <|eot_id|>
	// (128009) must survive the scalar-or-list parse or generation never stops on a turn.
	if len(cfg.EOSTokenIDs) < 2 {
		t.Fatalf("8B-Instruct EOS ids = %v, want a Llama-3 EOS list", cfg.EOSTokenIDs)
	}
	foundEOT := false
	for _, id := range cfg.EOSTokenIDs {
		if id == 128009 {
			foundEOT = true
		}
	}
	if !foundEOT {
		t.Fatalf("8B-Instruct EOS ids = %v, missing <|eot_id|> 128009", cfg.EOSTokenIDs)
	}
	// Provably not the tiny oracle.
	if cfg.HiddenSize <= 576 || cfg.VocabSize <= 49152 {
		t.Fatalf("production magnitudes collapsed to tiny-oracle scale: hidden=%d vocab=%d", cfg.HiddenSize, cfg.VocabSize)
	}
}

// TestLlama31_70BConfigShardsAcrossMultiGPU proves two things for the 70B: the loader
// derives its production axes, and its 8 KV-head groups shard cleanly across the GPU
// counts a 70B is actually run on (1/2/4/8) — the GPU-free, deterministic half of
// "Multi-GPU sharding for 70B" in #298. The tensor-parallel attention path shards whole
// KV-head groups (TensorParallelAttention requires plan.Dim == nKV), so the partitioned
// dimension here is NumKVHeads, not hidden.
func TestLlama31_70BConfigShardsAcrossMultiGPU(t *testing.T) {
	cfg := decodeProdConfig(t, `{
		"architectures": ["LlamaForCausalLM"],
		"model_type": "llama",
		"hidden_size": 8192,
		"num_hidden_layers": 80,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"intermediate_size": 28672,
		"vocab_size": 128256,
		"max_position_embeddings": 131072,
		"rms_norm_eps": 1e-05,
		"rope_theta": 500000.0,
		"eos_token_id": [128001, 128008, 128009],
		"rope_scaling": {
			"factor": 8.0,
			"low_freq_factor": 1.0,
			"high_freq_factor": 4.0,
			"original_max_position_embeddings": 8192,
			"rope_type": "llama3"
		}
	}`)

	if cfg.NumLayers != 80 || cfg.HiddenSize != 8192 || cfg.NumHeads != 64 ||
		cfg.NumKVHeads != 8 || cfg.IntermediateSize != 28672 {
		t.Fatalf("70B geometry = layers:%d hidden:%d heads:%d kv:%d inter:%d, want 80/8192/64/8/28672",
			cfg.NumLayers, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize)
	}
	if cfg.HeadDim != 128 {
		t.Fatalf("70B derived head_dim = %d, want 128 (8192/64)", cfg.HeadDim)
	}
	if cfg.RopeScaling != "llama3" || cfg.RopeFactor != 8 || cfg.RopeOrigContext != 8192 {
		t.Fatalf("70B llama3 rope = type:%q factor:%v orig:%d, want llama3/8/8192",
			cfg.RopeScaling, cfg.RopeFactor, cfg.RopeOrigContext)
	}

	// Multi-GPU tensor-parallel sharding of the 8 KV-head groups. The GPU counts that
	// divide 8 evenly keep whole KV-head groups (and their GQA query-head groups) intact
	// on each device.
	nKV := cfg.NumKVHeads
	for _, ranks := range []int{1, 2, 4, 8} {
		plan, err := NewTPPlan(nKV, ranks)
		if err != nil {
			t.Fatalf("70B NewTPPlan(nKV=%d, ranks=%d): %v", nKV, ranks, err)
		}
		assertTilesDim(t, plan, nKV)
		if len(plan.Shards) != ranks {
			t.Fatalf("70B ranks=%d produced %d shards, want %d", ranks, len(plan.Shards), ranks)
		}
		// Even division -> each device owns nKV/ranks KV groups and (NumHeads/ranks) query
		// heads, preserving the GQA grouping (query heads per KV group stays an integer).
		if nKV%ranks == 0 {
			kvPerRank := nKV / ranks
			if cfg.NumHeads%ranks != 0 {
				t.Fatalf("70B ranks=%d splits %d query heads unevenly", ranks, cfg.NumHeads)
			}
			qPerRank := cfg.NumHeads / ranks
			groupSize := cfg.NumHeads / nKV // 64/8 = 8 query heads per KV group
			if qPerRank != kvPerRank*groupSize {
				t.Fatalf("70B ranks=%d: %d query heads/rank != %d KV groups * %d group size (GQA grouping broken)",
					ranks, qPerRank, kvPerRank, groupSize)
			}
		}
	}

	// Fail closed: 8 KV-head groups cannot be spread over 16 GPUs without splitting a
	// group, so the plan refuses rather than emit a degenerate (empty-shard) tiling.
	if _, err := NewTPPlan(nKV, 16); err == nil {
		t.Fatalf("70B NewTPPlan(nKV=%d, ranks=16) should fail closed (more ranks than KV-head groups)", nKV)
	}
}
