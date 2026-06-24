package ggufload

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// writeKVBool writes a scalar GGUF boolean metadata entry (TypeBool, one byte).
// GLM-5.2 encodes expert_weights_norm this way; no other test fixture needed one
// yet, so the helper lives next to the test that exercises it.
func writeKVBool(b *bytes.Buffer, key string, value bool) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeBool))
	var x byte
	if value {
		x = 1
	}
	b.WriteByte(x)
}

// TestGLMMoeDsaConfig is the Pillar-1 first-slice golden: a glm_moe_dsa GGUF
// header (MoE + MLA + DSA-indexer metadata, no tensors) round-trips through
// (*File).Config() into the same model.Config the JSON/safetensors loader
// already produces for GLM-5.2 (config_test.go TestConfigDerives...). It asserts
// the family detection fires (isGLMMoeDsa/IsMoE) and every MoE/MLA/DSA scalar
// equals what was written — both with explicit head-dim keys and via the
// deepseek2 attention.key_length/value_length derivation fallback.
func TestGLMMoeDsaConfig(t *testing.T) {
	type kvWriter struct {
		buf bytes.Buffer
		n   uint64
	}
	ks := func(w *kvWriter, k, v string) { writeKVString(&w.buf, k, v); w.n++ }
	ku := func(w *kvWriter, k string, v uint32) { writeKVUint32(&w.buf, k, v); w.n++ }
	kf := func(w *kvWriter, k string, v float32) { writeKVFloat32(&w.buf, k, v); w.n++ }
	kb := func(w *kvWriter, k string, v bool) { writeKVBool(&w.buf, k, v); w.n++ }
	ka := func(w *kvWriter, k string, v []string) { writeKVStringArray(&w.buf, k, v); w.n++ }

	// generic + MoE + indexer metadata shared by both sub-cases. The per-case
	// closure then appends the head-dim metadata under test.
	base := func(w *kvWriter) {
		ks(w, "general.architecture", "glm_moe_dsa")
		ku(w, "general.alignment", 32)
		ku(w, "glm_moe_dsa.embedding_length", 32)
		ku(w, "glm_moe_dsa.block_count", 4)
		ku(w, "glm_moe_dsa.attention.head_count", 4)
		ku(w, "glm_moe_dsa.attention.head_count_kv", 2)
		ku(w, "glm_moe_dsa.feed_forward_length", 64)
		kf(w, "glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)
		kf(w, "glm_moe_dsa.rope.freq_base", 10000)
		// MoE FFN axis.
		ku(w, "glm_moe_dsa.expert_count", 4)
		ku(w, "glm_moe_dsa.expert_used_count", 2)
		ku(w, "glm_moe_dsa.expert_feed_forward_length", 48)
		ku(w, "glm_moe_dsa.expert_shared_count", 1)
		ku(w, "glm_moe_dsa.expert_shared_feed_forward_length", 48)
		ku(w, "glm_moe_dsa.leading_dense_block_count", 1)
		ku(w, "glm_moe_dsa.expert_group_count", 2)
		ku(w, "glm_moe_dsa.expert_group_used_count", 1)
		kf(w, "glm_moe_dsa.expert_weights_scale", 2.5)
		kb(w, "glm_moe_dsa.expert_weights_norm", true)
		// MLA latent-projection ranks.
		ku(w, "glm_moe_dsa.attention.q_lora_rank", 24)
		ku(w, "glm_moe_dsa.attention.kv_lora_rank", 16)
		// DSA learned-indexer axis.
		ku(w, "glm_moe_dsa.index_n_heads", 4)
		ku(w, "glm_moe_dsa.index_head_dim", 16)
		ku(w, "glm_moe_dsa.index_topk", 8)
		ka(w, "glm_moe_dsa.indexer_types", []string{"full", "shared"})
	}

	cases := []struct {
		name    string
		headDim func(w *kvWriter)
	}{
		{
			name: "explicit_head_dims",
			headDim: func(w *kvWriter) {
				ku(w, "glm_moe_dsa.attention.qk_nope_head_dim", 8)
				ku(w, "glm_moe_dsa.attention.qk_rope_head_dim", 4)
				ku(w, "glm_moe_dsa.attention.v_head_dim", 8)
			},
		},
		{
			// deepseek2 convention: n_embd_head_k (= qk_nope+qk_rope) under
			// attention.key_length, the rotary portion under rope.dimension_count,
			// and v_head_dim under attention.value_length. No explicit qk_* keys.
			name: "deepseek2_derived",
			headDim: func(w *kvWriter) {
				ku(w, "glm_moe_dsa.attention.key_length", 12)
				ku(w, "glm_moe_dsa.attention.value_length", 8)
				ku(w, "glm_moe_dsa.rope.dimension_count", 4)
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var w kvWriter
			base(&w)
			tc.headDim(&w)

			var b bytes.Buffer
			writeMinimalHeader(&b, 0, w.n)
			b.Write(w.buf.Bytes())

			gg, err := Read(bytes.NewReader(b.Bytes()))
			if err != nil {
				t.Fatalf("Read: %v", err)
			}
			cfg, err := gg.Config()
			if err != nil {
				t.Fatalf("Config: %v", err)
			}

			if cfg.ModelType != "glm_moe_dsa" {
				t.Fatalf("ModelType=%q, want glm_moe_dsa", cfg.ModelType)
			}
			if !cfg.IsMoE() {
				t.Fatalf("IsMoE()=false, want true (NumExperts=%d)", cfg.NumExperts)
			}

			// MoE FFN axis.
			if cfg.NumExperts != 4 || cfg.NumExpertsPerTok != 2 {
				t.Fatalf("MoE counts = experts:%d topk:%d, want 4/2", cfg.NumExperts, cfg.NumExpertsPerTok)
			}
			if cfg.MoEIntermediateSize != 48 || cfg.SharedIntermediateSize != 48 {
				t.Fatalf("MoE ffn = moe:%d shared:%d, want 48/48", cfg.MoEIntermediateSize, cfg.SharedIntermediateSize)
			}
			if cfg.NSharedExperts != 1 || cfg.FirstKDenseReplace != 1 {
				t.Fatalf("MoE structure = shared:%d firstKDense:%d, want 1/1", cfg.NSharedExperts, cfg.FirstKDenseReplace)
			}
			if cfg.NGroup != 2 || cfg.TopKGroup != 1 || cfg.RoutedScalingFactor != 2.5 {
				t.Fatalf("MoE routing = n_group:%d topk_group:%d scale:%v, want 2/1/2.5",
					cfg.NGroup, cfg.TopKGroup, cfg.RoutedScalingFactor)
			}
			if !cfg.NormTopKProb {
				t.Fatalf("NormTopKProb=false, want true (expert_weights_norm)")
			}

			// MLA latent-projection axis.
			if cfg.QLoraRank != 24 || cfg.KVLoraRank != 16 {
				t.Fatalf("MLA ranks = q:%d kv:%d, want 24/16", cfg.QLoraRank, cfg.KVLoraRank)
			}
			if cfg.QKNopeHeadDim != 8 || cfg.QKRopeHeadDim != 4 || cfg.VHeadDim != 8 {
				t.Fatalf("MLA head dims = nope:%d rope:%d v:%d, want 8/4/8",
					cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim)
			}

			// DSA learned-indexer axis.
			if cfg.IndexNHeads != 4 || cfg.IndexHeadDim != 16 || cfg.IndexTopK != 8 {
				t.Fatalf("DSA indexer = heads:%d dim:%d topk:%d, want 4/16/8",
					cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexTopK)
			}
			if len(cfg.IndexerTypes) != 2 || cfg.IndexerTypes[0] != "full" || cfg.IndexerTypes[1] != "shared" {
				t.Fatalf("IndexerTypes=%v, want [full shared]", cfg.IndexerTypes)
			}

			// The native GLM-DSA path is gated by cfg.isGLMMoeDsa(), which is a pure
			// function of ModelType ("glm_moe_dsa" -> family key "glmmoedsa" contains
			// both "glm" and "dsa"; proven in model/config_test.go). Asserting
			// ModelType (above) + IsMoE() here therefore proves, from the ggufload
			// package boundary, that the GGUF routes to the same DSA forward the JSON
			// loader does — without reaching into the model package's unexported helper.
		})
	}
}
