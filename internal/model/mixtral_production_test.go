package model

import (
	"math"
	"strings"
	"testing"
)

const mixtralOracleDir = ".cache/oracle-mixtral"
const mixtralOracleModel = "yujiepan/mixtral-tiny-random"
const mixtralOracleExportHint = "from fak/: python internal/model/export_oracle.py --online --model " +
	mixtralOracleModel + " --out internal/model/" + mixtralOracleDir

// TestConfigDerivesMixtralProductionCheckpoints is the weight-free production witness
// for #296: the public Mixtral 8x7B and 8x22B config magnitudes derive the MoE axes the
// loader and router need, without requiring 90GB+ of checkpoint weights on the build box.
func TestConfigDerivesMixtralProductionCheckpoints(t *testing.T) {
	tests := []struct {
		name      string
		js        string
		hidden    int
		layers    int
		heads     int
		kvHeads   int
		inter     int
		context   int
		vocab     int
		groupSize int
	}{
		{
			name: "8x7b",
			js: `{
				"architectures": ["MixtralForCausalLM"],
				"bos_token_id": 1,
				"eos_token_id": 2,
				"hidden_act": "silu",
				"hidden_size": 4096,
				"intermediate_size": 14336,
				"max_position_embeddings": 32768,
				"model_type": "mixtral",
				"num_attention_heads": 32,
				"num_experts_per_tok": 2,
				"num_hidden_layers": 32,
				"num_key_value_heads": 8,
				"num_local_experts": 8,
				"rms_norm_eps": 1e-05,
				"rope_theta": 1000000.0,
				"sliding_window": null,
				"tie_word_embeddings": false,
				"vocab_size": 32000
			}`,
			hidden: 4096, layers: 32, heads: 32, kvHeads: 8, inter: 14336,
			context: 32768, vocab: 32000, groupSize: 4,
		},
		{
			name: "8x22b",
			js: `{
				"architectures": ["MixtralForCausalLM"],
				"bos_token_id": 1,
				"eos_token_id": 2,
				"hidden_act": "silu",
				"hidden_size": 6144,
				"intermediate_size": 16384,
				"max_position_embeddings": 65536,
				"model_type": "mixtral",
				"num_attention_heads": 48,
				"num_experts_per_tok": 2,
				"num_hidden_layers": 56,
				"num_key_value_heads": 8,
				"num_local_experts": 8,
				"rms_norm_eps": 1e-05,
				"rope_theta": 1000000,
				"sliding_window": null,
				"tie_word_embeddings": false,
				"vocab_size": 32000
			}`,
			hidden: 6144, layers: 56, heads: 48, kvHeads: 8, inter: 16384,
			context: 65536, vocab: 32000, groupSize: 6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := decodeProdConfig(t, tt.js)
			if !strings.Contains(cfg.archFamilyKey(), "mixtral") {
				t.Fatalf("family = %q, want mixtral", cfg.archFamilyKey())
			}
			if cfg.HiddenSize != tt.hidden || cfg.NumLayers != tt.layers || cfg.NumHeads != tt.heads ||
				cfg.NumKVHeads != tt.kvHeads || cfg.IntermediateSize != tt.inter {
				t.Fatalf("dims = H:%d L:%d heads:%d kv:%d I:%d, want H:%d L:%d heads:%d kv:%d I:%d",
					cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.IntermediateSize,
					tt.hidden, tt.layers, tt.heads, tt.kvHeads, tt.inter)
			}
			if cfg.HeadDim != 128 {
				t.Fatalf("HeadDim = %d, want 128 derived from hidden/heads", cfg.HeadDim)
			}
			if cfg.GroupSize() != tt.groupSize {
				t.Fatalf("GQA group size = %d, want %d", cfg.GroupSize(), tt.groupSize)
			}
			if !cfg.IsMoE() || cfg.NumExperts != 8 || cfg.NumExpertsPerTok != 2 {
				t.Fatalf("MoE fields = experts:%d topk:%d IsMoE:%v, want 8/2/true",
					cfg.NumExperts, cfg.NumExpertsPerTok, cfg.IsMoE())
			}
			if !cfg.NormTopKProb {
				t.Fatal("Mixtral must normalize selected top-2 router weights to match HF")
			}
			if cfg.activationName() != "silu" || cfg.DenseMLP || cfg.ActGeluTanh || cfg.ActGeluErf {
				t.Fatalf("activation axes = name:%q dense:%v tanh:%v erf:%v, want SiLU SwiGLU",
					cfg.activationName(), cfg.DenseMLP, cfg.ActGeluTanh, cfg.ActGeluErf)
			}
			if cfg.windowForLayer(0) != -1 || cfg.windowForLayer(tt.layers-1) != -1 {
				t.Fatalf("sliding_window=null derived Window=%v, want full causal attention", cfg.Window)
			}
			if cfg.RopeTheta != 1000000 || cfg.MaxPositionEmbeddings != tt.context || cfg.VocabSize != tt.vocab {
				t.Fatalf("rope/context/vocab = %v/%d/%d, want 1000000/%d/%d",
					cfg.RopeTheta, cfg.MaxPositionEmbeddings, cfg.VocabSize, tt.context, tt.vocab)
			}
			if cfg.TieWordEmbeddings {
				t.Fatal("tie_word_embeddings=true, want false for production Mixtral checkpoints")
			}
			assertMixtralProductionCanonicalShapes(t, cfg, tt.hidden, tt.inter, tt.kvHeads*128)
			assertFullMixtralCheckpointResolves(t, cfg, tt.layers)
		})
	}
}

func TestMixtralConfigNormalizesTop2RoutingLikeHF(t *testing.T) {
	cfg := decodeProdConfig(t, `{
		"architectures": ["MixtralForCausalLM"],
		"hidden_act": "silu",
		"hidden_size": 2,
		"intermediate_size": 2,
		"model_type": "mixtral",
		"num_attention_heads": 1,
		"num_experts_per_tok": 2,
		"num_hidden_layers": 1,
		"num_key_value_heads": 1,
		"num_local_experts": 4,
		"vocab_size": 8
	}`)
	if !cfg.NormTopKProb {
		t.Fatal("Mixtral config should derive NormTopKProb=true")
	}
	m, err := NewFromF32Tensors(cfg, []NamedTensorF32{
		{Name: routerName(0), Shape: []int{4, 2}, Data: []float32{
			2, 0,
			0, 3,
			1, 1,
			-1, -1,
		}},
	})
	if err != nil {
		t.Fatalf("NewFromF32Tensors: %v", err)
	}
	picks := route(m, 0, []float32{1, 1}, f32Kernel{m})
	if len(picks) != 2 || picks[0].expert != 1 || picks[1].expert != 0 {
		t.Fatalf("Mixtral top-2 picks = %+v, want experts [1 0]", picks)
	}
	sum := picks[0].weight + picks[1].weight
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("Mixtral selected routing weights sum to %v, want 1", sum)
	}
}

func TestMixtralConfigPreservesExplicitTopKNormalization(t *testing.T) {
	cfg := decodeProdConfig(t, `{
		"architectures": ["MixtralForCausalLM"],
		"hidden_size": 2,
		"intermediate_size": 2,
		"model_type": "mixtral",
		"norm_topk_prob": false,
		"num_attention_heads": 1,
		"num_experts_per_tok": 2,
		"num_hidden_layers": 1,
		"num_key_value_heads": 1,
		"num_local_experts": 4,
		"vocab_size": 8
	}`)
	if cfg.NormTopKProb {
		t.Fatal("explicit norm_topk_prob=false was overwritten")
	}
}

// TestOptionalMixtralOracleForwardMatchesHF is the weight-backed argmax-exact gate for
// #296. It skips cleanly until the tiny Mixtral oracle is exported; when present, HF
// authors the hidden/logit/greedy oracle and the Go Mixtral path must match it.
func TestOptionalMixtralOracleForwardMatchesHF(t *testing.T) {
	m, doc := loadFixtureDir(t, mixtralOracleDir, true)
	cfg := m.Cfg
	if !strings.Contains(cfg.archFamilyKey(), "mixtral") {
		t.Fatalf("%s family = %q, want mixtral; regenerate: %s",
			mixtralOracleDir, cfg.archFamilyKey(), mixtralOracleExportHint)
	}
	if !cfg.IsMoE() || cfg.NumExperts != 8 || cfg.NumExpertsPerTok != 2 || !cfg.NormTopKProb {
		t.Fatalf("%s Mixtral MoE axes = experts:%d topk:%d norm:%v; regenerate: %s",
			mixtralOracleDir, cfg.NumExperts, cfg.NumExpertsPerTok, cfg.NormTopKProb, mixtralOracleExportHint)
	}
	if _, ok := m.ffnForLayer(0).(moeFFN); !ok {
		t.Fatalf("%s layer 0 selected %T, want moeFFN", mixtralOracleDir, m.ffnForLayer(0))
	}
	for _, name := range []string{
		routerName(0),
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 0, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"),
	} {
		if !m.has(name) {
			t.Fatalf("%s missing canonical Mixtral tensor %s", mixtralOracleDir, name)
		}
	}
	resolved, _ := resolveOracleDir(mixtralOracleDir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func assertMixtralProductionCanonicalShapes(t *testing.T, cfg Config, hidden, inter, kvRows int) {
	t.Helper()
	expect := []struct {
		name string
		got  []int
		want []int
	}{
		{layerName(0, "self_attn.q_proj.weight"), []int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize}, []int{hidden, hidden}},
		{layerName(0, "self_attn.k_proj.weight"), []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize}, []int{kvRows, hidden}},
		{layerName(0, "self_attn.v_proj.weight"), []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize}, []int{kvRows, hidden}},
		{layerName(0, "self_attn.o_proj.weight"), []int{cfg.HiddenSize, cfg.NumHeads * cfg.HeadDim}, []int{hidden, hidden}},
		{routerName(0), []int{cfg.NumExperts, cfg.HiddenSize}, []int{8, hidden}},
		{expertName(0, 0, "gate_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize}, []int{inter, hidden}},
		{expertName(0, 0, "up_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize}, []int{inter, hidden}},
		{expertName(0, 0, "down_proj.weight"), []int{cfg.HiddenSize, cfg.IntermediateSize}, []int{hidden, inter}},
	}
	for _, tt := range expect {
		if !sameShape(tt.got, tt.want) {
			t.Fatalf("%s derived shape = %v, want %v", tt.name, tt.got, tt.want)
		}
	}
}

func fullMixtralBlockSparseManifestNames(numLayers, numExperts int) []string {
	names := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"lm_head.weight",
	}
	for l := 0; l < numLayers; l++ {
		p := layerPrefix(l)
		block := p + "block_sparse_moe."
		names = append(names,
			p+"input_layernorm.weight",
			p+"self_attn.q_proj.weight",
			p+"self_attn.k_proj.weight",
			p+"self_attn.v_proj.weight",
			p+"self_attn.o_proj.weight",
			p+"post_attention_layernorm.weight",
			block+"gate.weight",
		)
		for e := 0; e < numExperts; e++ {
			expert := block + "experts." + itoa(e) + "."
			names = append(names,
				expert+"w1.weight",
				expert+"w2.weight",
				expert+"w3.weight",
			)
		}
	}
	return names
}

func assertFullMixtralCheckpointResolves(t *testing.T, cfg Config, numLayers int) {
	t.Helper()
	if cfg.NumLayers != numLayers {
		t.Fatalf("config NumLayers = %d, want production depth %d", cfg.NumLayers, numLayers)
	}
	man := manifestKeys(fullMixtralBlockSparseManifestNames(numLayers, cfg.NumExperts)...)
	res, err := ResolveTensorNames(cfg, man)
	if err != nil {
		t.Fatalf("full %d-layer Mixtral checkpoint must resolve every tensor: %v", numLayers, err)
	}
	if res.Family != "mixtral" {
		t.Fatalf("family = %q, want mixtral", res.Family)
	}
	want := 3 + numLayers*(7+cfg.NumExperts*3)
	if len(res.Resolved) != want {
		t.Fatalf("resolved %d tensors for %d layers, want %d", len(res.Resolved), numLayers, want)
	}

	deep := numLayers - 1
	deepBlock := layerName(deep, "block_sparse_moe.")
	if got := res.SourceFor(routerName(deep)); got != deepBlock+"gate.weight" {
		t.Fatalf("deepest Mixtral router resolved to %q, want %q", got, deepBlock+"gate.weight")
	}
	expert := cfg.NumExperts - 1
	for _, tt := range []struct {
		canonical string
		source    string
	}{
		{expertName(deep, expert, "gate_proj.weight"), deepBlock + "experts." + itoa(expert) + ".w1.weight"},
		{expertName(deep, expert, "down_proj.weight"), deepBlock + "experts." + itoa(expert) + ".w2.weight"},
		{expertName(deep, expert, "up_proj.weight"), deepBlock + "experts." + itoa(expert) + ".w3.weight"},
	} {
		if got := res.SourceFor(tt.canonical); got != tt.source {
			t.Fatalf("deepest Mixtral tensor %q resolved to %q, want %q", tt.canonical, got, tt.source)
		}
	}

	missing := deepBlock + "experts." + itoa(expert) + ".w2.weight"
	delete(man, missing)
	assertResolveError(t, cfg, man, "mixtral family", expertName(deep, expert, "down_proj.weight"), missing, "searched:")
}
