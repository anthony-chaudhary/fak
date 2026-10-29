package ggufload

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41HFOracle mirrors the published DeepSeek-V4.1-Flash HF config the Q2_K
// artifact was quantized from (internal/model/testdata/deepseek_v41_flash_config.json).
// Every expected value in TestDeepSeek41PublishedHeaderAdmission comes from this
// oracle, never from a constant transcribed into the test.
type v41HFOracle struct {
	BOSTokenID int `json:"bos_token_id"`
	EOSTokenID int `json:"eos_token_id"`
	Quantization struct {
		QuantMethod     string `json:"quant_method"`
		Activation      string `json:"activation_scheme"`
		WeightBlockSize []int  `json:"weight_block_size"`
		ScaleFmt        string `json:"scale_fmt"`
		ExpertDtype     string `json:"expert_dtype"`
	} `json:"quantization_config"`
	Text struct {
		VocabSize              int     `json:"vocab_size"`
		HiddenSize             int     `json:"hidden_size"`
		MoEIntermediateSize    int     `json:"moe_intermediate_size"`
		NumHiddenLayers        int     `json:"num_hidden_layers"`
		NumAttentionHeads      int     `json:"num_attention_heads"`
		HeadDim                int     `json:"head_dim"`
		QKRopeHeadDim          int     `json:"qk_rope_head_dim"`
		QLoraRank              int     `json:"q_lora_rank"`
		OLoraRank              int     `json:"o_lora_rank"`
		OGroups                int     `json:"o_groups"`
		SwigluLimit            float64 `json:"swiglu_limit"`
		NRoutedExperts         int     `json:"n_routed_experts"`
		NSharedExperts         int     `json:"n_shared_experts"`
		NumExpertsPerTok       int     `json:"num_experts_per_tok"`
		RoutedScalingFactor    float64 `json:"routed_scaling_factor"`
		NormTopKProb           bool    `json:"norm_topk_prob"`
		CompressRatios         []int   `json:"compress_ratios"`
		CompressRopeTheta      float64 `json:"compress_rope_theta"`
		KVSourceLayerIDs       []int   `json:"kv_source_layer_ids"`
		IndexSourceLayerIDs    []int   `json:"index_source_layer_ids"`
		IndexNHeads            int     `json:"index_n_heads"`
		IndexHeadDim           int     `json:"index_head_dim"`
		IndexTopK              int     `json:"index_topk"`
		CandidateSourceLayerID int     `json:"candidate_source_layer_id"`
		CandidateTopKBlocks    int     `json:"candidate_topk_blocks"`
		CandidateBlockSize     int     `json:"candidate_block_size"`
		HCMult                 int     `json:"hc_mult"`
		HCSinkhornIters        int     `json:"hc_sinkhorn_iters"`
		HCEps                  float64 `json:"hc_eps"`
		EngramLayerIDs         []int   `json:"engram_layer_ids"`
		EngramMaxNgramSize     int     `json:"engram_max_ngram_size"`
		EngramNHeads           int     `json:"engram_n_heads"`
		EngramHeadDim          int     `json:"engram_head_dim"`
		EngramPadTokenID       int     `json:"engram_pad_token_id"`
		EngramCompVocabSize    int     `json:"engram_compressed_vocab_size"`
	} `json:"text_config"`
}

func loadV41HFOracle(t *testing.T) v41HFOracle {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "model", "testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var orcl v41HFOracle
	if err := json.Unmarshal(raw, &orcl); err != nil {
		t.Fatal(err)
	}
	return orcl
}

// TestDeepSeek41PublishedHeaderAdmission is the end-to-end config-admission
// dogfood of the REAL published DeepSeek-V4.1-Flash Q2_K headers (#14102):
// load all seven shard headers, run the full f.Config() admission path, and pin
// every geometry axis admission produces against the published HF oracle. It is
// a regression witness: while every axis admits, this test locks the exact
// artifact identity; if any axis starts refusing, the minimal defect is fixed
// first, not the expectation weakened.
// fak-test:runtime medium est=1s lane=default
func TestDeepSeek41PublishedHeaderAdmission(t *testing.T) {
	orcl := loadV41HFOracle(t)
	f := loadV41HeaderFixture(t)
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("published real header refused by config admission: %v", err)
	}
	if cfg.ModelType != "deepseek41" {
		t.Fatalf("ModelType=%q, want the canonical deepseek41 GGUF identity", cfg.ModelType)
	}

	// ---- decoder / MoE / MLA flat axes vs the published oracle ---------------
	for _, ax := range []struct {
		name string
		got, want any
	}{
		{"num_hidden_layers", cfg.NumLayers, orcl.Text.NumHiddenLayers},
		{"hidden_size", cfg.HiddenSize, orcl.Text.HiddenSize},
		{"vocab_size", cfg.VocabSize, orcl.Text.VocabSize},
		{"num_attention_heads", cfg.NumHeads, orcl.Text.NumAttentionHeads},
		{"head_dim", cfg.HeadDim, orcl.Text.HeadDim},
		{"qk_rope_head_dim", cfg.QKRopeHeadDim, orcl.Text.QKRopeHeadDim},
		{"q_lora_rank", cfg.QLoraRank, orcl.Text.QLoraRank},
		{"o_lora_rank", cfg.OLoraRank, orcl.Text.OLoraRank},
		{"o_groups", cfg.OGroups, orcl.Text.OGroups},
		{"n_routed_experts", cfg.NumExperts, orcl.Text.NRoutedExperts},
		{"n_shared_experts", cfg.NSharedExperts, orcl.Text.NSharedExperts},
		{"num_experts_per_tok", cfg.NumExpertsPerTok, orcl.Text.NumExpertsPerTok},
		{"moe_intermediate_size", cfg.MoEIntermediateSize, orcl.Text.MoEIntermediateSize},
		{"index_n_heads", cfg.IndexNHeads, orcl.Text.IndexNHeads},
		{"index_head_dim", cfg.IndexHeadDim, orcl.Text.IndexHeadDim},
		{"index_topk", cfg.IndexTopK, orcl.Text.IndexTopK},
		{"hc_mult", cfg.HCMult, orcl.Text.HCMult},
		{"hc_sinkhorn_iters", cfg.HCSinkhornIters, orcl.Text.HCSinkhornIters},
	} {
		if ax.got != ax.want {
			t.Fatalf("%s=%v, published oracle %v", ax.name, ax.got, ax.want)
		}
	}
	for _, ax := range []struct {
		name string
		got, want float64
	}{
		{"routed_scaling_factor", cfg.RoutedScalingFactor, orcl.Text.RoutedScalingFactor},
		{"swiglu_limit", cfg.SwigluLimit, orcl.Text.SwigluLimit},
		{"compress_rope_theta", cfg.CompressRopeTheta, orcl.Text.CompressRopeTheta},
	} {
		if ax.got != ax.want {
			t.Fatalf("%s=%v, published oracle %v", ax.name, ax.got, ax.want)
		}
	}
	// HCEps travels as F32 in the header; compare at storage precision.
	if float32(cfg.HCEps) != float32(orcl.Text.HCEps) {
		t.Fatalf("hc_eps=%v, published oracle %v", cfg.HCEps, orcl.Text.HCEps)
	}
	if !cfg.NormTopKProb {
		t.Fatal("expert_weights_norm=false, published oracle true")
	}
	if cfg.BOSTokenID == nil || *cfg.BOSTokenID != orcl.BOSTokenID || cfg.EOSTokenID != orcl.EOSTokenID {
		t.Fatalf("special ids BOS=%v EOS=%d, published oracle %d/%d",
			cfg.BOSTokenID, cfg.EOSTokenID, orcl.BOSTokenID, orcl.EOSTokenID)
	}

	// ---- CED/CSA2 compress schedule ------------------------------------------
	n := orcl.Text.NumHiddenLayers
	if len(cfg.CompressRatios) < n || len(orcl.Text.CompressRatios) < n {
		t.Fatalf("compress schedule length gguf=%d hf=%d, want >= %d",
			len(cfg.CompressRatios), len(orcl.Text.CompressRatios), n)
	}
	if !slices.Equal(cfg.CompressRatios[:n], orcl.Text.CompressRatios[:n]) {
		t.Fatalf("compress_ratios=%v, published oracle %v", cfg.CompressRatios[:n], orcl.Text.CompressRatios[:n])
	}

	// ---- V4.1 envelope: shared-attention topology + candidate geometry -------
	d41 := cfg.DeepSeekV41
	if d41 == nil {
		t.Fatal("admitted config carries no DeepSeekV41 envelope")
	}
	if !slices.Equal(d41.KVSourceLayerIDs, orcl.Text.KVSourceLayerIDs) {
		t.Fatalf("kv source layers %v, published oracle %v", d41.KVSourceLayerIDs, orcl.Text.KVSourceLayerIDs)
	}
	if !slices.Equal(d41.IndexSourceLayerIDs, orcl.Text.IndexSourceLayerIDs) {
		t.Fatalf("index source layers %v, published oracle %v", d41.IndexSourceLayerIDs, orcl.Text.IndexSourceLayerIDs)
	}
	if !slices.Equal(d41.CompressRatios[:n], orcl.Text.CompressRatios[:n]) {
		t.Fatalf("envelope compress_ratios=%v, published oracle %v", d41.CompressRatios[:n], orcl.Text.CompressRatios[:n])
	}
	for _, ax := range []struct {
		name string
		got, want int
	}{
		{"candidate_source_layer_id", d41.CandidateSourceLayerID, orcl.Text.CandidateSourceLayerID},
		{"candidate_topk_blocks", d41.CandidateTopKBlocks, orcl.Text.CandidateTopKBlocks},
		{"candidate_block_size", d41.CandidateBlockSize, orcl.Text.CandidateBlockSize},
	} {
		if ax.got != ax.want {
			t.Fatalf("%s=%d, published oracle %d", ax.name, ax.got, ax.want)
		}
	}

	// ---- Engram stage metadata vs the published oracle ------------------------
	eng := f.DeepSeek41Engram
	if eng == nil {
		t.Fatal("missing DeepSeek V4.1 Engram metadata")
	}
	if !slices.Equal(eng.LayerIDs, orcl.Text.EngramLayerIDs) {
		t.Fatalf("engram layer ids %v, published oracle %v", eng.LayerIDs, orcl.Text.EngramLayerIDs)
	}
	for _, ax := range []struct {
		name string
		got, want int
	}{
		{"engram_max_ngram_size", eng.MaxNgramSize, orcl.Text.EngramMaxNgramSize},
		{"engram_n_heads", eng.NHeads, orcl.Text.EngramNHeads},
		{"engram_head_dim", eng.HeadDim, orcl.Text.EngramHeadDim},
		{"engram_pad_token_id", eng.PadTokenID, orcl.Text.EngramPadTokenID},
		{"engram_compressed_vocab_size", eng.CompressedVocabSize, orcl.Text.EngramCompVocabSize},
	} {
		if ax.got != ax.want {
			t.Fatalf("%s=%d, published oracle %d", ax.name, ax.got, ax.want)
		}
	}

	// ---- GGUF-load boundary: the HF global declarations are NOT inherited -----
	// The published checkpoint declares an FP8-dense quantization
	// (weight_block_size [32,32]); the Q2_K artifact is its quantization. The
	// GGUF admission path deliberately carries no global quantization or
	// attention-geometry declaration -- dtypes are observed per tensor -- so the
	// envelope axes stay zero-valued instead of falling through to the HF
	// defaults. Pin that fail-closed boundary.
	if orcl.Quantization.QuantMethod != "fp8" || !slices.Equal(orcl.Quantization.WeightBlockSize, []int{32, 32}) {
		t.Fatalf("oracle quantization %+v drifted from the published fp8 32x32 declaration", orcl.Quantization)
	}
	if d41.Quantization.Method != "" || d41.Attention.NumLayers != 0 {
		t.Fatalf("GGUF load inherited a global declaration: quant=%+v attention=%+v, want both zero-valued",
			d41.Quantization, d41.Attention)
	}
}

var _ = model.ErrV41ConfigAdmission // admission refusals are the typed contract this witness guards
