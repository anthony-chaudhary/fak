package ggufload

import (
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestClassifyTargetTensorQuantExcludesExactQwen38TrailingMTPBlock(t *testing.T) {
	cfg := model.Config{ModelType: "qwen35", NumLayers: 64, NumNextNPredictLayers: 1}
	tensors := []TensorInfo{
		// Target Q4_K_M inventory.
		{Name: "blk.63.attn_q.weight", Type: TensorQ4_K},
		{Name: "blk.63.attn_v.weight", Type: TensorQ6_K},
		{Name: "blk.63.ffn_gate.weight", Type: TensorQ4_K},
		{Name: "blk.63.ffn_down.weight", Type: TensorQ6_K},
		{Name: "blk.62.ssm_out.weight", Type: TensorQ5_K},
		{Name: "output_norm.weight", Type: TensorF32},
		// Exact Qwen3.8 trailing blk.64 MTP matrix mixture.
		{Name: "blk.64.nextn.eh_proj.weight", Type: TensorQ8_0},
		{Name: "blk.64.nextn.enorm.weight", Type: TensorF32},
		{Name: "blk.64.attn_q.weight", Type: TensorQ4_K},
		{Name: "blk.64.attn_k.weight", Type: TensorQ4_K},
		{Name: "blk.64.attn_v.weight", Type: TensorQ6_K},
		{Name: "blk.64.attn_output.weight", Type: TensorQ4_K},
		{Name: "blk.64.ffn_gate.weight", Type: TensorQ4_K},
		{Name: "blk.64.ffn_up.weight", Type: TensorQ4_K},
		{Name: "blk.64.ffn_down.weight", Type: TensorQ6_K},
		{Name: "blk.64.attn_norm.weight", Type: TensorF32},
		// A known non-target sidecar namespace must not perturb the recipe.
		{Name: "mm.vision.weight", Type: TensorIQ3_S},
	}

	got := ClassifyTargetTensorQuant(cfg, tensors)
	if got.Recipe != "Q4_K_M" || got.Name != "mixed(Q4_K+Q5_K+Q6_K)" || !got.Q4KResident {
		t.Fatalf("target-only classification=%+v, want Q4_K_M target", got)
	}
	if full := ClassifyTensorQuant(tensors); full.Recipe != "" {
		t.Fatalf("full mixed target+sidecar recipe=%q, want unmatched full inventory", full.Recipe)
	}

	// A target-only split-shard slice remains byte-for-byte classification compatible.
	targetShard := tensors[:4]
	if split, want := ClassifyTargetTensorQuant(cfg, targetShard), ClassifyTensorQuant(targetShard); split != want {
		t.Fatalf("target-only split classification=%+v, want unchanged %+v", split, want)
	}
	q5OnlyShard := tensors[4:5]
	if split, want := ClassifyTargetTensorQuant(cfg, q5OnlyShard), ClassifyTensorQuant(q5OnlyShard); split != want {
		t.Fatalf("Q5-only split classification=%+v, want unchanged %+v", split, want)
	}
	// The same names on an architecture without this sidecar contract are not filtered.
	other := cfg
	other.ModelType = "llama"
	if got := ClassifyTargetTensorQuant(other, []TensorInfo{{Name: "blk.64.attn_q.weight", Type: TensorQ5_K}}); got.Name != "Q5_K" {
		t.Fatalf("non-Qwen trailing tensor classification=%+v, want preserved Q5_K", got)
	}
}

func TestClassifyTargetTensorQuantKnownQwen38Q4KMHeader(t *testing.T) {
	path := os.Getenv("FAK_QWEN38_Q4KM_GGUF")
	if path == "" {
		t.Skip("set FAK_QWEN38_Q4KM_GGUF to validate a local Qwen3.8-27B-Q4_K_M artifact header")
	}
	if info, err := os.Stat(path); err != nil || info.IsDir() {
		t.Skip("known Qwen3.8-27B-Q4_K_M artifact header is unavailable")
	}
	// Open parses and closes only the GGUF header; it never reads tensor payloads.
	gg, err := Open(path)
	if err != nil {
		t.Fatalf("Open header: %v", err)
	}
	cfg, err := gg.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "qwen35" || cfg.NumLayers != 64 || cfg.NumNextNPredictLayers != 1 {
		t.Fatalf("known artifact target/MTP config=(%q,%d,%d), want (qwen35,64,1)", cfg.ModelType, cfg.NumLayers, cfg.NumNextNPredictLayers)
	}
	got := ClassifyTargetTensorQuant(cfg, gg.Tensors)
	if got.Recipe != "Q4_K_M" || !got.Q4KResident {
		var q5 []string
		for _, tensor := range gg.Tensors {
			if tensor.Type == TensorQ5_K && !targetQuantExcludesTensor(cfg, tensor.Name) {
				q5 = append(q5, tensor.Name)
			}
		}
		t.Fatalf("known artifact target-only classification=%+v, want Q4_K_M; target Q5_K=%s", got, strings.Join(q5, ","))
	}
}
