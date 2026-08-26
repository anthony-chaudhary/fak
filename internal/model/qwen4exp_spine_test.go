package model

import (
	"errors"
	"reflect"
	"testing"
)

func TestQwen4ExpFlashNextPinnedContract(t *testing.T) {
	cfg, err := LoadQwen4ExpFlashNextConfig("testdata/qwen4exp_flash_next_config.json")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelType != "qwen4_exp_text" || cfg.Architecture != "Qwen4ExpForConditionalGeneration" {
		t.Fatalf("identity = %q/%q", cfg.ModelType, cfg.Architecture)
	}
	if cfg.NumHiddenLayers != 48 || cfg.FullAttentionInterval != 4 || cfg.NumExperts != 512 || cfg.NumExpertsPerToken != 10 || cfg.SharedExpertIntermediateSize != 640 || cfg.IndexerBudget != 2048 || cfg.MambaSSMDType != "float32" || !cfg.MTP.Hybrid || cfg.NgramSize != 3 || cfg.MaxPositionEmbeddings != 262144 {
		t.Fatalf("contract = %+v", cfg)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestQwen4ExpFlashNextTensorInventoryAndFailClosedExecution(t *testing.T) {
	cfg, err := LoadQwen4ExpFlashNextConfig("testdata/qwen4exp_flash_next_config.json")
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := LoadQwen4ExpTensorInventory("testdata/qwen4exp_flash_next_tensor_index.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateQwen4ExpTensorInventory(cfg, inventory); err != nil {
		t.Fatal(err)
	}
	if got := inventory.Tensors["model.language_model.layers.47.mlp.experts.gate_up_proj"].Shape; !reflect.DeepEqual(got, []int{512, 1280, 2560}) {
		t.Fatalf("expert tensor shape = %v", got)
	}
	err = cfg.NativeExecution()
	var unsupported *UnsupportedQwen4ExpExecutionError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %T %v", err, err)
	}
	if unsupported.Engine != "fak-native" || unsupported.ModelType != "qwen4_exp_text" {
		t.Fatalf("unsupported = %+v", unsupported)
	}
}

func TestQwen4ExpFlashNextRejectsCoercionAndMissingTensorBeforeExecution(t *testing.T) {
	cfg, err := LoadQwen4ExpFlashNextConfig("testdata/qwen4exp_flash_next_config.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.ModelType = "qwen3_8"
	if err := cfg.Validate(); err == nil {
		t.Fatal("accepted architecture alias")
	}

	cfg, _ = LoadQwen4ExpFlashNextConfig("testdata/qwen4exp_flash_next_config.json")
	inventory, _ := LoadQwen4ExpTensorInventory("testdata/qwen4exp_flash_next_tensor_index.json")
	delete(inventory.Tensors, "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight")
	if err := ValidateQwen4ExpTensorInventory(cfg, inventory); err == nil {
		t.Fatal("accepted missing indexer tensor")
	}
}
