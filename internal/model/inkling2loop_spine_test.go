package model

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

const (
	inkling2MiniConfigPath    = "testdata/inkling2/inkling_2_mini_config.json"
	inkling2VarloopConfigPath = "testdata/inkling2/tml_d24_varloop_research_config.json"
	inkling2MiniInventoryPath = "testdata/inkling2/inkling_2_mini_tensor_index.json"
)

func TestInkling2LoopPinnedContractBothCheckpoints(t *testing.T) {
	for _, tc := range []struct {
		path    string
		varloop bool
	}{{inkling2MiniConfigPath, false}, {inkling2VarloopConfigPath, true}} {
		cfg, err := LoadInkling2LoopConfig(tc.path)
		if err != nil {
			t.Fatalf("%s: %v", tc.path, err)
		}
		if cfg.Loop.IsVarloop != tc.varloop {
			t.Fatalf("%s: is_varloop = %v, want %v", tc.path, cfg.Loop.IsVarloop, tc.varloop)
		}
		if got := cfg.LayerVisits(); got != 48 {
			t.Fatalf("%s: layer visits = %d, want 48", tc.path, got)
		}
		if cfg.IsLooped(5) || !cfg.IsLooped(6) || !cfg.IsLooped(17) || cfg.IsLooped(18) {
			t.Fatalf("%s: loop core must be exactly layers 6..17", tc.path)
		}
		if cfg.ExpertPrecision(19) != "NVFP4" || cfg.ExpertPrecision(20) != "MXFP8" {
			t.Fatalf("%s: expert precision split must be NVFP4 <20, MXFP8 >=20", tc.path)
		}
	}
}

func TestInkling2LoopTensorInventoryAndFailClosedExecution(t *testing.T) {
	cfg, err := LoadInkling2LoopConfig(inkling2MiniConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	inv, err := LoadInkling2LoopTensorInventory(inkling2MiniInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateInkling2LoopTensorInventory(cfg, inv); err != nil {
		t.Fatal(err)
	}
	var unsupported *UnsupportedInkling2LoopExecutionError
	if err := cfg.NativeExecution(); !errors.As(err, &unsupported) {
		t.Fatalf("native execution = %v, want UnsupportedInkling2LoopExecutionError", err)
	}
	if unsupported.Engine != "fak-native" || !strings.Contains(unsupported.Error(), "vllm@ab18334a4f") {
		t.Fatalf("refusal must name the engine and the OSS routes: %v", unsupported)
	}
}

func TestInkling2LoopRejectsCoercionBeforeExecution(t *testing.T) {
	data, err := os.ReadFile(inkling2MiniConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	for name, edit := range map[string][2]string{
		"passes":   {`"num_passes": 3`, `"num_passes": 4`},
		"experts":  {`"n_routed_experts": 128`, `"n_routed_experts": 384`},
		"quant":    {`"quant_algo": "NVFP4"`, `"quant_algo": "FP8"`},
		"identity": {`"model_type": "inkling2_loop_text"`, `"model_type": "inkling2_text"`},
	} {
		mutated := strings.Replace(string(data), edit[0], edit[1], 1)
		if mutated == string(data) {
			t.Fatalf("%s: fixture edit %q did not apply", name, edit[0])
		}
		if _, err := ParseInkling2LoopConfig([]byte(mutated)); err == nil {
			t.Fatalf("%s: drifted config accepted", name)
		}
	}
	cfg, _ := LoadInkling2LoopConfig(inkling2MiniConfigPath)
	inv, _ := LoadInkling2LoopTensorInventory(inkling2MiniInventoryPath)
	w := inv.Tensors["model.llm.layers.19.mlp.experts.w13_weight"]
	w.DType = "F8_E4M3"
	inv.Tensors["model.llm.layers.19.mlp.experts.w13_weight"] = w
	if err := ValidateInkling2LoopTensorInventory(cfg, inv); err == nil {
		t.Fatal("NVFP4 layer recorded as MXFP8 accepted")
	}
}

// The generic Config decode flattens text_config into an ordinary-looking MoE;
// classification must refuse rather than pick a generic forward.
func TestInkling2LoopGenericConfigClassifiesAsRefusal(t *testing.T) {
	for _, path := range []string{inkling2MiniConfigPath, inkling2VarloopConfigPath} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var cfg Config
		if err := json.Unmarshal(data, &cfg); err != nil {
			t.Fatalf("%s: generic decode: %v", path, err)
		}
		if !cfg.IsInkling2Loop() {
			t.Fatalf("%s: generic decode lost the family (model_type %q)", path, cfg.ModelType)
		}
		kind, err := ClassifyForwardPath(cfg, nil)
		var unsupported *UnsupportedInkling2LoopExecutionError
		if !errors.As(err, &unsupported) {
			t.Fatalf("%s: ClassifyForwardPath = %q, %v; want UnsupportedInkling2LoopExecutionError", path, kind, err)
		}
	}
}

func TestInkling2LoopVarloopPassesAndDSparkTapGuard(t *testing.T) {
	varloopCfg, err := LoadInkling2LoopConfig(inkling2VarloopConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	// Load-time override 1..5 is accepted on varloop checkpoints.
	for _, n := range []int{1, 2, 3, 4, 5} {
		loaded, err := varloopCfg.WithLoadPasses(n)
		if err != nil {
			t.Fatalf("WithLoadPasses(%d) failed: %v", n, err)
		}
		if loaded.Loop.NumPasses != n {
			t.Fatalf("WithLoadPasses(%d) got %d", n, loaded.Loop.NumPasses)
		}
	}
	if _, err := varloopCfg.WithLoadPasses(0); err == nil {
		t.Fatal("WithLoadPasses(0) should fail")
	}

	// Per-request passes <= loaded (5) are accepted; > loaded are refused.
	loaded5, err := varloopCfg.WithLoadPasses(5)
	if err != nil {
		t.Fatal(err)
	}
	req2, err := loaded5.WithRequestPasses(2)
	if err != nil || req2.Loop.NumPasses != 2 {
		t.Fatalf("WithRequestPasses(2) = %+v, %v", req2.Loop, err)
	}
	var pe *Inkling2LoopPassesError
	if _, err := loaded5.WithRequestPasses(6); !errors.As(err, &pe) || pe.Requested != 6 || pe.Max != 5 {
		t.Fatalf("WithRequestPasses(6) = %v, want Inkling2LoopPassesError(6, 5)", err)
	}

	// DSpark tap guard: passes=3 (48 visits) fits [5,23,41,48], passes=1 (24 visits) does not.
	if err := varloopCfg.DSparkTapAllowed(3); err != nil {
		t.Fatalf("DSparkTapAllowed(3) unexpected error: %v", err)
	}
	if err := varloopCfg.DSparkTapAllowed(1); err == nil {
		t.Fatal("DSparkTapAllowed(1) should refuse when MTP taps exceed 24 visits")
	}

	// Non-varloop checkpoint refuses load-time or per-request pass changes.
	fixedCfg, err := LoadInkling2LoopConfig(inkling2MiniConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixedCfg.WithLoadPasses(2); err == nil {
		t.Fatal("non-varloop WithLoadPasses(2) should fail")
	}
	if _, err := fixedCfg.WithRequestPasses(2); err == nil {
		t.Fatal("non-varloop WithRequestPasses(2) should fail")
	}
}
