package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

const (
	deepSeekV4FlashModelID      = "deepseek-ai/DeepSeek-V4-Flash-0731"
	deepSeekV4FlashRevision     = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
	deepSeekV4FlashConfigSHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
)

func readDeepSeekV4FlashConfig(t *testing.T) ([]byte, Config) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek_v4_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("parse official %s config at %s: %v", deepSeekV4FlashModelID, deepSeekV4FlashRevision, err)
	}
	return raw, cfg
}

func TestDeepSeekV4FlashOfficialConfigIsPinnedAndAdmitted(t *testing.T) {
	raw, cfg := readDeepSeekV4FlashConfig(t)
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != deepSeekV4FlashConfigSHA256 {
		t.Fatalf("official config digest = %s, want %s for %s@%s", got, deepSeekV4FlashConfigSHA256, deepSeekV4FlashModelID, deepSeekV4FlashRevision)
	}
	if err := AdmitDeepSeekV4Config(cfg); err != nil {
		t.Fatalf("official %s@%s config rejected: %v", deepSeekV4FlashModelID, deepSeekV4FlashRevision, err)
	}
}

func TestDeepSeekV4FlashRejectsMixedFlashAndProTuples(t *testing.T) {
	_, flash := readDeepSeekV4FlashConfig(t)
	pro := pinnedV4Config()

	cases := map[string]func() Config{
		"flash with Pro layers":         func() Config { cfg := flash; cfg.NumLayers = pro.NumLayers; return cfg },
		"flash with Pro hidden size":    func() Config { cfg := flash; cfg.HiddenSize = pro.HiddenSize; return cfg },
		"flash with Pro routed experts": func() Config { cfg := flash; cfg.NumExperts = pro.NumExperts; return cfg },
		"flash with Pro MoE width":      func() Config { cfg := flash; cfg.MoEIntermediateSize = pro.MoEIntermediateSize; return cfg },
		"flash with Pro routing scale":  func() Config { cfg := flash; cfg.RoutedScalingFactor = pro.RoutedScalingFactor; return cfg },
		"Pro with Flash layers":         func() Config { cfg := pro; cfg.NumLayers = flash.NumLayers; return cfg },
		"Pro with Flash hidden size":    func() Config { cfg := pro; cfg.HiddenSize = flash.HiddenSize; return cfg },
		"Pro with Flash routed experts": func() Config { cfg := pro; cfg.NumExperts = flash.NumExperts; return cfg },
		"Pro with Flash MoE width":      func() Config { cfg := pro; cfg.MoEIntermediateSize = flash.MoEIntermediateSize; return cfg },
		"Pro with Flash routing scale":  func() Config { cfg := pro; cfg.RoutedScalingFactor = flash.RoutedScalingFactor; return cfg },
	}

	for name, build := range cases {
		t.Run(name, func(t *testing.T) {
			cfg := build()
			if err := AdmitDeepSeekV4Config(cfg); !errors.Is(err, ErrV4ConfigAdmission) {
				t.Fatalf("mixed tuple admitted, err=%v; want ErrV4ConfigAdmission", err)
			}
		})
	}
}

func TestDeepSeekV4FlashAdmissionPreservesProAndRejectsUnknownIdentifiers(t *testing.T) {
	if err := AdmitDeepSeekV4Config(pinnedV4Config()); err != nil {
		t.Fatalf("pinned V4 Pro config regressed: %v", err)
	}
	for _, modelType := range []string{"deepseek_v4_1", "deepseek_v4_vision"} {
		t.Run(modelType, func(t *testing.T) {
			cfg := pinnedV4Config()
			cfg.ModelType = modelType
			if err := AdmitDeepSeekV4Config(cfg); !errors.Is(err, ErrV4ConfigAdmission) {
				t.Fatalf("unknown model_type %q admitted, err=%v; want ErrV4ConfigAdmission", modelType, err)
			}
		})
	}
}

func TestDeepSeekV4FlashQuantLoaderAdmissionRunsBeforeWeightIO(t *testing.T) {
	_, flash := readDeepSeekV4FlashConfig(t)
	pro := pinnedV4Config()
	mixed := flash
	mixed.HiddenSize = pro.HiddenSize

	opened := 0
	openSentinel := errors.New("weight opener reached")
	opener := func(string) (*safetensorsFile, error) {
		opened++
		return nil, openSentinel
	}

	if _, err := loadSafetensorsQuantDir(t.TempDir(), mixed, opener); !errors.Is(err, ErrV4ConfigAdmission) {
		t.Fatalf("mixed config loader error = %v, want ErrV4ConfigAdmission", err)
	}
	if opened != 0 {
		t.Fatalf("mixed config opened %d weight files before admission", opened)
	}

	if _, err := loadSafetensorsQuantDir(t.TempDir(), flash, opener); !errors.Is(err, openSentinel) {
		t.Fatalf("official Flash config did not pass admission to the weight opener: %v", err)
	}
	if opened != 1 {
		t.Fatalf("official Flash config opened %d weight files, want exactly one sentinel open", opened)
	}
}
