package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// deepseek_v4_config_test.go — witnesses the GGUF side of the DeepSeek-V4 CSA/HCA
// compression-rate axis (V4 attention seam map, Missing seam #7): applyGLMMoeDsaConfig
// reads the two rates best-effort under the existing attention.* namespace, and a
// file that ships neither key (every GLM-5.2 GGUF today) leaves both zero — no
// behavior change. Metadata-only: parsing a rate does not make V4 executable.
//
// A minimal in-memory File (metadata map, no tensor directory) is enough: the
// reads go through intValueOrZero -> File.Uint64 -> Metadata, and with no tensors
// the indexer-schedule fail-loud guard cannot fire.

func TestApplyGLMMoeDsaConfigReadsV4CompressionRates(t *testing.T) {
	const p = "glm_moe_dsa."
	f := &File{Metadata: map[string]Value{
		p + glmKeyCSACompRate: {Type: TypeUint32, Value: uint32(4)},
		p + glmKeyHCACompRate: {Type: TypeUint32, Value: uint32(128)},
	}}
	var cfg model.Config
	if err := applyGLMMoeDsaConfig(f, p, &cfg, 0); err != nil {
		t.Fatalf("applyGLMMoeDsaConfig: %v", err)
	}
	if cfg.CSACompressionRate != 4 || cfg.HCACompressionRate != 128 {
		t.Fatalf("rates = csa:%d hca:%d, want 4/128", cfg.CSACompressionRate, cfg.HCACompressionRate)
	}
	if csa, hca, ok := cfg.V4CompressionRates(); !ok || csa != 4 || hca != 128 {
		t.Fatalf("V4CompressionRates() = (%d, %d, %v), want (4, 128, true)", csa, hca, ok)
	}
}

// TestApplyGLMMoeDsaConfigV4RatesAbsentStaysZero pins the no-op default: a file
// with no compression-rate keys (the GLM-5.2 shape) leaves both fields zero and
// the accessor inert, so loading an existing DSA checkpoint is unaffected.
func TestApplyGLMMoeDsaConfigV4RatesAbsentStaysZero(t *testing.T) {
	const p = "glm_moe_dsa."
	f := &File{Metadata: map[string]Value{}}
	var cfg model.Config
	if err := applyGLMMoeDsaConfig(f, p, &cfg, 0); err != nil {
		t.Fatalf("applyGLMMoeDsaConfig: %v", err)
	}
	if cfg.CSACompressionRate != 0 || cfg.HCACompressionRate != 0 {
		t.Fatalf("rates = csa:%d hca:%d, want 0/0 (absent)", cfg.CSACompressionRate, cfg.HCACompressionRate)
	}
	if _, _, ok := cfg.V4CompressionRates(); ok {
		t.Fatal("accessor reported a V4 declaration with no rate keys present")
	}
}
