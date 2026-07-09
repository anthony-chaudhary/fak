package model

import (
	"encoding/json"
	"testing"
)

// deepseek_v4_config_test.go — witnesses the DeepSeek-V4 CSA/HCA compression-rate
// config axis (V4 attention seam map, Missing seam #7): the two rates parse from
// an HF config.json, the validated V4CompressionRates accessor enforces the
// two-plane invariant (hca > csa > 1), and every non-V4 config stays inert
// (accessor ok=false), so the load path for existing checkpoints is unchanged.
//
// Fence: these fields are METADATA ONLY. A well-formed pair does NOT mean V4 is
// executable — the co-resident two-plane kvLayout + dense-over-compressed attend
// are Missing seams #1-#3 and unbuilt. This test asserts parse + validation, not
// any forward behavior.

// TestDeepSeekV4CompressionRatesParseFromHFConfig drives the real HF-JSON parse
// path (Config.UnmarshalJSON) with the documented V4 rates and asserts they land
// on the exported fields and satisfy the validated accessor.
func TestDeepSeekV4CompressionRatesParseFromHFConfig(t *testing.T) {
	const js = `{
		"hidden_size": 7168,
		"num_hidden_layers": 4,
		"num_attention_heads": 128,
		"csa_compression_rate": 4,
		"hca_compression_rate": 128
	}`
	var cfg Config
	if err := json.Unmarshal([]byte(js), &cfg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if cfg.CSACompressionRate != 4 || cfg.HCACompressionRate != 128 {
		t.Fatalf("rates = csa:%d hca:%d, want 4/128", cfg.CSACompressionRate, cfg.HCACompressionRate)
	}
	csa, hca, ok := cfg.V4CompressionRates()
	if !ok || csa != 4 || hca != 128 {
		t.Fatalf("V4CompressionRates() = (%d, %d, %v), want (4, 128, true)", csa, hca, ok)
	}
}

// TestV4CompressionRatesValidation pins the accessor's two-plane invariant: a V4
// declaration is well-formed only when the heavily-compressed HCA plane is
// strictly more compressed than the light CSA plane, and the CSA plane is
// actually compressed (rate > 1). Malformed or absent rates report ok=false so a
// caller never routes a single-plane checkpoint down a two-plane path.
func TestV4CompressionRatesValidation(t *testing.T) {
	cases := []struct {
		name     string
		csa, hca int
		wantOK   bool
	}{
		{"documented V4 (4/128)", 4, 128, true},
		{"both unset (non-V4 default)", 0, 0, false},
		{"csa set, hca unset", 4, 0, false},
		{"csa unset, hca set", 0, 128, false},
		{"equal rates (no extra compression)", 4, 4, false},
		{"hca less compressed than csa (inverted)", 128, 4, false},
		{"csa==1 (not actually compressed)", 1, 128, false},
		{"minimal valid (2/4)", 2, 4, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{CSACompressionRate: tc.csa, HCACompressionRate: tc.hca}
			csa, hca, ok := cfg.V4CompressionRates()
			if ok != tc.wantOK {
				t.Fatalf("V4CompressionRates() ok = %v, want %v", ok, tc.wantOK)
			}
			// The rate passthrough always echoes the raw fields, ok or not.
			if csa != tc.csa || hca != tc.hca {
				t.Fatalf("rate passthrough = (%d, %d), want (%d, %d)", csa, hca, tc.csa, tc.hca)
			}
		})
	}
}

// TestV4CompressionRatesInertForNonV4Config guards the byte-identical promise:
// a config that never sets the V4 rates (every checkpoint fak loads today) must
// report ok=false, so nothing about the existing load path shifts.
func TestV4CompressionRatesInertForNonV4Config(t *testing.T) {
	var cfg Config // zero value = the shape a non-V4 unmarshal leaves these fields in
	if _, _, ok := cfg.V4CompressionRates(); ok {
		t.Fatal("zero-value config reported a V4 two-tier declaration; want inert (ok=false)")
	}
}
