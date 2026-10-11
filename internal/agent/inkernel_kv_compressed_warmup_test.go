package agent

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// fak-test:runtime fast est=1ms
func TestInKernelKVMemoryStatsFlashCompressedWarmup(t *testing.T) {
	raw, err := os.ReadFile("../model/testdata/deepseek_v4_flash_config.json")
	if err != nil {
		t.Fatal(err)
	}
	var cfg model.Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if err := model.AdmitDeepSeekV4Config(cfg); err != nil {
		t.Fatal(err)
	}
	// Geometry-only: loading this config must not allocate model weights or
	// pretend the not-yet-reachable compressed allocator has resident rows.
	p := &InKernelPlanner{m: &model.Model{Cfg: cfg}, tree: radixkv.New(0)}
	for _, tc := range []struct {
		gate         string
		slope, fixed int64
	}{{"0", 264192, 0}, {"1", 13760, 60893696}, {"0", 264192, 0}} {
		t.Setenv("FAK_HYBRID_KV", tc.gate)
		charge := cfg.ContextSizeConfigWithPrecision(compute.KVPrecisionF32)
		st := p.KVMemoryStats()
		if st.BytesPerToken != tc.slope || st.BytesPerToken != charge.KVBytesPerToken() {
			t.Fatalf("gate=%s slope=%d want=%d compute=%d", tc.gate, st.BytesPerToken, tc.slope, charge.KVBytesPerToken())
		}
		if st.FixedBytesPerStream != tc.fixed || st.FixedBytesPerStream != charge.FixedKVBytesPerStream() {
			t.Fatalf("gate=%s fixed=%d want=%d compute=%d", tc.gate, st.FixedBytesPerStream, tc.fixed, charge.FixedKVBytesPerStream())
		}
		if st.ResidentTokens != 0 || st.ResidentBytes != charge.KVBytes(0) {
			t.Fatalf("empty geometry fixture invented residency: %+v", st)
		}
	}
}

// fak-test:runtime fast est=1ms
func TestInKernelKVMemoryStatsUniformWarmupUnchanged(t *testing.T) {
	t.Setenv("FAK_HYBRID_KV", "1")
	for _, prec := range []model.KVPrecision{model.KVPrecisionFP32, model.KVPrecisionQ8_0} {
		cfg := tinyCfg()
		p := &InKernelPlanner{m: &model.Model{Cfg: cfg}, kvPrecision: prec}
		st := p.KVMemoryStats()
		kv := cfg.ContextSizeConfigWithPrecision(computeKVPrecisionFor(prec)).KV
		if st.BytesPerToken != compute.EstimateKVStoreBytes(kv, 1) || st.FixedBytesPerStream != 0 || st.DType != kv.Precision.StorageLabel() {
			t.Fatalf("uniform precision=%v changed: %+v", prec, st)
		}
	}
}

// This unchanged-API witness runs on both the pre-fix and candidate sources.
// fak-test:runtime fast est=1ms
func TestInKernelKVMemoryStatsFlashCompressedSlopeReachability(t *testing.T) {
	raw, err := os.ReadFile("../model/testdata/deepseek_v4_flash_config.json")
	if err != nil {
		t.Fatal(err)
	}
	var cfg model.Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if err := model.AdmitDeepSeekV4Config(cfg); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_HYBRID_KV", "1")
	p := &InKernelPlanner{m: &model.Model{Cfg: cfg}}
	if got := p.KVMemoryStats().BytesPerToken; got != 13760 {
		t.Fatalf("Flash compressed slope=%d, want 13760 from the existing compute contract", got)
	}
}
