package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// kv16RecordingBackend renames cpu-ref so the CUDA-only 16-bit admission runs without a
// GPU; NewKV records the requested tier and hands cpu-ref an f32 config.
type kv16RecordingBackend struct {
	compute.Backend
	name  string
	asked []compute.KVPrecision
}

func (b *kv16RecordingBackend) Name() string { return b.name }

func (b *kv16RecordingBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	b.asked = append(b.asked, cfg.Precision)
	cfg.Precision = compute.KVPrecisionF32
	return b.Backend.NewKV(cfg)
}

func kv16DenseConfig() Config {
	return Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 8, RopeTheta: 10000}
}

func kv16Session(be compute.Backend, cfg Config) *Session {
	return &Session{M: &Model{Cfg: cfg}, Backend: be, halKV: be.NewKV(halKVConfig(cfg))}
}

// fak-test:runtime fast est=5ms lane=default
func TestDeviceKVPrecisionCUDARealizes16Bit(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "")
	for _, prec := range []KVPrecision{KVPrecisionFP16, KVPrecisionBF16} {
		be := &kv16RecordingBackend{Backend: compute.Default(), name: "cuda"}
		s := kv16Session(be, kv16DenseConfig())
		if err := s.SetDeviceKVPrecision(prec); err != nil {
			t.Fatalf("%s on cuda: %v", prec, err)
		}
		want, _ := DeviceKVPrecision(prec)
		if got := s.HALKVPrecision(); got != want {
			t.Fatalf("%s: HAL KV precision = %v, want %v", prec, got, want)
		}
		if len(be.asked) != 2 || be.asked[1] != want {
			t.Fatalf("%s: NewKV tiers = %v, want [f32 %v]", prec, be.asked, want)
		}
		if err := s.SetDeviceKVPrecision(prec); err != nil || len(be.asked) != 2 {
			t.Fatalf("%s: re-applying the realized tier must be a no-op: err=%v asked=%v", prec, err, be.asked)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestDeviceKVPrecisionRefusesUnrealized(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "")
	hybrid := kv16DenseConfig()
	hybrid.LayerTypes = []string{"linear_attention", "full_attention"}
	mla := kv16DenseConfig()
	mla.ModelType = "deepseek2"
	for _, tc := range []struct {
		name    string
		backend string
		cfg     Config
		prec    KVPrecision
	}{
		{"vulkan store is f32", "vulkan", kv16DenseConfig(), KVPrecisionFP16},
		{"cpu-ref store is f32", "cpu-ref", kv16DenseConfig(), KVPrecisionBF16},
		{"hybrid off cuda", "vulkan", hybrid, KVPrecisionFP16},
		{"MLA layout bypasses the typed store", "cuda", mla, KVPrecisionFP16},
		{"q8_0 is a host tier", "cuda", kv16DenseConfig(), KVPrecisionQ8_0},
	} {
		be := &kv16RecordingBackend{Backend: compute.Default(), name: tc.backend}
		s := kv16Session(be, tc.cfg)
		before := s.halKV
		if err := s.SetDeviceKVPrecision(tc.prec); !errors.Is(err, ErrDeviceKVPrecisionUnrealized) {
			t.Fatalf("%s: err=%v, want ErrDeviceKVPrecisionUnrealized", tc.name, err)
		}
		if s.halKV != before || s.HALKVPrecision() != compute.KVPrecisionF32 || len(be.asked) != 1 {
			t.Fatalf("%s: refusal must keep the f32 store (asked=%v)", tc.name, be.asked)
		}
	}
	if reason := kv16DenseConfig().DeviceKVRefusal("vulkan", KVPrecisionFP32); reason != "" {
		t.Fatalf("f32 must be realizable everywhere, got %q", reason)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestDeviceKVPrecisionRefusesPopulatedStore(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "")
	be := &kv16RecordingBackend{Backend: compute.Default(), name: "cuda"}
	cfg := kv16DenseConfig()
	s := kv16Session(be, cfg)
	w := cfg.NumKVHeads * cfg.HeadDim
	row := compute.NewF32(be.Backend, []int{w}, make([]float32, w))
	for l := 0; l < cfg.NumLayers; l++ {
		s.halKV.AppendKV(l, row, row, row, 0)
	}
	if err := s.SetDeviceKVPrecision(KVPrecisionFP16); !errors.Is(err, ErrDeviceKVPrecisionUnrealized) {
		t.Fatalf("populated store: err=%v, want ErrDeviceKVPrecisionUnrealized", err)
	}
	if s.halKV.Len() != 1 || s.HALKVPrecision() != compute.KVPrecisionF32 {
		t.Fatal("refusal must keep the populated f32 store")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestParseKVPrecisionBF16(t *testing.T) {
	for _, raw := range []string{"bf16", "BF16", "bfloat16"} {
		if got, err := ParseKVPrecision(raw); err != nil || got != KVPrecisionBF16 {
			t.Fatalf("ParseKVPrecision(%q) = %q, %v; want bf16", raw, got, err)
		}
	}
	if got := KVVectorBytes(64, KVPrecisionBF16); got != 128 {
		t.Fatalf("bf16 vector bytes = %d, want 128", got)
	}
}

// Qwen3.5/3.8 hybrid full-attention planes live in the CUDA typed HAL store (the
// recurrent Gated-DeltaNet state is not KV), so a 16-bit tier is realized there.
// fak-test:runtime fast est=5ms lane=default
func TestDeviceKVPrecisionCUDARealizesQwen35Hybrid(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "")
	hybrid := kv16DenseConfig()
	hybrid.LayerTypes = []string{"linear_attention", "full_attention"}
	if !hybrid.IsQwen35Hybrid() {
		t.Fatal("precondition: config must be a Qwen3.5 hybrid")
	}
	for _, prec := range []KVPrecision{KVPrecisionFP16, KVPrecisionBF16} {
		be := &kv16RecordingBackend{Backend: compute.Default(), name: "cuda"}
		s := kv16Session(be, hybrid)
		if err := s.SetDeviceKVPrecision(prec); err != nil {
			t.Fatalf("%s on cuda hybrid: %v", prec, err)
		}
		want, _ := DeviceKVPrecision(prec)
		if got := s.HALKVPrecision(); got != want {
			t.Fatalf("%s: HAL KV precision = %v, want %v", prec, got, want)
		}
	}
	if reason := hybrid.DeviceKVRefusal("cuda", KVPrecisionQ8_0); reason == "" {
		t.Fatal("q8_0 stays a host tier for the hybrid device path")
	}
}
