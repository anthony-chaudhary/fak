package qwen4exp

import "testing"

func validMetalResidency() MetalResidency {
	return MetalResidency{Schema: MetalResidencySchema, Artifact: "sha256:x", ArtifactBytes: 360023351155, DType: "bf16", Chip: "Apple Silicon", UnifiedPhysicalBytes: 512 << 30, SystemReservedBytes: 32 << 30, RuntimePeakBytes: 32 << 30, StreamedBytes: 64 << 30, DiskFreeBytes: 100 << 30, Pressure: "nominal", Thermal: "nominal", Ops: map[string]bool{"gdn": true, "qsa_top2048": true, "sparse_moe": true, "shared_expert": true, "ple_ngram": true}, Engine: "fak-native", Fallback: "none"}
}
func TestMetalResidencyRejects36GiBExactBF16(t *testing.T) {
	p := validMetalResidency()
	p.UnifiedPhysicalBytes = 36 << 30
	p.SystemReservedBytes = 8 << 30
	p.StreamedBytes = 0
	if err := p.Validate(); err == nil {
		t.Fatal("impossible exact artifact admitted")
	}
}
func TestMetalResidencyAllowsMeasuredStreamingEnvelope(t *testing.T) {
	if err := validMetalResidency().Validate(); err != nil {
		t.Fatal(err)
	}
}
func TestMetalResidencyRequiresNoFallbackAndAllOps(t *testing.T) {
	p := validMetalResidency()
	p.Fallback = "mlx"
	if err := p.Validate(); err == nil {
		t.Fatal("fallback")
	}
	p = validMetalResidency()
	delete(p.Ops, "gdn")
	if err := p.Validate(); err == nil {
		t.Fatal("operation gap")
	}
}
