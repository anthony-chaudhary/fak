package qwen4exp

import "testing"

func validCUDAResidency() CUDAResidency {
	return CUDAResidency{Schema: CUDAResidencySchema, Artifact: "sha256:x", ArtifactBytes: 360023351155, DType: "bf16", GPUs: []CUDAGPU{{ID: "0", Architecture: "sm_80", PhysicalBytes: 80 << 30, ReservedBytes: 2 << 30}, {ID: "1", Architecture: "sm_80", PhysicalBytes: 80 << 30, ReservedBytes: 2 << 30}, {ID: "2", Architecture: "sm_80", PhysicalBytes: 80 << 30, ReservedBytes: 2 << 30}, {ID: "3", Architecture: "sm_80", PhysicalBytes: 80 << 30, ReservedBytes: 2 << 30}, {ID: "4", Architecture: "sm_80", PhysicalBytes: 80 << 30, ReservedBytes: 2 << 30}}, Ops: map[string]bool{"gdn": true, "qsa_top2048": true, "sparse_moe": true, "shared_expert": true, "ple_ngram": true}, Engine: "fak-native", Fallback: "none"}
}
func TestCUDAResidencyRequiresSufficientPhysicalResidency(t *testing.T) {
	p := validCUDAResidency()
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	p.GPUs = p.GPUs[:4]
	if err := p.Validate(); err == nil {
		t.Fatal("undersized plan admitted")
	}
}
func TestCUDAResidencyRequiresExactCoverageAndNoFallback(t *testing.T) {
	p := validCUDAResidency()
	delete(p.Ops, "gdn")
	if err := p.Validate(); err == nil {
		t.Fatal("coverage gap")
	}
	p = validCUDAResidency()
	p.Fallback = "vllm"
	if err := p.Validate(); err == nil {
		t.Fatal("fallback")
	}
}
