package qwen38quantrun

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestValidateFakNativeRequestIdentityMutationMatrix(t *testing.T) {
	valid := func() FakNativeRequestIdentity {
		return FakNativeRequestIdentity{
			CampaignEngine:  qwen38quant.EngineFakNative,
			ArmEngine:       qwen38quant.EngineFakNative,
			ExpectedBackend: compute.Qwen38VulkanDecodeBackend,
			Receipt: &model.NativeInferenceReceipt{
				Engine: "inkernel", Planner: "inkernel", Owner: "fak",
				Backend: compute.Qwen38VulkanDecodeBackend, ForwardPath: "device/generic",
			},
		}
	}

	if err := ValidateFakNativeRequestIdentity(valid()); err != nil {
		t.Fatalf("real fak-native class to in-kernel runtime tuple rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*FakNativeRequestIdentity)
	}{
		{"campaign llama.cpp", func(v *FakNativeRequestIdentity) { v.CampaignEngine = qwen38quant.EngineLlamaCpp }},
		{"arm llama.cpp", func(v *FakNativeRequestIdentity) { v.ArmEngine = qwen38quant.EngineLlamaCpp }},
		{"expected backend empty", func(v *FakNativeRequestIdentity) { v.ExpectedBackend = "" }},
		{"receipt missing", func(v *FakNativeRequestIdentity) { v.Receipt = nil }},
		// These engine mutations explicitly prove that neither the campaign label
		// nor the comparator identity is accepted as an in-kernel runtime receipt.
		{"rewritten engine fak-native", func(v *FakNativeRequestIdentity) { v.Receipt.Engine = qwen38quant.EngineFakNative }},
		{"comparator engine llama.cpp", func(v *FakNativeRequestIdentity) { v.Receipt.Engine = qwen38quant.EngineLlamaCpp }},
		{"engine empty", func(v *FakNativeRequestIdentity) { v.Receipt.Engine = "" }},
		{"planner proxy", func(v *FakNativeRequestIdentity) { v.Receipt.Planner = "proxy" }},
		{"planner empty", func(v *FakNativeRequestIdentity) { v.Receipt.Planner = "" }},
		{"owner external", func(v *FakNativeRequestIdentity) { v.Receipt.Owner = "external" }},
		{"owner empty", func(v *FakNativeRequestIdentity) { v.Receipt.Owner = "" }},
		{"backend mismatch", func(v *FakNativeRequestIdentity) { v.Receipt.Backend = "cpu-ref" }},
		{"backend empty", func(v *FakNativeRequestIdentity) { v.Receipt.Backend = "" }},
		{"forward path empty", func(v *FakNativeRequestIdentity) { v.Receipt.ForwardPath = "" }},
		{"forward path whitespace", func(v *FakNativeRequestIdentity) { v.Receipt.ForwardPath = " \t" }},
		{"fallback active", func(v *FakNativeRequestIdentity) { v.Receipt.FallbackActive = true }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			identity := valid()
			tc.mutate(&identity)
			if err := ValidateFakNativeRequestIdentity(identity); err == nil {
				t.Fatal("mutated request identity was accepted")
			}
		})
	}
}

func TestValidateFakNativePhysicalIdentityMutationMatrix(t *testing.T) {
	valid := func() FakNativePhysicalIdentity {
		receipt := validFakNativePhysicalReceipt(t)
		return FakNativePhysicalIdentity{
			CampaignEngine:  qwen38quant.EngineFakNative,
			ArmEngine:       qwen38quant.EngineFakNative,
			ExpectedBackend: compute.Qwen38VulkanDecodeBackend,
			Receipt:         &receipt,
		}
	}

	if err := ValidateFakNativePhysicalIdentity(valid()); err != nil {
		t.Fatalf("canonical fak-native Vulkan physical receipt rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*FakNativePhysicalIdentity)
	}{
		{"campaign llama.cpp", func(v *FakNativePhysicalIdentity) { v.CampaignEngine = qwen38quant.EngineLlamaCpp }},
		{"arm llama.cpp", func(v *FakNativePhysicalIdentity) { v.ArmEngine = qwen38quant.EngineLlamaCpp }},
		{"expected backend empty", func(v *FakNativePhysicalIdentity) { v.ExpectedBackend = "" }},
		{"declared backend mismatch", func(v *FakNativePhysicalIdentity) { v.ExpectedBackend = "cpu-ref" }},
		{"receipt missing", func(v *FakNativePhysicalIdentity) { v.Receipt = nil }},
		// Cross-schema confusion is fail-closed: the request runtime engine is not
		// the engine name of a physical modelbench receipt.
		{"request engine inkernel", func(v *FakNativePhysicalIdentity) { v.Receipt.Engine.Name = "inkernel" }},
		{"comparator engine llama.cpp", func(v *FakNativePhysicalIdentity) { v.Receipt.Engine.Name = qwen38quant.EngineLlamaCpp }},
		{"engine backend drift", func(v *FakNativePhysicalIdentity) { v.Receipt.Engine.Backend = "cpu-ref" }},
		{"receipt backend drift", func(v *FakNativePhysicalIdentity) { v.Receipt.Backend = "cpu-ref" }},
		{"runtime drift", func(v *FakNativePhysicalIdentity) { v.Receipt.Engine.Runtime = "external" }},
		{"fallback unobserved", func(v *FakNativePhysicalIdentity) { v.Receipt.Engine.FallbackCount = nil }},
		{"fallback nonzero", func(v *FakNativePhysicalIdentity) { one := uint64(1); v.Receipt.Engine.FallbackCount = &one }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			identity := valid()
			tc.mutate(&identity)
			if err := ValidateFakNativePhysicalIdentity(identity); err == nil {
				t.Fatal("mutated physical identity was accepted")
			}
		})
	}
}

func validFakNativePhysicalReceipt(t *testing.T) compute.Qwen38VulkanDecodeReceipt {
	t.Helper()
	clean, yes, no := false, true, false
	zero := uint64(0)
	processPeak, devicePeak := uint64(16<<30), uint64(12<<30)
	raw := compute.Qwen38VulkanRawDecodeResult{
		Source: compute.Qwen38VulkanSourceIdentity{
			GitCommit: strings.Repeat("a", 40), SourceArchiveSHA256: strings.Repeat("b", 64),
			BinarySHA256: strings.Repeat("c", 64), Dirty: &clean,
		},
		Model: compute.Qwen38VulkanModelIdentity{
			Name: "Qwen3.8-27B", ArtifactPath: "/models/qwen3.8.gguf",
			ArtifactSHA256:        compute.Qwen38VulkanDecodeGGUFSHA256,
			TensorInventorySHA256: strings.Repeat("d", 64), TokenizerSHA256: strings.Repeat("e", 64),
			TemplateSHA256: strings.Repeat("f", 64), Quantization: "Q4_K_M",
		},
		Device: compute.Qwen38VulkanDeviceIdentity{
			OS: "linux", Arch: "amd64", Kernel: "6.14", Name: "AMD Radeon 8060S Graphics|gfx1151",
			MesaVersion: "25.2", VulkanVersion: "1.4", Firmware: "amdgpu-test",
		},
		Engine: compute.Qwen38VulkanEngineIdentity{
			Name: qwen38quant.EngineFakNative, Backend: compute.Qwen38VulkanDecodeBackend,
			Runtime: compute.Qwen38VulkanDecodeRuntime, ExecutedPath: "modelbench/raw-decode/vulkan", FallbackCount: &zero,
		},
		CaptureCommand: "modelbench -raw-decode", PromptTokenIDs: []int32{1, 2}, GeneratedTokenLimit: 2,
		OutputTokenIDs: []int32{3, 4}, OutputText: "ok", FiniteLogits: &yes, CPUModelParity: &yes,
		Runs: []compute.Qwen38VulkanDecodeRun{{
			Repetition: 1, ContextLimit: 8, ContextTokens: 4, GeneratedTokenLimit: 2,
			ActualGeneratedTokens: 2, Sampler: "greedy", SeedPolicy: "not_applicable_greedy",
			IgnoreEOS: &no, EOSStopped: &no, OutputTokenIDs: []int32{3, 4},
			PrefillNanoseconds: 10, DecodeNanoseconds: 20, CandidateElapsedNanoseconds: 30,
		}},
		CandidateElapsedNanoseconds: 30, PeakProcessMemoryBytes: &processPeak, PeakDeviceMemoryBytes: &devicePeak,
		Counters: &compute.Qwen38VulkanDecodeCounters{
			ComputeDispatches: 2, Q4KMatmulDispatches: 1, OtherComputeDispatches: 1, DispatchSubmits: 1,
			D2D:        compute.Qwen38VulkanTransferCounters{Count: 1, Bytes: 64},
			TensorHome: compute.Qwen38VulkanTensorHomeCounters{Admissions: 1, CopiedBytes: 64, ResidentBytes: 64},
		},
	}
	receipt, err := compute.BuildQwen38VulkanDecodeReceipt(raw)
	if err != nil {
		t.Fatalf("build canonical physical receipt: %v", err)
	}
	return receipt
}
