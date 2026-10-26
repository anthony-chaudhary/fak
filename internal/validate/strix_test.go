package validate

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// TestIsGPURelatedValidationKeywordArm pins the trigger that decides whether
// `fak validate --mine` must run physical Strix hardware validation. It mirrors
// the table in cmd/fak/validate_strix_test.go; the two copies of the function
// must stay in sync.
func TestIsGPURelatedValidationKeywordArm(t *testing.T) {
	tests := []struct {
		name     string
		mine     []string
		expected bool
	}{
		{"non-gpu changes", []string{"cmd/fak/new_verb.go", "internal/policy/policy.go", "docs/README.md"}, false},
		{"amdgpu package changes", []string{"internal/amdgpu/strixhalo.go"}, true},
		{"compute package changes", []string{"internal/compute/vulkan.go"}, true},
		{"acceptance validation changes", []string{"cmd/fak/validate_acceptance.go"}, true},
		{"strix named source file changes", []string{"internal/devcmd/amd_strix_validate.go"}, true},
		{"pure-CPU model change must not trigger", []string{"internal/model/llm.go"}, false},
		{"model Vulkan physical test still triggers via model marker", []string{"internal/model/vulkan_glm_kda_physical_test.go"}, true},
		{"docs path mentioning strix and halo must not trigger", []string{"docs/tickets/strix-halo-franchise/INDEX.md"}, false},
		{"control-plane test file mentioning halo must not trigger", []string{"platform/obs/stack/halo_test.go"}, false},
		{"control-plane source under platform/strix must not trigger", []string{"platform/strix/opencode_runtime.go"}, false},
		{"private factory Halo validator helper must not trigger", []string{"cmd/fak-flow/halo_verify.go"}, false},
		{"private Halo appliance command must not trigger", []string{"cmd/fak-strix/work.go"}, false},
		{"private Ops Halo bridge must not trigger", []string{"cmd/fak-sync/ops_halo_ticket_admission.go"}, false},
		{"test file outside gpu roots mentioning halo must not trigger", []string{"internal/foo/halo_test.go"}, false},
		{"test file under gpu root still triggers via root", []string{"internal/amdgpu/x_test.go"}, true},
		{"public non-test source with vulkan keyword still triggers", []string{"internal/serve/vulkan_backend.go"}, true},
		{"model metal kernel still triggers via model marker", []string{"internal/model/metal_kernel.go"}, true},
		{"compute shader path containing vulkan still triggers", []string{"shaders/vulkan_matmul.comp"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isGPURelatedValidation(tt.mine); got != tt.expected {
				t.Errorf("isGPURelatedValidation(%v) = %v, want %v", tt.mine, got, tt.expected)
			}
		})
	}
}

func TestExecuteStrixValidationScopesAutomaticHardwareToPublicModule(t *testing.T) {
	privateRoot := writeValidationModule(t, "example.com/private-control-plane")
	publicRoot := writeValidationModule(t, "github.com/anthony-chaudhary/fak")
	gpuChanges := []string{"internal/amdgpu/kernel.go"}
	authorityErr := errors.New("test controller authority unavailable")

	originalAuthority := newStrixControllerAuthorityFn
	t.Cleanup(func() { newStrixControllerAuthorityFn = originalAuthority })

	t.Run("private module skips automatic hardware", func(t *testing.T) {
		calls := 0
		newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
			calls++
			return amdgpu.StrixControllerAuthority{}, authorityErr
		}

		res, recorder := newStrixPhaseTestRecorder()
		err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, res, recorder, privateRoot, false, "", "", "", gpuChanges)
		if err != nil {
			t.Fatalf("automatic validation for private module returned error: %v", err)
		}
		if calls != 0 {
			t.Fatalf("controller authority calls = %d, want 0", calls)
		}
		if len(res.SkippedPhases) != 1 || res.SkippedPhases[0] != "strix_validation" {
			t.Fatalf("skipped phases = %v, want [strix_validation]", res.SkippedPhases)
		}
	})

	for _, tc := range []struct {
		name     string
		root     string
		explicit bool
	}{
		{name: "public module invokes automatic hardware", root: publicRoot},
		{name: "explicit hardware remains fail closed for private module", root: privateRoot, explicit: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
				calls++
				return amdgpu.StrixControllerAuthority{}, authorityErr
			}

			res, recorder := newStrixPhaseTestRecorder()
			err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, res, recorder, tc.root, tc.explicit, "", "", "", gpuChanges)
			if !errors.Is(err, authorityErr) {
				t.Fatalf("validation error = %v, want controller authority failure", err)
			}
			if calls != 1 {
				t.Fatalf("controller authority calls = %d, want 1", calls)
			}
			if len(res.SkippedPhases) != 0 {
				t.Fatalf("skipped phases = %v, want none", res.SkippedPhases)
			}
			if len(res.Failures) != 1 || res.Failures[0].Step != "strix-controller-authority" {
				t.Fatalf("failures = %+v, want one strix-controller-authority failure", res.Failures)
			}
		})
	}
}

func writeValidationModule(t *testing.T, modulePath string) string {
	t.Helper()
	root := t.TempDir()
	contents := []byte("module " + modulePath + "\n\ngo 1.26\n")
	if err := os.WriteFile(filepath.Join(root, "go.mod"), contents, 0o600); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	return root
}

func newStrixPhaseTestRecorder() (*validateResult, *validateRecorder) {
	res := &validateResult{OK: true}
	return res, &validateRecorder{
		ctx:     context.Background(),
		stderr:  io.Discard,
		started: time.Now(),
		res:     res,
	}
}

var cudaOnlyComputeValidationPaths = []string{
	"internal/compute/cuda.go",
	"internal/compute/cuda/attention.cuh",
	"internal/compute/cuda/device_context.cuh",
	"internal/compute/cuda/device_context_table.h",
	"internal/compute/cuda/fattn_mma_hd256.cuh",
	"internal/compute/cuda/kernel_profile.cuh",
	"internal/compute/cuda/model_candidates.cuh",
	"internal/compute/cuda/peer_handoff.cuh",
	"internal/compute/cuda/q4k_mmq.cuh",
	"internal/compute/cuda/quant_matmul.cuh",
	"internal/compute/cuda/qwen35_gdn.cuh",
	"internal/compute/cuda/runtime_memory.cuh",
	"internal/compute/cuda/runtime_services.cuh",
	"internal/compute/cuda/spec_graph.cuh",
	"internal/compute/cuda/weight_formats.cuh",
	"internal/compute/cuda_backend.h",
	"internal/compute/cuda_backend_state.go",
	"internal/compute/cuda_clone_serialization_test.go",
	"internal/compute/cuda_collective.go",
	"internal/compute/cuda_collective_pg.go",
	"internal/compute/cuda_device_context.go",
	"internal/compute/cuda_device_context_table_test.go",
	"internal/compute/cuda_device_handoff.go",
	"internal/compute/cuda_devices.go",
	"internal/compute/cuda_devices_cuda_test.go",
	"internal/compute/cuda_devices_stub.go",
	"internal/compute/cuda_devices_stub_test.go",
	"internal/compute/cuda_f16_stage_ownership_test.go",
	"internal/compute/cuda_kernels.cu",
	"internal/compute/cuda_kernels.go",
	"internal/compute/cuda_kv.go",
	"internal/compute/cuda_kv_clone_ownership_test.go",
	"internal/compute/cuda_prompt_panel.go",
	"internal/compute/cuda_q2_allocation_ownership_test.go",
	"internal/compute/cuda_q8_allocation_ownership_test.go",
	"internal/compute/cuda_q8_width_validation_test.go",
	"internal/compute/cuda_qwen35_gdn.go",
	"internal/compute/cuda_qwen35_gdn_sequence.go",
	"internal/compute/cuda_qwen35_sequence.go",
	"internal/compute/cuda_qwen4exp_ple_hash_cuda.go",
	"internal/compute/cuda_resident_handoff.go",
	"internal/compute/cuda_routed_expert_gemv.go",
	"internal/compute/cuda_spec_verify.go",
	"internal/compute/cuda_specialized.go",
	"internal/compute/cuda_v41_bf16_publish.go",
	"internal/compute/cuda_v41_rope.go",
	"internal/compute/cuda_v41_rope_contract_test.go",
	"internal/compute/q4k_prefill_cuda.go",
	"internal/compute/testdata/cuda_device_context.cpp",
}

// fak-test:justify why=regression when=changed:internal/validate/**
// fak-test:runtime fast est=10ms
func TestStrixValidationSkipsCUDAOnlyComputePaths(t *testing.T) {
	for _, changed := range cudaOnlyComputeValidationPaths {
		t.Run(changed, func(t *testing.T) {
			if isGPURelatedValidation([]string{changed}) {
				t.Fatalf("CUDA-only compute path selected Strix: %s", changed)
			}
		})
	}
	if isGPURelatedValidation(cudaOnlyComputeValidationPaths) {
		t.Fatal("CUDA-only compute scope selected Strix")
	}
	for _, changed := range []string{
		"./internal/compute/cuda.go",
		"internal//compute/cuda.go",
		`internal\compute\cuda.go`,
		"internal/compute/./cuda/attention.cuh",
	} {
		if isGPURelatedValidation([]string{changed}) {
			t.Fatalf("normalized CUDA-only path selected Strix: %s", changed)
		}
	}
}

// fak-test:justify why=guard when=changed:internal/validate/** twin=TestStrixValidationSkipsCUDAOnlyComputePaths
// fak-test:runtime fast est=10ms
func TestStrixValidationKeepsSharedAndMixedComputePaths(t *testing.T) {
	for _, changed := range []string{
		"internal/compute",
		"internal/compute/backend.go",
		"internal/compute/vulkan.go",
		"internal/compute/hip.go",
		"internal/compute/rocm.go",
		"internal/compute/strix.go",
		"internal/compute/cuda_vulkan_bridge.go",
		"internal/compute/cuda/hip_bridge.cuh",
		"internal/compute/cuda/amd_bridge.h",
		"internal/compute/cuda/gfx1151_bridge.cuh",
		"internal/compute/cuda/metal_bridge.go",
		"internal/compute/cuda/backend.comp",
		"internal/compute/cuda_device_payload.bin",
		"internal/compute/cuda_generic/shared.go",
		"internal/amdgpu/cuda_bridge.go",
		"internal/roofline/cuda.go",
		"internal/model/cuda_hal.go",
		"cmd/fak/validate_acceptance.go",
		"./internal/compute/backend.go",
		`internal\compute\vulkan.go`,
		"internal/compute/cuda/../backend.go",
		"../internal/compute/cuda.go",
		"internal/compute/../../compute/cuda.go",
		"/internal/compute/cuda.go",
		"C:/internal/compute/cuda.go",
		`\\server\internal\compute\cuda.go`,
	} {
		t.Run(changed, func(t *testing.T) {
			if !isGPURelatedValidation([]string{changed}) {
				t.Fatalf("shared, mixed, or invalid compute path skipped Strix: %s", changed)
			}
		})
	}
	for _, changed := range [][]string{
		{"internal/compute/cuda.go", "internal/compute/backend.go"},
		{"internal/compute/backend.go", "internal/compute/cuda.go"},
		{"internal/compute/cuda/attention.cuh", "internal/amdgpu/strixhalo.go"},
		{"internal/amdgpu/strixhalo.go", "internal/compute/cuda/attention.cuh"},
	} {
		if !isGPURelatedValidation(changed) {
			t.Fatalf("mixed compute scope skipped Strix: %v", changed)
		}
	}
}

// fak-test:justify why=regression when=changed:internal/validate/**
// fak-test:runtime fast est=20ms
func TestStrixValidationCUDAOnlySkipsBeforeControllerAuthority(t *testing.T) {
	publicRoot := writeValidationModule(t, publicFakModulePath)
	originalAuthority := newStrixControllerAuthorityFn
	originalDiscovery := discoverStrixTargetFn
	originalArchive := buildStrixCandidateArchiveFn
	originalRun := runStrixValidationFn
	t.Cleanup(func() {
		newStrixControllerAuthorityFn = originalAuthority
		discoverStrixTargetFn = originalDiscovery
		buildStrixCandidateArchiveFn = originalArchive
		runStrixValidationFn = originalRun
	})

	for _, tc := range []struct {
		name          string
		mine          []string
		explicit      bool
		wantAuthority int
	}{
		{name: "automatic CUDA-only scope", mine: cudaOnlyComputeValidationPaths},
		{name: "automatic host framework scope", mine: []string{"internal/validate/strix.go", "internal/validate/strix_test.go"}},
		{name: "explicit host framework remains protected", mine: []string{"internal/validate/strix.go"}, explicit: true, wantAuthority: 1},
		{name: "mixed host framework remains protected", mine: []string{"internal/validate/strix.go", "internal/compute/vulkan.go"}, wantAuthority: 1},
		{name: "explicit CUDA scope remains protected", mine: []string{"internal/compute/cuda.go"}, explicit: true, wantAuthority: 1},
		{name: "generic compute remains protected", mine: []string{"internal/compute/backend.go"}, wantAuthority: 1},
		{name: "Strix scope remains protected", mine: []string{"internal/amdgpu/strixhalo.go"}, wantAuthority: 1},
		{name: "mixed compute remains protected", mine: []string{"internal/compute/cuda.go", "internal/compute/vulkan.go"}, wantAuthority: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			authorityErr := errors.New("test controller authority unavailable")
			authorityCalls, discoveryCalls, archiveCalls, runCalls := 0, 0, 0, 0
			newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
				authorityCalls++
				return amdgpu.StrixControllerAuthority{}, authorityErr
			}
			discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) {
				discoveryCalls++
				return nil, authorityErr
			}
			buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
				archiveCalls++
				return amdgpu.StrixCandidateArchive{}, authorityErr
			}
			runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
				runCalls++
				return nil, authorityErr
			}
			res, recorder := newStrixPhaseTestRecorder()
			err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, res, recorder, publicRoot, tc.explicit, "", "", "", tc.mine)
			if authorityCalls != tc.wantAuthority || discoveryCalls != 0 || archiveCalls != 0 || runCalls != 0 {
				t.Fatalf("phase calls authority/discovery/archive/run = %d/%d/%d/%d, want %d/0/0/0", authorityCalls, discoveryCalls, archiveCalls, runCalls, tc.wantAuthority)
			}
			if res.StrixValidation != nil {
				t.Fatal("scope selection earned a hardware receipt")
			}
			if tc.wantAuthority == 0 {
				if err != nil || !res.OK || len(res.Failures) != 0 {
					t.Fatalf("CUDA-only phase error=%v OK=%v failures=%d", err, res.OK, len(res.Failures))
				}
				if len(res.SkippedPhases) != 1 || res.SkippedPhases[0] != "strix_validation" {
					t.Fatalf("skipped phases = %v, want [strix_validation]", res.SkippedPhases)
				}
				return
			}
			if !errors.Is(err, authorityErr) || res.OK {
				t.Fatalf("protected phase error=%v OK=%v, want authority refusal", err, res.OK)
			}
			if len(res.SkippedPhases) != 0 || len(res.Failures) != 1 {
				t.Fatalf("protected phase skipped=%d failures=%d", len(res.SkippedPhases), len(res.Failures))
			}
			if res.Failures[0].Step != "strix-controller-authority" {
				t.Fatalf("failure step=%s, want strix-controller-authority", res.Failures[0].Step)
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/validate/**
// fak-test:runtime fast est=10ms
func TestStrixValidationSkipsHostFrameworkGoPaths(t *testing.T) {
	framework := []string{
		"internal/validate/strix.go",
		"internal/validate/strix_test.go",
		"internal/validate/halo_receipt.go",
		"internal/validate/vulkan_transport.go",
		"internal/validate/gfx115_probe.go",
		"./internal/validate/strix.go",
		`internal\validate\strix.go`,
	}
	for _, changed := range framework {
		if isGPURelatedValidation([]string{changed}) {
			t.Fatalf("host validator framework path selected Strix: %s", changed)
		}
	}
	if isGPURelatedValidation(framework) {
		t.Fatal("host validator framework scope selected Strix")
	}
}

// fak-test:justify why=guard when=changed:internal/validate/** twin=TestStrixValidationSkipsHostFrameworkGoPaths
// fak-test:runtime fast est=10ms
func TestStrixValidationKeepsFrameworkPayloadAndSiblingPaths(t *testing.T) {
	for _, changed := range []string{
		"internal/validate/strix.cu",
		"internal/validate/strix.cuh",
		"internal/validate/strix.hip",
		"internal/validate/strix.h",
		"internal/validate/vulkan.spv",
		"internal/validate/halo.comp",
		"internal/validate/gfx1151.bin",
		"internal/validate-extra/strix.go",
		"internal/validation/strix.go",
		"internal/validate/device/strix.go",
		"internal/devcmd/amd_strix_validate.go",
		"cmd/fak/validate_acceptance.go",
		"internal/validate/../compute/cuda.go",
		"../internal/validate/strix.go",
		"/internal/validate/strix.go",
		"C:/internal/validate/strix.go",
	} {
		t.Run(changed, func(t *testing.T) {
			if !isGPURelatedValidation([]string{changed}) {
				t.Fatalf("device payload, sibling, or invalid path skipped Strix: %s", changed)
			}
		})
	}
	for _, changed := range [][]string{
		{"internal/validate/strix.go", "internal/compute/backend.go"},
		{"internal/compute/backend.go", "internal/validate/strix.go"},
		{"internal/validate/strix.go", "internal/validate/strix.cu"},
		{"internal/amdgpu/strixhalo.go", "internal/validate/strix.go"},
	} {
		if !isGPURelatedValidation(changed) {
			t.Fatalf("mixed framework scope skipped Strix: %v", changed)
		}
	}
}
