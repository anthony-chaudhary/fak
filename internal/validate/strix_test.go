package validate

import "testing"

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
