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
