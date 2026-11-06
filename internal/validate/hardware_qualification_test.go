package validate

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// fak-test:justify why=regression when=changed:internal/validate/**
// fak-test:runtime fast est=20ms
func TestHardwareQualificationNeverImplicitlyDispatchesStrix(t *testing.T) {
	root := writeValidationModule(t, publicFakModulePath)
	original := newStrixControllerAuthorityFn
	oldDiscover, oldArchive, oldRun := discoverStrixTargetFn, buildStrixCandidateArchiveFn, runStrixValidationFn
	t.Cleanup(func() {
		newStrixControllerAuthorityFn = original
		discoverStrixTargetFn, buildStrixCandidateArchiveFn, runStrixValidationFn = oldDiscover, oldArchive, oldRun
	})
	discoverStrixTargetFn = func(context.Context, string) (*amdgpu.StrixTarget, error) {
		t.Fatal("implicit validation must not discover a device")
		return nil, errors.New("unexpected discovery")
	}
	buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
		t.Fatal("implicit validation must not prepare a device payload")
		return amdgpu.StrixCandidateArchive{}, errors.New("unexpected device payload")
	}
	runStrixValidationFn = func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
		t.Fatal("implicit validation must not run a device job")
		return nil, errors.New("unexpected device job")
	}
	for _, tc := range []struct {
		name string
		mine []string
	}{
		{"CUDA and unnamed host contract", []string{"internal/compute/cuda_qwen35_qknorm_rope_cache_vllm.go", "internal/compute/cuda/qwen35_qknorm_rope_cache_vllm.cuh", "internal/compute/qwen35_qknorm_rope_cache_vllm_contract_test.go"}},
		{"AMD implementation", []string{"internal/amdgpu/strix_validation.go"}},
		{"shared implementation", []string{"internal/compute/backend.go"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
				calls++
				return amdgpu.StrixControllerAuthority{}, errors.New("retired target must not be requested")
			}
			res, recorder := newStrixPhaseTestRecorder()
			err := executeStrixValidationPhase(context.Background(), io.Discard, io.Discard, res, recorder, root, false, "", "", "", tc.mine)
			if calls != 0 || err != nil || !res.OK || res.StrixValidation != nil {
				t.Fatalf("implicit dispatch calls=%d err=%v result=%+v", calls, err, res)
			}
			// JSON is the consumer contract: software OK must not imply hardware PASS.
			raw, err := json.Marshal(res)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Contains(raw, []byte(`"status":"PENDING_HARDWARE"`)) {
				t.Fatalf("missing pending hardware qualification: %s", raw)
			}
		})
	}
}

// fak-test:justify why=guard when=changed:internal/validate/** twin=TestHardwareQualificationNeverImplicitlyDispatchesStrix
// fak-test:runtime fast est=10ms
func TestHardwareQualificationSelectsChangedBackends(t *testing.T) {
	for _, tc := range []struct {
		mine []string
		want []string
	}{
		{[]string{"internal/compute/cuda.go", "internal/compute/qwen35_qknorm_rope_cache_vllm_contract_test.go"}, []string{"cuda"}},
		{[]string{"internal/compute/qwen35_qknorm_rope_cache_vllm_contract_test.go"}, nil},
		{[]string{"internal/compute/vulkan.go", "internal/compute/cuda.go"}, []string{"amd-vulkan", "cuda"}},
		{[]string{"internal/compute/hip.go"}, []string{"rocm"}},
		{[]string{"internal/model/metal_kernel.go"}, []string{"metal"}},
		{[]string{"internal/model/cuda_hal.go"}, []string{"cuda"}},
		{[]string{"internal/model/qwen38_cudadirect_swap.go"}, []string{"cuda"}},
		{[]string{"internal/model/rocmfp4.go"}, []string{"rocm"}},
		{[]string{"internal/model/ngram_amd64.go", "internal/model/quant_amd64_avx512.s"}, nil},
		{[]string{"internal/compute/rmsnorm_fp8_quant.cu"}, []string{"cuda"}},
		{[]string{"shaders/matmul.comp"}, []string{"amd-vulkan"}},
		{[]string{"internal/amdgpu/strix_validation.go"}, []string{"amd-vulkan"}},
		{[]string{"internal/compute/backend.go"}, []string{"shared"}},
		{[]string{"internal/compute/cuda/hip_bridge.cuh"}, []string{"cuda", "rocm"}},
		{[]string{`internal\compute\cuda.go`}, []string{"cuda"}},
		{[]string{"internal/compute/cuda/../backend.go"}, []string{"shared"}},
		{[]string{"docs/strix-halo.md", "internal/compute/README.md", "internal/validate/strix.go", "platform/strix/runtime.go"}, nil},
	} {
		got := hardwareQualificationBackends(tc.mine)
		if strings.Join(got, ",") != strings.Join(tc.want, ",") {
			t.Errorf("backends(%v)=%v, want %v", tc.mine, got, tc.want)
		}
	}
}

// fak-test:justify why=guard when=changed:internal/validate/** twin=TestHardwareQualificationNeverImplicitlyDispatchesStrix
// fak-test:runtime fast est=20ms
func TestHardwareQualificationKeepsExplicitReceiptGate(t *testing.T) {
	t.Setenv(hwReceiptEnv, "")
	root := writeValidationModule(t, publicFakModulePath)
	oldBuild, oldNow, oldAuthority := buildStrixCandidateArchiveFn, validateNow, newStrixControllerAuthorityFn
	t.Cleanup(func() {
		buildStrixCandidateArchiveFn, validateNow, newStrixControllerAuthorityFn = oldBuild, oldNow, oldAuthority
	})
	validateNow = func() time.Time { return hwTestNow }
	buildStrixCandidateArchiveFn = func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
		return amdgpu.StrixCandidateArchive{SourceArchiveSHA256: hwTestArchive}, nil
	}
	newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
		t.Fatal("receipt admission must not dispatch hardware")
		return amdgpu.StrixControllerAuthority{}, nil
	}
	for _, tc := range []struct {
		name        string
		raw         []byte
		mine        []string
		wantOK      bool
		wantPending []string
	}{
		{"missing supplied receipt", nil, hwTestMine, false, []string{"amd-vulkan"}},
		{"malformed supplied receipt", []byte("{invalid"), hwTestMine, false, []string{"amd-vulkan"}},
		{"valid exact binding remains scoped", hwTestReceipt(nil), hwTestMine, true, []string{"amd-vulkan"}},
		{"valid Strix proof leaves CUDA pending", hwTestReceipt(nil), append(append([]string(nil), hwTestMine...), "internal/compute/cuda.go"), true, []string{"amd-vulkan", "cuda"}},
		{"SSD restart does not qualify unrelated AMD kernel", hwTestReceipt(nil), append(append([]string(nil), hwTestMine...), "internal/compute/vulkan_rmsnorm.go"), true, []string{"amd-vulkan"}},
		{"missing receipt on host-only request still refuses", nil, []string{"internal/policy/policy.go"}, false, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			receipt := filepath.Join(t.TempDir(), "receipt.json")
			if tc.raw != nil {
				if err := os.WriteFile(receipt, tc.raw, 0o600); err != nil {
					t.Fatal(err)
				}
			}
			res, recorder := newStrixPhaseTestRecorder()
			res.Tip = hwTestTip
			runValidateHardwarePhase(context.Background(), io.Discard, io.Discard, res, recorder, root, validateOptions{strixReceipt: receipt}, tc.mine)
			if res.OK != tc.wantOK || res.StrixReceipt == nil || res.StrixReceipt.Admitted != tc.wantOK || res.StrixValidation != nil {
				t.Fatalf("receipt result=%+v, wantOK=%v", res, tc.wantOK)
			}
			var pending []string
			if res.HardwareQualification != nil {
				pending = res.HardwareQualification.Backends
			}
			if strings.Join(pending, ",") != strings.Join(tc.wantPending, ",") {
				t.Fatalf("pending=%v, want=%v", pending, tc.wantPending)
			}
			if len(pending) > 0 && res.HardwareQualification.Status != "PENDING_HARDWARE" {
				t.Fatal("unqualified backend was credited")
			}
		})
	}
	res, recorder := newStrixPhaseTestRecorder()
	runValidateHardwarePhase(context.Background(), io.Discard, io.Discard, res, recorder, root, validateOptions{}, []string{"internal/compute/cuda.go"})
	var text bytes.Buffer
	renderValidate(&text, *res)
	if !strings.Contains(text.String(), "PENDING_HARDWARE (cuda)") || !strings.Contains(text.String(), "Software checks do not qualify") || !reflect.DeepEqual(res.HardwareQualification.Backends, []string{"cuda"}) {
		t.Fatalf("text output hides qualification scope: %s", text.String())
	}
}

// fak-test:justify why=regression when=changed:internal/validate/**
// fak-test:runtime fast est=20ms
func TestHardwareQualificationScopesAmbientStrixReceipt(t *testing.T) {
	publicRoot := writeValidationModule(t, publicFakModulePath)
	privateRoot := writeValidationModule(t, "example.com/control-plane")
	missing := filepath.Join(t.TempDir(), "missing-receipt.json")
	t.Setenv(hwReceiptEnv, missing)
	oldAuthority := newStrixControllerAuthorityFn
	t.Cleanup(func() { newStrixControllerAuthorityFn = oldAuthority })
	newStrixControllerAuthorityFn = func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
		t.Fatal("receipt selection must never dispatch hardware")
		return amdgpu.StrixControllerAuthority{}, errors.New("unexpected hardware dispatch")
	}
	for _, tc := range []struct {
		name        string
		mine        []string
		private     bool
		opts        validateOptions
		wantReceipt bool
	}{
		{name: "CUDA", mine: []string{"internal/compute/cuda.go"}},
		{name: "Metal", mine: []string{"internal/compute/metal.go"}},
		{name: "ROCm", mine: []string{"internal/compute/rocm.go"}},
		{name: "host-only", mine: []string{"internal/policy/policy.go"}},
		{name: "private control plane", mine: []string{"internal/amdgpu/strix.go"}, private: true},
		{name: "AMD", mine: []string{"internal/compute/vulkan.go"}, wantReceipt: true},
		{name: "shared", mine: []string{"internal/compute/backend.go"}, wantReceipt: true},
		{name: "mixed AMD and CUDA", mine: []string{"internal/compute/vulkan.go", "internal/compute/cuda.go"}, wantReceipt: true},
		{name: "explicit Strix on CUDA", mine: []string{"internal/compute/cuda.go"}, opts: validateOptions{strix: true}, wantReceipt: true},
		{name: "explicit receipt on host-only", mine: []string{"internal/policy/policy.go"}, opts: validateOptions{strixReceipt: missing}, wantReceipt: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := publicRoot
			if tc.private {
				root = privateRoot
			}
			res, recorder := newStrixPhaseTestRecorder()
			runValidateHardwarePhase(context.Background(), io.Discard, io.Discard, res, recorder, root, tc.opts, tc.mine)
			if (res.StrixReceipt != nil) != tc.wantReceipt || res.OK == tc.wantReceipt || res.StrixValidation != nil {
				t.Fatalf("ambient receipt selection=%+v wantReceipt=%v", res, tc.wantReceipt)
			}
			if !tc.private && len(hardwareQualificationBackends(tc.mine)) > 0 && (res.HardwareQualification == nil || res.HardwareQualification.Status != "PENDING_HARDWARE") {
				t.Fatal("receipt routing erased pending device qualification")
			}
		})
	}
}

// fak-test:justify why=regression when=changed:internal/validate/**
// fak-test:runtime medium est=10s lane=default
func TestHardwareQualificationRunPreservesCUDADeviceDebt(t *testing.T) {
	eventFixtureEnvironment(t)
	root := writeValidationModule(t, publicFakModulePath)
	const source = "internal/compute/cuda.go"
	const contract = "internal/compute/qwen35_qknorm_rope_cache_vllm_contract_test.go"
	identityTestWrite(t, root, source, "package compute\n\nfunc Value() int { return 1 }\n")
	identityTestWrite(t, root, contract, "package compute\n\nimport \"testing\"\n\nfunc TestHostContract(t *testing.T) { if Value() != 1 { t.Fatal(\"old value\") } }\n")
	identityTestGit(t, root, "init", "-q")
	identityTestGit(t, root, "add", ".")
	identityTestGit(t, root, "-c", "user.name=Validator Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
	identityTestWrite(t, root, source, "package compute\n\nfunc Value() int { return 2 }\n")
	identityTestWrite(t, root, contract, "package compute\n\nimport \"testing\"\n\nfunc TestHostContract(t *testing.T) {\n\tif Value() != 2 {\n\t\tt.Fatal(\"candidate value\")\n\t}\n}\n")
	t.Setenv(hwReceiptEnv, filepath.Join(t.TempDir(), "unrelated-strix-receipt.json"))
	var stdout, stderr bytes.Buffer
	code := RunWithHooks(&stdout, &stderr, []string{"--root", root, "--mine", source, "--mine", contract, "--wsl-tests=false", "--progress=false", "--json"}, Hooks{
		NewStrixAuthority: func(context.Context, string) (amdgpu.StrixControllerAuthority, error) {
			t.Fatal("unexpected controller authority")
			return amdgpu.StrixControllerAuthority{}, nil
		},
		DiscoverStrixTarget: func(context.Context, string) (*amdgpu.StrixTarget, error) {
			t.Fatal("unexpected device discovery")
			return nil, nil
		},
		BuildStrixCandidate: func(context.Context, string, string, []string) (amdgpu.StrixCandidateArchive, error) {
			t.Fatal("unexpected device payload")
			return amdgpu.StrixCandidateArchive{}, nil
		},
		RunStrixValidation: func(context.Context, amdgpu.StrixValidationOpts) (*amdgpu.StrixValidationReceipt, error) {
			t.Fatal("unexpected hardware job")
			return nil, nil
		},
	})
	var res validateResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil || code != 0 || !res.OK || res.Mode != "full" {
		t.Fatalf("software validation code=%d err=%v stdout=%s stderr=%s", code, err, stdout.String(), stderr.String())
	}
	if res.StrixReceipt != nil || res.StrixValidation != nil || res.HardwareQualification == nil || res.HardwareQualification.Status != "PENDING_HARDWARE" || !reflect.DeepEqual(res.HardwareQualification.Backends, []string{"cuda"}) {
		t.Fatalf("software pass erased or misrouted physical evidence debt: %+v", res)
	}
	for _, required := range []string{"gofmt", "build", "vet", "platform_filename", "cross_build", "test"} {
		passed := false
		for _, phase := range res.Phases {
			passed = passed || phase.Name == required && phase.Status == "ok"
		}
		if !passed {
			t.Fatalf("required software phase %s did not pass: %+v", required, res.Phases)
		}
	}
}
