package computebuild

import (
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestCUDASM103BuildPlan(t *testing.T) {
	t.Parallel()
	const matrix = "sm_80\nsm_89\nsm_90\nsm_100\nsm_120\n"
	for _, target := range []string{"sm_103", "103"} {
		args, _, err := ParseCUDAArchMatrix(matrix, target)
		want := []string{"-gencode", "arch=compute_103,code=sm_103"}
		if err != nil || !reflect.DeepEqual(args, want) {
			t.Fatalf("target %s: args=%v err=%v, want %v", target, args, err, want)
		}
	}
	args, _, err := ParseCUDAArchMatrix(matrix, "")
	if err != nil || strings.Contains(strings.Join(args, " "), "103") || args[len(args)-1] != "arch=compute_120,code=compute_120" {
		t.Fatalf("default CUDA 12.8 matrix changed: args=%v err=%v", args, err)
	}
	for _, target := range []string{"sm_103a", "sm_103f"} {
		if _, _, err := ParseCUDAArchMatrix(matrix, target); err == nil {
			t.Fatalf("unexpected architecture-specific target admission: %s", target)
		}
	}
	for _, tc := range []struct {
		version string
		wantOK  bool
	}{
		{"Cuda compilation tools, release 12.8, V12.8.93", false},
		{"nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 12.9, V12.9.41\nBuild cuda_12.9.r12.9/compiler.36037853_0", true},
		{"Cuda compilation tools, release 13.0, V13.0.48\r\n", true},
		{"CUDA Version: 12.9", false},
		{"", false},
	} {
		if err := requireSM103NVCCVersion(tc.version); (err == nil) != tc.wantOK {
			t.Errorf("version %q: err=%v, wantOK=%v", tc.version, err, tc.wantOK)
		}
	}
}
