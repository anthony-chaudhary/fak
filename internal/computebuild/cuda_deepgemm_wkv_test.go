package computebuild

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
func TestCUDADeepGEMMWKVBuildContract(t *testing.T) {
	if (CUDAConfig{}).EnableDeepGEMMWKV || (CUDAConfig{}).DeepGEMMWKVSMs != 0 {
		t.Fatal("candidate and physical SM count must be explicit")
	}
	for _, version := range []string{"unknown", "release 12.9", "release 13.0"} {
		if !errors.Is(requireCUDADeepGEMMVersion(version), ErrCUDADeepGEMMWKV) {
			t.Fatal(version)
		}
	}
	if err := requireCUDADeepGEMMVersion("Cuda compilation tools, release 13.1, V13.1.80"); err != nil {
		t.Fatal(err)
	}
	args := cudaDeepGEMMCompileArgs(&Toolchain{}, "/repo", "/out/wkv.o", 148)
	var images []string
	for i, arg := range args {
		if arg == "-gencode" {
			images = append(images, args[i+1])
		}
	}
	if !reflect.DeepEqual(images, []string{"arch=compute_100f,code=sm_100f", "arch=compute_103f,code=sm_103f"}) {
		t.Fatal(images)
	}
	joined := " " + strings.Join(args, " ") + " "
	for _, flag := range []string{"-std=c++20", "--compiler-options=-O3", "--compiler-options=-fPIC", "--compiler-options=-fconcepts", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--ptxas-options=--register-usage-level=10", "--diag-suppress=39,161,174,177,186,940", "--compiler-options=-Wno-deprecated-declarations,-Wno-abi", "-DFAK_DEEPGEMM_WKV_SMS=148", "-I/repo/third_party/deepgemm/deep_gemm/include", "-I/repo/third_party/deepgemm_cutlass/include"} {
		if !strings.Contains(joined, " "+flag+" ") {
			t.Errorf("missing %s", flag)
		}
	}
	if strings.Contains(joined, "code=compute_") || strings.Contains(joined, "--use_fast_math") || strings.Contains(joined, "sm_100a") {
		t.Fatal("unexpected fallback or arithmetic change")
	}
	raw, err := json.Marshal(CUDADeepGEMMWKVReceipt{})
	if err != nil || !strings.Contains(string(raw), `"hardware_qualified":false`) {
		t.Fatal(string(raw), err)
	}
}
