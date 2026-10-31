package compute

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDABlockscaledWKVTimestampLabels(t *testing.T) {
	raw, err := os.ReadFile("cuda/blockscaled_wkv.cuh")
	if err != nil {
		t.Fatal(err)
	}
	src := string(raw)
	start := strings.Index(src, `extern "C" int fcuda_blockscaled_wkv_run(`)
	if start < 0 {
		t.Fatal("missing blockscaled run entry point")
	}
	src = src[start:]
	end := strings.Index(src, "\n#else")
	if end < 0 {
		t.Fatal("missing blockscaled toolkit fallback boundary")
	}
	src = src[:end]
	want := []string{"fill", "k_mxfp8_cast_f32", "repack", "cublaslt_matmul", "check_output"}
	classes := []string{KernelOpCopy, KernelOpOther, KernelOpOther, KernelOpMatmul, KernelOpOther}
	valid := func(src string) bool {
		var names []string
		for _, m := range cudaTSWrap.FindAllStringSubmatch(src, -1) {
			names = append(names, m[1])
		}
		if !reflect.DeepEqual(names, want) {
			return false
		}
		for i, name := range names {
			if KernelOpClass(name) != classes[i] || (classes[i] == KernelOpOther && !cudaKernelOtherAllowlist[name]) {
				return false
			}
		}
		launches := cudaLaunch.FindAllStringSubmatchIndex(src, -1)
		blas := cudaBLASCall.FindAllStringSubmatchIndex(src, -1)
		async := cudaAsyncCopy.FindAllStringSubmatchIndex(src, -1)
		if len(launches) != 3 || len(blas) != 1 || len(async) != 1 {
			return false
		}
		for _, m := range launches {
			name := cudaLaunchTimingName("cuda/blockscaled_wkv.cuh", src[m[0]:m[1]], src[m[4]:m[5]])
			if cudaTSEnclosing(src, m[0]) != name {
				return false
			}
		}
		m := blas[0]
		if !cudaBLASTimingValid("cuda/blockscaled_wkv.cuh", src[m[2]:m[3]], cudaTSEnclosing(src, m[0])) {
			return false
		}
		m = async[0]
		name, ok := cudaAsyncTimingName(src[m[2]:m[3]], src[m[1]:])
		return ok && cudaTSEnclosing(src, m[0]) == name
	}
	if !valid(src) {
		t.Fatal("blockscaled WKV labels must preserve canonical transfer/kernel/BLAS rows and op classes")
	}
	legacy := []string{"blockscaled-wkv-status", "blockscaled-wkv-cast", "blockscaled-wkv-repack", "blockscaled-wkv-matmul", "blockscaled-wkv-finite"}
	for i, name := range want {
		for _, wrong := range []string{legacy[i], "unknown_row"} {
			t.Run(name+"/"+wrong, func(t *testing.T) {
				mutated := strings.Replace(src, `FCUDA_TS("`+name+`",`, `FCUDA_TS("`+wrong+`",`, 1)
				if mutated == src || valid(mutated) {
					t.Fatal("accepted a missing or noncanonical profiling label")
				}
			})
		}
	}
}
