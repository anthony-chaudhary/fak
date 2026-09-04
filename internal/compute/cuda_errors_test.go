package compute

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestCUDAOpError_FormatAndUnwrap(t *testing.T) {
	cause := errors.New("underlying driver failure")
	err := &CUDAOpError{
		Op:    "MatMul",
		Site:  "gemv-row-check",
		Msg:   "dimension mismatch",
		Code:  42,
		Class: MemoryActivation,
		Err:   cause,
	}

	str := err.Error()
	if !strings.Contains(str, "compute: cuda MatMul: dimension mismatch") {
		t.Fatalf("unexpected message formatting: %s", str)
	}
	if !strings.Contains(str, "site: gemv-row-check") {
		t.Fatalf("site missing from error string: %s", str)
	}
	if !strings.Contains(str, "code 42") {
		t.Fatalf("code missing from error string: %s", str)
	}
	if !strings.Contains(str, "class: activation") {
		t.Fatalf("class missing from error string: %s", str)
	}

	if !errors.Is(err, cause) {
		t.Fatalf("errors.Is failed to match underlying cause")
	}
	if err.Unwrap() != cause {
		t.Fatalf("Unwrap returned %v, want %v", err.Unwrap(), cause)
	}

	// Test nil error
	var nilErr *CUDAOpError
	if nilErr.Error() != "compute: nil cuda op error" {
		t.Fatalf("nil error string = %q", nilErr.Error())
	}
	if nilErr.Unwrap() != nil {
		t.Fatalf("nil Unwrap = %v", nilErr.Unwrap())
	}
}

func TestCUDALaunchError_AsCUDAOpError(t *testing.T) {
	launchErr := &CUDALaunchError{
		CUDAOpError: CUDAOpError{
			Op:   "GraphEndLaunch",
			Site: "cuda-graph-launch",
			Msg:  "cuda graph capture/launch failed",
			Code: 2,
		},
	}

	var opErr *CUDAOpError
	if !errors.As(launchErr, &opErr) {
		t.Fatalf("errors.As(*CUDAOpError) failed on *CUDALaunchError")
	}
	if opErr.Op != "GraphEndLaunch" || opErr.Code != 2 {
		t.Fatalf("opErr fields mismatch: %+v", opErr)
	}

	var nilLaunch *CUDALaunchError
	if nilLaunch.Error() != "compute: nil cuda launch error" {
		t.Fatalf("nil launch error string = %q", nilLaunch.Error())
	}
	if nilLaunch.Unwrap() != nil {
		t.Fatalf("nil launch Unwrap = %v", nilLaunch.Unwrap())
	}
}

func TestCUDAError_Aliases(t *testing.T) {
	var err error = &CUDAError{
		Op:   "UploadConstantParam",
		Site: "upload-constant-param",
		Msg:  "destination buffer too small",
	}

	var backendErr *CUDABackendError
	if !errors.As(err, &backendErr) {
		t.Fatalf("errors.As(*CUDABackendError) failed on *CUDAError")
	}
	if backendErr.Op != "UploadConstantParam" {
		t.Fatalf("backendErr.Op = %s, want UploadConstantParam", backendErr.Op)
	}
}

func TestAsCUDAError_Conversion(t *testing.T) {
	// 1. Direct *CUDAOpError
	orig := &CUDAOpError{Op: "MatMul", Site: "test", Msg: "failed"}
	got, ok := AsCUDAError(orig)
	if !ok || got != orig {
		t.Fatalf("AsCUDAError(*CUDAOpError) = (%v, %v), want (%v, true)", got, ok, orig)
	}

	// 2. Direct *CUDALaunchError
	launch := &CUDALaunchError{CUDAOpError: *orig}
	got, ok = AsCUDAError(launch)
	if !ok || got.Op != orig.Op {
		t.Fatalf("AsCUDAError(*CUDALaunchError) = (%v, %v)", got, ok)
	}

	// 3. Wrapped *CUDAOpError
	wrapped := fmt.Errorf("wrapped: %w", orig)
	got, ok = AsCUDAError(wrapped)
	if !ok || got != orig {
		t.Fatalf("AsCUDAError(wrapped) = (%v, %v), want (%v, true)", got, ok, orig)
	}

	// 4. Bare strings for request-reachable CUDA failures
	cases := []struct {
		input string
		wantOp string
		wantSite string
	}{
		{"compute: cuda graph capture/launch failed", "GraphEndLaunch", "cuda-graph-launch"},
		{"compute: UploadConstantParam destination buffer too small", "UploadConstantParam", "upload-constant-param"},
		{"compute/cuda: sigmoid gate shape mismatch", "SigmoidMulInPlace", "sigmoid"},
		{"compute/cuda: Qwen query/gate projection shape mismatch", "SplitQwen35QueryGate", "split-qg"},
		{"compute: cuda MatMul produced wrong rows", "MatMul", "matmul"},
		{"compute: cuda BatchedMatMul P=1 produced 0", "BatchedMatMul", "batched-matmul"},
		{"compute: cuda Upload expects host data", "Upload", "upload"},
	}

	for _, tc := range cases {
		err, ok := AsCUDAError(tc.input)
		if !ok {
			t.Fatalf("AsCUDAError(%q) returned ok=false", tc.input)
		}
		if err.Op != tc.wantOp {
			t.Fatalf("AsCUDAError(%q).Op = %s, want %s", tc.input, err.Op, tc.wantOp)
		}
		if err.Site != tc.wantSite {
			t.Fatalf("AsCUDAError(%q).Site = %s, want %s", tc.input, err.Site, tc.wantSite)
		}

		// Also test as error wrapping string
		errFromErr, ok := AsCUDAError(errors.New(tc.input))
		if !ok || errFromErr.Op != tc.wantOp {
			t.Fatalf("AsCUDAError(error(%q)) = (%v, %v)", tc.input, errFromErr, ok)
		}
	}

	// 5. Non-CUDA failure should return false
	nonCUDACases := []any{
		"plain string panic",
		"index out of range [1] with length 1",
		errors.New("generic i/o timeout"),
		42,
		nil,
	}
	for _, non := range nonCUDACases {
		if _, ok := AsCUDAError(non); ok {
			t.Fatalf("AsCUDAError(%v) returned ok=true for non-CUDA failure", non)
		}
	}
}

func TestConvertCUDAPanic(t *testing.T) {
	// String panic converted with defaults
	err, ok := ConvertCUDAPanic("compute: cuda graph capture/launch failed", "GraphEndLaunch", "site-override")
	if !ok {
		t.Fatal("ConvertCUDAPanic returned ok=false")
	}
	var opErr *CUDAOpError
	if !errors.As(err, &opErr) || opErr.Op != "GraphEndLaunch" {
		t.Fatalf("ConvertCUDAPanic result = %T (%v)", err, err)
	}

	// DeviceAllocError preserved
	dae := &DeviceAllocError{Bytes: 1024, Site: "dalloc", Class: MemoryWeights}
	err, ok = ConvertCUDAPanic(dae, "op", "site")
	if !ok || err != dae {
		t.Fatalf("ConvertCUDAPanic(*DeviceAllocError) = (%v, %v)", err, ok)
	}

	// DeviceFaultError preserved
	dfe := &DeviceFaultError{Backend: "cuda", Site: "fault-site", Class: DeviceFaultExecution}
	err, ok = ConvertCUDAPanic(dfe, "op", "site")
	if !ok || err != dfe {
		t.Fatalf("ConvertCUDAPanic(*DeviceFaultError) = (%v, %v)", err, ok)
	}

	// Unclassified panic with op/site context
	err, ok = ConvertCUDAPanic("unexpected hardware fault", "KernelLaunch", "stream-sync")
	if !ok {
		t.Fatal("ConvertCUDAPanic with op/site returned ok=false")
	}
	if !errors.As(err, &opErr) || opErr.Op != "KernelLaunch" || opErr.Site != "stream-sync" {
		t.Fatalf("unexpected opErr: %+v", opErr)
	}
}

func TestRunGuardedCUDA(t *testing.T) {
	// 1. Success case
	ran := false
	err := RunGuardedCUDA("TestOp", "test-site", func() {
		ran = true
	})
	if err != nil || !ran {
		t.Fatalf("RunGuardedCUDA failed on normal path: err=%v, ran=%v", err, ran)
	}

	// 2. Panicking case with CUDA error
	err = RunGuardedCUDA("TestOp", "test-site", func() {
		panic("compute: cuda graph capture/launch failed")
	})
	if err == nil {
		t.Fatal("RunGuardedCUDA did not return error for panicking function")
	}
	var opErr *CUDAOpError
	if !errors.As(err, &opErr) {
		t.Fatalf("RunGuardedCUDA did not return *CUDAOpError: %T (%v)", err, err)
	}
	if opErr.Op != "GraphEndLaunch" {
		t.Fatalf("op = %s, want GraphEndLaunch", opErr.Op)
	}
}
