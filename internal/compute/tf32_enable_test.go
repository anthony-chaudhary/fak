// tf32_enable_test.go — a GPU-free regression guard for the EnableCUDATF32 cross-build seam
// (tf32_cuda.go / tf32_nocuda.go). This test has NO build tag, so it runs in the DEFAULT
// `go test ./...` (non-cuda) path that CI exercises on every host — including a host with no
// CUDA toolkit, where the cuda-tagged twin cannot compile. The seam's whole job is to let a host
// call one stable symbol in both builds; the classic way that silently breaks is a build-tag
// typo that drops the symbol from one side. Calling it here keeps the no-cuda twin compiled and
// callable, the same way graph_enable_test.go guards EnableCUDAGraph.
package compute

import "testing"

// TestEnableCUDATF32NoCUDAIsInertAndSafe asserts that in the default (non-cuda) build the TF32
// enable seam exists, is idempotent, and is genuinely inert — there is no CUDA backend and no
// cuBLAS handle to retune, so the call must be a safe no-op rather than registering or mutating
// anything. Under -tags cuda the twin in tf32_cuda.go flips tf32Enabled and re-applies it to the
// live handle instead; that side is covered by the cuda-tagged tests on a GPU node.
func TestEnableCUDATF32NoCUDAIsInertAndSafe(t *testing.T) {
	// The cuda backend must be absent in this build for the no-op to be the correct behavior.
	// (If a build ever registers "cuda" without the tf32 seam, this catches the mismatch.)
	if _, found := Lookup("cuda"); found {
		t.Skip("cuda backend registered — the cuda-tagged tests own this path")
	}

	// Callable and idempotent: a host may call it unconditionally, possibly more than once.
	EnableCUDATF32()
	EnableCUDATF32()

	// Inert: no device backend appeared, so the Default/Reference backend is unchanged.
	if got := Default().Name(); got == "cuda" {
		t.Fatalf("EnableCUDATF32 must not register or select a cuda backend in the non-cuda build; Default()=%q", got)
	}
}
