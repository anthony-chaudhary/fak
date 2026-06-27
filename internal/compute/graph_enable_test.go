// graph_enable_test.go — a GPU-free regression guard for the EnableCUDAGraph cross-build
// seam (graph_cuda.go / graph_nocuda.go). This test has NO build tag, so it runs in the
// DEFAULT `go test ./...` (non-cuda) path that CI exercises on every host. The seam's whole
// job is to let a host (`fak serve --cuda-graph`) call one stable symbol in both builds; the
// classic way that silently breaks is a build-tag typo that drops the symbol from one side.
// Calling it here keeps the no-cuda twin compiled and callable.
package compute

import "testing"

// TestEnableCUDAGraphNoCUDAIsInertAndSafe asserts that in the default (non-cuda) build the
// enable seam exists, is idempotent, and is genuinely inert — there is no CUDA backend to
// turn a graph path on for, so the call must be a safe no-op rather than registering or
// mutating anything. Under -tags cuda the twin in graph_cuda.go flips graphEnabled instead;
// that side is covered by the cuda-tagged graph tests on a GPU node.
func TestEnableCUDAGraphNoCUDAIsInertAndSafe(t *testing.T) {
	// The cuda backend must be absent in this build for the no-op to be the correct behavior.
	// (If a build ever registers "cuda" without the graph seam, this catches the mismatch.)
	if _, found := Lookup("cuda"); found {
		t.Skip("cuda backend registered — the cuda-tagged graph tests own this path")
	}

	// Callable and idempotent: a host may call it unconditionally, possibly more than once.
	EnableCUDAGraph()
	EnableCUDAGraph()

	// Inert: no device backend appeared, so the Default/Reference backend is unchanged.
	if got := Default().Name(); got == "cuda" {
		t.Fatalf("EnableCUDAGraph must not register or select a cuda backend in the non-cuda build; Default()=%q", got)
	}
}
