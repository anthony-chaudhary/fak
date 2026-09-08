package compute

import (
	"os"
	"strings"
	"testing"
)

// TestCUDAFiniteReductionSentinelContract keeps the physical edge regressions
// reachable from default CI: every maximum accumulator in this translation unit
// must admit the entire finite float32 range.
func TestCUDAFiniteReductionSentinelContract(t *testing.T) {
	source, err := os.ReadFile("cuda_kernels.cu")
	if err != nil {
		t.Fatalf("read cuda_kernels.cu: %v", err)
	}
	text := string(source)
	if strings.Contains(text, "-1e30f") {
		t.Fatal("cuda_kernels.cu retains a finite -1e30f reduction sentinel")
	}
	for want, count := range map[string]int{
		"float lm = -INFINITY;":             1,
		"float m = -INFINITY, l = 0.f;":     2,
		"float bv = -INFINITY; int bi = 0;": 1,
	} {
		if got := strings.Count(text, want); got != count {
			t.Errorf("cuda_kernels.cu true-bound contract %q count = %d, want %d", want, got, count)
		}
	}
}
