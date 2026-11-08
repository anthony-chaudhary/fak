package compute

import (
	"os"
	"regexp"
	"strconv"
	"testing"
)

// TestDequantGEMMScratchRowsMatchesKernelDispatch cross-pins every
// `if (P >= N)` dequant-GEMM dispatch literal in cuda/quant_matmul.cuh to
// DequantGEMMScratchRows (capacity.go): a drifted or newly added dispatch
// literal must fail here instead of silently re-pricing scratch capacity
// against a moved boundary.
func TestDequantGEMMScratchRowsMatchesKernelDispatch(t *testing.T) {
	src, err := os.ReadFile("cuda/quant_matmul.cuh")
	if err != nil {
		t.Fatalf("read cuda/quant_matmul.cuh: %v", err)
	}
	re := regexp.MustCompile(`(?m)^\s*if \(P >= (\d+)`)
	sites := re.FindAllStringSubmatch(string(src), -1)
	if len(sites) == 0 {
		t.Fatal("no `if (P >= N)` dispatch sites matched in cuda/quant_matmul.cuh; the dispatch shape or the file path moved - re-pin this test")
	}
	for _, s := range sites {
		n, err := strconv.Atoi(s[1])
		if err != nil {
			t.Fatalf("parse dispatch literal %q: %v", s[1], err)
		}
		if n != DequantGEMMScratchRows {
			t.Errorf("cuda/quant_matmul.cuh dispatch literal P >= %d != DequantGEMMScratchRows (%d): drifted boundary", n, DequantGEMMScratchRows)
		}
	}
}
