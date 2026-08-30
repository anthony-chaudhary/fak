package compute

import (
	"os"
	"strings"
	"testing"
)

func TestMetalSourceExportsCommandReceiptToComputeTrace(t *testing.T) {
	data, err := os.ReadFile("metal.go")
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	for _, want := range []string{
		"metalMatMulTraceEvent",
		"recordMetalMatMulTrace",
		"requireMetalMatMulF32(\"MatMul\", w, x)",
		"requireMetalMatMulF32(\"BatchedMatMul\", w, X)",
		"w.Numel()*w.Dtype.Bytes() + x.Numel()*x.Dtype.Bytes()",
		"y.Numel() * y.Dtype.Bytes()",
		"DeviceDurationNS",
		"metal_command_buffer",
		"mps_f32_matmul",
		"receipt.TimingAvailable",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("metal.go missing %q", want)
		}
	}
}
