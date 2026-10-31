package compute

import (
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// This is the real shared admission validator, not a simulated CUDA result.
func TestCUDAQ2KRejectsMalformedBeforeSubmission(t *testing.T) {
	weight := func() Tensor {
		return Tensor{Dtype: Q2_K, Layout: RowMajor, Shape: []int{128, 256}, Quant: &QuantSpec{Block: 256, Bits: 2, Axis: 2}}
	}
	input := func() Tensor { return Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{128, 256}} }
	const wn, xn = 128 * 84, 128 * 256 * 4
	if err := cudaQ2KCheckCapacity(weight(), input(), 128, true, wn, xn); err != nil {
		t.Fatal(err)
	}
	xv := input()
	xv.Shape = []int{256}
	if err := cudaQ2KCheckCapacity(weight(), xv, 1, false, wn, 256*4); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		edit   func(*Tensor, *Tensor)
		wb, xb int
	}{
		{"f16", func(w, x *Tensor) { x.Dtype = F16 }, wn, xn},
		{"shape", func(w, x *Tensor) { x.Shape = []int{127, 256} }, wn, xn},
		{"short-x", func(w, x *Tensor) {}, wn, xn - 1},
		{"short-w", func(w, x *Tensor) {}, wn - 1, xn},
		{"weight-rank", func(w, x *Tensor) { w.Shape = []int{256} }, wn, xn},
		{"block", func(w, x *Tensor) { w.Quant.Block = 32 }, wn, xn},
		{"k-tail", func(w, x *Tensor) { w.Shape[1] = 255 }, wn, xn},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, x := weight(), input()
			tc.edit(&w, &x)
			if err := cudaQ2KCheckCapacity(w, x, 128, true, tc.wb, tc.xb); err == nil {
				t.Fatal("malformed operand admitted")
			}
		})
	}
	for _, dims := range [][3]int{{131072, 16384, 128}, {1, 512, 2147483647}} {
		w, x := weight(), input()
		w.Shape = []int{dims[0], dims[1]}
		x.Shape = []int{dims[2], dims[1]}
		if _, _, err := cudaQ2KOperandBytes(w, x, dims[2], true); err == nil {
			t.Fatal("overflowing legacy launch admitted")
		}
	}
	// Source ordering complements the executed pure validator: both real CUDA
	// entry points must call it through the resident wrapper before trace or
	// allocation. This does not claim CUDA execution or device correctness.
	b, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	for _, name := range []string{"MatMul", "BatchedMatMul"} {
		start := strings.Index(s, "func (c *cudaBackend) "+name+"(")
		if start < 0 {
			t.Fatal(name)
		}
		body := s[start:]
		end := strings.Index(body, "\nfunc ")
		if end >= 0 {
			body = body[:end]
		}
		guard := strings.Index(body, "c.checkQ2KOperands(")
		for _, marker := range []string{"C.fcuda_event_elapsed_ms_start()", "c.devTr(", "out, in = w.Shape[0], w.Shape[1]"} {
			if at := strings.Index(body, marker); guard < 0 || at < 0 || guard >= at {
				t.Fatalf("%s guard must precede %s", name, marker)
			}
		}
	}
	b, err = os.ReadFile("cuda_q2k_admission.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "cudaQ2KCheckCapacity(w, x, p, panel, wb.n, xb.n)") || strings.Contains(string(b), "C.") {
		t.Fatal("resident admission must use shared capacity validator without CUDA submission")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAQ2KSequenceAdmissionBeforeAllocation(t *testing.T) {
	b, err := os.ReadFile("cuda_kernels.go")
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	start := strings.Index(s, "func (c *cudaBackend) qwen35SequenceMatMulLocked(")
	if start < 0 {
		t.Fatal("missing sequence entry")
	}
	s = s[start:]
	guard := strings.Index(s, "c.validateQ2KOperands(w, x, tokens, tokens != 1 || len(x.Shape) != 1)")
	allocation := strings.Index(s, "c.qwen35SequenceAllocLocked(")
	if guard < 0 || allocation < 0 || guard >= allocation {
		t.Fatal("Q2 sequence validation must precede shape indexing and allocation")
	}
}
