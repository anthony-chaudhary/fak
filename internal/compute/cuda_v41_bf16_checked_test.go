//go:build cuda && cgo && fak_cuda_v41_bf16_publish

package compute

import (
	"errors"
	"math"
	"os"
	"testing"
)

// This witness requires the existing native FAK_CUDA_V41_BF16_PUBLISH build
// option as well as the matching Go tag. No hardware evidence is implied by
// compiling this test or by skipping it when CUDA is unavailable.
// fak-test:runtime integration est=1s lane=optin
func TestCUDAV41FinalNormChecked(t *testing.T) {
	var absent *cudaBackend
	if absent.V41BF16CheckedAvailable() {
		t.Fatal("nil backend advertised final-norm publication")
	}
	// Refuse non-final-norm sites before touching backend/device state, even
	// when the broader publication primitive would accept that site.
	metadata := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{1}}
	if _, err := absent.V41BF16Checked(metadata, V41BF16PublishFFNNormInput); err == nil {
		t.Fatal("final-norm entry admitted an FFN site")
	}
	backend, ok := Lookup("cuda")
	if !ok || backend == nil {
		if os.Getenv("FAK_CUDA_V41_REQUIRED") == "1" {
			t.Fatal("FAK_CUDA_V41_REQUIRED=1: real CUDA backend required")
		}
		t.Skip("CUDA backend is unavailable")
	}
	c, ok := backend.(*cudaBackend)
	if !ok || !c.V41BF16CheckedAvailable() {
		t.Fatalf("CUDA final-norm capability unavailable on %T", backend)
	}
	words := []uint32{0, 0x80000000, 1, 0x00008000, 0x00018000, 0x3f808000, 0x3f818000, 0xbf818000}
	values := make([]float32, len(words))
	for i, word := range words {
		values[i] = math.Float32frombits(word)
	}
	input := c.Upload(NewF32(c, []int{len(values)}, values), F32)
	defer c.Free(input)
	for _, site := range []uint32{V41BF16CheckedFinalNormInput, V41BF16CheckedFinalNormOutput} {
		out, err := c.V41BF16Checked(input, site)
		if err != nil {
			t.Fatal(err)
		}
		if out.buf == input.buf {
			t.Fatal("checked output aliases input")
		}
		got := c.Read(out)
		c.Free(out)
		for i, word := range words {
			want := (word + 0x7fff + ((word >> 16) & 1)) & 0xffff0000
			if math.Float32bits(got[i]) != want {
				t.Fatalf("site %d word %d: got %08x want %08x", site, i, math.Float32bits(got[i]), want)
			}
		}
	}
	bad := c.Upload(NewF32(c, []int{2}, []float32{1, math.Float32frombits(0x7f7f8000)}), F32)
	defer c.Free(bad)
	out, err := c.V41BF16Checked(bad, V41BF16CheckedFinalNormOutput)
	var arithmetic *V41BF16CheckedArithmeticError
	if !errors.As(err, &arithmetic) || arithmetic.Site != V41BF16CheckedFinalNormOutput || arithmetic.Index != 1 || out.buf != nil {
		t.Fatalf("rounded overflow must withhold output and preserve site/index: out=%v err=%v", out.Shape, err)
	}
}
