package compute

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"runtime"
	"testing"
)

var cudaKQuantPromptPanelRequired = flag.Bool("fak-cuda-kquant-prompt-panel-required", false, "require real CUDA row-exact K-quant prompt-panel execution")

type cudaKQuantPromptPanelBackend interface {
	Backend
	PromptPanelRowExact(Dtype) bool
	PromptPanelBegin()
	PromptPanelChunk(int)
	PromptPanelEnd()
	PromptPanelMatMul(Tensor, Tensor, int) Tensor
}

// fak-test:runtime integration est=20s
func TestCUDAKQuantPromptPanelRowExact(t *testing.T) {
	if !*cudaKQuantPromptPanelRequired {
		t.Skip("select required CUDA K-quant prompt-panel execution")
	}
	raw, ok := Lookup("cuda")
	if !ok {
		t.Fatal("required native CUDA backend absent; CPU and zero-test fallbacks are refused")
	}
	be, ok := raw.(cudaKQuantPromptPanelBackend)
	if !ok {
		t.Fatal("required CUDA prompt-panel interface absent")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for _, dt := range []Dtype{Q4_K, Q6_K} {
		if !be.PromptPanelRowExact(dt) {
			t.Fatalf("%s row-exact panel not admitted", dt)
		}
		for _, in := range []int{256, 5120} {
			t.Run(fmt.Sprintf("%s-in%d", dt, in), func(t *testing.T) {
				const out, maxRows = 9, 513
				block := 144
				if dt == Q6_K {
					block = 210
				}
				codes := make([]byte, out*(in/256)*block)
				for i := range codes {
					codes[i] = byte((i*37 + 11) % 251)
				}
				for offset := 0; offset < len(codes); offset += block {
					if dt == Q4_K {
						binary.LittleEndian.PutUint16(codes[offset:], 0x3000)
						binary.LittleEndian.PutUint16(codes[offset+2:], 0x2c00)
					} else {
						binary.LittleEndian.PutUint16(codes[offset+208:], 0x3000)
					}
				}
				var weight Tensor
				if dt == Q4_K {
					weight = NewQ4K(Default(), []int{out, in}, codes)
				} else {
					weight = NewQ6K(Default(), []int{out, in}, codes)
				}
				w := be.Upload(weight, dt)
				defer be.Free(w)
				x := make([]float32, maxRows*in)
				for row := 0; row < maxRows; row++ {
					for k := 0; k < in; k++ {
						if row%11 != 3 {
							x[row*in+k] = float32(((row+3)*(k+7))%79-39) / 79
						}
					}
					if row%11 == 7 {
						x[row*in+(row*37)%in] = 4096
					}
				}
				want := make([]float32, maxRows*out)
				for row := 0; row < maxRows; row++ {
					dx := be.Upload(NewF32(Default(), []int{in}, x[row*in:(row+1)*in]), F32)
					dy := be.MatMul(w, dx)
					copy(want[row*out:], be.Read(dy))
					be.Free(dy)
					be.Free(dx)
				}
				for _, rows := range []int{1, 3, 4, 5, 127, 128, 129, 513} {
					t.Run(fmt.Sprintf("rows%d", rows), func(t *testing.T) {
						dx := be.Upload(NewF32(Default(), []int{rows, in}, x[:rows*in]), F32)
						defer be.Free(dx)
						be.PromptPanelBegin()
						defer be.PromptPanelEnd()
						be.PromptPanelChunk(rows)
						dy := be.PromptPanelMatMul(w, dx, rows)
						got := be.Read(dy)
						be.Free(dy)
						if len(got) != rows*out {
							t.Fatalf("output extent=%d want=%d", len(got), rows*out)
						}
						for i, v := range got {
							if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Float32bits(v) != math.Float32bits(want[i]) {
								t.Fatalf("row%d out%d panel=%08x decode=%08x", i/out, i%out, math.Float32bits(v), math.Float32bits(want[i]))
							}
						}
					})
				}
			})
		}
	}
}
