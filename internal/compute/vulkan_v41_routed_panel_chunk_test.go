//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=1s lane=optin
// Missing device/module is a skip unless physical qualification is required.
func TestVulkanV41RoutedPanelChunkParity(t *testing.T) {
	v := vk(t)
	const width = 65
	for _, rows := range []int{2, 4, 8} {
		func() {
			const panel = 16
			first := panel - rows
			gate, up := make([]float32, panel*width), make([]float32, panel*width)
			for i := range gate {
				gate[i] = float32(i%23-11) * .125
				up[i] = float32(i%17-8) * .25
			}
			gd := v.Upload(NewF32(Default(), []int{panel, width}, gate), F32)
			defer v.Free(gd)
			ud := v.Upload(NewF32(Default(), []int{panel, width}, up), F32)
			defer v.Free(ud)
			if !v.SupportsV41RoutedSwiGLUPanelChunk(gd, ud, first, rows) {
				if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
					t.Fatal("required chunk module unavailable")
				}
				t.Skip("optional chunk module unavailable")
			}
			weights := make([]float32, rows)
			for i := range weights {
				weights[i] = float32(i-3) * .25
			}
			weights[0] = math.Float32frombits(0x80000000)
			out, st, err := v.V41RoutedSwiGLUPanelChunkDeferred(gd, ud, first, .75, weights)
			if err != nil {
				t.Fatal(err)
			}
			defer v.Free(out)
			if st.Records != rows || st.Count != width || st.Site != 0 || st.Kind != V41DeferredRoutedSwiGLU {
				t.Fatal("wrong chunk status descriptor")
			}
			verdicts, got, err := v.V41ReadDeferred([]V41DeferredStatus{st}, []Tensor{out})
			if err != nil || len(verdicts) != rows || len(got) != 1 || len(got[0]) != rows*width {
				t.Fatalf("read %v", err)
			}
			for row := 0; row < rows; row++ {
				func() {
					if verdicts[row] != nil {
						t.Fatal(verdicts[row])
					}
					offset := (first + row) * width
					g := v.Upload(NewF32(Default(), []int{width}, gate[offset:offset+width]), F32)
					defer v.Free(g)
					u := v.Upload(NewF32(Default(), []int{width}, up[offset:offset+width]), F32)
					defer v.Free(u)
					y, s, e := v.V41RoutedSwiGLUDeferred(g, u, .75, weights[row])
					if e != nil {
						t.Fatal(e)
					}
					defer v.Free(y)
					vs, want, e := v.V41ReadDeferred([]V41DeferredStatus{s}, []Tensor{y})
					if e != nil || len(vs) != 1 || vs[0] != nil {
						t.Fatalf("scalar %v", e)
					}
					for i, x := range want[0] {
						if math.Float32bits(x) != math.Float32bits(got[0][row*width+i]) {
							t.Fatalf("rows=%d row=%d lane=%d", rows, row, i)
						}
					}
				}()
			}
		}()
	}
}

// fak-test:runtime integration est=1s lane=optin
func TestVulkanV41RoutedPanelChunkFaultRows(t *testing.T) {
	v := vk(t)
	const width = 65
	for _, rows := range []int{2, 4, 8} {
		func() {
			gate, up := make([]float32, (rows+2)*width), make([]float32, (rows+2)*width)
			for i := range gate {
				gate[i] = 1
				up[i] = 1
			}
			gate[0] = float32(math.NaN())
			for row := 0; row < rows; row++ {
				gate[(row+1)*width+row+3] = float32(math.NaN())
			}
			gd := v.Upload(NewF32(Default(), []int{rows + 2, width}, gate), F32)
			defer v.Free(gd)
			ud := v.Upload(NewF32(Default(), []int{rows + 2, width}, up), F32)
			defer v.Free(ud)
			if !v.SupportsV41RoutedSwiGLUPanelChunk(gd, ud, 1, rows) {
				if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
					t.Fatal("required chunk unavailable")
				}
				t.Skip("optional chunk unavailable")
			}
			weights := make([]float32, rows)
			for i := range weights {
				weights[i] = 1
			}
			out, st, err := v.V41RoutedSwiGLUPanelChunkDeferred(gd, ud, 1, 10, weights)
			if err != nil {
				t.Fatal(err)
			}
			defer v.Free(out)
			vs, _, err := v.V41ReadDeferred([]V41DeferredStatus{st}, nil)
			if err != nil || len(vs) != rows {
				t.Fatalf("read %v", err)
			}
			for row, e := range vs {
				var f *V41RoutedSwiGLUFault
				if !errors.As(e, &f) || f.Index != row+3 || f.Stage != V41RoutedSwiGLUStageActivation {
					t.Fatalf("row %d: %v", row, e)
				}
			}
		}()
	}
}
