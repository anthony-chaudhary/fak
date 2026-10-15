//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=1s lane=optin
// A skip is not a device witness. Set FAK_V41_ROUTED_PANEL_ROW=1 before process
// initialization and FAK_VULKAN_REQUIRE_DEVICE=1 for physical qualification.
func TestVulkanV41RoutedPanelRowParity(t *testing.T) {
	v := vk(t)
	const width = 65
	for _, n := range []int{1, 3, 32, 64} {
		func() {
			gate, up := make([]float32, n*width), make([]float32, n*width)
			for i := range gate {
				gate[i] = float32(i%23-11) * .125
				up[i] = float32(i%17-8) * .25
			}
			gd := v.Upload(NewF32(Default(), []int{n, width}, gate), F32)
			defer v.Free(gd)
			ud := v.Upload(NewF32(Default(), []int{n, width}, up), F32)
			defer v.Free(ud)
			row := n - 1
			if !v.SupportsV41RoutedSwiGLUPanelRow(gd, ud, row) {
				if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
					t.Fatal("required marked panel-row module unavailable")
				}
				t.Skip("marked panel-row module unavailable")
			}
			if v.SupportsV41RoutedSwiGLUPanelRow(gd, ud, -1) || v.SupportsV41RoutedSwiGLUPanelRow(gd, ud, n) || v.SupportsV41RoutedSwiGLUPanelRow(gd, gd, row) {
				t.Fatal("invalid row or aliased panel admitted")
			}
			rg := v.Upload(NewF32(Default(), []int{width}, gate[row*width:]), F32)
			defer v.Free(rg)
			ru := v.Upload(NewF32(Default(), []int{width}, up[row*width:]), F32)
			defer v.Free(ru)
			read := func(panel bool) []float32 {
				var out Tensor
				var st V41DeferredStatus
				var err error
				if panel {
					out, st, err = v.V41RoutedSwiGLUPanelRowDeferred(gd, ud, row, .75, -.5)
				} else {
					out, st, err = v.V41RoutedSwiGLUDeferred(rg, ru, .75, -.5)
				}
				if err != nil {
					t.Fatal(err)
				}
				defer v.Free(out)
				if st.Kind != V41DeferredRoutedSwiGLU || st.Count != width || st.Site != 0 {
					v.Free(st.Record)
					t.Fatal("wrong row-local status")
				}
				verdicts, values, err := v.V41ReadDeferred([]V41DeferredStatus{st}, []Tensor{out})
				if err != nil || len(verdicts) != 1 || verdicts[0] != nil || len(values) != 1 || len(values[0]) != width {
					t.Fatalf("read: %v %v", err, verdicts)
				}
				return values[0]
			}
			got, want := read(true), read(false)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("rows=%d lane=%d row=%g scalar=%g", n, i, got[i], want[i])
				}
			}
		}()
	}
}

// fak-test:runtime integration est=1s lane=optin
func TestVulkanV41RoutedPanelRowFaultIndex(t *testing.T) {
	v := vk(t)
	const width = 65
	gate, up := make([]float32, 3*width), make([]float32, 3*width)
	for i := range gate {
		gate[i], up[i] = 1, 1
	}
	gate[2] = float32(math.NaN())
	gate[2*width+7] = float32(math.NaN())
	gd := v.Upload(NewF32(Default(), []int{3, width}, gate), F32)
	defer v.Free(gd)
	ud := v.Upload(NewF32(Default(), []int{3, width}, up), F32)
	defer v.Free(ud)
	if !v.SupportsV41RoutedSwiGLUPanelRow(gd, ud, 2) {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required marked panel-row module unavailable")
		}
		t.Skip("marked panel-row module unavailable")
	}
	out, st, err := v.V41RoutedSwiGLUPanelRowDeferred(gd, ud, 2, 10, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer v.Free(out)
	verdicts, _, err := v.V41ReadDeferred([]V41DeferredStatus{st}, nil)
	if err != nil || len(verdicts) != 1 {
		t.Fatalf("read: %v", err)
	}
	var fault *V41RoutedSwiGLUFault
	if !errors.As(verdicts[0], &fault) || fault.Index != 7 || fault.Stage != V41RoutedSwiGLUStageActivation {
		t.Fatalf("row-local fault: %v", verdicts[0])
	}
}
