//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=1s lane=optin
// Qualification needs FAK_V41_ROUTED_PANEL_ROW=1 before initialization and
// FAK_VULKAN_REQUIRE_DEVICE=1. Skips are not physical witnesses.
func TestVulkanV41RoutedDownPanelRowParity(t *testing.T) {
	v := vk(t)
	const width = 65
	for _, rows := range []int{1, 3, 8, 64} {
		for _, poison := range []bool{false, true} {
			func() {
				row := rows - 1
				bits := []uint32{0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f800000, 0x3f808000, 0x3f818000, 0xbf808000, 0x7f7f0000}
				data := make([]float32, rows*width)
				for i := range data {
					data[i] = math.Float32frombits(bits[i%len(bits)])
				}
				if rows > 1 {
					data[0] = float32(math.NaN())
				}
				if poison {
					data[row*width+7] = math.Float32frombits(0x7f7fffff)
					data[row*width+11] = float32(math.NaN())
				}
				panel := v.Upload(NewF32(Default(), []int{rows, width}, data), F32)
				defer v.Free(panel)
				if !v.SupportsV41RoutedDownPanelRow(panel, row) {
					if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
						t.Fatal("required marked down-row module unavailable")
					}
					t.Skip("marked down-row module unavailable")
				}
				scalar := v.Upload(NewF32(Default(), []int{width}, data[row*width:(row+1)*width]), F32)
				defer v.Free(scalar)
				read := func(direct bool) ([]float32, error) {
					var out Tensor
					var st V41DeferredStatus
					var e error
					if direct {
						out, st, e = v.V41RoutedDownPanelRowDeferred(panel, row)
					} else {
						out, st, e = v.V41BF16PublishDeferred(scalar, V41BF16PublishRoutedDown)
					}
					if e != nil {
						t.Fatal(e)
					}
					defer v.Free(out)
					if st.Kind != V41DeferredBF16Publish || st.Count != width || st.Site != V41BF16PublishRoutedDown {
						v.Free(st.Record)
						t.Fatal("wrong fixed row status")
					}
					var outputs []Tensor
					if !poison {
						outputs = []Tensor{out}
					}
					verdicts, values, e := v.V41ReadDeferred([]V41DeferredStatus{st}, outputs)
					if e != nil || len(verdicts) != 1 {
						t.Fatalf("read %v", e)
					}
					if poison {
						return nil, verdicts[0]
					}
					if len(values) != 1 || len(values[0]) != width {
						t.Fatal("row extent")
					}
					return values[0], verdicts[0]
				}
				got, ge := read(true)
				want, we := read(false)
				if poison {
					var a, b *V41BF16CheckedArithmeticError
					if !errors.As(ge, &a) || !errors.As(we, &b) || a.Index != 7 || *a != *b {
						t.Fatalf("row-local fault %v / %v", ge, we)
					}
				} else {
					if ge != nil || we != nil {
						t.Fatalf("publication %v / %v", ge, we)
					}
					for i := range got {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("rows%d lane%d exact bits differ", rows, i)
						}
					}
				}
			}()
		}
	}
}
