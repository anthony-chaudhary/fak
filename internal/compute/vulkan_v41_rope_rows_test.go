//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=1s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1. A skipped test is no
// physical witness. Tolerances match TestVulkanV41TailRoPEQK: the prefix is
// bit-exact; tail abs/rel bounds are 2e-6.
func TestVulkanV41TailRoPERows(t *testing.T) {
	v := vk(t)
	if !v.V41TailRoPERowsAvailable() {
		t.Fatal("registered current Vulkan build lacks the V4.1 tail RoPE rows panel")
	}
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	for _, dims := range [][3]int{{1, 7, 4}, {3, 9, 4}, {64, 512, 64}} {
		for _, batched := range []bool{false, true} {
			t.Run(fmt.Sprintf("r%d_d%d_rot%d_batch%t", dims[0], dims[1], dims[2], batched), func(t *testing.T) {
				rows, width, rotary := dims[0], dims[1], dims[2]
				x, table := make([]float32, rows*width), make([]float32, rotary)
				for i := range x {
					x[i] = float32((i%31)-15)*0.03125 + 0.00390625
				}
				x[0] = math.Float32frombits(0x80000000)
				x[1] = math.Float32frombits(0x7fc01234)
				for j := 0; j < rotary/2; j++ {
					angle := float64(j+1) * 0.37
					// The inverse output rotation passes the conjugate [-sin,cos].
					table[2*j], table[2*j+1] = -float32(math.Sin(angle)), float32(math.Cos(angle))
				}
				upload := func(shape []int, values []float32) Tensor {
					d := v.UploadClass(NewF32(Default(), shape, values), F32, MemoryActivation, "V4.1 RoPE rows witness input")
					t.Cleanup(func() { v.Free(d) })
					return d
				}
				dx, dt := upload([]int{rows, width}, x), upload([]int{rotary / 2, 2}, table)
				if rows == 3 && !batched {
					bad := dx
					bad.Shape = []int{rows * width}
					if out, err := v.V41TailRoPERows(bad, dt, rows, width, rotary); err == nil || out.Buf() != nil {
						t.Fatal("malformed rows panel did not fail before publishing an output")
					}
				}
				if batched {
					v.BeginBatch()
				}
				out, err := v.V41TailRoPERows(dx, dt, rows, width, rotary)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { v.Free(out) })
				if out.Buf() == dx.Buf() || out.Buf() == dt.Buf() {
					t.Fatal("rows operation aliased an input")
				}
				got := v.Read(out)
				if batched {
					v.FlushBatch()
				}
				want := v41TailRoPEReference(x, table, width, rotary)
				if len(got) != len(want) {
					t.Fatalf("output length=%d want=%d", len(got), len(want))
				}
				for i := range want {
					if i%width < width-rotary {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("prefix lane %d bits=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
						}
						continue
					}
					delta := math.Abs(float64(got[i]) - float64(want[i]))
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || delta > 2e-6+2e-6*math.Abs(float64(want[i])) {
						t.Fatalf("tail lane %d: got=%g want=%g delta=%g", i, got[i], want[i], delta)
					}
				}
				for i, value := range v.Read(dx) {
					if math.Float32bits(value) != math.Float32bits(x[i]) {
						t.Fatalf("input mutated at lane %d", i)
					}
				}
			})
		}
	}
}
