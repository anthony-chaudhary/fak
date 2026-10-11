//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"reflect"
	"testing"
)

// fak-test:runtime integration est=2s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_V41_INDEX_SELECT=1 and the optional
// v41_index_select.spv module in FAK_VULKAN_SPIRV (it is outside the frozen
// V5/complete62 registry, so the default suite skips its absence). A skipped
// test is no physical witness. Selection is integer-keyed, so the device must
// match V41IndexSelectReference exactly.
func TestVulkanV41IndexSelect(t *testing.T) {
	v := vk(t)
	if !v.V41IndexSelectAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_V41_INDEX_SELECT") == "1" {
			t.Fatal("required V4.1 index-select witness: v41_index_select.spv not loaded")
		}
		t.Skip("optional V4.1 index-select module not loaded")
	}
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	rng := rand.New(rand.NewSource(13668))
	negZero := float32(math.Copysign(0, -1))
	ninf, inf := float32(math.Inf(-1)), float32(math.Inf(1))
	for _, rows := range []int{1, 5, 257, 1031, 4099} {
		scores := make([]float32, rows)
		for i := range scores {
			switch rng.Intn(13) {
			case 0:
				scores[i] = ninf
			case 1:
				scores[i] = negZero
			case 2:
				scores[i] = inf
			default:
				scores[i] = float32(rng.Intn(9)-4) * 0.25 // few values: dense ties
			}
		}
		for _, geometry := range [][3]int{{1, 0, 0}, {7, 0, 0}, {512, 0, 0}, {64, 2, 8}, {2048, 16, 8}, {9, 1, 1}} {
			for _, batched := range []bool{false, true} {
				topK, topKBlocks, blockSize := geometry[0], geometry[1], geometry[2]
				t.Run(fmt.Sprintf("rows%d_k%d_b%dx%d_batch%t", rows, topK, topKBlocks, blockSize, batched), func(t *testing.T) {
					want, err := V41IndexSelectReference(scores, topK, topKBlocks, blockSize)
					if err != nil {
						t.Fatal(err)
					}
					d := v.UploadClass(NewF32(Default(), []int{rows}, scores), F32, MemoryActivation, "V4.1 index-select witness scores")
					t.Cleanup(func() { v.Free(d) })
					if batched {
						v.BeginBatch()
					}
					out, err := v.V41IndexSelect(d, rows, topK, topKBlocks, blockSize)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() { v.Free(out) })
					values := v.Read(out)
					if batched {
						v.FlushBatch()
					}
					got := make([]int32, len(values))
					for i, value := range values {
						got[i] = int32(value)
						if float32(got[i]) != value {
							t.Fatalf("slot %d is not an exact integer: %g", i, value)
						}
					}
					if !reflect.DeepEqual(got, want) {
						t.Fatalf("device selection differs from the reference:\n got=%v\nwant=%v", got, want)
					}
				})
			}
		}
	}
	t.Run("nan-refused", func(t *testing.T) {
		d := v.UploadClass(NewF32(Default(), []int{3}, []float32{1, float32(math.NaN()), 2}), F32, MemoryActivation, "V4.1 index-select witness NaN")
		t.Cleanup(func() { v.Free(d) })
		if out, err := v.V41IndexSelect(d, 3, 2, 0, 0); err == nil || out.Buf() != nil {
			t.Fatal("NaN score produced a selection")
		}
	})
}
