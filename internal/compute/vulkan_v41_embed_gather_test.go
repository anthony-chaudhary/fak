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
// physical witness. The gather is a byte copy, so every lane is bit-exact,
// including signed zero and a NaN payload.
func TestVulkanV41EmbeddingGather(t *testing.T) {
	v := vk(t)
	if !v.V41EmbeddingGatherAvailable() {
		t.Fatal("Vulkan backend lacks the V4.1 embedding gather")
	}
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	const vocab, hidden = 97, 256
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32((i%37)-18)*0.0625 + float32(i/hidden)*0.001
	}
	table[5*hidden] = math.Float32frombits(0x80000000)
	table[5*hidden+1] = math.Float32frombits(0x7fc01234)
	dt := v.Upload(NewF32(Default(), []int{vocab, hidden}, table), F32)
	t.Cleanup(func() { v.Free(dt) })
	for _, ids := range [][]int{{0}, {5, 96, 5, 0, 13}, {96}} {
		for _, batched := range []bool{false, true} {
			t.Run(fmt.Sprintf("ids%v_batch%t", ids, batched), func(t *testing.T) {
				if batched {
					v.BeginBatch()
				}
				out, err := v.V41EmbeddingGather(dt, ids, hidden)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { v.Free(out) })
				got := v.Read(out)
				if batched {
					v.FlushBatch()
				}
				if len(got) != len(ids)*hidden {
					t.Fatalf("panel length=%d want=%d", len(got), len(ids)*hidden)
				}
				for row, id := range ids {
					for d := 0; d < hidden; d++ {
						if math.Float32bits(got[row*hidden+d]) != math.Float32bits(table[id*hidden+d]) {
							t.Fatalf("row %d (id %d) lane %d bits=%08x want=%08x", row, id, d, math.Float32bits(got[row*hidden+d]), math.Float32bits(table[id*hidden+d]))
						}
					}
				}
			})
		}
	}
	if out, err := v.V41EmbeddingGather(dt, []int{vocab}, hidden); err == nil || out.Buf() != nil {
		t.Fatal("out-of-range id produced a panel")
	}
}
