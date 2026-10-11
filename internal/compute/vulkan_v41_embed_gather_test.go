//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
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

// fak-test:runtime integration est=1s lane=optin
// Requires the optional v41_q2k_embed_gather.spv module in FAK_VULKAN_SPIRV
// (outside the trusted shader registry); FAK_VULKAN_REQUIRE_V41_Q2K_EMBED_GATHER=1
// makes its absence fatal. The panel must equal the host q2kDequantSuperBlock of
// the selected rows bit for bit, signed zeros included.
func TestVulkanV41Q2KEmbeddingGather(t *testing.T) {
	v := vk(t)
	if !v.V41Q2KEmbeddingGatherAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_V41_Q2K_EMBED_GATHER") == "1" {
			t.Fatal("required v41_q2k_embed_gather.spv not loaded")
		}
		t.Skip("optional V4.1 Q2_K embedding gather module not loaded")
	}
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	const vocab, hidden = 53, 1280
	rng := rand.New(rand.NewSource(13668))
	blocksPerRow := hidden / q2kSuper
	raw := make([]byte, vocab*blocksPerRow*q2kSuperBlock)
	halves := []uint16{0x3c00, 0xbc00, 0x2e66, 0xb555, 0x0001, 0x8000, 0x0000, 0x3555, 0x1400, 0x7bff}
	for b := 0; b < vocab*blocksPerRow; b++ {
		blk := raw[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
		rng.Read(blk[:80])
		if b%3 == 0 {
			for i := 0; i < 16; i++ {
				blk[i] &= 0xf0
			}
		}
		binary.LittleEndian.PutUint16(blk[80:], halves[rng.Intn(len(halves))])
		binary.LittleEndian.PutUint16(blk[82:], halves[rng.Intn(len(halves))])
	}
	dt := v.Upload(NewQ2K(Default(), []int{vocab, hidden}, raw), Q2_K)
	t.Cleanup(func() { v.Free(dt) })
	for _, ids := range [][]int{{0}, {52, 7, 7, 0, 31}, idsRange(vocab)} {
		for _, batched := range []bool{false, true} {
			t.Run(fmt.Sprintf("rows%d_batch%t", len(ids), batched), func(t *testing.T) {
				if batched {
					v.BeginBatch()
				}
				out, err := v.V41Q2KEmbeddingGather(dt, ids, hidden)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { v.Free(out) })
				got := v.Read(out)
				if batched {
					v.FlushBatch()
				}
				want := make([]float32, hidden)
				for row, id := range ids {
					for b := 0; b < blocksPerRow; b++ {
						off := (id*blocksPerRow + b) * q2kSuperBlock
						q2kDequantSuperBlock(want[b*q2kSuper:], raw[off:off+q2kSuperBlock])
					}
					for d := 0; d < hidden; d++ {
						if math.Float32bits(got[row*hidden+d]) != math.Float32bits(want[d]) {
							t.Fatalf("row %d (id %d) lane %d bits=%08x want=%08x", row, id, d, math.Float32bits(got[row*hidden+d]), math.Float32bits(want[d]))
						}
					}
				}
			})
		}
	}
	if out, err := v.V41Q2KEmbeddingGather(dt, []int{vocab}, hidden); err == nil || out.Buf() != nil {
		t.Fatal("out-of-range id produced a panel")
	}
}

func idsRange(n int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = n - 1 - i
	}
	return ids
}
