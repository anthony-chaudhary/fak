package compute

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

type vulkanOffloadWeightBackend interface {
	Backend
	UploadClass(Tensor, Dtype, MemoryClass, string) Tensor
	VulkanOffloadWeightCounters() (reused, fresh uint64)
	VulkanWeightArenaCounters() (memoryAllocations, bufferBindings, reservedBytes, liveBytes, peakReservedBytes uint64)
}

func vulkanOffloadQ2K(seed int64, out, in int) Tensor {
	rng := rand.New(rand.NewSource(seed))
	raw := make([]byte, out*(in/256)*84)
	_, _ = rng.Read(raw)
	for b := 0; b < len(raw); b += 84 {
		binaryPutFloat16(raw[b+80:b+82], 1.0/128)
		binaryPutFloat16(raw[b+82:b+84], 1.0/256)
	}
	return NewQ2K(Default(), []int{out, in}, raw)
}

// An expert ring pages more experts than it holds, evicting the oldest page on every miss.
// Evicted pages must hand their device buffer to the next same-size page-in instead of taking
// fresh bump-only arena bytes, which are released only when every arena buffer dies, so a
// long prefill would otherwise exhaust device memory. Every page, recycled or fresh, must
// still contract exactly like the same expert held in the arena.
// fak-test:runtime integration est=5s lane=optin
func TestVulkanOffloadWeightPagesReuseFreedBytes(t *testing.T) {
	be, ok := Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device is not registered")
		}
		t.Skip("Vulkan device is not registered; offload page reuse unverified")
	}
	v, ok := be.(vulkanOffloadWeightBackend)
	if !ok {
		t.Fatal("Vulkan backend lacks the classed upload or offload/arena counter contract")
	}
	const out, in, experts, ring, rounds = 64, 512, 6, 2, 4
	x := q2FusedUpload(t, v, q2FusedInput(in, 0.1))
	hosts := make([]Tensor, experts)
	want := make([][]float32, experts)
	for e := range hosts {
		hosts[e] = vulkanOffloadQ2K(int64(9100+e), out, in)
		resident := q2FusedUpload(t, v, hosts[e])
		want[e] = v.Read(v.MatMul(resident, x))
	}
	_, _, reserved0, live0, _ := v.VulkanWeightArenaCounters()
	reused0, fresh0 := v.VulkanOffloadWeightCounters()

	var pages []Tensor
	defer func() {
		for _, page := range pages {
			v.Free(page)
		}
	}()
	for round := 0; round < rounds; round++ {
		for e := range hosts {
			page := v.UploadClass(hosts[e], Q2_K, MemoryOffload, "test expert ring page")
			pages = append(pages, page)
			got := v.Read(v.MatMul(page, x))
			for i := range want[e] {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[e][i])) > 1e-5+1e-5*math.Abs(float64(want[e][i])) {
					t.Fatalf("round %d expert %d element %d: paged=%g resident=%g", round, e, i, got[i], want[e][i])
				}
			}
			if len(pages) > ring {
				v.Free(pages[0])
				pages = pages[1:]
			}
		}
	}

	_, _, reserved, live, _ := v.VulkanWeightArenaCounters()
	if reserved != reserved0 || live != live0 {
		t.Fatalf("expert pages touched the weight arena: reserved %d -> %d, live %d -> %d", reserved0, reserved, live0, live)
	}
	reused, fresh := v.VulkanOffloadWeightCounters()
	pageIns := uint64(rounds * experts)
	if reused-reused0+fresh-fresh0 != pageIns || fresh-fresh0 > ring+1 {
		t.Fatalf("page-ins reused=%d fresh=%d over %d page-ins, want at most %d fresh", reused-reused0, fresh-fresh0, pageIns, ring+1)
	}
	t.Logf("device=%s page-ins=%d reused=%d fresh=%d arena reserved=%d", v.Tier(), pageIns, reused-reused0, fresh-fresh0, reserved)
}
