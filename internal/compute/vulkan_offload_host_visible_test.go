//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"bytes"
	"math"
	"os"
	"os/exec"
	"testing"
)

// On a small-carve-out UMA device, expert-ring pages live in mapped host-visible memory, so a
// page-in is one memcpy: no staging copy, no submit, and no flush of the open batch. Pages must
// still contract exactly like the resident expert and recycle freed buffers. The dispatch
// profile is pinned at process start, so the check runs in a child with it enabled.
// fak-test:runtime integration est=5s lane=optin
func TestVulkanOffloadHostVisiblePagesSkipBatchFlush(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestVulkanOffloadHostVisiblePagesChild$", "-test.count=1", "-test.v")
	cmd.Env = append(os.Environ(), "FAK_VULKAN_DISPATCH_PROFILE=1", "FAK_VULKAN_OFFLOAD_HV_CHILD=1")
	out, err := cmd.CombinedOutput()
	if bytes.Contains(out, []byte("--- SKIP")) {
		t.Skipf("child skipped: %s", out)
	}
	if err != nil {
		t.Fatalf("child failed: %v\n%s", err, out)
	}
	t.Logf("%s", out)
}

func TestVulkanOffloadHostVisiblePagesChild(t *testing.T) {
	if os.Getenv("FAK_VULKAN_OFFLOAD_HV_CHILD") != "1" {
		t.Skip("child of TestVulkanOffloadHostVisiblePagesSkipBatchFlush")
	}
	v := vk(t)
	prev, applied := v.VulkanDebugSetOffloadHostVisible(true)
	defer v.VulkanDebugSetOffloadHostVisible(prev)
	if !applied {
		t.Skip("device has no mapped UMA weight path; host-visible expert pages are integrated-GPU only")
	}
	const out, in, experts, ring, rounds = 64, 512, 4, 2, 3
	x := q2FusedUpload(t, v, q2FusedInput(in, 0.1))
	hosts := make([]Tensor, experts)
	want := make([][]float32, experts)
	for e := range hosts {
		hosts[e] = vulkanOffloadQ2K(int64(9300+e), out, in)
		resident := q2FusedUpload(t, v, hosts[e])
		want[e] = v.Read(v.MatMul(resident, x))
	}
	pages0 := v.VulkanOffloadHostVisiblePages()
	direct0 := VulkanDirectWeightUploadBytes()
	pageBytes := uint64(len(hosts[0].buf.(HostBuffer).I8()))

	var pages []Tensor
	defer func() {
		for _, page := range pages {
			v.Free(page)
		}
	}()
	for round := 0; round < rounds; round++ {
		for e := range hosts {
			v.BeginBatch()
			before := v.VulkanDebugDispatchProfileSnapshot()
			page := v.UploadClass(hosts[e], Q2_K, MemoryOffload, "test expert ring page")
			after := v.VulkanDebugDispatchProfileSnapshot()
			v.FlushBatch()
			pages = append(pages, page)
			if after.BatchFlushes != before.BatchFlushes || after.OneShotSubmits != before.OneShotSubmits {
				t.Fatalf("round %d expert %d: page-in inside an open batch flushed %d times and made %d one-shot submits, want 0 and 0",
					round, e, after.BatchFlushes-before.BatchFlushes, after.OneShotSubmits-before.OneShotSubmits)
			}
			if !v.debugBufferHostVisible(page.buf.(*vulkanBuf)) {
				t.Fatalf("round %d expert %d: page is not host-visible", round, e)
			}
			got := v.Read(v.MatMul(page, x))
			for i := range want[e] {
				if math.IsNaN(float64(got[i])) || got[i] != want[e][i] {
					t.Fatalf("round %d expert %d element %d: host-visible page=%g resident=%g", round, e, i, got[i], want[e][i])
				}
			}
			if len(pages) > ring {
				v.Free(pages[0])
				pages = pages[1:]
			}
		}
	}
	pageIns := uint64(rounds * experts)
	if fresh := v.VulkanOffloadHostVisiblePages() - pages0; fresh == 0 || fresh > ring+1 {
		t.Fatalf("host-visible pages allocated = %d over %d page-ins, want 1..%d (freed pages recycled)", fresh, pageIns, ring+1)
	}
	if direct := VulkanDirectWeightUploadBytes() - direct0; direct != pageIns*pageBytes {
		t.Fatalf("direct upload bytes = %d, want %d (%d page-ins of %d bytes through the mapping)", direct, pageIns*pageBytes, pageIns, pageBytes)
	}
	t.Logf("device=%s page-ins=%d host-visible allocations=%d", v.Tier(), pageIns, v.VulkanOffloadHostVisiblePages()-pages0)
}
