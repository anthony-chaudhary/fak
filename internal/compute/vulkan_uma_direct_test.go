//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"bytes"
	"os"
	"os/exec"
	"testing"
)

// fak#13668: on UMA the dense base is written through a mapped DEVICE_LOCAL|HOST_VISIBLE weight
// arena with no staging copy. The device-type default is pinned per process at init, so the forced
// arm runs in a child process with FAK_VULKAN_UMA_DIRECT set.
// fak-test:runtime fast est=3s lane=default
func TestVulkanUMADirectWeightUploadSkipsStaging(t *testing.T) {
	runVulkanUMADirectChildArms(t, "TestVulkanUMADirectWeightUploadChild")
}

// fak#13668: a 512 MiB carve-out leaves the weights in the host-visible (GTT) fallback; with UMA
// direct upload on, that buffer is mapped too and the upload skips the staging copy.
// fak-test:runtime fast est=3s lane=default
func TestVulkanUMADirectHostVisibleWeightUploadSkipsStaging(t *testing.T) {
	runVulkanUMADirectChildArms(t, "TestVulkanUMADirectHostVisibleWeightChild")
}

func runVulkanUMADirectChildArms(t *testing.T, child string) {
	t.Helper()
	for _, arm := range []struct {
		env    string
		direct bool
	}{{"1", true}, {"0", false}} {
		cmd := exec.Command(os.Args[0], "-test.run=^"+child+"$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), "FAK_VULKAN_UMA_DIRECT="+arm.env, "FAK_VULKAN_UMA_DIRECT_CHILD=1")
		if arm.direct {
			cmd.Env = append(cmd.Env, "FAK_VULKAN_UMA_DIRECT_WANT=direct")
		}
		out, err := cmd.CombinedOutput()
		if bytes.Contains(out, []byte("--- SKIP")) {
			t.Skipf("child skipped: %s", out)
		}
		if err != nil {
			t.Fatalf("FAK_VULKAN_UMA_DIRECT=%s child failed: %v\n%s", arm.env, err, out)
		}
	}
}

func TestVulkanUMADirectWeightUploadChild(t *testing.T) {
	if os.Getenv("FAK_VULKAN_UMA_DIRECT_CHILD") != "1" {
		t.Skip("child of TestVulkanUMADirectWeightUploadSkipsStaging")
	}
	v := vk(t)
	assertVulkanUMADirectQ2KUpload(t, v, false)
}

func TestVulkanUMADirectHostVisibleWeightChild(t *testing.T) {
	if os.Getenv("FAK_VULKAN_UMA_DIRECT_CHILD") != "1" {
		t.Skip("child of TestVulkanUMADirectHostVisibleWeightUploadSkipsStaging")
	}
	v := vk(t)
	oldBudget := v.budgetBytes
	v.budgetBytes = v.dlUsed + 1 // any weight now exceeds the residency budget
	defer func() { v.budgetBytes = oldBudget }()
	assertVulkanUMADirectQ2KUpload(t, v, true)
}

func assertVulkanUMADirectQ2KUpload(t *testing.T, v *vulkanBackend, wantHostVisible bool) {
	t.Helper()
	const out, in = 8, 512
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	for i := range raw {
		raw[i] = byte(i*131 + 7)
	}
	before := VulkanDirectWeightUploadBytes()
	w := v.Upload(NewQ2K(Default(), []int{out, in}, raw), Q2_K)
	defer v.Free(w)
	if wantHostVisible && !v.debugBufferHostVisible(w.buf.(*vulkanBuf)) {
		t.Fatal("over-budget weight did not take the host-visible fallback")
	}
	direct := VulkanDirectWeightUploadBytes() - before
	wantDirect := os.Getenv("FAK_VULKAN_UMA_DIRECT_WANT") == "direct"
	switch {
	case wantDirect && direct != uint64(len(raw)):
		t.Fatalf("direct upload bytes = %d, want the whole %d-byte tensor written through the mapping", direct, len(raw))
	case !wantDirect && direct != 0:
		t.Fatalf("direct upload bytes = %d with UMA direct disabled, want the staging path", direct)
	}
	if got := v.VulkanDebugReadRestoreBuffer(w.buf.(*vulkanBuf)); !bytes.Equal(got[:len(raw)], raw) {
		t.Fatal("device weight bytes differ from the uploaded checkpoint bytes")
	}
}
