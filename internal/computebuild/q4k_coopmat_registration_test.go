package computebuild

import "testing"

// TestVulkanShadersIncludesQ4KCoopMat pins the q4k_matmul_coopmat shader in the
// canonical Vulkan shader bundle (fak#12177). The 2D cooperative-matrix Q4_K
// prefill arm records its shader here so a bundle build that omits it fails
// loudly instead of silently dropping the kernel at pipeline creation.
func TestVulkanShadersIncludesQ4KCoopMat(t *testing.T) {
	found := false
	for _, name := range VulkanShaders {
		if name == "q4k_matmul_coopmat" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("VulkanShaders is missing %q; the Q4_K cooperative-matrix arm would fail to build", "q4k_matmul_coopmat")
	}

	// The scalar and Wave32 Q4_K arms must remain registered so the coopmat arm can
	// always fall back rather than removing the decode path.
	for _, required := range []string{"q4k_matmul", "q4k_matmul_wave32"} {
		ok := false
		for _, name := range VulkanShaders {
			if name == required {
				ok = true
				break
			}
		}
		if !ok {
			t.Fatalf("VulkanShaders is missing %q; the Q4_K scalar/Wave32 fallback must stay registered", required)
		}
	}
}
