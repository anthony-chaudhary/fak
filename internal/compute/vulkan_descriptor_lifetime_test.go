package compute

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
// A source-ordering regression for all native destruction callers, including
// GDN verification scratch. This does not establish Vulkan driver behavior.
func TestVulkanDescriptorDestructionInvalidatesBindings(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	if err := validateVulkanDescriptorDestruction(source); err != nil {
		t.Fatal(err)
	}
	const invalidate = "        clearDescriptorBindingCache();\n"
	const destroy = "        vkDestroyBuffer(g_dev, b->buf, nullptr);\n"
	const quarantine = "    if (g_v41SubmissionPendingFailure) return; // referenced scratch/arena may still be in flight\n"
	body, err := vulkanRestoreLifetimeFunction(source, "void destroyBuffer(Buffer* b) {")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, old, replacement string
	}{
		{"missing invalidation", invalidate, ""},
		{"late invalidation", invalidate + destroy, destroy + invalidate},
		{"missing quarantine", quarantine, ""},
		{"mutation before quarantine", quarantine, invalidate + quarantine},
		{"missing buffer destruction", destroy, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if strings.Count(body, tc.old) != 1 {
				t.Fatalf("negative control must replace exactly one %q", tc.old)
			}
			mutated := strings.Replace(body, tc.old, tc.replacement, 1)
			if err := validateVulkanDescriptorDestruction(strings.Replace(source, body, mutated, 1)); err == nil {
				t.Fatal("unsafe destruction source passed the validator")
			}
		})
	}
}

func validateVulkanDescriptorDestruction(source string) error {
	if len(source) > 1<<20 {
		return fmt.Errorf("Vulkan source exceeds 1 MiB ordering-check bound")
	}
	body, err := vulkanRestoreLifetimeFunction(source, "void destroyBuffer(Buffer* b) {")
	if err != nil {
		return err
	}
	const prefix = "void destroyBuffer(Buffer* b) {\n" +
		"    if (g_v41SubmissionPendingFailure) return; // referenced scratch/arena may still be in flight\n" +
		"    if (!b) return;\n" +
		"    if (b->buf) {\n"
	if !strings.HasPrefix(body, prefix) {
		return fmt.Errorf("buffer destruction must preserve quarantine and null guards before mutation")
	}
	const invalidate = "clearDescriptorBindingCache();"
	const destroy = "vkDestroyBuffer(g_dev, b->buf, nullptr);"
	if strings.Count(body, invalidate) != 1 || strings.Count(body, destroy) != 1 ||
		!strings.Contains(body, invalidate+"\n        "+destroy+"\n    }") {
		return fmt.Errorf("every native buffer destruction must immediately follow descriptor invalidation")
	}
	return nil
}
