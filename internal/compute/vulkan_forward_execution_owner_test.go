//go:build vulkan && (windows || linux) && cgo

package compute

import "testing"

type vulkanExecutionOwnerEmbeddedWrapper struct{ *vulkanBackend }

// fak-test:runtime fast est=1ms lane=optin
// No native operation is called by this test. The Vulkan-tagged package still
// has cgo/link/init requirements, so this is not a CPU-only qualification.
func TestVulkanForwardExecutionOwnerDomain(t *testing.T) {
	a, b := new(vulkanBackend), new(vulkanBackend)
	providers := []ForwardExecutionOwnerBackend{
		a,
		b,
		&vulkanExecutionOwnerEmbeddedWrapper{a},
		&vulkanExecutionOwnerEmbeddedWrapper{b},
	}
	owner := a.ForwardExecutionOwner()
	if owner == nil || owner == &vulkanMu {
		t.Fatal("Vulkan needs a non-nil outer owner separate from vulkanMu")
	}
	for i, provider := range providers {
		if got := provider.ForwardExecutionOwner(); got != owner {
			t.Fatalf("provider %d has a different recording-domain owner", i)
		}
	}
	if !owner.TryLock() {
		t.Fatal("outer owner was unexpectedly busy")
	}
	defer owner.Unlock()
	for i, provider := range providers {
		if provider.ForwardExecutionOwner().TryLock() {
			provider.ForwardExecutionOwner().Unlock()
			t.Fatalf("provider %d escaped the held domain", i)
		}
	}
	if !vulkanMu.TryLock() {
		t.Fatal("holding the outer owner must not acquire vulkanMu")
	}
	vulkanMu.Unlock()
}
