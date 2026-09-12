//go:build !windows

package main

// probeVulkanLoaderFacts is a no-op off Windows: the cgo-free loader probe loads
// the Windows Vulkan runtime DLL (vulkan-1.dll) via NewLazySystemDLL, and
// Linux/other hosts resolve Vulkan through their own loader path (libvulkan.so),
// which this Windows-specific probe does not cover. Returning nil ("no fact")
// makes serveVulkanRow emit no gpu-vulkan row and keeps the readiness table's
// existing shape on non-Windows hosts, identical to an Applicable=false fact.
func probeVulkanLoaderFacts() *vulkanLoaderFacts {
	return nil
}
