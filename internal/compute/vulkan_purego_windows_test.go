//go:build windows

package compute

import (
	"errors"
	"os"
	"testing"
)

func TestVulkanPureGoLoader(t *testing.T) {
	receipt, err := ProbeVulkanPureGoLoader()
	if err != nil {
		var typed *VulkanPureGoLoaderError
		if errors.As(err, &typed) &&
			(typed.Kind == VulkanPureGoLoaderErrorLoader || typed.Kind == VulkanPureGoLoaderErrorSymbol) &&
			os.Getenv("FAK_VULKAN_LIVE_REQUIRED") != "1" {
			t.Skipf("Vulkan system loader unavailable: %v", err)
		}
		t.Fatalf("probe installed Vulkan loader: %v", err)
	}
	if receipt.VkResult != VulkanVKSuccess {
		t.Fatalf("VkResult = %d, want VK_SUCCESS (%d)", receipt.VkResult, VulkanVKSuccess)
	}
	if receipt.RawAPIVersion == 0 {
		t.Fatal("raw Vulkan API version is zero")
	}
	if receipt.SearchPolicy != vulkanSystemSearchPolicy {
		t.Fatalf("loader search policy = %q, want %q", receipt.SearchPolicy, vulkanSystemSearchPolicy)
	}
	if receipt.Major == 0 || receipt.Major > 9 {
		t.Fatalf("decoded Vulkan major version = %d, want a sane nonzero version", receipt.Major)
	}
	if receipt.Minor > 0x3ff || receipt.Patch > 0xfff {
		t.Fatalf("decoded Vulkan version is out of range: %d.%d.%d", receipt.Major, receipt.Minor, receipt.Patch)
	}
	t.Logf("Vulkan loader %s returned VK_SUCCESS, API version %d.%d.%d (raw %#x)",
		receipt.Loader, receipt.Major, receipt.Minor, receipt.Patch, receipt.RawAPIVersion)
}

func TestVulkanPureGoLoaderTypedFailures(t *testing.T) {
	t.Run("loader", func(t *testing.T) {
		_, err := probeVulkanPureGoLoader("fak-vulkan-loader-that-does-not-exist.dll", vulkanEnumerateVersion)
		assertVulkanPureGoLoaderError(t, err, VulkanPureGoLoaderErrorLoader)
	})

	t.Run("symbol", func(t *testing.T) {
		_, err := probeVulkanPureGoLoader("kernel32.dll", "fakVulkanSymbolThatDoesNotExist")
		assertVulkanPureGoLoaderError(t, err, VulkanPureGoLoaderErrorSymbol)
	})

	t.Run("call", func(t *testing.T) {
		_, err := decodeVulkanPureGoLoaderReceipt(vulkanPureGoLoaderName, vulkanEnumerateVersion, -3, 1)
		assertVulkanPureGoLoaderError(t, err, VulkanPureGoLoaderErrorCall)
	})

	t.Run("zero-version", func(t *testing.T) {
		_, err := decodeVulkanPureGoLoaderReceipt(vulkanPureGoLoaderName, vulkanEnumerateVersion, VulkanVKSuccess, 0)
		assertVulkanPureGoLoaderError(t, err, VulkanPureGoLoaderErrorZeroVersion)
	})

	t.Run("malformed-version", func(t *testing.T) {
		_, err := decodeVulkanPureGoLoaderReceipt(vulkanPureGoLoaderName, vulkanEnumerateVersion, VulkanVKSuccess, 1)
		assertVulkanPureGoLoaderError(t, err, VulkanPureGoLoaderErrorMalformedVersion)
	})
}

func TestVulkanPureGoLoaderVersionDecode(t *testing.T) {
	const packed = uint32(1<<29 | 1<<22 | 3<<12 | 275)
	receipt, err := decodeVulkanPureGoLoaderReceipt(vulkanPureGoLoaderName, vulkanEnumerateVersion, VulkanVKSuccess, packed)
	if err != nil {
		t.Fatalf("decode packed Vulkan version: %v", err)
	}
	if receipt.RawAPIVersion != packed || receipt.Variant != 1 || receipt.Major != 1 || receipt.Minor != 3 || receipt.Patch != 275 {
		t.Fatalf("decoded receipt = %+v, want raw=%#x variant=1 version=1.3.275", receipt, packed)
	}
}

func assertVulkanPureGoLoaderError(t *testing.T, err error, want VulkanPureGoLoaderErrorKind) {
	t.Helper()
	var typed *VulkanPureGoLoaderError
	if !errors.As(err, &typed) {
		t.Fatalf("error = %T %v, want *VulkanPureGoLoaderError", err, err)
	}
	if typed.Kind != want {
		t.Fatalf("error kind = %q, want %q", typed.Kind, want)
	}
}
