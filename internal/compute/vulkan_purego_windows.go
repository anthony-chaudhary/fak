//go:build windows

package compute

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	vulkanPureGoLoaderName   = "vulkan-1.dll"
	vulkanEnumerateVersion   = "vkEnumerateInstanceVersion"
	vulkanSystemSearchPolicy = "system32"

	// VulkanVKSuccess is the successful VkResult value returned by Vulkan calls.
	VulkanVKSuccess int32 = 0
)

// VulkanPureGoLoaderErrorKind identifies the failed part of the cgo-free
// Vulkan loader probe.
type VulkanPureGoLoaderErrorKind string

const (
	VulkanPureGoLoaderErrorLoader           VulkanPureGoLoaderErrorKind = "loader"
	VulkanPureGoLoaderErrorSymbol           VulkanPureGoLoaderErrorKind = "symbol"
	VulkanPureGoLoaderErrorCall             VulkanPureGoLoaderErrorKind = "call"
	VulkanPureGoLoaderErrorZeroVersion      VulkanPureGoLoaderErrorKind = "zero-version"
	VulkanPureGoLoaderErrorMalformedVersion VulkanPureGoLoaderErrorKind = "malformed-version"
)

// VulkanPureGoLoaderError reports a loader, symbol, Vulkan-call, or malformed
// version failure without selecting another backend.
type VulkanPureGoLoaderError struct {
	Kind          VulkanPureGoLoaderErrorKind
	Loader        string
	Symbol        string
	VkResult      int32
	RawAPIVersion uint32
	Err           error
}

func (e *VulkanPureGoLoaderError) Error() string {
	if e == nil {
		return "<nil>"
	}
	switch e.Kind {
	case VulkanPureGoLoaderErrorLoader:
		return fmt.Sprintf("load Vulkan system DLL %q: %v", e.Loader, e.Err)
	case VulkanPureGoLoaderErrorSymbol:
		return fmt.Sprintf("resolve Vulkan symbol %q from %q: %v", e.Symbol, e.Loader, e.Err)
	case VulkanPureGoLoaderErrorCall:
		return fmt.Sprintf("call Vulkan symbol %q from %q: VkResult=%d", e.Symbol, e.Loader, e.VkResult)
	case VulkanPureGoLoaderErrorZeroVersion:
		return fmt.Sprintf("call Vulkan symbol %q from %q: returned zero API version", e.Symbol, e.Loader)
	case VulkanPureGoLoaderErrorMalformedVersion:
		return fmt.Sprintf("call Vulkan symbol %q from %q: returned malformed API version %#x", e.Symbol, e.Loader, e.RawAPIVersion)
	default:
		return fmt.Sprintf("probe Vulkan system DLL %q: %v", e.Loader, e.Err)
	}
}

func (e *VulkanPureGoLoaderError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// VulkanPureGoLoaderReceipt is the observed API version returned by the
// installed Windows Vulkan loader. RawAPIVersion preserves Vulkan's packed
// uint32 representation; Major, Minor, and Patch decode VK_MAKE_API_VERSION.
type VulkanPureGoLoaderReceipt struct {
	Loader        string
	Symbol        string
	SearchPolicy  string
	VkResult      int32
	RawAPIVersion uint32
	Variant       uint32
	Major         uint32
	Minor         uint32
	Patch         uint32
}

// ProbeVulkanPureGoLoader calls the real Windows Vulkan loader without cgo.
// It performs no package initialization and never falls back to another
// backend when the loader, symbol, call, or returned version is invalid.
func ProbeVulkanPureGoLoader() (VulkanPureGoLoaderReceipt, error) {
	return probeVulkanPureGoLoader(vulkanPureGoLoaderName, vulkanEnumerateVersion)
}

func probeVulkanPureGoLoader(loaderName, symbolName string) (VulkanPureGoLoaderReceipt, error) {
	dll := windows.NewLazySystemDLL(loaderName)
	if err := dll.Load(); err != nil {
		return VulkanPureGoLoaderReceipt{}, &VulkanPureGoLoaderError{
			Kind:   VulkanPureGoLoaderErrorLoader,
			Loader: loaderName,
			Symbol: symbolName,
			Err:    err,
		}
	}

	proc := dll.NewProc(symbolName)
	if err := proc.Find(); err != nil {
		return VulkanPureGoLoaderReceipt{}, &VulkanPureGoLoaderError{
			Kind:   VulkanPureGoLoaderErrorSymbol,
			Loader: loaderName,
			Symbol: symbolName,
			Err:    err,
		}
	}

	var apiVersion uint32
	result, _, _ := proc.Call(uintptr(unsafe.Pointer(&apiVersion)))
	return decodeVulkanPureGoLoaderReceipt(loaderName, symbolName, int32(result), apiVersion)
}

func decodeVulkanPureGoLoaderReceipt(loaderName, symbolName string, result int32, apiVersion uint32) (VulkanPureGoLoaderReceipt, error) {
	if result != VulkanVKSuccess {
		return VulkanPureGoLoaderReceipt{}, &VulkanPureGoLoaderError{
			Kind:     VulkanPureGoLoaderErrorCall,
			Loader:   loaderName,
			Symbol:   symbolName,
			VkResult: result,
		}
	}
	if apiVersion == 0 {
		return VulkanPureGoLoaderReceipt{}, &VulkanPureGoLoaderError{
			Kind:   VulkanPureGoLoaderErrorZeroVersion,
			Loader: loaderName,
			Symbol: symbolName,
		}
	}
	major := (apiVersion >> 22) & 0x7f
	if major == 0 {
		return VulkanPureGoLoaderReceipt{}, &VulkanPureGoLoaderError{
			Kind:          VulkanPureGoLoaderErrorMalformedVersion,
			Loader:        loaderName,
			Symbol:        symbolName,
			RawAPIVersion: apiVersion,
		}
	}

	return VulkanPureGoLoaderReceipt{
		Loader:        loaderName,
		Symbol:        symbolName,
		SearchPolicy:  vulkanSystemSearchPolicy,
		VkResult:      result,
		RawAPIVersion: apiVersion,
		Variant:       apiVersion >> 29,
		Major:         major,
		Minor:         (apiVersion >> 12) & 0x3ff,
		Patch:         apiVersion & 0xfff,
	}, nil
}
