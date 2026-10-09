package compute

import (
	"errors"
	"fmt"
	"path"
	"strings"
)

type vulkanDRMRenderNodeProvider interface {
	VulkanDRMRenderNode() (major, minor uint64, available bool)
}

// vulkanDRMDeviceBinding is diagnostic identity for one observation, not a
// capacity probe or a device-lifetime lease. Rechecking detects observed changes
// but cannot make a sysfs read atomic with backend destruction or hot unplug.
type vulkanDRMDeviceBinding struct {
	major, minor                    uint64
	nodePath, devicePath, driverPath string
}

func bindSelectedVulkanDRMDevice(backend Backend, deps hostEnvironmentDeps, vendorID, deviceID string) (vulkanDRMDeviceBinding, error) {
	if deps.goos != "linux" || deps.readFile == nil || deps.evalSymlinks == nil {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan DRM binding requires Linux sysfs observation")
	}
	provider, ok := backend.(vulkanDRMRenderNodeProvider)
	if !ok {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan DRM render-node identity is unavailable")
	}
	major, minor, available := provider.VulkanDRMRenderNode()
	if !available || major == 0 {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan DRM render-node identity is unavailable")
	}
	number := fmt.Sprintf("%d:%d", major, minor)
	lookupPath := path.Join("/sys/dev/char", number)
	nodePath, err := resolveObservedSysfsPath(deps, lookupPath, "/sys/devices/", "DRM render node")
	if err != nil {
		return vulkanDRMDeviceBinding{}, err
	}
	// This validates the kernel's resolved node; it never chooses a device by
	// render-node order, a card number, a name, or a vendor/device-ID match.
	if path.Base(nodePath) != fmt.Sprintf("renderD%d", minor) {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan DRM node is not the reported render node")
	}
	observedNumber, err := readObservedLine(deps.readFile, path.Join(nodePath, "dev"), "DRM render-node number")
	if err != nil {
		return vulkanDRMDeviceBinding{}, err
	}
	if observedNumber != number {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan DRM node number does not match sysfs")
	}
	subsystem, err := resolveObservedSysfsPath(deps, path.Join(nodePath, "subsystem"), "/sys/", "DRM subsystem")
	if err != nil {
		return vulkanDRMDeviceBinding{}, err
	}
	if path.Base(subsystem) != "drm" {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan render node is not in the DRM subsystem")
	}
	devicePath, err := resolveObservedSysfsPath(deps, path.Join(lookupPath, "device"), "/sys/devices/", "DRM backing device")
	if err != nil {
		return vulkanDRMDeviceBinding{}, err
	}
	if !strings.HasPrefix(nodePath, devicePath+"/") {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan render node is not below its sysfs backing device")
	}
	subsystem, err = resolveObservedSysfsPath(deps, path.Join(devicePath, "subsystem"), "/sys/", "DRM backing-device subsystem")
	if err != nil {
		return vulkanDRMDeviceBinding{}, err
	}
	if path.Base(subsystem) != "pci" {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan DRM backing device is not PCI")
	}
	driverPath, err := resolveObservedSysfsPath(deps, path.Join(devicePath, "driver"), "/sys/", "DRM backing-device driver")
	if err != nil {
		return vulkanDRMDeviceBinding{}, err
	}
	if path.Base(driverPath) != "amdgpu" {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan DRM backing device is not bound to amdgpu")
	}
	vendor, err := readObservedPCIID(deps.readFile, path.Join(devicePath, "vendor"), "DRM vendor")
	if err != nil {
		return vulkanDRMDeviceBinding{}, err
	}
	device, err := readObservedPCIID(deps.readFile, path.Join(devicePath, "device"), "DRM device")
	if err != nil {
		return vulkanDRMDeviceBinding{}, err
	}
	if vendor != vendorID || device != deviceID {
		return vulkanDRMDeviceBinding{}, errors.New("compute: selected Vulkan DRM backing device does not match backend PCI identity")
	}
	return vulkanDRMDeviceBinding{major: major, minor: minor, nodePath: nodePath, devicePath: devicePath, driverPath: driverPath}, nil
}

func resolveObservedSysfsPath(deps hostEnvironmentDeps, name, prefix, field string) (string, error) {
	resolved, err := deps.evalSymlinks(name)
	if err != nil {
		return "", fmt.Errorf("compute: resolve %s: %w", field, err)
	}
	if path.Clean(resolved) != resolved || !strings.HasPrefix(resolved, prefix) {
		return "", fmt.Errorf("compute: %s did not resolve to a canonical sysfs path", field)
	}
	return resolved, nil
}
