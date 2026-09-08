package compute

const (
	vulkanAMDVendorID      = 0x1002
	vulkanGFX1151DeviceID  = 0x1586
	vulkanWave32SubgroupSz = 32
)

type vulkanQ4KWave32Device struct {
	vendorID             uint32
	deviceID             uint32
	computeSubgroupBasic bool
	computeSubgroupArith bool
	defaultSubgroupSize  uint32
	canRequireWave32     bool
}

// selectVulkanQ4KWave32 is the device-free model of the native admission gate.
// The physical candidate remains opt-in; an explicit scalar selection wins.
func selectVulkanQ4KWave32(device vulkanQ4KWave32Device, tokens int, optIn, forceScalar bool) bool {
	if forceScalar || !optIn || tokens != 1 {
		return false
	}
	return device.vendorID == vulkanAMDVendorID &&
		device.deviceID == vulkanGFX1151DeviceID &&
		device.computeSubgroupBasic &&
		device.computeSubgroupArith &&
		(device.defaultSubgroupSize == vulkanWave32SubgroupSz || device.canRequireWave32)
}
