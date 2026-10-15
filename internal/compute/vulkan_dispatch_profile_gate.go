package compute

// nativeVulkanDispatchProfileEnabled is assigned only during Vulkan package
// initialization, from the native profiler's captured startup state. Builds
// without the native Vulkan backend retain the disabled zero value.
var nativeVulkanDispatchProfileEnabled bool

// VulkanDispatchProfileEnabled reports the native profiler's immutable startup
// decision without initializing a device or rereading the environment. It is
// false when the native Vulkan backend is not included in this build.
func VulkanDispatchProfileEnabled() bool { return nativeVulkanDispatchProfileEnabled }
