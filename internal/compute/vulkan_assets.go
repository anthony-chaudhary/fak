package compute

import (
	"io/fs"
	"path/filepath"
)

// resolveVulkanSPIRVDir preserves an explicit shader directory or discovers a
// bundled spirv directory beside the resolved executable. An absent bundle
// leaves Vulkan unavailable so callers retain the portable backend fallback.
//
// This is the production shader-directory resolver (wired at vulkan.go); a
// duplicate non-invoked helper formerly lived in vulkan_resources.go and was
// removed rather than wired, so exactly one resolver is reachable.
func resolveVulkanSPIRVDir(
	explicit string,
	executable func() (string, error),
	evalSymlinks func(string) (string, error),
	stat func(string) (fs.FileInfo, error),
) string {
	if explicit != "" {
		return explicit
	}
	if executable == nil || evalSymlinks == nil || stat == nil {
		return ""
	}
	executablePath, err := executable()
	if err != nil || executablePath == "" {
		return ""
	}
	resolvedPath, err := evalSymlinks(executablePath)
	if err != nil || resolvedPath == "" {
		return ""
	}
	candidate := filepath.Clean(filepath.Join(filepath.Dir(resolvedPath), "spirv"))
	info, err := stat(candidate)
	if err != nil || !info.IsDir() {
		return ""
	}
	return candidate
}
