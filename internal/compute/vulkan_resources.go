package compute

import (
	"os"
	"path/filepath"
)

// vulkanSPIRVDir keeps explicit development bundles authoritative and locates
// release resources beside the executable, including when invoked by a symlink.
func vulkanSPIRVDir(override, executable string) string {
	if override != "" {
		return override
	}
	if executable == "" {
		return ""
	}
	if resolved, err := filepath.EvalSymlinks(executable); err == nil {
		executable = resolved
	}
	dir := filepath.Join(filepath.Dir(executable), "spirv")
	if info, err := os.Stat(dir); err == nil && info.IsDir() {
		return dir
	}
	return ""
}
