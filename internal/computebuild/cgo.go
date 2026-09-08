package computebuild

import (
	"path/filepath"
	"sort"
	"strings"
)

// QuoteFlag formats a compiler/linker flag (such as -I or -L) with quotes around
// the path if it contains spaces, matching Go's cgo splitQuoted parser.
func QuoteFlag(flag, path string) string {
	if path == "" {
		return ""
	}
	if strings.Contains(path, " ") {
		return flag + `"` + path + `"`
	}
	return flag + path
}

// QuoteArg wraps an argument in quotes if it contains spaces.
func QuoteArg(arg string) string {
	if strings.Contains(arg, " ") {
		return `"` + arg + `"`
	}
	return arg
}

// SynthesizeVulkanCgoEnv constructs the environment map required for `go build -tags vulkan`.
func SynthesizeVulkanCgoEnv(tc *Toolchain, pkgDir string) map[string]string {
	env := make(map[string]string)
	env["CGO_ENABLED"] = "1"

	if tc != nil {
		if tc.CC != "" {
			env["CC"] = tc.CC
		}
		if tc.CXX != "" {
			env["CXX"] = tc.CXX
		}
	}

	var cflags []string
	if tc != nil && tc.VulkanInc != "" {
		cflags = append(cflags, QuoteFlag("-I", tc.VulkanInc))
	}
	if pkgDir != "" {
		cflags = append(cflags, QuoteFlag("-I", pkgDir))
	}
	if len(cflags) > 0 {
		env["CGO_CFLAGS"] = strings.Join(cflags, " ")
	}

	var ldflags []string
	if pkgDir != "" {
		ldflags = append(ldflags, QuoteFlag("-L", pkgDir))
	}
	ldflags = append(ldflags, "-lfakvulkan")
	if tc != nil && tc.VulkanLib != "" {
		ldflags = append(ldflags, QuoteFlag("-L", tc.VulkanLib))
		if tc.IsWindows {
			ldflags = append(ldflags, "-lvulkan-1")
		} else {
			ldflags = append(ldflags, "-lvulkan")
		}
	}
	if tc != nil && tc.CxxRuntime != "" {
		ldflags = append(ldflags, tc.CxxRuntime)
	}
	env["CGO_LDFLAGS"] = strings.Join(ldflags, " ")

	if pkgDir != "" {
		env["FAK_VULKAN_SPIRV"] = filepath.Join(pkgDir, "spirv")
	}

	return env
}

// SynthesizeCUDACgoEnv constructs the environment map required for `go build -tags cuda`.
func SynthesizeCUDACgoEnv(tc *Toolchain, pkgDir string, enableNCCL bool, rpaths []string) map[string]string {
	env := make(map[string]string)
	env["CGO_ENABLED"] = "1"

	if tc != nil {
		if tc.CC != "" {
			env["CC"] = tc.CC
		}
		if tc.CXX != "" {
			env["CXX"] = tc.CXX
		}
	}

	var cflags []string
	if tc != nil {
		for _, inc := range tc.CUDAInc {
			if inc != "" {
				cflags = append(cflags, QuoteFlag("-I", inc))
			}
		}
	}
	if len(cflags) > 0 {
		env["CGO_CFLAGS"] = strings.Join(cflags, " ")
	}

	var ldflags []string
	if tc != nil && !tc.IsWindows && pkgDir != "" {
		ldflags = append(ldflags, QuoteFlag("-L", pkgDir))
	}
	if tc != nil {
		for _, lib := range tc.CUDALib {
			if lib != "" {
				ldflags = append(ldflags, QuoteFlag("-L", lib))
			}
		}
	}
	if enableNCCL {
		ldflags = append(ldflags, "-lnccl")
	}
	for _, rp := range rpaths {
		if rp != "" {
			ldflags = append(ldflags, "-Wl,-rpath,"+QuoteArg(rp))
		}
	}
	if len(ldflags) > 0 {
		env["CGO_LDFLAGS"] = strings.Join(ldflags, " ")
	}

	return env
}

// EnvMapToSlice converts a key-value map to a sorted slice of "KEY=VAL" strings.
func EnvMapToSlice(env map[string]string) []string {
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	res := make([]string, 0, len(env))
	for _, k := range keys {
		res = append(res, k+"="+env[k])
	}
	return res
}

// MergeEnviron combines a base environment slice with an overlay map.
// Existing keys in base matching overlay keys will be updated; new keys will be added.
func MergeEnviron(base []string, overlay map[string]string) []string {
	envMap := make(map[string]string, len(base)+len(overlay))
	for _, kv := range base {
		idx := strings.Index(kv, "=")
		if idx > 0 {
			envMap[kv[:idx]] = kv[idx+1:]
		}
	}
	for k, v := range overlay {
		envMap[k] = v
	}
	return EnvMapToSlice(envMap)
}
