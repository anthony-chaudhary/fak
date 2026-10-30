package computebuild

import (
	"fmt"
	"os"
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
	if tc != nil && !tc.IsWindows {
		if dir := wslCUDALinkDir(); dir != "" {
			ldflags = append(ldflags, QuoteFlag("-L", dir))
			env["LIBRARY_PATH"] = prependPathList(dir, os.Getenv("LIBRARY_PATH"))
		}
	}
	if enableNCCL {
		ldflags = append(ldflags, "-lnccl")
	}
	if tc != nil && !tc.IsWindows {
		var loader, toolkit []string
		appendUnique := func(paths []string, path string) []string {
			if path == "" || cudaStubPath(path) {
				return paths
			}
			for _, existing := range paths {
				if existing == path {
					return paths
				}
			}
			return append(paths, path)
		}
		for _, path := range []string{wslCUDADriverDir, "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu"} {
			if matches, _ := filepath.Glob(filepath.Join(path, "libcuda.so*")); len(matches) > 0 {
				loader = appendUnique(loader, path)
			}
		}
		for _, path := range tc.CUDALib {
			if cudaStubPath(path) {
				continue
			}
			if matches, _ := filepath.Glob(filepath.Join(path, "libcuda.so*")); len(matches) > 0 {
				loader = appendUnique(loader, path)
			} else {
				toolkit = appendUnique(toolkit, path)
				rpaths = appendUnique(rpaths, path)
			}
		}
		for _, path := range toolkit {
			loader = appendUnique(loader, path)
		}
		for _, path := range filepath.SplitList(os.Getenv("LD_LIBRARY_PATH")) {
			loader = appendUnique(loader, path)
		}
		env["LD_LIBRARY_PATH"] = strings.Join(loader, string(os.PathListSeparator))
	}
	for _, rp := range rpaths {
		if rp != "" && (tc == nil || tc.IsWindows || !cudaStubPath(rp)) {
			ldflags = append(ldflags, "-Wl,-rpath,"+QuoteArg(rp))
		}
	}
	if len(ldflags) > 0 {
		env["CGO_LDFLAGS"] = strings.Join(ldflags, " ")
	}

	return env
}

// wslCUDADriverDir is where WSL2 mounts the host GPU driver's libcuda.so.
// It is a variable so tests can point it at a fixture tree.
var wslCUDADriverDir = "/usr/lib/wsl/lib"

// wslCUDALinkDir returns the WSL driver directory when it holds the libcuda.so
// linker name, so `-lcuda` resolves without a caller-set LIBRARY_PATH.
func wslCUDALinkDir() string {
	if info, err := os.Stat(filepath.Join(wslCUDADriverDir, "libcuda.so")); err == nil && !info.IsDir() {
		return wslCUDADriverDir
	}
	return ""
}

func prependPathList(dir, list string) string {
	parts := []string{dir}
	for _, path := range filepath.SplitList(list) {
		if path != "" && path != dir {
			parts = append(parts, path)
		}
	}
	return strings.Join(parts, string(os.PathListSeparator))
}

func cudaStubPath(path string) bool {
	for _, part := range strings.Split(filepath.ToSlash(filepath.Clean(path)), "/") {
		if part == "stubs" {
			return true
		}
	}
	return false
}

// PrepareCUDANCCLToolchain resolves user-space NCCL layouts and their linker name.
func PrepareCUDANCCLToolchain(tc *Toolchain, home, repoRoot string) (*Toolchain, error) {
	if tc == nil {
		return nil, fmt.Errorf("nil CUDA toolchain")
	}
	prepared := *tc
	prepared.CUDAInc = append([]string(nil), tc.CUDAInc...)
	prepared.CUDALib = append([]string(nil), tc.CUDALib...)
	if home != "" {
		for _, relative := range []string{"include", "usr/include"} {
			path := filepath.Join(home, filepath.FromSlash(relative))
			if info, err := os.Stat(path); err == nil && info.IsDir() {
				prepared.CUDAInc = appendCUDAPath(prepared.CUDAInc, path)
			}
		}
		for _, relative := range []string{"lib64", "lib", "usr/lib/x86_64-linux-gnu", "usr/lib"} {
			path := filepath.Join(home, filepath.FromSlash(relative))
			if info, err := os.Stat(path); err == nil && info.IsDir() {
				prepared.CUDALib = appendCUDAPath(prepared.CUDALib, path)
			}
		}
	}
	if prepared.IsWindows {
		return &prepared, nil
	}
	for _, path := range prepared.CUDALib {
		if FileExists(filepath.Join(path, "libnccl.so")) {
			return &prepared, nil
		}
	}
	for _, path := range prepared.CUDALib {
		matches, err := filepath.Glob(filepath.Join(path, "libnccl.so.*"))
		if err != nil {
			return nil, err
		}
		for _, target := range matches {
			if !FileExists(target) {
				continue
			}
			target, err = filepath.Abs(target)
			if err != nil {
				return nil, err
			}
			stateDir := filepath.Join(repoRoot, ".fak")
			if err := os.Mkdir(stateDir, 0755); err != nil && !os.IsExist(err) {
				return nil, err
			}
			stateInfo, err := os.Lstat(stateDir)
			if err != nil {
				return nil, err
			}
			if !stateInfo.IsDir() || stateInfo.Mode()&os.ModeSymlink != 0 {
				return nil, fmt.Errorf("NCCL state directory is not a regular directory")
			}
			dir := filepath.Join(stateDir, "cuda-nccl-lib")
			if err := os.Mkdir(dir, 0755); err != nil && !os.IsExist(err) {
				return nil, err
			}
			info, err := os.Lstat(dir)
			if err != nil {
				return nil, err
			}
			if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
				return nil, fmt.Errorf("NCCL link directory is not a regular directory")
			}
			link := filepath.Join(dir, "libnccl.so")
			if current, err := os.Readlink(link); err == nil {
				if current != target {
					if err := os.Remove(link); err != nil {
						return nil, err
					}
				} else {
					prepared.CUDALib = appendCUDAPath(prepared.CUDALib, dir)
					return &prepared, nil
				}
			} else if _, err := os.Lstat(link); err == nil || !os.IsNotExist(err) {
				return nil, fmt.Errorf("NCCL linker name is not a replaceable symlink")
			}
			if err := os.Symlink(target, link); err != nil {
				return nil, err
			}
			prepared.CUDALib = appendCUDAPath(prepared.CUDALib, dir)
			return &prepared, nil
		}
	}
	return &prepared, nil
}

func appendCUDAPath(paths []string, path string) []string {
	for _, existing := range paths {
		if existing == path {
			return paths
		}
	}
	return append(paths, path)
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
