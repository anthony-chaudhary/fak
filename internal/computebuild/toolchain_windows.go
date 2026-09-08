//go:build windows

package computebuild

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// discoverToolchainOS implements Windows-specific toolchain discovery.
func discoverToolchainOS() (*Toolchain, error) {
	tc := &Toolchain{
		IsWindows: true,
	}

	// 1. Locate Visual Studio dev command environment.
	vsDevCmd := FindVsDevCmd()
	tc.VsDevCmd = vsDevCmd
	if vsDevCmd != "" {
		if env, err := IngestVsDevEnv(vsDevCmd); err == nil {
			tc.VsDevEnv = env
		}
	}

	// 2. Discover Vulkan SDK.
	vkRoot := FindVulkanSDK()
	tc.VulkanSDK = vkRoot
	if vkRoot != "" {
		tc.VulkanInc = filepath.Join(vkRoot, "Include")
		tc.VulkanLib = filepath.Join(vkRoot, "Lib")

		glslcPath := filepath.Join(vkRoot, "Bin", "glslc.exe")
		if !FileExists(glslcPath) {
			glslcPath = filepath.Join(vkRoot, "bin", "glslc.exe")
		}
		if FileExists(glslcPath) {
			tc.GLSLC = glslcPath
		}
	}
	if tc.GLSLC == "" {
		if p, err := exec.LookPath("glslc.exe"); err == nil {
			tc.GLSLC = p
		}
	}

	// 3. Discover Clang / GCC / AR / LLVM-AR.
	llvmBin := filepath.Join(os.Getenv("ProgramFiles"), "LLVM", "bin")

	// Search PATH and LLVM bin.
	gccPath, _ := exec.LookPath("gcc.exe")
	gxxPath, _ := exec.LookPath("g++.exe")
	arPath, _ := exec.LookPath("ar.exe")

	clangxxPath, _ := exec.LookPath("clang++.exe")
	if clangxxPath == "" && FileExists(filepath.Join(llvmBin, "clang++.exe")) {
		clangxxPath = filepath.Join(llvmBin, "clang++.exe")
	}

	clangPath, _ := exec.LookPath("clang.exe")
	if clangPath == "" && FileExists(filepath.Join(llvmBin, "clang.exe")) {
		clangPath = filepath.Join(llvmBin, "clang.exe")
	}

	llvmArPath, _ := exec.LookPath("llvm-ar.exe")
	if llvmArPath == "" && FileExists(filepath.Join(llvmBin, "llvm-ar.exe")) {
		llvmArPath = filepath.Join(llvmBin, "llvm-ar.exe")
	}

	// C compiler selection.
	if gccPath != "" {
		tc.CC = gccPath
	} else if clangPath != "" {
		tc.CC = clangPath
	}

	// C++ compiler selection.
	if gxxPath != "" {
		tc.CXX = gxxPath
		tc.CxxRuntime = "-lstdc++"
	} else if clangxxPath != "" {
		tc.CXX = clangxxPath
		tc.CxxRuntime = "-lmsvcprt"
	}

	// Archiver selection.
	if arPath != "" {
		tc.AR = arPath
	} else if llvmArPath != "" {
		tc.AR = llvmArPath
	}

	// 4. Discover NVCC / CUDA Toolkit.
	cudaRoot := os.Getenv("CUDA_PATH")
	if cudaRoot == "" {
		cudaRoot = os.Getenv("CUDA_HOME")
	}
	if cudaRoot != "" && FileExists(filepath.Join(cudaRoot, "bin", "nvcc.exe")) {
		tc.NVCC = filepath.Join(cudaRoot, "bin", "nvcc.exe")
	} else if nvccOnPath, err := exec.LookPath("nvcc.exe"); err == nil {
		tc.NVCC = nvccOnPath
		if cudaRoot == "" {
			cudaRoot = filepath.Dir(filepath.Dir(nvccOnPath))
		}
	}
	tc.CUDAHome = cudaRoot
	if cudaRoot != "" {
		tc.CUDAInc = []string{filepath.Join(cudaRoot, "include")}
		tc.CUDALib = []string{filepath.Join(cudaRoot, "lib", "x64")}
	}

	// 5. Discover SignTool.
	tc.SignTool = FindSignTool()

	return tc, nil
}

// FindVsWhere searches for vswhere.exe in standard locations and PATH.
func FindVsWhere() string {
	candidates := []string{
		filepath.Join(os.Getenv("ProgramFiles(x86)"), "Microsoft Visual Studio", "Installer", "vswhere.exe"),
		filepath.Join(os.Getenv("ProgramFiles"), "Microsoft Visual Studio", "Installer", "vswhere.exe"),
	}
	for _, cand := range candidates {
		if FileExists(cand) {
			return cand
		}
	}
	if p, err := exec.LookPath("vswhere.exe"); err == nil {
		return p
	}
	return ""
}

// FindVsDevCmd searches for VsDevCmd.bat using vswhere.exe and fallback directories.
func FindVsDevCmd() string {
	vswhere := FindVsWhere()
	if vswhere != "" {
		cmd := exec.Command(vswhere, "-latest", "-products", "*", "-property", "installationPath")
		if out, err := cmd.Output(); err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				cand := filepath.Join(line, "Common7", "Tools", "VsDevCmd.bat")
				if FileExists(cand) {
					return cand
				}
			}
		}
	}

	pfRoots := []string{
		os.Getenv("ProgramFiles(x86)"),
		os.Getenv("ProgramFiles"),
	}
	editions := []string{"BuildTools", "Community", "Professional", "Enterprise"}
	for _, pf := range pfRoots {
		if pf == "" {
			continue
		}
		for _, ed := range editions {
			cand := filepath.Join(pf, "Microsoft Visual Studio", "2022", ed, "Common7", "Tools", "VsDevCmd.bat")
			if FileExists(cand) {
				return cand
			}
		}
	}
	return ""
}

// IngestVsDevEnv executes VsDevCmd.bat and captures the environment variables.
func IngestVsDevEnv(vsDevCmdPath string) (map[string]string, error) {
	if !FileExists(vsDevCmdPath) {
		return nil, fmt.Errorf("VsDevCmd.bat not found at %s", vsDevCmdPath)
	}

	cmdLine := fmt.Sprintf(`"%s" -no_logo -arch=amd64 -host_arch=amd64 >nul && set`, vsDevCmdPath)
	cmd := exec.Command("cmd.exe", "/s", "/c", cmdLine)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("executing VsDevCmd.bat: %w", err)
	}

	return ParseEnvBlock(out), nil
}

// FindVulkanSDK discovers the Vulkan SDK installation directory.
func FindVulkanSDK() string {
	if envSDK := os.Getenv("VULKAN_SDK"); envSDK != "" && DirExists(envSDK) {
		return envSDK
	}

	vulkanRoot := `C:\VulkanSDK`
	entries, err := os.ReadDir(vulkanRoot)
	if err == nil && len(entries) > 0 {
		var versions []string
		for _, entry := range entries {
			if entry.IsDir() {
				versions = append(versions, entry.Name())
			}
		}
		if len(versions) > 0 {
			sort.Slice(versions, func(i, j int) bool {
				return compareVersionStrings(versions[i], versions[j]) > 0
			})
			cand := filepath.Join(vulkanRoot, versions[0])
			if DirExists(cand) {
				return cand
			}
		}
	}

	fallback := `C:\VulkanSDK\1.4.350.0`
	if DirExists(fallback) {
		return fallback
	}
	return ""
}

// FindSignTool searches for signtool.exe on PATH and in Windows Kits.
func FindSignTool() string {
	if p, err := exec.LookPath("signtool.exe"); err == nil {
		return p
	}

	roots := []string{
		filepath.Join(os.Getenv("ProgramFiles(x86)"), "Windows Kits", "10", "bin"),
		filepath.Join(os.Getenv("ProgramFiles"), "Windows Kits", "10", "bin"),
	}

	var found string
	for _, root := range roots {
		if !DirExists(root) {
			continue
		}
		_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil
			}
			if !info.IsDir() && strings.EqualFold(info.Name(), "signtool.exe") {
				if strings.Contains(strings.ToLower(path), `\x64\`) {
					found = path
				}
			}
			return nil
		})
		if found != "" {
			return found
		}
	}
	return ""
}
