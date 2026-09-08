//go:build !windows

package computebuild

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

// discoverToolchainOS implements Linux/macOS toolchain discovery.
func discoverToolchainOS() (*Toolchain, error) {
	tc := &Toolchain{
		IsWindows: false,
	}

	// 1. C/C++ compiler and archiver discovery.
	if cc := os.Getenv("CC"); cc != "" {
		tc.CC = cc
	} else if p, err := exec.LookPath("gcc"); err == nil {
		tc.CC = p
	} else if p, err := exec.LookPath("clang"); err == nil {
		tc.CC = p
	} else if p, err := exec.LookPath("cc"); err == nil {
		tc.CC = p
	}

	if cxx := os.Getenv("CXX"); cxx != "" {
		tc.CXX = cxx
	} else if p, err := exec.LookPath("g++"); err == nil {
		tc.CXX = p
	} else if p, err := exec.LookPath("clang++"); err == nil {
		tc.CXX = p
	} else if p, err := exec.LookPath("c++"); err == nil {
		tc.CXX = p
	}

	if ar := os.Getenv("AR"); ar != "" {
		tc.AR = ar
	} else if p, err := exec.LookPath("ar"); err == nil {
		tc.AR = p
	} else if p, err := exec.LookPath("llvm-ar"); err == nil {
		tc.AR = p
	}

	if runtime.GOOS == "darwin" {
		tc.CxxRuntime = "-lc++"
	} else {
		tc.CxxRuntime = "-lstdc++"
	}

	// 2. Vulkan SDK discovery.
	vkRoot := os.Getenv("VULKAN_SDK")
	if vkRoot == "" {
		if DirExists("/usr/include/vulkan") || FileExists("/usr/include/vulkan/vulkan.h") {
			vkRoot = "/usr"
		} else if DirExists("/usr/local/include/vulkan") {
			vkRoot = "/usr/local"
		}
	}
	tc.VulkanSDK = vkRoot
	if vkRoot != "" {
		tc.VulkanInc = filepath.Join(vkRoot, "include")
		tc.VulkanLib = filepath.Join(vkRoot, "lib")
		glslcPath := filepath.Join(vkRoot, "bin", "glslc")
		if FileExists(glslcPath) {
			tc.GLSLC = glslcPath
		}
	}
	if tc.GLSLC == "" {
		if p, err := exec.LookPath("glslc"); err == nil {
			tc.GLSLC = p
		}
	}

	// 3. CUDA discovery.
	cudaHome := os.Getenv("CUDA_HOME")
	if cudaHome == "" {
		homeDir := os.Getenv("HOME")
		if homeDir == "" {
			homeDir = "/opt"
		}
		micromambaCuda := filepath.Join(homeDir, "cudaenv")
		if FileExists(filepath.Join(micromambaCuda, "bin", "nvcc")) {
			cudaHome = micromambaCuda
		} else if DirExists("/usr/local/cuda") && FileExists("/usr/local/cuda/bin/nvcc") {
			cudaHome = "/usr/local/cuda"
		} else if p, err := exec.LookPath("nvcc"); err == nil {
			cudaHome = filepath.Dir(filepath.Dir(p))
		}
	}

	tc.CUDAHome = cudaHome
	if cudaHome != "" {
		nvccPath := filepath.Join(cudaHome, "bin", "nvcc")
		if FileExists(nvccPath) {
			tc.NVCC = nvccPath
		}
		// Resolve include paths that actually exist.
		for _, inc := range []string{
			filepath.Join(cudaHome, "include"),
			filepath.Join(cudaHome, "targets", "x86_64-linux", "include"),
		} {
			if DirExists(inc) {
				tc.CUDAInc = append(tc.CUDAInc, inc)
			}
		}
		// Resolve library paths that actually exist.
		for _, lib := range []string{
			filepath.Join(cudaHome, "lib64"),
			filepath.Join(cudaHome, "lib"),
			filepath.Join(cudaHome, "targets", "x86_64-linux", "lib"),
		} {
			if DirExists(lib) {
				tc.CUDALib = append(tc.CUDALib, lib)
			}
		}
	}
	if tc.NVCC == "" {
		if p, err := exec.LookPath("nvcc"); err == nil {
			tc.NVCC = p
		}
	}

	return tc, nil
}
