package computebuild

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// VulkanShaders contains the 34 GLSL compute shaders compiled for the Vulkan backend,
// exactly matching internal/compute/build_vulkan.ps1.
var VulkanShaders = []string{
	"matmul",
	"matmul_add",
	"matmul_argmax",
	"matmul_argmax_blocks",
	"matmul2",
	"matmul3",
	"rmsnorm",
	"rmsnorm_matmul",
	"rmsnorm_matmul2",
	"rmsnorm_matmul3",
	"rmsnorm_matmul_argmax_blocks",
	"rope",
	"swiglu",
	"swiglu_matmul_add",
	"add",
	"add_bias",
	"attention",
	"argmax",
	"argmax_pairs",
	"q8_matmul",
	"q8_matmul2",
	"q8_matmul3",
	"rmsnorm_q8_matmul2",
	"rmsnorm_q8_matmul3",
	"swiglu_q8_matmul_add",
	"qwen35_gdn_conv",
	"qwen35_gdn_recurrent",
	"q4k_matmul",
	"q2k_matmul",
	"qwen35_split_qg_panel",
	"qwen35_partial_rope_panel",
	"qwen35_causal_attention_panel",
	"sigmoid_mul",
	"q8_matmul_decode",
}

// BuildShaders compiles all registered GLSL shaders to SPIR-V using glslc.
func BuildShaders(ctx context.Context, tc *Toolchain, repoRoot string, stdout io.Writer) error {
	if tc == nil || tc.GLSLC == "" {
		return fmt.Errorf("glslc not found — set VULKAN_SDK or install Vulkan SDK")
	}

	pkgDir := filepath.Join(repoRoot, "internal", "compute")
	shaderSrc := filepath.Join(pkgDir, "shaders")
	spvOut := filepath.Join(pkgDir, "spirv")

	if err := os.MkdirAll(spvOut, 0755); err != nil {
		return fmt.Errorf("creating spirv output directory: %w", err)
	}

	compiledCount := 0
	for _, s := range VulkanShaders {
		src := filepath.Join(shaderSrc, s+".comp")
		dst := filepath.Join(spvOut, s+".spv")

		if !FileExists(src) {
			continue
		}

		if stdout != nil {
			fmt.Fprintf(stdout, "[vulkan] glslc %s.comp -> %s.spv\n", s, s)
		}
		args := []string{"-O", "--target-env=vulkan1.2", "-fshader-stage=comp", src, "-o", dst}
		if err := RunCmd(ctx, tc.GLSLC, args, pkgDir, nil, stdout, os.Stderr); err != nil {
			return fmt.Errorf("glslc failed on %s.comp: %w", s, err)
		}
		compiledCount++
	}

	if stdout != nil {
		fmt.Fprintf(stdout, "[vulkan] compiled %d SPIR-V modules to %s\n", compiledCount, spvOut)
	}
	return nil
}

// BuildVulkanShim compiles vulkan_shim.cpp into vulkan_shim.o and archives it to libfakvulkan.a.
func BuildVulkanShim(ctx context.Context, tc *Toolchain, repoRoot string, stdout io.Writer) (*BuildArtifact, error) {
	if tc == nil || tc.CXX == "" {
		return nil, fmt.Errorf("C++ compiler not found (clang++ or g++)")
	}
	if tc.AR == "" {
		return nil, fmt.Errorf("archiver not found (ar or llvm-ar)")
	}

	pkgDir := filepath.Join(repoRoot, "internal", "compute")
	shimCpp := filepath.Join(pkgDir, "vulkan_shim.cpp")
	shimObj := filepath.Join(pkgDir, "vulkan_shim.o")
	libOut := filepath.Join(pkgDir, "libfakvulkan.a")

	cxxArgs := []string{"-O3", "-std=c++17"}
	if tc.IsWindows {
		cxxArgs = append(cxxArgs, "-D_CRT_SECURE_NO_WARNINGS")
	} else {
		cxxArgs = append(cxxArgs, "-fPIC")
	}
	if tc.VulkanInc != "" {
		cxxArgs = append(cxxArgs, "-I"+tc.VulkanInc)
	}
	cxxArgs = append(cxxArgs, "-c", shimCpp, "-o", shimObj)

	if stdout != nil {
		fmt.Fprintf(stdout, "[vulkan] C++ compile vulkan_shim.cpp -> libfakvulkan.a\n")
	}

	env := os.Environ()
	if len(tc.VsDevEnv) > 0 {
		env = MergeEnviron(env, tc.VsDevEnv)
	}
	if err := RunCmd(ctx, tc.CXX, cxxArgs, pkgDir, env, stdout, os.Stderr); err != nil {
		return nil, fmt.Errorf("compiling vulkan_shim.cpp: %w", err)
	}

	arArgs := []string{"rcs", libOut, shimObj}
	if err := RunCmd(ctx, tc.AR, arArgs, pkgDir, env, stdout, os.Stderr); err != nil {
		return nil, fmt.Errorf("archiving libfakvulkan.a: %w", err)
	}

	fi, err := os.Stat(libOut)
	if err != nil {
		return nil, fmt.Errorf("stat libfakvulkan.a: %w", err)
	}

	if stdout != nil {
		fmt.Fprintf(stdout, "[vulkan] built libfakvulkan.a (%d bytes)\n", fi.Size())
	}
	return &BuildArtifact{
		Path:      libOut,
		SizeBytes: fi.Size(),
	}, nil
}

// RunVulkan orchestrates Vulkan build tasks according to cfg.Command.
func RunVulkan(ctx context.Context, cfg *VulkanConfig) error {
	if cfg == nil {
		return fmt.Errorf("nil VulkanConfig")
	}
	if cfg.RepoRoot == "" {
		root, err := ResolveRepoRoot("")
		if err != nil {
			return err
		}
		cfg.RepoRoot = root
	}
	if cfg.PkgDir == "" {
		cfg.PkgDir = filepath.Join(cfg.RepoRoot, "internal", "compute")
	}
	if cfg.Toolchain == nil {
		tc, err := DiscoverToolchain()
		if err != nil {
			return fmt.Errorf("discovering toolchain: %w", err)
		}
		cfg.Toolchain = tc
	}

	switch cfg.Command {
	case "shaders":
		return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
	case "lib":
		_, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		return err
	case "build":
		if err := BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout); err != nil {
			return err
		}
		if _, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"build", "-tags", "vulkan", "./internal/compute/"}
		return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
	case "binary":
		if cfg.OutPkg == "" || cfg.OutBin == "" {
			return fmt.Errorf("usage: binary <pkg> <out>")
		}
		if err := BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout); err != nil {
			return err
		}
		if _, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"build", "-tags", "vulkan", "-o", cfg.OutBin, cfg.OutPkg}
		return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
	case "test":
		if err := BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout); err != nil {
			return err
		}
		if _, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"test", "-tags", "vulkan", "-count=1", "-v", "-run", "Vulkan|HALDevice", "./internal/compute/", "./internal/model/"}
		return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
	default:
		return fmt.Errorf("unknown subcommand: %s (use shaders|lib|build|binary|test)", cfg.Command)
	}
}
