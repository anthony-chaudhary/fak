package computebuild

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// VulkanShaders contains the GLSL compute shaders compiled for the Vulkan backend,
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
	"glm_kda_recurrent_reread",
	"glm_kda_recurrent_wave32",
	"flash_attn_dequant",
	"qwen35_gdn_tiled_transpose",
	"coopmat_wave32_wmma",
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
			return fmt.Errorf("required shader source missing: %s", src)
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

	if compiledCount == 0 {
		return fmt.Errorf("no shaders compiled in %s", shaderSrc)
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

// TestCxxToolchain performs a shift-left C++ compilation probe, proving the C++
// compiler and standard library are functional before running heavy build tasks.
func TestCxxToolchain(ctx context.Context, tc *Toolchain) error {
	if tc == nil || tc.CXX == "" {
		return fmt.Errorf("C++ compiler not found (clang++ or g++)")
	}

	tmpDir, err := os.MkdirTemp("", "fak-cxx-probe-*")
	if err != nil {
		return fmt.Errorf("creating temp probe dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	probeSrc := filepath.Join(tmpDir, "probe.cpp")
	probeObj := filepath.Join(tmpDir, "probe.o")

	srcContent := []byte("#include <vector>\n#include <string>\nint main() {\n    std::vector<std::string> v;\n    v.push_back(\"fak_probe\");\n    return v.empty() ? 1 : 0;\n}\n")
	if err := os.WriteFile(probeSrc, srcContent, 0644); err != nil {
		return fmt.Errorf("writing probe source: %w", err)
	}

	cxxArgs := []string{"-O3", "-std=c++17"}
	if tc.IsWindows {
		cxxArgs = append(cxxArgs, "-D_CRT_SECURE_NO_WARNINGS")
	} else {
		cxxArgs = append(cxxArgs, "-fPIC")
	}
	cxxArgs = append(cxxArgs, "-c", probeSrc, "-o", probeObj)

	env := os.Environ()
	if len(tc.VsDevEnv) > 0 {
		env = MergeEnviron(env, tc.VsDevEnv)
	}

	if err := RunCmd(ctx, tc.CXX, cxxArgs, tmpDir, env, nil, nil); err != nil {
		return fmt.Errorf("C++ toolchain probe failed: %w", err)
	}
	return nil
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
	if cfg.ReceiptPath == "" {
		cfg.ReceiptPath = filepath.Join(cfg.RepoRoot, DefaultVulkanBuildReceiptPath)
	} else if !filepath.IsAbs(cfg.ReceiptPath) {
		cfg.ReceiptPath = filepath.Join(cfg.RepoRoot, cfg.ReceiptPath)
	}
	if cfg.Command == "binary" && !cfg.SkipSmoke {
		cfg.Smoke = true
	}

	tracker := newReceiptTracker("vulkan", cfg.Command, cfg.ReceiptPath)
	cfg.Receipt = tracker.receipt
	defer func() {
		_ = tracker.finish(cfg.ReceiptPath)
	}()

	switch cfg.Command {
	case "shaders":
		return tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		})

	case "lib":
		if err := tracker.recordPhase("toolchain_probe", func() error {
			return TestCxxToolchain(ctx, cfg.Toolchain)
		}); err != nil {
			return err
		}
		var artifact *BuildArtifact
		if err := tracker.recordPhase("shim", func() error {
			art, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
			if err != nil {
				return err
			}
			artifact = art
			return nil
		}); err != nil {
			return err
		}
		tracker.receipt.Artifact = artifact
		return nil

	case "build":
		if err := tracker.recordPhase("toolchain_probe", func() error {
			return TestCxxToolchain(ctx, cfg.Toolchain)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shim", func() error {
			_, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"build", "-tags", "vulkan", "./internal/compute/"}
		if err := tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		}); err != nil {
			return err
		}
		outBinPath := cfg.OutBin
		if outBinPath != "" && !filepath.IsAbs(outBinPath) {
			outBinPath = filepath.Join(cfg.RepoRoot, outBinPath)
		}
		if outBinPath != "" && FileExists(outBinPath) {
			if art, err := InspectArtifact(outBinPath); err == nil {
				tracker.receipt.Artifact = art
			}
			if cfg.Smoke {
				if err := tracker.recordSmoke(ctx, outBinPath); err != nil {
					return err
				}
			}
		}
		return nil

	case "binary":
		if cfg.OutPkg == "" || cfg.OutBin == "" {
			err := fmt.Errorf("usage: binary <pkg> <out>")
			tracker.receipt.Outcome = "failed"
			tracker.receipt.ExitCode = 1
			tracker.receipt.Error = err.Error()
			return err
		}
		if err := tracker.recordPhase("toolchain_probe", func() error {
			return TestCxxToolchain(ctx, cfg.Toolchain)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shim", func() error {
			_, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"build", "-tags", "vulkan", "-o", cfg.OutBin, cfg.OutPkg}
		if err := tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		}); err != nil {
			return err
		}
		outBinPath := cfg.OutBin
		if !filepath.IsAbs(outBinPath) {
			outBinPath = filepath.Join(cfg.RepoRoot, outBinPath)
		}
		if FileExists(outBinPath) {
			if art, err := InspectArtifact(outBinPath); err == nil {
				tracker.receipt.Artifact = art
			}
		}
		if cfg.Smoke && FileExists(outBinPath) {
			if err := tracker.recordSmoke(ctx, outBinPath); err != nil {
				return err
			}
		}
		return nil

	case "test":
		if err := tracker.recordPhase("toolchain_probe", func() error {
			return TestCxxToolchain(ctx, cfg.Toolchain)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shaders", func() error {
			return BuildShaders(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
		}); err != nil {
			return err
		}
		if err := tracker.recordPhase("shim", func() error {
			_, err := BuildVulkanShim(ctx, cfg.Toolchain, cfg.RepoRoot, cfg.Stdout)
			return err
		}); err != nil {
			return err
		}
		cgoEnv := SynthesizeVulkanCgoEnv(cfg.Toolchain, cfg.PkgDir)
		env := MergeEnviron(os.Environ(), cgoEnv)
		if len(cfg.Toolchain.VsDevEnv) > 0 {
			env = MergeEnviron(env, cfg.Toolchain.VsDevEnv)
		}
		args := []string{"test", "-tags", "vulkan", "-count=1", "-v", "-run", "Vulkan|HALDevice", "./internal/compute/", "./internal/model/"}
		return tracker.recordPhase("build_or_test", func() error {
			return RunCmd(ctx, "go", args, cfg.RepoRoot, env, cfg.Stdout, cfg.Stderr)
		})

	default:
		err := fmt.Errorf("unknown subcommand: %s (use shaders|lib|build|binary|test)", cfg.Command)
		tracker.receipt.Outcome = "failed"
		tracker.receipt.ExitCode = 1
		tracker.receipt.Error = err.Error()
		return err
	}
}
