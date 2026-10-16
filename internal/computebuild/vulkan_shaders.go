package computebuild

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// BuildShaders compiles all registered GLSL shaders to SPIR-V using glslc.
func BuildShaders(ctx context.Context, tc *Toolchain, repoRoot string, stdout io.Writer) error {
	if tc == nil || tc.GLSLC == "" {
		return fmt.Errorf("glslc not found — set VULKAN_SDK or install Vulkan SDK")
	}

	return buildVulkanShaders(repoRoot, stdout, func(src, dst string) error {
		args := []string{"-O", "--target-env=vulkan1.2", "-fshader-stage=comp", src, "-o", dst}
		return RunCmd(ctx, tc.GLSLC, args, filepath.Join(repoRoot, "internal", "compute"), nil, stdout, os.Stderr)
	})
}

func buildVulkanShaders(repoRoot string, stdout io.Writer, compile func(src, dst string) error) error {
	pkgDir := filepath.Join(repoRoot, "internal", "compute")
	shaderSrc := filepath.Join(pkgDir, "shaders")
	spvOut := filepath.Join(pkgDir, "spirv")

	if err := os.MkdirAll(spvOut, 0755); err != nil {
		return fmt.Errorf("creating spirv output directory: %w", err)
	}

	// Validate canonical source before compiling or replacing any module.
	qualified, err := readVulkanQualifiedIndexScale(repoRoot)
	if err != nil {
		return err
	}
	compiledCount := 0
	for _, s := range vulkanCurrentRegistry.stems() {
		src := filepath.Join(shaderSrc, s+".comp")
		dst := filepath.Join(spvOut, s+".spv")

		if !FileExists(src) {
			return fmt.Errorf("required shader source missing: %s", src)
		}

		if s == vulkanQualifiedIndexScaleStem {
			if err := writeVulkanQualifiedModule(dst, qualified); err != nil {
				return err
			}
			if stdout != nil {
				fmt.Fprintf(stdout, "[vulkan] qualified artifact %s -> %s.spv\n", vulkanQualifiedIndexScaleHeader, s)
			}
			compiledCount++
			continue
		}

		if stdout != nil {
			fmt.Fprintf(stdout, "[vulkan] glslc %s.comp -> %s.spv\n", s, s)
		}
		if err := compile(src, dst); err != nil {
			return fmt.Errorf("glslc failed on %s.comp: %w", s, err)
		}
		compiledCount++
	}

	if compiledCount == 0 {
		return fmt.Errorf("no shaders compiled in %s", shaderSrc)
	}

	if stdout != nil {
		fmt.Fprintf(stdout, "[vulkan] prepared %d SPIR-V modules to %s\n", compiledCount, spvOut)
	}
	return nil
}
