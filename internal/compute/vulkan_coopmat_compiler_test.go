package compute

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestVulkanCoopMatOptimizedCompile reproduces #12177's shaderc optimizer
// crash in the beta update. It compiles the real shader; no device is opened
// and a compiler pass does not establish numerical or hardware qualification.
func TestVulkanCoopMatOptimizedCompile(t *testing.T) {
	compiler, err := exec.LookPath("glslc")
	if err != nil {
		t.Skip("compiler witness unavailable: glslc is required")
	}
	source := filepath.Join("shaders", "coopmat_wave32_wmma.comp")
	contents, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	for _, variant := range []string{"fp16", "int8"} {
		t.Run(variant, func(t *testing.T) {
			output := filepath.Join(t.TempDir(), "coopmat.spv")
			args := []string{"-O", "--target-env=vulkan1.2", "-fshader-stage=comp"}
			if variant == "int8" {
				args = append(args, "-DWMMA_INT8=1")
			}
			args = append(args, source, "-o", output)
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			if log, err := exec.CommandContext(ctx, compiler, args...).CombinedOutput(); err != nil {
				t.Fatalf("optimized %s compile: %v\n%s", variant, err, log)
			}
			spv, err := os.ReadFile(output)
			if err != nil {
				t.Fatal(err)
			}
			if len(spv) < 20 || len(spv)%4 != 0 || binary.LittleEndian.Uint32(spv[:4]) != 0x07230203 {
				t.Fatal("compiler output lacks a complete SPIR-V header")
			}
			t.Logf("compiler=%s source_sha256=%x spirv_sha256=%x bytes=%d physical_executed=false", compiler, sha256.Sum256(contents), sha256.Sum256(spv), len(spv))
		})
	}
}
