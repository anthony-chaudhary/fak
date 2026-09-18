package computebuild

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestVulkanShadersIncludesQ4KCoopMat pins the q4k_matmul_coopmat shader in the
// canonical Vulkan shader bundle (fak#12177). The 2D cooperative-matrix Q4_K
// prefill arm records its shader here so a bundle build that omits it fails
// loudly instead of silently dropping the kernel at pipeline creation.
func TestVulkanShadersIncludesQ4KCoopMat(t *testing.T) {
	found := false
	for _, name := range VulkanShaders {
		if name == "q4k_matmul_coopmat" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("VulkanShaders is missing %q; the Q4_K cooperative-matrix arm would fail to build", "q4k_matmul_coopmat")
	}

	// The scalar and Wave32 Q4_K arms must remain registered so the coopmat arm can
	// always fall back rather than removing the decode path.
	for _, required := range []string{"q4k_matmul", "q4k_matmul_wave32"} {
		ok := false
		for _, name := range VulkanShaders {
			if name == required {
				ok = true
				break
			}
		}
		if !ok {
			t.Fatalf("VulkanShaders is missing %q; the Q4_K scalar/Wave32 fallback must stay registered", required)
		}
	}
}

// TestQ4KCoopMatShaderDeclaresRequiredExtensions pins the GLSL extension
// declarations the cooperative-matrix Q4_K prefill arm needs to compile at all
// (fak#13220). glslc 2026.3 rejects the bare shader: the `gl_ScopeSubgroup`
// scope argument used by every `coopmat<...>` declaration is supplied by
// GL_KHR_memory_scope_semantics, so a shader that enables only
// GL_KHR_cooperative_matrix + GL_KHR_shader_subgroup_basic fails with
// "'gl_ScopeSubgroup' : required extension not requested" and the whole
// Vulkan binary build aborts BEFORE any physical run. This is a build-break
// guard, not a runtime-capability guard: the arm still falls back to the
// scalar kernel when the device lacks cooperative-matrix support.
func TestQ4KCoopMatShaderDeclaresRequiredExtensions(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "compute", "shaders", "q4k_matmul_coopmat.comp"))
	if err != nil {
		t.Fatalf("read q4k_matmul_coopmat.comp: %v", err)
	}
	text := string(src)
	if !strings.Contains(text, "gl_ScopeSubgroup") {
		t.Skip("coopmat shader no longer uses gl_ScopeSubgroup; extension guard retired")
	}
	for _, ext := range []string{
		"GL_KHR_cooperative_matrix",
		"GL_KHR_memory_scope_semantics",
	} {
		if !strings.Contains(text, "#extension "+ext+" : enable") {
			t.Fatalf("q4k_matmul_coopmat.comp must enable %s; without it glslc rejects gl_ScopeSubgroup and the Vulkan binary build aborts", ext)
		}
	}
}
