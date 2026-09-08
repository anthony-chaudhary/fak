package compute

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func qwen35GDNComputeFixture(t *testing.T, relative string) (string, []byte) {
	t.Helper()
	for _, candidate := range []string{
		relative,
		filepath.Join("internal", "compute", relative),
		filepath.Join("..", "..", "internal", "compute", relative),
	} {
		raw, err := os.ReadFile(candidate)
		if err == nil {
			return candidate, raw
		}
	}
	t.Fatalf("cannot locate internal/compute/%s", relative)
	return "", nil
}

func TestQwen35GDNQ8InputProjectionShaderContract(t *testing.T) {
	shaderPath, raw := qwen35GDNComputeFixture(t, filepath.Join("shaders", "qwen35_gdn_q8_in_proj.comp"))
	shader := string(raw)
	for binding := 0; binding < 13; binding++ {
		token := "binding = " + strconv.Itoa(binding)
		if !strings.Contains(shader, token) {
			t.Errorf("fused Q8 GDN shader missing descriptor %d", binding)
		}
	}
	for _, token := range []string{
		"layout(local_size_x = 256) in;",
		"const uint SHARED_CAP = 2048u;",
		"for (uint wb = 0u; wb < nblk; wb += WIN_BLOCKS)",
		"uint totalOut = out0 + out1 + out2 + uint(pc.out3);",
		"Y0[outputRow] = acc;",
		"Y3[outputRow - out0 - out1 - out2] = acc;",
	} {
		if !strings.Contains(shader, token) {
			t.Errorf("fused Q8 GDN shader missing contract token %q", token)
		}
	}

	glslc := "glslc"
	if _, err := exec.LookPath(glslc); err != nil {
		vulkanSDK := os.Getenv("VULKAN_SDK")
		if vulkanSDK == "" {
			vulkanSDK = `C:\VulkanSDK\1.4.350.0`
		}
		candidate := filepath.Join(vulkanSDK, "Bin", "glslc.exe")
		if _, err := os.Stat(candidate); err == nil {
			glslc = candidate
		}
	}
	if _, err := exec.LookPath(glslc); err == nil || filepath.IsAbs(glslc) {
		spv := filepath.Join(t.TempDir(), "qwen35_gdn_q8_in_proj.spv")
		out, err := exec.Command(glslc, "-O", "--target-env=vulkan1.2", "-fshader-stage=comp", shaderPath, "-o", spv).CombinedOutput()
		if err != nil {
			t.Fatalf("glslc failed: %v\n%s", err, out)
		}
		if info, err := os.Stat(spv); err != nil || info.Size() == 0 {
			t.Fatalf("compiled SPIR-V missing or empty: %v", err)
		}
	}
}

func TestQwen35GDNQ8InputProjectionShimContract(t *testing.T) {
	_, shim := qwen35GDNComputeFixture(t, "vulkan_shim.cpp")
	src := string(shim)
	for _, token := range []string{
		"K_QWEN35_GDN_Q8_IN_PROJ",
		"case K_QWEN35_GDN_Q8_IN_PROJ: case K_QWEN35_GDN_CONV",
		"static constexpr int MAX_DISPATCH_BUFS = 13;",
		"P(\"qwen35_gdn_q8_in_proj.spv\"),",
		"13, 5 * sizeof(int)) ? 1 : 0;",
		"int fvk_qwen35_gdn_q8_in_proj_f32(",
		"Buffer* bufs[13]",
	} {
		if !strings.Contains(src, token) {
			t.Errorf("Vulkan shim missing fused Q8 GDN contract token %q", token)
		}
	}
	_, build := qwen35GDNComputeFixture(t, "build_vulkan.ps1")
	if !strings.Contains(string(build), `$shaders += "qwen35_gdn_q8_in_proj"`) {
		t.Error("Vulkan build does not compile the fused Q8 GDN shader")
	}
}
