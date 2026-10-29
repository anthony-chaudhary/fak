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
		"const uint SHARED_CAP = 8192u;",
		"subgroupShuffleXor(amax, 4u)",
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
		"static constexpr int MAX_DISPATCH_BUFS = 20;",
		"P(\"qwen35_gdn_q8_in_proj.spv\"),",
		"13, 5 * sizeof(int)) ? 1 : 0;",
		"int fvk_qwen35_gdn_q8_in_proj_f32(",
		"Buffer* bufs[13]",
	} {
		if !strings.Contains(src, token) {
			t.Errorf("Vulkan shim missing fused Q8 GDN contract token %q", token)
		}
	}
	// Membership in the shared GDN pre-dispatch case run must not depend on
	// adjacent kernels; adding another kernel to the run must preserve this
	// source witness (order-independence pattern from 07116aefba).
	const gdnAnchor = "case K_QWEN35_GDN_Q8_IN_PROJ:"
	gdnRun := shimCaseRun(src, gdnAnchor)
	if gdnRun == "" {
		t.Fatalf("missing shared GDN pre-dispatch case run anchored at %q", gdnAnchor)
	}
	if !strings.Contains(gdnRun, "case K_QWEN35_GDN_CONV:") {
		t.Error("shared GDN pre-dispatch case run lost K_QWEN35_GDN_CONV")
	}
	// Negative control: the witness must read the anchored run, not a coincident
	// occurrence elsewhere in the shim.
	broken := strings.Replace(src, gdnRun, strings.ReplaceAll(gdnRun, "case K_QWEN35_GDN_CONV:", "case K_QWEN35_GDN_RECURRENT:"), 1)
	if strings.Contains(shimCaseRun(broken, gdnAnchor), "case K_QWEN35_GDN_CONV:") {
		t.Error("source witness admitted GDN conv dispatch outside the anchored case run")
	}
	_, build := qwen35GDNComputeFixture(t, "build_vulkan.ps1")
	if !strings.Contains(string(build), `$shaders += "qwen35_gdn_q8_in_proj"`) {
		t.Error("Vulkan build does not compile the fused Q8 GDN shader")
	}
}

// fak-test:runtime fast est=1ms lane=default
// Source contract only: optional fused pipeline failures must not clear the
// four-kernel routed ID capability. This does not witness native execution.
func TestVulkanRoutedIDCapabilityIndependentOfFusedModules(t *testing.T) {
	_, shim := qwen35GDNComputeFixture(t, "vulkan_shim.cpp")
	src := string(shim)
	const prefix = "const bool v41RoutedIDBuilt ="
	start := strings.Index(src, prefix)
	if start < 0 {
		t.Fatal("missing routed ID capability initializer")
	}
	end := strings.Index(src[start:], ";")
	if end < 0 {
		t.Fatal("unterminated routed ID capability initializer")
	}
	initializer := strings.Join(strings.Fields(src[start:start+end+1]), " ")
	const want = `const bool v41RoutedIDBuilt = g_v41IndexerFloatControls && buildKernel(g_kern[K_V41_Q2K_MATVEC_ID], P("v41_q2k_matvec_id.spv"), 11, 16) && buildKernel(g_kern[K_V41_Q3K_MATVEC_ID], P("v41_q3k_matvec_id.spv"), 11, 16) && buildKernel(g_kern[K_V41_ROUTED_SWIGLU_ID], P("v41_routed_swiglu_id.spv"), 5, 12) && buildKernel(g_kern[K_V41_ROUTED_PUBLISH_SUM], P("v41_routed_publish_sum.spv"), 4, 12);`
	if initializer != want {
		t.Fatal("routed ID capability must depend only on float controls and its four legacy kernels")
	}
	const optional = `if (v41RoutedIDBuilt && buildKernel(g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID], P("v41_routed_gate_up_swiglu_id.spv"), 20, 20)) { buildKernel(g_kern[K_V41_ROUTED_DOWN_SUM], P("v41_routed_down_sum.spv"), 12, 16); }`
	if !strings.Contains(strings.Join(strings.Fields(src[start+end+1:]), " "), optional) {
		t.Fatal("optional fused pair must build separately after routed ID succeeds")
	}
}

// fak-test:runtime fast est=1ms lane=default
// Source contract only: either missing fused pipeline must refuse both C entry
// points before sticky errors or operand handling. No native execution is witnessed.
func TestVulkanRoutedFusedEntryPointAvailabilityContract(t *testing.T) {
	_, shim := qwen35GDNComputeFixture(t, "vulkan_shim.cpp")
	src := string(shim)
	body := func(t *testing.T, signature string) string {
		t.Helper()
		start := strings.Index(src, signature)
		if start < 0 {
			t.Fatalf("missing definition %q", signature)
		}
		open := strings.Index(src[start:], "{")
		if open < 0 {
			t.Fatalf("missing body for %q", signature)
		}
		start += open + 1
		end := strings.Index(src[start:], "\n}")
		if end < 0 {
			t.Fatalf("unterminated body for %q", signature)
		}
		return strings.Join(strings.Fields(src[start:start+end]), " ")
	}

	// Exact conjunction: gate/up and down/sum are independently mandatory,
	// including their descriptor counts, even when the legacy kernels are ready.
	const capability = `return g_ready && g_v41IndexerFloatControls && g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID].pipe != VK_NULL_HANDLE && g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID].nbuf == 20 && g_kern[K_V41_ROUTED_DOWN_SUM].pipe != VK_NULL_HANDLE && g_kern[K_V41_ROUTED_DOWN_SUM].nbuf == 12;`
	if got := body(t, "int fvk_have_v41_routed_fused(void)"); got != capability {
		t.Fatal("fused capability must require both complete pipelines")
	}
	const state = `if (!g_ready) return 1; if (!fvk_have_v41_routed_id() || (requireFused && !fvk_have_v41_routed_fused())) return 3; if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus; if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4; if (g_batching && !g_stepBatch) return 2; return 0;`
	if got := body(t, "static int v41RoutedIDState(bool requireFused = false)"); got != state {
		t.Fatal("routed state must keep readiness, capability, submission, restore and batching error precedence")
	}
	for _, tc := range []struct {
		entry string
		arg   string
	}{
		{"fvk_v41_matvec_id_f32", ""},
		{"fvk_v41_routed_swiglu_id_f32", ""},
		{"fvk_v41_routed_publish_sum_f32", ""},
		{"fvk_v41_routed_gate_up_swiglu_id_f32", "true"},
		{"fvk_v41_routed_down_sum_f32", "true"},
	} {
		t.Run(tc.entry, func(t *testing.T) {
			want := "const int state = v41RoutedIDState(" + tc.arg + "); if (state != 0) return state;"
			if !strings.HasPrefix(body(t, "int "+tc.entry+"("), want) {
				t.Fatal("entry point must check its complete capability before operands or dispatch")
			}
		})
	}
}
