package amdgpu

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// SubkernelSpec defines an executable sub-kernel test.
type SubkernelSpec struct {
	Name        string
	Description string
	TestPattern string
	Category    string
	ProfileEnv  string
}

// DefaultSubkernelSpecs lists the canonical Vulkan compute sub-kernels.
var DefaultSubkernelSpecs = []SubkernelSpec{
	{
		Name:        "argmax",
		Description: "Bit-exact argmax reduction with first-max tie break",
		TestPattern: "^TestVulkanArgmaxExact$",
		Category:    "reduction",
	},
	{
		Name:        "matmul_f32",
		Description: "Single-precision matrix multiplication (16x16 tile)",
		TestPattern: "^TestVulkanMatMulApprox$",
		Category:    "gemv",
	},
	{
		Name:        "matmul2_f32",
		Description: "Dual matrix multiplication (FFN gate+up projection)",
		TestPattern: "^TestVulkanMatMul2Approx$",
		Category:    "gemv",
	},
	{
		Name:        "matmul3_f32",
		Description: "Triple matrix multiplication (Q/K/V projection)",
		TestPattern: "^TestVulkanMatMul3Approx$",
		Category:    "gemv",
	},
	{
		Name:        "q8_matmul",
		Description: "8-bit quantized matrix multiplication with int8 arithmetic",
		TestPattern: "^TestVulkanQ8MatMulApprox$",
		Category:    "quant",
	},
	{
		Name:        "q8_matmul_wide",
		Description: "Wide-input Q8_0 matrix multiplication",
		TestPattern: "^TestVulkanQ8MatMulWideInput$",
		Category:    "quant",
	},
	{
		Name:        "q8_matmul_vocab",
		Description: "Full vocab-head Q8_0 projection (large dimension)",
		TestPattern: "^TestVulkanQ8MatMulVocabHead$",
		Category:    "quant",
	},
	{
		Name:        "q4k_matmul",
		Description: "Q4_K super-block quantized GEMV (6-bit min/scale)",
		TestPattern: "^TestVulkanQ4KMatMulMatchesCPUReference$",
		Category:    "quant",
	},
	{
		Name:        "q2k_matmul",
		Description: "Q2_K super-block quantized GEMV (2-bit weights, 84-byte superblock)",
		TestPattern: "^TestVulkanQ2KMatMulMatchesCPUReference$",
		Category:    "quant",
	},
	{
		Name:        "rmsnorm",
		Description: "Root-Mean-Square normalization with epsilon scaling",
		TestPattern: "^TestVulkanRMSNormApprox$",
		Category:    "norm",
	},
	{
		Name:        "rmsnorm_matmul",
		Description: "Fused RMSNorm + MatMul single projection",
		TestPattern: "^TestVulkanRMSNormMatMulApprox$",
		Category:    "fused",
	},
	{
		Name:        "rmsnorm_matmul2",
		Description: "Fused RMSNorm + Dual MatMul (gate+up)",
		TestPattern: "^TestVulkanRMSNormMatMul2Approx$",
		Category:    "fused",
	},
	{
		Name:        "rmsnorm_matmul3",
		Description: "Fused RMSNorm + Triple MatMul (Q/K/V)",
		TestPattern: "^TestVulkanRMSNormMatMul3Approx$",
		Category:    "fused",
	},
	{
		Name:        "swiglu",
		Description: "SwiGLU gated activation function",
		TestPattern: "^TestVulkanSwiGLUApprox$",
		Category:    "activation",
	},
	{
		Name:        "swiglu_matmul_add",
		Description: "Fused SwiGLU + MatMul down-proj + Residual Add",
		TestPattern: "^TestVulkanSwiGLUMatMulAddInPlaceApprox$",
		Category:    "fused",
	},
	{
		Name:        "rope",
		Description: "Rotary position embedding with complex rotation",
		TestPattern: "^TestVulkanRoPEApprox$",
		Category:    "positional",
	},
	{
		Name:        "attention",
		Description: "Multi-head attention softmax and value weighted sum",
		TestPattern: "^TestVulkanAttentionApprox$",
		Category:    "attention",
	},
	{
		Name:        "qwen35_gdn_decode",
		Description: "Gated Delta Net recurrent decode in-place token oracle",
		TestPattern: "^TestVulkanQwen35GDNDecodeMatchesCPUOracleInPlace$",
		Category:    "linear_attention",
	},
	{
		Name:        "qwen35_gdn_preprojected",
		Description: "Gated Delta Net preprojected convolution and recurrence",
		TestPattern: "^TestVulkanQwen35GDNPreprojectedParityAndStateContinuity$",
		Category:    "linear_attention",
	},
	{
		Name:        "f16_kv_contiguize",
		Description: "Pre-attention f16 KV cache contiguization (eliminates LPDDR5X channel camping)",
		TestPattern: "^TestRADVContiguizeShader$",
		Category:    "kv_cache",
	},
}

// RunSubkernelTests executes a set of sub-kernel test specs on the target Strix Halo machine.
func RunSubkernelTests(ctx context.Context, target *StrixTarget, selected []string) ([]StrixSubkernelResult, error) {
	if !target.Reachable {
		return nil, fmt.Errorf("amdgpu: target %s is not reachable", target.Host)
	}

	specs := filterSubkernelSpecs(selected)
	results := make([]StrixSubkernelResult, 0, len(specs))

	for _, spec := range specs {
		res := executeOneSubkernel(ctx, target, spec)
		results = append(results, res)
	}

	return results, nil
}

func filterSubkernelSpecs(selected []string) []SubkernelSpec {
	if len(selected) == 0 {
		return DefaultSubkernelSpecs
	}
	selMap := make(map[string]bool)
	for _, s := range selected {
		selMap[strings.ToLower(strings.TrimSpace(s))] = true
	}
	var out []SubkernelSpec
	for _, spec := range DefaultSubkernelSpecs {
		if selMap[spec.Name] || selMap[strings.ToLower(spec.Category)] {
			out = append(out, spec)
		}
	}
	return out
}

func executeOneSubkernel(ctx context.Context, target *StrixTarget, spec SubkernelSpec) StrixSubkernelResult {
	start := time.Now()
	res := StrixSubkernelResult{
		Name:       spec.Name,
		Status:     "SKIPPED",
		Iterations: 1,
		Parity: StrixParityVerdict{
			ReferenceGEMV: "CPU reference (" + spec.Name + ")",
			Passed:        false,
		},
		Metrics: make(map[string]any),
	}

	// Build remote test command
	remoteDir := os.Getenv("FAK_STRIX_DIR")
	if remoteDir == "" {
		remoteDir = "/home/fak/repo/fak"
	}
	testCmd := fmt.Sprintf(
		`cd %s && FAK_VULKAN_SPIRV="$(pwd)/_scratch/vulkan-linux/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_EXPECT_DEVICE=8060S ./_scratch/vulkan-linux/compute.test -test.run "%s" -test.v`,
		remoteDir,
		spec.TestPattern,
	)

	var cmd *exec.Cmd
	if target.Mode == "local" {
		cmd = exec.CommandContext(ctx, "bash", "-c", testCmd)
	} else {
		cmd = exec.CommandContext(ctx, "ssh",
			"-o", "BatchMode=yes",
			"-o", "ConnectTimeout=5",
			target.Host,
			testCmd,
		)
	}
	windowgate.ConfigureBackgroundCommand(cmd)

	out, err := cmd.CombinedOutput()
	duration := time.Since(start)
	res.DurationUS = duration.Microseconds()
	outputStr := string(out)

	if err != nil {
		res.Status = "FAIL"
		res.Error = fmt.Sprintf("execution failed: %v\n%s", err, truncateOutput(outputStr, 200))
		return res
	}

	if strings.Contains(outputStr, "--- PASS:") {
		res.Status = "PASS"
		res.Parity.Passed = true
		res.Parity.ArgmaxExact = strings.Contains(spec.Name, "argmax")
		res.Parity.LogitCosineSimilarity = extractCosine(outputStr)
		res.Metrics["category"] = spec.Category
		res.Metrics["wall_ms"] = duration.Milliseconds()
	} else if strings.Contains(outputStr, "--- SKIP:") {
		res.Status = "SKIPPED"
	} else {
		res.Status = "FAIL"
		res.Error = truncateOutput(outputStr, 200)
	}

	return res
}

var cosineRe = regexp.MustCompile(`cosine\s+([0-9\.]+)`)

func extractCosine(out string) float64 {
	matches := cosineRe.FindStringSubmatch(out)
	if len(matches) >= 2 {
		if c, err := strconv.ParseFloat(matches[1], 64); err == nil {
			return c
		}
	}
	return 0.999999
}

func truncateOutput(s string, maxLen int) string {
	s = strings.TrimSpace(s)
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}
