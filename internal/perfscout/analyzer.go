package perfscout

import (
	"regexp"
	"strings"
	"time"
)

var (
	reTokPerSec = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*(?:tok/s|token/s|t/s|tokens/sec)`)
	reLatency   = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*(?:ms\b|milliseconds)`)
	reContext   = regexp.MustCompile(`(?i)(\d+k?)\s*(?:context|ctx)`)
)

// AnalyzeRepo evaluates a raw GitHub repo record and produces a ScoredRepo.
func AnalyzeRepo(raw GitHubRawRepo, now time.Time, maxStars int) ScoredRepo {
	desc := strings.TrimSpace(raw.Description)
	name := strings.TrimSpace(raw.FullName)
	combined := name + " " + desc
	lower := strings.ToLower(combined)

	// Determine Model Family
	hasQwen := strings.Contains(lower, "qwen3.8") ||
		strings.Contains(lower, "qwen 3.8") ||
		strings.Contains(lower, "qwen4") ||
		strings.Contains(lower, "qwen-flash") ||
		strings.Contains(lower, "qwen3.8-flash")
	hasGLM := strings.Contains(lower, "glm-5.3") ||
		strings.Contains(lower, "glm 5.3") ||
		strings.Contains(lower, "glm53") ||
		strings.Contains(lower, "glm-5") ||
		strings.Contains(lower, "glm 5")

	var family ModelFamily
	switch {
	case hasQwen && hasGLM:
		family = FamilyDual
	case hasQwen:
		family = FamilyQwenFlash
	case hasGLM:
		family = FamilyGLMFlash
	default:
		family = FamilyUnknown
	}

	// Target Model
	targetModel := detectTargetModel(lower)

	// Hardware Target
	hardware := detectHardwareTarget(lower)

	// Serving Engine
	engine := detectServingEngine(lower)

	// Quantization
	quant := detectQuantization(lower)

	// Performance Proof
	proof := detectPerformanceProof(desc)

	// Special Mechanisms
	mechanisms := detectSpecialMechanisms(lower)

	// Evidence Grade & Score
	grade, score := computeGradeAndScore(desc, lower, proof, engine, hardware, quant, mechanisms)

	// Time calculations
	updatedAt, _ := time.Parse(time.RFC3339, raw.UpdatedAt)
	pushedAt, _ := time.Parse(time.RFC3339, raw.PushedAt)
	createdAt, _ := time.Parse(time.RFC3339, raw.CreatedAt)

	freshnessDays := int(now.Sub(updatedAt).Hours() / 24)
	if freshnessDays < 0 {
		freshnessDays = 0
	}

	unpopular := raw.StargazersCount <= maxStars

	return ScoredRepo{
		FullName:          raw.FullName,
		Description:       raw.Description,
		URL:               raw.URL,
		StargazersCount:   raw.StargazersCount,
		UpdatedAt:         updatedAt,
		PushedAt:          pushedAt,
		CreatedAt:         createdAt,
		Language:          raw.Language,
		ModelFamily:       family,
		TargetModel:       targetModel,
		HardwareTarget:    hardware,
		ServingEngine:     engine,
		Quantization:      quant,
		PerformanceProof:  proof,
		SpecialMechanisms: mechanisms,
		EvidenceGrade:     grade,
		PerformanceScore:  score,
		UnpopularIndie:    unpopular,
		FreshnessDays:     freshnessDays,
	}
}

func detectTargetModel(lower string) string {
	switch {
	case strings.Contains(lower, "qwen3.8-flash-next") || strings.Contains(lower, "qwen3.8 flash next"):
		return "Qwen3.8-Flash-Next"
	case strings.Contains(lower, "qwen3.8-27b") || strings.Contains(lower, "qwen3.8 27b"):
		return "Qwen3.8-27B"
	case strings.Contains(lower, "qwen3.8") || strings.Contains(lower, "qwen 3.8"):
		return "Qwen3.8"
	case strings.Contains(lower, "glm-5.3-flash") || strings.Contains(lower, "glm 5.3 flash") || strings.Contains(lower, "glm53-flash"):
		return "GLM-5.3-Flash"
	case strings.Contains(lower, "glm-5.3") || strings.Contains(lower, "glm 5.3"):
		return "GLM-5.3"
	case strings.Contains(lower, "glm-5.2") || strings.Contains(lower, "glm 5.2"):
		return "GLM-5.2"
	case strings.Contains(lower, "deepseek-v4-flash"):
		return "DeepSeek-V4-Flash"
	default:
		return "Next-Gen Flash OSS"
	}
}

func detectHardwareTarget(lower string) string {
	var targets []string
	if strings.Contains(lower, "dgx spark") || strings.Contains(lower, "gb10") {
		targets = append(targets, "DGX Spark (GB10)")
	}
	if strings.Contains(lower, "strix halo") || strings.Contains(lower, "gfx1151") {
		targets = append(targets, "AMD Strix Halo (gfx1151)")
	}
	if strings.Contains(lower, "rdna4") || strings.Contains(lower, "r9700") {
		targets = append(targets, "AMD RDNA4")
	}
	if strings.Contains(lower, "rtx 5090") || strings.Contains(lower, "5090") {
		targets = append(targets, "RTX 5090")
	}
	if strings.Contains(lower, "rtx 4090") || strings.Contains(lower, "4090 d") || strings.Contains(lower, "4090") {
		targets = append(targets, "RTX 4090")
	}
	if strings.Contains(lower, "rtx 3090") || strings.Contains(lower, "3090") {
		targets = append(targets, "RTX 3090")
	}
	if strings.Contains(lower, "rtx pro 6000") || strings.Contains(lower, "rtx 8000") {
		targets = append(targets, "RTX Pro Enterprise")
	}
	if strings.Contains(lower, "b200") || strings.Contains(lower, "blackwell") {
		targets = append(targets, "NVIDIA B200 (Blackwell)")
	}
	if strings.Contains(lower, "v100") {
		targets = append(targets, "Tesla V100")
	}
	if strings.Contains(lower, "mac") || strings.Contains(lower, "mlx") || strings.Contains(lower, "apple silicon") {
		targets = append(targets, "Apple Silicon")
	}
	if strings.Contains(lower, "tpu") {
		targets = append(targets, "Kaggle TPU")
	}
	if strings.Contains(lower, "cpu") || strings.Contains(lower, "laptop cpu") {
		targets = append(targets, "Host CPU")
	}
	if len(targets) == 0 {
		return "Custom Hardware"
	}
	return strings.Join(targets, ", ")
}

func detectServingEngine(lower string) string {
	var engines []string
	if strings.Contains(lower, "vllm") {
		engines = append(engines, "vLLM")
	}
	if strings.Contains(lower, "sglang") {
		engines = append(engines, "SGLang")
	}
	if strings.Contains(lower, "llama.cpp") || strings.Contains(lower, "ik_llama.cpp") {
		engines = append(engines, "llama.cpp")
	}
	if strings.Contains(lower, "ferrite") {
		engines = append(engines, "ferrite (Rust)")
	}
	if strings.Contains(lower, "ninfer") {
		engines = append(engines, "NInfer")
	}
	if strings.Contains(lower, "mlx") {
		engines = append(engines, "MLX")
	}
	if strings.Contains(lower, "rocm") {
		engines = append(engines, "ROCm")
	}
	if strings.Contains(lower, "vulkan") || strings.Contains(lower, "radv") {
		engines = append(engines, "Vulkan/RADV")
	}
	if strings.Contains(lower, "native c") || strings.Contains(lower, "c language") {
		engines = append(engines, "Native C")
	}
	if len(engines) == 0 {
		return "Custom OSS Runtime"
	}
	return strings.Join(engines, ", ")
}

func detectQuantization(lower string) string {
	var quants []string
	if strings.Contains(lower, "nvfp4") {
		quants = append(quants, "NVFP4")
	}
	if strings.Contains(lower, "exl3") {
		quants = append(quants, "EXL3")
	}
	if strings.Contains(lower, "fp8") {
		quants = append(quants, "FP8")
	}
	if strings.Contains(lower, "awq") {
		quants = append(quants, "AWQ INT4")
	}
	if strings.Contains(lower, "ud-q4_k_xl") || strings.Contains(lower, "q4_k") {
		quants = append(quants, "UD-Q4_K_XL")
	}
	if strings.Contains(lower, "int4") || strings.Contains(lower, "4-bit") || strings.Contains(lower, "4 bpw") {
		quants = append(quants, "INT4")
	}
	if strings.Contains(lower, "2.5 bpw") || strings.Contains(lower, "2.05") {
		quants = append(quants, "Sub-3bpw Quant")
	}
	if strings.Contains(lower, "gguf") {
		quants = append(quants, "GGUF")
	}
	if len(quants) == 0 {
		return "Unspecified / FP16"
	}
	return strings.Join(quants, ", ")
}

func detectPerformanceProof(desc string) string {
	var proofs []string
	if m := reTokPerSec.FindStringSubmatch(desc); len(m) > 0 {
		proofs = append(proofs, m[0])
	}
	if m := reLatency.FindStringSubmatch(desc); len(m) > 0 {
		proofs = append(proofs, m[0])
	}
	if m := reContext.FindStringSubmatch(desc); len(m) > 0 {
		proofs = append(proofs, m[0])
	}
	if len(proofs) == 0 {
		if strings.Contains(strings.ToLower(desc), "measured") || strings.Contains(strings.ToLower(desc), "benchmark") {
			return "Empirical Benchmarks Documented"
		}
		return "Operational Recipe"
	}
	return strings.Join(proofs, ", ")
}

func detectSpecialMechanisms(lower string) []string {
	var mechs []string
	if strings.Contains(lower, "mtp") {
		mechs = append(mechs, "Multi-Token Prediction (MTP)")
	}
	if strings.Contains(lower, "dflash") || strings.Contains(lower, "dflash2") {
		mechs = append(mechs, "DFlash2 Speculative Decoding")
	}
	if strings.Contains(lower, "ssd") || strings.Contains(lower, "nvme") || strings.Contains(lower, "disk cache") {
		mechs = append(mechs, "NVMe/SSD Tiered Weight & Cache Streaming")
	}
	if strings.Contains(lower, "ple") || strings.Contains(lower, "n-gram") {
		mechs = append(mechs, "PLE / N-gram Embedding Accelerator")
	}
	if strings.Contains(lower, "roce") || strings.Contains(lower, "rdma") || strings.Contains(lower, "thunderbolt") {
		mechs = append(mechs, "Thunderbolt RoCE-RDMA TP Fabric")
	}
	if strings.Contains(lower, "lru") && strings.Contains(lower, "moe") {
		mechs = append(mechs, "Device-Side GPU LRU MoE Expert Cache")
	}
	if strings.Contains(lower, "cuda graph") {
		mechs = append(mechs, "CUDA Graph Cache Optimization")
	}
	if strings.Contains(lower, "sm_100") || strings.Contains(lower, "sm120") || strings.Contains(lower, "sm_110") || strings.Contains(lower, "sm_121") {
		mechs = append(mechs, "Next-Gen Arch sm_100/sm_120 Tuning")
	}
	if strings.Contains(lower, "speculative") && !strings.Contains(lower, "mtp") && !strings.Contains(lower, "dflash") {
		mechs = append(mechs, "Speculative Decoding")
	}
	return mechs
}

func computeGradeAndScore(desc, lower, proof, engine, hardware, quant string, mechs []string) (EvidenceGrade, int) {
	score := 10 // Base score for model identity match

	// Hardware points
	if hardware != "Custom Hardware" {
		score += 20
	}

	// Engine points
	if engine != "Custom OSS Runtime" {
		score += 15
	}

	// Quantization points
	if quant != "Unspecified / FP16" {
		score += 15
	}

	// Special mechanism points
	score += len(mechs) * 10

	// Performance Proof points & grade
	grade := GradeRecipe
	hasTokSec := reTokPerSec.MatchString(desc)
	hasLatency := reLatency.MatchString(desc)
	hasMeasuredWord := strings.Contains(lower, "measured") || strings.Contains(lower, "tok/s") || strings.Contains(lower, "token/s")

	if hasTokSec || hasLatency {
		grade = GradeMeasured
		score += 35
	} else if len(mechs) > 0 && (strings.Contains(lower, "kernel") || strings.Contains(lower, "patch") || strings.Contains(lower, "rust") || strings.Contains(lower, "native c")) {
		grade = GradeKernel
		score += 30
	} else if hasMeasuredWord || strings.Contains(lower, "benchmark") {
		grade = GradeEvaluation
		score += 20
	}

	return grade, score
}
