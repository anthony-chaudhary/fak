package macbench

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// AgenticComparisonSchema is the canonical schema for multi-agent shared cache head-to-head comparisons.
const AgenticComparisonSchema = "fak.macbench.agentic-comparison.v1"

// MinAgenticSpeedupRatio is the minimum speedup ratio required for agentic comparison verification.
// Under honest post-#11855 methodology (without unphysical queue wait double-counting), baseline
// speedup is ~1.86x (and ~2.51x for 20k context).
const MinAgenticSpeedupRatio = 1.50

// AgenticComparisonPacket records a complete, matched-envelope head-to-head comparison
// between fak-native (inkernel) and llama.cpp (reference) on agentic shared cache workloads.
type AgenticComparisonPacket struct {
	Schema            string                  `json:"schema"`
	GeneratedAt       string                  `json:"generated_at"`
	CampaignID        string                  `json:"campaign_id"`
	HostID            string                  `json:"host_id"`
	Provenance        string                  `json:"provenance"` // "MODELED" or "WITNESSED" per docs/standards/simulated-results-discipline.md
	IsPhysicalSilicon bool                    `json:"is_physical_silicon"`
	UnmodeledEffects  []string                `json:"unmodeled_effects,omitempty"`
	Model             ComparisonModel         `json:"model"`
	Hardware          ComparisonHardware      `json:"hardware"`
	OS                ComparisonOS            `json:"os"`
	Workload          AgenticWorkloadShape    `json:"workload"`
	QualityPolicy     ComparisonQualityPolicy `json:"quality_policy"`
	Arms              []AgenticComparisonArm  `json:"arms"`
	Summary           AgenticSummary          `json:"summary"`
	MinSpeedupRatio   float64                 `json:"min_speedup_ratio,omitempty"`
}

// AgenticWorkloadShape defines the multi-agent session parameters.
type AgenticWorkloadShape struct {
	Concurrency        int `json:"concurrency"`          // K agents (e.g. 4 or 8)
	Horizon            int `json:"horizon"`              // H interaction turns per agent (e.g. 20)
	SharedPrefixTokens int `json:"shared_prefix_tokens"` // P tokens in shared system+tool preamble (e.g. 4096)
	TurnDeltaTokens    int `json:"turn_delta_tokens"`    // Input tokens ingested per turn (e.g. 128)
	TurnOutputTokens   int `json:"turn_output_tokens"`   // Assistant tokens generated per turn (e.g. 64)
}

// AgenticComparisonArm records performance and memory telemetry for one engine arm.
type AgenticComparisonArm struct {
	Name              string                  `json:"name"`                // "fak-native" or "llama.cpp"
	Engine            string                  `json:"engine"`              // "fak-native" or "llama.cpp"
	Runtime           string                  `json:"runtime"`             // "inkernel" or "reference"
	RuntimeRevision   string                  `json:"runtime_revision"`    // Git revision / engine version
	EvidenceKind      string                  `json:"evidence_kind"`       // "modeled" or "observed"
	PrefixStrategy    string                  `json:"prefix_strategy"`     // "radix-shared" or "slot-isolated"
	PrefixEvalCount   int                     `json:"prefix_eval_count"`   // Number of times prefix was evaluated
	PromptTokens      uint64                  `json:"prompt_tokens"`       // Total prompt tokens presented
	ReusedTokens      uint64                  `json:"reused_tokens"`       // Prompt tokens served from cache
	ReuseRatio        float64                 `json:"reuse_ratio"`         // ReusedTokens / PromptTokens
	TotalWallMS       float64                 `json:"total_wall_ms"`       // Total elapsed wall-clock milliseconds
	PrefillMS         float64                 `json:"prefill_ms"`          // Time spent in prefill compute
	DecodeMS          float64                 `json:"decode_ms"`           // Time spent in autoregressive decode
	QueueContentionMS float64                 `json:"queue_contention_ms"` // Prefill queue / slot serialization wait
	P50TTFTMS         float64                 `json:"p50_ttft_ms"`         // Median time-to-first-token in ms
	P95TTFTMS         float64                 `json:"p95_ttft_ms"`         // 95th percentile TTFT in ms
	PeakMemoryMB      float64                 `json:"peak_memory_mb"`      // Peak memory consumption in MB
	AgentsPerGB       float64                 `json:"agents_per_gb"`       // Density: Concurrency / PeakMemoryGB
	EffectiveTokS     float64                 `json:"effective_tok_s"`     // Total output tokens / (TotalWallMS / 1000)
	Quality           ComparisonQualityResult `json:"quality"`             // Strict quality/token parity receipt
	RawResult         ComparisonRawResult     `json:"raw_result"`          // Path and SHA256 of raw sample log
	Repro             []string                `json:"repro"`               // Reproduction commands
}

// AgenticSummary aggregates head-to-head comparison ratios.
type AgenticSummary struct {
	SpeedupRatio   float64 `json:"speedup_ratio"`    // llama.cpp TotalWallMS / fak-native TotalWallMS (must be >= 1.50)
	MemorySavedMB  float64 `json:"memory_saved_mb"`  // Peak memory savings from prefix sharing
	TTFTSpeedupP50 float64 `json:"ttft_speedup_p50"` // llama.cpp P50TTFTMS / fak-native P50TTFTMS
	Verified       bool    `json:"verified"`         // True if all gate conditions hold
}

// ValidateAgenticComparisonPacket validates an agentic comparison packet against strict fail-closed criteria.
func ValidateAgenticComparisonPacket(p AgenticComparisonPacket) error {
	var problems []string
	require := func(ok bool, field, detail string) {
		if !ok {
			problems = append(problems, field+": "+detail)
		}
	}

	require(p.Schema == AgenticComparisonSchema, "schema", "must be "+AgenticComparisonSchema)
	generatedAt, generatedErr := time.Parse(time.RFC3339, p.GeneratedAt)
	require(generatedErr == nil, "generated_at", "must be RFC3339")
	require(strings.TrimSpace(p.CampaignID) != "", "campaign_id", "is required")
	require(validSHA256(p.HostID), "host_id", "must be a valid SHA-256")

	// Provenance validation per docs/standards/simulated-results-discipline.md
	provenance := strings.ToUpper(strings.TrimSpace(p.Provenance))
	require(provenance == "MODELED" || provenance == "SIMULATED" || provenance == "WITNESSED",
		"provenance", "must be MODELED, SIMULATED, or WITNESSED")
	if !p.IsPhysicalSilicon {
		require(provenance == "MODELED" || provenance == "SIMULATED",
			"provenance", "non-physical run must be labeled MODELED or SIMULATED")
		require(len(p.UnmodeledEffects) >= 3, "unmodeled_effects",
			"modeled run must articulate unmodeled physical effects (at least 3)")
	}

	// Model validation: must strictly be Qwen3.8-27B Q4_K_M
	require(strings.EqualFold(strings.TrimSpace(p.Model.Family), "Qwen3.8"), "model.family", "must be exactly Qwen3.8")
	modelID := strings.ToLower(strings.TrimSpace(p.Model.ID))
	require(modelID == "qwen3.8" || strings.HasPrefix(modelID, "qwen3.8-"), "model.id", "must identify Qwen3.8")
	require(strings.TrimSpace(p.Model.SourceRevision) != "", "model.source_revision", "is required")
	require(validSHA256(p.Model.CanonicalWeightsSHA256), "model.canonical_weights_sha256", "must be a valid SHA-256")
	require(strings.EqualFold(strings.TrimSpace(p.Model.Quant), "Q4_K_M"), "model.quant", "must be Q4_K_M")

	// Hardware and OS validation
	require(strings.TrimSpace(p.Hardware.Model) != "", "hardware.model", "is required")
	require(strings.TrimSpace(p.Hardware.Chip) != "", "hardware.chip", "is required")
	require(p.Hardware.MemoryBytes > 0, "hardware.memory_bytes", "must be positive")
	require(strings.TrimSpace(p.OS.Name) != "", "os.name", "is required")
	require(strings.TrimSpace(p.OS.Version) != "", "os.version", "is required")
	require(strings.TrimSpace(p.OS.Build) != "", "os.build", "is required")

	// Workload validation
	require(p.Workload.Concurrency >= 2, "workload.concurrency", "must be at least 2 agents")
	require(p.Workload.Horizon >= 5, "workload.horizon", "must be at least 5 turns")
	require(p.Workload.SharedPrefixTokens >= 1024, "workload.shared_prefix_tokens", "must be at least 1024 tokens")
	require(p.Workload.TurnDeltaTokens > 0, "workload.turn_delta_tokens", "must be positive")
	require(p.Workload.TurnOutputTokens > 0, "workload.turn_output_tokens", "must be positive")

	// Quality policy validation
	require(strings.TrimSpace(p.QualityPolicy.ID) != "", "quality_policy.id", "is required")
	require(strings.TrimSpace(p.QualityPolicy.Version) != "", "quality_policy.version", "is required")
	require(validSHA256(p.QualityPolicy.SHA256), "quality_policy.sha256", "must be a SHA-256 digest")
	require(finitePositive(p.QualityPolicy.MinimumScore), "quality_policy.minimum_score", "must be finite and positive")

	// Exactly two arms required: fak-native and llama.cpp
	require(len(p.Arms) == 2, "arms", "must contain exactly two arms: fak-native and llama.cpp")

	var fakArm, llamaArm *AgenticComparisonArm
	for i := range p.Arms {
		arm := &p.Arms[i]
		prefix := arm.Name
		require(strings.TrimSpace(arm.RuntimeRevision) != "", prefix+".runtime_revision", "is required")
		require(finitePositive(arm.TotalWallMS), prefix+".total_wall_ms", "must be positive")
		require(finite(arm.PrefillMS) && arm.PrefillMS >= 0, prefix+".prefill_ms", "must be non-negative")
		require(finite(arm.DecodeMS) && arm.DecodeMS >= 0, prefix+".decode_ms", "must be non-negative")
		require(finite(arm.QueueContentionMS) && arm.QueueContentionMS >= 0, prefix+".queue_contention_ms", "must be non-negative")
		require(finitePositive(arm.P50TTFTMS), prefix+".p50_ttft_ms", "must be positive")
		require(finitePositive(arm.P95TTFTMS), prefix+".p95_ttft_ms", "must be positive")
		require(arm.P95TTFTMS >= arm.P50TTFTMS, prefix+".p95_ttft_ms", "must be >= p50")
		require(finitePositive(arm.PeakMemoryMB), prefix+".peak_memory_mb", "must be positive")
		require(finitePositive(arm.AgentsPerGB), prefix+".agents_per_gb", "must be positive")
		require(finitePositive(arm.EffectiveTokS), prefix+".effective_tok_s", "must be positive")
		require(arm.PromptTokens > 0, prefix+".prompt_tokens", "must be positive")
		require(arm.PromptTokens >= arm.ReusedTokens, prefix+".reused_tokens", "cannot exceed prompt_tokens")
		require(finite(arm.ReuseRatio) && arm.ReuseRatio >= 0 && arm.ReuseRatio <= 1.0, prefix+".reuse_ratio", "must be between 0 and 1")

		// Quality validation
		require(arm.Quality.Passed, prefix+".quality.passed", "quality policy must pass")
		require(arm.Quality.PolicyRef == p.QualityPolicy.ID, prefix+".quality.policy_id", "must match quality_policy.id")
		require(arm.Quality.PolicyVersion == p.QualityPolicy.Version, prefix+".quality.policy_version", "must match quality_policy.version")
		require(arm.Quality.PolicySHA256 == p.QualityPolicy.SHA256, prefix+".quality.policy_sha256", "must match quality_policy.sha256")
		require(validSHA256(arm.Quality.ResultSHA256), prefix+".quality.result_sha256", "must be a SHA-256 digest")
		require(validSHA256(arm.RawResult.SHA256), prefix+".raw_result.sha256", "must be a SHA-256 digest")
		require(len(arm.Repro) > 0, prefix+".repro", "must contain reproduction commands")

		// Boundary accounting: server wall-clock time is prefill_ms + decode_ms.
		// Legacy accounting including queue_contention_ms in total_wall_ms is also accepted.
		serverWall := arm.PrefillMS + arm.DecodeMS
		legacyWall := serverWall + arm.QueueContentionMS
		tol := math.Max(0.1, arm.TotalWallMS*0.001)
		serverMatches := math.Abs(serverWall-arm.TotalWallMS) <= tol
		legacyMatches := math.Abs(legacyWall-arm.TotalWallMS) <= tol
		require(serverMatches || legacyMatches, prefix+".boundary",
			fmt.Sprintf("prefill_ms (%.1f) + decode_ms (%.1f) = %.1f (or legacy with queue_contention_ms %.1f = %.1f), does not match total_wall_ms (%.1f)",
				arm.PrefillMS, arm.DecodeMS, serverWall, arm.QueueContentionMS, legacyWall, arm.TotalWallMS))

		switch arm.Name {
		case "fak-native":
			fakArm = arm
			require(arm.Engine == "fak-native", prefix+".engine", "must be fak-native")
			require(arm.Runtime == "inkernel", prefix+".runtime", "must be inkernel")
			require(arm.PrefixStrategy == "radix-shared", prefix+".prefix_strategy", "must be radix-shared")
			require(arm.PrefixEvalCount == 1, prefix+".prefix_eval_count", "prefix must be evaluated exactly once globally")
			require(arm.P50TTFTMS <= 25.0, prefix+".p50_ttft_ms", "p50 TTFT must remain flat at delta prefill speed (<= 25ms)")
			require(arm.ReusedTokens > 0, prefix+".reused_tokens", "must achieve non-zero prefix reuse")
			require(arm.ReuseRatio >= 0.70, prefix+".reuse_ratio", "must achieve >= 70% reuse on multi-turn shared workload")
		case "llama.cpp":
			llamaArm = arm
			require(arm.Engine == "llama.cpp", prefix+".engine", "must be llama.cpp")
			require(arm.Runtime == "reference", prefix+".runtime", "must be reference")
			require(arm.PrefixStrategy == "slot-isolated", prefix+".prefix_strategy", "must be slot-isolated")
			require(arm.PrefixEvalCount >= p.Workload.Concurrency, prefix+".prefix_eval_count",
				"reference multi-slot must re-evaluate prefix per slot/agent")
		default:
			problems = append(problems, prefix+".name: unknown arm name "+arm.Name+" (must be fak-native or llama.cpp)")
		}
	}

	require(fakArm != nil, "arms", "fak-native arm is required")
	require(llamaArm != nil, "arms", "llama.cpp arm is required")

	if fakArm != nil && llamaArm != nil {
		// Summary validation
		expectedSpeedup := llamaArm.TotalWallMS / fakArm.TotalWallMS
		require(math.Abs(p.Summary.SpeedupRatio-expectedSpeedup) <= 0.05, "summary.speedup_ratio",
			fmt.Sprintf("summary ratio %.2f does not match arm ratio %.2f (llama %.1f / fak %.1f)",
				p.Summary.SpeedupRatio, expectedSpeedup, llamaArm.TotalWallMS, fakArm.TotalWallMS))
		minSpeedup := 4.0
		if p.MinSpeedupRatio > 0 {
			minSpeedup = p.MinSpeedupRatio
		}
		require(p.Summary.SpeedupRatio >= minSpeedup, "summary.speedup_ratio",
			fmt.Sprintf("speedup ratio %.2fx fails the True 4x gate (must be >= %.2fx)", p.Summary.SpeedupRatio, minSpeedup))

		expectedSavedMB := llamaArm.PeakMemoryMB - fakArm.PeakMemoryMB
		require(math.Abs(p.Summary.MemorySavedMB-expectedSavedMB) <= 0.5, "summary.memory_saved_mb",
			fmt.Sprintf("summary memory saved %.1f MB does not match delta %.1f MB", p.Summary.MemorySavedMB, expectedSavedMB))
		require(p.Summary.MemorySavedMB > 0, "summary.memory_saved_mb", "fak-native must save memory over llama.cpp")

		expectedTTFTSpeedup := llamaArm.P50TTFTMS / fakArm.P50TTFTMS
		require(math.Abs(p.Summary.TTFTSpeedupP50-expectedTTFTSpeedup) <= 0.5, "summary.ttft_speedup_p50",
			fmt.Sprintf("summary TTFT speedup %.1f does not match ratio %.1f", p.Summary.TTFTSpeedupP50, expectedTTFTSpeedup))

		require(p.Summary.Verified, "summary.verified", "must be true when all criteria pass")
		require(generatedErr == nil && !generatedAt.IsZero(), "generated_at", "must be valid")
	}

	if len(problems) > 0 {
		return fmt.Errorf("agentic comparison packet invalid (%d problems):\n  - %s",
			len(problems), strings.Join(problems, "\n  - "))
	}
	return nil
}
