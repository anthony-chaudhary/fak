package macbench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"
)

const Schema = "fak.macbench.result.v1"

const (
	ComparisonSchema                = "fak.macbench.comparison.v1"
	ComparisonRawSamplesSchema      = "fak.macbench.comparison.raw-samples.v1"
	ComparisonQualityEvidenceSchema = "fak.macbench.comparison.quality.v1"
	MinimumComparisonSamples        = 20
)

type ComparisonPacket struct {
	Schema        string                  `json:"schema"`
	GeneratedAt   string                  `json:"generated_at"`
	CampaignID    string                  `json:"campaign_id"`
	HostID        string                  `json:"host_id"`
	Model         ComparisonModel         `json:"model"`
	Hardware      ComparisonHardware      `json:"hardware"`
	OS            ComparisonOS            `json:"os"`
	PromptSet     ComparisonPromptSet     `json:"prompt_set"`
	ContextTokens int                     `json:"context_tokens"`
	OutputTokens  int                     `json:"output_tokens"`
	QualityPolicy ComparisonQualityPolicy `json:"quality_policy"`
	Arms          []ComparisonArm         `json:"arms"`
}

type ComparisonModel struct {
	Family                 string `json:"family"`
	ID                     string `json:"id"`
	SourceRevision         string `json:"source_revision"`
	CanonicalWeightsSHA256 string `json:"canonical_weights_sha256"`
	Quant                  string `json:"quant"`
}

type ComparisonHardware struct {
	Model       string `json:"model"`
	Chip        string `json:"chip"`
	MemoryBytes uint64 `json:"memory_bytes"`
}

type ComparisonOS struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Build   string `json:"build"`
}

type ComparisonPromptSet struct {
	ID      string             `json:"id"`
	SHA256  string             `json:"sha256"`
	Prompts []ComparisonPrompt `json:"prompts"`
}

type ComparisonPrompt struct {
	ID     string `json:"id"`
	SHA256 string `json:"sha256"`
}

type ComparisonQualityPolicy struct {
	ID           string  `json:"id"`
	Version      string  `json:"version"`
	SHA256       string  `json:"sha256"`
	MinimumScore float64 `json:"minimum_score"`
}

type ComparisonArm struct {
	Name            string                  `json:"name"`
	EvidenceKind    string                  `json:"evidence_kind"`
	RunID           string                  `json:"run_id"`
	StartedAt       string                  `json:"started_at"`
	FinishedAt      string                  `json:"finished_at"`
	HostID          string                  `json:"host_id"`
	Engine          string                  `json:"engine"`
	Runtime         string                  `json:"runtime"`
	RuntimeRevision string                  `json:"runtime_revision"`
	Fallback        string                  `json:"fallback"`
	FallbackCount   int                     `json:"fallback_count"`
	ModelID         string                  `json:"model_id"`
	Artifact        ComparisonArtifact      `json:"artifact"`
	Hardware        ComparisonHardware      `json:"hardware"`
	OS              ComparisonOS            `json:"os"`
	PromptSetSHA256 string                  `json:"prompt_set_sha256"`
	ContextTokens   int                     `json:"context_tokens"`
	OutputTokens    int                     `json:"output_tokens"`
	Quality         ComparisonQualityResult `json:"quality"`
	Metrics         ComparisonMetrics       `json:"metrics"`
	Samples         []ComparisonSample      `json:"samples"`
	RawResult       ComparisonRawResult     `json:"raw_result"`
	Repro           []string                `json:"repro"`
}

type ComparisonArtifact struct {
	Identity               string `json:"identity"`
	SHA256                 string `json:"sha256"`
	Format                 string `json:"format"`
	SourceRevision         string `json:"source_revision"`
	CanonicalWeightsSHA256 string `json:"canonical_weights_sha256"`
	Quant                  string `json:"quant"`
}

type ComparisonQualityResult struct {
	PolicyRef     string  `json:"policy_id"`
	PolicyVersion string  `json:"policy_version"`
	PolicySHA256  string  `json:"policy_sha256"`
	Passed        bool    `json:"passed"`
	Score         float64 `json:"score"`
	ResultPath    string  `json:"result_path"`
	ResultSHA256  string  `json:"result_sha256"`
}

type ComparisonMetrics struct {
	Prefill ComparisonPrefillMetrics `json:"prefill"`
	Decode  ComparisonDecodeMetrics  `json:"decode"`
}

type ComparisonPrefillMetrics struct {
	TTFTMS  ComparisonDistribution `json:"ttft_ms"`
	TokPerS ComparisonDistribution `json:"throughput_tok_s"`
}

type ComparisonDecodeMetrics struct {
	ITLMS   ComparisonDistribution `json:"itl_ms"`
	TokPerS ComparisonDistribution `json:"throughput_tok_s"`
}

type ComparisonDistribution struct {
	P50 float64 `json:"p50"`
	P95 float64 `json:"p95"`
}

type ComparisonSample struct {
	ID              string                    `json:"id"`
	PromptID        string                    `json:"prompt_id"`
	PromptSHA256    string                    `json:"prompt_sha256"`
	Ordinal         int                       `json:"ordinal"`
	InputTokens     int                       `json:"input_tokens"`
	OutputTokens    int                       `json:"output_tokens"`
	Engine          string                    `json:"engine"`
	Runtime         string                    `json:"runtime"`
	RuntimeRevision string                    `json:"runtime_revision"`
	Fallback        string                    `json:"fallback"`
	FallbackCount   int                       `json:"fallback_count"`
	ArtifactSHA256  string                    `json:"artifact_sha256"`
	TTFTMS          float64                   `json:"ttft_ms"`
	ITLMS           float64                   `json:"itl_ms"`
	PrefillTokPerS  float64                   `json:"prefill_tok_s"`
	DecodeTokPerS   float64                   `json:"decode_tok_s"`
	Boundary        ComparisonRequestBoundary `json:"request_boundary"`
}

type ComparisonRequestBoundary struct {
	TotalMS        float64 `json:"total_ms"`
	QueueMS        float64 `json:"queue_ms"`
	SetupMS        float64 `json:"setup_ms"`
	PrefillMS      float64 `json:"prefill_ms"`
	DecodeMS       float64 `json:"decode_ms"`
	VerificationMS float64 `json:"verification_ms"`
	RecoveryMS     float64 `json:"recovery_ms"`
	OtherMS        float64 `json:"other_ms"`
}

type ComparisonRawResult struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type ComparisonRawSamplesFile struct {
	Schema     string             `json:"schema"`
	Arm        string             `json:"arm"`
	CampaignID string             `json:"campaign_id"`
	RunID      string             `json:"run_id"`
	HostID     string             `json:"host_id"`
	StartedAt  string             `json:"started_at"`
	FinishedAt string             `json:"finished_at"`
	Samples    []ComparisonSample `json:"samples"`
}

type ComparisonQualityEvidenceFile struct {
	Schema          string  `json:"schema"`
	Arm             string  `json:"arm"`
	RunID           string  `json:"run_id"`
	PolicyRef       string  `json:"policy_id"`
	PolicyVersion   string  `json:"policy_version"`
	PolicySHA256    string  `json:"policy_sha256"`
	Passed          bool    `json:"passed"`
	Score           float64 `json:"score"`
	ArtifactSHA256  string  `json:"artifact_sha256"`
	PromptSetSHA256 string  `json:"prompt_set_sha256"`
}

// ValidateComparisonPacket is the publication gate for a three-way Mac result.
// It accepts exact, matched Qwen3.8 evidence only; llama.cpp and MLX are named
// reference arms, while the product arm must prove in-kernel fak-native execution.
func ValidateComparisonPacket(packet ComparisonPacket) error {
	var problems []string
	require := func(ok bool, field, detail string) {
		if !ok {
			problems = append(problems, field+": "+detail)
		}
	}
	require(packet.Schema == ComparisonSchema, "schema", "must be "+ComparisonSchema)
	generatedAt, generatedErr := time.Parse(time.RFC3339, packet.GeneratedAt)
	require(generatedErr == nil, "generated_at", "must be RFC3339")
	require(strings.TrimSpace(packet.CampaignID) != "", "campaign_id", "is required")
	require(validSHA256(packet.HostID), "host_id", "must be a SHA-256 host identity")
	require(strings.EqualFold(strings.TrimSpace(packet.Model.Family), "Qwen3.8"), "model.family", "must be exactly Qwen3.8")
	modelID := strings.ToLower(strings.TrimSpace(packet.Model.ID))
	require(modelID == "qwen3.8" || strings.HasPrefix(modelID, "qwen3.8-"), "model.id", "must identify Qwen3.8")
	require(strings.TrimSpace(packet.Model.SourceRevision) != "", "model.source_revision", "is required")
	require(validSHA256(packet.Model.CanonicalWeightsSHA256), "model.canonical_weights_sha256", "must be a SHA-256 digest")
	require(strings.TrimSpace(packet.Model.Quant) != "", "model.quant", "is required")
	validateHardware := func(prefix string, hardware ComparisonHardware) {
		require(strings.TrimSpace(hardware.Model) != "", prefix+".model", "is required")
		require(strings.TrimSpace(hardware.Chip) != "", prefix+".chip", "is required")
		require(hardware.MemoryBytes > 0, prefix+".memory_bytes", "must be positive")
	}
	validateOS := func(prefix string, os ComparisonOS) {
		require(strings.TrimSpace(os.Name) != "", prefix+".name", "is required")
		require(strings.TrimSpace(os.Version) != "", prefix+".version", "is required")
		require(strings.TrimSpace(os.Build) != "", prefix+".build", "is required")
	}
	validateHardware("hardware", packet.Hardware)
	validateOS("os", packet.OS)
	require(strings.TrimSpace(packet.PromptSet.ID) != "", "prompt_set.id", "is required")
	require(validSHA256(packet.PromptSet.SHA256), "prompt_set.sha256", "must be a SHA-256 digest")
	require(len(packet.PromptSet.Prompts) > 0, "prompt_set.prompts", "must bind at least one prompt")
	promptIDs := make(map[string]struct{}, len(packet.PromptSet.Prompts))
	for i, prompt := range packet.PromptSet.Prompts {
		prefix := fmt.Sprintf("prompt_set.prompts[%d]", i)
		require(strings.TrimSpace(prompt.ID) != "", prefix+".id", "is required")
		_, duplicate := promptIDs[prompt.ID]
		require(!duplicate, prefix+".id", "must be unique")
		promptIDs[prompt.ID] = struct{}{}
		require(validSHA256(prompt.SHA256), prefix+".sha256", "must be a SHA-256 digest")
	}
	require(packet.ContextTokens > 0, "context_tokens", "must be positive")
	require(packet.OutputTokens > 1, "output_tokens", "must be greater than one so decode ITL is measurable")
	require(strings.TrimSpace(packet.QualityPolicy.ID) != "", "quality_policy.id", "is required")
	require(strings.TrimSpace(packet.QualityPolicy.Version) != "", "quality_policy.version", "is required")
	require(validSHA256(packet.QualityPolicy.SHA256), "quality_policy.sha256", "must be a SHA-256 digest")
	require(finitePositive(packet.QualityPolicy.MinimumScore), "quality_policy.minimum_score", "must be finite and positive")
	require(len(packet.Arms) == 3, "arms", "must contain exactly three arms")

	wantArms := map[string]struct {
		engine  string
		runtime string
	}{
		"fak-native": {engine: "fak-native", runtime: "inkernel"},
		"llama.cpp":  {engine: "llama.cpp", runtime: "reference"},
		"mlx":        {engine: "mlx", runtime: "reference"},
	}
	seenArms := make(map[string]int, len(packet.Arms))
	var matchedSampleKeys []string
	for i, arm := range packet.Arms {
		prefix := fmt.Sprintf("arms[%d]", i)
		seenArms[arm.Name]++
		want, known := wantArms[arm.Name]
		require(known, prefix+".name", "must be fak-native, llama.cpp, or mlx")
		if known {
			require(arm.Engine == want.engine, prefix+".engine", "must be "+want.engine)
			require(arm.Runtime == want.runtime, prefix+".runtime", "must be "+want.runtime)
		}
		require(strings.TrimSpace(arm.RuntimeRevision) != "", prefix+".runtime_revision", "is required")
		require(arm.EvidenceKind == "observed", prefix+".evidence_kind", "must be observed")
		require(strings.TrimSpace(arm.RunID) != "", prefix+".run_id", "is required")
		startedAt, startedErr := time.Parse(time.RFC3339, arm.StartedAt)
		finishedAt, finishedErr := time.Parse(time.RFC3339, arm.FinishedAt)
		require(startedErr == nil, prefix+".started_at", "must be RFC3339")
		require(finishedErr == nil, prefix+".finished_at", "must be RFC3339")
		if startedErr == nil && finishedErr == nil {
			require(finishedAt.After(startedAt), prefix+".finished_at", "must be after started_at")
			if generatedErr == nil {
				require(!finishedAt.After(generatedAt), prefix+".finished_at", "must not be after generated_at")
			}
		}
		require(arm.HostID == packet.HostID, prefix+".host_id", "must match packet host_id")
		require(arm.FallbackCount == 0, prefix+".fallback_count", "must be zero")
		require(arm.Fallback == "none", prefix+".fallback", "must be none")
		require(arm.ModelID == packet.Model.ID, prefix+".model_id", "must match model.id")
		require(strings.TrimSpace(arm.Artifact.Identity) != "", prefix+".artifact.identity", "is required")
		require(validSHA256(arm.Artifact.SHA256), prefix+".artifact.sha256", "must be a SHA-256 digest")
		require(strings.TrimSpace(arm.Artifact.Format) != "", prefix+".artifact.format", "is required")
		require(arm.Artifact.SourceRevision == packet.Model.SourceRevision, prefix+".artifact.source_revision", "must match model.source_revision")
		require(arm.Artifact.CanonicalWeightsSHA256 == packet.Model.CanonicalWeightsSHA256, prefix+".artifact.canonical_weights_sha256", "must match model.canonical_weights_sha256")
		require(arm.Artifact.Quant == packet.Model.Quant, prefix+".artifact.quant", "must match model.quant")
		require(arm.Hardware == packet.Hardware, prefix+".hardware", "must match packet hardware exactly")
		require(arm.OS == packet.OS, prefix+".os", "must match packet OS exactly")
		require(arm.PromptSetSHA256 == packet.PromptSet.SHA256, prefix+".prompt_set_sha256", "must match prompt_set.sha256")
		require(arm.ContextTokens == packet.ContextTokens, prefix+".context_tokens", "must match packet context_tokens")
		require(arm.OutputTokens == packet.OutputTokens, prefix+".output_tokens", "must match packet output_tokens")
		require(arm.Quality.PolicyRef == packet.QualityPolicy.ID, prefix+".quality.policy_id", "must match quality_policy.id")
		require(arm.Quality.PolicyVersion == packet.QualityPolicy.Version, prefix+".quality.policy_version", "must match quality_policy.version")
		require(arm.Quality.PolicySHA256 == packet.QualityPolicy.SHA256, prefix+".quality.policy_sha256", "must match quality_policy.sha256")
		require(arm.Quality.Passed, prefix+".quality.passed", "must be true")
		require(finite(arm.Quality.Score) && arm.Quality.Score >= packet.QualityPolicy.MinimumScore, prefix+".quality.score", "must meet quality_policy.minimum_score")
		require(strings.TrimSpace(arm.Quality.ResultPath) != "", prefix+".quality.result_path", "is required")
		require(validSHA256(arm.Quality.ResultSHA256), prefix+".quality.result_sha256", "must be a SHA-256 digest")
		require(len(arm.Samples) >= MinimumComparisonSamples, prefix+".samples", fmt.Sprintf("must contain at least %d raw samples", MinimumComparisonSamples))
		sampleKeys := validateComparisonSamples(prefix, arm, packet, promptIDs, require)
		if matchedSampleKeys == nil {
			matchedSampleKeys = sampleKeys
		} else {
			require(equalStrings(matchedSampleKeys, sampleKeys), prefix+".samples", "must have the same prompt/ordinal keys as every other arm")
		}
		validateComparisonMetrics(prefix, arm.Metrics, arm.Samples, require)
		require(strings.TrimSpace(arm.RawResult.Path) != "", prefix+".raw_result.path", "is required")
		require(validSHA256(arm.RawResult.SHA256), prefix+".raw_result.sha256", "must be a SHA-256 digest")
		require(len(arm.Repro) > 0, prefix+".repro", "must contain a copy-pasteable command")
		for j, arg := range arm.Repro {
			require(strings.TrimSpace(arg) != "", fmt.Sprintf("%s.repro[%d]", prefix, j), "must not be empty")
		}
	}
	for name := range wantArms {
		require(seenArms[name] == 1, "arms", "must contain exactly one "+name+" arm")
	}
	if len(problems) > 0 {
		return fmt.Errorf("comparison packet invalid: %s", strings.Join(problems, "; "))
	}
	return nil
}

func validateComparisonSamples(prefix string, arm ComparisonArm, packet ComparisonPacket, promptIDs map[string]struct{}, require func(bool, string, string)) []string {
	seen := make(map[string]struct{}, len(arm.Samples))
	keys := make([]string, 0, len(arm.Samples))
	promptDigests := make(map[string]string, len(packet.PromptSet.Prompts))
	for _, prompt := range packet.PromptSet.Prompts {
		promptDigests[prompt.ID] = prompt.SHA256
	}
	for i, sample := range arm.Samples {
		field := fmt.Sprintf("%s.samples[%d]", prefix, i)
		key := fmt.Sprintf("%s#%d", sample.PromptID, sample.Ordinal)
		require(sample.ID == key, field+".id", "must equal prompt_id#ordinal")
		_, duplicate := seen[sample.ID]
		require(!duplicate, field+".id", "must be unique")
		seen[sample.ID] = struct{}{}
		keys = append(keys, key)
		_, knownPrompt := promptIDs[sample.PromptID]
		require(knownPrompt, field+".prompt_id", "must name a prompt in prompt_set")
		require(sample.PromptSHA256 == promptDigests[sample.PromptID], field+".prompt_sha256", "must match the named prompt")
		require(sample.Ordinal > 0, field+".ordinal", "must be positive")
		require(sample.InputTokens == packet.ContextTokens, field+".input_tokens", "must match packet context_tokens")
		require(sample.OutputTokens == packet.OutputTokens, field+".output_tokens", "must match packet output_tokens")
		require(sample.Engine == arm.Engine, field+".engine", "must match arm engine")
		require(sample.Runtime == arm.Runtime, field+".runtime", "must match arm runtime")
		require(sample.RuntimeRevision == arm.RuntimeRevision, field+".runtime_revision", "must match arm runtime_revision")
		require(sample.Fallback == "none" && sample.Fallback == arm.Fallback, field+".fallback", "must be none and match arm fallback")
		require(sample.FallbackCount == 0 && sample.FallbackCount == arm.FallbackCount, field+".fallback_count", "must be zero and match arm fallback_count")
		require(sample.ArtifactSHA256 == arm.Artifact.SHA256, field+".artifact_sha256", "must match arm artifact.sha256")
		require(finitePositive(sample.TTFTMS), field+".ttft_ms", "must be finite and positive")
		require(finitePositive(sample.ITLMS), field+".itl_ms", "must be finite and positive")
		require(finitePositive(sample.PrefillTokPerS), field+".prefill_tok_s", "must be finite and positive")
		require(finitePositive(sample.DecodeTokPerS), field+".decode_tok_s", "must be finite and positive")
		boundary := sample.Boundary
		parts := []float64{boundary.QueueMS, boundary.SetupMS, boundary.PrefillMS, boundary.DecodeMS, boundary.VerificationMS, boundary.RecoveryMS, boundary.OtherMS}
		accounted := 0.0
		validParts := true
		for _, part := range parts {
			validParts = validParts && finite(part) && part >= 0
			accounted += part
		}
		tolerance := math.Max(0.01, boundary.TotalMS*0.001)
		require(finitePositive(boundary.TotalMS) && validParts && math.Abs(accounted-boundary.TotalMS) <= tolerance, field+".request_boundary", "components must fully account for total_ms")
		expectedTTFT := boundary.QueueMS + boundary.SetupMS + boundary.PrefillMS
		expectedPrefillRate := float64(sample.InputTokens) * 1000 / boundary.PrefillMS
		expectedITL := boundary.DecodeMS / float64(sample.OutputTokens-1)
		expectedDecodeRate := float64(sample.OutputTokens-1) * 1000 / boundary.DecodeMS
		require(withinRelative(sample.TTFTMS, expectedTTFT, 0.001), field+".ttft_ms", "must equal queue_ms + setup_ms + prefill_ms")
		require(withinRelative(sample.PrefillTokPerS, expectedPrefillRate, 0.001), field+".prefill_tok_s", "must reconcile input_tokens with prefill_ms")
		require(withinRelative(sample.ITLMS, expectedITL, 0.001), field+".itl_ms", "must reconcile output_tokens with decode_ms")
		require(withinRelative(sample.DecodeTokPerS, expectedDecodeRate, 0.001), field+".decode_tok_s", "must reconcile output_tokens with decode_ms")
	}
	sort.Strings(keys)
	return keys
}

func validateComparisonMetrics(prefix string, got ComparisonMetrics, samples []ComparisonSample, require func(bool, string, string)) {
	want := SummarizeComparisonSamples(samples)
	checks := []struct {
		field string
		got   ComparisonDistribution
		want  ComparisonDistribution
	}{
		{field: "prefill.ttft_ms", got: got.Prefill.TTFTMS, want: want.Prefill.TTFTMS},
		{field: "prefill.throughput_tok_s", got: got.Prefill.TokPerS, want: want.Prefill.TokPerS},
		{field: "decode.itl_ms", got: got.Decode.ITLMS, want: want.Decode.ITLMS},
		{field: "decode.throughput_tok_s", got: got.Decode.TokPerS, want: want.Decode.TokPerS},
	}
	for _, check := range checks {
		valid := finitePositive(check.got.P50) && finitePositive(check.got.P95) && check.got.P95 >= check.got.P50
		matched := nearlyEqual(check.got.P50, check.want.P50) && nearlyEqual(check.got.P95, check.want.P95)
		require(valid && matched, prefix+".metrics."+check.field, "must be positive p50/p95 values derived from raw samples")
	}
}

// SummarizeComparisonSamples calculates deterministic nearest-rank p50/p95
// values for the phase-split metrics bound into a comparison packet.
func SummarizeComparisonSamples(samples []ComparisonSample) ComparisonMetrics {
	values := func(read func(ComparisonSample) float64) ComparisonDistribution {
		ordered := make([]float64, 0, len(samples))
		for _, sample := range samples {
			ordered = append(ordered, read(sample))
		}
		sort.Float64s(ordered)
		return ComparisonDistribution{P50: nearestRank(ordered, 0.50), P95: nearestRank(ordered, 0.95)}
	}
	return ComparisonMetrics{
		Prefill: ComparisonPrefillMetrics{
			TTFTMS:  values(func(sample ComparisonSample) float64 { return sample.TTFTMS }),
			TokPerS: values(func(sample ComparisonSample) float64 { return sample.PrefillTokPerS }),
		},
		Decode: ComparisonDecodeMetrics{
			ITLMS:   values(func(sample ComparisonSample) float64 { return sample.ITLMS }),
			TokPerS: values(func(sample ComparisonSample) float64 { return sample.DecodeTokPerS }),
		},
	}
}

func nearestRank(ordered []float64, percentile float64) float64 {
	if len(ordered) == 0 {
		return 0
	}
	index := int(math.Ceil(percentile*float64(len(ordered)))) - 1
	if index < 0 {
		index = 0
	}
	return ordered[index]
}

func validSHA256(raw string) bool {
	if len(raw) != 64 {
		return false
	}
	_, err := hex.DecodeString(raw)
	return err == nil
}

func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finitePositive(value float64) bool {
	return finite(value) && value > 0
}

func nearlyEqual(a, b float64) bool {
	return math.Abs(a-b) <= math.Max(1e-9, math.Max(math.Abs(a), math.Abs(b))*1e-9)
}

func withinRelative(a, b, tolerance float64) bool {
	return finite(a) && finite(b) && math.Abs(a-b) <= math.Max(1e-9, math.Abs(b)*tolerance)
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type Suite string

const (
	SuiteAll          Suite = "all"
	SuiteDecodeLong   Suite = "decode-longgen"
	SuitePrefillSweep Suite = "prefill-sweep"
	SuiteTwoStream    Suite = "2stream"
	SuiteHealth       Suite = "health"
)

type Options struct {
	Gateway       string
	Model         string
	Key           string
	Suite         Suite
	DecodeTokens  []int
	PrefillTokens []int
	Concurrency   int
	HTTPClient    *http.Client
	Now           func() time.Time
}

type Report struct {
	Schema      string   `json:"schema"`
	GeneratedAt string   `json:"generated_at"`
	Suite       Suite    `json:"suite"`
	Gateway     string   `json:"gateway"`
	Model       string   `json:"model"`
	Health      Health   `json:"health"`
	Rows        []Row    `json:"rows,omitempty"`
	Headline    string   `json:"headline,omitempty"`
	Errors      []string `json:"errors,omitempty"`
}

type Health struct {
	OK      bool   `json:"ok"`
	Engine  string `json:"engine,omitempty"`
	Planner string `json:"planner,omitempty"`
	Model   string `json:"model,omitempty"`
	Error   string `json:"error,omitempty"`
}

const RecoverySchema = "fak.macbench.recovery.v1"

type RecoverySignals struct {
	WatcherRunning bool
	ResultPresent  bool
	LatestReport   *Report
	// LogPresent reports whether the watch log the caller named actually
	// exists. nil means no log path was inspected, so presence is unknown.
	// A known-false value is NOT the same as "the watcher has not polled
	// yet": there is no log to poll into, so waiting cannot make progress.
	LogPresent    *bool
	TailnetOnline *bool
	SSHReachable  *bool
	WakeHelper    *bool
}

type RecoveryPlan struct {
	Schema   string           `json:"schema"`
	State    string           `json:"state"`
	Severity string           `json:"severity"`
	Summary  string           `json:"summary"`
	Evidence []string         `json:"evidence,omitempty"`
	Actions  []RecoveryAction `json:"actions"`
}

type RecoveryAction struct {
	ID     string `json:"id"`
	Title  string `json:"title"`
	Detail string `json:"detail"`
}

type Row struct {
	Name                   string  `json:"name"`
	Kind                   string  `json:"kind"`
	Streams                int     `json:"streams,omitempty"`
	PromptRequested        int     `json:"prompt_requested,omitempty"`
	PromptTokens           int     `json:"prompt_tokens,omitempty"`
	MaxTokens              int     `json:"max_tokens,omitempty"`
	CompletionTokens       int     `json:"completion_tokens,omitempty"`
	WallSeconds            float64 `json:"wall_seconds,omitempty"`
	TTFTSeconds            float64 `json:"ttft_seconds,omitempty"`
	TokensPerSecond        float64 `json:"tokens_per_second,omitempty"`
	PrefillTokensPerSecond float64 `json:"prefill_tokens_per_second,omitempty"`
	FinishReason           string  `json:"finish_reason,omitempty"`
	HTTPStatus             int     `json:"http_status,omitempty"`
	Headline               string  `json:"headline,omitempty"`
	Error                  string  `json:"error,omitempty"`
}

func DefaultOptions() Options {
	return Options{
		Gateway:       "http://127.0.0.1:8080",
		Model:         "qwen38:27b",
		Suite:         SuiteAll,
		DecodeTokens:  []int{16, 32, 64, 128, 256, 512},
		PrefillTokens: []int{128, 512, 2048, 4096},
		Concurrency:   2,
	}
}

func PlanRecovery(sig RecoverySignals) RecoveryPlan {
	plan := RecoveryPlan{
		Schema:   RecoverySchema,
		State:    "no_health_report",
		Severity: "info",
		Summary:  "macbench has not observed a gateway health report yet",
		Actions: []RecoveryAction{{
			ID:     "wait-first-poll",
			Title:  "wait for first health poll",
			Detail: "Keep the watcher running until it writes the first sanitized health report.",
		}},
	}
	if sig.ResultPresent {
		plan.State = "result_present"
		plan.Severity = "info"
		plan.Summary = "macbench result artifact is present"
		plan.Actions = []RecoveryAction{{
			ID:     "record-result",
			Title:  "record result",
			Detail: "Fold the result artifact into the nightrun ledger or benchmark summary.",
		}}
		return plan
	}
	if !sig.WatcherRunning {
		plan.State = "watcher_not_running"
		plan.Severity = "action"
		plan.Summary = "macbench watcher is not running"
		plan.Actions = []RecoveryAction{{
			ID:     "restart-watch",
			Title:  "restart macbench watch",
			Detail: "Start `fak macbench watch` with the same sanitized log and result paths.",
		}}
		return plan
	}
	if boolKnownFalse(sig.LogPresent) {
		plan.State = "log_missing"
		plan.Severity = "action"
		plan.Summary = "macbench watch log is absent at the named path; there is no evidence to recover from"
		plan.Actions = []RecoveryAction{
			{
				ID:     "confirm-log-path",
				Title:  "confirm the watch log path",
				Detail: "Check the run id and box directory: the watch log does not exist at the path passed to --log.",
			},
			{
				ID:     "start-fresh-watch",
				Title:  "start a fresh watch run",
				Detail: "Nightrun artifacts are host-local and rotate, so a run whose log has aged out cannot be revived. Start `fak macbench watch` and bind follow-up work to the NEW run id.",
			},
		}
		return plan
	}
	if sig.LatestReport == nil {
		return plan
	}

	rep := *sig.LatestReport
	gateway := sanitizedReportGateway(rep.Gateway)
	plan.Evidence = append(plan.Evidence, fmt.Sprintf("latest suite=%s gateway=%s health=%t", rep.Suite, gateway, rep.Health.OK))
	if rep.Health.Error != "" {
		plan.Evidence = append(plan.Evidence, "health_error="+sanitizeGatewayInText(rep.Health.Error, rep.Gateway))
	}
	if rep.Health.OK {
		plan.State = "gateway_ready"
		plan.Severity = "info"
		plan.Summary = "gateway health is OK; waiting for full benchmark result"
		plan.Actions = []RecoveryAction{{
			ID:     "wait-full-suite",
			Title:  "wait for full suite",
			Detail: "Let the watcher run the full `all` suite and write the result artifact.",
		}}
		return plan
	}

	errText := strings.ToLower(rep.Health.Error + "\n" + strings.Join(rep.Errors, "\n"))
	switch {
	case boolKnownFalse(sig.TailnetOnline):
		plan.State = "tailnet_offline"
		plan.Severity = "operator"
		plan.Summary = "Mac benchmark peer is offline; gateway cannot be recovered from this host"
		plan.Actions = []RecoveryAction{
			{
				ID:     "wake-or-power-mac",
				Title:  "wake or power the Mac",
				Detail: "Use the private lab control path or physical access to bring the Mac back onto the tailnet.",
			},
			{
				ID:     "confirm-tailnet-online",
				Title:  "confirm tailnet peer online",
				Detail: "Re-check the tailnet peer status before restarting the gateway.",
			},
			{
				ID:     "restart-gateway",
				Title:  "restart fak gateway",
				Detail: "Once reachable, start or restart the Mac `fak serve --metal` gateway and keep the watcher running.",
			},
		}
	case boolKnownFalse(sig.SSHReachable):
		plan.State = "control_path_down"
		plan.Severity = "operator"
		plan.Summary = "Mac control path is unreachable; gateway restart cannot be attempted"
		plan.Actions = []RecoveryAction{
			{
				ID:     "restore-control-path",
				Title:  "restore control path",
				Detail: "Bring SSH/tailnet connectivity back before trying to read the gateway key or restart the gateway.",
			},
			{
				ID:     "keep-watch-running",
				Title:  "keep watcher running",
				Detail: "The watcher will write the full result automatically once gateway health succeeds.",
			},
		}
	case strings.Contains(errText, "deadline exceeded") ||
		strings.Contains(errText, "timeout") ||
		strings.Contains(errText, "connection refused") ||
		strings.Contains(errText, "no route to host"):
		plan.State = "gateway_unreachable"
		plan.Severity = "operator"
		plan.Summary = "gateway health probe is timing out or refusing connections"
		plan.Actions = []RecoveryAction{
			{
				ID:     "check-peer-online",
				Title:  "check Mac peer",
				Detail: "Confirm the Mac is awake and visible on the private network.",
			},
			{
				ID:     "restart-gateway",
				Title:  "restart fak gateway",
				Detail: "Restart the Mac gateway, then re-run `fak macbench watch-status` against the existing log/result paths.",
			},
		}
	default:
		plan.State = "waiting_for_gateway"
		plan.Severity = "watch"
		plan.Summary = "gateway health is still false"
		plan.Actions = []RecoveryAction{{
			ID:     "keep-watch-running",
			Title:  "keep watcher running",
			Detail: "Continue polling while investigating the latest sanitized health error.",
		}}
	}
	if boolKnownFalse(sig.WakeHelper) {
		plan.Actions = append(plan.Actions, RecoveryAction{
			ID:     "document-wake-helper-gap",
			Title:  "document wake helper gap",
			Detail: "Track the missing wake/restart helper so the next run has an operator-usable recovery path.",
		})
	}
	return plan
}

func (r Report) HasErrors() bool {
	if len(r.Errors) > 0 {
		return true
	}
	for _, row := range r.Rows {
		if row.Error != "" {
			return true
		}
	}
	return false
}

func Run(ctx context.Context, opts Options) (Report, error) {
	opts = normalizeOptions(opts)
	base, err := NormalizeGateway(opts.Gateway)
	if err != nil {
		return Report{}, err
	}
	opts.Gateway = base
	now := opts.Now()
	rep := Report{
		Schema:      Schema,
		GeneratedAt: now.UTC().Format(time.RFC3339),
		Suite:       opts.Suite,
		Gateway:     SanitizeGatewayForReport(base),
		Model:       opts.Model,
	}
	rep.Health = probeHealth(ctx, opts)
	rep.Health.Error = sanitizeGatewayInText(rep.Health.Error, base)
	if !rep.Health.OK {
		rep.Errors = append(rep.Errors, "healthz failed: "+rep.Health.Error)
		if opts.Suite == SuiteHealth {
			return rep, nil
		}
	}

	switch opts.Suite {
	case SuiteHealth:
	case SuiteAll:
		rep.Rows = append(rep.Rows, runDecodeLong(ctx, opts)...)
		rep.Rows = append(rep.Rows, runPrefillSweep(ctx, opts)...)
		rep.Rows = append(rep.Rows, runTwoStream(ctx, opts)...)
	case SuiteDecodeLong:
		rep.Rows = append(rep.Rows, runDecodeLong(ctx, opts)...)
	case SuitePrefillSweep:
		rep.Rows = append(rep.Rows, runPrefillSweep(ctx, opts)...)
	case SuiteTwoStream:
		rep.Rows = append(rep.Rows, runTwoStream(ctx, opts)...)
	default:
		return Report{}, fmt.Errorf("unknown suite %q", opts.Suite)
	}
	sanitizeReportErrors(&rep, base)
	rep.Headline = headline(rep.Rows)
	return rep, nil
}

func normalizeOptions(opts Options) Options {
	def := DefaultOptions()
	if strings.TrimSpace(opts.Gateway) == "" {
		opts.Gateway = def.Gateway
	}
	if strings.TrimSpace(opts.Model) == "" {
		opts.Model = def.Model
	}
	if opts.Suite == "" {
		opts.Suite = def.Suite
	}
	if len(opts.DecodeTokens) == 0 {
		opts.DecodeTokens = def.DecodeTokens
	}
	if len(opts.PrefillTokens) == 0 {
		opts.PrefillTokens = def.PrefillTokens
	}
	if opts.Concurrency <= 0 {
		opts.Concurrency = def.Concurrency
	}
	if opts.HTTPClient == nil {
		opts.HTTPClient = &http.Client{}
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	return opts
}

func NormalizeGateway(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		raw = "http://127.0.0.1:8080"
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" {
		u.Scheme = "http"
	}
	if u.Host == "" {
		return "", fmt.Errorf("gateway %q has no host", raw)
	}
	u.Path = strings.TrimSuffix(u.Path, "/")
	u.Path = strings.TrimSuffix(u.Path, "/v1")
	u.RawQuery = ""
	u.Fragment = ""
	return strings.TrimSuffix(u.String(), "/"), nil
}

func SanitizeGatewayForReport(raw string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return "<gateway>"
	}
	host := u.Hostname()
	if strings.EqualFold(host, "localhost") {
		return u.Scheme + "://localhost" + portSuffix(u)
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return u.Scheme + "://" + host + portSuffix(u)
	}
	return "<remote-gateway>"
}

func sanitizeReportErrors(rep *Report, rawGateway string) {
	rep.Health.Error = sanitizeGatewayInText(rep.Health.Error, rawGateway)
	for i := range rep.Errors {
		rep.Errors[i] = sanitizeGatewayInText(rep.Errors[i], rawGateway)
	}
	for i := range rep.Rows {
		rep.Rows[i].Error = sanitizeGatewayInText(rep.Rows[i].Error, rawGateway)
	}
}

func sanitizeGatewayInText(s, rawGateway string) string {
	if s == "" {
		return s
	}
	return strings.ReplaceAll(s, rawGateway, SanitizeGatewayForReport(rawGateway))
}

func sanitizedReportGateway(raw string) string {
	raw = strings.TrimSpace(raw)
	if strings.HasPrefix(raw, "<") && strings.HasSuffix(raw, ">") {
		return raw
	}
	return SanitizeGatewayForReport(raw)
}

func boolKnownFalse(v *bool) bool {
	return v != nil && !*v
}

func portSuffix(u *url.URL) string {
	if p := u.Port(); p != "" {
		return ":" + p
	}
	return ""
}

func probeHealth(ctx context.Context, opts Options) Health {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, opts.Gateway+"/healthz", nil)
	if err != nil {
		return Health{Error: err.Error()}
	}
	resp, err := opts.HTTPClient.Do(req)
	if err != nil {
		return Health{Error: err.Error()}
	}
	defer resp.Body.Close()
	var h Health
	if resp.StatusCode/100 != 2 {
		h.Error = fmt.Sprintf("status %d", resp.StatusCode)
		return h
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&h); err != nil {
		return Health{Error: err.Error()}
	}
	return h
}

func runDecodeLong(ctx context.Context, opts Options) []Row {
	rows := make([]Row, 0, len(opts.DecodeTokens))
	for _, maxTok := range opts.DecodeTokens {
		row := runBuffered(ctx, opts, "decode-longgen", fmt.Sprintf("decode-%d", maxTok), 25, maxTok)
		rows = append(rows, row)
	}
	return rows
}

func runPrefillSweep(ctx context.Context, opts Options) []Row {
	rows := make([]Row, 0, len(opts.PrefillTokens))
	for _, promptTok := range opts.PrefillTokens {
		row := runStreamed(ctx, opts, "prefill-sweep", fmt.Sprintf("prefill-%d", promptTok), promptTok, 16)
		rows = append(rows, row)
	}
	return rows
}

func runTwoStream(ctx context.Context, opts Options) []Row {
	streams := opts.Concurrency
	rows := make([]Row, streams)
	var wg sync.WaitGroup
	start := time.Now()
	for i := 0; i < streams; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			rows[i] = runBuffered(ctx, opts, "2stream", fmt.Sprintf("2stream-%d", i+1), 25, 128)
			rows[i].Streams = 1
		}()
	}
	wg.Wait()
	wall := elapsedSeconds(start)
	agg := Row{Name: "2stream-aggregate", Kind: "2stream", Streams: streams, PromptRequested: 25, MaxTokens: 128, WallSeconds: round(wall)}
	for _, row := range rows {
		agg.CompletionTokens += row.CompletionTokens
		if row.Error != "" {
			if agg.Error != "" {
				agg.Error += "; "
			}
			agg.Error += row.Name + ": " + row.Error
		}
	}
	if wall > 0 && agg.CompletionTokens > 0 {
		agg.TokensPerSecond = round(float64(agg.CompletionTokens) / wall)
		agg.Headline = fmt.Sprintf("%.2f tok/s", agg.TokensPerSecond)
	}
	out := []Row{agg}
	out = append(out, rows...)
	return out
}

func runBuffered(ctx context.Context, opts Options, kind, name string, promptTokens, maxTokens int) Row {
	row := Row{Name: name, Kind: kind, PromptRequested: promptTokens, MaxTokens: maxTokens}
	body := chatBody(opts.Model, prompt(promptTokens), maxTokens, false)
	start := time.Now()
	resp, err := doChat(ctx, opts, body)
	wall := elapsedSeconds(start)
	row.WallSeconds = round(wall)
	if err != nil {
		row.Error = err.Error()
		return row
	}
	defer resp.Body.Close()
	row.HTTPStatus = resp.StatusCode
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 32<<20))
	if resp.StatusCode/100 != 2 {
		row.Error = fmt.Sprintf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
		return row
	}
	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		row.Error = err.Error()
		return row
	}
	row.PromptTokens = cr.Usage.PromptTokens
	row.CompletionTokens = cr.Usage.CompletionTokens
	if len(cr.Choices) > 0 {
		row.FinishReason = cr.Choices[0].FinishReason
	}
	if row.CompletionTokens > 0 && wall > 0 {
		row.TokensPerSecond = round(float64(row.CompletionTokens) / wall)
		row.Headline = fmt.Sprintf("%.2f tok/s", row.TokensPerSecond)
	}
	return row
}

func runStreamed(ctx context.Context, opts Options, kind, name string, promptTokens, maxTokens int) Row {
	row := Row{Name: name, Kind: kind, PromptRequested: promptTokens, MaxTokens: maxTokens}
	body := chatBody(opts.Model, prompt(promptTokens), maxTokens, true)
	start := time.Now()
	resp, err := doChat(ctx, opts, body)
	if err != nil {
		row.Error = err.Error()
		return row
	}
	defer resp.Body.Close()
	row.HTTPStatus = resp.StatusCode
	if resp.StatusCode/100 != 2 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		row.Error = fmt.Sprintf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
		return row
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4<<20)
	chunkTokens := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var sr streamResponse
		if err := json.Unmarshal([]byte(payload), &sr); err != nil {
			continue
		}
		if sr.Usage != nil {
			row.PromptTokens = sr.Usage.PromptTokens
			row.CompletionTokens = sr.Usage.CompletionTokens
		}
		for _, choice := range sr.Choices {
			if choice.FinishReason != nil {
				row.FinishReason = *choice.FinishReason
			}
			if choice.Delta.Content != "" {
				chunkTokens++
				if row.TTFTSeconds == 0 {
					row.TTFTSeconds = elapsedSeconds(start)
				}
			}
		}
	}
	row.WallSeconds = elapsedSeconds(start)
	if row.CompletionTokens == 0 {
		row.CompletionTokens = chunkTokens
	}
	if row.PromptTokens == 0 {
		row.PromptTokens = promptTokens
	}
	if err := sc.Err(); err != nil {
		row.Error = err.Error()
		return row
	}
	if row.TTFTSeconds > 0 && row.PromptTokens > 0 {
		row.PrefillTokensPerSecond = round(float64(row.PromptTokens) / row.TTFTSeconds)
		row.Headline = fmt.Sprintf("%.2f tok/s", row.PrefillTokensPerSecond)
	}
	if row.CompletionTokens > 0 && row.WallSeconds > 0 {
		row.TokensPerSecond = round(float64(row.CompletionTokens) / row.WallSeconds)
	}
	return row
}

func doChat(ctx context.Context, opts Options, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, opts.Gateway+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(opts.Key) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(opts.Key))
	}
	return opts.HTTPClient.Do(req)
}

func chatBody(model, prompt string, maxTokens int, stream bool) []byte {
	body := map[string]any{
		"model":       model,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"max_tokens":  maxTokens,
		"temperature": 0,
	}
	if stream {
		body["stream"] = true
	}
	b, _ := json.Marshal(body)
	return b
}

func prompt(tokens int) string {
	if tokens <= 0 {
		tokens = 25
	}
	var b strings.Builder
	b.WriteString("Continue with plain short filler text. Context:")
	for i := 0; i < tokens; i++ {
		fmt.Fprintf(&b, " token%d", i%97)
	}
	b.WriteString("\nAnswer with neutral text only.")
	return b.String()
}

func headline(rows []Row) string {
	for _, row := range rows {
		if row.Headline != "" {
			return row.Headline
		}
	}
	return ""
}

func round(v float64) float64 {
	if v > 0 && v < 0.001 {
		return 0.001
	}
	return float64(int(v*1000+0.5)) / 1000
}

func elapsedSeconds(start time.Time) float64 {
	v := time.Since(start).Seconds()
	if v <= 0 {
		return 0.001
	}
	return round(v)
}

type chatResponse struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

type streamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage,omitempty"`
}
