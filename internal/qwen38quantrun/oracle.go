package qwen38quantrun

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"slices"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/quality"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

const (
	OracleConfigSchema        = "fak.qwen38-llamacpp-oracle-config/1"
	OracleMeasurementSchema   = "fak.qwen38-llamacpp-measurement/1"
	OracleMeasurementV2Schema = "fak.qwen38-llamacpp-measurement/2"
	OracleReportSchema        = "fak.qwen38-llamacpp-oracle/1"
	OracleArchiveSchema       = "fak.qwen38-llamacpp-oracle-raw/1"
	PinnedLlamaCPPBuild       = "9828"
	PinnedLlamaCPPRevision    = "ebd048fc5e4b43ec4e0b4abe0b9bf66e1724dad0"
	PinnedLlamaCPPLicense     = "MIT"
)

const (
	oracleRuntimeReference = "reference"
	oracleRuntimeCandidate = "candidate"
	oracleNativeEngine     = "inkernel"
	oracleMetalBackend     = "metal"
	oracleNativeForward    = "metal/qwen35-hybrid-session-v1"
	oracleLlamaForward     = "llama.cpp/metal"
)

// OracleConfig composes two maintained AdapterConfig runs. Runtime lifecycle
// remains owned by those adapters; this layer only binds and compares evidence.
type OracleConfig struct {
	Schema            string              `json:"schema"`
	LlamaCPPRevision  string              `json:"llama_cpp_revision"`
	LlamaCPPLicense   string              `json:"llama_cpp_license"`
	RevisionCommand   []string            `json:"revision_command"`
	BuildFlags        []string            `json:"build_flags"`
	LogitTolerance    float64             `json:"logit_tolerance"`
	Reference         OracleRuntimeConfig `json:"reference"`
	Candidate         OracleRuntimeConfig `json:"candidate"`
	StaleAfter        string              `json:"stale_after"`
	RollbackThreshold string              `json:"rollback_threshold"`
}

type OracleRuntimeConfig struct {
	Name               string   `json:"name"`
	AdapterConfig      string   `json:"adapter_config"`
	CampaignReport     string   `json:"campaign_report"`
	CampaignArchive    string   `json:"campaign_archive"`
	MeasurementCommand []string `json:"measurement_command"`
}

type OracleMeasurementRun struct {
	Schema         string                   `json:"schema"`
	Runtime        string                   `json:"runtime"`
	CorpusID       string                   `json:"corpus_id"`
	CorpusSHA256   string                   `json:"corpus_sha256"`
	ArtifactSHA256 string                   `json:"artifact_sha256"`
	Hardware       string                   `json:"hardware"`
	Software       string                   `json:"software"`
	Command        []string                 `json:"command"`
	Samples        []OracleMeasurement      `json:"samples"`
	Execution      *OracleExecutionIdentity `json:"execution,omitempty"`
	Matched        *OracleMatchedEnvelope   `json:"matched,omitempty"`
}

// OracleExecutionIdentity names the runtime that performed model math. It is
// separate from transport and endpoint identity so a comparator cannot be
// mistaken for a fak-native recovery arm.
type OracleExecutionIdentity struct {
	Engine         string `json:"engine"`
	Backend        string `json:"backend"`
	ForwardPath    string `json:"forward_path"`
	Q4K            bool   `json:"q4k"`
	FallbackActive *bool  `json:"fallback_active"`
	ComparatorOnly *bool  `json:"comparator_only"`
}

// OracleMatchedEnvelope carries the fixed-shape performance receipt alongside
// the frozen-corpus samples. Keeping the two separate preserves the existing
// quality corpus while making the exact P32/T64 repetitions explicit.
type OracleMatchedEnvelope struct {
	PromptSHA256  string                    `json:"prompt_sha256"`
	Temperature   float64                   `json:"temperature"`
	PrefillTokens int                       `json:"prefill_tokens"`
	DecodeTokens  int                       `json:"decode_tokens"`
	Repetitions   []OracleMatchedRepetition `json:"repetitions"`
}

type OracleMatchedRepetition struct {
	Repetition             int       `json:"repetition"`
	Tokens                 []string  `json:"tokens"`
	Logits                 []float64 `json:"logits"`
	TTFTMS                 float64   `json:"ttft_ms"`
	PrefillSeconds         float64   `json:"prefill_seconds"`
	PrefillTokensPerSecond float64   `json:"prefill_tokens_per_second"`
	DecodeSeconds          float64   `json:"decode_seconds"`
	DecodeTokensPerSecond  float64   `json:"decode_tokens_per_second"`
	RSSBytes               uint64    `json:"rss_bytes"`
	OSFootprintBytes       uint64    `json:"os_footprint_bytes"`
}

type OracleMeasurement struct {
	FixtureID              string    `json:"fixture_id"`
	CacheState             string    `json:"cache_state"`
	PromptSHA256           string    `json:"prompt_sha256"`
	Tokens                 []string  `json:"tokens"`
	Logits                 []float64 `json:"logits"`
	TTFTMS                 float64   `json:"ttft_ms"`
	PrefillTokens          int       `json:"prefill_tokens"`
	PrefillSeconds         float64   `json:"prefill_seconds"`
	PrefillTokensPerSecond float64   `json:"prefill_tokens_per_second"`
	DecodeTokens           int       `json:"decode_tokens"`
	DecodeSeconds          float64   `json:"decode_seconds"`
	DecodeTokensPerSecond  float64   `json:"decode_tokens_per_second"`
	RSSBytes               uint64    `json:"rss_bytes"`
	VRAMBytes              uint64    `json:"vram_bytes"`
}

type OracleReport struct {
	Schema            string                 `json:"schema"`
	CorpusID          string                 `json:"corpus_id"`
	CorpusSHA256      string                 `json:"corpus_sha256"`
	ArtifactSHA256    string                 `json:"artifact_sha256"`
	PromptSHA256      map[string]string      `json:"prompt_sha256"`
	LlamaCPPRevision  string                 `json:"llama_cpp_revision"`
	LlamaCPPLicense   string                 `json:"llama_cpp_license"`
	BuildFlags        []string               `json:"build_flags"`
	LogitTolerance    float64                `json:"logit_tolerance"`
	Reference         OracleRuntimeReport    `json:"reference"`
	Candidate         OracleRuntimeReport    `json:"candidate"`
	NumericQuality    []OracleNumericSummary `json:"numeric_quality"`
	Performance       []OraclePerformance    `json:"performance"`
	Verdict           string                 `json:"verdict"`
	RawArchiveSHA256  string                 `json:"raw_archive_sha256"`
	StaleAfter        string                 `json:"stale_after"`
	RollbackThreshold string                 `json:"rollback_threshold"`
}

type OracleRuntimeReport struct {
	Name                string             `json:"name"`
	AdapterConfigSHA256 string             `json:"adapter_config_sha256"`
	Campaign            qwen38quant.Report `json:"campaign"`
	Hardware            string             `json:"hardware"`
	Software            string             `json:"software"`
	MeasurementCommand  []string           `json:"measurement_command"`
}

type OracleNumericSummary struct {
	FixtureID        string  `json:"fixture_id"`
	CacheState       string  `json:"cache_state"`
	Pass             bool    `json:"pass"`
	TokensEqual      bool    `json:"tokens_equal"`
	MaxAbsLogitDiff  float64 `json:"max_abs_logit_diff"`
	CosineSimilarity float64 `json:"cosine_similarity"`
}

type OraclePerformance struct {
	Runtime                    string   `json:"runtime"`
	CacheState                 string   `json:"cache_state"`
	Samples                    int      `json:"samples"`
	TTFTP50MS                  float64  `json:"ttft_p50_ms"`
	PrefillTokensPerSecondMean float64  `json:"prefill_tokens_per_second_mean"`
	DecodeTokensPerSecondMean  float64  `json:"decode_tokens_per_second_mean"`
	PeakRSSBytes               uint64   `json:"peak_rss_bytes"`
	PeakVRAMBytes              uint64   `json:"peak_vram_bytes"`
	Command                    []string `json:"command"`
}

type OracleArchive struct {
	Schema            string               `json:"schema"`
	LlamaCPPRevision  string               `json:"llama_cpp_revision"`
	LlamaCPPLicense   string               `json:"llama_cpp_license"`
	RevisionCommand   []string             `json:"revision_command"`
	BuildFlags        []string             `json:"build_flags"`
	LogitTolerance    float64              `json:"logit_tolerance"`
	Reference         OracleRuntimeArchive `json:"reference"`
	Candidate         OracleRuntimeArchive `json:"candidate"`
	Quality           []quality.Result     `json:"quality"`
	StaleAfter        string               `json:"stale_after"`
	RollbackThreshold string               `json:"rollback_threshold"`
}

type OracleRuntimeArchive struct {
	Name                string               `json:"name"`
	AdapterConfigSHA256 string               `json:"adapter_config_sha256"`
	Campaign            qwen38quant.Report   `json:"campaign"`
	CampaignArchive     json.RawMessage      `json:"campaign_archive"`
	Measurement         OracleMeasurementRun `json:"measurement"`
}

// RunOracle validates the independently captured campaign and measurement
// evidence, applies the numeric gate, and atomically emits a hash-bound pair.
func RunOracle(ctx context.Context, configPath, corpusPath, reportPath, archivePath string) (OracleReport, error) {
	var cfg OracleConfig
	if err := decodeFile(configPath, &cfg); err != nil {
		return OracleReport{}, fmt.Errorf("config: %w", err)
	}
	if err := validateOracleConfig(cfg); err != nil {
		return OracleReport{}, fmt.Errorf("config: %w", err)
	}
	corpusBytes, err := os.ReadFile(corpusPath)
	if err != nil {
		return OracleReport{}, fmt.Errorf("corpus: %w", err)
	}
	corpus, err := qwen38quant.DecodeCorpus(corpusBytes)
	if err != nil {
		return OracleReport{}, fmt.Errorf("corpus: %w", err)
	}
	revision, err := runArgv(ctx, cfg.RevisionCommand)
	if err != nil {
		return OracleReport{}, fmt.Errorf("llama.cpp revision: %w", err)
	}
	if !matchesPinnedLlamaCPPRevision(revision) {
		return OracleReport{}, fmt.Errorf("llama.cpp revision drift: got %q want build %s at %s", strings.TrimSpace(string(revision)), PinnedLlamaCPPBuild, PinnedLlamaCPPRevision)
	}
	reference, err := loadOracleRuntime(ctx, cfg.Reference, corpus, oracleRuntimeReference)
	if err != nil {
		return OracleReport{}, fmt.Errorf("reference: %w", err)
	}
	candidate, err := loadOracleRuntime(ctx, cfg.Candidate, corpus, oracleRuntimeCandidate)
	if err != nil {
		return OracleReport{}, fmt.Errorf("candidate: %w", err)
	}
	archive := OracleArchive{
		Schema: OracleArchiveSchema, LlamaCPPRevision: cfg.LlamaCPPRevision, LlamaCPPLicense: cfg.LlamaCPPLicense,
		RevisionCommand: slices.Clone(cfg.RevisionCommand), BuildFlags: slices.Clone(cfg.BuildFlags), LogitTolerance: cfg.LogitTolerance,
		Reference: reference, Candidate: candidate, StaleAfter: cfg.StaleAfter, RollbackThreshold: cfg.RollbackThreshold,
	}
	archive.Quality, err = compareOracleRuns(reference.Measurement, candidate.Measurement, cfg.LogitTolerance)
	if err != nil {
		return OracleReport{}, err
	}
	report, err := oracleReportFromArchive(archive, corpus)
	if err != nil {
		return OracleReport{}, err
	}
	raw, err := canonicalJSON(archive)
	if err != nil {
		return OracleReport{}, err
	}
	sum := sha256.Sum256(raw)
	report.RawArchiveSHA256 = hex.EncodeToString(sum[:])
	if err := ValidateOracleArtifacts(report, raw, corpus); err != nil {
		return OracleReport{}, fmt.Errorf("validate oracle artifacts: %w", err)
	}
	reportJSON, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return OracleReport{}, err
	}
	reportJSON = append(reportJSON, '\n')
	if err := writeAtomic(archivePath, raw, 0o600); err != nil {
		return OracleReport{}, fmt.Errorf("archive: %w", err)
	}
	if err := writeAtomic(reportPath, reportJSON, 0o644); err != nil {
		_ = os.Remove(archivePath)
		return OracleReport{}, fmt.Errorf("report: %w", err)
	}
	return report, nil
}

// matchesPinnedLlamaCPPRevision binds llama-server's build-number/short-SHA
// identity to the immutable full revision stored in every oracle artifact. The
// alternate full-SHA form supports revision commands such as git rev-parse HEAD.
func matchesPinnedLlamaCPPRevision(raw []byte) bool {
	output := strings.TrimSpace(strings.ReplaceAll(string(raw), "\r\n", "\n"))
	if output == PinnedLlamaCPPRevision {
		return true
	}
	lines := strings.Split(output, "\n")
	versionLine := fmt.Sprintf("version: %s (%s)", PinnedLlamaCPPBuild, PinnedLlamaCPPRevision[:9])
	return len(lines) == 2 && lines[0] == versionLine && strings.HasPrefix(lines[1], "built with ") && len(strings.TrimSpace(strings.TrimPrefix(lines[1], "built with "))) > 0
}

func validateOracleConfig(cfg OracleConfig) error {
	if cfg.Schema != OracleConfigSchema {
		return fmt.Errorf("schema: got %q", cfg.Schema)
	}
	if cfg.LlamaCPPRevision != PinnedLlamaCPPRevision || cfg.LlamaCPPLicense != PinnedLlamaCPPLicense {
		return errors.New("llama.cpp revision and MIT license must match the pinned oracle")
	}
	if len(cfg.RevisionCommand) == 0 || len(cfg.BuildFlags) == 0 || !finitePositive(cfg.LogitTolerance) {
		return errors.New("revision_command, build_flags, and a positive finite logit_tolerance are required")
	}
	if cfg.StaleAfter == "" || cfg.RollbackThreshold == "" {
		return errors.New("stale_after and rollback_threshold are required")
	}
	if err := validateRuntimeConfig("reference", cfg.Reference); err != nil {
		return err
	}
	if err := validateRuntimeConfig("candidate", cfg.Candidate); err != nil {
		return err
	}
	return nil
}

func validateRuntimeConfig(label string, cfg OracleRuntimeConfig) error {
	if cfg.Name == "" || cfg.AdapterConfig == "" || cfg.CampaignReport == "" || cfg.CampaignArchive == "" || len(cfg.MeasurementCommand) == 0 {
		return fmt.Errorf("%s runtime requires name, adapter_config, campaign_report, campaign_archive, and measurement_command", label)
	}
	for i, arg := range cfg.MeasurementCommand {
		lower := strings.ToLower(arg)
		if strings.Contains(lower, "api_key=") || strings.Contains(lower, "api-key=") || strings.Contains(lower, "authorization=") || (i > 0 && (lower == "--api-key" || lower == "--authorization")) {
			return fmt.Errorf("%s measurement_command must use environment-only secrets", label)
		}
	}
	return nil
}

func loadOracleRuntime(ctx context.Context, cfg OracleRuntimeConfig, corpus qwen38quant.Corpus, role string) (OracleRuntimeArchive, error) {
	adapterBytes, err := os.ReadFile(cfg.AdapterConfig)
	if err != nil {
		return OracleRuntimeArchive{}, fmt.Errorf("adapter_config: %w", err)
	}
	var adapter AdapterConfig
	if err := decodeJSONStrict(adapterBytes, &adapter); err != nil {
		return OracleRuntimeArchive{}, fmt.Errorf("adapter_config: %w", err)
	}
	var campaign qwen38quant.Report
	if err := decodeFile(cfg.CampaignReport, &campaign); err != nil {
		return OracleRuntimeArchive{}, fmt.Errorf("campaign_report: %w", err)
	}
	campaignRaw, err := os.ReadFile(cfg.CampaignArchive)
	if err != nil {
		return OracleRuntimeArchive{}, fmt.Errorf("campaign_archive: %w", err)
	}
	if err := validateAdapterBinding(adapter, campaign); err != nil {
		return OracleRuntimeArchive{}, err
	}
	measurementRaw, err := runArgv(ctx, cfg.MeasurementCommand)
	if err != nil {
		return OracleRuntimeArchive{}, fmt.Errorf("measurement_command: %w", err)
	}
	var measurement OracleMeasurementRun
	if err := decodeJSONStrict(measurementRaw, &measurement); err != nil {
		return OracleRuntimeArchive{}, fmt.Errorf("measurement: %w", err)
	}
	measurement.Command = slices.Clone(cfg.MeasurementCommand)
	sum := sha256.Sum256(adapterBytes)
	runtime := OracleRuntimeArchive{
		Name: cfg.Name, AdapterConfigSHA256: hex.EncodeToString(sum[:]), Campaign: campaign,
		CampaignArchive: json.RawMessage(bytes.TrimSpace(campaignRaw)), Measurement: measurement,
	}
	if err := validateRuntimeArchive(runtime, corpus, role); err != nil {
		return OracleRuntimeArchive{}, err
	}
	return runtime, nil
}

func validateAdapterBinding(adapter AdapterConfig, report qwen38quant.Report) error {
	if adapter.Arm != report.Arm || !reflect.DeepEqual(adapter.Expected, report.Identity) || !slices.Equal(adapter.Command, report.Environment.Command) {
		return errors.New("adapter config does not bind the campaign report identity, arm, and command")
	}
	if len(adapter.ObservationCommand) == 0 || len(adapter.RestartCommand) == 0 || len(adapter.ReadyCommand) == 0 || len(adapter.CleanupCommand) == 0 {
		return errors.New("adapter config lacks lifecycle commands")
	}
	return nil
}

func validateRuntimeArchive(runtime OracleRuntimeArchive, corpus qwen38quant.Corpus, role string) error {
	if runtime.Name == "" || !validOracleSHA256(runtime.AdapterConfigSHA256) {
		return errors.New("runtime name and adapter config hash are required")
	}
	if err := qwen38quant.Validate(runtime.Campaign, corpus); err != nil {
		return fmt.Errorf("campaign report: %w", err)
	}
	if err := validateCampaignBinding(runtime.Campaign, runtime.CampaignArchive, corpus); err != nil {
		return fmt.Errorf("campaign archive: %w", err)
	}
	if runtime.Measurement.Runtime != runtime.Name || runtime.Measurement.ArtifactSHA256 != runtime.Campaign.Identity.ArtifactSHA256 {
		return errors.New("measurement runtime or artifact does not match campaign")
	}
	if err := validateMeasurement(runtime.Measurement, corpus); err != nil {
		return err
	}
	if runtime.Measurement.Schema == OracleMeasurementV2Schema {
		return validateExecutionIdentity(runtime.Measurement.Execution, runtime.Campaign.ExecutionEngine, role)
	}
	return nil
}

func validateCampaignBinding(report qwen38quant.Report, raw []byte, corpus qwen38quant.Corpus) error {
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != report.RawArchiveSHA256 {
		return errors.New("raw archive hash mismatch")
	}
	var archive Archive
	if err := decodeJSONStrict(raw, &archive); err != nil {
		return err
	}
	if archive.Schema != "fak.qwen38-quant-raw/1" || archive.CorpusID != corpus.ID || archive.Arm != report.Arm || !archive.CleanupOK {
		return errors.New("campaign archive identity or cleanup mismatch")
	}
	if !reflect.DeepEqual(archive.Before.Identity, report.Identity) || !reflect.DeepEqual(archive.After.Identity, report.Identity) {
		return errors.New("campaign observation identity mismatch")
	}
	trials := make([]qwen38quant.Trial, 0, len(archive.Results))
	for _, result := range archive.Results {
		trials = append(trials, qwen38quant.Trial{Workload: result.Workload, Repetition: result.Repeat, Quality: result.Quality, LatencyMS: result.LatencyMS, Failure: result.Failure, CompletionTokens: result.Usage["completion_tokens"]})
	}
	if !reflect.DeepEqual(trials, report.Trials) {
		return errors.New("campaign results do not bind report trials")
	}
	return nil
}

func validateMeasurement(run OracleMeasurementRun, corpus qwen38quant.Corpus) error {
	if (run.Schema != OracleMeasurementSchema && run.Schema != OracleMeasurementV2Schema) || run.Runtime == "" || run.CorpusID != corpus.ID || run.CorpusSHA256 != qwen38quant.CorpusDigest(corpus) || !validOracleSHA256(run.ArtifactSHA256) {
		return errors.New("measurement identity or corpus mismatch")
	}
	if run.Schema == OracleMeasurementSchema && (run.Execution != nil || run.Matched != nil) {
		return errors.New("v1 measurement cannot carry v2 execution or matched-envelope fields")
	}
	if run.Hardware == "" || run.Software == "" || len(run.Command) == 0 {
		return errors.New("measurement hardware, software, and argv command are required")
	}
	want := make(map[string]string, len(corpus.Fixtures))
	for _, fixture := range corpus.Fixtures {
		sum := sha256.Sum256([]byte(materialize(fixture)))
		want[fixture.ID] = hex.EncodeToString(sum[:])
	}
	seen := map[string]bool{}
	for _, sample := range run.Samples {
		key := sample.FixtureID + "/" + sample.CacheState
		if seen[key] || (sample.CacheState != "cold" && sample.CacheState != "warm") || want[sample.FixtureID] == "" || sample.PromptSHA256 != want[sample.FixtureID] {
			return fmt.Errorf("invalid, duplicate, or drifted measurement %q", key)
		}
		seen[key] = true
		if len(sample.Tokens) == 0 || len(sample.Logits) == 0 || sample.PrefillTokens <= 0 || sample.DecodeTokens <= 0 || sample.RSSBytes == 0 || sample.VRAMBytes == 0 ||
			!finitePositive(sample.TTFTMS) || !finitePositive(sample.PrefillSeconds) || !finitePositive(sample.PrefillTokensPerSecond) || !finitePositive(sample.DecodeSeconds) || !finitePositive(sample.DecodeTokensPerSecond) {
			return fmt.Errorf("measurement %q lacks numeric, timing, RSS, or VRAM evidence", key)
		}
		for _, logit := range sample.Logits {
			if math.IsNaN(logit) || math.IsInf(logit, 0) {
				return fmt.Errorf("measurement %q has non-finite logits", key)
			}
		}
	}
	for fixture := range want {
		for _, state := range []string{"cold", "warm"} {
			if !seen[fixture+"/"+state] {
				return fmt.Errorf("measurement missing %s/%s", fixture, state)
			}
		}
	}
	if run.Schema == OracleMeasurementV2Schema {
		return validateMatchedEnvelope(run.Matched)
	}
	return nil
}

func validateExecutionIdentity(identity *OracleExecutionIdentity, campaignEngine, role string) error {
	if identity == nil || identity.Backend != oracleMetalBackend || !identity.Q4K || identity.FallbackActive == nil || *identity.FallbackActive || identity.ComparatorOnly == nil {
		return fmt.Errorf("%s execution identity is incomplete or unsafe", role)
	}
	switch role {
	case oracleRuntimeReference:
		if campaignEngine != qwen38quant.EngineLlamaCpp || identity.Engine != qwen38quant.EngineLlamaCpp || identity.ForwardPath != oracleLlamaForward || !*identity.ComparatorOnly {
			return errors.New("reference execution identity must be pinned llama.cpp Metal and comparator-only")
		}
	case oracleRuntimeCandidate:
		if campaignEngine != qwen38quant.EngineFakNative || identity.Engine != oracleNativeEngine || identity.ForwardPath != oracleNativeForward || *identity.ComparatorOnly {
			return errors.New("candidate execution identity must be fak-native in-kernel Metal with no comparator role")
		}
	default:
		return fmt.Errorf("unknown oracle runtime role %q", role)
	}
	return nil
}

func validateMatchedEnvelope(matched *OracleMatchedEnvelope) error {
	if matched == nil || !validOracleSHA256(matched.PromptSHA256) || matched.Temperature != 0 || matched.PrefillTokens != 32 || matched.DecodeTokens != 64 || len(matched.Repetitions) != 3 {
		return errors.New("matched envelope requires one prompt hash and exactly three temperature-zero P32/T64 repetitions")
	}
	seen := map[int]bool{}
	var deterministicTokens []string
	for _, repetition := range matched.Repetitions {
		if repetition.Repetition < 1 || repetition.Repetition > 3 || seen[repetition.Repetition] {
			return errors.New("matched envelope has a missing, duplicate, or out-of-range repetition")
		}
		seen[repetition.Repetition] = true
		if len(repetition.Tokens) != matched.DecodeTokens || len(repetition.Logits) == 0 || repetition.RSSBytes == 0 || repetition.OSFootprintBytes == 0 ||
			!finitePositive(repetition.TTFTMS) || !finitePositive(repetition.PrefillSeconds) || !finitePositive(repetition.PrefillTokensPerSecond) || !finitePositive(repetition.DecodeSeconds) || !finitePositive(repetition.DecodeTokensPerSecond) {
			return fmt.Errorf("matched repetition %d lacks generated tokens, timing, RSS, or OS footprint evidence", repetition.Repetition)
		}
		for _, logit := range repetition.Logits {
			if math.IsNaN(logit) || math.IsInf(logit, 0) {
				return fmt.Errorf("matched repetition %d has non-finite logits", repetition.Repetition)
			}
		}
		for _, token := range repetition.Tokens {
			if token == "" {
				return fmt.Errorf("matched repetition %d has an empty generated token", repetition.Repetition)
			}
		}
		if deterministicTokens == nil {
			deterministicTokens = slices.Clone(repetition.Tokens)
		} else if !slices.Equal(deterministicTokens, repetition.Tokens) {
			return errors.New("matched repetitions have nondeterministic generated tokens")
		}
	}
	return nil
}

func validateMatchedPair(reference, candidate OracleMeasurementRun, tolerance float64) error {
	if reference.Schema == OracleMeasurementSchema && candidate.Schema == OracleMeasurementSchema {
		return nil
	}
	if reference.Schema != OracleMeasurementV2Schema || candidate.Schema != OracleMeasurementV2Schema {
		return errors.New("reference and candidate measurement schema versions must match")
	}
	ref, got := reference.Matched, candidate.Matched
	if ref == nil || got == nil || ref.PromptSHA256 != got.PromptSHA256 || ref.Temperature != got.Temperature || ref.PrefillTokens != got.PrefillTokens || ref.DecodeTokens != got.DecodeTokens {
		return errors.New("reference and candidate matched envelopes differ")
	}
	candidates := make(map[int]OracleMatchedRepetition, len(got.Repetitions))
	for _, repetition := range got.Repetitions {
		candidates[repetition.Repetition] = repetition
	}
	for _, want := range ref.Repetitions {
		candidateRepetition, ok := candidates[want.Repetition]
		if !ok || !slices.Equal(want.Tokens, candidateRepetition.Tokens) || maxAbsDiff(want.Logits, candidateRepetition.Logits) > tolerance {
			return fmt.Errorf("matched repetition %d failed generated-token or logit parity", want.Repetition)
		}
	}
	return nil
}

func compareOracleRuns(reference, candidate OracleMeasurementRun, tolerance float64) ([]quality.Result, error) {
	if reference.CorpusID != candidate.CorpusID || reference.CorpusSHA256 != candidate.CorpusSHA256 || reference.ArtifactSHA256 != candidate.ArtifactSHA256 {
		return nil, errors.New("reference and candidate must use the same corpus and artifact")
	}
	candidates := map[string]OracleMeasurement{}
	for _, sample := range candidate.Samples {
		candidates[sample.FixtureID+"/"+sample.CacheState] = sample
	}
	results := make([]quality.Result, 0, len(reference.Samples))
	for _, ref := range reference.Samples {
		key := ref.FixtureID + "/" + ref.CacheState
		eng, ok := candidates[key]
		if !ok || eng.PromptSHA256 != ref.PromptSHA256 {
			return nil, fmt.Errorf("candidate missing prompt-identical sample %s", key)
		}
		caseSpec := quality.QualityCase{
			Schema: quality.CaseSchema, ID: key, Version: 1,
			Prompt:    fmt.Sprintf("Qwen3.8 llama.cpp oracle tol=%g", tolerance),
			Params:    quality.SamplingParams{Temperature: 0, MaxTokens: len(ref.Tokens)},
			Reference: quality.Trace{Runner: reference.Runtime, Tokens: slices.Clone(ref.Tokens), Logits: [][]float64{slices.Clone(ref.Logits)}, Text: strings.Join(ref.Tokens, "")},
			Oracles:   []string{"greedy-token-diff", "logit-parity-tolerance"},
		}
		oracles, err := quality.Lookup(caseSpec.Oracles)
		if err != nil {
			return nil, err
		}
		result, err := quality.RunCase(caseSpec, quality.ReferenceRunner{}, quality.ScriptedRunner{Label: candidate.Runtime, Trace: quality.Trace{Runner: candidate.Runtime, Tokens: slices.Clone(eng.Tokens), Logits: [][]float64{slices.Clone(eng.Logits)}, Text: strings.Join(eng.Tokens, "")}}, oracles)
		if err != nil {
			return nil, err
		}
		results = append(results, result)
	}
	return results, nil
}

func oracleReportFromArchive(archive OracleArchive, corpus qwen38quant.Corpus) (OracleReport, error) {
	if archive.Schema != OracleArchiveSchema || archive.LlamaCPPRevision != PinnedLlamaCPPRevision || archive.LlamaCPPLicense != PinnedLlamaCPPLicense || len(archive.BuildFlags) == 0 || !finitePositive(archive.LogitTolerance) {
		return OracleReport{}, errors.New("invalid oracle archive policy")
	}
	if err := validateRuntimeArchive(archive.Reference, corpus, oracleRuntimeReference); err != nil {
		return OracleReport{}, fmt.Errorf("reference: %w", err)
	}
	if err := validateRuntimeArchive(archive.Candidate, corpus, oracleRuntimeCandidate); err != nil {
		return OracleReport{}, fmt.Errorf("candidate: %w", err)
	}
	wantQuality, err := compareOracleRuns(archive.Reference.Measurement, archive.Candidate.Measurement, archive.LogitTolerance)
	if err != nil {
		return OracleReport{}, err
	}
	if !reflect.DeepEqual(wantQuality, archive.Quality) {
		return OracleReport{}, errors.New("quality results do not match raw measurements")
	}
	if err := validateMatchedPair(archive.Reference.Measurement, archive.Candidate.Measurement, archive.LogitTolerance); err != nil {
		return OracleReport{}, err
	}
	promote := archive.Reference.Campaign.Verdict == "PROMOTE" && archive.Candidate.Campaign.Verdict == "PROMOTE"
	numeric := make([]OracleNumericSummary, 0, len(archive.Quality))
	refSamples := sampleMap(archive.Reference.Measurement.Samples)
	candSamples := sampleMap(archive.Candidate.Measurement.Samples)
	for _, result := range archive.Quality {
		ref, candidate := refSamples[result.CaseID], candSamples[result.CaseID]
		numeric = append(numeric, OracleNumericSummary{
			FixtureID: ref.FixtureID, CacheState: ref.CacheState, Pass: result.Pass,
			TokensEqual: slices.Equal(ref.Tokens, candidate.Tokens), MaxAbsLogitDiff: maxAbsDiff(ref.Logits, candidate.Logits), CosineSimilarity: cosine(ref.Logits, candidate.Logits),
		})
		promote = promote && result.Pass
	}
	sort.Slice(numeric, func(i, j int) bool {
		if numeric[i].FixtureID == numeric[j].FixtureID {
			return numeric[i].CacheState < numeric[j].CacheState
		}
		return numeric[i].FixtureID < numeric[j].FixtureID
	})
	verdict := "HOLD"
	if promote {
		verdict = "PROMOTE"
	}
	return OracleReport{
		Schema: OracleReportSchema, CorpusID: corpus.ID, CorpusSHA256: qwen38quant.CorpusDigest(corpus), ArtifactSHA256: archive.Reference.Measurement.ArtifactSHA256,
		PromptSHA256: promptHashes(corpus), LlamaCPPRevision: archive.LlamaCPPRevision, LlamaCPPLicense: archive.LlamaCPPLicense, BuildFlags: slices.Clone(archive.BuildFlags), LogitTolerance: archive.LogitTolerance,
		Reference: runtimeReport(archive.Reference), Candidate: runtimeReport(archive.Candidate), NumericQuality: numeric,
		Performance: append(performance(archive.Reference.Measurement), performance(archive.Candidate.Measurement)...), Verdict: verdict,
		StaleAfter: archive.StaleAfter, RollbackThreshold: archive.RollbackThreshold,
	}, nil
}

func runtimeReport(runtime OracleRuntimeArchive) OracleRuntimeReport {
	return OracleRuntimeReport{Name: runtime.Name, AdapterConfigSHA256: runtime.AdapterConfigSHA256, Campaign: runtime.Campaign, Hardware: runtime.Measurement.Hardware, Software: runtime.Measurement.Software, MeasurementCommand: slices.Clone(runtime.Measurement.Command)}
}

func performance(run OracleMeasurementRun) []OraclePerformance {
	var out []OraclePerformance
	for _, state := range []string{"cold", "warm"} {
		var ttft []float64
		var prefill, decode float64
		var rss, vram uint64
		for _, sample := range run.Samples {
			if sample.CacheState != state {
				continue
			}
			ttft = append(ttft, sample.TTFTMS)
			prefill += sample.PrefillTokensPerSecond
			decode += sample.DecodeTokensPerSecond
			rss = max(rss, sample.RSSBytes)
			vram = max(vram, sample.VRAMBytes)
		}
		sort.Float64s(ttft)
		n := len(ttft)
		out = append(out, OraclePerformance{Runtime: run.Runtime, CacheState: state, Samples: n, TTFTP50MS: median(ttft), PrefillTokensPerSecondMean: prefill / float64(n), DecodeTokensPerSecondMean: decode / float64(n), PeakRSSBytes: rss, PeakVRAMBytes: vram, Command: slices.Clone(run.Command)})
	}
	return out
}

func ValidateOracleReport(report OracleReport, corpus qwen38quant.Corpus) error {
	if report.Schema != OracleReportSchema || report.CorpusID != corpus.ID || report.CorpusSHA256 != qwen38quant.CorpusDigest(corpus) || !validOracleSHA256(report.ArtifactSHA256) || !validOracleSHA256(report.RawArchiveSHA256) {
		return errors.New("oracle report identity, corpus, or raw hash is invalid")
	}
	if report.LlamaCPPRevision != PinnedLlamaCPPRevision || report.LlamaCPPLicense != PinnedLlamaCPPLicense || len(report.BuildFlags) == 0 || !finitePositive(report.LogitTolerance) {
		return errors.New("oracle report lacks pinned upstream provenance")
	}
	if report.StaleAfter == "" || report.RollbackThreshold == "" || (report.Verdict != "PROMOTE" && report.Verdict != "HOLD") {
		return errors.New("oracle report lifecycle or verdict is invalid")
	}
	if len(report.NumericQuality) != len(corpus.Fixtures)*2 || len(report.Performance) != 4 {
		return errors.New("oracle report lacks complete cold/warm quality or performance evidence")
	}
	for _, summary := range report.Performance {
		if summary.Samples != len(corpus.Fixtures) || !finitePositive(summary.TTFTP50MS) || !finitePositive(summary.PrefillTokensPerSecondMean) || !finitePositive(summary.DecodeTokensPerSecondMean) || summary.PeakRSSBytes == 0 || summary.PeakVRAMBytes == 0 || len(summary.Command) == 0 {
			return errors.New("oracle report has incomplete performance evidence")
		}
	}
	if report.Verdict == "PROMOTE" {
		if report.Reference.Campaign.Verdict != "PROMOTE" || report.Candidate.Campaign.Verdict != "PROMOTE" {
			return errors.New("campaign quality failure cannot be promoted")
		}
		for _, result := range report.NumericQuality {
			if !result.Pass || !result.TokensEqual || result.MaxAbsLogitDiff > report.LogitTolerance {
				return errors.New("numeric or exact-token failure cannot be promoted")
			}
		}
	}
	return nil
}

func ValidateOracleArtifacts(report OracleReport, raw []byte, corpus qwen38quant.Corpus) error {
	if err := ValidateOracleReport(report, corpus); err != nil {
		return err
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != report.RawArchiveSHA256 {
		return errors.New("oracle raw archive hash mismatch")
	}
	var archive OracleArchive
	if err := decodeJSONStrict(raw, &archive); err != nil {
		return err
	}
	want, err := oracleReportFromArchive(archive, corpus)
	if err != nil {
		return err
	}
	want.RawArchiveSHA256 = report.RawArchiveSHA256
	if !reflect.DeepEqual(want, report) {
		return errors.New("oracle report does not match raw archive")
	}
	return nil
}

func decodeJSONStrict(raw []byte, dst any) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if err := d.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		if err == nil {
			return errors.New("multiple JSON values")
		}
		return err
	}
	return nil
}

func promptHashes(corpus qwen38quant.Corpus) map[string]string {
	out := make(map[string]string, len(corpus.Fixtures))
	for _, fixture := range corpus.Fixtures {
		sum := sha256.Sum256([]byte(materialize(fixture)))
		out[fixture.ID] = hex.EncodeToString(sum[:])
	}
	return out
}

func sampleMap(samples []OracleMeasurement) map[string]OracleMeasurement {
	out := make(map[string]OracleMeasurement, len(samples))
	for _, sample := range samples {
		out[sample.FixtureID+"/"+sample.CacheState] = sample
	}
	return out
}

func maxAbsDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var out float64
	for i := range a {
		out = max(out, math.Abs(a[i]-b[i]))
	}
	return out
}

func cosine(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, aa, bb float64
	for i := range a {
		dot += a[i] * b[i]
		aa += a[i] * a[i]
		bb += b[i] * b[i]
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / math.Sqrt(aa*bb)
}

func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	mid := len(values) / 2
	if len(values)%2 == 1 {
		return values[mid]
	}
	return (values[mid-1] + values[mid]) / 2
}

func finitePositive(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validOracleSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
