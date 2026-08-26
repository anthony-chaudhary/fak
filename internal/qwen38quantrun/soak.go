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
	"net/http"
	"os"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

const (
	SoakSchema             = "fak.qwen38-quant-soak/1"
	SoakArchiveSchema      = "fak.qwen38-quant-soak-raw/1"
	DecodeCampaignSchema   = "fak.qwen38-decode-campaign-raw/1"
	MinimumSoakFinalists   = 3
	MinimumCodingTasks     = 30
	defaultCancellationLag = 250 * time.Millisecond
)

var requiredSoakScenarios = []string{
	"context_pressure",
	"malformed_call",
	"cancellation",
	"restart",
	"cache_recovery",
}

// CodingTask is a bounded, exact-effect coding prompt used identically for
// every finalist. Expected output remains outside the model prompt.
type CodingTask struct {
	ID              string `json:"id"`
	Prompt          string `json:"prompt"`
	ExpectedExact   string `json:"expected_exact"`
	MaxOutputTokens int    `json:"max_output_tokens"`
}

type SoakScenario struct {
	Name      string      `json:"name"`
	Outcome   string      `json:"outcome"`
	Failure   string      `json:"failure,omitempty"`
	LatencyMS float64     `json:"latency_ms,omitempty"`
	Readback  Observation `json:"readback"`
}

type SoakMetrics struct {
	CodingLatencyP50MS float64              `json:"coding_latency_p50_ms"`
	CodingThroughput   float64              `json:"coding_output_tokens_per_second"`
	PeakMemoryBytes    uint64               `json:"peak_memory_bytes"`
	PeakPowerWatts     float64              `json:"peak_power_watts"`
	CacheColdMS        float64              `json:"cache_cold_ms"`
	CacheWarmMS        float64              `json:"cache_warm_ms"`
	CacheAfterRestart  float64              `json:"cache_after_restart_ms"`
	CacheSavedMS       float64              `json:"cache_saved_ms"`
	DecodeWindows      *DecodeWindowSummary `json:"decode_windows,omitempty"`
}

type SoakArmResult struct {
	Arm                   string                      `json:"arm"`
	Campaign              qwen38quant.Report          `json:"campaign"`
	CampaignArchiveSHA256 string                      `json:"campaign_archive_sha256"`
	Coding                []Result                    `json:"coding"`
	Scenarios             []SoakScenario              `json:"scenarios"`
	Metrics               SoakMetrics                 `json:"metrics"`
	MatchedDecode         *MatchedDecodeWindowSummary `json:"matched_decode,omitempty"`
	Verdict               string                      `json:"verdict"`
	ArchiveSHA256         string                      `json:"archive_sha256"`
}

type SoakReport struct {
	Schema            string          `json:"schema"`
	CorpusID          string          `json:"corpus_id"`
	CorpusSHA256      string          `json:"corpus_sha256"`
	Arms              []SoakArmResult `json:"arms"`
	SelectedArm       string          `json:"selected_arm,omitempty"`
	Verdict           string          `json:"verdict"`
	RawArchiveSHA256  string          `json:"raw_archive_sha256"`
	RollbackThreshold string          `json:"rollback_threshold"`
}

type SoakArmConfig struct {
	Campaign          CampaignConfig
	CancellationAfter time.Duration
	LongDecode        *LongDecodeCampaignConfig
}

// LongDecodeCampaignConfig adds one measurement-only long-output fixture to a
// soak arm without changing the frozen quality corpus. Comparator results are
// caller-captured b9828 raw events and remain optional for a native-only run.
type LongDecodeCampaignConfig struct {
	Fixture    qwen38quant.Fixture       `json:"fixture"`
	Comparator []LlamaClientDecodeResult `json:"comparator,omitempty"`
}

type DecodeCampaignArchive struct {
	Schema     string                    `json:"schema"`
	FixtureID  string                    `json:"fixture_id"`
	Native     []Result                  `json:"native"`
	Comparator []LlamaClientDecodeResult `json:"comparator,omitempty"`
}

type soakArmArchive struct {
	Schema    string                 `json:"schema"`
	Arm       string                 `json:"arm"`
	Coding    []Result               `json:"coding"`
	Scenarios []SoakScenario         `json:"scenarios"`
	Campaign  json.RawMessage        `json:"campaign"`
	Decode    *DecodeCampaignArchive `json:"decode,omitempty"`
}

type soakArchive struct {
	Schema string            `json:"schema"`
	Arms   []json.RawMessage `json:"arms"`
}

// DefaultSoakTasks is the frozen issue-8319 coding set. The prompts exercise
// small Go semantics while keeping every response bounded and independently
// gradeable as one exact value.
func DefaultSoakTasks() []CodingTask {
	specs := [][3]string{
		{"coding-01", "In Go, a loop adds integers 1 through 10 inclusive. Reply with only the final integer.", "55"},
		{"coding-02", "In Go, fib(0)=0 and fib(1)=1. Reply with only fib(12).", "144"},
		{"coding-03", "In Go, a factorial loop multiplies 1 through 6. Reply with only the result.", "720"},
		{"coding-04", "In Go, reply with only len([]byte(\"gopher\")).", "6"},
		{"coding-05", "In Go, reply with only len([]int{8,5,3,2,1}).", "5"},
		{"coding-06", "In Go, range over []int{2,3,4,5} and sum the values. Reply with only the result.", "14"},
		{"coding-07", "In Go, m := map[string]int{\"x\":2}; m[\"x\"] = 7. Reply with only m[\"x\"].", "7"},
		{"coding-08", "In Go, x starts at 4 and is incremented three times. Reply with only x.", "7"},
		{"coding-09", "Using Go integer division, reply with only 17 / 5.", "3"},
		{"coding-10", "Using Go integer remainder, reply with only 17 % 5.", "2"},
		{"coding-11", "In Go, reply with only the decimal value of 1 << 6.", "64"},
		{"coding-12", "In Go, reply with only the decimal value of 13 & 7.", "5"},
		{"coding-13", "A Go Euclidean gcd function receives 48 and 18. Reply with only the gcd.", "6"},
		{"coding-14", "Go sorts []int{3,1,2} ascending. Reply with only the value at index 1.", "2"},
		{"coding-15", "In Go, strings.Repeat(\"ab\", 3) is assigned to s. Reply with only len(s).", "6"},
		{"coding-16", "Two Go loops run i=0..2 and j=0..3 and increment n each iteration. Reply with only n.", "12"},
		{"coding-17", "A Go loop counts even integers from 1 through 10 inclusive. Reply with only the count.", "5"},
		{"coding-18", "A Go max scan reads []int{-1,9,4,7}. Reply with only the maximum.", "9"},
		{"coding-19", "A Go min scan reads []int{3,-4,8,0}. Reply with only the minimum.", "-4"},
		{"coding-20", "In Go, r := 'A' + 2. Reply with only the character represented by r.", "C"},
		{"coding-21", "In Go, evaluate 3 < 5 && 5 < 8. Reply with only true or false.", "true"},
		{"coding-22", "In Go, reply with only len(map[string]int{\"a\":1, \"b\":2}).", "2"},
		{"coding-23", "In Go, append 4 to []int{1,2,3} and reply with only the new length.", "4"},
		{"coding-24", "A Go loop sums the squares of 1, 2, 3, and 4. Reply with only the sum.", "30"},
		{"coding-25", "A Go loop sums integers 1 through 20 inclusive. Reply with only the sum.", "210"},
		{"coding-26", "A Go loop multiplies x by 2 ten times starting from x=1. Reply with only x.", "1024"},
		{"coding-27", "A Go rune loop reverses \"abcd\". Reply with only the reversed string.", "dcba"},
		{"coding-28", "A Go loop counts vowels in \"agentkernel\". Reply with only the count.", "4"},
		{"coding-29", "In Go, strconv.Atoi(\"314\") succeeds and the result is incremented once. Reply with only the integer.", "315"},
		{"coding-30", "A Go closure captures x=7 and returns x+3. Reply with only the returned integer.", "10"},
	}
	out := make([]CodingTask, 0, len(specs))
	for _, spec := range specs {
		out = append(out, CodingTask{ID: spec[0], Prompt: spec[1], ExpectedExact: spec[2], MaxOutputTokens: 32})
	}
	return out
}

func (r Runner) RunSoakArm(ctx context.Context, cfg SoakArmConfig, corpus qwen38quant.Corpus, tasks []CodingTask) (arm SoakArmResult, archive []byte, err error) {
	if err := validateSoakTasks(tasks); err != nil {
		return arm, nil, err
	}
	if cfg.Campaign.Probe == nil || cfg.Campaign.Lifecycle == nil {
		return arm, nil, errors.New("probe and lifecycle are required")
	}
	if cfg.Campaign.Arm == "" || cfg.Campaign.RollbackThreshold == "" {
		return arm, nil, errors.New("arm and rollback threshold are required")
	}
	if cfg.Campaign.Endpoint.NativeDecodeTrace && cfg.Campaign.ExecutionEngine != qwen38quant.EngineFakNative {
		return arm, nil, errors.New("native decode trace requires explicit fak-native execution; comparator transport timings are not token commits")
	}
	if cfg.LongDecode != nil && cfg.Campaign.ExecutionEngine != qwen38quant.EngineFakNative {
		return arm, nil, errors.New("long decode campaign requires explicit fak-native execution")
	}

	cleanupOwnedByCampaign := false
	defer func() {
		if cleanupOwnedByCampaign {
			return
		}
		if cleanupErr := cfg.Campaign.Lifecycle.Cleanup(ctx); cleanupErr != nil {
			err = errors.Join(err, fmt.Errorf("cleanup: %w", cleanupErr))
		}
	}()

	before, err := cfg.Campaign.Probe.Observe(ctx)
	if err != nil {
		return arm, nil, fmt.Errorf("soak preflight: %w", err)
	}
	if err := admitObservation(before, cfg.Campaign); err != nil {
		return arm, nil, err
	}
	client := r.client(cfg.Campaign.Endpoint.Timeout)
	if err := preflightModel(ctx, client, cfg.Campaign.Endpoint); err != nil {
		return arm, nil, err
	}

	coding := r.runCodingTasks(ctx, client, cfg.Campaign, tasks)
	malformed := runMalformedScenario(ctx, client, cfg.Campaign.Endpoint)
	malformed.Readback, malformed.Failure = readbackScenario(ctx, cfg.Campaign, malformed.Failure)
	if malformed.Failure != "" {
		malformed.Outcome = "FAIL"
	}
	cancelled := r.runCancellationScenario(ctx, client, cfg)
	if readyErr := cfg.Campaign.Lifecycle.Ready(ctx); readyErr != nil {
		cancelled.Failure = joinFailure(cancelled.Failure, "readiness after cancellation: "+readyErr.Error())
	}
	cancelled.Readback, cancelled.Failure = readbackScenario(ctx, cfg.Campaign, cancelled.Failure)
	if cancelled.Failure != "" {
		cancelled.Outcome = "FAIL"
	}

	var decodeArchive *DecodeCampaignArchive
	var matchedDecode *MatchedDecodeWindowSummary
	if cfg.LongDecode != nil {
		endpoint := cfg.Campaign.Endpoint
		endpoint.NativeDecodeTrace = true
		endpoint.Repetitions = MinimumDecodeRepetitions
		endpoint.Sample = func(sampleCtx context.Context) (ResourceSample, error) {
			observed, sampleErr := cfg.Campaign.Probe.Observe(sampleCtx)
			if sampleErr != nil {
				return ResourceSample{}, sampleErr
			}
			if err := admitObservation(observed, cfg.Campaign); err != nil {
				return ResourceSample{}, err
			}
			return ResourceSample{MemoryBytes: observed.MemoryBytes, PowerWatts: observed.PowerWatts}, nil
		}
		native, decodeErr := r.RunLongDecode(ctx, endpoint, cfg.LongDecode.Fixture)
		if decodeErr != nil {
			return arm, nil, fmt.Errorf("long decode campaign: %w", decodeErr)
		}
		decodeArchive = &DecodeCampaignArchive{Schema: DecodeCampaignSchema, FixtureID: cfg.LongDecode.Fixture.ID, Native: native, Comparator: append([]LlamaClientDecodeResult(nil), cfg.LongDecode.Comparator...)}
		if len(cfg.LongDecode.Comparator) > 0 {
			matched, matchErr := FoldMatchedDecodeCampaign(native, cfg.LongDecode.Comparator)
			if matchErr != nil {
				return arm, nil, fmt.Errorf("matched decode campaign: %w", matchErr)
			}
			matchedDecode = &matched
		}
	}

	cleanupOwnedByCampaign = true
	campaignCfg := cfg.Campaign
	if cfg.LongDecode != nil {
		campaignCfg.Endpoint.NativeDecodeTrace = false
	}
	campaign, campaignErr := r.RunCampaign(ctx, campaignCfg, corpus)
	if campaignErr != nil {
		return arm, nil, campaignErr
	}
	var campaignArchive Archive
	if err := json.Unmarshal(campaign.Archive, &campaignArchive); err != nil {
		return arm, nil, fmt.Errorf("decode campaign archive: %w", err)
	}
	scenarios := []SoakScenario{
		contextScenario(campaignArchive),
		malformed,
		cancelled,
		restartScenario(campaignArchive),
		cacheScenario(campaignArchive),
	}
	metrics := summarizeMetrics(coding, campaignArchive.Results, cfg.Campaign.Endpoint.NativeDecodeTrace && cfg.LongDecode == nil)
	if decodeArchive != nil {
		summary := FoldDecodeResults(decodeArchive.Native)
		metrics.DecodeWindows = &summary
	}
	campaignHash := sha256.Sum256(campaign.Archive)
	arm = SoakArmResult{
		Arm:                   cfg.Campaign.Arm,
		Campaign:              campaign.Report,
		CampaignArchiveSHA256: hex.EncodeToString(campaignHash[:]),
		Coding:                coding,
		Scenarios:             scenarios,
		Metrics:               metrics,
		MatchedDecode:         matchedDecode,
	}
	arm.Verdict = deriveArmVerdict(arm)
	armRaw := soakArmArchive{Schema: SoakArchiveSchema, Arm: arm.Arm, Coding: coding, Scenarios: scenarios, Campaign: json.RawMessage(campaign.Archive), Decode: decodeArchive}
	archive, err = canonicalJSON(armRaw)
	if err != nil {
		return SoakArmResult{}, nil, err
	}
	archive = scrubSecret(archive, cfg.Campaign.Endpoint.APIKey)
	armHash := sha256.Sum256(archive)
	arm.ArchiveSHA256 = hex.EncodeToString(armHash[:])
	return arm, archive, nil
}

func (r Runner) client(timeout time.Duration) *http.Client {
	if r.Client != nil {
		return r.Client
	}
	if timeout == 0 {
		timeout = 10 * time.Minute
	}
	return &http.Client{Timeout: timeout}
}

func (r Runner) runCodingTasks(ctx context.Context, client *http.Client, cfg CampaignConfig, tasks []CodingTask) []Result {
	endpoint := cfg.Endpoint
	endpoint.NativeDecodeTrace = false
	out := make([]Result, 0, len(tasks))
	for _, task := range tasks {
		fixture := qwen38quant.Fixture{ID: task.ID, Workload: "coding_reasoning", Prompt: task.Prompt, ExpectedExact: task.ExpectedExact, MaxOutputTokens: task.MaxOutputTokens}
		result := Result{FixtureID: task.ID, Workload: fixture.Workload, Repeat: 1, Quality: "FAIL"}
		start := time.Now()
		response, err := runOne(ctx, client, endpoint, fixture)
		result.LatencyMS = float64(time.Since(start).Microseconds()) / 1000
		if err != nil {
			result.Failure = err.Error()
			out = append(out, result)
			continue
		}
		result.Usage = response.Usage
		result.CachedInputTokens = response.UsageDetails.CachedTokens
		observed, observeErr := cfg.Probe.Observe(ctx)
		if observeErr != nil {
			result.Failure = "resource readback: " + observeErr.Error()
			out = append(out, result)
			continue
		}
		if admitErr := admitObservation(observed, cfg); admitErr != nil {
			result.Failure = "resource readback: " + admitErr.Error()
			out = append(out, result)
			continue
		}
		result.Resource = ResourceSample{MemoryBytes: observed.MemoryBytes, PowerWatts: observed.PowerWatts}
		grade(&result, fixture, response)
		out = append(out, result)
	}
	return out
}

func runMalformedScenario(ctx context.Context, client *http.Client, cfg Config) SoakScenario {
	scenario := SoakScenario{Name: "malformed_call", Outcome: "FAIL"}
	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.Endpoint, "/")+"/v1/chat/completions", bytes.NewBufferString(`{"model":`))
	if err != nil {
		scenario.Failure = err.Error()
		return scenario
	}
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	rsp, err := client.Do(req)
	scenario.LatencyMS = float64(time.Since(start).Microseconds()) / 1000
	if err != nil {
		scenario.Failure = err.Error()
		return scenario
	}
	defer rsp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(rsp.Body, 1<<20))
	if rsp.StatusCode/100 != 4 {
		scenario.Failure = fmt.Sprintf("malformed request returned HTTP %d", rsp.StatusCode)
		return scenario
	}
	scenario.Outcome = "PASS"
	return scenario
}

func (r Runner) runCancellationScenario(ctx context.Context, client *http.Client, cfg SoakArmConfig) SoakScenario {
	scenario := SoakScenario{Name: "cancellation", Outcome: "FAIL"}
	delay := cfg.CancellationAfter
	if delay <= 0 {
		delay = defaultCancellationLag
	}
	cancelCtx, cancel := context.WithCancel(ctx)
	timer := time.AfterFunc(delay, cancel)
	start := time.Now()
	fixture := qwen38quant.Fixture{ID: "cancellation-v1", Workload: "coding_reasoning", Prompt: "Write a complete Go program that enumerates prime numbers below one million, then explain its complexity.", ExpectedExact: "cancelled", MaxOutputTokens: 2048}
	_, err := runOne(cancelCtx, client, cfg.Campaign.Endpoint, fixture)
	timer.Stop()
	cancel()
	scenario.LatencyMS = float64(time.Since(start).Microseconds()) / 1000
	if err == nil {
		scenario.Failure = "request completed before cancellation"
		return scenario
	}
	if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) && !strings.Contains(strings.ToLower(err.Error()), "canceled") {
		scenario.Failure = "unexpected cancellation result: " + err.Error()
		return scenario
	}
	scenario.Outcome = "PASS"
	return scenario
}

func readbackScenario(ctx context.Context, cfg CampaignConfig, failure string) (Observation, string) {
	observed, err := cfg.Probe.Observe(ctx)
	if err != nil {
		return Observation{}, joinFailure(failure, "independent readback: "+err.Error())
	}
	if err := admitObservation(observed, cfg); err != nil {
		return observed, joinFailure(failure, "independent readback: "+err.Error())
	}
	return observed, failure
}

func contextScenario(archive Archive) SoakScenario {
	return resultsScenario("context_pressure", archive.Results, "long_context_retrieval", archive.After)
}

func restartScenario(archive Archive) SoakScenario {
	s := SoakScenario{Name: "restart", Outcome: "PASS", Readback: archive.After}
	if !archive.RestartReady {
		s.Outcome, s.Failure = "FAIL", "restart readiness was not witnessed"
	}
	return s
}

func cacheScenario(archive Archive) SoakScenario {
	s := resultsScenario("cache_recovery", archive.Results, "repeated_workflow_cache", archive.After)
	if !archive.RestartReady {
		s.Outcome, s.Failure = "FAIL", joinFailure(s.Failure, "restart readiness was not witnessed")
	}
	return s
}

func resultsScenario(name string, results []Result, workload string, readback Observation) SoakScenario {
	s := SoakScenario{Name: name, Outcome: "PASS", Readback: readback}
	found := 0
	for _, result := range results {
		if result.Workload != workload {
			continue
		}
		found++
		s.LatencyMS += result.LatencyMS
		if result.Quality != "PASS" {
			s.Outcome = "FAIL"
			s.Failure = joinFailure(s.Failure, fmt.Sprintf("%s/%d: %s", result.FixtureID, result.Repeat, result.Failure))
		}
	}
	if found == 0 {
		s.Outcome, s.Failure = "FAIL", "scenario was not executed"
	} else {
		s.LatencyMS /= float64(found)
	}
	return s
}

func summarizeMetrics(coding, campaign []Result, requireDecodeWindows bool) SoakMetrics {
	latencies := make([]float64, 0, len(coding))
	var outputTokens int
	var elapsedSeconds float64
	var metrics SoakMetrics
	for _, result := range append(append([]Result(nil), coding...), campaign...) {
		if result.Resource.MemoryBytes > metrics.PeakMemoryBytes {
			metrics.PeakMemoryBytes = result.Resource.MemoryBytes
		}
		if result.Resource.PowerWatts > metrics.PeakPowerWatts {
			metrics.PeakPowerWatts = result.Resource.PowerWatts
		}
	}
	for _, result := range coding {
		latencies = append(latencies, result.LatencyMS)
		outputTokens += usageValue(result.Usage, "completion_tokens", "output_tokens")
		elapsedSeconds += result.LatencyMS / 1000
	}
	if len(latencies) > 0 {
		sort.Float64s(latencies)
		middle := len(latencies) / 2
		metrics.CodingLatencyP50MS = latencies[middle]
		if len(latencies)%2 == 0 {
			metrics.CodingLatencyP50MS = (latencies[middle-1] + latencies[middle]) / 2
		}
	}
	if elapsedSeconds > 0 {
		metrics.CodingThroughput = float64(outputTokens) / elapsedSeconds
	}
	for _, result := range campaign {
		if result.Workload != "repeated_workflow_cache" {
			continue
		}
		switch result.Phase {
		case "cold":
			metrics.CacheColdMS = result.LatencyMS
		case "warm":
			metrics.CacheWarmMS = result.LatencyMS
		case "warm_after_restart":
			metrics.CacheAfterRestart = result.LatencyMS
		}
	}
	metrics.CacheSavedMS = metrics.CacheColdMS - metrics.CacheWarmMS
	if requireDecodeWindows {
		summary := FoldDecodeResults(campaign)
		metrics.DecodeWindows = &summary
	}
	return metrics
}

func usageValue(usage map[string]int, keys ...string) int {
	for _, key := range keys {
		if usage[key] > 0 {
			return usage[key]
		}
	}
	return 0
}

func AssembleSoakReport(arms []SoakArmResult, corpus qwen38quant.Corpus, selectedArm, rollbackThreshold string) SoakReport {
	report := SoakReport{
		Schema:            SoakSchema,
		CorpusID:          corpus.ID,
		CorpusSHA256:      qwen38quant.CorpusDigest(corpus),
		Arms:              append([]SoakArmResult(nil), arms...),
		SelectedArm:       selectedArm,
		Verdict:           "HOLD",
		RollbackThreshold: rollbackThreshold,
	}
	if selectedArm != "" {
		for _, arm := range arms {
			if arm.Arm == selectedArm && arm.Verdict == "PROMOTE" {
				report.Verdict = "PROMOTE"
			}
		}
	}
	return report
}

func ValidateSoakReport(report SoakReport, corpus qwen38quant.Corpus) error {
	if err := corpus.Validate(); err != nil {
		return err
	}
	if report.Schema != SoakSchema || report.CorpusID != corpus.ID || report.CorpusSHA256 != qwen38quant.CorpusDigest(corpus) {
		return errors.New("soak corpus identity mismatch")
	}
	if len(report.Arms) < MinimumSoakFinalists {
		return fmt.Errorf("soak requires at least %d finalists", MinimumSoakFinalists)
	}
	if report.RollbackThreshold == "" || !validSHA256(report.RawArchiveSHA256) {
		return errors.New("soak lifecycle or archive evidence is incomplete")
	}
	if report.Verdict != "PROMOTE" && report.Verdict != "HOLD" && report.Verdict != "EXCLUDE" {
		return errors.New("invalid soak verdict")
	}
	seen, selectedExists, selectedPromotable := map[string]bool{}, false, false
	for _, arm := range report.Arms {
		if arm.Arm == "" || seen[arm.Arm] {
			return errors.New("missing or duplicate soak arm")
		}
		seen[arm.Arm] = true
		if err := validateSoakArm(arm, corpus); err != nil {
			return fmt.Errorf("arm %s: %w", arm.Arm, err)
		}
		if arm.Arm == report.SelectedArm {
			selectedExists = true
			selectedPromotable = arm.Verdict == "PROMOTE"
		}
	}
	if report.SelectedArm != "" && !selectedExists {
		return errors.New("selected soak arm is absent")
	}
	if report.Verdict == "PROMOTE" && (!selectedPromotable || report.SelectedArm == "") {
		return errors.New("PROMOTE requires a fully qualified selected arm")
	}
	return nil
}

// ValidateSoakArtifacts binds a validator-clean report to the archive bytes it
// names, then checks that every embedded arm archive has the recorded hash.
func ValidateSoakArtifacts(report SoakReport, archive []byte, corpus qwen38quant.Corpus) error {
	if err := ValidateSoakReport(report, corpus); err != nil {
		return err
	}
	sum := sha256.Sum256(archive)
	if hex.EncodeToString(sum[:]) != report.RawArchiveSHA256 {
		return errors.New("soak raw archive hash mismatch")
	}
	var raw soakArchive
	if err := json.Unmarshal(archive, &raw); err != nil {
		return fmt.Errorf("decode soak archive: %w", err)
	}
	if raw.Schema != SoakArchiveSchema || len(raw.Arms) != len(report.Arms) {
		return errors.New("soak archive arm set mismatch")
	}
	want := make(map[string]SoakArmResult, len(report.Arms))
	for _, arm := range report.Arms {
		want[arm.Arm] = arm
	}
	seen := make(map[string]bool, len(raw.Arms))
	for _, encoded := range raw.Arms {
		var arm soakArmArchive
		if err := json.Unmarshal(encoded, &arm); err != nil {
			return fmt.Errorf("decode soak arm archive: %w", err)
		}
		reported, ok := want[arm.Arm]
		if arm.Arm == "" || seen[arm.Arm] || !ok {
			return errors.New("soak archive arm set mismatch")
		}
		seen[arm.Arm] = true
		if !reflect.DeepEqual(arm.Coding, reported.Coding) || !reflect.DeepEqual(arm.Scenarios, reported.Scenarios) {
			return fmt.Errorf("arm %s archive results mismatch", arm.Arm)
		}
		var campaign Archive
		if err := json.Unmarshal(arm.Campaign, &campaign); err != nil {
			return fmt.Errorf("decode arm %s campaign archive: %w", arm.Arm, err)
		}
		if campaign.Arm != arm.Arm || campaign.CorpusID != report.CorpusID {
			return fmt.Errorf("arm %s campaign archive identity mismatch", arm.Arm)
		}
		if arm.Decode != nil {
			if arm.Decode.Schema != DecodeCampaignSchema || arm.Decode.FixtureID == "" {
				return fmt.Errorf("arm %s decode campaign identity mismatch", arm.Arm)
			}
			for _, result := range arm.Decode.Native {
				if result.FixtureID != arm.Decode.FixtureID {
					return fmt.Errorf("arm %s native decode fixture mismatch", arm.Arm)
				}
			}
			for _, result := range arm.Decode.Comparator {
				if result.FixtureID != arm.Decode.FixtureID {
					return fmt.Errorf("arm %s comparator decode fixture mismatch", arm.Arm)
				}
			}
			rebuilt := FoldDecodeResults(arm.Decode.Native)
			if reported.Metrics.DecodeWindows == nil || !reflect.DeepEqual(rebuilt, *reported.Metrics.DecodeWindows) {
				return fmt.Errorf("arm %s decode-window archive mismatch", arm.Arm)
			}
			if len(arm.Decode.Comparator) > 0 {
				matched, err := FoldMatchedDecodeCampaign(arm.Decode.Native, arm.Decode.Comparator)
				if err != nil || reported.MatchedDecode == nil || !reflect.DeepEqual(matched, *reported.MatchedDecode) {
					return fmt.Errorf("arm %s matched decode archive mismatch: %v", arm.Arm, err)
				}
			} else if reported.MatchedDecode != nil {
				return fmt.Errorf("arm %s matched decode report lacks comparator raw evidence", arm.Arm)
			}
		} else {
			hasDecodeTrace := false
			for _, result := range campaign.Results {
				hasDecodeTrace = hasDecodeTrace || result.DecodeTrace != nil || result.DecodeWindows != nil
			}
			if hasDecodeTrace || reported.Metrics.DecodeWindows != nil {
				rebuilt := FoldDecodeResults(campaign.Results)
				if reported.Metrics.DecodeWindows == nil || !reflect.DeepEqual(rebuilt, *reported.Metrics.DecodeWindows) {
					return fmt.Errorf("arm %s decode-window archive mismatch", arm.Arm)
				}
			}
			if reported.MatchedDecode != nil {
				return fmt.Errorf("arm %s matched decode report lacks raw evidence", arm.Arm)
			}
		}
		campaignCanonical, err := canonicalJSON(campaign)
		if err != nil {
			return err
		}
		campaignSum := sha256.Sum256(campaignCanonical)
		if hex.EncodeToString(campaignSum[:]) != reported.CampaignArchiveSHA256 {
			return fmt.Errorf("arm %s campaign archive hash mismatch", arm.Arm)
		}
		canonical, err := canonicalJSON(arm)
		if err != nil {
			return err
		}
		armSum := sha256.Sum256(canonical)
		if hex.EncodeToString(armSum[:]) != reported.ArchiveSHA256 {
			return fmt.Errorf("arm %s archive hash mismatch", arm.Arm)
		}
	}
	return nil
}

func validateSoakArm(arm SoakArmResult, corpus qwen38quant.Corpus) error {
	if arm.Campaign.Arm != arm.Arm {
		return errors.New("campaign arm mismatch")
	}
	if err := qwen38quant.Validate(arm.Campaign, corpus); err != nil {
		return fmt.Errorf("campaign: %w", err)
	}
	if !validSHA256(arm.CampaignArchiveSHA256) || !validSHA256(arm.ArchiveSHA256) {
		return errors.New("missing archive hash")
	}
	if len(arm.Coding) != MinimumCodingTasks {
		return fmt.Errorf("coding tasks: got %d want %d", len(arm.Coding), MinimumCodingTasks)
	}
	wantTasks := DefaultSoakTasks()
	wantIDs := make(map[string]bool, len(wantTasks))
	for _, task := range wantTasks {
		wantIDs[task.ID] = true
	}
	seenTasks := map[string]bool{}
	for _, result := range arm.Coding {
		if !wantIDs[result.FixtureID] || seenTasks[result.FixtureID] || result.Workload != "coding_reasoning" {
			return errors.New("coding task set mismatch")
		}
		seenTasks[result.FixtureID] = true
		if result.Quality != "PASS" && result.Failure == "" {
			return fmt.Errorf("coding task %s lost its failure", result.FixtureID)
		}
	}
	if err := validateScenarios(arm.Scenarios, arm.Campaign.Identity); err != nil {
		return err
	}
	metrics := arm.Metrics
	if metrics.CodingLatencyP50MS <= 0 || metrics.CodingThroughput < 0 || math.IsNaN(metrics.CodingThroughput) || math.IsInf(metrics.CodingThroughput, 0) || metrics.PeakMemoryBytes == 0 || metrics.PeakPowerWatts <= 0 || metrics.CacheColdMS <= 0 || metrics.CacheWarmMS <= 0 || metrics.CacheAfterRestart <= 0 || math.IsNaN(metrics.CacheSavedMS) || math.IsInf(metrics.CacheSavedMS, 0) {
		return fmt.Errorf("latency, throughput, memory, power, or cache metrics missing: %+v", metrics)
	}
	if metrics.DecodeWindows != nil {
		if err := validateDecodeWindowSummary(*metrics.DecodeWindows); err != nil {
			return err
		}
	}
	if arm.MatchedDecode != nil {
		if err := validateMatchedDecodeWindowSummary(*arm.MatchedDecode); err != nil {
			return err
		}
	}
	if arm.Verdict != deriveArmVerdict(arm) {
		return errors.New("arm verdict does not follow retained quality evidence")
	}
	return nil
}

func validateScenarios(scenarios []SoakScenario, identity qwen38quant.Identity) error {
	if len(scenarios) != len(requiredSoakScenarios) {
		return errors.New("failure-scenario set incomplete")
	}
	want := map[string]bool{}
	for _, name := range requiredSoakScenarios {
		want[name] = true
	}
	seen := map[string]bool{}
	for _, scenario := range scenarios {
		if !want[scenario.Name] || seen[scenario.Name] {
			return errors.New("failure-scenario set mismatch")
		}
		seen[scenario.Name] = true
		if scenario.Outcome != "PASS" && scenario.Outcome != "FAIL" && scenario.Outcome != "TIMEOUT" {
			return fmt.Errorf("scenario %s has invalid outcome", scenario.Name)
		}
		if scenario.Outcome != "PASS" && scenario.Failure == "" {
			return fmt.Errorf("scenario %s lost its failure", scenario.Name)
		}
		if !reflect.DeepEqual(scenario.Readback.Identity, identity) || !scenario.Readback.Resident || scenario.Readback.FallbackActive {
			return fmt.Errorf("scenario %s lacks independent identity/residency readback", scenario.Name)
		}
	}
	return nil
}

func deriveArmVerdict(arm SoakArmResult) string {
	if arm.Campaign.Verdict == "EXCLUDE" {
		return "EXCLUDE"
	}
	if arm.Campaign.Verdict != "PROMOTE" {
		return "HOLD"
	}
	if arm.Metrics.DecodeWindows != nil && arm.Metrics.DecodeWindows.Verdict != "PASS" {
		return "HOLD"
	}
	if arm.MatchedDecode != nil && arm.MatchedDecode.Verdict != "PASS" {
		return "HOLD"
	}
	for _, result := range arm.Coding {
		if result.Quality != "PASS" {
			return "HOLD"
		}
	}
	for _, scenario := range arm.Scenarios {
		if scenario.Outcome != "PASS" {
			return "HOLD"
		}
	}
	return "PROMOTE"
}

func validateSoakTasks(tasks []CodingTask) error {
	want := DefaultSoakTasks()
	if len(tasks) != len(want) {
		return fmt.Errorf("coding tasks: got %d want %d", len(tasks), len(want))
	}
	byID := make(map[string]CodingTask, len(tasks))
	for _, task := range tasks {
		if task.ID == "" || task.Prompt == "" || task.ExpectedExact == "" || task.MaxOutputTokens <= 0 || byID[task.ID].ID != "" {
			return errors.New("invalid or duplicate coding task")
		}
		byID[task.ID] = task
	}
	for _, task := range want {
		if !reflect.DeepEqual(byID[task.ID], task) {
			return fmt.Errorf("coding task %s drifted", task.ID)
		}
	}
	return nil
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func joinFailure(current, next string) string {
	if current == "" {
		return next
	}
	if next == "" {
		return current
	}
	return current + "; " + next
}

// SoakAdapterConfig is the file-backed three-finalist operator contract.
// Secrets are named by environment variable and never accepted inline.
type SoakAdapterConfig struct {
	Schema            string           `json:"schema"`
	Finalists         []SoakAdapterArm `json:"finalists"`
	SelectedArm       string           `json:"selected_arm,omitempty"`
	RollbackThreshold string           `json:"rollback_threshold"`
	CancellationLagMS int              `json:"cancellation_lag_ms,omitempty"`
}

type SoakAdapterArm struct {
	Campaign   AdapterConfig             `json:"campaign"`
	APIKeyEnv  string                    `json:"api_key_env"`
	Setup      []string                  `json:"setup_command,omitempty"`
	LongDecode *LongDecodeCampaignConfig `json:"long_decode,omitempty"`
}

func RunSoakAdapter(ctx context.Context, configPath, corpusPath, reportPath, archivePath string) error {
	var cfg SoakAdapterConfig
	if err := decodeFile(configPath, &cfg); err != nil {
		return fmt.Errorf("soak config: %w", err)
	}
	if cfg.Schema != SoakSchema || len(cfg.Finalists) < MinimumSoakFinalists || cfg.RollbackThreshold == "" {
		return errors.New("soak config requires schema, three finalists, and rollback threshold")
	}
	corpusBytes, err := os.ReadFile(corpusPath)
	if err != nil {
		return fmt.Errorf("corpus: %w", err)
	}
	corpus, err := qwen38quant.DecodeCorpus(corpusBytes)
	if err != nil {
		return fmt.Errorf("corpus: %w", err)
	}
	lag := time.Duration(cfg.CancellationLagMS) * time.Millisecond
	var arms []SoakArmResult
	rawArms := make([]json.RawMessage, 0, len(cfg.Finalists))
	seenArms := make(map[string]bool, len(cfg.Finalists))
	for _, finalist := range cfg.Finalists {
		campaignCfg := finalist.Campaign
		if campaignCfg.Arm == "" || seenArms[campaignCfg.Arm] {
			return errors.New("soak finalists require unique non-empty arms")
		}
		seenArms[campaignCfg.Arm] = true
		if len(campaignCfg.ObservationCommand) == 0 {
			return fmt.Errorf("arm %s: observation_command is required", campaignCfg.Arm)
		}
		if len(campaignCfg.RestartCommand) == 0 || len(campaignCfg.ReadyCommand) == 0 || len(campaignCfg.CleanupCommand) == 0 {
			return fmt.Errorf("arm %s: restart_command, ready_command, and cleanup_command are required", campaignCfg.Arm)
		}
		if finalist.Campaign.Endpoint.APIKey != "" || finalist.APIKeyEnv == "" {
			return fmt.Errorf("arm %s: use api_key_env; inline API keys are refused", campaignCfg.Arm)
		}
		apiKey := os.Getenv(finalist.APIKeyEnv)
		if apiKey == "" {
			return fmt.Errorf("arm %s: API key environment %s is empty", campaignCfg.Arm, finalist.APIKeyEnv)
		}
		if len(finalist.Setup) > 0 {
			if _, err := runArgv(ctx, finalist.Setup); err != nil {
				return fmt.Errorf("arm %s setup: %w", campaignCfg.Arm, err)
			}
		}
		campaignCfg.Endpoint.APIKey = apiKey
		arm, raw, err := (Runner{}).RunSoakArm(ctx, SoakArmConfig{
			Campaign: CampaignConfig{
				Endpoint: campaignCfg.Endpoint.runnerConfig(), ExecutionEngine: campaignCfg.ExecutionEngine, Arm: campaignCfg.Arm, Expected: campaignCfg.Expected,
				Command: campaignCfg.Command, RequireDevice: campaignCfg.RequireDevice, StaleAfter: campaignCfg.StaleAfter,
				RollbackThreshold: campaignCfg.RollbackThreshold,
				Probe:             commandProbe{argv: campaignCfg.ObservationCommand},
				Lifecycle: commandLifecycle{
					restart: campaignCfg.RestartCommand,
					ready:   campaignCfg.ReadyCommand,
					cleanup: campaignCfg.CleanupCommand,
				},
			},
			CancellationAfter: lag,
			LongDecode:        finalist.LongDecode,
		}, corpus, DefaultSoakTasks())
		if err != nil {
			return fmt.Errorf("arm %s: %w", campaignCfg.Arm, err)
		}
		arms = append(arms, arm)
		rawArms = append(rawArms, json.RawMessage(raw))
	}
	archiveBytes, err := canonicalJSON(soakArchive{Schema: SoakArchiveSchema, Arms: rawArms})
	if err != nil {
		return err
	}
	report := AssembleSoakReport(arms, corpus, cfg.SelectedArm, cfg.RollbackThreshold)
	archiveHash := sha256.Sum256(archiveBytes)
	report.RawArchiveSHA256 = hex.EncodeToString(archiveHash[:])
	if err := ValidateSoakArtifacts(report, archiveBytes, corpus); err != nil {
		return fmt.Errorf("validate soak report: %w", err)
	}
	reportBytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	reportBytes = append(reportBytes, '\n')
	if err := writeAtomic(archivePath, archiveBytes, 0o600); err != nil {
		return fmt.Errorf("soak archive: %w", err)
	}
	if err := writeAtomic(reportPath, reportBytes, 0o644); err != nil {
		_ = os.Remove(archivePath)
		return fmt.Errorf("soak report: %w", err)
	}
	return nil
}
