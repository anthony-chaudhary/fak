package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

const (
	qwen38PagedSwapSchema      = "fak.modelbench.qwen38-paged-swap/1"
	qwen38PagedSwapEngine      = "fak-native"
	qwen38PagedSwapBackend     = "metal"
	qwen38PagedSwapForwardPath = "native-scheduler/qwen38-metal-q4k/paged-swap-v1"
	qwen38PagedSwapRepetitions = 3
)

// The transition rules mirror primary OSS practice without copying source:
// Apache-2.0 vLLM scheduler.py@f49777ba62b4926d0f8c100ab06edb03c5c10098
// keeps LATER victims queued and publishes RUNNING only after swap_in succeeds;
// https://github.com/vllm-project/vllm/blob/f49777ba62b4926d0f8c100ab06edb03c5c10098/vllm/core/scheduler.py
// Apache-2.0 SGLang hiradix_cache.py@5ab97c4f441462f3d2adb1ffa954cc92518beece
// protects host state during load-back and publishes device state only after transfer.
// https://github.com/sgl-project/sglang/blob/5ab97c4f441462f3d2adb1ffa954cc92518beece/python/sglang/srt/mem_cache/hiradix_cache.py
type qwen38PagedSwapReceipt struct {
	Schema            string                   `json:"schema"`
	BindingSHA256     string                   `json:"binding_sha256"`
	Verdict           string                   `json:"verdict"`
	Engine            string                   `json:"engine"`
	Backend           string                   `json:"backend"`
	ForwardPath       string                   `json:"forward_path"`
	Artifact          nativeArtifactIdentity   `json:"artifact"`
	ModelConfig       map[string]any           `json:"model_config"`
	ModelConfigSHA256 string                   `json:"model_config_sha256"`
	Host              nativeHostIdentity       `json:"host"`
	Source            nativeSourceIdentity     `json:"source"`
	Binary            nativeFileIdentity       `json:"binary"`
	Controls          map[string]string        `json:"controls"`
	ArrivalTrace      []qwen38PagedSwapArrival `json:"arrival_trace"`
	Repetitions       []qwen38PagedSwapPair    `json:"repetitions"`
}

type qwen38PagedSwapArrival struct {
	Request   string `json:"request"`
	Ordinal   int    `json:"ordinal"`
	PromptSHA string `json:"prompt_sha256"`
	Tokens    int    `json:"tokens"`
}

type qwen38PagedSwapPair struct {
	Repetition int                `json:"repetition"`
	Off        qwen38PagedSwapArm `json:"off"`
	On         qwen38PagedSwapArm `json:"on"`
}

type qwen38PagedSwapArm struct {
	Pressure                 string                                  `json:"pressure"`
	Requests                 []qwen38PagedSwapRequest                `json:"requests"`
	SessionExecutions        []qwen38PagedSwapSessionExecution       `json:"session_executions"`
	StateIdentities          []model.Qwen35MetalStateIdentityReceipt `json:"state_identities"`
	SwapUsage                []modelperfobs.QwenSwapWeeklyUsage      `json:"swap_usage,omitempty"`
	SwapTotal                int64                                   `json:"swap_total"`
	ReadmittedTotal          int64                                   `json:"readmitted_total"`
	SwapBytes                int64                                   `json:"swap_bytes"`
	RestoredBytes            int64                                   `json:"restored_bytes"`
	RecomputeTotal           int64                                   `json:"recompute_total"`
	KVMaxBlocks              int                                     `json:"kv_max_blocks"`
	PeakRunning              int                                     `json:"peak_running"`
	PeakUsedBlocks           int                                     `json:"peak_used_blocks"`
	PeakRSSBytes             uint64                                  `json:"peak_rss_bytes"`
	TTFTP50Milliseconds      float64                                 `json:"ttft_p50_ms"`
	TTFTP95Milliseconds      float64                                 `json:"ttft_p95_ms"`
	ITLP50Milliseconds       float64                                 `json:"itl_p50_ms"`
	ITLP95Milliseconds       float64                                 `json:"itl_p95_ms"`
	AggregateTokensPerSecond float64                                 `json:"aggregate_tokens_per_second"`
	FallbackTotal            int                                     `json:"fallback_total"`
	ErrorTotal               int                                     `json:"error_total"`
	RefusalTotal             int                                     `json:"refusal_total"`
	TeardownMilliseconds     float64                                 `json:"teardown_ms"`
	TeardownComplete         bool                                    `json:"teardown_complete"`
}

type qwen38PagedSwapRequest struct {
	Request          string    `json:"request"`
	TokenIDs         []int     `json:"token_ids"`
	OutputSHA256     string    `json:"output_sha256"`
	TTFTMilliseconds float64   `json:"ttft_ms"`
	ITLMilliseconds  []float64 `json:"itl_ms"`
}

type qwen38PagedSwapSessionExecution struct {
	Lifecycle modelengine.NativeSessionLifecycle `json:"lifecycle"`
	Execution metalgemm.ExecutionReceipt         `json:"execution"`
	Fallbacks model.MetalFallbackReceipt         `json:"fallbacks"`
}

type qwen38PagedSwapProfiler struct {
	lifecycle modelengine.NativeSessionLifecycle
	profiler  *model.PhaseProfiler
}

var qwen38PagedSwapRequiredEnvironment = map[string]string{
	"FAK_NATIVE_MAX_RUNNING":     "2",
	"FAK_NATIVE_KV_BLOCK_TOKENS": "16",
	"FAK_NATIVE_KV_PREEMPT_MODE": "swap",
}

var qwen38PagedSwapOptionalEnvironment = map[string]bool{
	"FAK_NATIVE_KV_MAX_BLOCKS":  true,
	"FAK_NATIVE_KV_VICTIM_RULE": true,
}

func qwen38PagedSwapControls(lookup func(string) (string, bool), environ []string, budget float64) (map[string]string, int, error) {
	filtered := make([]string, 0, len(environ))
	for _, declaration := range environ {
		key, _, _ := strings.Cut(declaration, "=")
		if _, ok := qwen38PagedSwapRequiredEnvironment[key]; ok {
			continue
		}
		if qwen38PagedSwapOptionalEnvironment[key] {
			continue
		}
		filtered = append(filtered, declaration)
	}
	controls, err := nativeProfileControlEnvironment(lookup, filtered, budget, model.Qwen35DecodeHandoffAuto)
	if err != nil {
		return nil, 0, err
	}
	if controls[nativeProfileSequenceSelector] != nativeProfileUnset && controls[nativeProfileSequenceSelector] != nativeProfileSelectorOff {
		return nil, 0, fmt.Errorf("qwen38 paged-swap requires %s unset or OFF", nativeProfileSequenceSelector)
	}
	for key, want := range qwen38PagedSwapRequiredEnvironment {
		got, ok := lookup(key)
		if !ok || strings.TrimSpace(got) != want {
			return nil, 0, fmt.Errorf("qwen38 paged-swap requires %s=%s", key, want)
		}
		controls[key] = want
	}
	raw, ok := lookup("FAK_NATIVE_KV_MAX_BLOCKS")
	maxBlocks, err := strconv.Atoi(strings.TrimSpace(raw))
	if !ok || err != nil || maxBlocks < 2 || maxBlocks > 3 {
		return nil, 0, fmt.Errorf("qwen38 paged-swap requires FAK_NATIVE_KV_MAX_BLOCKS=2 or 3 for the exact two-P32/block16 trace")
	}
	controls["FAK_NATIVE_KV_MAX_BLOCKS"] = strconv.Itoa(maxBlocks)
	victim := "most-recent"
	if raw, ok := lookup("FAK_NATIVE_KV_VICTIM_RULE"); ok {
		victim = strings.TrimSpace(raw)
	}
	if victim != "most-recent" && victim != "newest" {
		return nil, 0, fmt.Errorf("qwen38 paged-swap requires most-recent victim selection")
	}
	controls["FAK_NATIVE_KV_VICTIM_RULE"] = "most-recent"
	if err := validateQwen38PagedSwapControls(controls); err != nil {
		return nil, 0, err
	}
	return controls, maxBlocks, nil
}

func validateQwen38PagedSwapControls(controls map[string]string) error {
	base := make(map[string]string, len(controls))
	for key, value := range controls {
		if _, ok := qwen38PagedSwapRequiredEnvironment[key]; ok {
			continue
		}
		if qwen38PagedSwapOptionalEnvironment[key] {
			continue
		}
		base[key] = value
	}
	if err := validateNativeProfileControls(base); err != nil {
		return err
	}
	for key, want := range qwen38PagedSwapRequiredEnvironment {
		if controls[key] != want {
			return fmt.Errorf("qwen38 paged-swap control %s=%q, want %q", key, controls[key], want)
		}
	}
	maxBlocks, err := strconv.Atoi(controls["FAK_NATIVE_KV_MAX_BLOCKS"])
	if err != nil || maxBlocks < 2 || maxBlocks > 3 {
		return fmt.Errorf("qwen38 paged-swap control FAK_NATIVE_KV_MAX_BLOCKS must be 2 or 3")
	}
	if controls["FAK_NATIVE_KV_VICTIM_RULE"] != "most-recent" {
		return fmt.Errorf("qwen38 paged-swap victim control is not most-recent")
	}
	if len(controls) != len(base)+len(qwen38PagedSwapRequiredEnvironment)+len(qwen38PagedSwapOptionalEnvironment) {
		return fmt.Errorf("qwen38 paged-swap control receipt has unknown or missing fields")
	}
	return nil
}

func runQwen38PagedSwap(f *benchFlags, m *model.Model, controls map[string]string, maxBlocks int) error {
	if !*f.metal || !metalgemm.Available() {
		return fmt.Errorf("native Metal backend unavailable; refusing CPU fallback")
	}
	if err := validateQwen38PagedSwapControls(controls); err != nil {
		return err
	}
	artifact, err := fileIdentity(*f.gguf)
	if err != nil {
		return err
	}
	envelope, err := exactMetalProfileEnvelope(nativeperf.ActiveGraph(), artifact)
	if err != nil {
		return err
	}
	if envelope.Engine != qwen38PagedSwapEngine || envelope.Backend != qwen38PagedSwapBackend {
		return fmt.Errorf("exact envelope engine/backend=%s/%s, want %s/%s", envelope.Engine, envelope.Backend, qwen38PagedSwapEngine, qwen38PagedSwapBackend)
	}
	host, err := captureNativeHost()
	if err != nil {
		return err
	}
	if err := validateNativeHost(envelope, host); err != nil {
		return err
	}
	source, binary, err := captureNativeBuild()
	if err != nil {
		return err
	}
	config, err := nativeModelConfigIdentity(m.Cfg)
	if err != nil {
		return err
	}
	configSHA, err := sha256JSON(config)
	if err != nil {
		return err
	}
	prompts := [][]int{qwen38PagedSwapPrompt(m.Cfg.VocabSize, 7), qwen38PagedSwapPrompt(m.Cfg.VocabSize, 29)}
	arrivals := make([]qwen38PagedSwapArrival, len(prompts))
	for i, prompt := range prompts {
		digest, err := sha256JSON(prompt)
		if err != nil {
			return err
		}
		arrivals[i] = qwen38PagedSwapArrival{Request: fmt.Sprintf("request-%d", i+1), Ordinal: i + 1, PromptSHA: digest, Tokens: len(prompt)}
	}
	receipt := qwen38PagedSwapReceipt{
		Schema: qwen38PagedSwapSchema, Verdict: "KEEP",
		Engine: qwen38PagedSwapEngine, Backend: qwen38PagedSwapBackend, ForwardPath: qwen38PagedSwapForwardPath,
		Artifact:    nativeArtifactIdentity{nativeFileIdentity: artifact, Model: envelope.Model, ModelRevision: envelope.ModelRevision},
		ModelConfig: config, ModelConfigSHA256: configSHA, Host: host, Source: source, Binary: binary,
		Controls: controls, ArrivalTrace: arrivals,
	}
	for repetition := 1; repetition <= qwen38PagedSwapRepetitions; repetition++ {
		off, err := runQwen38PagedSwapArm(m, prompts, false, maxBlocks)
		if err != nil {
			return fmt.Errorf("repetition %d OFF: %w", repetition, err)
		}
		on, err := runQwen38PagedSwapArm(m, prompts, true, maxBlocks)
		if err != nil {
			return fmt.Errorf("repetition %d ON: %w", repetition, err)
		}
		receipt.Repetitions = append(receipt.Repetitions, qwen38PagedSwapPair{Repetition: repetition, Off: off, On: on})
	}
	receipt.BindingSHA256, err = qwen38PagedSwapBinding(receipt)
	if err != nil {
		return err
	}
	if err := validateQwen38PagedSwapReceipt(receipt); err != nil {
		return fmt.Errorf("qwen38 paged-swap receipt self-check: %w", err)
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(*f.qwenSwapOut), 0o755); err != nil {
		return err
	}
	out, err := os.OpenFile(*f.qwenSwapOut, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("create immutable receipt: %w", err)
	}
	if _, err := out.Write(data); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "wrote", *f.qwenSwapOut)
	return nil
}

func qwen38PagedSwapPrompt(vocab, seed int) []int {
	out := make([]int, 32)
	x := seed
	for i := range out {
		x = (x*48271 + 1) % vocab
		out[i] = x
	}
	return out
}

func runQwen38PagedSwapArm(m *model.Model, prompts [][]int, pressure bool, maxBlocks int) (qwen38PagedSwapArm, error) {
	arm := qwen38PagedSwapArm{Pressure: map[bool]string{false: "OFF", true: "ON"}[pressure]}
	s := modelengine.NewNativeScheduler(m)
	s.SetMaxRunning(2)
	s.SetResidentQ4K(true)
	s.EnableQwen35MetalStateIdentityCapture()
	var profilerMu sync.Mutex
	var profilers []qwen38PagedSwapProfiler
	s.SetSessionProfilerFactory(func(lifecycle modelengine.NativeSessionLifecycle) *model.PhaseProfiler {
		profiler := model.NewPhaseProfiler()
		profilerMu.Lock()
		profilers = append(profilers, qwen38PagedSwapProfiler{lifecycle: lifecycle, profiler: profiler})
		profilerMu.Unlock()
		return profiler
	})
	ledgerDir, err := os.MkdirTemp("", "fak-qwen38-paged-swap-")
	if err != nil {
		return arm, err
	}
	defer os.RemoveAll(ledgerDir)
	policy := modelengine.NativePreemptionPolicy{
		Mode: modelengine.NativePreemptSwap, BlockTokens: 16,
		VictimRule: modelengine.NativePreemptVictimMostRecent,
	}
	if pressure {
		policy.MaxBlocks = maxBlocks
		policy.UsageLedgerPath = filepath.Join(ledgerDir, "swap.jsonl")
	}
	s.SetKVPreemptionPolicy(policy)
	arm.KVMaxBlocks = policy.MaxBlocks

	type admission struct {
		name    string
		arrival time.Time
		req     abi.EngineRequest
	}
	admittedRequests := make([]admission, 0, len(prompts))
	started := time.Now()
	for i, prompt := range prompts {
		name := fmt.Sprintf("request-%d", i+1)
		arrival := time.Now()
		req, err := s.AdmitTokenIDs(context.Background(), name, prompt)
		if err != nil {
			s.Close()
			return arm, err
		}
		admittedRequests = append(admittedRequests, admission{name: name, arrival: arrival, req: req})
	}

	type completed struct {
		request qwen38PagedSwapRequest
		err     error
	}
	done := make(chan completed, len(admittedRequests))
	for _, admitted := range admittedRequests {
		go func(admitted admission) {
			var ids []int
			var times []time.Time
			for token := range admitted.req.Tokens() {
				ids = append(ids, token.ID)
				times = append(times, time.Now())
			}
			result, err := admitted.req.Result()
			if err != nil {
				done <- completed{err: err}
				return
			}
			outputSHA, err := sha256JSON(result.Payload)
			if err != nil {
				done <- completed{err: err}
				return
			}
			request := qwen38PagedSwapRequest{Request: admitted.name, TokenIDs: ids, OutputSHA256: outputSHA}
			if len(times) > 0 {
				request.TTFTMilliseconds = float64(times[0].Sub(admitted.arrival).Nanoseconds()) / 1e6
				for i := 1; i < len(times); i++ {
					request.ITLMilliseconds = append(request.ITLMilliseconds, float64(times[i].Sub(times[i-1]).Nanoseconds())/1e6)
				}
			}
			done <- completed{request: request}
		}(admitted)
	}
	for range admittedRequests {
		completed := <-done
		if completed.err != nil {
			s.Close()
			return arm, completed.err
		}
		arm.Requests = append(arm.Requests, completed.request)
	}
	sort.Slice(arm.Requests, func(i, j int) bool { return arm.Requests[i].Request < arm.Requests[j].Request })
	elapsed := time.Since(started)
	stats := s.KVPreemptionStats()
	arm.SwapTotal = stats.SwapPreemptions
	arm.ReadmittedTotal = stats.Readmitted
	arm.SwapBytes = stats.SwapBytes
	arm.RestoredBytes = stats.SwapRestoredBytes
	arm.RecomputeTotal = stats.RecomputeCount
	arm.PeakRunning = s.MaxObservedRunning()
	arm.PeakUsedBlocks = s.MaxObservedKVBlocks()
	arm.StateIdentities = s.Qwen35MetalStateIdentityReceipts()
	arm.PeakRSSBytes, err = peakRSSBytes()
	if err != nil {
		s.Close()
		return arm, err
	}
	teardownStarted := time.Now()
	teardownCtx, cancelTeardown := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelTeardown()
	if err := s.CloseAndWait(teardownCtx); err != nil {
		return arm, fmt.Errorf("scheduler teardown: %w", err)
	}
	arm.TeardownMilliseconds = float64(time.Since(teardownStarted).Nanoseconds()) / 1e6
	arm.TeardownComplete = true

	profilerMu.Lock()
	captured := append([]qwen38PagedSwapProfiler(nil), profilers...)
	profilerMu.Unlock()
	for _, item := range captured {
		execution, err := item.profiler.MetalExecutionReceipt()
		if err != nil {
			return arm, err
		}
		if err := metalgemm.ValidateExecutionReceipt(execution); err != nil {
			return arm, err
		}
		fallbacks, err := item.profiler.MetalFallbackReceipt()
		if err != nil {
			return arm, err
		}
		if err := model.ValidateMetalFallbackReceipt(fallbacks); err != nil {
			return arm, err
		}
		arm.FallbackTotal += fallbacks.PromisedCPUFallbacks
		arm.SessionExecutions = append(arm.SessionExecutions, qwen38PagedSwapSessionExecution{
			Lifecycle: item.lifecycle, Execution: execution, Fallbacks: fallbacks,
		})
	}
	if pressure {
		arm.SwapUsage, err = modelperfobs.FoldQwenSwapUsage(policy.UsageLedgerPath)
		if err != nil {
			return arm, err
		}
		for _, usage := range arm.SwapUsage {
			arm.ErrorTotal += usage.Errors
			arm.RefusalTotal += usage.Refused
		}
	}
	var ttft, itl []float64
	totalTokens := 0
	for _, request := range arm.Requests {
		ttft = append(ttft, request.TTFTMilliseconds)
		itl = append(itl, request.ITLMilliseconds...)
		totalTokens += len(request.TokenIDs)
	}
	arm.TTFTP50Milliseconds, arm.TTFTP95Milliseconds = qwen38PagedSwapPercentiles(ttft)
	arm.ITLP50Milliseconds, arm.ITLP95Milliseconds = qwen38PagedSwapPercentiles(itl)
	arm.AggregateTokensPerSecond = float64(totalTokens) / elapsed.Seconds()
	return arm, nil
}

func qwen38PagedSwapPercentiles(values []float64) (p50, p95 float64) {
	if len(values) == 0 {
		return 0, 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	at := func(p float64) float64 {
		index := int(float64(len(sorted)-1)*p + 0.5)
		return sorted[index]
	}
	return at(.50), at(.95)
}

func qwen38PagedSwapBinding(receipt qwen38PagedSwapReceipt) (string, error) {
	receipt.BindingSHA256 = ""
	return sha256JSON(receipt)
}

func validateQwen38PagedSwapReceipt(receipt qwen38PagedSwapReceipt) error {
	if receipt.Schema != qwen38PagedSwapSchema || receipt.Verdict != "KEEP" ||
		receipt.Engine != qwen38PagedSwapEngine || receipt.Backend != qwen38PagedSwapBackend ||
		receipt.ForwardPath != qwen38PagedSwapForwardPath {
		return fmt.Errorf("invalid qwen38 paged-swap identity")
	}
	envelope, err := exactMetalProfileEnvelope(nativeperf.ActiveGraph(), receipt.Artifact.nativeFileIdentity)
	if err != nil || receipt.Artifact.Model != envelope.Model || receipt.Artifact.ModelRevision != envelope.ModelRevision {
		return fmt.Errorf("artifact identity does not match exact Metal envelope")
	}
	if err := validateNativeHost(envelope, receipt.Host); err != nil {
		return err
	}
	configSHA, err := sha256JSON(receipt.ModelConfig)
	if err != nil || configSHA != receipt.ModelConfigSHA256 {
		return fmt.Errorf("model config digest mismatch")
	}
	if receipt.Source.Revision == "" || receipt.Binary.Bytes <= 0 || len(receipt.Binary.SHA256) != sha256.Size*2 {
		return fmt.Errorf("source or binary identity incomplete")
	}
	if receipt.Source.Modified && (receipt.Source.DiffBytes <= 0 || len(receipt.Source.DiffSHA256) != sha256.Size*2) {
		return fmt.Errorf("modified source lacks an exact diff identity")
	}
	if !receipt.Source.Modified && (receipt.Source.DiffBytes != 0 || receipt.Source.DiffSHA256 != "") {
		return fmt.Errorf("clean source carries a contradictory diff identity")
	}
	if err := validateQwen38PagedSwapControls(receipt.Controls); err != nil {
		return err
	}
	controlMaxBlocks, err := strconv.Atoi(receipt.Controls["FAK_NATIVE_KV_MAX_BLOCKS"])
	if err != nil {
		return fmt.Errorf("invalid paged-swap max-block control")
	}
	if len(receipt.ArrivalTrace) != 2 || len(receipt.Repetitions) != qwen38PagedSwapRepetitions {
		return fmt.Errorf("arrival trace or raw repetition count mismatch")
	}
	for i, arrival := range receipt.ArrivalTrace {
		if arrival.Request != fmt.Sprintf("request-%d", i+1) || arrival.Ordinal != i+1 || arrival.Tokens != 32 || len(arrival.PromptSHA) != sha256.Size*2 {
			return fmt.Errorf("arrival trace entry %d is incomplete", i+1)
		}
	}
	for i, pair := range receipt.Repetitions {
		if pair.Repetition != i+1 {
			return fmt.Errorf("repetition ordinal=%d, want %d", pair.Repetition, i+1)
		}
		if pair.Off.KVMaxBlocks != 0 || pair.On.KVMaxBlocks != controlMaxBlocks {
			return fmt.Errorf("repetition %d max-block control was not applied", pair.Repetition)
		}
		if err := validateQwen38PagedSwapPair(pair); err != nil {
			return fmt.Errorf("repetition %d: %w", pair.Repetition, err)
		}
	}
	binding, err := qwen38PagedSwapBinding(receipt)
	if err != nil || binding != receipt.BindingSHA256 {
		return fmt.Errorf("qwen38 paged-swap binding digest mismatch")
	}
	return nil
}

func validateQwen38PagedSwapPair(pair qwen38PagedSwapPair) error {
	if pair.Off.Pressure != "OFF" || pair.On.Pressure != "ON" {
		return fmt.Errorf("pressure labels are not OFF/ON")
	}
	if pair.Off.SwapTotal != 0 || pair.Off.ReadmittedTotal != 0 || pair.Off.SwapBytes != 0 || pair.Off.RestoredBytes != 0 || pair.Off.RecomputeTotal != 0 {
		return fmt.Errorf("OFF arm observed preemption")
	}
	if pair.On.SwapTotal != 1 || pair.On.ReadmittedTotal != 1 || pair.On.SwapBytes <= 0 ||
		pair.On.RestoredBytes != pair.On.SwapBytes || pair.On.RecomputeTotal != 0 {
		return fmt.Errorf("ON arm lacks byte-exact swap/readmission or used recompute")
	}
	if len(pair.On.SwapUsage) == 0 {
		return fmt.Errorf("ON arm lacks durable swap/readmission usage evidence")
	}
	if pair.On.KVMaxBlocks <= 0 || pair.On.PeakUsedBlocks <= pair.On.KVMaxBlocks || pair.On.PeakRunning != 2 {
		return fmt.Errorf("ON arm lacks pressure occupancy evidence")
	}
	for _, arm := range []*qwen38PagedSwapArm{&pair.Off, &pair.On} {
		if arm.FallbackTotal != 0 || arm.ErrorTotal != 0 || arm.RefusalTotal != 0 || !arm.TeardownComplete ||
			arm.TeardownMilliseconds <= 0 || arm.PeakRSSBytes == 0 || arm.AggregateTokensPerSecond <= 0 ||
			arm.TTFTP50Milliseconds <= 0 || arm.ITLP50Milliseconds <= 0 {
			return fmt.Errorf("%s arm has fallback/error/refusal/metric/teardown failure", arm.Pressure)
		}
		wantExecutions := 2
		if arm.Pressure == "ON" {
			wantExecutions = 3
		}
		if len(arm.Requests) != 2 || len(arm.StateIdentities) != 2 || len(arm.SessionExecutions) != wantExecutions {
			return fmt.Errorf("%s arm has incomplete requests/state/execution evidence", arm.Pressure)
		}
		for i, request := range arm.Requests {
			if request.Request != fmt.Sprintf("request-%d", i+1) || len(request.TokenIDs) == 0 || len(request.OutputSHA256) != sha256.Size*2 {
				return fmt.Errorf("%s arm request %d is incomplete", arm.Pressure, i+1)
			}
		}
		restored := 0
		for _, session := range arm.SessionExecutions {
			if session.Lifecycle == modelengine.NativeSessionRestored {
				restored++
			}
			if err := metalgemm.ValidateExecutionReceipt(session.Execution); err != nil {
				return err
			}
			if err := model.ValidateMetalFallbackReceipt(session.Fallbacks); err != nil || session.Fallbacks.PromisedCPUFallbacks != 0 {
				return fmt.Errorf("%s arm invalid fallback receipt", arm.Pressure)
			}
		}
		if arm.Pressure == "ON" && restored != 1 {
			return fmt.Errorf("ON arm has %d restored Metal session executions, want 1", restored)
		}
		if arm.Pressure == "OFF" && restored != 0 {
			return fmt.Errorf("OFF arm unexpectedly restored a session")
		}
		for _, identity := range arm.StateIdentities {
			if err := model.ValidateQwen35MetalStateIdentityReceipt(identity); err != nil {
				return err
			}
		}
	}
	if !qwen38PagedSwapOutputsEqual(pair.Off.Requests, pair.On.Requests) {
		return fmt.Errorf("OFF/ON accepted token IDs or outputs differ")
	}
	if !qwen38PagedSwapStatesEqual(pair.Off.StateIdentities, pair.On.StateIdentities) {
		return fmt.Errorf("OFF/ON Qwen state digests differ")
	}
	var usageSwapOut, usageRestoreIn int
	var usageBytes int64
	for _, usage := range pair.On.SwapUsage {
		if usage.Errors != 0 || usage.Refused != 0 || usage.Succeeded != usage.Invocations {
			return fmt.Errorf("ON arm swap usage is not committed-only")
		}
		usageSwapOut += usage.SwapOut
		usageRestoreIn += usage.RestoreIn
		usageBytes += usage.Bytes
	}
	if int64(usageSwapOut) != pair.On.SwapTotal || int64(usageRestoreIn) != pair.On.ReadmittedTotal ||
		usageBytes != pair.On.SwapBytes+pair.On.RestoredBytes {
		return fmt.Errorf("ON arm swap usage does not bind runtime counters")
	}
	return nil
}

func qwen38PagedSwapOutputsEqual(a, b []qwen38PagedSwapRequest) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Request != b[i].Request || a[i].OutputSHA256 != b[i].OutputSHA256 || !reflect.DeepEqual(a[i].TokenIDs, b[i].TokenIDs) {
			return false
		}
	}
	return true
}

func qwen38PagedSwapStatesEqual(a, b []model.Qwen35MetalStateIdentityReceipt) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Authority != b[i].Authority || a[i].TokenLineageSHA256 != b[i].TokenLineageSHA256 ||
			!reflect.DeepEqual(a[i].States, b[i].States) {
			return false
		}
	}
	return true
}

func runQwen38PagedSwapReadback(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var receipt qwen38PagedSwapReceipt
	if err := decodeExactJSON(data, &receipt); err != nil {
		return err
	}
	return validateQwen38PagedSwapReceipt(receipt)
}
