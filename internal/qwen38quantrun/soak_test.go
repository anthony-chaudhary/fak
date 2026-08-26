package qwen38quantrun

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestIssue8319SoakContractRejectsLessThanThreeFinalists(t *testing.T) {
	corpus := qwen38quant.DefaultCorpus()
	report := SoakReport{
		Schema:            SoakSchema,
		CorpusID:          corpus.ID,
		CorpusSHA256:      qwen38quant.CorpusDigest(corpus),
		Arms:              []SoakArmResult{{Arm: "fp8"}, {Arm: "q4_k_m"}},
		RawArchiveSHA256:  hash64('a'),
		RollbackThreshold: "any selected-arm quality failure",
		Verdict:           "HOLD",
	}
	if err := ValidateSoakReport(report, corpus); err == nil || !strings.Contains(err.Error(), "3 finalists") {
		t.Fatalf("err=%v, want three-finalist refusal", err)
	}
}

func TestIssue8319DefaultSoakCarriesThirtyCodingTasks(t *testing.T) {
	tasks := DefaultSoakTasks()
	if len(tasks) != MinimumCodingTasks {
		t.Fatalf("tasks=%d want=%d", len(tasks), MinimumCodingTasks)
	}
	if err := validateSoakTasks(tasks); err != nil {
		t.Fatal(err)
	}
	drifted := append([]CodingTask(nil), tasks...)
	drifted[0].Prompt += " changed"
	if err := validateSoakTasks(drifted); err == nil || !strings.Contains(err.Error(), "drifted") {
		t.Fatalf("err=%v, want task-drift refusal", err)
	}
}

func TestIssue8319SoakAdapterRefusesInlineSecrets(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "soak.json")
	cfg := SoakAdapterConfig{
		Schema:            SoakSchema,
		RollbackThreshold: "any quality failure",
	}
	for _, arm := range []string{"bf16", "fp8", "q4_k_m"} {
		cfg.Finalists = append(cfg.Finalists, SoakAdapterArm{
			Campaign: AdapterConfig{
				Arm: arm, ExecutionEngine: qwen38quant.EngineFakNative, Endpoint: EndpointConfig{APIKey: "inline-secret"},
				ObservationCommand: []string{"observe"}, RestartCommand: []string{"restart"},
				ReadyCommand: []string{"ready"}, CleanupCommand: []string{"cleanup"},
			},
			APIKeyEnv: "QWEN38_TEST_KEY",
		})
	}
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	err = RunSoakAdapter(context.Background(), configPath, filepath.Join("..", "..", "docs", "benchmarks", "qwen38-quant", "corpus.json"), filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json"))
	if err == nil || !strings.Contains(err.Error(), "inline API keys are refused") {
		t.Fatalf("err=%v, want inline-secret refusal", err)
	}
}

func TestDecodeWindowsSoakAggregation(t *testing.T) {
	report, err := BuildDecodeWindows(nativeLinearTrace(MinimumLongDecodeTokens, 10), NativeDecodeContract, MinimumLongDecodeTokens)
	if err != nil {
		t.Fatal(err)
	}
	results := make([]Result, 3)
	for i := range results {
		trace := nativeLinearTrace(MinimumLongDecodeTokens, 10)
		window := report
		window.Windows = append([]DecodeWindow(nil), report.Windows...)
		results[i] = Result{FixtureID: "long-output", Repeat: i + 1, Usage: map[string]int{"completion_tokens": MinimumLongDecodeTokens}, Quality: "PASS", DecodeTrace: &trace, DecodeWindows: &window}
	}
	metrics := summarizeMetrics(nil, results, true)
	if metrics.DecodeWindows == nil || metrics.DecodeWindows.Verdict != "PASS" || metrics.DecodeWindows.Confidence.Repetitions != 3 {
		t.Fatalf("metrics=%+v", metrics)
	}
	results[1].DecodeTrace.Events[1].TokenIndex = 1
	metrics = summarizeMetrics(nil, results, true)
	if metrics.DecodeWindows == nil || metrics.DecodeWindows.Verdict != "HOLD" || !strings.Contains(metrics.DecodeWindows.Failure, "duplicate") {
		t.Fatalf("tampered metrics=%+v", metrics)
	}
	if got := deriveArmVerdict(SoakArmResult{Campaign: qwen38quant.Report{Verdict: "PROMOTE"}, Metrics: metrics}); got != "HOLD" {
		t.Fatalf("verdict=%s want HOLD", got)
	}
}

func TestDecodeWindowsSoakRefusesComparatorNativeReceipt(t *testing.T) {
	_, _, err := (Runner{}).RunSoakArm(context.Background(), SoakArmConfig{Campaign: CampaignConfig{
		Endpoint: Config{NativeDecodeTrace: true}, ExecutionEngine: qwen38quant.EngineLlamaCpp,
		Arm: "llama.cpp", RollbackThreshold: "late/early below 0.85", Probe: &staticProbe{}, Lifecycle: &lifecycleSpy{},
	}}, qwen38quant.DefaultCorpus(), DefaultSoakTasks())
	if err == nil || !strings.Contains(err.Error(), "comparator transport timings are not token commits") {
		t.Fatalf("err=%v", err)
	}
}

func TestRunSoakArmCapturesCodingAndFailureReadbacks(t *testing.T) {
	corpus := frozenCorpus(t)
	tasks := DefaultSoakTasks()
	wants := make(map[string]string, len(tasks))
	for _, task := range tasks {
		wants[task.Prompt] = task.ExpectedExact
	}

	nativeEvents := make([]map[string]any, MinimumLongDecodeTokens)
	for i := range nativeEvents {
		nativeEvents[i] = map[string]any{"token_index": i + 1, "elapsed_ns": int64(i/2+1) * 1_000_000}
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/v1/models" {
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "exact"}}})
			return
		}
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
			Tools          []any `json:"tools"`
			FakDecodeTrace bool  `json:"fak_decode_trace"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, "bad JSON", http.StatusBadRequest)
			return
		}
		prompt := body.Messages[0].Content
		if strings.Contains(prompt, "prime numbers below one million") {
			<-req.Context().Done()
			return
		}
		if body.FakDecodeTrace {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model": "exact", "choices": []any{map[string]any{"message": map[string]any{"content": "bounded"}}},
				"usage": map[string]int{"prompt_tokens": 9, "completion_tokens": MinimumLongDecodeTokens},
				"fak":   map[string]any{"decode_trace": map[string]any{"schema": NativeDecodeTraceSchema, "engine": NativeDecodeTraceEngine, "events": nativeEvents}},
			})
			return
		}
		message := map[string]any{"content": wants[prompt]}
		switch {
		case len(body.Tools) > 0:
			message["content"] = ""
			message["tool_calls"] = []any{map[string]any{"function": map[string]any{"name": "lookup_ticket", "arguments": `{"ticket_id":314}`}}}
		case strings.Contains(prompt, "release record"):
			message["content"] = `{"version":3,"ready":true}`
		case strings.Contains(prompt, "fib(0)=0") && strings.Contains(prompt, "n=10"):
			message["content"] = "55"
		case strings.Contains(prompt, "record 1537"):
			message["content"] = "ORCHID-7319"
		case strings.Contains(prompt, "CACHE-2048"):
			message["content"] = "CACHE-2048"
		case strings.Contains(prompt, "exactly Q38"):
			message["content"] = "Q38"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "exact",
			"choices": []any{map[string]any{"message": message}},
			"usage":   map[string]int{"prompt_tokens": 9, "completion_tokens": 2},
		})
	}))
	defer server.Close()

	identity := qwen38quant.Identity{
		Model: "Qwen3.8-27B", CheckpointSHA256: hash64('a'), ArtifactSHA256: hash64('b'),
		TokenizerSHA256: hash64('c'), TemplateSHA256: hash64('d'), QuantizerRevision: "quant-r1",
		RuntimeRevision: "runtime-r1", FakModuleRev: "internal/qwen38quantrun@r1+gabcdef0",
	}
	observation := Observation{
		Identity: identity, Hardware: "A100 40GB", Software: "runtime-r1", Device: "NVIDIA A100",
		ContextTokens: 16384, CacheMode: "prefix", Resident: true, MemoryBytes: 24 << 30, PowerWatts: 217,
	}
	probe, lifecycle := &staticProbe{observation: observation}, &lifecycleSpy{}
	comparator := makeLlamaDecodeResults("long-output", MinimumLongDecodeTokens)
	arm, raw, err := (Runner{Client: server.Client()}).RunSoakArm(context.Background(), SoakArmConfig{
		Campaign: CampaignConfig{
			Endpoint: Config{Endpoint: server.URL, APIKey: "secret", Model: "exact"}, ExecutionEngine: qwen38quant.EngineFakNative, Arm: "q4_k_m",
			Expected: identity, Command: []string{"fak", "serve"}, RequireDevice: "A100",
			StaleAfter: "2026-09-21", RollbackThreshold: "any quality failure",
			Probe: probe, Lifecycle: lifecycle,
		},
		CancellationAfter: 5 * time.Millisecond,
		LongDecode: &LongDecodeCampaignConfig{
			Fixture:    qwen38quant.Fixture{ID: "long-output", Workload: LongDecodeWorkload, Prompt: "continue to the output limit", MaxOutputTokens: MinimumLongDecodeTokens},
			Comparator: comparator,
		},
	}, corpus, tasks)
	if err != nil {
		t.Fatal(err)
	}
	if len(arm.Coding) != MinimumCodingTasks || len(arm.Scenarios) != len(requiredSoakScenarios) {
		t.Fatalf("coding=%d scenarios=%d", len(arm.Coding), len(arm.Scenarios))
	}
	if arm.Verdict != "PROMOTE" || len(raw) == 0 || strings.Contains(string(raw), "secret") {
		t.Fatalf("verdict=%s campaign=%s coding=%#v scenarios=%#v raw=%d secret=%t", arm.Verdict, arm.Campaign.Verdict, arm.Coding, arm.Scenarios, len(raw), strings.Contains(string(raw), "secret"))
	}
	if lifecycle.restarts != 1 || lifecycle.ready != 2 || lifecycle.cleanup != 1 {
		t.Fatalf("lifecycle=%+v", lifecycle)
	}
	if arm.Metrics.CodingThroughput <= 0 || arm.Metrics.PeakMemoryBytes == 0 || arm.Metrics.PeakPowerWatts == 0 {
		t.Fatalf("metrics=%+v", arm.Metrics)
	}
	if arm.Metrics.DecodeWindows == nil || arm.Metrics.DecodeWindows.Verdict != "PASS" || arm.MatchedDecode == nil || arm.MatchedDecode.Verdict != "PASS" {
		t.Fatalf("decode evidence metrics=%+v matched=%+v", arm.Metrics.DecodeWindows, arm.MatchedDecode)
	}
	var liveDecodeArchive soakArmArchive
	if err := json.Unmarshal(raw, &liveDecodeArchive); err != nil {
		t.Fatal(err)
	}
	if liveDecodeArchive.Decode == nil || len(liveDecodeArchive.Decode.Native) != MinimumDecodeRepetitions || len(liveDecodeArchive.Decode.Comparator) != MinimumDecodeRepetitions || len(liveDecodeArchive.Decode.Comparator[0].Events) != MinimumLongDecodeTokens {
		t.Fatalf("raw matched decode archive=%+v", liveDecodeArchive.Decode)
	}

	arms := []SoakArmResult{arm, cloneSoakArm(arm, "fp8"), cloneSoakArm(arm, "q5_k_m")}
	report := AssembleSoakReport(arms, corpus, "q4_k_m", "any selected-arm quality failure")
	armArchives := make([]json.RawMessage, 0, len(arms))
	armArchives = append(armArchives, json.RawMessage(raw))
	for _, cloned := range arms[1:] {
		var decoded soakArmArchive
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatal(err)
		}
		decoded.Arm = cloned.Arm
		var campaign Archive
		if err := json.Unmarshal(decoded.Campaign, &campaign); err != nil {
			t.Fatal(err)
		}
		campaign.Arm = cloned.Arm
		campaignRaw, err := canonicalJSON(campaign)
		if err != nil {
			t.Fatal(err)
		}
		campaignSum := sha256.Sum256(campaignRaw)
		cloned.CampaignArchiveSHA256 = hex.EncodeToString(campaignSum[:])
		decoded.Campaign = json.RawMessage(campaignRaw)
		encoded, err := canonicalJSON(decoded)
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(encoded)
		cloned.ArchiveSHA256 = hex.EncodeToString(sum[:])
		report.Arms[len(armArchives)] = cloned
		armArchives = append(armArchives, json.RawMessage(encoded))
	}
	combined, err := canonicalJSON(soakArchive{Schema: SoakArchiveSchema, Arms: armArchives})
	if err != nil {
		t.Fatal(err)
	}
	combinedSum := sha256.Sum256(combined)
	report.RawArchiveSHA256 = hex.EncodeToString(combinedSum[:])
	if err := ValidateSoakArtifacts(report, combined, corpus); err != nil {
		t.Fatal(err)
	}
	recomputedTamperReport := report
	recomputedTamperReport.Arms = append([]SoakArmResult(nil), report.Arms...)
	recomputedTamperArchives := append([]json.RawMessage(nil), armArchives...)
	var recomputedTamper soakArmArchive
	if err := json.Unmarshal(recomputedTamperArchives[0], &recomputedTamper); err != nil {
		t.Fatal(err)
	}
	for i := 700; i < len(recomputedTamper.Decode.Comparator[0].Events); i++ {
		recomputedTamper.Decode.Comparator[0].Events[i].ElapsedNS += 1_000
	}
	recomputedTamperRaw, err := canonicalJSON(recomputedTamper)
	if err != nil {
		t.Fatal(err)
	}
	recomputedTamperArchives[0] = json.RawMessage(recomputedTamperRaw)
	recomputedTamperArmSum := sha256.Sum256(recomputedTamperRaw)
	recomputedTamperReport.Arms[0].ArchiveSHA256 = hex.EncodeToString(recomputedTamperArmSum[:])
	recomputedTamperCombined, err := canonicalJSON(soakArchive{Schema: SoakArchiveSchema, Arms: recomputedTamperArchives})
	if err != nil {
		t.Fatal(err)
	}
	recomputedTamperCombinedSum := sha256.Sum256(recomputedTamperCombined)
	recomputedTamperReport.RawArchiveSHA256 = hex.EncodeToString(recomputedTamperCombinedSum[:])
	if err := ValidateSoakArtifacts(recomputedTamperReport, recomputedTamperCombined, corpus); err == nil || !strings.Contains(err.Error(), "matched decode archive mismatch") {
		t.Fatalf("recomputed comparator tamper err=%v", err)
	}
	legacyReport := report
	legacyReport.Arms = append([]SoakArmResult(nil), report.Arms...)
	legacyArchives := make([]json.RawMessage, 0, len(armArchives))
	for i, encoded := range armArchives {
		var legacy soakArmArchive
		if err := json.Unmarshal(encoded, &legacy); err != nil {
			t.Fatal(err)
		}
		legacy.Decode = nil
		legacyRaw, err := canonicalJSON(legacy)
		if err != nil {
			t.Fatal(err)
		}
		legacyArchives = append(legacyArchives, json.RawMessage(legacyRaw))
		legacyReport.Arms[i].Metrics.DecodeWindows = nil
		legacyReport.Arms[i].MatchedDecode = nil
		legacyReport.Arms[i].Verdict = deriveArmVerdict(legacyReport.Arms[i])
		legacySum := sha256.Sum256(legacyRaw)
		legacyReport.Arms[i].ArchiveSHA256 = hex.EncodeToString(legacySum[:])
	}
	legacyCombined, err := canonicalJSON(soakArchive{Schema: SoakArchiveSchema, Arms: legacyArchives})
	if err != nil {
		t.Fatal(err)
	}
	legacyCombinedSum := sha256.Sum256(legacyCombined)
	legacyReport.RawArchiveSHA256 = hex.EncodeToString(legacyCombinedSum[:])
	if err := ValidateSoakArtifacts(legacyReport, legacyCombined, corpus); err != nil {
		t.Fatalf("legacy archive roundtrip: %v", err)
	}
	zeroThroughput := report
	zeroThroughput.Arms = append([]SoakArmResult(nil), report.Arms...)
	zeroThroughput.Arms[0].Metrics.CodingThroughput = 0
	zeroThroughput.Arms[0].Verdict = "HOLD"
	zeroThroughput.Arms[0].Campaign.Verdict = "HOLD"
	zeroThroughput.Verdict = "HOLD"
	if err := ValidateSoakReport(zeroThroughput, corpus); err != nil {
		t.Fatalf("a measured zero throughput on a failing arm is valid: %v", err)
	}
	tampered := append([]byte(nil), combined...)
	tampered[len(tampered)-2] ^= 1
	if err := ValidateSoakArtifacts(report, tampered, corpus); err == nil || !strings.Contains(err.Error(), "hash mismatch") {
		t.Fatalf("err=%v, want archive-hash refusal", err)
	}
	report.Arms[0].Scenarios = report.Arms[0].Scenarios[:len(report.Arms[0].Scenarios)-1]
	if err := ValidateSoakReport(report, corpus); err == nil || !strings.Contains(err.Error(), "scenario") {
		t.Fatalf("err=%v, want scenario refusal", err)
	}
}

func frozenCorpus(t *testing.T) qwen38quant.Corpus {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "docs", "benchmarks", "qwen38-quant", "corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	corpus, err := qwen38quant.DecodeCorpus(raw)
	if err != nil {
		t.Fatal(err)
	}
	return corpus
}

func cloneSoakArm(src SoakArmResult, arm string) SoakArmResult {
	clone := src
	clone.Arm = arm
	clone.Campaign.Arm = arm
	return clone
}

func makeLlamaDecodeResults(fixtureID string, tokens int) []LlamaClientDecodeResult {
	results := make([]LlamaClientDecodeResult, MinimumDecodeRepetitions)
	for repetition := range results {
		events := make([]LlamaClientArrivalEvent, tokens)
		for i := range events {
			events[i] = LlamaClientArrivalEvent{TokenIDs: []int{10_000 + i}, TokensPredicted: i + 1, ElapsedNS: 250_000_000 + int64(i/2+1)*1_000_000}
		}
		results[repetition] = LlamaClientDecodeResult{FixtureID: fixtureID, Repeat: repetition + 1, Events: events, Final: LlamaClientFinal{StopType: "limit", TokensPredicted: tokens}}
	}
	return results
}
