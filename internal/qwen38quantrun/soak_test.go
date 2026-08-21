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
				Arm: arm, Endpoint: EndpointConfig{APIKey: "inline-secret"},
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

func TestRunSoakArmCapturesCodingAndFailureReadbacks(t *testing.T) {
	corpus := frozenCorpus(t)
	tasks := DefaultSoakTasks()
	wants := make(map[string]string, len(tasks))
	for _, task := range tasks {
		wants[task.Prompt] = task.ExpectedExact
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
			Tools []any `json:"tools"`
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
	arm, raw, err := (Runner{Client: server.Client()}).RunSoakArm(context.Background(), SoakArmConfig{
		Campaign: CampaignConfig{
			Endpoint: Config{Endpoint: server.URL, APIKey: "secret", Model: "exact"}, Arm: "q4_k_m",
			Expected: identity, Command: []string{"fak", "serve"}, RequireDevice: "A100",
			StaleAfter: "2026-09-21", RollbackThreshold: "any quality failure",
			Probe: probe, Lifecycle: lifecycle,
		},
		CancellationAfter: 5 * time.Millisecond,
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
