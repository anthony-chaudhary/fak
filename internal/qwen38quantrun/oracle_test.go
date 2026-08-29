package qwen38quantrun

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestRunOracleBindsQualityBeforePerformance(t *testing.T) {
	dir := t.TempDir()
	corpus := qwen38quant.DefaultCorpus()
	corpusPath := filepath.Join(dir, "corpus.json")
	writeJSONTest(t, corpusPath, corpus)
	ref := writeOracleRuntimeFixture(t, dir, "llama.cpp", corpus, 0)
	candidate := writeOracleRuntimeFixture(t, dir, "fak", corpus, 0.0001)
	cfg := validOracleConfig(ref, candidate)
	configPath := filepath.Join(dir, "oracle.json")
	writeJSONTest(t, configPath, cfg)
	reportPath, archivePath := filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json")

	report, err := RunOracle(context.Background(), configPath, corpusPath, reportPath, archivePath)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "HOLD" || len(report.NumericQuality) != len(corpus.Fixtures)*2 || len(report.Performance) != 4 {
		t.Fatalf("verdict=%s quality=%d performance=%d", report.Verdict, len(report.NumericQuality), len(report.Performance))
	}
	raw, err := os.ReadFile(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateOracleArtifacts(report, raw, corpus); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "adapter-secret") {
		t.Fatal("raw oracle archive retained the adapter secret")
	}
}

func TestRunOracleBindsMatchedP32T64Receipt(t *testing.T) {
	dir := t.TempDir()
	corpus := qwen38quant.DefaultCorpus()
	corpusPath := filepath.Join(dir, "corpus.json")
	writeJSONTest(t, corpusPath, corpus)
	ref := writeOracleRuntimeFixture(t, dir, "llama.cpp", corpus, 0)
	candidate := writeOracleRuntimeFixture(t, dir, "fak", corpus, 0.0001)
	writeMatchedMeasurementFixture(t, ref.measurementPath, oracleRuntimeReference)
	writeMatchedMeasurementFixture(t, candidate.measurementPath, oracleRuntimeCandidate)
	cfg := validOracleConfig(ref, candidate)
	configPath := filepath.Join(dir, "oracle.json")
	writeJSONTest(t, configPath, cfg)

	report, err := RunOracle(context.Background(), configPath, corpusPath, filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json"))
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "HOLD" {
		t.Fatalf("verdict=%q want comparator-bound HOLD", report.Verdict)
	}
}

func TestRunOracleRefusesInvalidMatchedP32T64Receipt(t *testing.T) {
	tests := []struct {
		name   string
		target string
		mutate func(*OracleMeasurementRun)
	}{
		{name: "missing repetition", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions = run.Matched.Repetitions[:2]
		}},
		{name: "duplicate repetition", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions[2].Repetition = 2
		}},
		{name: "out of range repetition", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions[0].Repetition = 0
		}},
		{name: "prefill shape drift", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.PrefillTokens = 31
		}},
		{name: "decode shape drift", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.DecodeTokens = 63
		}},
		{name: "temperature drift", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Temperature = 0.1
		}},
		{name: "engine tamper", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Execution.Engine = "llama.cpp"
		}},
		{name: "backend tamper", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Execution.Backend = "cpu"
		}},
		{name: "forward path tamper", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Execution.ForwardPath = "cpu/reference"
		}},
		{name: "q4k tamper", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Execution.Q4K = false
		}},
		{name: "fallback active", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Execution.FallbackActive = oracleBool(true)
		}},
		{name: "fallback state omitted", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Execution.FallbackActive = nil
		}},
		{name: "candidate marked comparator", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Execution.ComparatorOnly = oracleBool(true)
		}},
		{name: "reference not comparator", target: oracleRuntimeReference, mutate: func(run *OracleMeasurementRun) {
			run.Execution.ComparatorOnly = oracleBool(false)
		}},
		{name: "comparator state omitted", target: oracleRuntimeReference, mutate: func(run *OracleMeasurementRun) {
			run.Execution.ComparatorOnly = nil
		}},
		{name: "missing RSS", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions[0].RSSBytes = 0
		}},
		{name: "missing OS footprint", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions[0].OSFootprintBytes = 0
		}},
		{name: "generated token tamper", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions[0].Tokens[0] = "drift"
		}},
		{name: "empty generated token", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions[0].Tokens[0] = ""
		}},
		{name: "nondeterministic generated tokens", target: "both", mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions[1].Tokens[0] = "other"
		}},
		{name: "logit tamper", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.Repetitions[0].Logits[0] += 1
		}},
		{name: "prompt hash tamper", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Matched.PromptSHA256 = strings.Repeat("d", 64)
		}},
		{name: "mixed schema versions", target: oracleRuntimeCandidate, mutate: func(run *OracleMeasurementRun) {
			run.Schema = OracleMeasurementSchema
			run.Execution = nil
			run.Matched = nil
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			corpus := qwen38quant.DefaultCorpus()
			corpusPath := filepath.Join(dir, "corpus.json")
			writeJSONTest(t, corpusPath, corpus)
			ref := writeOracleRuntimeFixture(t, dir, "llama.cpp", corpus, 0)
			candidate := writeOracleRuntimeFixture(t, dir, "fak", corpus, 0)
			writeMatchedMeasurementFixture(t, ref.measurementPath, oracleRuntimeReference)
			writeMatchedMeasurementFixture(t, candidate.measurementPath, oracleRuntimeCandidate)
			targets := []string{candidate.measurementPath}
			switch tt.target {
			case oracleRuntimeReference:
				targets = []string{ref.measurementPath}
			case "both":
				targets = []string{ref.measurementPath, candidate.measurementPath}
			}
			for _, target := range targets {
				var measurement OracleMeasurementRun
				readJSONTest(t, target, &measurement)
				tt.mutate(&measurement)
				writeJSONTest(t, target, measurement)
			}
			cfg := validOracleConfig(ref, candidate)
			configPath := filepath.Join(dir, "oracle.json")
			writeJSONTest(t, configPath, cfg)
			reportPath, archivePath := filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json")

			if _, err := RunOracle(context.Background(), configPath, corpusPath, reportPath, archivePath); err == nil {
				t.Fatal("invalid matched receipt was accepted")
			}
			for _, path := range []string{reportPath, archivePath} {
				if _, err := os.Stat(path); !os.IsNotExist(err) {
					t.Fatalf("%s exists after refusal", path)
				}
			}
		})
	}
}

func TestRunOracleHoldsPerformanceOnLogitDrift(t *testing.T) {
	dir := t.TempDir()
	corpus := qwen38quant.DefaultCorpus()
	corpusPath := filepath.Join(dir, "corpus.json")
	writeJSONTest(t, corpusPath, corpus)
	ref := writeOracleRuntimeFixture(t, dir, "llama.cpp", corpus, 0)
	candidate := writeOracleRuntimeFixture(t, dir, "fak", corpus, 0.01)
	cfg := validOracleConfig(ref, candidate)
	configPath := filepath.Join(dir, "oracle.json")
	writeJSONTest(t, configPath, cfg)

	report, err := RunOracle(context.Background(), configPath, corpusPath, filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json"))
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "HOLD" || report.NumericQuality[0].Pass {
		t.Fatalf("verdict=%s first=%+v", report.Verdict, report.NumericQuality[0])
	}
}

func TestRunOracleRefusesMissingHardwareEvidence(t *testing.T) {
	dir := t.TempDir()
	corpus := qwen38quant.DefaultCorpus()
	corpusPath := filepath.Join(dir, "corpus.json")
	writeJSONTest(t, corpusPath, corpus)
	ref := writeOracleRuntimeFixture(t, dir, "llama.cpp", corpus, 0)
	candidate := writeOracleRuntimeFixture(t, dir, "fak", corpus, 0)
	var measurement OracleMeasurementRun
	readJSONTest(t, candidate.measurementPath, &measurement)
	measurement.Samples[0].VRAMBytes = 0
	writeJSONTest(t, candidate.measurementPath, measurement)
	cfg := validOracleConfig(ref, candidate)
	configPath := filepath.Join(dir, "oracle.json")
	writeJSONTest(t, configPath, cfg)
	reportPath, archivePath := filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json")

	_, err := RunOracle(context.Background(), configPath, corpusPath, reportPath, archivePath)
	if err == nil || !strings.Contains(err.Error(), "RSS, or VRAM") {
		t.Fatalf("err=%v", err)
	}
	for _, path := range []string{reportPath, archivePath} {
		if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
			t.Fatalf("%s exists after refusal", path)
		}
	}
}

func TestRunOracleRefusesRevisionDrift(t *testing.T) {
	dir := t.TempDir()
	corpus := qwen38quant.DefaultCorpus()
	corpusPath := filepath.Join(dir, "corpus.json")
	writeJSONTest(t, corpusPath, corpus)
	ref := writeOracleRuntimeFixture(t, dir, "llama.cpp", corpus, 0)
	candidate := writeOracleRuntimeFixture(t, dir, "fak", corpus, 0)
	cfg := validOracleConfig(ref, candidate)
	cfg.RevisionCommand = oracleHelperCommand("revision-drift", "")
	configPath := filepath.Join(dir, "oracle.json")
	writeJSONTest(t, configPath, cfg)
	reportPath, archivePath := filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json")

	_, err := RunOracle(context.Background(), configPath, corpusPath, reportPath, archivePath)
	if err == nil || !strings.Contains(err.Error(), "revision drift") {
		t.Fatalf("err=%v", err)
	}
	for _, path := range []string{reportPath, archivePath} {
		if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
			t.Fatalf("%s exists after revision refusal", path)
		}
	}
}

func TestValidateOracleConfigRefusesPinnedRevisionDrift(t *testing.T) {
	paths := oracleFixturePaths{
		adapterPath: "adapter.json", reportPath: "report.json", archivePath: "archive.json", measurementPath: "measurement.json",
	}
	cfg := validOracleConfig(paths, paths)
	cfg.LlamaCPPRevision = strings.Repeat("b", 40)
	if err := validateOracleConfig(cfg); err == nil || !strings.Contains(err.Error(), "pinned oracle") {
		t.Fatalf("err=%v", err)
	}
}

func TestMatchesPinnedLlamaCPPRevision(t *testing.T) {
	const buildLine = "built with AppleClang 21.0.0.21000101 for Darwin arm64\n"
	tests := []struct {
		name   string
		output string
		want   bool
	}{
		{name: "immutable full revision", output: PinnedLlamaCPPRevision + "\n", want: true},
		{name: "installed b9828 identity", output: "version: 9828 (ebd048fc5)\n" + buildLine, want: true},
		{name: "wrong build", output: "version: 9827 (ebd048fc5)\n" + buildLine},
		{name: "wrong revision", output: "version: 9828 (deadbeef0)\n" + buildLine},
		{name: "dirty revision suffix", output: "version: 9828 (ebd048fc5-dirty)\n" + buildLine},
		{name: "missing build metadata", output: "version: 9828 (ebd048fc5)\n"},
		{name: "identity not first", output: "warning\nversion: 9828 (ebd048fc5)\n" + buildLine},
		{name: "drifted full revision", output: strings.Repeat("b", 40) + "\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := matchesPinnedLlamaCPPRevision([]byte(tt.output)); got != tt.want {
				t.Fatalf("matchesPinnedLlamaCPPRevision(%q)=%v want %v", tt.output, got, tt.want)
			}
		})
	}
}

func TestOracleHelperProcess(t *testing.T) {
	if len(os.Args) < 3 || os.Args[len(os.Args)-3] != "--" {
		command := oracleHelperCommand("revision", "")
		output, err := exec.Command(command[0], command[1:]...).Output()
		if err != nil {
			t.Fatalf("revision helper failed: %v", err)
		}
		wantRevision := PinnedLlamaCPPBuild + " (" + PinnedLlamaCPPRevision[:9] + ")"
		if text := string(output); !strings.Contains(text, wantRevision) || !strings.Contains(text, "Darwin arm64") {
			t.Fatalf("revision output = %q, want revision %q and platform", text, wantRevision)
		}
		return
	}
	mode, path := os.Args[len(os.Args)-2], os.Args[len(os.Args)-1]
	switch mode {
	case "revision":
		os.Stdout.WriteString("version: " + PinnedLlamaCPPBuild + " (" + PinnedLlamaCPPRevision[:9] + ")\n")
		os.Stdout.WriteString("built with AppleClang 21.0.0.21000101 for Darwin arm64\n")
	case "revision-drift":
		os.Stdout.WriteString(strings.Repeat("b", 40) + "\n")
	case "measurement":
		data, err := os.ReadFile(path)
		if err != nil {
			os.Exit(7)
		}
		_, _ = os.Stdout.Write(data)
	default:
		os.Exit(8)
	}
	os.Exit(0)
}

type oracleFixturePaths struct {
	adapterPath, reportPath, archivePath, measurementPath string
}

func writeOracleRuntimeFixture(t *testing.T, dir, name string, corpus qwen38quant.Corpus, logitDelta float64) oracleFixturePaths {
	t.Helper()
	runtimeDir := filepath.Join(dir, strings.ReplaceAll(name, ".", "-"))
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	hash := strings.Repeat("a", 64)
	identity := qwen38quant.Identity{
		Model: "Qwen/Qwen3.8-27B", CheckpointSHA256: hash, ArtifactSHA256: hash, TokenizerSHA256: hash, TemplateSHA256: hash,
		QuantizerRevision: "llama.cpp@" + PinnedLlamaCPPRevision, RuntimeRevision: name + "@test", FakModuleRev: "internal/qwen38quantrun@r1+gtest",
	}
	command := []string{name, "serve", "--model", "Qwen3.8-27B-Q4_K_M.gguf"}
	observation := Observation{Identity: identity, Hardware: "H100-80GB", Software: "CUDA 13", Device: "CUDA", ContextTokens: 32768, CacheMode: "explicit", Resident: true, MemoryBytes: 32 << 30}
	var results []Result
	var trials []qwen38quant.Trial
	for _, fixture := range corpus.Fixtures {
		for repetition := 1; repetition <= corpus.MinimumRepetitions; repetition++ {
			result := Result{FixtureID: fixture.ID, Workload: fixture.Workload, Repeat: repetition, LatencyMS: float64(repetition), Quality: "PASS", Output: fixture.ExpectedExact, Usage: map[string]int{"completion_tokens": 1}}
			results = append(results, result)
			trials = append(trials, qwen38quant.Trial{Workload: fixture.Workload, Repetition: repetition, Quality: "PASS", LatencyMS: float64(repetition), CompletionTokens: 1})
		}
	}
	archive := Archive{Schema: "fak.qwen38-quant-raw/1", CorpusID: corpus.ID, Arm: "q4_k_m", Before: observation, After: observation, Results: results, RestartReady: true, CleanupOK: true}
	archiveRaw, err := canonicalJSON(archive)
	if err != nil {
		t.Fatal(err)
	}
	archiveSum := sha256.Sum256(bytes.TrimSpace(archiveRaw))
	report := qwen38quant.Report{
		Schema: qwen38quant.Schema, ExecutionEngine: qwen38quant.EngineFakNative, CorpusID: corpus.ID, CorpusSHA256: qwen38quant.CorpusDigest(corpus), Arm: "q4_k_m", Identity: identity,
		Environment: qwen38quant.Environment{Command: command, Hardware: observation.Hardware, Software: observation.Software, ContextTokens: observation.ContextTokens, CacheMode: observation.CacheMode, RequireDevice: "CUDA", DenyFallback: true},
		Trials:      trials, Verdict: "PROMOTE", EvidenceClass: "CAMPAIGN", RawArchiveSHA256: hex.EncodeToString(archiveSum[:]), StaleAfter: "2026-11-21", RollbackThreshold: "any exact-effect failure",
	}
	if name == "llama.cpp" {
		report.ExecutionEngine = qwen38quant.EngineLlamaCpp
		report.Verdict = "HOLD"
	}
	adapter := AdapterConfig{
		Endpoint: EndpointConfig{Endpoint: "http://127.0.0.1:9999", APIKey: "adapter-secret", Model: identity.Model}, ExecutionEngine: report.ExecutionEngine, Arm: "q4_k_m", Expected: identity, Command: command,
		RequireDevice: "CUDA", StaleAfter: report.StaleAfter, RollbackThreshold: report.RollbackThreshold,
		ObservationCommand: []string{"observe"}, RestartCommand: []string{"restart"}, ReadyCommand: []string{"ready"}, CleanupCommand: []string{"cleanup"},
	}
	measurement := OracleMeasurementRun{Schema: OracleMeasurementSchema, Runtime: name, CorpusID: corpus.ID, CorpusSHA256: qwen38quant.CorpusDigest(corpus), ArtifactSHA256: hash, Hardware: observation.Hardware, Software: observation.Software}
	for _, fixture := range corpus.Fixtures {
		promptSum := sha256.Sum256([]byte(materialize(fixture)))
		for _, state := range []string{"cold", "warm"} {
			measurement.Samples = append(measurement.Samples, OracleMeasurement{
				FixtureID: fixture.ID, CacheState: state, PromptSHA256: hex.EncodeToString(promptSum[:]), Tokens: []string{"ok"}, Logits: []float64{4 + logitDelta, 2, 1},
				TTFTMS: 10, PrefillTokens: 128, PrefillSeconds: 0.1, PrefillTokensPerSecond: 1280, DecodeTokens: 8, DecodeSeconds: 0.1, DecodeTokensPerSecond: 80,
				RSSBytes: 40 << 30, VRAMBytes: 32 << 30,
			})
		}
	}
	paths := oracleFixturePaths{
		adapterPath: filepath.Join(runtimeDir, "adapter.json"), reportPath: filepath.Join(runtimeDir, "report.json"), archivePath: filepath.Join(runtimeDir, "archive.json"), measurementPath: filepath.Join(runtimeDir, "measurement.json"),
	}
	writeJSONTest(t, paths.adapterPath, adapter)
	writeJSONTest(t, paths.reportPath, report)
	if err := os.WriteFile(paths.archivePath, archiveRaw, 0o600); err != nil {
		t.Fatal(err)
	}
	writeJSONTest(t, paths.measurementPath, measurement)
	return paths
}

func writeMatchedMeasurementFixture(t *testing.T, path, role string) {
	t.Helper()
	var measurement OracleMeasurementRun
	readJSONTest(t, path, &measurement)
	measurement.Schema = OracleMeasurementV2Schema
	measurement.Matched = &OracleMatchedEnvelope{
		PromptSHA256: strings.Repeat("c", 64), Temperature: 0, PrefillTokens: 32, DecodeTokens: 64,
	}
	for repetition := 1; repetition <= 3; repetition++ {
		measurement.Matched.Repetitions = append(measurement.Matched.Repetitions, OracleMatchedRepetition{
			Repetition: repetition, Tokens: slices.Repeat([]string{"tok"}, 64), Logits: []float64{4, 2, 1},
			TTFTMS: 10, PrefillSeconds: 0.1, PrefillTokensPerSecond: 320,
			DecodeSeconds: 1, DecodeTokensPerSecond: 64, RSSBytes: 24 << 30, OSFootprintBytes: 26 << 30,
		})
	}
	switch role {
	case oracleRuntimeReference:
		measurement.Execution = &OracleExecutionIdentity{
			Engine: qwen38quant.EngineLlamaCpp, Backend: oracleMetalBackend, ForwardPath: oracleLlamaForward,
			Q4K: true, FallbackActive: oracleBool(false), ComparatorOnly: oracleBool(true),
		}
	case oracleRuntimeCandidate:
		measurement.Execution = &OracleExecutionIdentity{
			Engine: oracleNativeEngine, Backend: oracleMetalBackend, ForwardPath: oracleNativeForward,
			Q4K: true, FallbackActive: oracleBool(false), ComparatorOnly: oracleBool(false),
		}
	default:
		t.Fatalf("unknown role %q", role)
	}
	writeJSONTest(t, path, measurement)
}

func oracleBool(value bool) *bool { return &value }

func validOracleConfig(ref, candidate oracleFixturePaths) OracleConfig {
	return OracleConfig{
		Schema: OracleConfigSchema, LlamaCPPRevision: PinnedLlamaCPPRevision, LlamaCPPLicense: PinnedLlamaCPPLicense,
		RevisionCommand: oracleHelperCommand("revision", ""), BuildFlags: []string{"-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DLLAMA_BUILD_TESTS=OFF"}, LogitTolerance: 0.001,
		Reference:  OracleRuntimeConfig{Name: "llama.cpp", AdapterConfig: ref.adapterPath, CampaignReport: ref.reportPath, CampaignArchive: ref.archivePath, MeasurementCommand: oracleHelperCommand("measurement", ref.measurementPath)},
		Candidate:  OracleRuntimeConfig{Name: "fak", AdapterConfig: candidate.adapterPath, CampaignReport: candidate.reportPath, CampaignArchive: candidate.archivePath, MeasurementCommand: oracleHelperCommand("measurement", candidate.measurementPath)},
		StaleAfter: "2026-11-21", RollbackThreshold: "any exact-effect, token, or logit failure",
	}
}

func oracleHelperCommand(mode, path string) []string {
	return []string{os.Args[0], "-test.run=^TestOracleHelperProcess$", "--", mode, path}
}

func readJSONTest(t *testing.T, path string, dst any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatal(err)
	}
}
