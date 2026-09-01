package qwen38quantrun

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
	"github.com/anthony-chaudhary/fak/internal/serverlifecycle"
	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

type managedFixture struct {
	dir, baseURL, model string
	generation          uint64
	started             string
}

func newManagedFixture(t *testing.T, baseURL, model string) *managedFixture {
	t.Helper()
	started, ok := serverlifecycle.ProcessIdentity(os.Getpid())
	if !ok {
		t.Skip("current process identity unavailable")
	}
	f := &managedFixture{dir: t.TempDir(), baseURL: baseURL, model: model, generation: 7, started: started}
	f.write(t)
	return f
}

func (f *managedFixture) write(t *testing.T) string {
	t.Helper()
	now := time.Now().UTC().Format(time.RFC3339Nano)
	state := map[string]any{"schema": "fak.server-lifecycle-state/v1", "state": "ready", "instance_id": "managed-test", "generation": f.generation, "process_id": os.Getpid(), "process_start_identity": f.started, "base_url": f.baseURL, "updated_at": now}
	writeJSONTest(t, filepath.Join(f.dir, serverlifecycle.StateFilename), state)
	digest := "sha256:" + strings.Repeat("a", 64)
	receipt := serverproduct.ServerReceipt{
		Schema: serverproduct.SchemaV1, State: serverproduct.ReceiptStateReady,
		Identity: serverproduct.ServerIdentity{ServerName: "managed-server", InstanceID: "managed-test"}, SpecDigest: digest, Generation: f.generation, CreatedAt: now,
		Artifact: serverproduct.ArtifactIdentity{Reference: filepath.Join(f.dir, "model.gguf"), Digest: digest}, Adapter: serverproduct.AdapterIdentity{Name: "llama-server", Version: "test", ExecutableDigest: digest},
		Endpoint: serverproduct.LoopbackEndpoint{BaseURL: f.baseURL}, ModelAlias: f.model, Auth: serverproduct.AuthReference{Mode: serverproduct.AuthNone},
		Protocol:  serverproduct.ProtocolObservation{Family: serverproduct.ProtocolOpenAIHTTP, Revision: "v1", Capabilities: []string{"chat-completions", "models"}},
		Readiness: serverproduct.ReadinessEvidence{Probe: "models", ProbeDigest: digest, ObservedAt: now}, Ownership: serverproduct.OwnershipReference{InstanceID: "managed-test", ProcessID: os.Getpid(), ProcessStartIdentity: f.started},
		Provenance: serverproduct.ReceiptProvenance{Spec: serverproduct.Provenance{Kind: serverproduct.ProvenanceAuthored, Source: "test"}, Artifact: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"}, Adapter: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"}, Endpoint: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"}, Readiness: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"}, Ownership: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"}},
	}
	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(filepath.Join(f.dir, serverlifecycle.ReceiptFilename), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	return fmt.Sprintf("sha256:%x", sum)
}

func managedAdapterConfig(t *testing.T, endpoint, model, marker, observationPath string) AdapterConfig {
	t.Helper()
	h := strings.Repeat("a", 64)
	id := qwen38quant.Identity{Model: "Qwen3.8-27B", CheckpointSHA256: h, ArtifactSHA256: h, TokenizerSHA256: h, TemplateSHA256: h, QuantizerRevision: "quant-r1", RuntimeRevision: "runtime-r1", FakModuleRev: "internal/model@r1+gabcdef0"}
	return AdapterConfig{
		Endpoint: EndpointConfig{Endpoint: "http://127.0.0.1:1", Model: "hand-copied", Repetitions: 3}, ExecutionEngine: qwen38quant.EngineFakNative, Arm: "q4_k_m", Expected: id,
		Command: []string{"fak", "serve"}, RequireDevice: "test-device", StaleAfter: "2026-09-20", RollbackThreshold: "quality pass rate below 100%",
		ObservationCommand: adapterHelperCommand("observation", marker, observationPath), RestartCommand: adapterHelperCommand("managed-restart", marker), ReadyCommand: adapterHelperCommand("ok", ""), CleanupCommand: adapterHelperCommand("effect", marker),
		ManagedServer: &ManagedServerConfig{ProtocolFamily: serverproduct.ProtocolOpenAIHTTP, ProtocolRevision: "v1", Capabilities: []string{"models", "chat-completions"}, ModelAlias: model, BaseURL: endpoint, MinimumGeneration: 7},
	}
}

func TestCheckedInAdapterExamplesSelfcheck(t *testing.T) {
	tests := []struct {
		name   string
		engine string
		arms   []string
	}{
		{"llama.cpp.json", qwen38quant.EngineLlamaCpp, []string{"q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs"}},
		{"vllm.json", qwen38quant.EngineVLLM, []string{"bf16", "fp8", "awq_int4", "gptq_int4"}},
		{"managed-llama.json", qwen38quant.EngineLlamaCpp, []string{"q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join("..", "..", "examples", "qwen38quantrun", test.name)
			cfg, err := SelfcheckAdapterConfig(path)
			if err != nil {
				t.Fatal(err)
			}
			if cfg.ExecutionEngine != test.engine || !slices.Equal(cfg.SupportedArms, test.arms) {
				t.Fatalf("engine/arms = %q/%v", cfg.ExecutionEngine, cfg.SupportedArms)
			}
		})
	}
}

func TestAdapterExampleSelfcheckRejectsDrift(t *testing.T) {
	path := filepath.Join("..", "..", "examples", "qwen38quantrun", "vllm.json")
	good, err := SelfcheckAdapterConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		mutate func(*AdapterConfig)
	}{
		{"arm matrix", func(cfg *AdapterConfig) { cfg.SupportedArms = cfg.SupportedArms[:3] }},
		{"inline secret", func(cfg *AdapterConfig) { cfg.Endpoint.APIKey = "secret" }},
		{"runtime pin", func(cfg *AdapterConfig) { cfg.ReadyCommand[5] = "X-Fak-Runtime-Revision: drift" }},
		{"shell lifecycle", func(cfg *AdapterConfig) {
			cfg.CleanupCommand = []string{"sh", "-c", "docker stop " + cfg.Expected.RuntimeRevision}
		}},
		{"implementation fallback", func(cfg *AdapterConfig) {
			for i := range cfg.Command {
				if cfg.Command[i] == "vllm" && i > 0 && cfg.Command[i-1] == "--model-impl" {
					cfg.Command[i] = "auto"
				}
			}
		}},
		{"residency witness", func(cfg *AdapterConfig) {
			for i := range cfg.ObservationCommand {
				cfg.ObservationCommand[i] = strings.ReplaceAll(cfg.ObservationCommand[i], ".State.Running", "true")
			}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := good
			cfg.SupportedArms = slices.Clone(good.SupportedArms)
			cfg.Command = slices.Clone(good.Command)
			cfg.ObservationCommand = slices.Clone(good.ObservationCommand)
			cfg.RestartCommand = slices.Clone(good.RestartCommand)
			cfg.ReadyCommand = slices.Clone(good.ReadyCommand)
			cfg.CleanupCommand = slices.Clone(good.CleanupCommand)
			test.mutate(&cfg)
			if err := validateMaintainedAdapter(cfg); err == nil {
				t.Fatal("accepted drifted adapter")
			}
		})
	}
}

func TestRunAdapterRequiresRealLifecycleCommands(t *testing.T) {
	dir := t.TempDir()
	cfg := AdapterConfig{ObservationCommand: []string{"probe"}}
	configPath := filepath.Join(dir, "config.json")
	writeJSONTest(t, configPath, cfg)
	corpusPath := filepath.Join(dir, "corpus.json")
	writeJSONTest(t, corpusPath, qwen38quant.DefaultCorpus())
	err := RunAdapter(context.Background(), configPath, corpusPath, filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json"))
	if err == nil || !contains(err.Error(), "restart_command") {
		t.Fatalf("err=%v", err)
	}
}

func TestRunAdapterRejectsInlineAPIKey(t *testing.T) {
	dir := t.TempDir()
	cfg := AdapterConfig{
		Endpoint: EndpointConfig{APIKey: "secret"}, ObservationCommand: []string{"probe"},
		RestartCommand: []string{"restart"}, ReadyCommand: []string{"ready"}, CleanupCommand: []string{"cleanup"},
	}
	configPath := filepath.Join(dir, "config.json")
	writeJSONTest(t, configPath, cfg)
	corpusPath := filepath.Join(dir, "corpus.json")
	writeJSONTest(t, corpusPath, qwen38quant.DefaultCorpus())
	err := RunAdapter(context.Background(), configPath, corpusPath, filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json"))
	if err == nil || !contains(err.Error(), "inline api_key") {
		t.Fatalf("err=%v", err)
	}
}

func TestRunAdapterRejectsMaintainedDriftBeforeEffects(t *testing.T) {
	var requests atomic.Int32
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		http.Error(w, "must not contact endpoint", http.StatusServiceUnavailable)
	}))
	defer endpoint.Close()

	tests := []struct {
		name, example, want string
		mutate              func(*AdapterConfig)
	}{
		{"llama shell lifecycle", "llama.cpp.json", "cleanup_command", func(cfg *AdapterConfig) { cfg.CleanupCommand = []string{"sh", "-c", cfg.Expected.RuntimeRevision} }},
		{"llama image drift", "llama.cpp.json", "llama.cpp command", func(cfg *AdapterConfig) {
			replaceAdapterArg(cfg.Command, "llama.cpp:"+cfg.Expected.RuntimeRevision, "llama.cpp:drift")
		}},
		{"vllm runtime revision", "vllm.json", "runtime_source", func(cfg *AdapterConfig) { cfg.Expected.RuntimeRevision = strings.Repeat("0", 40) }},
		{"vllm inline secret", "vllm.json", "inline api_key", func(cfg *AdapterConfig) { cfg.Endpoint.APIKey = "must-not-leak" }},
		{"vllm implementation fallback", "vllm.json", "vLLM command", func(cfg *AdapterConfig) { replaceAdapterArg(cfg.Command, "vllm", "auto") }},
		{"vllm observation drift", "vllm.json", "observation_command", func(cfg *AdapterConfig) { replaceAdapterArg(cfg.ObservationCommand, ".State.Running", "true") }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			marker := filepath.Join(dir, "helper-called")
			cfg := maintainedAdapterRunTestConfig(t, test.example, endpoint.URL, marker)
			t.Setenv(cfg.APIKeyEnv, "must-not-leak")
			test.mutate(&cfg)
			configPath := filepath.Join(dir, "config.json")
			corpusPath := filepath.Join(dir, "corpus.json")
			reportPath := filepath.Join(dir, "report.json")
			archivePath := filepath.Join(dir, "archive.json")
			writeJSONTest(t, configPath, cfg)
			writeJSONTest(t, corpusPath, qwen38quant.DefaultCorpus())
			beforeRequests := requests.Load()

			err := RunAdapter(context.Background(), configPath, corpusPath, reportPath, archivePath)
			if err == nil || !contains(err.Error(), test.want) || !contains(err.Error(), "SelfcheckAdapterConfig") {
				t.Fatalf("err=%v, want field %q and recovery action", err, test.want)
			}
			if contains(err.Error(), "must-not-leak") {
				t.Fatalf("refusal leaked credential value: %v", err)
			}
			if requests.Load() != beforeRequests {
				t.Fatalf("drifted config contacted endpoint: requests %d -> %d", beforeRequests, requests.Load())
			}
			for _, path := range []string{marker, reportPath, archivePath} {
				if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
					t.Fatalf("%s exists after structural refusal", filepath.Base(path))
				}
			}
		})
	}
}

func TestRunAdapterValidMaintainedConfigReachesCampaign(t *testing.T) {
	var requests atomic.Int32
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		http.Error(w, "campaign seam reached", http.StatusServiceUnavailable)
	}))
	defer endpoint.Close()
	dir := t.TempDir()
	marker := filepath.Join(dir, "helper-called")
	cfg := maintainedAdapterRunTestConfig(t, "vllm.json", endpoint.URL, marker)
	t.Setenv(cfg.APIKeyEnv, "test-key")
	configPath, corpusPath := filepath.Join(dir, "config.json"), filepath.Join(dir, "corpus.json")
	writeJSONTest(t, configPath, cfg)
	writeJSONTest(t, corpusPath, qwen38quant.DefaultCorpus())
	err := RunAdapter(context.Background(), configPath, corpusPath, filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json"))
	if err == nil || requests.Load() == 0 {
		t.Fatalf("valid maintained config did not reach campaign seam: requests=%d err=%v", requests.Load(), err)
	}
	if _, statErr := os.Stat(marker); statErr != nil {
		t.Fatalf("valid maintained config did not execute observation helper: %v", statErr)
	}
}

func maintainedAdapterRunTestConfig(t *testing.T, example, endpoint, marker string) AdapterConfig {
	t.Helper()
	cfg, err := SelfcheckAdapterConfig(filepath.Join("..", "..", "examples", "qwen38quantrun", example))
	if err != nil {
		t.Fatal(err)
	}
	revision := cfg.Expected.RuntimeRevision
	observationPath := filepath.Join(t.TempDir(), "observation.json")
	writeJSONTest(t, observationPath, Observation{
		Identity: cfg.Expected, Hardware: "test", Software: "test", Device: cfg.RequireDevice,
		ContextTokens: 1, CacheMode: "test", Resident: true,
	})
	cfg.Endpoint.Endpoint = endpoint
	cfg.ObservationCommand = adapterHelperCommand("observation", marker, observationPath, revision, ".State.Running", ".HostConfig.DeviceRequests")
	cfg.RestartCommand = adapterHelperCommand("effect", marker, revision)
	cfg.ReadyCommand = adapterHelperCommand("effect", marker, revision, "/health")
	cfg.CleanupCommand = adapterHelperCommand("effect", marker, revision)
	cfg.Command = adapterHelperCommand("ok", "", revision, cfg.APIKeyEnv, "--pull", "never", "--gpus", "all")
	if cfg.ExecutionEngine == qwen38quant.EngineLlamaCpp {
		cfg.Command = append(cfg.Command, "llama.cpp:"+revision, "--model", "/models/model.gguf", "--n-gpu-layers", "all")
	} else {
		cfg.Command = append(cfg.Command, "vllm:"+revision, "serve", "/models/model", "--model-impl", "vllm")
	}
	if err := validateMaintainedAdapter(cfg); err != nil {
		t.Fatalf("test adapter is not maintained-contract clean: %v", err)
	}
	return cfg
}

func replaceAdapterArg(argv []string, old, new string) {
	for i := range argv {
		if argv[i] == old {
			argv[i] = new
			return
		}
	}
}

func TestCommandProbeRejectsUnknownObservationFields(t *testing.T) {
	argv := helperCommand("observation-unknown")
	_, err := (commandProbe{argv: argv}).Observe(context.Background())
	if err == nil || !contains(err.Error(), "unknown field") {
		t.Fatalf("err=%v", err)
	}
}

func TestRunAdapterWritesNothingWhenProbeFails(t *testing.T) {
	dir := t.TempDir()
	cfg := AdapterConfig{
		Endpoint: EndpointConfig{Endpoint: "http://invalid", Model: "exact"}, ExecutionEngine: qwen38quant.EngineFakNative, Arm: "q4_k_m",
		ObservationCommand: helperCommand("probe-fail"), RestartCommand: helperCommand("ok"),
		ReadyCommand: helperCommand("ok"), CleanupCommand: helperCommand("ok"),
	}
	configPath, corpusPath := filepath.Join(dir, "config.json"), filepath.Join(dir, "corpus.json")
	writeJSONTest(t, configPath, cfg)
	writeJSONTest(t, corpusPath, qwen38quant.DefaultCorpus())
	report, archive := filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json")
	if err := RunAdapter(context.Background(), configPath, corpusPath, report, archive); err == nil {
		t.Fatal("expected probe failure")
	}
	for _, path := range []string{report, archive} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("%s exists after failure", path)
		}
	}
}

func TestRunAdapterManagedServerRefusesDriftBeforeTimedRequestOrCleanup(t *testing.T) {
	var models, chats atomic.Int32
	var fixture *managedFixture
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			models.Add(1)
			fixture.model = "drifted-model"
			fixture.write(t)
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "exact"}}})
		case "/v1/chat/completions":
			chats.Add(1)
			http.Error(w, "timed request must not run", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	defer endpoint.Close()
	fixture = newManagedFixture(t, endpoint.URL, "exact")
	dir := t.TempDir()
	marker := filepath.Join(dir, "cleanup-called")
	observationMarker := filepath.Join(dir, "observation-called")
	observationPath := filepath.Join(dir, "observation.json")
	cfg := managedAdapterConfig(t, endpoint.URL, "exact", marker, observationPath)
	cfg.ObservationCommand = adapterHelperCommand("observation", observationMarker, observationPath)
	cfg.ManagedServer.Directory = fixture.dir
	cfg.RestartCommand = adapterHelperCommand("managed-restart", fixture.dir)
	writeJSONTest(t, observationPath, Observation{Identity: cfg.Expected, Hardware: "test", Software: "test", Device: cfg.RequireDevice, ContextTokens: 16384, CacheMode: "test", Resident: true})
	t.Setenv("FAK_QWEN38_MANAGED_TEST_KEY", "test-key")
	cfg.APIKeyEnv = "FAK_QWEN38_MANAGED_TEST_KEY"
	configPath, corpusPath := filepath.Join(dir, "config.json"), filepath.Join(dir, "corpus.json")
	writeJSONTest(t, configPath, cfg)
	writeJSONTest(t, corpusPath, qwen38quant.DefaultCorpus())
	reportPath, archivePath := filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json")
	err := RunAdapter(context.Background(), configPath, corpusPath, reportPath, archivePath)
	if err == nil || !contains(err.Error(), "managed server READY identity changed") {
		t.Fatalf("err=%v", err)
	}
	if models.Load() != 1 || chats.Load() != 0 {
		t.Fatalf("requests models=%d chats=%d", models.Load(), chats.Load())
	}
	assertNoAdapterArtifacts(t, marker, reportPath, archivePath)
	if temps, globErr := filepath.Glob(filepath.Join(dir, ".*.tmp")); globErr != nil || len(temps) != 0 {
		t.Fatalf("temporary artifacts=%v err=%v", temps, globErr)
	}
}

func TestRunAdapterManagedServerBindsMatchingGenerationChain(t *testing.T) {
	var chats atomic.Int32
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "exact"}}})
			return
		}
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		call := chats.Add(1)
		msg := map[string]any{"content": "ok"}
		switch {
		case call <= 3:
		case call <= 6:
			msg["content"] = `{"ok":true}`
		case call <= 9:
			msg["content"] = ""
			msg["tool_calls"] = []any{map[string]any{"function": map[string]any{"name": "x", "arguments": `null`}}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "exact", "choices": []any{map[string]any{"message": msg}}, "usage": map[string]int{"prompt_tokens": 10, "completion_tokens": 2}})
	}))
	defer endpoint.Close()
	fixture := newManagedFixture(t, endpoint.URL, "exact")
	dir := t.TempDir()
	marker := filepath.Join(dir, "lifecycle-called")
	observationMarker := filepath.Join(dir, "observation-called")
	observationPath := filepath.Join(dir, "observation.json")
	cfg := managedAdapterConfig(t, endpoint.URL, "exact", marker, observationPath)
	cfg.ObservationCommand = adapterHelperCommand("observation", observationMarker, observationPath)
	cfg.ManagedServer.Directory = fixture.dir
	cfg.RestartCommand = adapterHelperCommand("managed-restart", fixture.dir)
	writeJSONTest(t, observationPath, Observation{Identity: cfg.Expected, Hardware: "test", Software: "test", Device: cfg.RequireDevice, ContextTokens: 16384, CacheMode: "test", Resident: true})
	t.Setenv("FAK_QWEN38_MANAGED_TEST_KEY", "test-key")
	cfg.APIKeyEnv = "FAK_QWEN38_MANAGED_TEST_KEY"
	configPath, corpusPath := filepath.Join(dir, "config.json"), filepath.Join(dir, "corpus.json")
	writeJSONTest(t, configPath, cfg)
	writeJSONTest(t, corpusPath, qwen38quant.DefaultCorpus())
	reportPath, archivePath := filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json")
	if err := RunAdapter(context.Background(), configPath, corpusPath, reportPath, archivePath); err != nil {
		t.Fatal(err)
	}
	if chats.Load() != int32(len(qwen38quant.RequiredWorkloads)*3) {
		t.Fatalf("timed requests=%d", chats.Load())
	}
	var archive managedArchive
	archiveRaw, err := os.ReadFile(archivePath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(archiveRaw, &archive); err != nil {
		t.Fatal(err)
	}
	if archive.Schema != "fak.qwen38-quant-managed-raw/1" || len(archive.ServerIdentity) != 2 {
		t.Fatalf("managed archive=%+v", archive)
	}
	if archive.ServerIdentity[0].Generation != 7 || archive.ServerIdentity[1].Generation != 8 || archive.ServerIdentity[0].ReceiptDigest == archive.ServerIdentity[1].ReceiptDigest {
		t.Fatalf("identity chain=%+v", archive.ServerIdentity)
	}
	var report qwen38quant.Report
	reportRaw, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(reportRaw, &report); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(archiveRaw)
	if report.RawArchiveSHA256 != fmt.Sprintf("%x", sum) || report.Verdict != "PROMOTE" {
		t.Fatalf("report archive/verdict=%s/%s", report.RawArchiveSHA256, report.Verdict)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("validated cleanup was not executed: %v", err)
	}
}

func assertNoAdapterArtifacts(t *testing.T, paths ...string) {
	t.Helper()
	for _, path := range paths {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("%s exists after refusal: %v", filepath.Base(path), err)
		}
	}
}

func TestAdapterHelperProcess(t *testing.T) {
	separator := slices.Index(os.Args, "--")
	if separator < 0 || separator+1 >= len(os.Args) {
		dir := t.TempDir()
		marker := filepath.Join(dir, "marker")
		observation := filepath.Join(dir, "observation.json")
		want := []byte(`{"identity":{"runtime":"fak"}}`)
		if err := os.WriteFile(observation, want, 0o600); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(helperCommand("observation")[0], append(helperCommand("observation")[1:], marker, observation)...)
		output, err := cmd.Output()
		if err != nil {
			t.Fatalf("observation helper failed: %v", err)
		}
		if !slices.Equal(output, want) {
			t.Fatalf("observation output = %q, want %q", output, want)
		}
		if markerRaw, err := os.ReadFile(marker); err != nil || string(markerRaw) != "called\n" {
			t.Fatalf("observation marker = %q, %v", markerRaw, err)
		}
		return
	}
	args := os.Args[separator+1:]
	switch args[0] {
	case "ok":
		os.Exit(0)
	case "managed-restart":
		if len(args) < 2 {
			os.Exit(4)
		}
		if err := advanceManagedFixture(args[1]); err != nil {
			os.Stderr.WriteString(err.Error())
			os.Exit(8)
		}
		os.Exit(0)
	case "effect":
		if len(args) < 2 {
			os.Exit(4)
		}
		if err := os.WriteFile(args[1], []byte("called\n"), 0o600); err != nil {
			os.Exit(5)
		}
		os.Exit(0)
	case "observation":
		if len(args) < 3 {
			os.Exit(4)
		}
		if err := os.WriteFile(args[1], []byte("called\n"), 0o600); err != nil {
			os.Exit(5)
		}
		raw, err := os.ReadFile(args[2])
		if err != nil {
			os.Exit(6)
		}
		_, _ = os.Stdout.Write(raw)
		os.Exit(0)
	case "observation-unknown":
		os.Stdout.WriteString(`{"identity":{},"unknown":true}`)
		os.Exit(0)
	default:
		os.Stderr.WriteString("probe failed")
		os.Exit(7)
	}
}

func advanceManagedFixture(dir string) error {
	statePath := filepath.Join(dir, serverlifecycle.StateFilename)
	receiptPath := filepath.Join(dir, serverlifecycle.ReceiptFilename)
	var state map[string]any
	stateRaw, err := os.ReadFile(statePath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(stateRaw, &state); err != nil {
		return err
	}
	generation, ok := state["generation"].(float64)
	if !ok {
		return errors.New("fixture generation missing")
	}
	generation++
	state["generation"] = generation
	state["updated_at"] = time.Now().UTC().Format(time.RFC3339Nano)
	stateRaw, err = json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	stateRaw = append(stateRaw, '\n')
	if err := os.WriteFile(statePath, stateRaw, 0o600); err != nil {
		return err
	}
	var receipt serverproduct.ServerReceipt
	receiptRaw, err := os.ReadFile(receiptPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(receiptRaw, &receipt); err != nil {
		return err
	}
	receipt.Generation = uint64(generation)
	receipt.CreatedAt = state["updated_at"].(string)
	receipt.Readiness.ObservedAt = receipt.CreatedAt
	receiptRaw, err = json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	receiptRaw = append(receiptRaw, '\n')
	return os.WriteFile(receiptPath, receiptRaw, 0o600)
}

func helperCommand(mode string) []string {
	return []string{os.Args[0], "-test.run=^TestAdapterHelperProcess$", "--", mode}
}

func adapterHelperCommand(mode, marker string, extra ...string) []string {
	argv := []string{os.Args[0], "-test.run=^TestAdapterHelperProcess$", "--", mode}
	if marker != "" {
		argv = append(argv, marker)
	}
	return append(argv, extra...)
}

func writeJSONTest(t *testing.T, path string, v any) {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
