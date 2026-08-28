package qwen38quantrun

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

func TestCheckedInAdapterExamplesSelfcheck(t *testing.T) {
	tests := []struct {
		name   string
		engine string
		arms   []string
	}{
		{"llama.cpp.json", qwen38quant.EngineLlamaCpp, []string{"q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs"}},
		{"vllm.json", qwen38quant.EngineVLLM, []string{"bf16", "fp8", "awq_int4", "gptq_int4"}},
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

func TestAdapterHelperProcess(t *testing.T) {
	if len(os.Args) < 2 || os.Args[len(os.Args)-2] != "--" {
		return
	}
	switch os.Args[len(os.Args)-1] {
	case "ok":
		os.Exit(0)
	case "observation-unknown":
		os.Stdout.WriteString(`{"identity":{},"unknown":true}`)
		os.Exit(0)
	default:
		os.Stderr.WriteString("probe failed")
		os.Exit(7)
	}
}

func helperCommand(mode string) []string {
	return []string{os.Args[0], "-test.run=^TestAdapterHelperProcess$", "--", mode}
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
