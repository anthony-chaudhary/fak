package qwen38quantrun

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

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
		Endpoint: EndpointConfig{Endpoint: "http://invalid", Model: "exact"}, Arm: "q4_k_m",
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
