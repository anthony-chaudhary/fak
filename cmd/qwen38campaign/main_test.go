package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
)

func TestRunSoakDispatchesToProductionSoakAdapter(t *testing.T) {
	dir := t.TempDir()
	config := filepath.Join(dir, "config.json")
	if err := os.WriteFile(config, []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	exit := run(&stdout, &stderr, []string{
		"--soak",
		"--config", config,
		"--corpus", filepath.Join("..", "..", "docs", "benchmarks", "qwen38-quant", "corpus.json"),
		"--report", filepath.Join(dir, "report.json"),
		"--archive", filepath.Join(dir, "archive.json"),
	})
	if exit != 1 || !strings.Contains(stderr.String(), "soak config requires") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
}

func TestRunRequiresExplicitOutputs(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if exit := run(&stdout, &stderr, []string{"--soak"}); exit != 2 || !strings.Contains(stderr.String(), "usage:") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
}

func TestRunOracleDispatchesToPinnedOracle(t *testing.T) {
	dir := t.TempDir()
	config := filepath.Join(dir, "config.json")
	if err := os.WriteFile(config, []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	exit := run(&stdout, &stderr, []string{
		"--oracle",
		"--config", config,
		"--corpus", filepath.Join("..", "..", "docs", "benchmarks", "qwen38-quant", "corpus.json"),
		"--report", filepath.Join(dir, "report.json"),
		"--archive", filepath.Join(dir, "archive.json"),
	})
	if exit != 1 || !strings.Contains(stderr.String(), "config: schema") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
}

func TestRunRejectsConflictingModes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	exit := run(&stdout, &stderr, []string{"--soak", "--oracle", "--config", "c", "--report", "r", "--archive", "a"})
	if exit != 2 || !strings.Contains(stderr.String(), "--soak | --oracle") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
}

func TestRunAMDScoreboardWritesComparableReport(t *testing.T) {
	dir := t.TempDir()
	config := filepath.Join(dir, "config.json")
	input := qwen38quantrunTestInput()
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(config, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	report := filepath.Join(dir, "report.json")
	exit := run(&stdout, &stderr, []string{"--amd-scoreboard", "--config", config, "--report", report})
	if exit != 0 || !strings.Contains(stdout.String(), "comparable") {
		t.Fatalf("exit=%d stdout=%q stderr=%q", exit, stdout.String(), stderr.String())
	}
	if _, err := os.Stat(report); err != nil {
		t.Fatal(err)
	}
}

func TestRunRejectsThreeConflictingModes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	exit := run(&stdout, &stderr, []string{"--soak", "--oracle", "--amd-scoreboard", "--config", "c", "--report", "r", "--archive", "a"})
	if exit != 2 || !strings.Contains(stderr.String(), "--amd-scoreboard") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
}

func qwen38quantrunTestInput() qwen38quantrun.AMDScoreboardInput {
	sha := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	prompt := strings.Repeat("a", 64)
	arm := qwen38quantrun.AMDArmReceipt{Name: "fak", Engine: "fak-native", Backend: "vulkan", Runtime: "native", ArtifactSHA256: sha, PromptSHA256: prompt, PromptTokenIDs: []int{1}, ContextTokens: 256, ContextBudgetBytes: 1 << 30, KVTypeK: "f16", KVTypeV: "f16", KVOffload: "gpu", FlashAttention: true, GPUMemoryBudget: 6 << 30, HostSpillPolicy: "bounded", PrefillTokens: 1, DecodeTokens: 1, Hardware: "RX 7600", SoftwareRevision: "fak@1", BuildFlags: []string{"vulkan"}, PeakRSSBytes: 1, PeakVRAMBytes: 1, ResidentModelBytes: 1}
	for i := 1; i <= 3; i++ {
		arm.Trials = append(arm.Trials, qwen38quantrun.AMDScoreboardTrial{Repetition: i, ColdSetupSeconds: 1, PrefillSeconds: 1, PrefillTokensPerSecond: 1, WarmDecodeSeconds: 1, WarmDecodeTokensPerSecond: 1, OutputTokenIDs: []int{2}, Logits: []float64{1}, H2DBytes: 1, D2HBytes: 1, QueueSubmissions: 1})
	}
	ref := arm
	ref.Name = "llama.cpp"
	ref.Engine = "llama.cpp"
	ref.ComparatorOnly = true
	ref.SoftwareRevision = "llama.cpp@1"
	ref.PromptTokenIDs = []int{1}
	ref.Trials = append([]qwen38quantrun.AMDScoreboardTrial(nil), arm.Trials...)
	return qwen38quantrun.AMDScoreboardInput{Schema: qwen38quantrun.AMDScoreboardInputSchema, LogitTolerance: 1e-3, Candidate: arm, Reference: ref}
}
