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

func TestRunManagedServerRefusalLeavesNoOutputs(t *testing.T) {
	dir := t.TempDir()
	config := filepath.Join(dir, "config.json")
	cfg := qwen38quantrun.AdapterConfig{
		Endpoint:        qwen38quantrun.EndpointConfig{Endpoint: "http://127.0.0.1:1", Model: "hand-copied"},
		ExecutionEngine: "fak-native", Arm: "q4_k_m", ObservationCommand: []string{"probe"}, RestartCommand: []string{"restart"}, ReadyCommand: []string{"ready"}, CleanupCommand: []string{"cleanup"},
		ManagedServer: &qwen38quantrun.ManagedServerConfig{Directory: filepath.Join(dir, "missing"), ProtocolFamily: "openai-http", ProtocolRevision: "v1", Capabilities: []string{"models", "chat-completions"}, ModelAlias: "exact"},
	}
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(config, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	report, archive := filepath.Join(dir, "report.json"), filepath.Join(dir, "archive.json")
	var stdout, stderr bytes.Buffer
	exit := run(&stdout, &stderr, []string{"--config", config, "--corpus", filepath.Join("..", "..", "docs", "benchmarks", "qwen38-quant", "corpus.json"), "--report", report, "--archive", archive})
	if exit != 1 || !strings.Contains(stderr.String(), "managed server READY identity") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
	for _, path := range []string{report, archive} {
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("%s exists after refusal", filepath.Base(path))
		}
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
	input := qwen38quantrunTestInput(t)
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
	if exit != 0 || !strings.HasPrefix(stdout.String(), "comparable ") {
		t.Fatalf("exit=%d stdout=%q stderr=%q", exit, stdout.String(), stderr.String())
	}
	if _, err := os.Stat(report); err != nil {
		t.Fatal(err)
	}
}

func TestRunVerifyPromptPacketFailsClosed(t *testing.T) {
	dir := t.TempDir()
	valid := qwen38quantrun.PromptTokenPacket{
		Schema: qwen38quantrun.PromptTokenPacketSchema, PacketID: "cli-regression",
		ArtifactSHA256: strings.Repeat("a", 64), TokenizerIdentity: "test-tokenizer",
		TokenizerDigest: strings.Repeat("b", 64), TemplateDigest: strings.Repeat("c", 64),
		PromptTokenIDs: []int{1, 2, 3}, ContextBudget: qwen38quantrun.ContextBudget{ContextTokens: 256, ContextBudgetBytes: 1 << 20},
		GenerationControls: qwen38quantrun.GenerationControls{Temperature: 0, TopP: 1, TopK: 1, MaxOutputTokens: 4},
	}
	frozen, err := qwen38quantrun.FreezePromptPacket(valid)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("verified v2 packet materializes without execution", func(t *testing.T) {
		path := filepath.Join(dir, "valid-v2.json")
		if err := qwen38quantrun.WritePromptPacketFile(path, frozen); err != nil {
			t.Fatal(err)
		}
		var stdout, stderr bytes.Buffer
		exit := run(&stdout, &stderr, []string{"--verify-prompt-packet", path})
		if exit != 0 || stderr.Len() != 0 || !strings.Contains(stdout.String(), "digest="+frozen.PacketDigest) {
			t.Fatalf("exit=%d stdout=%q stderr=%q", exit, stdout.String(), stderr.String())
		}
	})

	t.Run("template identity does not match sealed packet", func(t *testing.T) {
		mismatched := frozen
		mismatched.TemplateDigest = strings.Repeat("d", 64)
		path := filepath.Join(dir, "mismatched-v2.json")
		raw, err := json.Marshal(mismatched)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, raw, 0o600); err != nil {
			t.Fatal(err)
		}
		var stdout, stderr bytes.Buffer
		exit := run(&stdout, &stderr, []string{"--verify-prompt-packet", path})
		if exit != 1 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "prompt packet digest mismatch") {
			t.Fatalf("exit=%d stdout=%q stderr=%q", exit, stdout.String(), stderr.String())
		}
	})

	t.Run("legacy packet remains readable but receives no credit", func(t *testing.T) {
		legacy := frozen
		legacy.Schema = "fak.qwen38.prompt-token-packet.v1"
		legacy.TemplateDigest = ""
		legacy.PacketDigest = ""
		digest, err := qwen38quantrun.ComputePromptPacketDigest(legacy)
		if err != nil {
			t.Fatal(err)
		}
		legacy.PacketDigest = digest
		path := filepath.Join(dir, "legacy-v1.json")
		if err := qwen38quantrun.WritePromptPacketFile(path, legacy); err != nil {
			t.Fatal(err)
		}
		var stdout, stderr bytes.Buffer
		exit := run(&stdout, &stderr, []string{"--verify-prompt-packet", path})
		if exit != 1 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "readable but not eligible") {
			t.Fatalf("exit=%d stdout=%q stderr=%q", exit, stdout.String(), stderr.String())
		}
	})

	t.Run("explicit empty packet path cannot fall through to scoreboard", func(t *testing.T) {
		config := filepath.Join(dir, "scoreboard-input.json")
		raw, err := json.Marshal(qwen38quantrunTestInput(t))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(config, raw, 0o600); err != nil {
			t.Fatal(err)
		}
		report := filepath.Join(dir, "must-not-exist.json")
		var stdout, stderr bytes.Buffer
		exit := run(&stdout, &stderr, []string{"--verify-prompt-packet=", "--amd-scoreboard", "--config", config, "--report", report})
		if exit != 2 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "usage: qwen38campaign --verify-prompt-packet PACKET.json") {
			t.Fatalf("exit=%d stdout=%q stderr=%q", exit, stdout.String(), stderr.String())
		}
		if _, err := os.Stat(report); !os.IsNotExist(err) {
			t.Fatalf("scoreboard report exists after empty packet-path refusal: %v", err)
		}
	})
}

func TestRunRejectsThreeConflictingModes(t *testing.T) {
	var stdout, stderr bytes.Buffer
	exit := run(&stdout, &stderr, []string{"--soak", "--oracle", "--amd-scoreboard", "--config", "c", "--report", "r", "--archive", "a"})
	if exit != 2 || !strings.Contains(stderr.String(), "--amd-scoreboard") {
		t.Fatalf("exit=%d stderr=%q", exit, stderr.String())
	}
}

func qwen38quantrunTestInput(t *testing.T) qwen38quantrun.AMDScoreboardInput {
	t.Helper()
	sha := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	prompt := strings.Repeat("a", 64)
	packet, err := qwen38quantrun.FreezePromptPacket(qwen38quantrun.PromptTokenPacket{
		Schema: qwen38quantrun.PromptTokenPacketSchema, PacketID: "scoreboard-cli-test", ArtifactSHA256: sha,
		TokenizerIdentity: "test-tokenizer", TokenizerDigest: strings.Repeat("b", 64), TemplateDigest: strings.Repeat("c", 64),
		PromptTokenIDs: []int{1}, ContextBudget: qwen38quantrun.ContextBudget{ContextTokens: 256, ContextBudgetBytes: 1 << 30},
		GenerationControls: qwen38quantrun.GenerationControls{Temperature: 0, TopP: 1, MaxOutputTokens: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	arm := qwen38quantrun.AMDArmReceipt{Name: "fak", Engine: "fak-native", Backend: "vulkan", Runtime: "native", ArtifactSHA256: sha, PromptSHA256: prompt, PromptTokenIDs: []int{1}, ContextTokens: 256, ContextBudgetBytes: 1 << 30, KVTypeK: "f16", KVTypeV: "f16", KVOffload: "gpu", FlashAttention: true, GPUMemoryBudget: 6 << 30, HostSpillPolicy: "bounded", PrefillTokens: 1, DecodeTokens: 1, Hardware: "RX 7600", SoftwareRevision: "fak@1", BuildFlags: []string{"vulkan"}, PeakRSSBytes: 1, PeakVRAMBytes: 1, ResidentModelBytes: 1, TokenizerDigest: packet.TokenizerDigest, TemplateDigest: packet.TemplateDigest, PromptPacketDigest: packet.PacketDigest, TopP: packet.GenerationControls.TopP, TopK: packet.GenerationControls.TopK, PromptPacket: &packet}
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
