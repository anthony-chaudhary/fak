package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38campaign"
)

func TestRunBenchSubagentHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBenchSubagent(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("runBenchSubagent --help failed with code %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "Usage of fak bench subagent") && !strings.Contains(stderr.String(), "-scenario") {
		t.Errorf("expected usage output in stderr, got:\n%s", stderr.String())
	}
}

func TestRunBenchSubagentJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--scenario=shared_prefix_forked", "--concurrency=4", "--runs=5", "--json"}

	code := runBenchSubagent(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchSubagent failed with code %d, stderr: %s", code, stderr.String())
	}

	var receipt qwen38campaign.SubagentFanoutReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v\nOutput: %s", err, stdout.String())
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	if receipt.Schema != qwen38campaign.SubagentFanoutSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, qwen38campaign.SubagentFanoutSchema)
	}
	if receipt.Engine != "fak-native" {
		t.Errorf("engine = %q, want %q", receipt.Engine, "fak-native")
	}
	if receipt.Config.Scenario != qwen38campaign.ScenarioSharedPrefixForked {
		t.Errorf("scenario = %q, want %q", receipt.Config.Scenario, qwen38campaign.ScenarioSharedPrefixForked)
	}
	if receipt.Config.Concurrency != 4 {
		t.Errorf("concurrency = %d, want 4", receipt.Config.Concurrency)
	}
	if receipt.Summary.RunsCount != 5 {
		t.Errorf("runs count = %d, want 5", receipt.Summary.RunsCount)
	}
	if !receipt.Summary.ParityPassed {
		t.Errorf("logit cosine parity failed: mean = %f", receipt.Summary.MeanLogitCosineParity)
	}
	if receipt.Summary.MeanMALLHitRate < 0.85 {
		t.Errorf("expected MALL hit rate >= 0.85, got %f", receipt.Summary.MeanMALLHitRate)
	}
}

func TestRunBenchSubagentWithSubcommandPrefix(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"subagent", "--scenario=warm_same_prefix", "--concurrency=2", "--runs=5", "--json"}

	code := runBenchSubagent(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchSubagent with subcommand prefix failed with code %d, stderr: %s", code, stderr.String())
	}

	var receipt qwen38campaign.SubagentFanoutReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v\nOutput: %s", err, stdout.String())
	}

	if receipt.Config.Scenario != qwen38campaign.ScenarioWarmSamePrefix {
		t.Errorf("scenario = %q, want %q", receipt.Config.Scenario, qwen38campaign.ScenarioWarmSamePrefix)
	}
	if receipt.Config.Concurrency != 2 {
		t.Errorf("concurrency = %d, want 2", receipt.Config.Concurrency)
	}
}

func TestRunBenchSubagentColdScenario(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--scenario=cold", "--concurrency=1", "--runs=5", "--json"}

	code := runBenchSubagent(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchSubagent cold scenario failed with code %d, stderr: %s", code, stderr.String())
	}

	var receipt qwen38campaign.SubagentFanoutReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v\nOutput: %s", err, stdout.String())
	}

	if receipt.Config.Scenario != qwen38campaign.ScenarioCold {
		t.Errorf("scenario = %q, want %q", receipt.Config.Scenario, qwen38campaign.ScenarioCold)
	}
	if receipt.Config.Concurrency != 1 {
		t.Errorf("concurrency = %d, want 1", receipt.Config.Concurrency)
	}
}

func TestRunBenchSubagentHumanOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--scenario=shared_prefix_forked", "--concurrency=2", "--runs=5"}

	code := runBenchSubagent(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchSubagent human output failed with code %d, stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	requiredSnippets := []string{
		"Strix Halo Subagent Fan-Out Benchmark Receipt",
		"MALL Cache Size:     32 MB",
		"Mean Throughput:",
		"host_dispatch:",
		"prefix_tree_lookup:",
		"kv_allocation:",
		"gpu_kernel:",
		"token_sampling:",
		"Verification Digest:",
	}
	for _, snippet := range requiredSnippets {
		if !strings.Contains(out, snippet) {
			t.Errorf("human output missing snippet %q; output:\n%s", snippet, out)
		}
	}
}

func TestRunBenchSubagentOutFile(t *testing.T) {
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "subagent_receipt.json")

	var stdout, stderr bytes.Buffer
	args := []string{"--scenario=shared_prefix_forked", "--concurrency=4", "--runs=5", "--json", "--out=" + outPath}

	code := runBenchSubagent(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchSubagent with --out failed with code %d, stderr: %s", code, stderr.String())
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read written file: %v", err)
	}

	var receipt qwen38campaign.SubagentFanoutReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("failed to unmarshal written JSON: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("written receipt failed validation: %v", err)
	}
}

func TestRunBenchSubagentInvalidFlags(t *testing.T) {
	tests := []struct {
		name     string
		args     []string
		wantCode int
	}{
		{
			name:     "invalid-scenario",
			args:     []string{"--scenario=unknown_scenario"},
			wantCode: 2,
		},
		{
			name:     "invalid-concurrency",
			args:     []string{"--concurrency=3"},
			wantCode: 2,
		},
		{
			name:     "invalid-runs-count",
			args:     []string{"--runs=2"},
			wantCode: 2,
		},
		{
			name:     "unknown-flag",
			args:     []string{"--nonexistent-flag=1"},
			wantCode: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runBenchSubagent(&stdout, &stderr, tc.args)
			if code != tc.wantCode {
				t.Errorf("runBenchSubagent code = %d, want %d (stderr: %s)", code, tc.wantCode, stderr.String())
			}
		})
	}
}

func TestRunBenchSubagentMatrix(t *testing.T) {
	concurrencies := []int{1, 2, 4, 8}
	for _, b := range concurrencies {
		var stdout, stderr bytes.Buffer
		args := []string{
			"--scenario=shared_prefix_forked",
			"--concurrency=" + string(rune('0'+b)),
			"--runs=5",
			"--gen-tokens=16",
			"--json",
		}

		code := runBenchSubagent(&stdout, &stderr, args)
		if code != 0 {
			t.Fatalf("runBenchSubagent matrix B=%d failed with code %d, stderr: %s", b, code, stderr.String())
		}

		var receipt qwen38campaign.SubagentFanoutReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatalf("matrix B=%d unmarshal error: %v", b, err)
		}
		if receipt.Config.Concurrency != b {
			t.Errorf("matrix B=%d receipt concurrency = %d", b, receipt.Config.Concurrency)
		}
	}
}
