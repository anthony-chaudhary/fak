package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agentxbench"
)

func TestAgentXSelfCheck(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--selfcheck", "--agents=2", "--turns=2"}

	code := runAgentX(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("expected code 0, got %d. stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "AgentX Benchmark Result:") {
		t.Fatalf("expected benchmark result header, got: %s", out)
	}
	if !strings.Contains(out, "Status         : SIMULATED") {
		t.Fatalf("expected SIMULATED, got: %s", out)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAgentXLiveRunNeedsEngineAndHardware(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAgentX(&stdout, &stderr, []string{"--endpoint=http://127.0.0.1:1/v1", "--agents=1", "--turns=1"})
	if code == 0 {
		t.Fatalf("live run without --engine/--hardware succeeded; stdout: %s", stdout.String())
	}
}

func TestAgentXSelfCheckJSONAndValidate(t *testing.T) {
	tmpDir := t.TempDir()
	receiptPath := filepath.Join(tmpDir, "agentx-receipt.json")

	var stdout, stderr bytes.Buffer
	args := []string{
		"--selfcheck",
		"--agents=2",
		"--turns=2",
		"--json",
		"--out=" + receiptPath,
	}

	code := runAgentX(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("expected code 0, got %d. stderr: %s", code, stderr.String())
	}

	var receipt agentxbench.AgentXReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v", err)
	}
	if receipt.Schema != agentxbench.SchemaIdentifier {
		t.Fatalf("unexpected schema: %s", receipt.Schema)
	}
	if receipt.ValidationStatus != agentxbench.StatusSimulated {
		t.Fatalf("expected SIMULATED, got: %s", receipt.ValidationStatus)
	}

	// Verify file was written
	data, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("failed to read written receipt: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("written receipt is empty")
	}

	// A self-check receipt is never admitted as evidence.
	stdout.Reset()
	stderr.Reset()
	valCode := runAgentX(&stdout, &stderr, []string{"--validate=" + receiptPath})
	if valCode == 0 {
		t.Fatalf("validate admitted a mock receipt: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "SIMULATED") {
		t.Fatalf("expected SIMULATED refusal, got: %s", stderr.String())
	}
}
