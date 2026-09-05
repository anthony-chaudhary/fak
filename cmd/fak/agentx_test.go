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
	if !strings.Contains(out, "Status         : VERIFIED_PASS") {
		t.Fatalf("expected VERIFIED_PASS, got: %s", out)
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
	if receipt.ValidationStatus != "VERIFIED_PASS" {
		t.Fatalf("expected VERIFIED_PASS, got: %s", receipt.ValidationStatus)
	}

	// Verify file was written
	data, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("failed to read written receipt: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("written receipt is empty")
	}

	// Test validate subcommand on the written file
	stdout.Reset()
	stderr.Reset()
	valCode := runAgentX(&stdout, &stderr, []string{"--validate=" + receiptPath})
	if valCode != 0 {
		t.Fatalf("validate returned code %d. stderr: %s", valCode, stderr.String())
	}
	if !strings.Contains(stdout.String(), "PASS: receipt") {
		t.Fatalf("expected PASS in validate output, got: %s", stdout.String())
	}
}
