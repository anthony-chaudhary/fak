package model

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms lane=default
func TestSoftwareReceiptTestsPreserveRepositoryEvidence(t *testing.T) {
	root := t.TempDir()
	packageDir := filepath.Join(root, "internal", "model")
	if err := os.MkdirAll(packageDir, 0755); err != nil {
		t.Fatal(err)
	}
	issues := []string{"issue-10946-nccl-ep-collective", "issue-10949-hf-parity"}
	sentinel := []byte("retained historical evidence\n")
	for _, issue := range issues {
		path := filepath.Join(root, "docs", "_witnesses", issue, "receipt.json")
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, sentinel, 0644); err != nil {
			t.Fatal(err)
		}
	}
	// Execute the real generators in a disposable repository-shaped directory.
	// The legacy names keep this witness executable against the unfixed base.
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "-test.run=^(TestLocalCollectiveExpertParallelDeltaReceipt|TestSyntheticGLMMLPSoftwareReceipt|TestNCCLCollectiveExpertParallelForward|TestHFReferenceForwardParity_Qwen38_GLM53)$", "-test.v")
	cmd.Dir = packageDir
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("receipt generators failed: %v\n%s", err, output)
	}
	for _, names := range [][2]string{
		{"TestLocalCollectiveExpertParallelDeltaReceipt", "TestNCCLCollectiveExpertParallelForward"},
		{"TestSyntheticGLMMLPSoftwareReceipt", "TestHFReferenceForwardParity_Qwen38_GLM53"},
	} {
		if !strings.Contains(string(output), "--- PASS: "+names[0]+" ") && !strings.Contains(string(output), "--- PASS: "+names[1]+" ") {
			t.Fatalf("receipt generator did not execute: %v\n%s", names, output)
		}
	}
	for _, issue := range issues {
		path := filepath.Join(root, "docs", "_witnesses", issue, "receipt.json")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(data, sentinel) {
			t.Errorf("%s: receipt generator overwrote retained repository evidence", issue)
		}
	}
}
