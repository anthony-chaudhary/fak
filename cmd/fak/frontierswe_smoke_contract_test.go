package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

func TestFrontiersweSmokeContractEmitsJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"smoke-contract", "--tasks", repoRootTasksDir, "--task", "git-to-zig",
		"--model", "claude-opus-4-6",
	})
	if code != 0 {
		t.Fatalf("smoke-contract exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}
	var c frontierswe.RawFakContract
	if err := json.Unmarshal(stdout.Bytes(), &c); err != nil {
		t.Fatalf("stdout is not contract JSON: %v\n%s", err, stdout.String())
	}
	if c.Schema != frontierswe.RawFakContractSchema {
		t.Fatalf("schema = %q", c.Schema)
	}
	if c.ResultClaimAllowed {
		t.Fatal("contract must not allow a result claim")
	}
	if c.TaskSelection.Task != "git-to-zig" || len(c.Arms) != 2 {
		t.Fatalf("contract task/arms wrong: task=%+v arms=%+v", c.TaskSelection, c.Arms)
	}
	for _, want := range []string{"score_parity_gate_declared", "tts_metric_declared"} {
		if !contractHasGate(c, want) {
			t.Fatalf("contract missing gate %q: %+v", want, c.Gates)
		}
	}
	if !strings.Contains(stderr.String(), "claim allowed: false") {
		t.Errorf("stderr should surface claim refusal; got:\n%s", stderr.String())
	}
}

func TestFrontiersweSmokeContractWritesMarkdown(t *testing.T) {
	dir := t.TempDir()
	jsonPath := filepath.Join(dir, "contract.json")
	mdPath := filepath.Join(dir, "contract.md")
	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"smoke-contract", "--tasks", repoRootTasksDir, "--task", "git-to-zig",
		"--model", "claude-opus-4-6", "--out", jsonPath, "--md", mdPath,
	})
	if code != 0 {
		t.Fatalf("smoke-contract --out/--md exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("--out should keep stdout empty, got:\n%s", stdout.String())
	}
	jb, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read json: %v", err)
	}
	var c frontierswe.RawFakContract
	if err := json.Unmarshal(jb, &c); err != nil {
		t.Fatalf("json file invalid: %v", err)
	}
	mb, err := os.ReadFile(mdPath)
	if err != nil {
		t.Fatalf("read markdown: %v", err)
	}
	md := string(mb)
	for _, want := range []string{"FrontierSWE Raw-vs-fak Smoke Contract", "Required Before Any Result Claim", "score_parity_gate_declared"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func contractHasGate(c frontierswe.RawFakContract, name string) bool {
	for _, gate := range c.Gates {
		if gate.Name == name && gate.OK {
			return true
		}
	}
	return false
}
