package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBenchTB4CLI(t *testing.T) {
	// 1. Test help flag
	var stdout, stderr bytes.Buffer
	code := runBenchTB4(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help, got %d", code)
	}
	helpText := stdout.String()
	subcommands := []string{"preflight", "run", "eval", "compare", "replay"}
	for _, sub := range subcommands {
		if !strings.Contains(helpText, sub) {
			t.Errorf("help text missing subcommand %s: %s", sub, helpText)
		}
	}

	// 2. Test preflight subcommand
	stdout.Reset()
	stderr.Reset()
	code = runBenchTB4(&stdout, &stderr, []string{"preflight"})
	if code != 0 {
		t.Errorf("expected exit code 0 for preflight, got %d", code)
	}
	preflightOut := stdout.String()
	if !strings.Contains(preflightOut, "Preflight Gate Check") {
		t.Errorf("preflight missing header: %s", preflightOut)
	}

	// 3. Test run subcommand with missing dataset
	stdout.Reset()
	stderr.Reset()
	code = runBenchTB4(&stdout, &stderr, []string{"run"})
	if code != 1 {
		t.Errorf("expected exit code 1 for run without dataset, got %d", code)
	}

	// 4. Test compare subcommand without required dirs
	stdout.Reset()
	stderr.Reset()
	code = runBenchTB4(&stdout, &stderr, []string{"compare"})
	if code != 1 {
		t.Errorf("expected exit code 1 for compare without args, got %d", code)
	}
}

func TestBenchTB4RunEvalComparePipeline(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "tb4-cli-pipeline-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	dataset := filepath.Join("..", "..", "testdata", "tb4bench", "synthetic_suite.json")
	if _, err := os.Stat(dataset); err != nil {
		dataset = filepath.Join("testdata", "tb4bench", "synthetic_suite.json")
	}

	var stdout, stderr bytes.Buffer

	// 1. Run command with --mock
	code := runBenchTB4(&stdout, &stderr, []string{
		"run",
		"--mock",
		"--dataset", dataset,
		"--out", tempDir,
	})
	if code != 0 {
		t.Fatalf("run failed with code %d: stderr: %s", code, stderr.String())
	}
	runOut := stdout.String()
	if !strings.Contains(runOut, "Starting TB4 Benchmark Run") {
		t.Errorf("run output missing start banner: %s", runOut)
	}
	if !strings.Contains(runOut, "Run completed successfully") {
		t.Errorf("run output missing completion message: %s", runOut)
	}

	// 2. Eval command
	stdout.Reset()
	stderr.Reset()
	code = runBenchTB4(&stdout, &stderr, []string{
		"eval",
		"--run-dir", tempDir,
		"--dataset", dataset,
	})
	if code != 0 {
		t.Fatalf("eval failed with code %d: stderr: %s", code, stderr.String())
	}
	evalOut := stdout.String()
	if !strings.Contains(evalOut, "Evaluating tasks in") {
		t.Errorf("eval output missing evaluating banner: %s", evalOut)
	}
	if !strings.Contains(evalOut, "SOLVED") {
		t.Errorf("eval output missing SOLVED verdicts: %s", evalOut)
	}

	// 3. Compare command
	stdout.Reset()
	stderr.Reset()
	jsonOut := filepath.Join(tempDir, "out.json")
	mdOut := filepath.Join(tempDir, "out.md")
	code = runBenchTB4(&stdout, &stderr, []string{
		"compare",
		"--fak-dir", filepath.Join(tempDir, "fak"),
		"--opencode-dir", filepath.Join(tempDir, "opencode"),
		"--dataset", dataset,
		"--out-json", jsonOut,
		"--out-md", mdOut,
	})
	if code != 0 {
		t.Fatalf("compare failed with code %d: stderr: %s", code, stderr.String())
	}
	compOut := stdout.String()
	if !strings.Contains(compOut, "Terminal-Bench 4 Comparative Analysis") {
		t.Errorf("compare output missing title: %s", compOut)
	}
	if !strings.Contains(compOut, "Executive Summary & Authoritative Solve Rates") {
		t.Errorf("compare output missing executive summary: %s", compOut)
	}
	if _, err := os.Stat(jsonOut); err != nil {
		t.Errorf("expected JSON report on disk: %v", err)
	}
	if _, err := os.Stat(mdOut); err != nil {
		t.Errorf("expected MD report on disk: %v", err)
	}

	// 4. Replay command check
	stdout.Reset()
	stderr.Reset()
	code = runBenchTB4(&stdout, &stderr, []string{
		"replay",
		"--run-dir", filepath.Join(tempDir, "fak"),
		"--task", "tb4-synth-01-syntax-fix",
	})
	if code != 0 {
		t.Fatalf("replay failed with code %d: stderr: %s", code, stderr.String())
	}
	replayOut := stdout.String()
	if !strings.Contains(replayOut, "TB4 Run Replay") {
		t.Errorf("replay output missing replay banner: %s", replayOut)
	}
}
