package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
