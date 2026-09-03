package main

import (
	"bytes"
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
