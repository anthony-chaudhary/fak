package main

import (
	"testing"
	"time"
)

// TestParseWebbenchEvalConfigTimeout proves the #3913 wire on `fak webbench eval`:
// the --timeout flag lands in EvalConfig.Timeout, and an omitted flag stays 0 so
// EvalConfig.harnessTimeout() keeps the built-in 15m default rather than running
// unbounded.
func TestParseWebbenchEvalConfigTimeout(t *testing.T) {
	cfg, out, err := parseWebbenchEvalConfig([]string{"--predictions", "p.json", "--timeout", "90s"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if cfg.Timeout != 90*time.Second {
		t.Errorf("--timeout 90s must land in EvalConfig.Timeout, got %s", cfg.Timeout)
	}
	if cfg.PredictionsPath != "p.json" {
		t.Errorf("predictions path = %q", cfg.PredictionsPath)
	}
	if out != "" {
		t.Errorf("unexpected out = %q", out)
	}

	// Omitted flag -> zero -> package default via harnessTimeout(); never unbounded.
	cfg, _, err = parseWebbenchEvalConfig([]string{"--predictions", "p.json"})
	if err != nil {
		t.Fatalf("parse (no timeout): %v", err)
	}
	if cfg.Timeout != 0 {
		t.Errorf("omitted --timeout must leave EvalConfig.Timeout zero, got %s", cfg.Timeout)
	}

	// --predictions is required.
	if _, _, err := parseWebbenchEvalConfig([]string{"--timeout", "5m"}); err == nil {
		t.Error("missing --predictions must be an error")
	}
}

// TestParseWebbenchCompareArgsTimeout proves the #3913 wire on the compare path:
// --timeout is parsed and (per cmdWebbenchCompare) threaded into
// CompareInputs.Timeout, which report.go hands to the --predictions RunEval. This
// is the confusion-risk the issue flagged: wiring only eval would leave compare
// stuck on the default.
func TestParseWebbenchCompareArgsTimeout(t *testing.T) {
	a, err := parseWebbenchCompareArgs([]string{"--dataset", "d.jsonl", "--predictions", "p.json", "--timeout", "5m"})
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if a.timeout != 5*time.Minute {
		t.Errorf("--timeout 5m must land in webbenchCompareArgs.timeout, got %s", a.timeout)
	}
	if a.preds != "p.json" {
		t.Errorf("predictions = %q", a.preds)
	}

	// Omitted flag -> zero -> the built-in default downstream.
	a, err = parseWebbenchCompareArgs([]string{"--dataset", "d.jsonl"})
	if err != nil {
		t.Fatalf("parse (no timeout): %v", err)
	}
	if a.timeout != 0 {
		t.Errorf("omitted --timeout must leave compare timeout zero, got %s", a.timeout)
	}

	// --dataset is required.
	if _, err := parseWebbenchCompareArgs([]string{"--timeout", "5m"}); err == nil {
		t.Error("missing --dataset must be an error")
	}
}
