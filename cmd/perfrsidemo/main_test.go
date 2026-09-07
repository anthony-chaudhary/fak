package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/perfrsiscore"
)

func TestStandardRun(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{})
	if code != 0 {
		t.Fatalf("run() code = %d, stderr = %s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Errorf("expected empty stderr, got %q", stderr.String())
	}
	out := stdout.String()
	for _, needle := range []string{
		"performance RSI: fixture",
		"loop health:",
		"dominant bottleneck: evaluation_latency",
		"cycle_time",
		"improvement_yield",
		"evaluation_latency",
		"receipt_coverage",
	} {
		if !strings.Contains(out, needle) {
			t.Errorf("stdout missing %q; got:\n%s", needle, out)
		}
	}
	if strings.Contains(out, "selfcheck: PASS") {
		t.Errorf("standard run should not output selfcheck PASS line; got:\n%s", out)
	}
}

func TestSelfcheck(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{"-selfcheck"})
	if code != 0 {
		t.Fatalf("run(-selfcheck) code = %d, stderr = %s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Errorf("expected empty stderr, got %q", stderr.String())
	}
	out := stdout.String()
	for _, needle := range []string{
		"performance RSI: fixture",
		"dominant bottleneck: evaluation_latency",
		"selfcheck: PASS (deterministic performance-rsi scorecard)",
	} {
		if !strings.Contains(out, needle) {
			t.Errorf("stdout missing %q; got:\n%s", needle, out)
		}
	}
}

func TestUsageInvalidFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{"-unknown-flag"})
	if code != 2 {
		t.Fatalf("run(-unknown-flag) code = %d, want 2", code)
	}
}

func TestUsageExtraArg(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{"extra-arg"})
	if code != 2 {
		t.Fatalf("run(extra-arg) code = %d, want 2", code)
	}
}

func TestValidateSelfcheck(t *testing.T) {
	evidence, err := perfrsiscore.Decode(strings.NewReader(deterministicFixture))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	report := perfrsiscore.Score(evidence)
	if err := validateSelfcheck(report); err != nil {
		t.Fatalf("validateSelfcheck(valid) failed: %v", err)
	}

	badSchema := report
	badSchema.Schema = "bad-schema"
	if err := validateSelfcheck(badSchema); err == nil {
		t.Errorf("expected error on bad schema, got nil")
	}

	badDims := report
	badDims.Dimensions = report.Dimensions[:len(report.Dimensions)-1]
	if err := validateSelfcheck(badDims); err == nil {
		t.Errorf("expected error on missing dimension, got nil")
	}

	badBottleneck := report
	badBottleneck.DominantBottleneck = "cycle_time"
	if err := validateSelfcheck(badBottleneck); err == nil {
		t.Errorf("expected error on wrong dominant bottleneck, got nil")
	}
}
