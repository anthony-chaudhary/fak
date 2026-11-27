package main

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func TestSelectHarnessVerifyEff(t *testing.T) {
	h, err := selectHarness("verifyeff", ".", "main", "", "", "")
	if err != nil {
		t.Fatalf("selectHarness(verifyeff): %v", err)
	}
	if h.MetricName != rsiloop.VerifyEffMetricName || h.LowerBetter {
		t.Fatalf("verifyeff harness should expose higher-better %q, got %q lower=%v",
			rsiloop.VerifyEffMetricName, h.MetricName, h.LowerBetter)
	}
}

func TestVerifyEffImproveCLISequence(t *testing.T) {
	h, err := selectHarness("verifyeff", ".", "main", "", "", "")
	if err != nil {
		t.Fatalf("selectHarness(verifyeff): %v", err)
	}
	j, err := rsiloop.NewJournal("-")
	if err != nil {
		t.Fatalf("NewJournal: %v", err)
	}
	defer j.Close()

	// Capture stdout from runImprove.
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	code := runImprove(h, j, 3, 0, nil)
	w.Close()
	os.Stdout = oldStdout

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(r); err != nil {
		t.Fatalf("read stdout: %v", err)
	}
	out := buf.String()
	if code != 0 {
		t.Fatalf("runImprove exit=%d, want 0\n%s", code, out)
	}
	for _, want := range []string{
		"-> REVERT (kept=false",
		"-> KEEP (kept=true",
		"SUMMARY cycles=3 kept=2",
		"escalated=false",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("output missing %q:\n%s", want, out)
		}
	}
}
