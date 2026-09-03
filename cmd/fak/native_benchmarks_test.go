package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativebench"
)

func TestNativeBenchmarksJSONExposesMissingWitnesses(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runNativeBenchmarks(&out, &errb, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var report nativebench.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Complete || len(report.Findings) == 0 || report.Coverage.NativeLeaves == 0 || len(report.Coverage.MissingLeaves) == 0 {
		t.Fatalf("report must honestly expose missing witnesses: %+v", report)
	}
}

func TestNativeBenchmarksCheckFailsOpenCoverageDebt(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runNativeBenchmarks(&out, &errb, []string{"--check"}); code != 1 {
		t.Fatalf("code=%d, want 1; out=%s err=%s", code, out.String(), errb.String())
	}
}

func TestNativeBenchmarksJSONHasNoUnreadableWitnessFindings(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runNativeBenchmarks(&out, &errb, []string{"--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var report nativebench.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	for _, f := range report.Findings {
		if bytes.Contains([]byte(f.Reason), []byte("is not readable")) {
			t.Errorf("falsely unreadable finding for %s: %s", f.Capability, f.Reason)
		}
	}
}
