package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/portabilitylab"
)

func TestRunRejectsUnknownFlag(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run([]string{"-bogus"}, &out, &errb); code != 2 {
		t.Fatalf("exit=%d, want 2", code)
	}
	if out.Len() != 0 {
		t.Fatalf("stdout not empty: %q", out.String())
	}
}

func TestRunHelpExitsZero(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run([]string{"-h"}, &out, &errb); code != 0 {
		t.Fatalf("exit=%d, want 0", code)
	}
}

func TestRunWritesPassingReportToStdoutAndOut(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "nested", "report.json")
	var out, errb bytes.Buffer
	if code := run([]string{"-work", filepath.Join(dir, "lab"), "-out", outPath}, &out, &errb); code != 0 {
		t.Fatalf("exit=%d stderr=%q", code, errb.String())
	}
	var printed portabilitylab.Report
	if err := json.Unmarshal(out.Bytes(), &printed); err != nil {
		t.Fatalf("stdout is not a report: %v", err)
	}
	if printed.Schema != portabilitylab.Schema || printed.Verdict != "pass" || !printed.Hermetic {
		t.Fatalf("schema=%q verdict=%q hermetic=%t", printed.Schema, printed.Verdict, printed.Hermetic)
	}
	raw, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("report file: %v", err)
	}
	var written portabilitylab.Report
	if err := json.Unmarshal(raw, &written); err != nil {
		t.Fatalf("report file is not a report: %v", err)
	}
	if written.Schema != portabilitylab.Schema || written.Verdict != "pass" {
		t.Fatalf("file schema=%q verdict=%q", written.Schema, written.Verdict)
	}
}
