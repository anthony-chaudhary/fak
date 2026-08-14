package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestArmbenchPairedReportCLI(t *testing.T) {
	p := filepath.Join("..", "..", "docs", "_witnesses", "armbench-paired-receipts-2026-08-14.json")
	var out, errout bytes.Buffer
	if code := runArmbench(&out, &errout, []string{"paired-report", "--receipts", p}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	if !strings.Contains(out.String(), `"tuned_baseline": "baseline-tuned"`) || !strings.Contains(out.String(), `"claim_check_input"`) {
		t.Fatalf("incomplete output: %s", out.String())
	}
	want, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "armbench-paired-report-2026-08-14.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out.Bytes(), want) {
		t.Fatal("committed paired report is stale")
	}
}
