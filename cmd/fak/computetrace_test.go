package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func TestComputeTraceCaptureAndSummary(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.json")
	var out, errOut bytes.Buffer
	if code := runComputeTrace(&out, &errOut, []string{"capture", "--out", path, "--limit", "1", "--run", "r", "--request", "q"}); code != 0 {
		t.Fatalf("capture code=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	if code := runComputeTrace(&out, &errOut, []string{"summary", "--in", path}); code != 0 {
		t.Fatalf("summary code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "schema=fak.compute_trace.v1 events=1 dropped=0") {
		t.Fatalf("summary=%q", out.String())
	}
}
