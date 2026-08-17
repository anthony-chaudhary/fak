package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunToolprocReplayCapturedSummary(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trace.jsonl")
	body := strings.Join([]string{
		`{"id":"a","turn":1,"tool":"read_file","args":{"path":"a"},"read_only":true,"state_epoch":"s","prompt_units":128000,"needed":true,"result_id":"r1","succeeded":true}`,
		`{"id":"b","turn":2,"tool":"read_file","args":{"path":"a"},"read_only":true,"state_epoch":"s","prompt_units":128000,"needed":false,"result_id":"r2","succeeded":true}`,
	}, "\n")
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if code := runToolprocReplay(&out, &stderr, []string{"--trace", path}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	got := out.String()
	for _, want := range []string{"identical trace in every arm", "exact-reuse", "NEEDED SUPPRESSED", "exposure proxies, not measured dollars"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render missing %q:\n%s", want, got)
		}
	}
}
