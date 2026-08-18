package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessControlStudyCLIReportsNotYetWithoutPairs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "study.json")
	raw := `{"schema":"fak-harness-control-study/1","study_id":"pilot","task_digest":"sha256:` + strings.Repeat("a", 64) + `","min_pairs":2,"rows":[]}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"study", "control", "--input", path})
	var report struct {
		Verdict string   `json:"verdict"`
		Reasons []string `json:"reasons"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if code != 3 || errb.Len() != 0 || report.Verdict != "not_yet" || len(report.Reasons) == 0 {
		t.Fatalf("code=%d stderr=%q report=%+v", code, errb.String(), report)
	}
}
