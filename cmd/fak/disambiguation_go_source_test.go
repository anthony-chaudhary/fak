package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"testing"
)

func TestDisambiguationGoSourceSelfTestCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"go-source-self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.GoSourceSelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Candidates) != 3 || !report.Deterministic || !report.TestsExcluded || !report.GeneratedExcluded || !report.UnexportedExcluded {
		t.Fatalf("report=%#v", report)
	}
}
