package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"testing"
)

func TestDisambiguationOwnershipSourceSelfTestCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"ownership-source-self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.OwnershipSourceSelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Binding.ModuleAtRev == "" || !report.LeafMismatchTyped || !report.LaneMismatchTyped || !report.StampMismatchTyped {
		t.Fatalf("report=%#v", report)
	}
}
