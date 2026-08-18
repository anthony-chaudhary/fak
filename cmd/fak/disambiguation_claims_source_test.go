package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"testing"
)

func TestDisambiguationClaimsSourceSelfTestCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"claims-source-self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.ClaimsSourceSelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.CanonicalTerms) != 6 || !report.MissingBaselineRejected || !report.MissingProvenanceRejected || !report.MissingScopeRejected {
		t.Fatalf("report=%#v", report)
	}
}
