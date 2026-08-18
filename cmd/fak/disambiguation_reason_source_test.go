package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func TestDisambiguationReasonSourceSelfTestCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDisambiguation(&stdout, &stderr, []string{"reason-source-self-test", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.ReasonSourceSelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Terms) != 6 || !report.IncompatibleRejected || !report.CrossPackageAliasAllowed {
		t.Fatalf("report=%#v", report)
	}
}
