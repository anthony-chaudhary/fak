package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"testing"
)

func TestDisambiguationLifecycleSourceSelfTestCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"lifecycle-source-self-test", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.LifecycleSourceSelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Ladders) != 3 || !report.IncompatibleSpellingsRejected {
		t.Fatalf("report=%#v", report)
	}
}
