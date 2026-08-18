package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func TestDisambiguationCacheSourceSelfTestCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDisambiguation(&stdout, &stderr, []string{"cache-source-self-test", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.CacheSourceSelfTestReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Resolutions) != 4 || !report.Pairwise {
		t.Fatalf("report=%#v", report)
	}
}
