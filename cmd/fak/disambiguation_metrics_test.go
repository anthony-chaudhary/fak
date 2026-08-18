package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/disambiguation"
	"testing"
)

func TestDisambiguationMetricsCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDisambiguation(&stdout, &stderr, []string{"metrics", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report disambiguation.MetricsReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Total == 0 || disambiguation.SumMetrics(report.Freshness) != report.Total || disambiguation.SumMetrics(report.Owners) != report.Total {
		t.Fatalf("report=%#v", report)
	}
}
