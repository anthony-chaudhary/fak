package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestBenchmarksListOfflineIncludesVCache(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runBenchmarks(&out, &errb, []string{"list", "--offline"}); code != 0 {
		t.Fatalf("benchmarks list exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "vcache") || !strings.Contains(s, "offline") ||
		!strings.Contains(s, "vCache 2x readiness scorecard") {
		t.Fatalf("offline list missing vcache scorecard:\n%s", s)
	}
}

func TestBenchmarksDescribeVCacheShowsScorecardInputs(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runBenchmarks(&out, &errb, []string{"describe", "vcache"}); code != 0 {
		t.Fatalf("benchmarks describe exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{
		"fak vcache bench --json",
		"--telemetry",
		"--anchors-file",
		"--index-out",
		"--plan-out",
		"--two-x",
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("describe missing %q:\n%s", want, s)
		}
	}
}

func TestBenchmarksRunVCacheExecutesScoreGate(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runBenchmarks(&out, &errb, []string{"run", "vcache"}); code != 0 {
		t.Fatalf("benchmarks run exit=%d stderr=%s", code, errb.String())
	}
	var rep struct {
		Schema     string `json:"schema"`
		Status     string `json:"status"`
		TwoXBetter bool   `json:"two_x_better"`
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("run output is not score JSON: %v\n%s", err, out.String())
	}
	if rep.Schema != "fak.vcache.score.v1" || rep.Status != "2x_ready" || !rep.TwoXBetter {
		t.Fatalf("score = %+v, want default vCache 2x gate pass", rep)
	}
}
