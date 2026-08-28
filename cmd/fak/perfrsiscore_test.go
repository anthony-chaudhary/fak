package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func runPerfRSI(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var out, err bytes.Buffer
	c := runPerformanceRSIScorecard(&out, &err, args)
	return c, out.String(), err.String()
}
func fixturePath() string {
	return filepath.Join("..", "..", "internal", "perfrsiscore", "testdata", "complete.json")
}

func TestPerformanceRSIRenderers(t *testing.T) {
	for _, tc := range [][]string{{"--input", fixturePath()}, {"--input", fixturePath(), "--json"}, {"--input", fixturePath(), "--markdown"}} {
		code, out, err := runPerfRSI(t, tc...)
		if code != 0 {
			t.Fatalf("%v code=%d err=%s", tc, code, err)
		}
		if !strings.Contains(out, "cycle_time") {
			t.Fatalf("%v missing dimension", tc)
		}
	}
}
func TestPerformanceRSIComparison(t *testing.T) {
	code, out, err := runPerfRSI(t, "--input", fixturePath(), "--json")
	if code != 0 {
		t.Fatal(err)
	}
	var p map[string]any
	if json.Unmarshal([]byte(out), &p) != nil {
		t.Fatal("bad json")
	}
	p["snapshot"] = "prior"
	b, _ := json.Marshal(p)
	path := filepath.Join(t.TempDir(), "prior.json")
	if os.WriteFile(path, b, 0600) != nil {
		t.Fatal("write")
	}
	code, out, err = runPerfRSI(t, "--input", fixturePath(), "--prior", path, "--json")
	if code != 0 || !strings.Contains(out, `"prior_snapshot": "prior"`) {
		t.Fatalf("code=%d out=%s err=%s", code, out, err)
	}
}
func TestPerformanceRSIRequiresInput(t *testing.T) {
	code, _, _ := runPerfRSI(t)
	if code != 2 {
		t.Fatal(code)
	}
}

func TestPerformanceRSICycleWitness(t *testing.T) {
	witness := filepath.Join("..", "..", "docs", "_witnesses", "issue-9780-performance-rsi-cycle.json")
	code, out, err := runPerfRSI(t, "--input", witness, "--json")
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, err)
	}
	var report struct {
		Dimensions []struct {
			ID      string   `json:"id"`
			Status  string   `json:"status"`
			Current *float64 `json:"current"`
		} `json:"dimensions"`
	}
	if err := json.Unmarshal([]byte(out), &report); err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{
		"cycle_time": false, "evaluation_latency": false,
		"experiment_throughput": false, "automation_coverage": false,
	}
	for _, d := range report.Dimensions {
		if _, ok := want[d.ID]; ok {
			if d.Status == "UNKNOWN" || d.Current == nil {
				t.Errorf("%s remained UNKNOWN", d.ID)
			}
			want[d.ID] = true
		}
	}
	for id, found := range want {
		if !found {
			t.Errorf("missing derived dimension %s", id)
		}
	}
}
