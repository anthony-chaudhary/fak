package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTrendLedger drops a two-row history whose cumulative counters give a
// clean 6-hour window: lands 10→22 (2.0/hr), resumes 100→130 (5.0/hr),
// deaths 4→10 (1.0/hr), goodput 12/(12+6)=67%.
func writeTrendLedger(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "history.jsonl")
	body := `{"ts":"2026-07-01T00:00:00Z","usable":2,"lands":10,"resumes":100,"deaths":4}
{"ts":"2026-07-01T06:00:00Z","usable":2,"lands":22,"resumes":130,"deaths":10}
`
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunFleetTrendThroughputRow(t *testing.T) {
	path := writeTrendLedger(t)
	var out, errb bytes.Buffer
	if code := runFleetTrend(&out, &errb, strings.NewReader(""), []string{"-ledger", path, "-show"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"throughput: lands 2.0/hr", "resumes 5.0/hr", "deaths 1.0/hr",
		"goodput 67%", "over 6.0h · 2 ticks", "[lands: self-reported]",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("--show output %q missing %q", got, want)
		}
	}
}

func TestRunFleetTrendThroughputJSON(t *testing.T) {
	path := writeTrendLedger(t)
	var out, errb bytes.Buffer
	if code := runFleetTrend(&out, &errb, strings.NewReader(""), []string{"-ledger", path, "-json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var payload struct {
		Throughput struct {
			Lands struct {
				PerHour float64 `json:"per_hour"`
			} `json:"lands"`
			Goodput     float64 `json:"goodput"`
			WindowHours float64 `json:"window_hours"`
		} `json:"throughput"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("decode json: %v\n%s", err, out.String())
	}
	if payload.Throughput.Lands.PerHour != 2 || payload.Throughput.WindowHours != 6 {
		t.Fatalf("throughput json = %+v", payload.Throughput)
	}
}

// TestRunFleetTrendCounterFreeSilent proves a history with no throughput
// counters renders only the trend line, not a row of n/a figures.
func TestRunFleetTrendCounterFreeSilent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	body := "{\"ts\":\"2026-07-01T00:00:00Z\",\"usable\":3}\n{\"ts\":\"2026-07-01T06:00:00Z\",\"usable\":1}\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runFleetTrend(&out, &errb, strings.NewReader(""), []string{"-ledger", path, "-show"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if strings.Contains(out.String(), "throughput:") {
		t.Fatalf("counter-free history printed a throughput row: %q", out.String())
	}
}
