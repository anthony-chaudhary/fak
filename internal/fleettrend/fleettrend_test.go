package fleettrend

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSpark(t *testing.T) {
	if got := Spark(nil); got != "" {
		t.Fatalf("empty spark = %q, want empty", got)
	}
	if got := Spark([]float64{3, 3, 3}); got != "▁▁▁" {
		t.Fatalf("flat spark = %q", got)
	}
	if got := Spark([]float64{7}); got != "▁" {
		t.Fatalf("single spark = %q", got)
	}
	ramp := []rune(Spark([]float64{0, 1, 2, 3, 4, 5, 6, 7}))
	if len(ramp) != 8 || ramp[0] != '▁' || ramp[len(ramp)-1] != '█' {
		t.Fatalf("ramp = %q", string(ramp))
	}
}

func TestMetricsOf(t *testing.T) {
	snap := map[string]any{
		"sessions": map[string]any{"total": 5, "by_category": map[string]any{"LIVE": 2, "AGENT": 3}},
		"accounts": map[string]any{"usable": 1, "total": 4},
		"system":   map[string]any{"verdict": "NEEDS_YOU", "escalate": 2, "self_healing": 1},
	}
	got := MetricsOf(snap)
	want := map[string]float64{"usable": 1, "live": 2, "sessions": 5, "escalate": 2}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("metric %s = %v, want %v (all=%v)", k, got[k], v, got)
		}
	}
	zero := MetricsOf(map[string]any{})
	for _, k := range []string{"usable", "live", "sessions", "escalate"} {
		if zero[k] != 0 {
			t.Fatalf("partial metric %s = %v, want 0", k, zero[k])
		}
	}
}

func TestLedgerAppendTailAndCap(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "history.jsonl")
	if _, err := Append(path, map[string]float64{"usable": 3, "live": 1, "sessions": 4, "escalate": 0}, "2026-07-01T00:00:00Z", DefaultCap); err != nil {
		t.Fatal(err)
	}
	if _, err := Append(path, map[string]float64{"usable": 2, "live": 1, "sessions": 4, "escalate": 1}, "2026-07-01T01:00:00Z", DefaultCap); err != nil {
		t.Fatal(err)
	}
	rows := Tail(path, 24)
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(rows))
	}
	if number(rows[0]["usable"]) != 3 || number(rows[1]["escalate"]) != 1 || rows[1]["ts"] != "2026-07-01T01:00:00Z" {
		t.Fatalf("rows = %+v", rows)
	}

	capPath := filepath.Join(t.TempDir(), "history.jsonl")
	for i := 0; i < 10; i++ {
		if _, err := Append(capPath, map[string]float64{"usable": float64(i)}, "2026-07-01T00:00:00Z", 3); err != nil {
			t.Fatal(err)
		}
	}
	rows = Tail(capPath, 100)
	if len(rows) != 3 || number(rows[0]["usable"]) != 7 || number(rows[2]["usable"]) != 9 {
		t.Fatalf("cap rows = %+v", rows)
	}
}

func TestTailMissingAndTornLine(t *testing.T) {
	dir := t.TempDir()
	if got := Tail(filepath.Join(dir, "missing.jsonl"), 5); len(got) != 0 {
		t.Fatalf("missing tail = %+v", got)
	}
	path := filepath.Join(dir, "history.jsonl")
	if err := os.WriteFile(path, []byte("{\"ts\":\"a\",\"usable\":3}\n{ this is not json\n{\"ts\":\"b\",\"usable\":1}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	rows := Tail(path, 24)
	if len(rows) != 2 || number(rows[0]["usable"]) != 3 || number(rows[1]["usable"]) != 1 {
		t.Fatalf("torn rows = %+v", rows)
	}
}

func TestRenderLine(t *testing.T) {
	if got := RenderLine(nil); got != "" {
		t.Fatalf("empty render = %q", got)
	}
	one := RenderLine([]map[string]any{{"ts": "a", "usable": 2, "escalate": 0}})
	if !strings.HasPrefix(one, "trend: ") || !strings.Contains(one, "usable 2 ") || strings.Contains(one, "→") || strings.Contains(one, "over") {
		t.Fatalf("single render = %q", one)
	}
	rows := []map[string]any{
		{"ts": "a", "usable": 3, "live": 1, "sessions": 4, "escalate": 0},
		{"ts": "b", "usable": 2, "live": 1, "sessions": 4, "escalate": 0},
		{"ts": "c", "usable": 1, "live": 1, "sessions": 4, "escalate": 1},
	}
	line := RenderLine(rows)
	for _, want := range []string{"usable 3→1", "(-2 over 3)", "escalate 0→1", "(+1 over 3)", "live 1→1"} {
		if !strings.Contains(line, want) {
			t.Fatalf("render %q missing %q", line, want)
		}
	}
	if strings.Contains(line, "live 1→1 ▁▁▁ (") {
		t.Fatalf("flat metric carried a delta: %q", line)
	}
}
