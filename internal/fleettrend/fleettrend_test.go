package fleettrend

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
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

// TestFoldedTailMatchesFullRead is AC3: the converted Tail (an incremental
// jsonlledger.TailFold over an in-process checkpoint) yields the same aggregate
// — and the same rendered pane line — as a from-scratch read, across a pure-
// append growth phase and the cap rewrites that shift the ledger's oldest rows
// out. If the delta fold ever diverged from a full read, one of these steps
// would mismatch.
func TestFoldedTailMatchesFullRead(t *testing.T) {
	resetFoldCache()
	defer resetFoldCache()
	path := filepath.Join(t.TempDir(), "history.jsonl")

	const capRows = 4
	for i := 0; i < 12; i++ { // grows past capRows, so later steps rewrite in place
		metrics := map[string]float64{
			"usable":   float64(20 - i),
			"live":     float64(i % 3),
			"sessions": float64(300 + i),
			"escalate": float64(i % 2),
		}
		if _, err := Append(path, metrics, fmt.Sprintf("2026-07-01T%02d:00:00Z", i), capRows); err != nil {
			t.Fatal(err)
		}

		cached := Tail(path, 24) // delta-folded read through the checkpoint cache
		fresh := readRows(path)  // from-scratch whole-file read
		if !reflect.DeepEqual(cached, fresh) {
			t.Fatalf("step %d: folded tail %+v != full read %+v", i, cached, fresh)
		}
		if a, b := RenderLine(cached), RenderLine(fresh); a != b {
			t.Fatalf("step %d: folded render %q != full render %q", i, a, b)
		}
	}
}

// TestFoldedTailAdvancesOffset proves the conversion actually folds incrementally
// rather than silently re-reading: a below-cap append advances the cached
// checkpoint's offset instead of resetting it, and the aggregate still matches a
// full read.
func TestFoldedTailAdvancesOffset(t *testing.T) {
	resetFoldCache()
	defer resetFoldCache()
	path := filepath.Join(t.TempDir(), "history.jsonl")
	key, _ := filepath.Abs(path)

	if _, err := Append(path, map[string]float64{"usable": 1}, "2026-07-01T00:00:00Z", DefaultCap); err != nil {
		t.Fatal(err)
	}
	Tail(path, 24) // warm the checkpoint
	off1 := foldCkpts[key].Offset
	if off1 <= 0 {
		t.Fatalf("first folded read left no offset: %d", off1)
	}

	if _, err := Append(path, map[string]float64{"usable": 2}, "2026-07-01T01:00:00Z", DefaultCap); err != nil {
		t.Fatal(err)
	}
	Tail(path, 24)
	off2 := foldCkpts[key].Offset
	if off2 <= off1 {
		t.Fatalf("below-cap append should advance the fold offset, got %d -> %d", off1, off2)
	}
	if a, b := RenderLine(Tail(path, 24)), RenderLine(readRows(path)); a != b {
		t.Fatalf("resumed read diverged from full read: %q vs %q", a, b)
	}
}

func TestWindowRatesArithmetic(t *testing.T) {
	// A synthetic 6-hour window: lands 10→22 (12 over 6h = 2.0/hr), resumes
	// 100→130 (30 over 6h = 5.0/hr), deaths 4→10 (6 over 6h = 1.0/hr). Goodput
	// = 12 / (12 + 6) = 66.67%.
	rows := []map[string]any{
		{"ts": "2026-07-01T00:00:00Z", "lands": 10, "resumes": 100, "deaths": 4, "lands_witnessed": 1},
		{"ts": "2026-07-01T03:00:00Z", "lands": 16, "resumes": 115, "deaths": 7, "lands_witnessed": 1},
		{"ts": "2026-07-01T06:00:00Z", "lands": 22, "resumes": 130, "deaths": 10, "lands_witnessed": 1},
	}
	r, ok := WindowRates(rows)
	if !ok {
		t.Fatal("WindowRates(counter rows) reported no data")
	}
	if !r.Lands.Present || r.Lands.PerHour != 2 || r.Lands.Delta != 12 {
		t.Fatalf("lands = %+v, want 2.0/hr delta 12", r.Lands)
	}
	if !r.Resumes.Present || r.Resumes.PerHour != 5 {
		t.Fatalf("resumes = %+v, want 5.0/hr", r.Resumes)
	}
	if !r.Deaths.Present || r.Deaths.PerHour != 1 {
		t.Fatalf("deaths = %+v, want 1.0/hr", r.Deaths)
	}
	if !r.GoodputPresent || r.Goodput < 0.66 || r.Goodput > 0.67 {
		t.Fatalf("goodput = %v (present=%v), want ~0.6667", r.Goodput, r.GoodputPresent)
	}
	if r.WindowHours != 6 || r.Ticks != 3 || !r.LandsWitnessed {
		t.Fatalf("window = %.1fh ticks=%d witnessed=%v, want 6h/3/true", r.WindowHours, r.Ticks, r.LandsWitnessed)
	}
}

func TestWindowRatesGuards(t *testing.T) {
	// No counters at all → not derivable.
	if _, ok := WindowRates([]map[string]any{{"ts": "2026-07-01T00:00:00Z", "usable": 3}}); ok {
		t.Fatal("gauge-only window should report no rate data")
	}
	// A single counter-bearing row cannot form a rate.
	single := []map[string]any{{"ts": "2026-07-01T00:00:00Z", "lands": 5}}
	if r, _ := WindowRates(single); r.Lands.Present {
		t.Fatalf("single point yielded a rate: %+v", r.Lands)
	}
	// A counter reset (last < first) clamps to a 0 delta, never a negative rate.
	reset := []map[string]any{
		{"ts": "2026-07-01T00:00:00Z", "lands": 20},
		{"ts": "2026-07-01T02:00:00Z", "lands": 3},
	}
	if r, _ := WindowRates(reset); !r.Lands.Present || r.Lands.PerHour != 0 || r.Lands.Delta != 0 {
		t.Fatalf("counter reset = %+v, want clamped 0/hr", r.Lands)
	}
	// deaths present but lands absent → goodput has no numerator.
	noLands := []map[string]any{
		{"ts": "2026-07-01T00:00:00Z", "deaths": 1},
		{"ts": "2026-07-01T01:00:00Z", "deaths": 3},
	}
	if r, _ := WindowRates(noLands); r.GoodputPresent {
		t.Fatalf("goodput present without lands: %+v", r)
	}
}

func TestRenderThroughput(t *testing.T) {
	if got := RenderThroughput(nil); got != "" {
		t.Fatalf("empty throughput = %q, want empty", got)
	}
	rows := []map[string]any{
		{"ts": "2026-07-01T00:00:00Z", "lands": 10, "resumes": 100, "deaths": 4},
		{"ts": "2026-07-01T06:00:00Z", "lands": 22, "resumes": 130, "deaths": 10},
	}
	line := RenderThroughput(rows)
	for _, want := range []string{
		"throughput: lands 2.0/hr", "resumes 5.0/hr", "deaths 1.0/hr",
		"goodput 67%", "over 6.0h · 2 ticks", "[lands: self-reported]",
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("throughput %q missing %q", line, want)
		}
	}
	// A git-witnessed lands total flips the provenance tag; absent counters read n/a.
	witnessed := RenderThroughput([]map[string]any{
		{"ts": "2026-07-01T00:00:00Z", "lands": 1, "lands_witnessed": 1},
		{"ts": "2026-07-01T02:00:00Z", "lands": 5, "lands_witnessed": 1},
	})
	if !strings.Contains(witnessed, "[lands: git-witnessed]") || !strings.Contains(witnessed, "resumes n/a") {
		t.Fatalf("witnessed throughput = %q", witnessed)
	}
}

func TestThroughputCountersPersist(t *testing.T) {
	// The counter columns MetricsOf reads from the throughput seam round-trip
	// through Append into the ledger the rates derive from — no new ledger.
	snap := map[string]any{
		"accounts":   map[string]any{"usable": 2},
		"throughput": map[string]any{"lands": 12, "resumes": 30, "deaths": 5, "lands_witness": "git"},
	}
	m := MetricsOf(snap)
	if m["lands"] != 12 || m["resumes"] != 30 || m["deaths"] != 5 || m[landsWitnessedKey] != 1 {
		t.Fatalf("MetricsOf throughput = %+v", m)
	}
	path := filepath.Join(t.TempDir(), "history.jsonl")
	if _, err := Append(path, m, "2026-07-01T00:00:00Z", DefaultCap); err != nil {
		t.Fatal(err)
	}
	rows := Tail(path, 24)
	if len(rows) != 1 || number(rows[0]["lands"]) != 12 || number(rows[0][landsWitnessedKey]) != 1 {
		t.Fatalf("persisted throughput row = %+v", rows)
	}
	// A snapshot with no throughput object leaves the counter columns unset.
	bare := MetricsOf(map[string]any{"accounts": map[string]any{"usable": 1}})
	if _, ok := bare["lands"]; ok {
		t.Fatalf("bare snapshot set a lands counter: %+v", bare)
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
