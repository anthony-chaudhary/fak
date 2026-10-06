package usagelog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func metricRows() []Row {
	return []Row{
		{Schema: SchemaV1, Seq: 1, TSUnixNano: 1000, Verb: "commit", ExitCode: 0, DurationMS: 10},
		{Schema: SchemaV1, Seq: 2, TSUnixNano: 2000, Verb: "commit", ExitCode: 3, DurationMS: 20},
		{Schema: SchemaV1, Seq: 3, TSUnixNano: 3000, Verb: "commit", ExitCode: 0, DurationMS: 30},
		{Schema: SchemaV1, Seq: 4, TSUnixNano: 4000, Verb: "wip", ExitCode: 0, DurationMS: 500},
		{Schema: "foreign/1", Seq: 5, TSUnixNano: 5000, Verb: "ignore", ExitCode: 0, DurationMS: 1},
	}
}

func TestProjectRowsGroupsByVerbAndOutcome(t *testing.T) {
	p := ProjectRows(metricRows(), MetricOptions{})
	if p.Rows != 4 {
		t.Fatalf("rows = %d, want 4 (foreign schema skipped)", p.Rows)
	}
	if p.Errors != 1 {
		t.Fatalf("errors = %d, want 1", p.Errors)
	}
	if len(p.Series) != 3 {
		t.Fatalf("series = %d, want 3", len(p.Series))
	}
	// Count descending: commit/success (2) first, then wip/success (1) or commit/refused (1).
	if p.Series[0].Verb != "commit" || p.Series[0].Outcome != OutcomeSuccess || p.Series[0].Count != 2 {
		t.Fatalf("series[0] = %+v, want commit/success count 2", p.Series[0])
	}
	if p.Series[0].SumMS != 40 || p.Series[0].MaxMS != 30 || p.Series[0].LastMS != 30 {
		t.Fatalf("commit/success agg = %+v, want sum 40 max 30 last 30", p.Series[0])
	}
	// Ties on count break by verb, then outcome rank.
	if p.Series[1].Verb != "commit" || p.Series[1].Outcome != OutcomeRefused {
		t.Fatalf("series[1] = %+v, want commit/refused", p.Series[1])
	}
	if p.Series[2].Verb != "wip" || p.Series[2].Outcome != OutcomeSuccess {
		t.Fatalf("series[2] = %+v, want wip/success", p.Series[2])
	}
	if p.OldestTSUnixNano != 1000 || p.NewestTSUnixNano != 4000 {
		t.Fatalf("time range = %d..%d, want 1000..4000", p.OldestTSUnixNano, p.NewestTSUnixNano)
	}
}

func TestProjectRowsSinceCutoff(t *testing.T) {
	p := ProjectRows(metricRows(), MetricOptions{Since: time.Unix(0, 3000)})
	if p.Rows != 2 {
		t.Fatalf("rows = %d, want 2 after cutoff", p.Rows)
	}
}

func TestProjectRowsBoundsSeriesAndReportsTruncation(t *testing.T) {
	p := ProjectRows(metricRows(), MetricOptions{MaxSeries: 1})
	if len(p.Series) != 1 {
		t.Fatalf("series = %d, want 1 (bounded)", len(p.Series))
	}
	if p.Truncated != 2 {
		t.Fatalf("truncated = %d, want 2", p.Truncated)
	}
	if p.Rows != 4 {
		t.Fatalf("rows = %d, want 4 (truncation must not drop rows)", p.Rows)
	}
}

func TestRenderPrometheusIsWellFormedAndDeterministic(t *testing.T) {
	p := ProjectRows(metricRows(), MetricOptions{})
	now := time.Unix(1_700_000_000, 0)
	out := p.RenderPrometheus(now)
	if out != p.RenderPrometheus(now) {
		t.Fatal("render is not deterministic")
	}
	for _, want := range []string{
		"# TYPE fak_cli_invocations_total counter",
		`fak_cli_invocations_total{verb="commit",outcome="success"} 2`,
		`fak_cli_invocation_duration_ms_total{verb="commit",outcome="success"} 40`,
		"fak_cli_usage_errors_in_window 1",
		"fak_cli_usage_series_truncated 0",
		"fak_cli_usage_render_timestamp_seconds 1700000000",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n---\n%s", want, out)
		}
	}
}

func TestRenderPrometheusEscapesLabelValues(t *testing.T) {
	p := ProjectRows([]Row{{Schema: SchemaV1, Seq: 1, TSUnixNano: 1, Verb: `we"ird\v`, ExitCode: 0, DurationMS: 1}}, MetricOptions{})
	out := p.RenderPrometheus(time.Unix(1, 0))
	if !strings.Contains(out, `verb="we\"ird\\v"`) {
		t.Fatalf("label value not escaped:\n%s", out)
	}
}

func TestFoldPathMissingJournalIsEmptyNotError(t *testing.T) {
	p, err := FoldPath(filepath.Join(t.TempDir(), "absent.jsonl"), MetricOptions{})
	if err != nil {
		t.Fatalf("FoldPath on missing journal: %v", err)
	}
	if p.Rows != 0 || len(p.Series) != 0 {
		t.Fatalf("missing journal projection = %+v, want empty", p)
	}
	// A freshness gauge is still rendered so a stalled exporter is visible.
	out := p.RenderPrometheus(time.Unix(5, 0))
	if !strings.Contains(out, "fak_cli_usage_render_timestamp_seconds 5") {
		t.Fatalf("render missing freshness gauge:\n%s", out)
	}
}

func TestFoldPathReadsRealJournal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "usage.jsonl")
	lg, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, verb := range []string{"commit", "commit", "wip"} {
		if _, err := lg.Append(Row{Verb: verb, ExitCode: 0, DurationMS: 5}); err != nil {
			t.Fatal(err)
		}
	}
	_ = lg.Close()
	p, err := FoldPath(path, MetricOptions{})
	if err != nil {
		t.Fatalf("FoldPath: %v", err)
	}
	if p.Rows != 3 {
		t.Fatalf("rows = %d, want 3", p.Rows)
	}
	if p.Series[0].Verb != "commit" || p.Series[0].Count != 2 {
		t.Fatalf("series[0] = %+v, want commit count 2", p.Series[0])
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("journal not written: %v", err)
	}
}

// TestReadTailBoundsLargeJournal pins the tail bound: a journal larger than
// maxBytes yields only its trailing rows (in order), with the partial prefix of
// the window dropped, so `fak fleet metrics --serve` cannot stall on a
// multi-hundred-MB usage.jsonl that grows one row per fak invocation forever.
func TestReadTailBoundsLargeJournal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "usage.jsonl")
	lg, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 500; i++ {
		if _, err := lg.Append(Row{Verb: "commit", ExitCode: 0, DurationMS: 1}); err != nil {
			t.Fatal(err)
		}
	}
	_ = lg.Close()

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	full, err := ReadRows(path)
	if err != nil {
		t.Fatal(err)
	}
	// A tail window far smaller than the file must return FEWER rows than the
	// full read, and they must be the LAST rows (highest seq), in order.
	tail, err := ReadTail(path, info.Size()/4)
	if err != nil {
		t.Fatalf("ReadTail: %v", err)
	}
	if len(tail) == 0 || len(tail) >= len(full) {
		t.Fatalf("tail rows = %d, full = %d; want a non-empty strict subset", len(tail), len(full))
	}
	if tail[len(tail)-1].Seq != full[len(full)-1].Seq {
		t.Fatalf("tail must reach the newest row: tail last seq=%d, full last seq=%d", tail[len(tail)-1].Seq, full[len(full)-1].Seq)
	}
	for i := 1; i < len(tail); i++ {
		if tail[i].Seq != tail[i-1].Seq+1 {
			t.Fatalf("tail rows out of order/gapped at %d: %d then %d", i, tail[i-1].Seq, tail[i].Seq)
		}
	}

	// A maxBytes larger than the file reads everything, with no partial-line drop.
	all, err := ReadTail(path, info.Size()+1)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != len(full) {
		t.Fatalf("bounded-by-file tail = %d rows, want full %d", len(all), len(full))
	}
}
