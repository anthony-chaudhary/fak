package projectreport

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trendreport"
)

func TestRowFromReportNormalizesHorizons(t *testing.T) {
	r := Fold([]Item{
		{Issue: 1, Status: "Todo", Generation: "now", Priority: "P1"},
		{Issue: 2, Status: "Todo", Generation: "gen/now", Priority: "P2"}, // gen/ prefix normalizes
		{Issue: 3, Status: "Backlog", Generation: "second-next", Priority: "P3"},
		{Issue: 4, Status: "Backlog", Generation: "someday", Priority: "P3"}, // off-vocab -> gen_unclassified
	}, FoldOpts{Commit: "c0ffee", Date: "2026-07-11", GeneratedAt: "2026-07-11T00:00:00Z"})
	row := RowFromReport(r)
	if row.Schema != LedgerSchema {
		t.Fatalf("schema = %q, want %q", row.Schema, LedgerSchema)
	}
	if row.Now != 2 {
		t.Fatalf("now = %d, want 2 (now + gen/now)", row.Now)
	}
	if row.SecondNext != 1 {
		t.Fatalf("second_next = %d, want 1", row.SecondNext)
	}
	if row.GenUnclassified != 1 {
		t.Fatalf("gen_unclassified = %d, want 1 (the off-vocab 'someday')", row.GenUnclassified)
	}
	if row.Total != 4 || row.Verdict != "OK" {
		t.Fatalf("row total/verdict = %d/%q, want 4/OK", row.Total, row.Verdict)
	}
}

func TestTrendVsLastDirections(t *testing.T) {
	base := LedgerRow{Date: "2026-07-04", GeneratedAt: "2026-07-04T00:00:00Z", Total: 5, Unclassified: 2}

	first := TrendVsLast(base, nil)
	if first.Direction != "new" {
		t.Fatalf("no prior row => direction %q, want new", first.Direction)
	}

	improved := TrendVsLast(LedgerRow{Date: "2026-07-11", GeneratedAt: "2026-07-11T00:00:00Z", Total: 5, Unclassified: 1}, []LedgerRow{base})
	if improved.Direction != "improved" || improved.UnclassifiedDelta != -1 {
		t.Fatalf("shedding an unclassified item => %q delta %d, want improved -1", improved.Direction, improved.UnclassifiedDelta)
	}

	regressed := TrendVsLast(LedgerRow{Date: "2026-07-11", GeneratedAt: "2026-07-11T00:00:00Z", Total: 6, Unclassified: 3}, []LedgerRow{base})
	if regressed.Direction != "regressed" {
		t.Fatalf("gaining an unclassified item => %q, want regressed", regressed.Direction)
	}

	grew := TrendVsLast(LedgerRow{Date: "2026-07-11", GeneratedAt: "2026-07-11T00:00:00Z", Total: 8, Unclassified: 2}, []LedgerRow{base})
	if grew.Direction != "improved" || grew.TotalDelta != 3 {
		t.Fatalf("more tracked items at flat drift => %q delta %d, want improved +3", grew.Direction, grew.TotalDelta)
	}
}

func TestTrendVsLastSkipsSameGeneration(t *testing.T) {
	// A re-append with the SAME generated_at must not compare a row against itself.
	self := LedgerRow{Date: "2026-07-11", GeneratedAt: "2026-07-11T00:00:00Z", Total: 5, Unclassified: 1}
	tr := TrendVsLast(self, []LedgerRow{self})
	if tr.Direction != "new" {
		t.Fatalf("re-append of the same tick => %q, want new (self excluded)", tr.Direction)
	}
}

func TestLedgerRoundTrip(t *testing.T) {
	row := RowFromReport(Fold([]Item{{Issue: 1, Status: "Todo", Generation: "now"}}, FoldOpts{Date: "2026-07-11"}))
	line, err := trendreport.AppendLedgerLine(row)
	if err != nil {
		t.Fatalf("append: %v", err)
	}
	if strings.Contains(line, "\n") {
		t.Fatalf("append line must not embed a newline: %q", line)
	}
	back := ParseLedger(line + "\n\n") // tolerate a trailing blank line
	if len(back) != 1 {
		t.Fatalf("round-trip parsed %d rows, want 1", len(back))
	}
	if back[0].Now != row.Now || back[0].Total != row.Total || back[0].Unclassified != row.Unclassified {
		t.Fatalf("round-trip mismatch: wrote %+v, parsed %+v", row, back[0])
	}
}

func TestWithTrendAttaches(t *testing.T) {
	r := Fold([]Item{{Issue: 1, Status: "Todo", Generation: "now"}}, FoldOpts{})
	if r.Trend != nil {
		t.Fatalf("fold must not set a trend")
	}
	tr := TrendVsLast(RowFromReport(r), nil)
	r2 := r.WithTrend(tr)
	if r2.Trend == nil || r2.Trend.Direction != "new" {
		t.Fatalf("WithTrend did not attach the trend: %+v", r2.Trend)
	}
	// WithTrend is verdict-neutral: attaching a trend never flips the measured verdict.
	if r2.Verdict != r.Verdict {
		t.Fatalf("WithTrend changed the verdict %q -> %q", r.Verdict, r2.Verdict)
	}
}

func TestSelfcheckPasses(t *testing.T) {
	if err := Selfcheck(); err != nil {
		t.Fatalf("Selfcheck: %v", err)
	}
}
