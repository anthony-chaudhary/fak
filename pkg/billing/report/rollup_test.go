package report_test

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
	"github.com/anthony-chaudhary/fak/pkg/billing/report"
)

// i64 boxes a token count for the record pointer fields.
func i64(v int64) *int64 { return &v }

var (
	d1 = time.Date(2026, 11, 3, 8, 0, 0, 0, time.UTC)
	d2 = time.Date(2026, 11, 4, 8, 0, 0, 0, time.UTC)
	d3 = time.Date(2026, 11, 5, 8, 0, 0, 0, time.UTC)
)

// rec builds a valid recorded UsageRecord; nil token pointers stay
// unreported.
func rec(event, tenant, key, model string, at time.Time, prompt, completion, cached *int64) billing.UsageRecord {
	return billing.UsageRecord{
		SchemaVersion:    billing.SchemaVersionV1,
		EventID:          event,
		TenantID:         tenant,
		KeyID:            key,
		Model:            model,
		RequestedAt:      at,
		PromptTokens:     prompt,
		CompletionTokens: completion,
		CachedTokens:     cached,
		Classification:   billing.UsageRecorded,
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRollupGroupsByTenantModelDay(t *testing.T) {
	records := []billing.UsageRecord{
		rec("e1", "t1", "k1", "m1", d1, i64(100), i64(10), i64(5)),
		rec("e2", "t1", "k1", "m1", d1, i64(200), i64(20), i64(0)),
		rec("e3", "t1", "k1", "m1", d2, i64(50), i64(5), nil),
		rec("e4", "t2", "k9", "m1", d1, i64(300), i64(30), nil),
		rec("e5", "t1", "k2", "m2", d1, i64(10), i64(1), nil),
	}
	rows, skipped := report.Rollup(records)
	if skipped != 0 {
		t.Fatalf("skipped %d valid records", skipped)
	}
	if len(rows) != 4 {
		t.Fatalf("got %d rows, want 4: %+v", len(rows), rows)
	}

	wantOrder := [][3]string{
		{"t1", "m1", "2026-11-03"},
		{"t1", "m1", "2026-11-04"},
		{"t1", "m2", "2026-11-03"},
		{"t2", "m1", "2026-11-03"},
	}
	for i, w := range wantOrder {
		got := rows[i]
		if got.TenantID != w[0] || got.Model != w[1] || got.Day != w[2] {
			t.Fatalf("row %d = (%s,%s,%s), want (%s,%s,%s)", i, got.TenantID, got.Model, got.Day, w[0], w[1], w[2])
		}
	}

	a := rows[0] // t1/m1/day1: two fully-reported records
	if a.Records != 2 {
		t.Fatalf("row t1/m1/d1 records=%d, want 2", a.Records)
	}
	if a.Prompt != (report.TokenSum{Value: 300, Known: true}) {
		t.Fatalf("prompt sum %+v, want {300 known}", a.Prompt)
	}
	if a.Completion != (report.TokenSum{Value: 30, Known: true}) {
		t.Fatalf("completion sum %+v, want {30 known}", a.Completion)
	}
	if a.Cached != (report.TokenSum{Value: 5, Known: true}) {
		t.Fatalf("cached sum %+v, want {5 known}", a.Cached)
	}

	b := rows[1] // t1/m1/day2: cached left unreported
	if b.Records != 1 || b.Prompt != (report.TokenSum{Value: 50, Known: true}) {
		t.Fatalf("row t1/m1/d2 wrong: %+v", b)
	}
	if b.Cached.Known {
		t.Fatal("axis with one unreported record must come back unknown")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRollupPreservesUnknownAxes(t *testing.T) {
	records := []billing.UsageRecord{
		// Group A: every axis has at least one unreported contributor.
		rec("a1", "ta", "k1", "m", d1, i64(100), nil, nil),
		rec("a2", "ta", "k1", "m", d1, nil, i64(50), nil),
		// Group B: every axis fully reported by both records.
		rec("b1", "tb", "k1", "m", d1, i64(10), i64(5), i64(2)),
		rec("b2", "tb", "k1", "m", d1, i64(20), i64(4), i64(1)),
	}
	rows, skipped := report.Rollup(records)
	if skipped != 0 || len(rows) != 2 {
		t.Fatalf("rows=%d skipped=%d, want 2 rows 0 skipped", len(rows), skipped)
	}

	a := rows[0]
	if a.Prompt != (report.TokenSum{Value: 100, Known: false}) {
		t.Fatalf("partial prompt sum %+v, want {100 unknown}", a.Prompt)
	}
	if a.Completion != (report.TokenSum{Value: 50, Known: false}) {
		t.Fatalf("partial completion sum %+v, want {50 unknown}", a.Completion)
	}
	if a.Cached != (report.TokenSum{Value: 0, Known: false}) {
		t.Fatalf("all-unreported cached axis %+v, want {0 unknown}", a.Cached)
	}

	b := rows[1]
	if b.Prompt != (report.TokenSum{Value: 30, Known: true}) ||
		b.Completion != (report.TokenSum{Value: 9, Known: true}) ||
		b.Cached != (report.TokenSum{Value: 3, Known: true}) {
		t.Fatalf("fully-reported group wrong: %+v", b)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRollupSkipsInvalidRecordsAndDoesNotDedup(t *testing.T) {
	good := rec("g1", "t1", "k1", "m", d1, i64(100), i64(10), nil)
	bad := good
	bad.EventID = "g2"
	neg := int64(-5)
	bad.PromptTokens = &neg // Validate rejects negative counts

	rows, skipped := report.Rollup([]billing.UsageRecord{good, bad})
	if skipped != 1 {
		t.Fatalf("skipped=%d, want 1", skipped)
	}
	if len(rows) != 1 || rows[0].Prompt.Value != 100 {
		t.Fatalf("invalid record leaked into the sums: %+v", rows)
	}

	// Rollup is a transform, not a ledger: duplicate event IDs both count.
	rows, skipped = report.Rollup([]billing.UsageRecord{good, good})
	if skipped != 0 || len(rows) != 1 || rows[0].Records != 2 || rows[0].Prompt.Value != 200 {
		t.Fatalf("duplicate event ids must both count: rows=%+v skipped=%d", rows, skipped)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRollupDeterministicForShuffledInput(t *testing.T) {
	base := []billing.UsageRecord{
		rec("e1", "t1", "k1", "m1", d1, i64(100), nil, nil),
		rec("e2", "t2", "k1", "m1", d1, i64(200), nil, nil),
		rec("e3", "t1", "k1", "m2", d2, i64(300), nil, nil),
	}
	reversed := []billing.UsageRecord{base[2], base[1], base[0]}

	a, sa := report.Rollup(base)
	b, sb := report.Rollup(reversed)
	if sa != 0 || sb != 0 || len(a) != len(b) {
		t.Fatalf("shuffled input changed the row set: %d vs %d rows", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("row %d differs under input order: %+v vs %+v", i, a[i], b[i])
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDayOfUsesUTCAndHandlesZeroTime(t *testing.T) {
	local := time.Date(2026, 11, 3, 23, 30, 0, 0, time.FixedZone("plus-one-hour-east-of-utc", 3600))
	if got := report.DayOf(rec("e1", "t", "k", "m", local, i64(1), nil, nil)); got != "2026-11-03" {
		t.Fatalf("DayOf %q, want 2026-11-03 (UTC day of the instant)", got)
	}
	var zero billing.UsageRecord
	if got := report.DayOf(zero); got != "0001-01-01" {
		t.Fatalf("DayOf of a zero time %q, want 0001-01-01", got)
	}
}
