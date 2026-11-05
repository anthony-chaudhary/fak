package report_test

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
	"github.com/anthony-chaudhary/fak/pkg/billing/report"
)

// sampleRecords builds three records spanning two UTC days, exercising the
// full v1 field surface: optional fields set and unset, explicit-zero
// pointers, and unreported (nil) axes.
func sampleRecords() []billing.UsageRecord {
	completed := d2.Add(30 * time.Millisecond)
	zero := int64(0)
	return []billing.UsageRecord{
		{
			SchemaVersion:    billing.SchemaVersionV1,
			EventID:          "evt-a",
			RequestID:        "req-a",
			TenantID:         "tenant-1",
			KeyID:            "key-1",
			Model:            "model-x",
			ModelRevision:    "rev-3",
			Backend:          "upstream-1",
			Route:            "route-main",
			RequestedAt:      d1,
			CompletedAt:      &completed,
			PromptTokens:     i64(1200),
			CompletionTokens: i64(300),
			CachedTokens:     i64(256),
			Classification:   billing.UsageRecorded,
			PriceBookVersion: "stamp-1",
			WitnessSHA:       "witness-hex",
		},
		{
			SchemaVersion:    billing.SchemaVersionV1,
			EventID:          "evt-b",
			TenantID:         "tenant-1",
			Model:            "model-x",
			RequestedAt:      d2,
			PromptTokens:     i64(800),
			CompletionTokens: nil, // unreported axis must survive the round trip
			CachedTokens:     &zero,
			Classification:   billing.UsageRecorded,
		},
		{
			SchemaVersion:  billing.SchemaVersionV1,
			EventID:        "evt-c",
			TenantID:       "tenant-2",
			Model:          "model-y",
			RequestedAt:    d2,
			Classification: billing.UsageMissing,
		},
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestJSONLRoundTrip(t *testing.T) {
	dir := t.TempDir()
	st, err := report.OpenStore(dir)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	want := sampleRecords()
	for _, rec := range want {
		if err := st.Append(rec); err != nil {
			t.Fatalf("append %s: %v", rec.EventID, err)
		}
	}
	got, err := st.Read(report.ReadOptions{})
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("replayed %d records, want %d", len(got), len(want))
	}
	for i := range want {
		if !reflect.DeepEqual(got[i], want[i]) {
			t.Fatalf("record %d differs after round trip:\n got %+v\nwant %+v", i, got[i], want[i])
		}
	}
	if err := st.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestJSONLAppendAfterRestartReplaysAndContinues(t *testing.T) {
	dir := t.TempDir()

	// First process: append one record per day, then exit.
	st1, err := report.OpenStore(dir)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	first := sampleRecords()
	if err := st1.Append(first[0]); err != nil { // day 1
		t.Fatalf("append: %v", err)
	}
	if err := st1.Append(first[1]); err != nil { // day 2
		t.Fatalf("append: %v", err)
	}
	if err := st1.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Restart: a fresh store over the same directory replays both lines.
	st2, err := report.OpenStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer st2.Close()
	got, err := st2.Read(report.ReadOptions{})
	if err != nil {
		t.Fatalf("replay after restart: %v", err)
	}
	if len(got) != 2 || got[0].EventID != "evt-a" || got[1].EventID != "evt-b" {
		t.Fatalf("restart replay wrong: %+v", got)
	}

	// Appending after restart continues the SAME day file, not a new one.
	if err := st2.Append(first[2]); err != nil { // also day 2
		t.Fatalf("append after restart: %v", err)
	}
	got, err = st2.Read(report.ReadOptions{})
	if err != nil {
		t.Fatalf("re-read: %v", err)
	}
	if len(got) != 3 || got[2].EventID != "evt-c" {
		t.Fatalf("post-append replay wrong: %+v", got)
	}

	// Exactly one file per day on disk; day 2 now carries two lines.
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("list dir: %v", err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	if len(entries) != 2 {
		t.Fatalf("directory holds %d files (%v), want one per day", len(entries), names)
	}
	data, err := os.ReadFile(filepath.Join(dir, report.DayFile("2026-11-04")))
	if err != nil {
		t.Fatalf("read day-2 file: %v", err)
	}
	lines := 0
	for _, b := range data {
		if b == '\n' {
			lines++
		}
	}
	if lines != 2 {
		t.Fatalf("day-2 file holds %d lines, want 2", lines)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestJSONLReadStopsAtMalformedLine(t *testing.T) {
	dir := t.TempDir()
	st, err := report.OpenStore(dir)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer st.Close()

	good := sampleRecords()[0]
	line, _ := json.Marshal(good)
	data := append(append(line, '\n'), []byte("{'not json")...)
	dayPath := filepath.Join(dir, report.DayFile("2026-11-03"))
	if err := os.WriteFile(dayPath, data, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	_, err = st.Read(report.ReadOptions{})
	if !errors.Is(err, report.ErrMalformedLine) {
		t.Fatalf("want ErrMalformedLine, got %v", err)
	}
	if !errors.Is(err, billing.ErrMalformedPayload) {
		t.Fatalf("malformed line must wrap the wire sentinel: %v", err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestJSONLReadRefusesSmuggledFields(t *testing.T) {
	dir := t.TempDir()
	st, err := report.OpenStore(dir)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer st.Close()

	good := sampleRecords()[0]
	line, _ := json.Marshal(good)
	// Inject a body-carrying field outside the v1 schema: strict decoding
	// must refuse it exactly as on the wire.
	smuggled := append([]byte{}, line...)
	smuggled = smuggled[:len(smuggled)-1]
	smuggled = append(smuggled, `,"prompt_body":"secret"`...)
	smuggled = append(smuggled, '}')
	smuggled = append(smuggled, '\n')
	dayPath := filepath.Join(dir, report.DayFile("2026-11-03"))
	if err := os.WriteFile(dayPath, smuggled, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	_, err = st.Read(report.ReadOptions{})
	if !errors.Is(err, report.ErrMalformedLine) {
		t.Fatalf("smuggled payload must be refused, got %v", err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestJSONLDayWindowFilter(t *testing.T) {
	dir := t.TempDir()
	st, err := report.OpenStore(dir)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer st.Close()
	recs := sampleRecords()
	for _, r := range recs {
		if err := st.Append(r); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	// A third-day record extends the window.
	third := recs[0]
	third.EventID = "evt-d"
	third.RequestedAt = d3
	if err := st.Append(third); err != nil {
		t.Fatalf("append third day: %v", err)
	}

	check := func(name string, opts report.ReadOptions, wantIDs ...string) {
		t.Helper()
		got, err := st.Read(opts)
		if err != nil {
			t.Fatalf("%s: read: %v", name, err)
		}
		var ids []string
		for _, r := range got {
			ids = append(ids, r.EventID)
		}
		if !reflect.DeepEqual(ids, wantIDs) {
			t.Fatalf("%s: replayed %v, want %v", name, ids, wantIDs)
		}
	}

	check("all days", report.ReadOptions{}, "evt-a", "evt-b", "evt-c", "evt-d")
	check("middle day only", report.ReadOptions{Since: d2, Until: d2}, "evt-b", "evt-c")
	check("from day 2", report.ReadOptions{Since: d2}, "evt-b", "evt-c", "evt-d")
	check("through day 2", report.ReadOptions{Until: d2}, "evt-a", "evt-b", "evt-c")
	check("empty window", report.ReadOptions{Since: d1, Until: d1}, "evt-a")
}

// fak-test:runtime fast est=5ms lane=default
func TestJSONLAppendValidatesBeforeWriting(t *testing.T) {
	dir := t.TempDir()
	st, err := report.OpenStore(dir)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	defer st.Close()

	bad := sampleRecords()[0]
	bad.Model = ""
	err = st.Append(bad)
	if !errors.Is(err, billing.ErrInvalidRecord) {
		t.Fatalf("want ErrInvalidRecord, got %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("list dir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("invalid record left %d files behind", len(entries))
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestJSONLClosedStoreRefusesAppendButStillReads(t *testing.T) {
	dir := t.TempDir()
	st, err := report.OpenStore(dir)
	if err != nil {
		t.Fatalf("OpenStore: %v", err)
	}
	recs := sampleRecords()
	if err := st.Append(recs[0]); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := st.Close(); err != nil {
		t.Fatalf("second close must be a no-op: %v", err)
	}

	if err := st.Append(recs[1]); !errors.Is(err, report.ErrStoreClosed) {
		t.Fatalf("append after close: want ErrStoreClosed, got %v", err)
	}
	got, err := st.Read(report.ReadOptions{})
	if err != nil {
		t.Fatalf("read after close: %v", err)
	}
	if len(got) != 1 || got[0].EventID != "evt-a" {
		t.Fatalf("read after close wrong: %+v", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestJSONLNILStoreFailsClosed(t *testing.T) {
	var st *report.Store
	rec := sampleRecords()[0]
	if err := st.Append(rec); !errors.Is(err, report.ErrStoreClosed) {
		t.Fatalf("nil-store append: want ErrStoreClosed, got %v", err)
	}
	if _, err := st.Read(report.ReadOptions{}); !errors.Is(err, report.ErrStoreClosed) {
		t.Fatalf("nil-store read: want ErrStoreClosed, got %v", err)
	}
	if st.Dir() != "" {
		t.Fatal("nil-store Dir must report empty")
	}
}
