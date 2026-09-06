package jsonlledger

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type row struct {
	Date string `json:"date"`
	N    int    `json:"n"`
}

func TestParseSkipsBlankMalformedAndRejected(t *testing.T) {
	content := "" +
		`{"date":"2026-07-01","n":1}` + "\n" +
		"\n" + // blank line skipped
		"   \n" + // whitespace-only skipped
		`{"date":"","n":2}` + "\n" + // rejected by keep (empty date)
		`{not json}` + "\n" + // malformed skipped
		`{"date":"2026-07-02","n":3}` + "\n"

	got := Parse(content, func(r row) bool { return r.Date != "" })
	if len(got) != 2 {
		t.Fatalf("want 2 kept rows, got %d: %+v", len(got), got)
	}
	if got[0].Date != "2026-07-01" || got[0].N != 1 {
		t.Errorf("row 0 = %+v, want {2026-07-01 1}", got[0])
	}
	if got[1].Date != "2026-07-02" || got[1].N != 3 {
		t.Errorf("row 1 = %+v, want {2026-07-02 3}", got[1])
	}
}

func TestParseNilKeepAcceptsEveryWellFormedRow(t *testing.T) {
	content := `{"date":"a"}` + "\n" + `{"date":""}` + "\n"
	got := Parse[row](content, nil)
	if len(got) != 2 {
		t.Fatalf("nil keep should accept both well-formed rows, got %d", len(got))
	}
}

func TestLatestBefore(t *testing.T) {
	date := func(r row) string { return r.Date }
	// tiebreak reuses a field; model it with a second helper over a struct that
	// carries a gen stamp.
	type grow struct {
		Date string
		Gen  string
	}
	d := func(r grow) string { return r.Date }
	tb := func(r grow) string { return r.Gen }

	prior := []grow{
		{Date: "2026-07-01", Gen: "a"},
		{Date: "2026-07-03", Gen: "b"},
		{Date: "2026-07-02", Gen: "c"},
	}
	// newest by date is 2026-07-03/b
	got, ok := LatestBefore(grow{Date: "2026-07-04", Gen: "self"}, prior, d, tb)
	if !ok || got.Gen != "b" {
		t.Fatalf("want newest row b, got %+v ok=%v", got, ok)
	}

	// self-exclusion: the reference row's own non-empty tiebreak is skipped.
	got, ok = LatestBefore(grow{Date: "z", Gen: "b"}, prior, d, tb)
	if !ok || got.Gen != "c" {
		t.Fatalf("want c after excluding self gen b, got %+v ok=%v", got, ok)
	}

	// empty prior -> zero,false
	if _, ok := LatestBefore(row{}, nil, date, date); ok {
		t.Errorf("empty prior should return ok=false")
	}

	// all prior rows excluded by self-tiebreak -> zero,false
	allSelf := []grow{
		{Date: "2026-07-01", Gen: "self"},
		{Date: "2026-07-02", Gen: "self"},
	}
	if _, ok := LatestBefore(grow{Date: "2026-07-03", Gen: "self"}, allSelf, d, tb); ok {
		t.Errorf("all prior matching self tiebreak should return ok=false")
	}

	// same date, tiebreak breaks tie
	sameDate := []grow{
		{Date: "2026-07-01", Gen: "alpha"},
		{Date: "2026-07-01", Gen: "beta"},
	}
	if got, ok := LatestBefore(grow{Date: "2026-07-02", Gen: "ref"}, sameDate, d, tb); !ok || got.Gen != "beta" {
		t.Fatalf("want beta by tiebreak, got %+v ok=%v", got, ok)
	}

	// same date and same tiebreak: later element in prior wins (stable ordering)
	sameDateSameTiebreak := []grow{
		{Date: "2026-07-01", Gen: "same"},
		{Date: "2026-07-01", Gen: "same"},
	}
	sameDateSameTiebreak[0].Date = "2026-07-01-first"
	sameDateSameTiebreak[1].Date = "2026-07-01-second"
	dateKey := func(r grow) string { return "2026-07-01" } // both return same date
	if got, ok := LatestBefore(grow{Date: "2026-07-02", Gen: "ref"}, sameDateSameTiebreak, dateKey, tb); !ok || got.Date != "2026-07-01-second" {
		t.Fatalf("want later candidate on tie, got %+v ok=%v", got, ok)
	}
}

func TestLatestBeforeZeroAllocations(t *testing.T) {
	type grow struct {
		Date string
		Gen  string
	}
	d := func(r grow) string { return r.Date }
	tb := func(r grow) string { return r.Gen }

	prior := []grow{
		{Date: "2026-07-01", Gen: "a"},
		{Date: "2026-07-03", Gen: "b"},
		{Date: "2026-07-02", Gen: "c"},
	}
	ref := grow{Date: "2026-07-04", Gen: "self"}

	// Warm up
	LatestBefore(ref, prior, d, tb)

	allocs := testing.AllocsPerRun(100, func() {
		got, ok := LatestBefore(ref, prior, d, tb)
		if !ok || got.Gen != "b" {
			t.Fatalf("unexpected result: %+v, ok=%v", got, ok)
		}
	})
	if allocs != 0 {
		t.Fatalf("want 0 allocs/op, got %f", allocs)
	}
}

func TestParseHandlesLineLongerThanDefaultScannerBuffer(t *testing.T) {
	// bufio.Scanner's default 64 KiB token limit would drop this line; the 1 MiB
	// buffer this package configures must let it through.
	long := make([]byte, 200*1024)
	for i := range long {
		long[i] = 'x'
	}
	content := `{"date":"d","tag":"` + string(long) + `"}` + "\n"
	got := Parse(content, func(r row) bool { return r.Date != "" })
	if len(got) != 1 {
		t.Fatalf("want 1 row for a 200 KiB line, got %d (buffer not enlarged?)", len(got))
	}
}

func jsonl(rows ...row) string {
	var b strings.Builder
	for _, r := range rows {
		line, _ := json.Marshal(r)
		b.Write(line)
		b.WriteByte('\n')
	}
	return b.String()
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func appendRow(acc []row, r row) []row { return append(acc, r) }

// counting wraps appendRow so a test can assert how many rows a fold touched —
// the difference between a delta fold and a full re-fold.
func counting(calls *int) func([]row, row) []row {
	return func(acc []row, r row) []row {
		*calls++
		return appendRow(acc, r)
	}
}

// TestTailFoldFoldsOnlyAppendedDelta is AC1: after a first fold, appending M
// rows makes the next fold touch exactly those M rows — not the whole file —
// while still producing the same aggregate a whole-file Parse would.
func TestTailFoldFoldsOnlyAppendedDelta(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ledger.jsonl")
	first := jsonl(row{"2026-07-01", 1}, row{"2026-07-02", 2}, row{"2026-07-03", 3})
	writeFile(t, p, first)

	calls := 0
	ck, err := TailFold(p, Checkpoint[[]row]{}, nil, counting(&calls))
	if err != nil {
		t.Fatal(err)
	}
	if calls != 3 {
		t.Fatalf("first fold: want 3 step calls, got %d", calls)
	}

	appended := jsonl(row{"2026-07-04", 4}, row{"2026-07-05", 5})
	writeFile(t, p, first+appended)

	calls = 0
	ck, err = TailFold(p, ck, nil, counting(&calls))
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("second fold: want 2 delta step calls, got %d", calls)
	}
	if want := Parse[row](first+appended, nil); !reflect.DeepEqual(ck.State, want) {
		t.Fatalf("delta fold state %+v != whole-file Parse %+v", ck.State, want)
	}
}

// TestTailFoldNoChangeFoldsNothing: re-folding an unchanged file touches no rows
// and preserves the aggregate.
func TestTailFoldNoChangeFoldsNothing(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ledger.jsonl")
	content := jsonl(row{"2026-07-01", 1}, row{"2026-07-02", 2})
	writeFile(t, p, content)

	ck, _ := TailFold(p, Checkpoint[[]row]{}, nil, appendRow)
	calls := 0
	ck2, err := TailFold(p, ck, nil, counting(&calls))
	if err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("unchanged file: want 0 step calls, got %d", calls)
	}
	if !reflect.DeepEqual(ck2.State, ck.State) {
		t.Fatalf("state changed across a no-op fold: %+v vs %+v", ck2.State, ck.State)
	}
}

// TestTailFoldTruncationRefolds is AC2 (shrink): a file that shrank below the
// checkpoint offset is fully re-folded, matching a whole-file Parse.
func TestTailFoldTruncationRefolds(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ledger.jsonl")
	writeFile(t, p, jsonl(row{"2026-07-01", 1}, row{"2026-07-02", 2}, row{"2026-07-03", 3}, row{"2026-07-04", 4}))
	ck, _ := TailFold(p, Checkpoint[[]row]{}, nil, appendRow)

	shorter := jsonl(row{"2026-07-01", 1}, row{"2026-07-02", 2})
	writeFile(t, p, shorter)

	calls := 0
	ck, err := TailFold(p, ck, nil, counting(&calls))
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("truncation: want a 2-row full re-fold, got %d step calls", calls)
	}
	if want := Parse[row](shorter, nil); !reflect.DeepEqual(ck.State, want) {
		t.Fatalf("re-fold state %+v != Parse %+v", ck.State, want)
	}
}

// TestTailFoldRewriteRefolds is AC2 (rotation): an in-place rewrite that drops
// the oldest row and appends a new one — a capped ledger's steady state — keeps
// the file the same length yet shifts its bytes. The boundary fingerprint no
// longer matches, so TailFold re-folds the whole file rather than mis-resuming.
func TestTailFoldRewriteRefolds(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ledger.jsonl")
	writeFile(t, p, jsonl(row{"2026-07-01", 1}, row{"2026-07-02", 2}, row{"2026-07-03", 3}))
	ck, _ := TailFold(p, Checkpoint[[]row]{}, nil, appendRow)

	rotated := jsonl(row{"2026-07-02", 2}, row{"2026-07-03", 3}, row{"2026-07-04", 4})
	writeFile(t, p, rotated)

	calls := 0
	ck, err := TailFold(p, ck, nil, counting(&calls))
	if err != nil {
		t.Fatal(err)
	}
	if calls != 3 {
		t.Fatalf("cap rewrite: want a 3-row full re-fold, got %d step calls", calls)
	}
	if want := Parse[row](rotated, nil); !reflect.DeepEqual(ck.State, want) {
		t.Fatalf("re-fold state %+v != Parse %+v", ck.State, want)
	}
}

// TestTailFoldLeavesPartialTrailingLine: a row still being written (no trailing
// newline) is not folded until the writer finishes it, and is then folded once.
func TestTailFoldLeavesPartialTrailingLine(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ledger.jsonl")
	complete := `{"date":"2026-07-01","n":1}` + "\n"
	writeFile(t, p, complete+`{"date":"2026-07-02","n":2`) // second line unterminated

	calls := 0
	ck, err := TailFold(p, Checkpoint[[]row]{}, nil, counting(&calls))
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("want only the complete line folded, got %d step calls", calls)
	}
	if ck.Offset != int64(len(complete)) {
		t.Fatalf("offset %d should stop at the last newline (%d)", ck.Offset, len(complete))
	}

	writeFile(t, p, complete+`{"date":"2026-07-02","n":2}`+"\n") // finish the second line
	calls = 0
	ck, err = TailFold(p, ck, nil, counting(&calls))
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("want the now-complete line folded once, got %d step calls", calls)
	}
	if len(ck.State) != 2 {
		t.Fatalf("want 2 rows after completion, got %d", len(ck.State))
	}
}

// TestTailFoldMissingFile: a checkpoint that points at a deleted file folds to
// the initial state without error (a rotation, not a failure).
func TestTailFoldMissingFile(t *testing.T) {
	ck, err := TailFold(filepath.Join(t.TempDir(), "absent.jsonl"), Checkpoint[[]row]{}, nil, appendRow)
	if err != nil {
		t.Fatalf("missing file should not error: %v", err)
	}
	if len(ck.State) != 0 {
		t.Fatalf("missing file should fold to empty, got %d rows", len(ck.State))
	}
}
