package jsonlledger

import "testing"

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
