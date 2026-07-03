package memview

import (
	"encoding/json"
	"strings"
	"testing"
)

func mustSurface(t *testing.T, title string, fields []string, rows []Row) Surface {
	t.Helper()
	s, err := NewSurface(title, fields, rows)
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}
	return s
}

// TestNewSurfaceRefusesRaggedRows proves the fail-closed shape guard: a row whose
// value count does not match Fields is refused at construction, before any encoder
// can produce a TOON header whose declared field count lies about the row shape.
func TestNewSurfaceRefusesRaggedRows(t *testing.T) {
	_, err := NewSurface("t", []string{"a", "b"}, []Row{{"1", "2"}, {"only-one"}})
	if err == nil {
		t.Fatal("expected an error for a ragged row, got nil")
	}
	if !strings.Contains(err.Error(), "row 1") {
		t.Fatalf("error should name the offending row: %v", err)
	}
}

// TestEncodeUnknownFormatFailsClosed proves an unregistered format name is refused
// with the known-set named, not silently rendered under some default.
func TestEncodeUnknownFormatFailsClosed(t *testing.T) {
	s := mustSurface(t, "t", []string{"a"}, []Row{{"1"}})
	_, err := Encode(Format("yaml"), s)
	if err == nil {
		t.Fatal("expected an error for an unknown format")
	}
	if !strings.Contains(err.Error(), "yaml") || !strings.Contains(err.Error(), "markdown") {
		t.Fatalf("error should name the bad format and the known set: %v", err)
	}
}

// TestKnownFormatsIsSortedAndComplete pins the three shipped formats and their
// deterministic ordering (the `--list-formats` contract).
func TestKnownFormatsIsSortedAndComplete(t *testing.T) {
	got := KnownFormats()
	want := []string{"json", "markdown", "toon"}
	if len(got) != len(want) {
		t.Fatalf("KnownFormats() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("KnownFormats() = %v, want %v", got, want)
		}
	}
}

// TestEncodeMarkdownTable pins the header/separator/row shape.
func TestEncodeMarkdownTable(t *testing.T) {
	s := mustSurface(t, "Notes", []string{"id", "verdict"}, []Row{
		{"n1", "fresh"},
		{"n2", "withheld:stale"},
	})
	got := string(EncodeMarkdown(s))
	want := "## Notes\n\n" +
		"| id | verdict |\n" +
		"| --- | --- |\n" +
		"| n1 | fresh |\n" +
		"| n2 | withheld:stale |\n"
	if got != want {
		t.Fatalf("EncodeMarkdown mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// TestEncodeMarkdownEscapesPipesAndNewlines proves a cell value can never break the
// table's column structure.
func TestEncodeMarkdownEscapesPipesAndNewlines(t *testing.T) {
	s := mustSurface(t, "", []string{"detail"}, []Row{{"a | b\nc"}})
	got := string(EncodeMarkdown(s))
	if strings.Contains(got, "a | b") {
		t.Fatalf("unescaped pipe broke the table: %q", got)
	}
	if strings.Contains(got, "\nc |") {
		t.Fatalf("unescaped newline broke the row: %q", got)
	}
}

// TestEncodeJSONRoundTrips proves the JSON encoder is a lossless, order-preserving
// array-of-objects that any JSON decoder can parse back.
func TestEncodeJSONRoundTrips(t *testing.T) {
	s := mustSurface(t, "t", []string{"id", "note"}, []Row{
		{"n1", `has "quotes" and, a comma`},
		{"n2", "plain"},
	})
	body := EncodeJSON(s)
	var decoded []map[string]string
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("EncodeJSON produced invalid JSON: %v\n%s", err, body)
	}
	if len(decoded) != 2 {
		t.Fatalf("got %d objects, want 2", len(decoded))
	}
	if decoded[0]["id"] != "n1" || decoded[0]["note"] != `has "quotes" and, a comma` {
		t.Fatalf("round-trip mismatch: %+v", decoded[0])
	}
}

// TestEncodeTOONGolden pins the exact TOON tabular shape: one header line declaring
// row count + field names once, then indented comma-joined rows with no per-row key
// repetition — the load-bearing token-savings property.
func TestEncodeTOONGolden(t *testing.T) {
	s := mustSurface(t, "Notes", []string{"id", "verdict"}, []Row{
		{"n1", "fresh"},
		{"n2", "withheld"},
	})
	got := string(EncodeTOON(s))
	want := "notes[2]{id,verdict}:\n" +
		"  n1,fresh\n" +
		"  n2,withheld\n"
	if got != want {
		t.Fatalf("EncodeTOON mismatch:\ngot:\n%q\nwant:\n%q", got, want)
	}
}

// TestEncodeTOONQuotesAmbiguousCells proves a value that would be misread as a
// delimiter, a typed scalar, or an empty cell is quoted, while a plain scalar stays
// bare — the "quote only when required" rule that keeps the format token-cheap.
func TestEncodeTOONQuotesAmbiguousCells(t *testing.T) {
	s := mustSurface(t, "r", []string{"a", "b", "c", "d", "e"}, []Row{
		{"plain", "has,comma", "42", "", `has"quote`},
	})
	got := string(EncodeTOON(s))
	line := strings.TrimPrefix(strings.TrimSuffix(strings.SplitN(got, "\n", 2)[1], "\n"), "  ")
	cells := splitTOONRowForTest(line)
	if cells[0] != "plain" {
		t.Fatalf("plain scalar should be bare, got %q", cells[0])
	}
	if cells[1] != `"has,comma"` {
		t.Fatalf("comma-bearing cell should be quoted, got %q", cells[1])
	}
	if cells[2] != `"42"` {
		t.Fatalf("numeric-looking cell should be quoted (else misread as a number), got %q", cells[2])
	}
	if cells[3] != `""` {
		t.Fatalf("empty cell should be quoted (not a bare zero-width column), got %q", cells[3])
	}
	if cells[4] != `"has\"quote"` {
		t.Fatalf("quote-bearing cell should be escaped+quoted, got %q", cells[4])
	}
}

// splitTOONRowForTest is a test-only naive splitter good enough for the fixed,
// known cell count this test constructs (it does not need to be a general TOON
// parser — no production code decodes TOON here).
func splitTOONRowForTest(line string) []string {
	var out []string
	var cur strings.Builder
	inQuote := false
	escaped := false
	for _, r := range line {
		switch {
		case escaped:
			cur.WriteRune(r)
			escaped = false
		case r == '\\' && inQuote:
			cur.WriteRune(r)
			escaped = true
		case r == '"':
			inQuote = !inQuote
			cur.WriteRune(r)
		case r == ',' && !inQuote:
			out = append(out, cur.String())
			cur.Reset()
		default:
			cur.WriteRune(r)
		}
	}
	out = append(out, cur.String())
	return out
}

// TestEncodeTOONEmptyRows proves a zero-row Surface still renders a valid header
// (N=0) rather than an empty/panic-prone byte slice.
func TestEncodeTOONEmptyRows(t *testing.T) {
	s := mustSurface(t, "empty", []string{"a", "b"}, nil)
	got := string(EncodeTOON(s))
	want := "empty[0]{a,b}:\n"
	if got != want {
		t.Fatalf("EncodeTOON(empty) = %q, want %q", got, want)
	}
}

// TestSweepFormatsIsDeterministic proves the same Surface encodes to byte-identical
// output across repeated sweeps — the property SweepFormats' cross-format token
// deltas depend on to mean something rather than measure encoder jitter.
func TestSweepFormatsIsDeterministic(t *testing.T) {
	s := mustSurface(t, "Notes", []string{"id", "verdict", "detail"}, []Row{
		{"n1", "fresh", "no concrete claims"},
		{"n2", "withheld:stale", "artifact no longer exists"},
	})
	first, err := SweepFormats(s, nil)
	if err != nil {
		t.Fatalf("SweepFormats: %v", err)
	}
	second, err := SweepFormats(s, nil)
	if err != nil {
		t.Fatalf("SweepFormats: %v", err)
	}
	if len(first) != len(second) {
		t.Fatalf("sweep length changed: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("sweep row %d not deterministic: %+v vs %+v", i, first[i], second[i])
		}
	}
}

// TestSweepFormatsCoversEveryRegisteredFormat proves an empty formats list defaults
// to the full registry, not a hardcoded subset — a future Register call is picked up
// without editing this call site.
func TestSweepFormatsCoversEveryRegisteredFormat(t *testing.T) {
	s := mustSurface(t, "t", []string{"a"}, []Row{{"1"}})
	got, err := SweepFormats(s, nil)
	if err != nil {
		t.Fatalf("SweepFormats: %v", err)
	}
	if len(got) != len(KnownFormats()) {
		t.Fatalf("SweepFormats(nil) covered %d formats, want %d (%v)", len(got), len(KnownFormats()), KnownFormats())
	}
}

// TestSweepFormatsUnknownFormatFailsWholeSweep proves a typo in a requested format
// list refuses the whole sweep rather than silently reporting fewer arms than asked.
func TestSweepFormatsUnknownFormatFailsWholeSweep(t *testing.T) {
	s := mustSurface(t, "t", []string{"a"}, []Row{{"1"}})
	_, err := SweepFormats(s, []Format{FormatJSON, "xml"})
	if err == nil {
		t.Fatal("expected an error for an unknown format in the sweep list")
	}
}

// TestTOONIsCheaperThanJSONForUniformRows is the load-bearing ablation witness: for a
// uniform array of records (the shape every memory Surface in this codebase is), TOON
// must cost fewer estimated tokens than JSON, because JSON repeats every field name
// once per row while TOON declares the field list exactly once. This is the concrete,
// measured version of the "TOON can be used/ablated at the right time" goal — a caller
// deciding which format to surface can read this delta off SweepFormats instead of
// taking the claim on faith.
func TestTOONIsCheaperThanJSONForUniformRows(t *testing.T) {
	fields := []string{"id", "title", "verdict", "detail"}
	rows := make([]Row, 0, 20)
	for i := 0; i < 20; i++ {
		rows = append(rows, Row{"note-id-value", "a reasonably long title field", "fresh", "no concrete artifact claims to check"})
	}
	s := mustSurface(t, "Notes", fields, rows)

	metrics, err := SweepFormats(s, []Format{FormatJSON, FormatTOON})
	if err != nil {
		t.Fatalf("SweepFormats: %v", err)
	}
	byFormat := map[Format]FormatMetrics{}
	for _, m := range metrics {
		byFormat[m.Format] = m
	}
	jsonM, toonM := byFormat[FormatJSON], byFormat[FormatTOON]
	if toonM.Bytes >= jsonM.Bytes {
		t.Fatalf("expected toon (%d bytes) < json (%d bytes) for %d uniform rows", toonM.Bytes, jsonM.Bytes, len(rows))
	}
	if toonM.EstimatedTokens >= jsonM.EstimatedTokens {
		t.Fatalf("expected toon (%d tokens) < json (%d tokens)", toonM.EstimatedTokens, jsonM.EstimatedTokens)
	}
}

// TestRegisterAddsAFormat proves the open extension seam: a caller-registered format
// becomes visible to KnownFormats/Encode/SweepFormats without a memview code change.
func TestRegisterAddsAFormat(t *testing.T) {
	Register(Format("test-upper"), func(s Surface) []byte {
		return []byte(strings.ToUpper(strings.Join(s.Fields, ",")))
	})
	defer delete(formatRegistry, Format("test-upper"))

	s := mustSurface(t, "t", []string{"a", "b"}, nil)
	body, err := Encode(Format("test-upper"), s)
	if err != nil {
		t.Fatalf("Encode(test-upper): %v", err)
	}
	if string(body) != "A,B" {
		t.Fatalf("registered encoder not used: %q", body)
	}
	found := false
	for _, f := range KnownFormats() {
		if f == "test-upper" {
			found = true
		}
	}
	if !found {
		t.Fatal("KnownFormats did not include the registered format")
	}
}
