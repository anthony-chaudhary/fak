package logvault

import "testing"

func TestFootprintFoldsPerSourceWitnessedStats(t *testing.T) {
	rows := []ManifestRow{
		{TSUnixNano: 100, Op: OpFull, Source: "s", RelPath: "a.jsonl", SizeAfter: 5, SHA256: "x1"},
		{TSUnixNano: 200, Op: OpAppend, Source: "s", RelPath: "a.jsonl", SizeAfter: 10, SHA256: "x2"},
		{TSUnixNano: 150, Op: OpFull, Source: "s", RelPath: "b.log", SizeAfter: 3, SHA256: "y1"},
		{TSUnixNano: 250, Op: OpSkip, Source: "s", RelPath: "c.txt"}, // read failure: no SHA, advances nothing but Errors
		{TSUnixNano: 300, Op: OpSkip, Source: "t", RelPath: "only-errors"},
	}
	fps := Footprint(rows)
	if len(fps) != 2 {
		t.Fatalf("footprint sources = %d, want 2", len(fps))
	}
	// Sorted by source id: "s" before "t".
	s := fps[0]
	if s.Source != "s" {
		t.Fatalf("first source = %q, want s", s.Source)
	}
	if s.Files != 2 {
		t.Fatalf("s.Files = %d, want 2 (a.jsonl + b.log; the skip-error file is not tracked)", s.Files)
	}
	if s.Bytes != 13 {
		t.Fatalf("s.Bytes = %d, want 13 (latest SizeAfter 10 + 3)", s.Bytes)
	}
	if s.ManifestRows != 4 {
		t.Fatalf("s.ManifestRows = %d, want 4", s.ManifestRows)
	}
	if s.Errors != 1 {
		t.Fatalf("s.Errors = %d, want 1", s.Errors)
	}
	if s.LastCaptureUnixNano != 200 {
		t.Fatalf("s.LastCaptureUnixNano = %d, want 200 (newest SUCCESSFUL op, NOT the ts=250 skip-error)", s.LastCaptureUnixNano)
	}

	tt := fps[1]
	if tt.Source != "t" || tt.Files != 0 || tt.Bytes != 0 || tt.Errors != 1 || tt.LastCaptureUnixNano != 0 {
		t.Fatalf("skip-error-only source = %+v, want files/bytes/last-capture all zero with 1 error", tt)
	}

	if got := NewestCaptureUnixNano(fps); got != 200 {
		t.Fatalf("NewestCaptureUnixNano = %d, want 200", got)
	}
	if got := TotalBytes(fps); got != 13 {
		t.Fatalf("TotalBytes = %d, want 13", got)
	}
}

func TestFootprintEmptyIsValidEmpty(t *testing.T) {
	if fps := Footprint(nil); len(fps) != 0 {
		t.Fatalf("empty manifest footprint = %v, want none", fps)
	}
	if got := NewestCaptureUnixNano(nil); got != 0 {
		t.Fatalf("NewestCaptureUnixNano(nil) = %d, want 0", got)
	}
}
