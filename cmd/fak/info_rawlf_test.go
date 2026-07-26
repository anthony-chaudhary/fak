package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestCRLFWriterTUIInsertsCR pins the core mapping: every bare "\n" gains a "\r",
// and an already-CRLF stream passes through unchanged (no "\r\r\n" inflation).
func TestCRLFWriterTUIInsertsCR(t *testing.T) {
	cases := []struct{ in, want string }{
		{"a\nb\nc", "a\r\nb\r\nc"},
		{"a\r\nb", "a\r\nb"},
		{"\n", "\r\n"},
		{"no newline", "no newline"},
		{"tail\n", "tail\r\n"},
		{"\r\n\n", "\r\n\r\n"},
	}
	for _, tc := range cases {
		var buf bytes.Buffer
		if n, err := newCRLFWriterTUI(&buf).Write([]byte(tc.in)); err != nil || n != len(tc.in) {
			t.Fatalf("Write(%q) = (%d, %v), want (%d, nil)", tc.in, n, err, len(tc.in))
		}
		if got := buf.String(); got != tc.want {
			t.Fatalf("Write(%q) wrote %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestCRLFWriterTUIStatefulAcrossWrites pins the chunk-boundary case: a "\r" at the
// end of one Write followed by a "\n" at the start of the next must stay ONE CRLF —
// the writer's state has to survive between calls.
func TestCRLFWriterTUIStatefulAcrossWrites(t *testing.T) {
	var buf bytes.Buffer
	w := newCRLFWriterTUI(&buf)
	for _, chunk := range []string{"row\r", "\nnext", "\n", "last"} {
		if _, err := w.Write([]byte(chunk)); err != nil {
			t.Fatalf("Write(%q): %v", chunk, err)
		}
	}
	if got, want := buf.String(), "row\r\nnext\r\nlast"; got != want {
		t.Fatalf("stateful writes produced %q, want %q", got, want)
	}
}

// TestWriteGuardInfoFrameThroughCRLFWriter is the #5370 regression seam: the
// interactive frame writer joins rows with bare "\n", which raw mode (OPOST off)
// renders as a staircase. Through the CRLF wrapper the emitted bytes must carry
// "\r\n" between every row so each row starts at column 0 in ANY tty mode.
func TestWriteGuardInfoFrameThroughCRLFWriter(t *testing.T) {
	var buf bytes.Buffer
	rows := writeGuardInfoFrame(newCRLFWriterTUI(&buf), "row one\nrow two\nrow three", 0)
	if rows != 3 {
		t.Fatalf("rows = %d, want 3", rows)
	}
	got := buf.String()
	if strings.Contains(strings.ReplaceAll(got, "\r\n", ""), "\n") {
		t.Fatalf("frame still carries a bare LF (staircase in raw mode): %q", got)
	}
	if got != "row one\r\nrow two\r\nrow three" {
		t.Fatalf("frame bytes = %q", got)
	}
}
