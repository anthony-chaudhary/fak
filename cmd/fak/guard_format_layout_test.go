package main

import (
	"strings"
	"testing"
)

// TestGuardRowAlignsValueColumn pins the core promise of the exit-summary grammar: the value
// column lines up down a section regardless of label length, so the variable data reads as a
// column set off from the fixed label prose to its left. A short label is padded to the value
// column; a label carrying a multi-byte glyph is padded by DISPLAY width (rune count), not
// byte length, so the "⚠" fak-fault marker does not short the column.
func TestGuardRowAlignsValueColumn(t *testing.T) {
	short := guardRow("state", "X")
	glyph := guardRow("  ⚠ anchor-starved", "X")
	plain := guardRow("  bailed: under_budget", "X")
	// The value "X" must start at the same column in every row — find the byte offset of the
	// value by rune-counting up to it, since the glyph row has multi-byte runes.
	col := func(line string) int {
		i := strings.IndexByte(line, 'X')
		if i < 0 {
			t.Fatalf("row has no value marker: %q", line)
		}
		return len([]rune(line[:i]))
	}
	if col(short) != col(glyph) || col(short) != col(plain) {
		t.Errorf("value column misaligned:\n short=%q (col %d)\n glyph=%q (col %d)\n plain=%q (col %d)",
			short, col(short), glyph, col(glyph), plain, col(plain))
	}
}

// TestGuardSectionNamesOnce pins that a section header carries the "guard · <name>" spine and
// a trailing rule, so the rows beneath need not each repeat the "fak guard:" prefix the old
// wall-of-text summary stamped on every line.
func TestGuardSectionNamesOnce(t *testing.T) {
	h := guardSection("audit")
	if !strings.Contains(h, "guard · audit") {
		t.Errorf("section header must name the section: %q", h)
	}
	if !strings.HasSuffix(h, "\n") || !strings.Contains(h, "──") {
		t.Errorf("section header must be a newline-terminated rule: %q", h)
	}
}

// TestGuardNoteDemotesAndSkipsEmpty pins the continuation-line contract: a note is prefixed
// with the "↳" demotion marker, and an empty note yields nothing (so an absent caveat never
// prints a bare arrow).
func TestGuardNoteDemotesAndSkipsEmpty(t *testing.T) {
	if n := guardNote("the caveat"); !strings.Contains(n, "↳") || !strings.Contains(n, "the caveat") {
		t.Errorf("note must carry the demotion marker and text: %q", n)
	}
	if n := guardNote("   "); n != "" {
		t.Errorf("a blank note must yield no line, got %q", n)
	}
}

// TestGuardColorizeSummaryTTYGated pins the print-time color contract: on a real terminal the
// section rules and demoted notes are wrapped in SGR so the chrome recedes and the value rows
// stand out — but a non-TTY sink (a file, a `-p` JSON capture) gets the text back BYTE-FOR-BYTE,
// the same byte-clean discipline guardSummaryResetPrefix follows and the formatter tests assert.
func TestGuardColorizeSummaryTTYGated(t *testing.T) {
	plain := guardSection("audit") + guardRow("47 kernel decision(s)", "44 allowed") + guardNote("a caveat")

	// Non-TTY: unchanged, byte-for-byte.
	if got := guardColorizeSummary(plain, false); got != plain {
		t.Errorf("a non-TTY summary must be returned byte-clean:\n got %q\n want %q", got, plain)
	}

	// TTY: section rule and note get SGR chrome; the raw text is still all present.
	colored := guardColorizeSummary(plain, true)
	if !strings.Contains(colored, tuiSGRCyanBold) {
		t.Errorf("a TTY summary must color the section rule:\n%q", colored)
	}
	if !strings.Contains(colored, tuiSGRDim) {
		t.Errorf("a TTY summary must dim the demoted note:\n%q", colored)
	}
	if !strings.Contains(colored, tuiSGRReset) {
		t.Errorf("a TTY summary must reset SGR after coloring:\n%q", colored)
	}
	// Stripping the SGR escapes must recover the exact plain text — color adds chrome, never
	// mangles or drops the underlying data.
	stripped := colored
	for _, esc := range []string{tuiSGRCyanBold, tuiSGRDim, tuiSGRReset} {
		stripped = strings.ReplaceAll(stripped, esc, "")
	}
	if stripped != plain {
		t.Errorf("stripping SGR must recover the plain text:\n got %q\n want %q", stripped, plain)
	}
	// The value row (no section/note prefix) must be untouched even on a TTY — the data stands
	// out precisely because only the chrome around it is colored.
	valueRow := guardRow("47 kernel decision(s)", "44 allowed")
	if !strings.Contains(colored, valueRow) {
		t.Errorf("a TTY summary must leave the value row uncolored:\n%q", colored)
	}
}
