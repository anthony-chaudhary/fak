package main

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// TestTrimTUIIsRuneSafe pins the keystone fix: trimTUI must never emit invalid
// UTF-8. The old byte-indexed implementation sliced s[:width-3] which could cut a
// multibyte rune in half — e.g. trimTUI("ab—cdef…", 6) returned "ab\xe2..." (the
// em-dash's first byte), which a terminal draws as a replacement glyph. Every
// pane column routes user/issue/loop text (em-dashes, arrows, emoji) through
// trimTUI, so a single mid-rune cut corrupts the output.
func TestTrimTUIIsRuneSafe(t *testing.T) {
	cases := []string{
		"ab—cdefghijklmnop",                  // em-dash (3 bytes) straddling the cut
		"loop summary — reads repaid",        // realistic summary with an em-dash
		"→→→→→→→→→→",                         // all multibyte arrows
		"🚀🚀🚀🚀🚀",                              // 4-byte emoji (also 2 cells each)
		"日本語のテキストです",                         // CJK wide runes
		"plain ascii only no surprises here", // control: pure ASCII
	}
	for _, s := range cases {
		for w := 0; w <= dispWidthTUI(s)+2; w++ {
			out := trimTUI(s, w)
			if !utf8.ValidString(out) {
				t.Fatalf("trimTUI(%q, %d) = %q is not valid UTF-8", s, w, out)
			}
			if dispWidthTUI(out) > maxTUI(w, 0) && w > 0 {
				t.Fatalf("trimTUI(%q, %d) = %q has display width %d > budget %d",
					s, w, out, dispWidthTUI(out), w)
			}
		}
	}
}

// TestTrimTUIMidRuneCutRegression is the exact reproduction from the bug: at
// width 6 the cut lands inside the em-dash. The fix must keep whole runes.
func TestTrimTUIMidRuneCutRegression(t *testing.T) {
	got := trimTUI("ab—cdefghijklmnop", 6)
	if !utf8.ValidString(got) {
		t.Fatalf("trimTUI mid-rune cut produced invalid UTF-8: %q", got)
	}
	if strings.ContainsRune(got, '�') {
		t.Fatalf("trimTUI produced a replacement glyph: %q", got)
	}
}

// TestDispWidthTUICounts pins the cell model the column math depends on: an
// em-dash is one cell, a CJK glyph or emoji is two, a combining mark is zero.
func TestDispWidthTUICounts(t *testing.T) {
	cases := []struct {
		s    string
		want int
	}{
		{"abc", 3},
		{"a—b", 3}, // em-dash = 1 cell
		{"→", 1},   // arrow = 1 cell
		{"日本", 4},  // 2 CJK glyphs = 4 cells
		{"🚀", 2},   // emoji = 2 cells
		{"é", 1},  // 'e' + combining acute = 1 cell (mark is zero-width)
		{"", 0},
	}
	for _, c := range cases {
		if got := dispWidthTUI(c.s); got != c.want {
			t.Fatalf("dispWidthTUI(%q) = %d, want %d", c.s, got, c.want)
		}
	}
}

// TestPadRightTUIAlignsByCells proves the rune-aware pad keeps columns aligned
// when text carries multibyte runes — fmt's "%-*s" pads by bytes and would draw
// an em-dash column two cells too wide, shearing every column to its right.
func TestPadRightTUIAlignsByCells(t *testing.T) {
	plain := padRightTUI("ab", 6)
	dash := padRightTUI("a—", 6) // 'a' + em-dash = 2 cells, like "ab"
	if dispWidthTUI(plain) != dispWidthTUI(dash) {
		t.Fatalf("padRightTUI misaligned: %q (w=%d) vs %q (w=%d)",
			plain, dispWidthTUI(plain), dash, dispWidthTUI(dash))
	}
	if dispWidthTUI(plain) != 6 {
		t.Fatalf("padRightTUI(\"ab\", 6) display width = %d, want 6", dispWidthTUI(plain))
	}
	// An already-too-wide field is returned unchanged, not negatively padded.
	wide := padRightTUI("abcdef", 3)
	if wide != "abcdef" {
		t.Fatalf("padRightTUI over-wide field mangled: %q", wide)
	}
}
