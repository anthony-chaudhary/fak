package main

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// This file holds the SHARED VISUAL GRAMMAR for the `fak guard` exit summary. Before it,
// every summary line was a run-on sentence that repeated the "fak guard:" prefix and buried
// its one variable number (a count, a token total, a path) mid-clause, so the whole roll-up
// read as an undifferentiated wall of prose. These primitives give the summary ONE grammar:
//
//   - a single section HEADER (a labelled rule) replaces the per-line prefix, so a section's
//     lines group under one banner instead of each re-announcing "fak guard";
//   - an aligned key → value ROW sets the variable data in a fixed column so the eye scans the
//     values straight down, separated from the fixed label prose to its left;
//   - a NOTE continuation ("↳ …") demotes a long explanation off the scannable row so the
//     primary line stays short and the caveat is still one glance away.
//
// They are PURE string builders — no color, no I/O, no TTY probing. Color is layered on at
// print time in guard_child.go (the same discipline guardSummaryResetPrefix already follows:
// a piped/JSON sink must stay byte-clean), and the formatters are unit-tested by calling them
// directly, so keeping escape bytes out here is what lets both stay true.

const (
	// guardRowIndent is the left margin every row/note sits at, under its header.
	guardRowIndent = "  "
	// guardLabelWidth is the fixed column the value starts at: the label is padded to this
	// width so the value column lines up down a section regardless of label length. Chosen to
	// clear the longest routine label ("avoided-call amplification") without wrapping an 80-col
	// terminal once the value is appended.
	guardLabelWidth = 26
	// guardNotePrefix marks a demoted continuation line — a caveat or fix-hint pulled OFF the
	// scannable row so the row stays short. The arrow reads as "belongs to the row above".
	guardNotePrefix = guardRowIndent + "  ↳ "
)

// guardSection renders a section header: a short rule that names the section once, so the
// rows beneath it need not each repeat "fak guard". Example: `── guard · audit ──────`.
// The trailing rule is padded to a stable width so successive headers align into a spine.
func guardSection(name string) string {
	const width = 60
	head := "── guard · " + name + " "
	if pad := width - len(head); pad > 0 {
		head += strings.Repeat("─", pad)
	}
	return head + "\n"
}

// guardRow renders one aligned "label   value" row under a section header. The label is
// left-padded to guardLabelWidth so the value column lines up down the section; a label that
// overruns the column still gets a single separating space rather than colliding with its
// value. value is the variable data (the counts/tokens/path) — the part the operator scans.
func guardRow(label, value string) string {
	// Pad by rune count, not byte length: a label carrying a multi-byte glyph (the "⚠"
	// fak-fault marker) is several bytes but one display cell, so byte-length padding would
	// short the value column and break the alignment the whole grammar exists to give.
	if w := utf8.RuneCountInString(label); w < guardLabelWidth {
		label = label + strings.Repeat(" ", guardLabelWidth-w)
	} else {
		label = label + " "
	}
	return guardRowIndent + label + value + "\n"
}

// guardNote renders a demoted continuation line under the row it annotates: the long
// explanation, caveat, or fix-hint that would otherwise bloat the row into a paragraph.
// Multiple sentences are kept on one wrapped note rather than split, so the caveat reads as
// a single aside. Empty text yields no line.
func guardNote(text string) string {
	if strings.TrimSpace(text) == "" {
		return ""
	}
	return guardNotePrefix + text + "\n"
}

// guardColorizeSummary layers TTY color onto an already-assembled exit summary at PRINT time,
// so the pure formatters stay byte-clean (a piped/JSON sink gets isTTY=false and the text
// verbatim — the same discipline guardSummaryResetPrefix follows). It dims the structural
// chrome — the "── guard · … ──" section rules and the "↳ …" continuation notes — so the
// value-bearing rows between them stand out at normal weight without needing to color the
// variable data itself (which would mean re-parsing it). Idempotent per line and safe on any
// input: a line that matches nothing passes through unchanged.
func guardColorizeSummary(s string, isTTY bool) string {
	if !isTTY || s == "" {
		return s
	}
	lines := strings.Split(s, "\n")
	for i, ln := range lines {
		switch {
		case strings.HasPrefix(ln, "── guard · "):
			// Section rule: cyan-bold banner so each section reads as a heading.
			lines[i] = tuiSGRCyanBold + ln + tuiSGRReset
		case strings.HasPrefix(ln, guardNotePrefix):
			// Demoted note: dim the whole caveat so it recedes behind the rows it annotates.
			lines[i] = tuiSGRDim + ln + tuiSGRReset
		}
	}
	return strings.Join(lines, "\n")
}

// guardKV joins short "k=v" facets into the compact, bracketed detail tail some rows carry
// (e.g. the cache-attribution breakdown). Rendered as `[a=1 · b=2 · c=3]` so the facets read
// as a set of labelled values, not a comma-run of bare numbers.
func guardKV(pairs ...[2]string) string {
	if len(pairs) == 0 {
		return ""
	}
	parts := make([]string, 0, len(pairs))
	for _, p := range pairs {
		parts = append(parts, fmt.Sprintf("%s=%s", p[0], p[1]))
	}
	return "[" + strings.Join(parts, " · ") + "]"
}
