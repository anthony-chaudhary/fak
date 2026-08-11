// bar.go — the one non-text mark a card can carry.
//
// WHY A MARK AT ALL, in a renderer that is otherwise entirely glyphs. A row of
// figures makes the viewer parse before they can compare: "0.2782 -> 0.3619"
// is two numbers to hold in working memory and one subtraction to perform, per
// row, against a card that is already moving on. A bar is compared by eye, at a
// glance, and the comparison it invites — this one is about a third of that one
// — is the actual claim the beat exists to land. In a cut ranked by importance,
// a beat the viewer has to do arithmetic in is a beat they leave.
//
// ⛔ THE HONESTY CONSTRAINT, and it is why the scale is not a knob. A bar's
// track is always the metric's OWN maximum — 1.0 for an AP, 100% for a recall,
// N for a count out of N — so there is no per-card axis for an author to pick,
// and therefore no axis choice that can flatter a number. A +0.08 lift cannot be
// drawn as half the screen by cropping the scale to 0.25..0.40, and two bars on
// one card are comparable precisely because neither one got its own ruler. The
// config supplies the fraction, the author's own text supplies the number, and
// the renderer invents neither: it cannot compute a fraction from prose, so it
// never pretends to have checked one. What it DOES enforce is that the fraction
// lies in 0..1, because a value past the metric's maximum means the denominator
// is wrong, and a bar clamped silently at full width would hide that.
//
// The mark is drawn as filled cell rectangles rather than block glyphs, for two
// reasons that both matter here. No font face is consulted, so the bar is the
// same SHAPE in the bitmap GIF path and the antialiased video path instead of
// depending on whether either face happens to carry U+2588. And it cannot be
// silently truncated at the right margin the way a long line of text is,
// because its width is fixed by the renderer, not by the author's line.
package main

import (
	"fmt"
	"strconv"
	"strings"
)

// barFillRune marks a cell that the rasterisers paint as a solid rectangle in
// the cell's colour instead of looking a glyph up for it. It is deliberately a
// control code: nothing a card or a script(1) capture can contain reaches it, so
// no text can be mistaken for a bar or vice versa.
const barFillRune = '\x01'

// barToken introduces a bar on a card line, after that line's colour token.
const barToken = "@bar"

const (
	// barCells is the track width in terminal cells, shared by every bar in
	// every cut. One shared width is the other half of what makes two bars
	// comparable — a per-bar width would be a second axis to choose, and the
	// scale comment above spent a paragraph removing the first one.
	barCells = 20
	// barGap separates a track from its label.
	barGap = 2
)

// barMark is a parsed bar line: a track filled to Frac, then Label beside it.
type barMark struct {
	Frac  float64
	Label string
}

// parseBar recognises a bar line, whose form is "@bar <frac> <label>" once the
// line's colour token has been taken off the front. ok reports whether this is
// a bar line at all; ordinary prose returns false with no error.
//
// ⛔ err is non-nil only for a line that MEANT to be a bar and got it wrong, and
// it is a hard failure rather than a fallback to prose on purpose. A malformed
// bar that quietly rendered as the literal text "@bar 1.3 engines identical"
// would ship a card that looks authored, reads as deliberate, and silently
// carries no mark at all — the same shape as every other green-and-blind result
// the config's verify block exists to refuse. Caught at load, it is a one-line
// message with a segment and a line number instead.
func parseBar(text string) (barMark, bool, error) {
	rest, found := strings.CutPrefix(text, barToken)
	if !found {
		return barMark{}, false, nil
	}
	// "@barrier tests pass" is prose that happens to open with the token.
	if rest != "" && rest[0] != ' ' {
		return barMark{}, false, nil
	}
	fields := strings.Fields(rest)
	if len(fields) == 0 {
		return barMark{}, true, fmt.Errorf("%s needs a fraction and a label", barToken)
	}
	frac, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return barMark{}, true, fmt.Errorf("%s fraction %q does not parse as a number", barToken, fields[0])
	}
	if frac < 0 || frac > 1 {
		return barMark{}, true, fmt.Errorf("%s fraction %v is outside 0..1 — a track is the metric's "+
			"own maximum, so a value past it means the denominator is wrong", barToken, frac)
	}
	// The label keeps its interior spacing: cards align their figures into
	// columns by hand, and collapsing runs of spaces here would break them.
	label := strings.TrimLeft(rest, " ")
	label = strings.TrimLeft(strings.TrimPrefix(label, fields[0]), " ")
	return barMark{Frac: frac, Label: label}, true, nil
}

// width is the cells a bar line occupies. A cardLeft block shares one left
// margin across every line, so a bar has to be able to state its own width or
// the prose around it would be measured on a different ruler and the column
// would break.
func (b barMark) width() int { return barCells + barGap + len([]rune(b.Label)) }

// fill is how much of the track is bar rather than backing.
//
// The rounding has one floor: a non-zero fraction always gets at least one
// cell, because "small but real" and "nothing" must not draw identically —
// 16-32 px recall at 15.5% and a hole at 0 of 11 are different findings, and a
// reader who cannot tell them apart has been told the wrong one.
func (b barMark) fill() int {
	n := int(b.Frac*float64(barCells) + 0.5)
	if n == 0 && b.Frac > 0 {
		n = 1
	}
	if n > barCells {
		n = barCells
	}
	return n
}

// cardLineWidth is the on-screen width of one authored card line, bar or prose.
// Both the block-margin computation and the per-line centring go through it, so
// they cannot disagree about how wide a bar line is.
func cardLineWidth(line string) int {
	_, _, rest := cardColorToken(line)
	if b, ok, err := parseBar(rest); ok && err == nil {
		return b.width()
	}
	return len([]rune(rest))
}

// putBar writes one bar at the cursor: the filled cells in the line's own
// colour, the remainder of the track in the recessive backing tone, then the
// gap and the label.
//
// The backing is always drawn, and that is the point rather than decoration. A
// bar with no 100% mark beside it is a length with nothing to be a length OF,
// which is the reading a viewer cannot do from the figure alone. It is also the
// only way a zero-length bar says anything: "0 of 11 artifacts byte-identical"
// draws as a full, empty track, which is a far more emphatic image than a blank
// row would be, and an honest one.
func (t *term) putBar(b barMark, fg uint8) {
	n := b.fill()
	for i := 0; i < barCells; i++ {
		if t.cx >= t.cols {
			break
		}
		c := uint8(cBarTrack)
		if i < n {
			c = fg
		}
		t.cells[t.cy*t.cols+t.cx] = cell{r: barFillRune, fg: c, age: t.age}
		t.cx++
	}
	t.fg, t.bold = fg, false
	for i := 0; i < barGap; i++ {
		t.put(' ')
	}
	for _, r := range b.Label {
		if t.cx >= t.cols {
			break
		}
		t.put(printable(r, t.face))
	}
}

// barInset is the breathing space above and below a bar inside its cell row.
//
// A bar drawn to the full cell height butts against the rows either side of it
// and reads as a filled block rather than a bar. A sixth of the row at each end
// reads as a bar. It is derived from the row height instead of being a fixed
// pixel count so the GIF's 18-pixel rows and the video's 36-pixel rows get the
// same SHAPE — the same reason the line spacing is a renderer property and not
// a per-video workaround.
func barInset(cellH int) int {
	n := cellH / 6
	if n < 1 {
		n = 1
	}
	return n
}

// validateCards refuses a config whose card markup cannot be rendered as
// authored. It runs before anything rasterises, so `-verify` is the gate and a
// bad bar costs a second rather than a full render.
func validateCards(cfg config) error {
	for si, seg := range cfg.Segments {
		for li, line := range seg.Card {
			_, _, rest := cardColorToken(line)
			if _, _, err := parseBar(rest); err != nil {
				name := seg.Chapter
				if name == "" {
					name = "untitled"
				}
				return fmt.Errorf("segment %d (%s), card line %d: %w", si+1, name, li+1, err)
			}
		}
	}
	return nil
}
