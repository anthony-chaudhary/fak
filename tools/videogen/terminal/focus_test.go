package main

import (
	"image/color"
	"math"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/image/font/inconsolata"
)

func TestActiveCardConceptMovesOnlyWhenNewInformationArrives(t *testing.T) {
	lines := []string{
		"@c TITLE",
		"",
		"@w first line of concept two",
		"@g second line of concept two",
		"",
		"@y concept three",
	}
	cases := []struct {
		show       int
		start, end int
	}{
		{1, 0, 1},
		{2, 0, 1}, // a spacer keeps focus on the completed concept
		{3, 2, 3},
		{4, 2, 4},
		{5, 2, 4}, // another spacer keeps the two-line concept together
		{6, 5, 6},
	}
	for _, tc := range cases {
		start, end := activeCardConcept(lines, tc.show)
		if start != tc.start || end != tc.end {
			t.Errorf("show %d: active concept = [%d,%d), want [%d,%d)",
				tc.show, start, end, tc.start, tc.end)
		}
	}
}

func TestCardConceptAgesKeepAMultiLineConceptTogether(t *testing.T) {
	lines := []string{"@c A1", "@c A2", "", "@y B", "", "@w C"}
	want := []uint8{2, 2, 0, 1, 0, 0}
	got := cardConceptAges(lines, len(lines))
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("line %d (%q) age = %d, want %d", i, lines[i], got[i], want[i])
		}
	}
}

// The card path's version of the ramp: each older concept sits one step further
// back, so a card carries its own reading order rather than a bright line and a
// wall of grey.
func TestRenderCardFadesEachOlderConceptOneStepFurther(t *testing.T) {
	term := newTerm(40, 12, inconsolata.Regular8x16)
	lines := []string{
		"@g FIRST", "",
		"@c SECOND", "",
		"@y THIRD", "",
		"@w FOURTH",
	}
	renderCard(term, lines, false, len(lines))

	firstOnRow := func(row int) cell {
		t.Helper()
		for x := 0; x < term.cols; x++ {
			if c := term.cells[row*term.cols+x]; c.r != ' ' {
				return c
			}
		}
		t.Fatalf("row %d is blank", row)
		return cell{}
	}
	// renderCard centres the card vertically: 12 rows, 7 lines => top row 2.
	cases := []struct {
		row  int
		what string
		age  uint8
		fg   uint8
		bold bool
	}{
		{8, "FOURTH, the live concept", 0, cWhite, true},
		{6, "THIRD, one beat back", 1, cYellow, false},
		{4, "SECOND, two beats back", 2, cFade1, false},
		{2, "FIRST, three beats back", 3, cFade2, false},
	}
	for _, tc := range cases {
		c := firstOnRow(tc.row)
		if c.age != tc.age {
			t.Errorf("%s: age %d, want %d", tc.what, c.age, tc.age)
		}
		if fg, bold := c.faded(); fg != tc.fg || bold != tc.bold {
			t.Errorf("%s: drawn as {fg:%d bold:%v}, want {fg:%d bold:%v}",
				tc.what, fg, bold, tc.fg, tc.bold)
		}
	}
}

// The replay path's version: one step per emitted unit, and the step is small.
func TestAgeExistingWalksOneStepPerUnitAndSettlesOnTheFloor(t *testing.T) {
	term := newTerm(40, 4, inconsolata.Regular8x16)
	term.fg, term.bold = cGreen, true
	term.writeString("old")

	want := []struct {
		fg   uint8
		bold bool
	}{
		{cGreen, true},  // the unit that just landed
		{cGreen, false}, // hue kept, weight dropped
		{cFade1, false},
		{cFade2, false},
		{cFade3, false},
		{cFade3, false}, // the floor holds, however long the segment runs
		{cFade3, false},
	}
	for i, w := range want {
		fg, bold := term.cells[0].faded()
		if fg != w.fg || bold != w.bold {
			t.Errorf("%d unit(s) later: {fg:%d bold:%v}, want {fg:%d bold:%v}",
				i, fg, bold, w.fg, w.bold)
		}
		ageExisting(term)
	}
	// Ageing has to stop, or two identical screens would compare unequal and
	// the encoders would emit a fresh frame per unit forever.
	if got := term.cells[0].age; got != fadeFloorAge {
		t.Errorf("age kept climbing to %d, want it pinned at the floor %d", got, fadeFloorAge)
	}
}

func TestAgeExistingLeavesTheNewestUnitBrightest(t *testing.T) {
	term := newTerm(40, 4, inconsolata.Regular8x16)
	term.fg, term.bold = cGreen, true
	term.writeString("old")

	ageExisting(term)
	term.fg, term.bold = cWhite, true
	term.writeString("\nnew")

	if fg, bold := term.cells[0].faded(); fg != cGreen || bold {
		t.Errorf("previous unit = {fg:%d bold:%v}, want its own colour at regular weight", fg, bold)
	}
	if fg, bold := term.cells[term.cols].faded(); fg != cWhite || !bold {
		t.Errorf("new unit = {fg:%d bold:%v}, want white and bold", fg, bold)
	}
}

// The defect the ramp exists to fix: superseded text used to arrive at the
// resting tone in a single beat. Pin the route, not only the destination.
func TestFadeReachesTheFloorNoSoonerThanTheLastBeat(t *testing.T) {
	c := cell{r: 'x', fg: cFG, bold: true}
	for age := uint8(0); age < fadeFloorAge; age++ {
		c.age = age
		if fg, _ := c.faded(); fg == cFade3 {
			t.Errorf("age %d is already on the floor; the fade is meant to take %d beats",
				age, fadeFloorAge)
		}
	}
	c.age = fadeFloorAge
	if fg, _ := c.faded(); fg != cFade3 {
		t.Errorf("age %d draws as palette %d, want the floor cFade3", fadeFloorAge, fg)
	}
}

// "Slight fade, still readable" is the requirement, so it gets a threshold
// rather than an opinion. Every rung has to clear body-text contrast.
func TestEveryFadeStepStaysReadableAgainstTheBackdrop(t *testing.T) {
	const wantMin = 4.5 // WCAG 2.x AA, normal-size text
	for i, idx := range fadeLadder {
		if got := contrastRatio(pal[idx], pal[cBG]); got < wantMin {
			t.Errorf("fade step %d (palette %d) contrast %.2f:1, want at least %.1f:1",
				i+1, idx, got, wantMin)
		}
	}
}

// Gradual means monotone: no rung may be brighter than the one before it, or
// the text would flare on its way out.
func TestFadeRampDescendsWithoutAWobble(t *testing.T) {
	prev, prevName := relLuminance(pal[cFG]), "the authored foreground"
	for i, idx := range fadeLadder {
		l := relLuminance(pal[idx])
		if l >= prev {
			t.Errorf("fade step %d (palette %d) luminance %.4f is not below %s (%.4f)",
				i+1, idx, l, prevName, prev)
		}
		prev, prevName = l, "the previous step"
	}
}

// The ramp changes the route, not the destination: it still comes to rest on
// the tone the old single switch jumped straight to.
func TestFadeFloorMatchesTheToneTheOldSwitchLandedOn(t *testing.T) {
	got, want := relLuminance(pal[cFade3]), relLuminance(pal[cGrey])
	if d := math.Abs(got-want) / want; d > 0.05 {
		t.Errorf("floor luminance %.4f vs the previous grey %.4f, %.1f%% apart; want within 5%%",
			got, want, 100*d)
	}
}

// TestAuthoredGreyDoesNotFlareOnItsWayOut pins the one colour the ramp must not
// touch: cGrey is what `@d` markup and SGR 90 already mean — recessive text — so
// running it down a ramp of brighter neutrals would flare it before it settled.
func TestAuthoredGreyDoesNotFlareOnItsWayOut(t *testing.T) {
	for age := uint8(0); age <= fadeFloorAge+2; age++ {
		c := cell{r: 'x', fg: cGrey, bold: age == 0, age: age}
		if fg, _ := c.faded(); fg != cGrey {
			t.Errorf("authored grey at age %d draws as palette %d, want cGrey", age, fg)
		}
	}
}

func TestPaletteStillFitsFourBitsPerPixel(t *testing.T) {
	if len(pal) > 16 {
		t.Fatalf("palette has %d entries, want at most 16 so the GIF stays 4bpp", len(pal))
	}
}

// contrastRatio and relLuminance are the WCAG 2.x formulas. They live in the
// test rather than the renderer on purpose: the ramp's readability is a
// property to be checked, never a number the renderer recomputes at draw time.
func contrastRatio(a, b color.Color) float64 {
	la, lb := relLuminance(a), relLuminance(b)
	if la < lb {
		la, lb = lb, la
	}
	return (la + 0.05) / (lb + 0.05)
}

func relLuminance(c color.Color) float64 {
	lin := func(v uint32) float64 {
		s := float64(v) / 65535
		if s <= 0.03928 {
			return s / 12.92
		}
		return math.Pow((s+0.055)/1.055, 2.4)
	}
	r, g, b, _ := c.RGBA()
	return 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b)
}

func TestRowsHaveExtraSpaceBetweenLinesAtBothResolutions(t *testing.T) {
	r := newRenderer(120, 32, 10)
	if gap := r.ch - r.reg.Height; gap < 2 {
		t.Fatalf("GIF row height %d - glyph height %d = %dpx, want at least 2px of line spacing",
			r.ch, r.reg.Height, gap)
	}
	if defaultHiresCellHeight != 2*r.ch {
		t.Fatalf("hi-res row height = %d, want exactly 2x the GIF's %dpx rows",
			defaultHiresCellHeight, r.ch)
	}
}

func TestRenderGIFCreatesAProjectOutputDirectory(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out", "example.gif")
	cfg := config{Cols: 12, Rows: 4, Pad: 1, Out: out}
	if _, err := renderGIF(cfg, "", 1); err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(out); err != nil {
		t.Fatalf("rendered GIF: %v", err)
	} else if st.Size() == 0 {
		t.Fatal("rendered GIF is empty")
	}
}
