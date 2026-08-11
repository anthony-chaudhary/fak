package main

import (
	"image"
	"strings"
	"testing"

	"golang.org/x/image/font/inconsolata"
)

func TestParseBarSplitsFractionFromLabelAndLeavesProseAlone(t *testing.T) {
	cases := []struct {
		in    string
		ok    bool
		frac  float64
		label string
	}{
		{"@bar 0.3619 zero human labels    0.3619", true, 0.3619, "zero human labels    0.3619"},
		{"@bar 0 byte-identical across 4 rigs    0 of 11", true, 0, "byte-identical across 4 rigs    0 of 11"},
		{"@bar 1 layer tactics identical", true, 1, "layer tactics identical"},
		// Prose is never a bar, including prose that opens with the token's letters.
		{"the aggregate lies in both directions", false, 0, ""},
		{"@barrier tests pass", false, 0, ""},
	}
	for _, tc := range cases {
		b, ok, err := parseBar(tc.in)
		if err != nil {
			t.Errorf("%q: unexpected error %v", tc.in, err)
			continue
		}
		if ok != tc.ok {
			t.Errorf("%q: recognised as bar = %v, want %v", tc.in, ok, tc.ok)
			continue
		}
		if !ok {
			continue
		}
		if b.Frac != tc.frac {
			t.Errorf("%q: frac = %v, want %v", tc.in, b.Frac, tc.frac)
		}
		if b.Label != tc.label {
			t.Errorf("%q: label = %q, want %q", tc.in, b.Label, tc.label)
		}
	}
}

// ⛔ The point of this test is that a malformed bar is a FAILURE and never a
// fallback to prose. A bar that silently rendered as the literal text "@bar 1.3
// engines identical" would ship a card that looks authored and carries no mark.
func TestParseBarRefusesAMalformedBarRatherThanRenderingItAsProse(t *testing.T) {
	for _, in := range []string{
		"@bar",                     // no fraction, no label
		"@bar half of them",        // a word where the fraction goes
		"@bar 1.3 engines match",   // past the metric's own maximum
		"@bar -0.058 vehicle",      // a signed delta is not a fraction of a track
		"@bar 58% relative uplift", // percent sign, not a 0..1 fraction
	} {
		b, ok, err := parseBar(in)
		if !ok {
			t.Errorf("%q: not recognised as a bar at all, so a bad bar would render as prose", in)
			continue
		}
		if err == nil {
			t.Errorf("%q: accepted as frac=%v label=%q, want a refusal", in, b.Frac, b.Label)
		}
	}
}

// validateCards is the gate that turns the refusal above into a `-verify`
// failure, naming the segment so the author does not have to hunt for it.
func TestValidateCardsNamesTheSegmentAndLineOfABadBar(t *testing.T) {
	cfg := config{Segments: []segment{
		{Chapter: "1 . fine", Card: []string{"@c TITLE", "@g @bar 0.5 half"}},
		{Chapter: "2 . broken", Card: []string{"@c TITLE", "", "@r @bar 2.0 impossible"}},
	}}
	err := validateCards(cfg)
	if err == nil {
		t.Fatal("a fraction of 2.0 was accepted")
	}
	for _, want := range []string{"segment 2", "2 . broken", "card line 3"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not locate the bad bar: want %q in it", err, want)
		}
	}
	if err := validateCards(config{Segments: cfg.Segments[:1]}); err != nil {
		t.Errorf("a well-formed card was refused: %v", err)
	}
}

// A non-zero fraction must never draw as nothing: "16-32 px recall is 15.5%" and
// "0 of 11 are byte-identical" are different findings.
func TestBarFillNeverRoundsARealValueAwayToNothing(t *testing.T) {
	cases := []struct {
		frac float64
		want int
	}{
		{0, 0},     // a genuine zero IS an empty track
		{0.001, 1}, // ...but a small real value is not
		{0.155, 3}, // 0.155 * 20 = 3.1
		{0.5, 10},
		{1, barCells},
	}
	for _, tc := range cases {
		if got := (barMark{Frac: tc.frac}).fill(); got != tc.want {
			t.Errorf("frac %v: fill = %d cells, want %d", tc.frac, got, tc.want)
		}
	}
}

// A bar line has to state its own width, or a cardLeft block would measure it on
// the ruler for its markup text and the column of figures beside it would break.
func TestCardLineWidthMeasuresABarAsItsTrackPlusLabel(t *testing.T) {
	const label = "zero human labels"
	got := cardLineWidth("@g @bar 0.3619 " + label)
	want := barCells + barGap + len(label)
	if got != want {
		t.Errorf("bar line width = %d, want %d (track %d + gap %d + label %d)",
			got, want, barCells, barGap, len(label))
	}
	if got := cardLineWidth("@g plain prose"); got != len("plain prose") {
		t.Errorf("prose line width = %d, want %d", got, len("plain prose"))
	}
}

// stripCardMarkup feeds the frame log and the concept-boundary rule. A bar line
// must strip to the text the viewer reads, not to its markup — and above all not
// to something that tests as blank, which would make it a concept boundary.
func TestStripCardMarkupReducesABarToItsLabel(t *testing.T) {
	const line = "@g @bar 0.33 zero human labels, 16-32 px    33.0%"
	got := stripCardMarkup(line)
	want := "zero human labels, 16-32 px    33.0%"
	if got != want {
		t.Errorf("stripped to %q, want %q", got, want)
	}
	if strings.TrimSpace(got) == "" {
		t.Error("a bar line stripped to blank, so it would read as a concept boundary")
	}
}

func TestRenderCardDrawsBarsAsFilledCellsAndReportsHowMany(t *testing.T) {
	term := newTerm(60, 10, inconsolata.Regular8x16)
	lines := []string{
		"@c TITLE",
		"",
		"@d @bar 0.2782 base                0.2782",
		"@g @bar 0.3619 zero human labels   0.3619",
	}
	bars := renderCard(term, lines, true, len(lines))
	if bars != 2 {
		t.Fatalf("renderCard reported %d bars drawn, want 2", bars)
	}
	// 10 rows, 4 lines => top row 3, so the bars are on rows 5 and 6.
	countFill := func(row int) (filled, track int) {
		for x := 0; x < term.cols; x++ {
			c := term.cells[row*term.cols+x]
			if c.r != barFillRune {
				continue
			}
			if c.fg == cBarTrack {
				track++
			} else {
				filled++
			}
		}
		return filled, track
	}
	for _, tc := range []struct {
		row    int
		filled int
	}{
		{5, 6}, // 0.2782 * 20 = 5.56 -> 6
		{6, 7}, // 0.3619 * 20 = 7.24 -> 7
	} {
		filled, track := countFill(tc.row)
		if filled != tc.filled {
			t.Errorf("row %d: %d filled cells, want %d", tc.row, filled, tc.filled)
		}
		// The track is always drawn in full: a bar with no 100% mark is a length
		// with nothing to be a length of.
		if filled+track != barCells {
			t.Errorf("row %d: track is %d cells wide, want %d", tc.row, filled+track, barCells)
		}
	}
}

// ⛔ The regression this pins is a silent one and it changes the NUMBER on
// screen: the fade ramp walks a colour toward the neutrals, cBarTrack is a
// neutral, so a bar left to fade has its filled cells converge on its empty ones
// and a 28% bar reads as a full one two beats later.
func TestABarDoesNotFadeIntoItsOwnTrack(t *testing.T) {
	for age := uint8(0); age <= fadeFloorAge+1; age++ {
		c := cell{r: barFillRune, fg: cGreen, age: age}
		fg, bold := c.faded()
		if fg != cGreen {
			t.Errorf("age %d: a bar cell drew as colour %d, want cGreen (%d) at every age",
				age, fg, cGreen)
		}
		if bold {
			t.Errorf("age %d: a bar cell asked for bold, which means nothing to a rectangle", age)
		}
	}
	// The prose beside it still fades — the reading order has not been switched
	// off, it is just no longer carried by the mark.
	if fg, _ := (cell{r: 'x', fg: cGreen, age: 2}).faded(); fg == cGreen {
		t.Error("ordinary text stopped fading, so the card lost its reading order")
	}
}

// Both rasterisers paint a bar; neither consults a face for it. That is what
// makes the mark the same SHAPE in the bitmap GIF and the antialiased video.
func TestBothRasterisersPaintABarWithoutAGlyph(t *testing.T) {
	term := newTerm(30, 5, inconsolata.Regular8x16)
	if bars := renderCard(term, []string{"@g @bar 1 all of them"}, true, 1); bars != 1 {
		t.Fatalf("renderCard drew %d bars, want 1", bars)
	}
	row := 2 // 5 rows, 1 line => top row 2

	r := newRenderer(term.cols, term.rows, 4)
	gif := r.draw(term, cBG)
	ins := barInset(r.ch)
	// Inside the bar's band the pixel is the bar's colour; in the inset above it
	// is still background, which is what keeps it reading as a bar rather than a
	// filled block butting into the row above.
	at := func(img image.Image, x, y int) uint8 {
		p, ok := img.(*image.Paletted)
		if !ok {
			t.Fatalf("GIF path did not produce a paletted image")
		}
		return p.Pix[y*p.Stride+x]
	}
	midX := r.pad + 2*r.cw
	if got := at(gif, midX, r.pad+row*r.ch+ins+1); got != cGreen {
		t.Errorf("GIF: bar band pixel = palette %d, want cGreen (%d)", got, cGreen)
	}
	if got := at(gif, midX, r.pad+row*r.ch); got != cBG {
		t.Errorf("GIF: the inset above the bar = palette %d, want cBG (%d)", got, cBG)
	}

	dir := t.TempDir()
	hs, err := newHiresSink(dir, term.cols, term.rows, 4, defaultHiresCellWidth, defaultHiresCellHeight)
	if err != nil {
		t.Fatalf("newHiresSink: %v", err)
	}
	hi := hs.render(term)
	hins := barInset(hs.face.ch)
	wantR, wantG, wantB, _ := pal[cGreen].RGBA()
	gotR, gotG, gotB, _ := hi.At(4+2*hs.face.cw, 4+row*hs.face.ch+hins+1).RGBA()
	if gotR != wantR || gotG != wantG || gotB != wantB {
		t.Errorf("hi-res: bar band pixel = (%d,%d,%d), want cGreen (%d,%d,%d)",
			gotR, gotG, gotB, wantR, wantG, wantB)
	}
}

// The floor exists so a cut cannot drift back to a wall of figures with every
// other pacing assertion still green — which is the failure mode it is named
// after: the bars go one edit at a time and nothing else notices.
func TestVerifyRefusesACutThatLostItsBars(t *testing.T) {
	// A timeline that satisfies every OTHER rule in checkPacing, so the only
	// failure it can report is the bar floor. Zeroed verify floors are inert.
	minimal := func(bars int) *timeline {
		return &timeline{
			Frames:   1,
			TotalSec: 1.0,
			Bars:     bars,
			Counts:   map[string]int{"step": 1, "cmd": 1, "rc": 1},
			Secs:     map[string]float64{},
			MinDwell: map[string]float64{},
			Log:      []tlFrame{{N: 1, Start: 0, Secs: 1.0, Class: "out"}},
		}
	}
	cfg := config{Verify: verify{MinBars: 6}}

	fails := checkPacing(cfg, minimal(3))
	if len(fails) != 1 {
		t.Fatalf("3 bars against a floor of 6 reported %d failures %q, want exactly 1", len(fails), fails)
	}
	if !strings.Contains(fails[0], "bar marks drawn") {
		t.Errorf("the failure does not name the bar floor: %q", fails[0])
	}

	if fails := checkPacing(cfg, minimal(6)); len(fails) != 0 {
		t.Errorf("6 bars against a floor of 6 was refused: %q", fails)
	}
	// ⛔ And an unset floor must stay silent rather than defaulting to 1: a cut
	// with no bars in it is a legitimate cut, and a floor nobody asked for would
	// turn every other project's config red.
	if fails := checkPacing(config{}, minimal(0)); len(fails) != 0 {
		t.Errorf("a config with no minBars was refused for having no bars: %q", fails)
	}
}
