// The shared terminal renderer turns script(1) captures and narrative cards
// into an animated GIF and a chaptered MP4.
//
// System explainers often start as a `script(1)` capture but need a paced,
// chaptered visual that can be reviewed without replaying the command. The
// renderer is shared here so each explainer owns only its story and evidence.
// It is Go because that is the only project language (Law 1), and because the
// capture remains renderable without asciinema, agg, or a running container.
//
// The capture it was written for contains SGR colour escapes and nothing else
// (verified: no cursor motion, no erase), so the emulator below is deliberately
// small. Unknown CSI sequences are skipped rather than guessed at.
//
// Frames are emitted one per timing chunk, then diffed: only the changed
// bounding box is encoded, with GIF disposal=None. That keeps a two-minute
// terminal session down to a few MB.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/gif"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/font/inconsolata"
	"golang.org/x/image/math/fixed"
)

// ── palette ──────────────────────────────────────────────────────────────────
// Kept to 16 entries so the GIF encodes at 4 bits per pixel. 14 are spent, and
// three of those are the fade ramp below, so a new accent still fits.

const (
	cBG = iota
	cFG
	cGreen
	cGrey
	cYellow
	cCyan
	cWhite
	cRed
	cBlue
	cMagenta
	cDimBG    // title-card backdrop
	cBarTrack // the unfilled remainder of a bar; see bar.go
	cFade1    // ── the fade ramp; see fadeLadder ──
	cFade2
	cFade3
)

var pal = color.Palette{
	cBG:      color.RGBA{0x0b, 0x0e, 0x14, 0xff},
	cFG:      color.RGBA{0xc9, 0xd1, 0xd9, 0xff},
	cGreen:   color.RGBA{0x56, 0xd3, 0x64, 0xff},
	cGrey:    color.RGBA{0x76, 0x7f, 0x8a, 0xff},
	cYellow:  color.RGBA{0xe3, 0xb3, 0x41, 0xff},
	cCyan:    color.RGBA{0x56, 0xc7, 0xe8, 0xff},
	cWhite:   color.RGBA{0xff, 0xff, 0xff, 0xff},
	cRed:     color.RGBA{0xf8, 0x51, 0x49, 0xff},
	cBlue:    color.RGBA{0x79, 0xc0, 0xff, 0xff},
	cMagenta: color.RGBA{0xd2, 0xa8, 0xff, 0xff},
	cDimBG:   color.RGBA{0x14, 0x1a, 0x24, 0xff},

	// cFG mixed toward cBG at 78%: light enough to read as the rest of the
	// meter against the background, dark enough that a full-width track never
	// competes with the prose beside it for the eye.
	cBarTrack: color.RGBA{0x35, 0x39, 0x3f, 0xff},

	// cFG mixed toward cBG at 14%, 28% and 42%. The last step lands on the tone
	// superseded text used to jump to in one go, so the fade ends exactly where
	// it always did and only the route there is new.
	cFade1: color.RGBA{0xae, 0xb6, 0xbd, 0xff},
	cFade2: color.RGBA{0x94, 0x9a, 0xa2, 0xff},
	cFade3: color.RGBA{0x79, 0x7f, 0x86, 0xff},
}

var uniforms = func() []image.Image {
	u := make([]image.Image, len(pal))
	for i, c := range pal {
		u[i] = image.NewUniform(c)
	}
	return u
}()

// ── the fade ─────────────────────────────────────────────────────────────────
// Text that is no longer the thing to read recedes; it does not switch off. A
// cell therefore carries an age — how many beats ago it lost focus — and the
// rasterisers resolve that age to a colour at draw time. The authored colour is
// never overwritten, so ageing is reversible and cannot accumulate error.
//
// The ramp is deliberately shallow and deliberately long. Step one keeps the
// authored hue and only drops the weight, which the eye reads as "settled"
// rather than "gone"; the neutral steps then walk down to the resting tone over
// three further beats. Every step stays comfortably readable — the ramp exists
// to rank what is on screen, not to hide it.
//
// What recedes is chroma as much as brightness, and the two must not be
// confused. An accent that loses its hue reads as background even where the
// neutral it lands on is no darker measured in isolation: cRed is the standing
// counter-example, dimmer by luma than the floor it fades onto and far louder
// on screen. So the ladder is not gated on a brightness comparison.

// fadeLadder is the neutral tail of the ramp, walked one entry per beat once
// the weight has already been dropped.
var fadeLadder = [...]uint8{cFade1, cFade2, cFade3}

// fadeFloorAge is the age at which a cell has reached the last ladder step.
// Ageing stops there: beyond it nothing would change, and a counter that kept
// climbing would make two identical frames compare unequal and re-encode.
const fadeFloorAge = uint8(1 + len(fadeLadder))

// sgrColor maps an SGR colour parameter onto a palette index.
func sgrColor(n int) (uint8, bool) {
	switch n {
	case 30, 90:
		return cGrey, true
	case 31, 91:
		return cRed, true
	case 32, 92:
		return cGreen, true
	case 33, 93:
		return cYellow, true
	case 34, 94:
		return cBlue, true
	case 35, 95:
		return cMagenta, true
	case 36, 96:
		return cCyan, true
	case 37, 97:
		return cWhite, true
	case 39:
		return cFG, true
	}
	return 0, false
}

// ── glyph coverage ───────────────────────────────────────────────────────────
// inconsolata8x16 covers Latin-1 and a scattering of Latin Extended-A, but none
// of the punctuation the capture's prose uses. Substituting one ASCII rune keeps
// column alignment exact, which matters: the capture is full of hashes and
// aligned tables.

var translit = map[rune]rune{
	'—': '-', '–': '-', '−': '-', // em/en dash, minus
	'→': '>', '⇒': '>', '←': '<', // arrows
	'…': '.', '·': '.', '•': '*', // ellipsis, middot, bullet
	'‘': '\'', '’': '\'', // curly single quotes
	'“': '"', '”': '"', // curly double quotes
	'─': '-', '│': '|', '┌': '+', '┐': '+',
	'└': '+', '┘': '+', '├': '+', '┤': '+',
	'═': '=', '║': '|',
	'✓': 'y', '✗': 'x', '⛔': '!', '⭐': '*',
	' ': ' ',
}

func printable(r rune, face *basicfont.Face) rune {
	if r < 0x20 {
		return ' '
	}
	if s, ok := translit[r]; ok {
		r = s
	}
	if r < 0x80 {
		return r
	}
	for _, rg := range face.Ranges {
		if r >= rg.Low && r < rg.High {
			return r
		}
	}
	return '?'
}

// ── terminal ─────────────────────────────────────────────────────────────────

type cell struct {
	r    rune
	fg   uint8 // the authored colour, never overwritten by the fade
	bold bool
	age  uint8 // beats since this cell lost focus; 0 is the live one
}

// faded resolves a cell's authored attributes into what is actually drawn. Both
// rasterisers go through it, so the GIF and the video cannot disagree about how
// far a line has receded.
//
// cGrey is exempt from the ladder. It is the palette's designated recessive
// tone — what `@d` markup and SGR 90 mean — and it already sits on the floor
// (within a point of cFade3), so walking it down a ramp built from brighter
// neutrals would make it flare before it settled.
func (c cell) faded() (uint8, bool) {
	switch {
	case c.age == 0:
		return c.fg, c.bold
	// ⛔ A bar is exempt from the ramp entirely, and this is a correctness rule
	// rather than a style one. The ramp walks a colour toward the neutrals, and
	// the neutrals are where cBarTrack already lives — so a receding bar's
	// FILLED cells converge on the tone of its own EMPTY ones, and a 30% bar
	// silently becomes a 100% bar two beats after the card moves on. A mark
	// whose whole meaning is its length may not change length. The card's
	// reading order is carried by the prose beside it, which still fades.
	case c.r == barFillRune:
		return c.fg, false
	// Step one is the authored colour at regular weight: the smallest change
	// that still reads as a handover.
	case c.age == 1 || c.fg == cGrey:
		return c.fg, false
	}
	return fadeLadder[min(int(c.age)-2, len(fadeLadder)-1)], false
}

type term struct {
	cols, rows int
	cells      []cell // row-major, len == cols*rows
	cx, cy     int
	fg         uint8
	bold       bool
	age        uint8 // stamped onto every cell put; only renderCard sets it
	face       *basicfont.Face
	esc        []byte // partial escape sequence
	inEsc      bool
	pend       []byte // partial UTF-8 sequence
}

func newTerm(cols, rows int, face *basicfont.Face) *term {
	t := &term{cols: cols, rows: rows, face: face, fg: cFG}
	t.cells = make([]cell, cols*rows)
	t.clear()
	return t
}

func (t *term) clear() {
	for i := range t.cells {
		t.cells[i] = cell{r: ' ', fg: cFG}
	}
	t.cx, t.cy = 0, 0
	t.fg, t.bold, t.age = cFG, false, 0
}

func (t *term) scroll() {
	copy(t.cells, t.cells[t.cols:])
	last := t.cells[(t.rows-1)*t.cols:]
	for i := range last {
		last[i] = cell{r: ' ', fg: cFG}
	}
	t.cy = t.rows - 1
}

func (t *term) newline() {
	t.cx = 0
	t.cy++
	if t.cy >= t.rows {
		t.scroll()
	}
}

func (t *term) put(r rune) {
	if t.cx >= t.cols {
		t.newline()
	}
	t.cells[t.cy*t.cols+t.cx] = cell{r: r, fg: t.fg, bold: t.bold, age: t.age}
	t.cx++
}

// applySGR handles the only escape family the capture actually uses.
func (t *term) applySGR(params string) {
	if params == "" {
		params = "0"
	}
	for _, p := range strings.Split(params, ";") {
		n, err := strconv.Atoi(p)
		if err != nil {
			continue
		}
		switch {
		case n == 0:
			t.fg, t.bold = cFG, false
		case n == 1:
			t.bold = true
		case n == 22:
			t.bold = false
		default:
			if c, ok := sgrColor(n); ok {
				t.fg = c
			}
		}
	}
}

func (t *term) writeByte(b byte) {
	if t.inEsc {
		t.esc = append(t.esc, b)
		// CSI: ESC [ params final. Final byte is in @..~
		if len(t.esc) == 1 {
			if b != '[' { // a non-CSI escape; a single-byte one, drop it
				t.inEsc, t.esc = false, nil
			}
			return
		}
		if b >= '@' && b <= '~' {
			if b == 'm' {
				t.applySGR(string(t.esc[1 : len(t.esc)-1]))
			}
			// every other CSI is deliberately ignored
			t.inEsc, t.esc = false, nil
		}
		return
	}
	switch b {
	case 0x1b:
		t.inEsc, t.esc = true, t.esc[:0]
		return
	case '\r':
		t.cx = 0
		return
	case '\n':
		t.newline()
		return
	case '\t':
		n := 8 - t.cx%8
		for i := 0; i < n; i++ {
			t.put(' ')
		}
		return
	case '\b':
		if t.cx > 0 {
			t.cx--
		}
		return
	case 0x07:
		return
	}
	if b < 0x80 {
		t.put(printable(rune(b), t.face))
		return
	}
	// UTF-8: buffer until a full rune decodes
	t.pend = append(t.pend, b)
	if r, size := utf8.DecodeRune(t.pend); r != utf8.RuneError || size > 1 {
		t.put(printable(r, t.face))
		t.pend = t.pend[:0]
	} else if len(t.pend) >= 4 {
		t.pend = t.pend[:0]
	}
}

func (t *term) writeString(s string) {
	for i := 0; i < len(s); i++ {
		t.writeByte(s[i])
	}
}

// ── rendering ────────────────────────────────────────────────────────────────

type renderer struct {
	cols, rows, pad int
	cw, ch, base    int
	reg, bold       *basicfont.Face
	w, h            int
}

const (
	defaultGIFCellHeight   = 18
	defaultHiresCellWidth  = 16
	defaultHiresCellHeight = 36
)

func newRenderer(cols, rows, pad int) *renderer {
	r := &renderer{
		cols: cols, rows: rows, pad: pad,
		// The bitmap face is 16 pixels tall. An 18-pixel row gives each line a
		// two-pixel breathing space without changing the terminal's 120x32
		// content grid. The hi-res path uses the exact 2x counterpart (16x36).
		cw: 8, ch: defaultGIFCellHeight, base: 13,
		reg: inconsolata.Regular8x16, bold: inconsolata.Bold8x16,
	}
	r.w = cols*r.cw + 2*pad
	r.h = rows*r.ch + 2*pad
	return r
}

func (r *renderer) draw(t *term, bg uint8) *image.Paletted {
	img := image.NewPaletted(image.Rect(0, 0, r.w, r.h), pal)
	if bg != 0 {
		for i := range img.Pix {
			img.Pix[i] = bg
		}
	}
	d := &font.Drawer{Dst: img}
	for y := 0; y < r.rows; y++ {
		row := t.cells[y*r.cols : (y+1)*r.cols]
		x := 0
		for x < r.cols {
			c := row[x]
			// A bar cell is painted, not drawn from a face. The palette index is
			// written straight into Pix because fg already IS an index, so the
			// mark cannot be requantised into a neighbouring colour.
			if c.r == barFillRune {
				fg, _ := c.faded()
				ins := barInset(r.ch)
				x0 := r.pad + x*r.cw
				for py := r.pad + y*r.ch + ins; py < r.pad+(y+1)*r.ch-ins; py++ {
					o := py*img.Stride + x0
					for i := 0; i < r.cw; i++ {
						img.Pix[o+i] = fg
					}
				}
				x++
				continue
			}
			if c.r == ' ' {
				x++
				continue
			}
			// group a run of identical drawn attributes into one DrawString.
			// Grouping on the faded result rather than the authored one keeps
			// runs intact once a whole region has settled on the same step.
			fg, bold := c.faded()
			run := []rune{c.r}
			j := x + 1
			for j < r.cols && row[j].r != ' ' {
				nfg, nbold := row[j].faded()
				if nfg != fg || nbold != bold {
					break
				}
				run = append(run, row[j].r)
				j++
			}
			d.Src = uniforms[fg]
			if bold {
				d.Face = r.bold
			} else {
				d.Face = r.reg
			}
			d.Dot = fixed.P(r.pad+x*r.cw, r.pad+y*r.ch+r.base)
			d.DrawString(string(run))
			x = j
		}
	}
	return img
}

// diffRect returns the bounding box of pixels that differ between a and b.
// An empty rectangle means the two frames are identical.
func diffRect(a, b *image.Paletted) image.Rectangle {
	w, h := a.Rect.Dx(), a.Rect.Dy()
	minX, minY, maxX, maxY := w, h, -1, -1
	for y := 0; y < h; y++ {
		ao, bo := y*a.Stride, y*b.Stride
		for x := 0; x < w; x++ {
			if a.Pix[ao+x] != b.Pix[bo+x] {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if maxX < 0 {
		return image.Rectangle{}
	}
	return image.Rect(minX, minY, maxX+1, maxY+1)
}

func crop(src *image.Paletted, r image.Rectangle) *image.Paletted {
	out := image.NewPaletted(r, pal)
	draw.Draw(out, r, src, r.Min, draw.Src)
	return out
}

// ── config ───────────────────────────────────────────────────────────────────

type segment struct {
	// A card segment: full-screen text, revealed a line at a time and then held
	// for CardSecs on the completed card.
	Card     []string `json:"card"`
	CardSecs float64  `json:"cardSecs"`
	CardLeft bool     `json:"cardLeft"` // share one left margin across the block
	// A replay segment: a script(1) pair.
	Typescript string `json:"typescript"`
	Timing     string `json:"timing"`
	// Clear the screen before this segment.
	Clear bool `json:"clear"`
	// Chapter opens a named MP4 chapter here. Replay segments name their own
	// chapters from the step banners inside them; this is for the cards.
	Chapter string `json:"chapter"`
}

type config struct {
	Cols int    `json:"cols"`
	Rows int    `json:"rows"`
	Pad  int    `json:"pad"`
	Out  string `json:"out"` // the GIF
	MP4  string `json:"mp4"` // the video, encoded by -all

	// Pacing is content-driven; see pacing.go for why the old minFrame/maxGap/
	// speed triple could not be tuned into a watchable cut (#750).
	Pacing pacing `json:"pacing"`
	Verify verify `json:"verify"`

	Segments []segment `json:"segments"`
}

// verify holds the floors the render asserts about itself. They are the DoD of
// #750 written as numbers a program can fail on: a pacing claim nothing can
// falsify is a preference, not a property.
type verify struct {
	MinStepSecs        float64 `json:"minStepSecs"`        // no proof step may fly past
	MinCmdSecs         float64 `json:"minCmdSecs"`         // a `$ ` line is readable before its output
	MinRCSecs          float64 `json:"minRCSecs"`          // a verdict is readable
	MinEmphasisSecs    float64 `json:"minEmphasisSecs"`    // a money line is readable
	MinCardRevealSecs  float64 `json:"minCardRevealSecs"`  // a card line must arrive smoothly
	MinConceptHoldSecs float64 `json:"minConceptHoldSecs"` // one concept gets a reading beat
	MinOpeningSecs     float64 `json:"minOpeningSecs"`     // the opening establishes the story
	TotalSecsMin       float64 `json:"totalSecsMin"`       // guards a pacing edit that guts the cut
	TotalSecsMax       float64 `json:"totalSecsMax"`       // ...and one that bloats it
	MinChapters        int     `json:"minChapters"`        // scrubbing has to stay possible
	MP4ToleranceSecs   float64 `json:"mp4ToleranceSecs"`   // encoded vs intended duration
	// MinBars is the floor on non-text marks the render actually DREW. A cut
	// whose comparisons are carried by bars can otherwise lose them one edit at
	// a time — a beat deleted here, a bar line reworded into prose there — and
	// arrive back at a wall of figures with every other floor still green.
	MinBars int `json:"minBars"`
}

func (v *verify) defaults() {
	setf := func(dst *float64, d float64) {
		if *dst <= 0 {
			*dst = d
		}
	}
	setf(&v.MinStepSecs, 2.0)
	setf(&v.MinCmdSecs, 0.4)
	setf(&v.MinRCSecs, 0.5)
	setf(&v.MinEmphasisSecs, 1.5)
	setf(&v.MinCardRevealSecs, 0.25)
	setf(&v.MinConceptHoldSecs, 0.8)
	setf(&v.MinOpeningSecs, 10)
	setf(&v.TotalSecsMax, 600)
	setf(&v.MP4ToleranceSecs, 0.5)
}

// stripCardMarkup drops a leading colour token, leaving the text the viewer
// actually reads. Both the layout and the frame log need it — the log because a
// card frame's trigger line should be what is on screen, not the config's markup
// for it (#750 B1).
func stripCardMarkup(s string) string {
	_, _, rest := cardColorToken(s)
	// A bar line strips to its LABEL. The frame log should say what the viewer
	// reads on that row, and the concept-boundary rule tests these strings for
	// emptiness — a bar line is never a boundary, and its label keeps it from
	// looking like one.
	if b, ok, err := parseBar(rest); ok && err == nil {
		return b.Label
	}
	return rest
}

// cardColorToken splits a leading "@x " colour token off a card line, returning
// the colour it names, whether it named one, and the text after it.
func cardColorToken(s string) (uint8, bool, string) {
	if len(s) > 3 && s[0] == '@' {
		if c, ok := cardColor[s[:2]]; ok {
			return c, true, s[3:]
		}
	}
	return cFG, false, s
}

// cardMarkup: a leading "@x " picks the colour for the whole line.
var cardColor = map[string]uint8{
	"@c": cCyan, "@y": cYellow, "@g": cGreen, "@d": cGrey,
	"@w": cWhite, "@r": cRed, "@b": cBlue, "@m": cMagenta,
}

// renderCard lays a card out centred vertically. With block=true every line
// shares one left margin, so tables and dot-leaders stay in column; otherwise
// each line is centred on its own, which reads better for a title.
//
// show is how many lines to draw. The layout is always computed from the FULL
// card, so a partial draw is the finished card with its tail not yet arrived —
// nothing shifts as the rest lands. That is what makes the reveal readable
// instead of a re-flow.
//
// Blank lines divide concepts. Only the newest visible concept is drawn at full
// strength; each earlier one is one step further down the fade ramp, so a card
// carries its own reading order — brightest is where you are, and the concepts
// behind it stay legible as context rather than being switched off.
// It returns how many bar marks it actually drew, which is what the verify
// floor counts. Counting the marks the RENDERER put on screen rather than the
// ones the config authored is the difference between a floor and a wish: a bar
// line pushed off the bottom of a tall card is authored and invisible, and only
// the drawn count can tell those apart.
func renderCard(t *term, lines []string, block bool, show int) int {
	t.clear()
	top := (t.rows - len(lines)) / 2
	if top < 0 {
		top = 0
	}
	blockLeft := 0
	if block {
		maxLen := 0
		for _, ln := range lines {
			if n := cardLineWidth(ln); n > maxLen {
				maxLen = n
			}
		}
		blockLeft = (t.cols - maxLen) / 2
		if blockLeft < 0 {
			blockLeft = 0
		}
	}
	ages := cardConceptAges(lines, show)
	bars := 0
	for i, ln := range lines {
		if i >= show {
			break
		}
		fg, named, rest := cardColorToken(ln)
		bold := named
		ln = rest
		t.age = ages[i]
		if top+i >= t.rows {
			break
		}
		bar, isBar, _ := parseBar(ln) // a bad bar was refused at load; see validateCards
		width := len([]rune(ln))
		if isBar {
			width = bar.width()
		}
		left := blockLeft
		if !block {
			left = (t.cols - width) / 2
		}
		if left < 0 {
			left = 0
		}
		t.cy, t.cx = top+i, left
		t.fg, t.bold = fg, bold
		if isBar {
			t.putBar(bar, fg)
			bars++
			continue
		}
		for _, r := range ln {
			if t.cx >= t.cols {
				break
			}
			t.put(printable(r, t.face))
		}
	}
	t.fg, t.bold, t.age = cFG, false, 0
	return bars
}

// cardConceptAges gives every visible line the age renderCard should stamp on
// it: 0 for the concept being read now, 1 for the one before it, and so on up
// the card. A concept therefore leaves the foreground over as many beats as
// there are ramp steps instead of all at once, and the lines of a multi-line
// concept always recede together.
func cardConceptAges(lines []string, show int) []uint8 {
	if show > len(lines) {
		show = len(lines)
	}
	ages := make([]uint8, show)
	activeStart, _ := activeCardConcept(lines, show)
	blank := func(i int) bool { return strings.TrimSpace(stripCardMarkup(lines[i])) == "" }
	age := uint8(0)
	for i := activeStart - 1; i >= 0; i-- {
		if blank(i) {
			continue
		}
		if blank(i + 1) { // crossing a divider upward starts an older concept
			age = min(age+1, fadeFloorAge)
		}
		ages[i] = age
	}
	return ages
}

// activeCardConcept returns the half-open range of the newest non-blank block
// among the first show lines. A just-revealed spacer keeps focus on the concept
// above it; focus moves only when the next non-blank line arrives.
func activeCardConcept(lines []string, show int) (start, end int) {
	if show > len(lines) {
		show = len(lines)
	}
	end = show
	for end > 0 && strings.TrimSpace(stripCardMarkup(lines[end-1])) == "" {
		end--
	}
	if end == 0 {
		return 0, 0
	}
	start = end - 1
	for start > 0 && strings.TrimSpace(stripCardMarkup(lines[start-1])) != "" {
		start--
	}
	return start, end
}

// ageExisting moves the terminal's prior state one step further back without
// changing a byte of the capture. The next emitted unit is then the brightest
// information on screen — command, output, verdict, and finally the next step —
// while the units behind it trail off down the ramp over the following beats.
// One step per unit is what makes the recession readable: the line you just
// finished is still nearly as legible as it was, and only the history several
// units back has settled onto the floor.
func ageExisting(t *term) {
	for i := range t.cells {
		if t.cells[i].r == ' ' || t.cells[i].age >= fadeFloorAge {
			continue
		}
		t.cells[i].age++
	}
}

// readTiming parses the classic `script --timing` format: "<delay> <nbytes>".
func readTiming(path string) ([][2]float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out [][2]float64
	for _, ln := range strings.Split(string(raw), "\n") {
		f := strings.Fields(ln)
		if len(f) != 2 {
			continue
		}
		d, e1 := strconv.ParseFloat(f[0], 64)
		n, e2 := strconv.ParseFloat(f[1], 64)
		if e1 != nil || e2 != nil {
			continue
		}
		out = append(out, [2]float64{d, n})
	}
	return out, nil
}

func main() {
	cfgPath := flag.String("config", "", "path to the JSON render config")
	pngDir := flag.String("png", "", "also dump composited full frames here, as PNGs (for eyeballing)")
	pngEvery := flag.Int("png-every", 25, "with -png, dump every Nth frame")
	framesDir := flag.String("frames", "", "hi-res mode: write an antialiased PNG sequence + ffconcat playlist here instead of a GIF")
	cellW := flag.Int("cell-w", defaultHiresCellWidth, "with -frames, cell width in px (the GIF's bitmap face is 8)")
	cellH := flag.Int("cell-h", defaultHiresCellHeight, "with -frames, cell height in px (2x the GIF's 18-pixel rows)")
	recTS := flag.String("record-typescript", "", "record mode: tee stdin to this typescript file (see record.go)")
	recTim := flag.String("record-timing", "", "record mode: write the companion timing file here")
	all := flag.Bool("all", false, "the default regen: GIF + hi-res frames + chapters + the ffmpeg encode, then verify the result")
	verifyOnly := flag.Bool("verify", false, "plan the render and assert the pacing floors without rasterising anything")
	tlPath := flag.String("timeline", "timeline.json", "where to write the render's own account of its pacing")
	flag.Parse()
	// Record mode is the inverse of everything below it and shares none of it —
	// no config, no terminal, no rasteriser — so it returns before the render
	// path's -config requirement can reject a perfectly valid recording run.
	if *recTS != "" || *recTim != "" {
		if *recTS == "" || *recTim == "" {
			fmt.Fprintln(os.Stderr, "video: -record-typescript and -record-timing must be given together")
			os.Exit(2)
		}
		if err := recordMode(*recTS, *recTim); err != nil {
			fmt.Fprintln(os.Stderr, "video: record:", err)
			os.Exit(1)
		}
		return
	}
	if *cfgPath == "" {
		fmt.Fprintln(os.Stderr, "video: -config is required")
		os.Exit(2)
	}
	raw, err := os.ReadFile(*cfgPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "video:", err)
		os.Exit(1)
	}
	var cfg config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		fmt.Fprintln(os.Stderr, "video: bad config:", err)
		os.Exit(1)
	}
	if err := cfg.Pacing.defaults(); err != nil {
		fmt.Fprintln(os.Stderr, "video: bad pacing config:", err)
		os.Exit(1)
	}
	cfg.Verify.defaults()
	// Card markup is checked before anything else runs, so a malformed bar costs
	// a second and names its own segment instead of rendering as prose.
	if err := validateCards(cfg); err != nil {
		fmt.Fprintln(os.Stderr, "video: bad card:", err)
		os.Exit(1)
	}
	// Paths in the config are relative to the config file, not to the working
	// directory. The config ships next to the capture it renders, so anchoring
	// it anywhere else means the copy that gets shared only runs on the one box
	// whose absolute paths were baked in.
	cfgDir := filepath.Dir(*cfgPath)
	rel := func(p string) string {
		if p == "" || filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(cfgDir, p)
	}
	cfg.Out = rel(cfg.Out)
	cfg.MP4 = rel(cfg.MP4)
	for i := range cfg.Segments {
		cfg.Segments[i].Typescript = rel(cfg.Segments[i].Typescript)
		cfg.Segments[i].Timing = rel(cfg.Segments[i].Timing)
	}

	// -all is the default regen: one command produces every shipped artifact and
	// then checks the one it cannot see. Everything it does is also reachable
	// one piece at a time, because a debug loop that can only run the whole
	// pipeline is not a debug loop.
	hiresPath := *framesDir
	if *all && hiresPath == "" {
		hiresPath = filepath.Join(cfgDir, "frames-hires")
	}
	timelinePath := rel(*tlPath)

	// Plan-only: walk the whole render with no rasteriser attached, so the
	// pacing can be inspected and asserted in under a second.
	if *verifyOnly {
		tl := walk(cfg, func(float64) {})
		report(cfg, tl, timelinePath)
		if fails := checkPacing(cfg, tl); len(fails) > 0 {
			for _, f := range fails {
				fmt.Fprintln(os.Stderr, "video: VERIFY FAIL:", f)
			}
			os.Exit(1)
		}
		fmt.Println("video: verify OK")
		return
	}

	var gifTL, hiTL *timeline

	if *framesDir == "" || *all {
		tl, err := renderGIF(cfg, *pngDir, *pngEvery)
		if err != nil {
			fmt.Fprintln(os.Stderr, "video:", err)
			os.Exit(1)
		}
		gifTL = tl
	}

	if hiresPath != "" {
		tl, list, err := renderHires(cfg, hiresPath, *cellW, *cellH)
		if err != nil {
			fmt.Fprintln(os.Stderr, "video:", err)
			os.Exit(1)
		}
		hiTL = tl
		chapters := filepath.Join(hiresPath, "chapters.ffmetadata")
		if err := tl.writeChapters(chapters); err != nil {
			fmt.Fprintln(os.Stderr, "video:", err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s: %d chapters\n", chapters, len(tl.Chapters))
		if *all {
			if cfg.MP4 == "" {
				fmt.Fprintln(os.Stderr, "video: -all needs \"mp4\" in the config")
				os.Exit(1)
			}
			if err := encodeMP4(list, chapters, cfg.MP4, tl.TotalSec, cfg.Verify.MP4ToleranceSecs); err != nil {
				fmt.Fprintln(os.Stderr, "video:", err)
				os.Exit(1)
			}
		}
	}

	tl := hiTL
	if tl == nil {
		tl = gifTL
	}
	// The two rasterisers share one timing walk precisely so they cannot
	// disagree; when both ran, say so out loud rather than assuming it.
	if gifTL != nil && hiTL != nil && math.Abs(gifTL.TotalSec-hiTL.TotalSec) > 0.01 {
		fmt.Fprintf(os.Stderr, "video: GIF and MP4 timelines disagree: %.2fs vs %.2fs\n",
			gifTL.TotalSec, hiTL.TotalSec)
		os.Exit(1)
	}
	report(cfg, tl, timelinePath)
	if fails := checkPacing(cfg, tl); len(fails) > 0 {
		for _, f := range fails {
			fmt.Fprintln(os.Stderr, "video: VERIFY FAIL:", f)
		}
		os.Exit(1)
	}
}

// walk runs the whole render with the given frame sink. Every mode goes through
// it, so the plan, the GIF and the video are three readings of one timeline.
func walk(cfg config, emit func(secs float64)) *timeline {
	r := newRenderer(cfg.Cols, cfg.Rows, cfg.Pad)
	t := newTerm(cfg.Cols, cfg.Rows, r.reg)
	tl := newTimeline()
	runSegments(cfg, t, tl, emit)
	// Hold the last frame so a loop does not snap straight back to the title.
	tl.mark("tail", "", tailHold, true)
	emit(tailHold)
	tl.finish(tl.Now)
	return tl
}

// tailHold is the pause on the closing card before the GIF loops.
const tailHold = 3.0

func renderGIF(cfg config, pngDir string, pngEvery int) (*timeline, error) {
	r := newRenderer(cfg.Cols, cfg.Rows, cfg.Pad)
	t := newTerm(cfg.Cols, cfg.Rows, r.reg)
	g := &gif.GIF{
		Config:    image.Config{ColorModel: pal, Width: r.w, Height: r.h},
		LoopCount: 0,
	}
	var prev *image.Paletted
	totalCS := 0

	// emit renders the current terminal state and appends it, coalescing frames
	// that changed nothing into the previous frame's delay. That coalescing is
	// why the paced cut costs no bytes for its pauses: a hold is a longer delay
	// on a frame that already exists, not a new one.
	emit := func(secs float64) {
		cs := int(secs*100 + 0.5)
		if cs < 2 {
			cs = 2 // browsers clamp anything under 0.02s anyway
		}
		img := r.draw(t, cBG)
		if pngDir != "" {
			nth := len(g.Image)
			if nth%pngEvery == 0 {
				if f, err := os.Create(fmt.Sprintf("%s/frame-%04d.png", pngDir, nth)); err == nil {
					_ = png.Encode(f, img)
					_ = f.Close()
				}
			}
		}
		if prev == nil {
			g.Image = append(g.Image, img)
			g.Delay = append(g.Delay, cs)
			g.Disposal = append(g.Disposal, gif.DisposalNone)
			prev = img
			totalCS += cs
			return
		}
		rect := diffRect(prev, img)
		if rect.Empty() {
			g.Delay[len(g.Delay)-1] += cs
			totalCS += cs
			return
		}
		g.Image = append(g.Image, crop(img, rect))
		g.Delay = append(g.Delay, cs)
		g.Disposal = append(g.Disposal, gif.DisposalNone)
		prev = img
		totalCS += cs
	}

	tl := newTimeline()
	runSegments(cfg, t, tl, emit)
	tl.mark("tail", "", tailHold, true)
	emit(tailHold)
	tl.finish(tl.Now)

	if err := os.MkdirAll(filepath.Dir(cfg.Out), 0o755); err != nil {
		return nil, fmt.Errorf("create GIF output directory: %w", err)
	}
	f, err := os.Create(cfg.Out)
	if err != nil {
		return nil, err
	}
	if err := gif.EncodeAll(f, g); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("gif encode: %w", err)
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	st, _ := os.Stat(cfg.Out)
	fmt.Printf("wrote %s: %dx%d, %d frames, %.1fs, %.1f MiB\n",
		cfg.Out, r.w, r.h, len(g.Image), float64(totalCS)/100, float64(st.Size())/(1<<20))
	return tl, nil
}

// renderHires shares the terminal emulator and the timing walk with the GIF path
// and differs only in the rasteriser. See hires.go for why upscaling the GIF
// instead is not the same picture.
func renderHires(cfg config, dir string, cellW, cellH int) (*timeline, string, error) {
	r := newRenderer(cfg.Cols, cfg.Rows, cfg.Pad)
	t := newTerm(cfg.Cols, cfg.Rows, r.reg)
	// Scale the margin with the cell so the frame keeps its proportions.
	hs, err := newHiresSink(dir, cfg.Cols, cfg.Rows, cfg.Pad*cellW/8, cellW, cellH)
	if err != nil {
		return nil, "", err
	}
	var emitErr error
	tl := newTimeline()
	emit := func(secs float64) {
		if emitErr == nil {
			emitErr = hs.emit(t, secs)
		}
	}
	runSegments(cfg, t, tl, emit)
	tl.mark("tail", "", tailHold, true)
	emit(tailHold)
	tl.finish(tl.Now)
	if emitErr != nil {
		return nil, "", emitErr
	}
	list, err := hs.close()
	if err != nil {
		return nil, "", err
	}
	return tl, list, nil
}

// runSegments walks the config's cards and script(1) replays, calling emit once
// per frame with that frame's on-screen time and recording what it did in tl.
// Both rasterisers drive it, so the GIF and the video cannot drift apart on
// content or on timing — and both get the same timeline.json and chapters.
func runSegments(cfg config, t *term, tl *timeline, emit func(secs float64)) {
	for si, seg := range cfg.Segments {
		if len(seg.Card) > 0 {
			sx := tl.openSegment("card", seg.Chapter)
			if seg.Chapter != "" {
				tl.chapter(seg.Chapter)
			}
			// Reveal line by line, then hold the finished card. Blank lines are
			// concept boundaries: the completed concept gets a reading beat,
			// and when the next begins renderCard steps everything before it
			// one rung further down the fade ramp.
			for n := 1; n <= len(seg.Card); n++ {
				bars := renderCard(t, seg.Card, seg.CardLeft, n)
				// Counted on the COMPLETED card only. Every earlier reveal is a
				// prefix of it, so charging each one would count the same mark
				// once per line that follows it.
				if n == len(seg.Card) {
					tl.Bars += bars
				}
				d := cfg.Pacing.CardReveal
				line := strings.TrimSpace(stripCardMarkup(seg.Card[n-1]))
				nextBlank := n == len(seg.Card) ||
					strings.TrimSpace(stripCardMarkup(seg.Card[n])) == ""
				if line != "" && nextBlank {
					d += cfg.Pacing.CardConceptHold
				}
				if n == len(seg.Card) {
					hold := seg.CardSecs
					if hold <= 0 {
						hold = 2.5
					}
					d += hold
				}
				tl.mark("card", stripCardMarkup(seg.Card[n-1]), d, true)
				emit(d)
			}
			tl.closeSegment(sx)
			continue
		}
		if seg.Typescript == "" {
			continue
		}
		if seg.Clear {
			t.clear()
		}
		body, err := os.ReadFile(seg.Typescript)
		if err != nil {
			fmt.Fprintln(os.Stderr, "video:", err)
			os.Exit(1)
		}
		// script(1) writes a "Script started on ..." header that the timing file
		// does not account for. Drop exactly that first line.
		if i := strings.IndexByte(string(body), '\n'); i >= 0 {
			body = body[i+1:]
		}
		chunks, err := readTiming(seg.Timing)
		if err != nil {
			fmt.Fprintln(os.Stderr, "video:", err)
			os.Exit(1)
		}
		sx := tl.openSegment("replay", filepath.Base(seg.Typescript))
		units := planSegment(body, chunks, &cfg.Pacing)
		wrote, held, skipped := 0, 0.0, 0
		for _, u := range units {
			if u.Class == "step" {
				tl.chapter(u.Line)
			}
			if u.Emit {
				ageExisting(t)
			}
			t.writeString(string(u.Raw))
			wrote += len(u.Raw)
			tl.mark(u.Class, u.Line, u.Dwell, u.Emit)
			if !u.Emit {
				skipped++
				continue
			}
			held += u.Dwell
			emit(u.Dwell)
		}
		tl.closeSegment(sx)
		fmt.Fprintf(os.Stderr, "segment %d: %s -> %d chunks, %d units (%d skipped), %d of %d bytes, %.1fs\n",
			si, filepath.Base(seg.Typescript), len(chunks), len(units), skipped, wrote, len(body), held)
	}
}
