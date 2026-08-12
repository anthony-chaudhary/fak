// hires.go — the crisp path.
//
// Why it exists: the GIF path rasterises with inconsolata8x16, a BITMAP face.
// Its glyphs only exist at one size, so the only way to get a 1960x1192 video
// out of it is to render 980x596 and upscale, and an upscale invents no detail
// — it just makes every stroke a 2px staircase. Worse, H.264's yuv420p halves
// the chroma plane, so at a 980-wide source the green and cyan text carried
// 490 samples of colour across 120 columns: four per glyph.
//
// This path rasterises Go Mono, a SCALABLE face, natively at the output size
// with antialiasing, and hands ffmpeg full frames. Same 1960x1192, but the
// detail is real: strokes get grey edge pixels instead of stair-steps, and the
// chroma plane lands at 980 for a 980-column-pair image rather than 490.
//
// Frames go out as PNGs plus an ffconcat playlist, so the per-frame durations
// the capture recorded survive into the encode instead of being resampled by
// a fixed-rate GIF decode.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gomono"
	"golang.org/x/image/font/gofont/gomonobold"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/font/sfnt"
	"golang.org/x/image/math/fixed"
)

type glyphKey struct {
	r    rune
	bold bool
}

// aaFace rasterises Go Mono into per-cell alpha masks, one per (rune, weight),
// cached. The cache is what makes this affordable: the capture is ~800 frames
// of 120x32 cells, but it draws from an alphabet of well under 200 glyphs.
type aaFace struct {
	reg, bold    font.Face
	cw, ch, base int
	cache        map[glyphKey]*image.Alpha
}

// fitSize picks the largest point size whose advance still fits the cell.
//
// Hinting is full, which quantises the advance to whole pixels — that is the
// point. A fractional advance would put each column at a different subpixel
// phase and the capture is full of aligned hash tables and dot-leaders that
// have to stay in column.
func fitSize(f *sfnt.Font, cw int) (float64, error) {
	for s := float64(cw) * 2.5; s >= 4; s -= 0.05 {
		face, err := opentype.NewFace(f, &opentype.FaceOptions{Size: s, DPI: 72, Hinting: font.HintingFull})
		if err != nil {
			continue
		}
		adv, ok := face.GlyphAdvance('M')
		_ = face.Close()
		if ok && adv.Ceil() <= cw {
			return s, nil
		}
	}
	return 0, fmt.Errorf("no point size fits a %dpx cell", cw)
}

func newAAFace(cw, ch int) (*aaFace, error) {
	regSF, err := opentype.Parse(gomono.TTF)
	if err != nil {
		return nil, fmt.Errorf("parse gomono: %w", err)
	}
	boldSF, err := opentype.Parse(gomonobold.TTF)
	if err != nil {
		return nil, fmt.Errorf("parse gomonobold: %w", err)
	}
	size, err := fitSize(regSF, cw)
	if err != nil {
		return nil, err
	}
	opts := &opentype.FaceOptions{Size: size, DPI: 72, Hinting: font.HintingFull}
	reg, err := opentype.NewFace(regSF, opts)
	if err != nil {
		return nil, err
	}
	bold, err := opentype.NewFace(boldSF, opts)
	if err != nil {
		return nil, err
	}
	m := reg.Metrics()
	asc, desc := m.Ascent.Ceil(), m.Descent.Ceil()
	base := (ch-(asc+desc))/2 + asc
	if base < asc {
		base = asc
	}
	fmt.Fprintf(os.Stderr, "hires: Go Mono %.2fpx in a %dx%d cell, baseline %d (ascent %d descent %d)\n",
		size, cw, ch, base, asc, desc)
	return &aaFace{reg: reg, bold: bold, cw: cw, ch: ch, base: base,
		cache: map[glyphKey]*image.Alpha{}}, nil
}

// mask returns the coverage map for one glyph, drawn at the cell origin.
func (f *aaFace) mask(r rune, bold bool) *image.Alpha {
	k := glyphKey{r, bold}
	if m, ok := f.cache[k]; ok {
		return m
	}
	m := image.NewAlpha(image.Rect(0, 0, f.cw, f.ch))
	d := &font.Drawer{Dst: m, Src: image.NewUniform(color.Alpha{A: 0xff}), Face: f.reg}
	if bold {
		d.Face = f.bold
	}
	d.Dot = fixed.P(0, f.base)
	d.DrawString(string(r))
	f.cache[k] = m
	return m
}

// hiresSink renders full RGBA frames and writes them as a PNG sequence with an
// ffconcat playlist carrying the original per-frame durations.
type hiresSink struct {
	dir        string
	cols, rows int
	pad        int
	face       *aaFace
	w, h       int
	uni        []*image.Uniform
	prev       []cell
	prevVisual image.Image
	names      []string
	durs       []float64
}

func newHiresSink(dir string, cols, rows, pad, cw, ch int) (*hiresSink, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	// A stale frame from a previous run with a different config would be
	// picked up by the playlist and silently spliced into the video.
	old, _ := filepath.Glob(filepath.Join(dir, "frame-*.png"))
	for _, p := range old {
		_ = os.Remove(p)
	}
	face, err := newAAFace(cw, ch)
	if err != nil {
		return nil, err
	}
	u := make([]*image.Uniform, len(pal))
	for i, c := range pal {
		u[i] = image.NewUniform(c)
	}
	return &hiresSink{
		dir: dir, cols: cols, rows: rows, pad: pad, face: face,
		w: cols*cw + 2*pad, h: rows*ch + 2*pad, uni: u,
	}, nil
}

func (s *hiresSink) render(t *term) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, s.w, s.h))
	draw.Draw(img, img.Bounds(), s.uni[cBG], image.Point{}, draw.Src)
	if t.visual != nil {
		drawVisual(img, t.visual)
		return img
	}
	for y := 0; y < s.rows; y++ {
		row := t.cells[y*s.cols : (y+1)*s.cols]
		for x := 0; x < s.cols; x++ {
			c := row[x]
			px := s.pad + x*s.face.cw
			py := s.pad + y*s.face.ch
			// A bar is a filled rectangle on both paths. Nothing here consults
			// the face or its glyph cache, which is exactly why the mark is the
			// same shape in the bitmap GIF and in this antialiased video.
			if c.r == barFillRune {
				fg, _ := c.faded()
				ins := barInset(s.face.ch)
				draw.Draw(img, image.Rect(px, py+ins, px+s.face.cw, py+s.face.ch-ins),
					s.uni[fg], image.Point{}, draw.Src)
				continue
			}
			if c.r == ' ' || c.r == 0 {
				continue
			}
			fg, bold := c.faded()
			draw.DrawMask(img, image.Rect(px, py, px+s.face.cw, py+s.face.ch),
				s.uni[fg], image.Point{}, s.face.mask(c.r, bold), image.Point{}, draw.Over)
		}
	}
	return img
}

// emit writes one frame, or extends the previous frame's duration when the
// terminal state did not change. Comparing cells rather than pixels is both
// cheaper and exact: the cell grid fully determines the image.
func (s *hiresSink) emit(t *term, secs float64) error {
	if s.prev != nil && s.prevVisual == t.visual && cellsEqual(s.prev, t.cells) {
		s.durs[len(s.durs)-1] += secs
		return nil
	}
	name := fmt.Sprintf("frame-%05d.png", len(s.names))
	f, err := os.Create(filepath.Join(s.dir, name))
	if err != nil {
		return err
	}
	if err := png.Encode(f, s.render(t)); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	s.prev = append(s.prev[:0], t.cells...)
	s.prevVisual = t.visual
	s.names = append(s.names, name)
	s.durs = append(s.durs, secs)
	return nil
}

// hold extends the last frame, so a loop does not snap straight back.
func (s *hiresSink) hold(secs float64) {
	if len(s.durs) > 0 {
		s.durs[len(s.durs)-1] += secs
	}
}

// playlistFPS is the rate the playlist is quantised to. It must match the fps
// filter in the encode, and 15 is plenty: this is a terminal, not motion.
const playlistFPS = 15

// close writes the ffconcat playlist, one entry per output frame.
//
// The uniform rate is not laziness, it is the fix for a demuxer edge case that
// bit twice. The concat demuxer gives the FINAL entry the duration of the entry
// BEFORE it, ignoring the one it declares — measured on a two-image playlist
// (2 s then 5 s): 4 s out, not 7 s. So a variable-duration playlist silently
// loses (last - second-to-last) seconds; here that ate 4 s of the closing card
// and shipped an 83.3 s cut as 79.3 s. Naming the last file again does not help
// either: the repeat is itself an entry and inherits its predecessor's duration,
// which turned the same playlist into 95.3 s.
//
// Quantising every entry to one frame time makes that rule harmless — the final
// entry inherits 1/fps, so the worst case is one frame short. It also hands the
// encoder exactly the constant rate its fps filter would have produced anyway.
func (s *hiresSink) close() (string, error) {
	if len(s.names) == 0 {
		return "", fmt.Errorf("hires: no frames")
	}
	const tick = 1.0 / playlistFPS
	var b strings.Builder
	b.WriteString("ffconcat version 1.0\n")
	// Round against the running total, not against each duration on its own.
	// Per-frame rounding is biased upward here — minFrame is 0.12 s, which is
	// 1.8 ticks and always rounds to 2 — and over 268 frames that bias added
	// 3.2 s of drift. Charging each frame the difference between where the
	// capture says it should end and where the playlist actually is keeps the
	// total honest to within one tick.
	var want, total float64
	entries := 0
	for i, n := range s.names {
		want += s.durs[i]
		reps := int((want-total)*playlistFPS + 0.5)
		if reps < 1 {
			reps = 1
		}
		for r := 0; r < reps; r++ {
			fmt.Fprintf(&b, "file '%s'\nduration %.4f\n", n, tick)
		}
		total += float64(reps) * tick
		entries += reps
	}
	list := filepath.Join(s.dir, "frames.ffconcat")
	if err := os.WriteFile(list, []byte(b.String()), 0o644); err != nil {
		return "", err
	}
	fmt.Fprintf(os.Stderr, "hires: %d distinct frames, %d entries at %d fps, %dx%d, %.1fs -> %s\n",
		len(s.names), entries, playlistFPS, s.w, s.h, total, list)
	return list, nil
}

func cellsEqual(a, b []cell) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
