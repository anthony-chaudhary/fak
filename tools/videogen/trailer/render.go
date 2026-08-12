package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gobold"
	"golang.org/x/image/font/gofont/gomono"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

var bg = color.RGBA{8, 12, 18, 255}
var white = color.RGBA{242, 246, 250, 255}
var muted = color.RGBA{148, 163, 184, 255}
var cyan = color.RGBA{39, 214, 255, 255}
var green = color.RGBA{64, 232, 151, 255}
var red = color.RGBA{255, 84, 104, 255}

type painter struct {
	reg, bold, mono *opentype.Font
	faces           map[string]font.Face
}

func newPainter() (*painter, error) {
	r, err := opentype.Parse(goregular.TTF)
	if err != nil {
		return nil, err
	}
	b, err := opentype.Parse(gobold.TTF)
	if err != nil {
		return nil, err
	}
	m, err := opentype.Parse(gomono.TTF)
	if err != nil {
		return nil, err
	}
	return &painter{r, b, m, map[string]font.Face{}}, nil
}
func (p *painter) face(size float64, bold, mono bool) font.Face {
	k := fmt.Sprintf("%.1f/%v/%v", size, bold, mono)
	if f := p.faces[k]; f != nil {
		return f
	}
	ft := p.reg
	if bold {
		ft = p.bold
	}
	if mono {
		ft = p.mono
	}
	f, _ := opentype.NewFace(ft, &opentype.FaceOptions{Size: size, DPI: 96, Hinting: font.HintingFull})
	p.faces[k] = f
	return f
}
func fill(im *image.RGBA, r image.Rectangle, c color.Color) {
	draw.Draw(im, r, image.NewUniform(c), image.Point{}, draw.Src)
}
func text(im *image.RGBA, p *painter, x, y int, s string, size float64, c color.Color, bold, mono bool) {
	d := font.Drawer{Dst: im, Src: image.NewUniform(c), Face: p.face(size, bold, mono), Dot: fixed.P(x, y)}
	d.DrawString(s)
}
func center(im *image.RGBA, p *painter, y int, s string, size float64, c color.Color, bold, mono bool) {
	f := p.face(size, bold, mono)
	w := font.MeasureString(f, s).Ceil()
	text(im, p, (im.Bounds().Dx()-w)/2, y, s, size, c, bold, mono)
}
func ease(t float64) float64                   { return 1 - math.Pow(1-t, 3) }
func alpha(c color.RGBA, a float64) color.RGBA { c.A = uint8(255 * max(0, min(1, a))); return c }
func sceneFrame(c Config, s Scene, t float64, p *painter) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, c.Width, c.Height))
	fill(im, im.Bounds(), bg)
	fade := min(1, t/.35) * min(1, (s.Secs-t)/.3)
	yoff := int((1 - ease(min(1, t/.7))) * 34)
	// quiet cinematic horizon and moving checkpoint line
	fill(im, image.Rect(0, c.Height-8, c.Width, c.Height), color.RGBA{12, 26, 36, 255})
	x := int(float64(c.Width) * min(1, t/s.Secs))
	fill(im, image.Rect(0, c.Height-8, x, c.Height), cyan)
	switch s.Kind {
	case "hook":
		center(im, p, 250+yoff, s.Title, 82, alpha(white, fade), true, false)
		center(im, p, 340+yoff, s.Subtitle, 44, alpha(muted, fade), false, false)
		// one giant dangerous call moving toward execution
		card := image.Rect(220, 465, c.Width-220, 650)
		fill(im, card, color.RGBA{15, 23, 34, 255})
		text(im, p, 270, 525, "TOOL CALL", 30, muted, true, false)
		text(im, p, 270, 600, s.Action, 58, white, true, true)
		text(im, p, 790, 600, s.Detail, 58, red, true, true)
	case "checkpoint":
		center(im, p, 105, s.Eyebrow, 30, cyan, true, false)
		center(im, p, 205+yoff, s.Title, 88, alpha(white, fade), true, false)
		// One continuous object: the dangerous call travels toward execution and
		// meets the checkpoint. The red stop bar lands late in the beat.
		cy := 455
		callX := 110 + int(ease(min(1, t/1.6))*290)
		fill(im, image.Rect(callX, cy-70, callX+360, cy+70), color.RGBA{17, 25, 37, 255})
		stroke(im, image.Rect(callX, cy-70, callX+360, cy+70), white, 2)
		centerBoxText(im, p, image.Rect(callX, cy-70, callX+360, cy+70), s.Action, 30, white)
		gate := image.Rect(820, cy-125, 875, cy+125)
		fill(im, gate, cyan)
		text(im, p, 804, cy-155, "fak", 34, cyan, true, false)
		if t > 1.5 {
			fill(im, image.Rect(915, cy-75, 1200, cy+75), color.RGBA{44, 18, 25, 255})
			centerBoxText(im, p, image.Rect(915, cy-75, 1200, cy+75), "STOP", 48, red)
			line(im, 875, cy, 915, cy, red, 8)
		}
	case "proof":
		center(im, p, 270+yoff, s.Title, 150, alpha(red, fade), true, false)
		center(im, p, 405, s.Action, 40, white, true, true)
		center(im, p, 520, s.Verdict, 68, green, true, false)
	case "cta":
		center(im, p, 145, s.Eyebrow, 32, cyan, true, false)
		center(im, p, 265+yoff, s.Title, 82, alpha(white, fade), true, false)
		r := image.Rect(165, 390, c.Width-165, 555)
		fill(im, r, color.RGBA{13, 22, 32, 255})
		stroke(im, r, cyan, 3)
		center(im, p, 498, s.Command, 84, white, true, true)
		center(im, p, 650, s.Subtitle, 38, muted, false, false)
	}
	return im
}
func centerBoxText(im *image.RGBA, p *painter, r image.Rectangle, s string, size float64, c color.Color) {
	f := p.face(size, true, false)
	w := font.MeasureString(f, s).Ceil()
	text(im, p, r.Min.X+(r.Dx()-w)/2, r.Min.Y+r.Dy()/2+int(size/3), s, size, c, true, false)
}
func stroke(im *image.RGBA, r image.Rectangle, c color.Color, w int) {
	fill(im, image.Rect(r.Min.X, r.Min.Y, r.Max.X, r.Min.Y+w), c)
	fill(im, image.Rect(r.Min.X, r.Max.Y-w, r.Max.X, r.Max.Y), c)
	fill(im, image.Rect(r.Min.X, r.Min.Y, r.Min.X+w, r.Max.Y), c)
	fill(im, image.Rect(r.Max.X-w, r.Min.Y, r.Max.X, r.Max.Y), c)
}
func line(im *image.RGBA, x1, y1, x2, y2 int, c color.Color, w int) {
	fill(im, image.Rect(x1, y1-w/2, x2, y2+w/2), c)
}

func renderAll(c Config, ff string, a Audit) error {
	p, e := newPainter()
	if e != nil {
		return e
	}
	dir := filepath.Join(filepath.Dir(c.MP4), "frames")
	os.RemoveAll(dir)
	if e = os.MkdirAll(dir, 0755); e != nil {
		return e
	}
	frame := 0
	var sceneStarts []int
	for _, s := range c.Scenes {
		sceneStarts = append(sceneStarts, frame)
		n := int(math.Round(s.Secs * float64(c.FPS)))
		for i := 0; i < n; i++ {
			im := sceneFrame(c, s, float64(i)/float64(c.FPS), p)
			f, e := os.Create(filepath.Join(dir, fmt.Sprintf("frame-%05d.png", frame)))
			if e != nil {
				return e
			}
			e = png.Encode(f, im)
			f.Close()
			if e != nil {
				return e
			}
			frame++
		}
	}
	if e = encode(ff, c, dir); e != nil {
		return e
	}
	if e = pngAt(c.Poster, sceneFrame(c, c.Scenes[len(c.Scenes)-1], 1, p)); e != nil {
		return e
	}
	if e = contact(c.ContactSheet, c, p); e != nil {
		return e
	}
	writeAudit(c.Audit, a)
	fmt.Printf("trailer: wrote %.1fs, %d frames, %dx%d @ %dfps\n", a.Duration, frame, c.Width, c.Height, c.FPS)
	return nil
}
func encode(ff string, c Config, dir string) error {
	_ = os.MkdirAll(filepath.Dir(c.MP4), 0755)
	in := filepath.Join(dir, "frame-%05d.png")
	cmd := exec.Command(ff, "-y", "-framerate", fmt.Sprint(c.FPS), "-i", in, "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", c.MP4)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if e := cmd.Run(); e != nil {
		return e
	}
	cmd = exec.Command(ff, "-y", "-framerate", fmt.Sprint(c.FPS), "-i", in, "-vf", "fps=15,scale=680:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", c.GIF)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
func pngAt(path string, im image.Image) error {
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	f, e := os.Create(path)
	if e != nil {
		return e
	}
	defer f.Close()
	return png.Encode(f, im)
}
func contact(path string, c Config, p *painter) error {
	thumbW := 480
	thumbH := 270
	sheet := image.NewRGBA(image.Rect(0, 0, thumbW*2, thumbH*((len(c.Scenes)+1)/2)))
	fill(sheet, sheet.Bounds(), bg)
	for i, s := range c.Scenes {
		src := sceneFrame(c, s, s.Secs*.55, p)
		dst := image.Rect((i%2)*thumbW, (i/2)*thumbH, (i%2+1)*thumbW, (i/2+1)*thumbH)
		xdraw.ApproxBiLinear.Scale(sheet, dst, src, src.Bounds(), draw.Src, nil)
	}
	return pngAt(path, sheet)
}

var _ = strings.TrimSpace
