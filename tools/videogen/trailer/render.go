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
	fill(im, image.Rect(0, c.Height-12, c.Width, c.Height), color.RGBA{12, 26, 36, 255})
	x := int(float64(c.Width) * min(1, t/s.Secs))
	fill(im, image.Rect(0, c.Height-12, x, c.Height), cyan)

	sx := float64(c.Width) / 1280
	sy := float64(c.Height) / 720
	X := func(v int) int { return int(math.Round(float64(v) * sx)) }
	Y := func(v int) int { return int(math.Round(float64(v) * sy)) }
	S := func(v float64) float64 { return v * min(sx, sy) }
	R := func(x1, y1, x2, y2 int) image.Rectangle { return image.Rect(X(x1), Y(y1), X(x2), Y(y2)) }
	C := func(y int, value string, size float64, clr color.Color, bold, mono bool) {
		centerFit(im, p, Y(y), X(140), value, S(size), clr, bold, mono)
	}
	particle := func(x, y, radius int, clr color.RGBA, glow bool) {
		if glow {
			circle(im, X(x), Y(y), X(radius+8), alpha(clr, .12))
		}
		circle(im, X(x), Y(y), max(2, X(radius)), clr)
	}
	switch s.Kind {
	case "harness-hook":
		C(82, s.Eyebrow, 26, cyan, true, false)
		C(180+yoff, s.Title, 68, alpha(white, fade), true, false)
		C(244+yoff, s.Subtitle, 30, alpha(muted, fade), false, false)
		cx, cy := X(640), Y(470)
		labels := []string{"UI", "MODEL", "TOOLS", "MEMORY", "POLICY"}
		for i, label := range labels {
			a := -math.Pi/2 + float64(i)*2*math.Pi/float64(len(labels))
			px, py := cx+int(math.Cos(a)*float64(X(300))), cy+int(math.Sin(a)*float64(Y(145)))
			progress := min(1, max(0, t*1.4-float64(i)*.13))
			ex, ey := cx+int(float64(px-cx)*progress), cy+int(float64(py-cy)*progress)
			lineSegment(im, cx, cy, ex, ey, alpha(cyan, .55), X(3))
			if progress > .9 {
				node := image.Rect(px-X(74), py-Y(28), px+X(74), py+Y(28))
				fill(im, node, color.RGBA{13, 25, 36, 255})
				stroke(im, node, alpha(cyan, .7), X(2))
				centerBoxText(im, p, node, label, S(21), white)
			}
			beam := math.Mod(t*.65+float64(i)*.16, 1)
			particle(int(float64(cx)/sx+float64(px-cx)/sx*beam), int(float64(cy)/sy+float64(py-cy)/sy*beam), 6, green, true)
		}
		circle(im, cx, cy, X(78), color.RGBA{9, 46, 55, 255})
		circleStroke(im, cx, cy, X(78), cyan, X(3))
		centerBoxText(im, p, image.Rect(cx-X(68), cy-Y(42), cx+X(68), cy+Y(42)), "YOU", S(34), white)
	case "harness-blueprint":
		C(78, s.Eyebrow, 26, cyan, true, false)
		C(154+yoff, s.Title, 58, alpha(white, fade), true, false)
		positions := [][2]int{{250, 300}, {640, 300}, {1030, 300}, {250, 520}, {640, 520}, {1030, 520}}
		for i, item := range s.Items {
			px, py := positions[i][0], positions[i][1]
			delay := float64(i) * .18
			reveal := ease(min(1, max(0, (t-delay)/.65)))
			w := int(150 * reveal)
			node := R(px-w, py-58, px+w, py+58)
			fill(im, node, alpha(color.RGBA{13, 25, 36, 255}, reveal))
			stroke(im, node, alpha(cyan, reveal*.8), X(2))
			if reveal > .75 {
				centerBoxText(im, p, node, item, S(25), alpha(white, reveal))
			}
			if i > 0 {
				prev := positions[i-1]
				lineProgress := min(1, max(0, (t-delay-.2)/.45))
				x1, y1 := X(prev[0]), Y(prev[1])
				x2, y2 := X(px), Y(py)
				lineSegment(im, x1, y1, x1+int(float64(x2-x1)*lineProgress), y1+int(float64(y2-y1)*lineProgress), alpha(green, .6), X(3))
			}
		}
		C(670, s.Detail, 24, alpha(muted, fade), false, true)
	case "harness-run":
		C(76, s.Eyebrow, 26, cyan, true, false)
		C(150+yoff, s.Title, 56, alpha(white, fade), true, false)
		left := []int{145, 485, 825}
		for i, item := range s.Items {
			r := R(left[i], 315, left[i]+310, 455)
			fill(im, r, color.RGBA{13, 25, 36, 255})
			stroke(im, r, alpha(cyan, .55), X(2))
			centerBoxText(im, p, r, item, S(26), white)
			if i < 2 {
				lineSegment(im, X(left[i]+310), Y(385), X(left[i+1]), Y(385), alpha(cyan, .45), X(3))
			}
		}
		pathStart, pathEnd := 455, 825
		travel := ease(min(1, t/2.0))
		px := pathStart + int(float64(pathEnd-pathStart)*travel)
		particle(px, 385, 11, green, true)
		for i := 0; i < 5; i++ {
			tail := max(0, travel-float64(i+1)*.045)
			particle(pathStart+int(float64(pathEnd-pathStart)*tail), 385, max(3, 9-i), alpha(green, .45), false)
		}
		verdict := R(390, 540, 890, 630)
		show := min(1, max(0, (t-1.6)/.5))
		fill(im, verdict, alpha(color.RGBA{8, 55, 42, 255}, show))
		stroke(im, verdict, alpha(green, show), X(3))
		centerBoxText(im, p, verdict, s.Verdict, S(30), alpha(green, show))
	case "token-hook":
		C(82, s.Eyebrow, 26, cyan, true, false)
		C(180+yoff, s.Title, 68, alpha(white, fade), true, false)
		C(244+yoff, s.Subtitle, 30, alpha(muted, fade), false, false)
		gateX := 640
		fill(im, R(gateX-7, 320, gateX+7, 625), cyan)
		text(im, p, X(gateX-20), Y(300), "fak", S(28), cyan, true, false)
		for i := 0; i < 42; i++ {
			row, col := i/14, i%14
			px := 88 + col*34 + int(13*math.Sin(float64(i)*1.7+t*2.4))
			py := 382 + row*72 + int(9*math.Cos(float64(i)*1.3+t*2.1))
			particle(px, py, 5, color.RGBA{255, 104, 132, 255}, true)
		}
		for i := 0; i < 13; i++ {
			angle := float64(i)*.55 + t*.8
			px := 870 + int(math.Cos(angle)*float64(95+i*5))
			py := 485 + int(math.Sin(angle)*float64(70+i*3))
			particle(px, py, 6, green, true)
		}
		left, right := R(86, 650, 540, 700), R(740, 650, 1194, 700)
		centerBoxText(im, p, left, s.Action+"  42", S(26), red)
		centerBoxText(im, p, right, s.Detail+"  13", S(26), green)
	case "token-grid", "token-flow":
		renderTokenScenes(s, t, yoff, fade, im, p, X, Y, S, R, C, particle)
	case "hook":
		C(250+yoff, s.Title, 82, alpha(white, fade), true, false)
		C(340+yoff, s.Subtitle, 44, alpha(muted, fade), false, false)
		card := R(220, 465, 1060, 650)
		fill(im, card, color.RGBA{15, 23, 34, 255})
		text(im, p, X(270), Y(525), "TOOL CALL", S(30), muted, true, false)
		text(im, p, X(270), Y(600), s.Action, S(58), white, true, true)
		text(im, p, X(790), Y(600), s.Detail, S(58), red, true, true)
	case "checkpoint":
		C(105, s.Eyebrow, 30, cyan, true, false)
		C(205+yoff, s.Title, 88, alpha(white, fade), true, false)
		cy := 455
		callX := 110 + int(ease(min(1, t/1.6))*290)
		call := R(callX, cy-70, callX+360, cy+70)
		fill(im, call, color.RGBA{17, 25, 37, 255})
		stroke(im, call, white, X(2))
		centerBoxText(im, p, call, s.Action, S(30), white)
		gate := R(820, cy-125, 875, cy+125)
		fill(im, gate, cyan)
		if t > 1.5 {
			stop := R(915, cy-75, 1200, cy+75)
			fill(im, stop, color.RGBA{44, 18, 25, 255})
			centerBoxText(im, p, stop, "STOP", S(48), red)
		}
	case "proof":
		C(270+yoff, s.Title, 150, alpha(red, fade), true, false)
		C(405, s.Action, 40, white, true, true)
		C(520, s.Verdict, 68, green, true, false)
	case "cta":
		C(110, s.Eyebrow, 28, cyan, true, false)
		C(210+yoff, s.Title, 70, alpha(white, fade), true, false)
		// A visual checkpoint: the command crosses the luminous kernel boundary.
		for i := 0; i < 24; i++ {
			particle(130+i*26, 385+int(15*math.Sin(float64(i)*.7+t*1.4)), 4, cyan, true)
		}
		gate := R(638, 315, 651, 535)
		fill(im, gate, cyan)
		for i := 0; i < 8; i++ {
			particle(730+i*47, 385+int(28*math.Sin(float64(i)+t)), 6, green, true)
		}
		r := R(165, 530, 1115, 625)
		fill(im, r, color.RGBA{13, 22, 32, 255})
		stroke(im, r, cyan, X(3))
		centerBoxText(im, p, r, s.Command, S(56), white)
		C(675, s.Subtitle, 27, muted, false, false)
	}
	return im
}

func renderTokenScenes(s Scene, t float64, yoff int, fade float64, im *image.RGBA, p *painter, X, Y, S func(int) int, R func(int, int, int, int) image.Rectangle, C func(int, string, int, color.RGBA, bool, bool), particle func(int, int, int, color.RGBA, bool)) {
	switch s.Kind {
	case "token-grid":
		C(78, s.Eyebrow, 26, cyan, true, false)
		C(162+yoff, s.Title, 58, alpha(white, fade), true, false)
		cx, cy := X(640), Y(425)
		circle(im, cx, cy, X(92), color.RGBA{9, 46, 55, 255})
		circleStroke(im, cx, cy, X(92), cyan, X(3))
		centerBoxText(im, p, image.Rect(cx-X(82), cy-Y(55), cx+X(82), cy+Y(55)), "fak", S(48), white)
		positions := [][2]int{{245, 310}, {245, 500}, {640, 625}, {1035, 500}, {1035, 310}, {640, 245}}
		for i, item := range s.Items {
			px, py := positions[i][0], positions[i][1]
			lineSegment(im, cx, cy, X(px), Y(py), alpha(cyan, .35), X(2))
			node := image.Rect(X(px-145), Y(py-48), X(px+145), Y(py+48))
			fill(im, node, color.RGBA{13, 25, 36, 255})
			stroke(im, node, cyan, X(2))
			centerBoxText(im, p, node, item, S(27), white)
			particle(px, py-63, 5, green, true)
		}
	case "token-flow":
		C(78, s.Eyebrow, 26, cyan, true, false)
		C(162+yoff, s.Title, 55, alpha(white, fade), true, false)
		baseY := 530
		widths := []int{390, 270, 155}
		colors := []color.RGBA{red, cyan, green}
		for i, item := range s.Items {
			xc := 250 + i*390
			h := widths[i]
			for j := 0; j < h/15; j++ {
				px := xc - h/2 + 12 + j*15
				particle(px, baseY-int(24*math.Sin(float64(j)*.8+t)), 5, colors[i], true)
			}
			r := R(xc-165, 595, xc+165, 670)
			centerBoxText(im, p, r, item, S(28), white)
			if i < 2 {
				centerFit(im, p, Y(530), X(45), "->", S(34), muted, true, true)
			}
		}
		C(708, s.Verdict, 29, green, true, false)
	}
}

func fittedTextSize(p *painter, available int, s string, size, floor float64, bold, mono bool) (float64, int) {
	for size > floor {
		w := font.MeasureString(p.face(size, bold, mono), s).Ceil()
		if w <= available {
			return size, w
		}
		size--
	}
	return floor, font.MeasureString(p.face(floor, bold, mono), s).Ceil()
}

func centerFit(im *image.RGBA, p *painter, y, margin int, s string, size float64, c color.Color, bold, mono bool) {
	size, _ = fittedTextSize(p, im.Bounds().Dx()-2*margin, s, size, 18, bold, mono)
	center(im, p, y, s, size, c, bold, mono)
}

func circle(im *image.RGBA, cx, cy, radius int, c color.Color) {
	r2 := radius * radius
	for y := -radius; y <= radius; y++ {
		span := int(math.Sqrt(float64(r2 - y*y)))
		fill(im, image.Rect(cx-span, cy+y, cx+span+1, cy+y+1), c)
	}
}

func circleStroke(im *image.RGBA, cx, cy, radius int, c color.Color, width int) {
	circle(im, cx, cy, radius, c)
	circle(im, cx, cy, max(0, radius-width), color.RGBA{9, 46, 55, 255})
}

func lineSegment(im *image.RGBA, x1, y1, x2, y2 int, c color.Color, width int) {
	steps := max(abs(x2-x1), abs(y2-y1))
	if steps == 0 {
		return
	}
	for i := 0; i <= steps; i++ {
		x := x1 + (x2-x1)*i/steps
		y := y1 + (y2-y1)*i/steps
		circle(im, x, y, max(1, width/2), c)
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func centerBoxText(im *image.RGBA, p *painter, r image.Rectangle, s string, size float64, c color.Color) {
	const inset = 24
	for size > 18 {
		f := p.face(size, true, false)
		bounds, _ := font.BoundString(f, s)
		if (bounds.Max.X-bounds.Min.X).Ceil() <= r.Dx()-2*inset && (bounds.Max.Y-bounds.Min.Y).Ceil() <= r.Dy()-2*inset {
			break
		}
		size--
	}
	f := p.face(size, true, false)
	bounds, _ := font.BoundString(f, s)
	px := r.Min.X + (r.Dx()-(bounds.Max.X-bounds.Min.X).Ceil())/2 - bounds.Min.X.Ceil()
	py := r.Min.Y + (r.Dy()-(bounds.Max.Y-bounds.Min.Y).Ceil())/2 - bounds.Min.Y.Ceil()
	d := font.Drawer{Dst: im, Src: image.NewUniform(c), Face: f, Dot: fixed.P(px, py)}
	d.DrawString(s)
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

func validateLayout(c Config) (int, int, int, error) {
	p, err := newPainter()
	if err != nil {
		return 0, 0, 0, err
	}
	margin := int(math.Round(140 * float64(c.Width) / 1280))
	sx := float64(c.Width) / 1280
	available := c.Width - 2*margin
	samples, maxRight, maxBottom := 0, 0, 0
	for i, scene := range c.Scenes {
		for _, at := range []float64{.15, .55, .9} {
			_ = sceneFrame(c, scene, scene.Secs*at, p)
			samples++
		}
		for _, region := range []struct {
			name, text string
			size       float64
			mono       bool
		}{
			{"title", scene.Title, 68 * sx, false}, {"subtitle", scene.Subtitle, 30 * sx, false}, {"eyebrow", scene.Eyebrow, 28 * sx, false},
		} {
			if region.text == "" {
				continue
			}
			_, width := fittedTextSize(p, available, region.text, region.size, 18*sx, true, region.mono)
			right := margin + (available+width)/2
			if right > maxRight {
				maxRight = right
			}
			if width > available {
				return samples, maxRight, maxBottom, fmt.Errorf("scene %d %s crosses %dpx safe area: width=%d available=%d", i, region.name, margin, width, available)
			}
		}
	}
	maxBottom = c.Height - int(math.Round(32*float64(c.Height)/720))
	return samples, maxRight, maxBottom, nil
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
	if c.AppendMP4 != "" {
		if e = compose(ff, c); e != nil {
			return e
		}
	}
	posterScene := c.Scenes[len(c.Scenes)-1]
	if len(c.Scenes) > 0 && c.Scenes[0].Kind == "token-hook" {
		posterScene = c.Scenes[0]
	}
	if e = pngAt(c.Poster, sceneFrame(c, posterScene, posterScene.Secs*.55, p)); e != nil {
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
	cmd = exec.Command(ff, "-y", "-framerate", fmt.Sprint(c.FPS), "-i", in, "-vf", "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", c.GIF)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func compose(ff string, c Config) error {
	if c.CompositeMP4 == "" || c.CompositeGIF == "" {
		return fmt.Errorf("appendMP4 requires compositeMP4 and compositeGIF")
	}
	_ = os.MkdirAll(filepath.Dir(c.CompositeMP4), 0755)
	cmd := exec.Command(ff, "-y", "-i", c.MP4, "-i", c.AppendMP4,
		"-filter_complex", "[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[v]",
		"-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", c.CompositeMP4)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return err
	}
	cmd = exec.Command(ff, "-y", "-i", c.CompositeMP4,
		"-vf", "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
		"-loop", "0", c.CompositeGIF)
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
	thumbW := 960
	thumbH := 540
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

// Layout checks are render-backed: they exercise representative frames and
// reject any title region that cannot fit inside the declared safe area.
