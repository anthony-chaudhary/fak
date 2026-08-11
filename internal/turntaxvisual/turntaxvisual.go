// Package turntaxvisual renders the checked-in turn-tax efficiency visual from
// its JSON source of truth.
package turntaxvisual

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	DefaultData = "tools/hero_turntax.data.json"
	DefaultOut  = "visuals/60-hero-turntax-curves.svg"

	Base     = "#3b4a7a"
	BaseFill = "#e7ebf4"
	FAK      = "#2e8b57"
	FAKFill  = "#e9f5ee"
	Grid     = "#e3e9ee"
	Axis     = "#b9c4cf"

	width        = 1320
	marginLeft   = 64
	marginRight  = 64
	panelGap     = 40
	panelCount   = 3
	panelWidth   = (width - marginLeft - marginRight - (panelCount-1)*panelGap) / panelCount
	plotPadLeft  = 40
	plotPadRight = 10
	panelsTop    = 196
	plotHeight   = 250
	titleGap     = 56
	height       = panelsTop + titleGap + plotHeight + 120
)

const style = `  <defs>
    <linearGradient id="tt-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#fbfcfd"/>
      <stop offset="60%" stop-color="#f4f7f9"/>
      <stop offset="100%" stop-color="#f6f4f0"/>
    </linearGradient>
    <style>
      .k    { font: 800 14px "Segoe UI", Arial, sans-serif; fill: #2e8b57; letter-spacing: 1.6px; }
      .ti   { font: 900 33px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .sub  { font: 400 16px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .pt   { font: 800 18px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .pst  { font: 400 12.5px "Segoe UI", Arial, sans-serif; fill: #5b6a7a; }
      .axt  { font: 600 12px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .axl  { font: 700 12.5px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .ytk  { font: 400 11px "Segoe UI", Arial, sans-serif; fill: #8090a0; }
      .mult { font: 900 50px "Segoe UI", Arial, sans-serif; fill: #14202b; }
      .msub { font: 700 13px "Segoe UI", Arial, sans-serif; fill: #415266; }
      .mfoot{ font: 400 11.5px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
      .leg  { font: 600 14px "Segoe UI", Arial, sans-serif; fill: #1b2733; }
      .foot { font: 400 12px "Segoe UI", Arial, sans-serif; fill: #6a7685; }
    </style>
  </defs>`

type Data struct {
	Meta struct {
		Title  string `json:"title"`
		OutSVG string `json:"out_svg"`
	} `json:"meta"`
	Kicker   string   `json:"kicker"`
	Headline string   `json:"headline"`
	Subhead  string   `json:"subhead"`
	Panels   []Panel  `json:"panels"`
	Legend   []Legend `json:"legend"`
	Footer   string   `json:"footer"`
}

type Panel struct {
	Title     string    `json:"title"`
	Subtitle  string    `json:"subtitle"`
	XLabel    string    `json:"x_label"`
	YLabel    string    `json:"y_label"`
	XVals     []float64 `json:"x_vals"`
	XTicks    []string  `json:"x_ticks"`
	YMax      float64   `json:"y_max"`
	YTicks    []Tick    `json:"y_ticks"`
	Baseline  []float64 `json:"baseline"`
	FAK       []float64 `json:"fak"`
	AnchorIdx *int      `json:"anchor_idx"`
	Mult      string    `json:"mult"`
	MultSub   string    `json:"mult_sub"`
	MultFoot  string    `json:"mult_foot"`
}

type Tick struct {
	V float64 `json:"v"`
	T string  `json:"t"`
}

type Legend struct {
	Label string `json:"label"`
	Color string `json:"color"`
}

func Load(path string) (Data, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Data{}, err
	}
	var d Data
	if err := json.Unmarshal(b, &d); err != nil {
		return Data{}, err
	}
	if len(d.Panels) != panelCount {
		return Data{}, fmt.Errorf("turn-tax visual: want %d panels, got %d", panelCount, len(d.Panels))
	}
	for i, p := range d.Panels {
		if len(p.XVals) < 2 || len(p.XTicks) != len(p.XVals) || len(p.Baseline) != len(p.XVals) || len(p.FAK) != len(p.XVals) || p.YMax <= 0 || p.XVals[len(p.XVals)-1] == p.XVals[0] {
			return Data{}, fmt.Errorf("turn-tax visual: panel %d has inconsistent curve data", i+1)
		}
	}
	return d, nil
}

func esc(v any) string {
	s := fmt.Sprint(v)
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	return strings.ReplaceAll(s, ">", "&gt;")
}

func panelX(i int) int    { return marginLeft + i*(panelWidth+panelGap) }
func f1(v float64) string { return strconv.FormatFloat(v, 'f', 1, 64) }

func Render(d Data) ([]byte, error) {
	if len(d.Panels) != panelCount {
		return nil, fmt.Errorf("turn-tax visual: want %d panels, got %d", panelCount, len(d.Panels))
	}
	var out []string
	out = append(out, fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img" aria-labelledby="tt-title tt-desc">`, width, height, width, height))
	out = append(out, fmt.Sprintf(`  <title id="tt-title">%s</title>`, esc(d.Meta.Title)))
	mults := make([]string, len(d.Panels))
	for i := range d.Panels {
		mults[i] = d.Panels[i].Mult
	}
	out = append(out, fmt.Sprintf(`  <desc id="tt-desc">Three panels — per-turn prefill cost vs context, WebVoyager fleet prefill vs workers, and 50-turn fleet serving work vs turns. In each, a baseline re-prefill curve rises ~linearly while fak's resident-KV curve stays flat; multipliers %s. Panel 2 is modeled geometry vs the naive floor (643 WebVoyager tasks); panel 3 is the conservative number vs a tuned SOTA cache.</desc>`, esc(strings.Join(mults, " / "))))
	out = append(out, style)
	out = append(out, fmt.Sprintf(`  <rect width="%d" height="%d" fill="url(#tt-bg)"/>`, width, height))
	out = append(out, fmt.Sprintf(`  <text x="%d" y="58" class="k">%s</text>`, marginLeft, esc(d.Kicker)))
	out = append(out, fmt.Sprintf(`  <text x="%d" y="96" class="ti">%s</text>`, marginLeft, esc(d.Headline)))
	out = append(out, fmt.Sprintf(`  <text x="%d" y="124" class="sub">%s</text>`, marginLeft, esc(d.Subhead)))
	for i, p := range d.Panels {
		panelSVG(&out, i, p)
	}
	legY := panelsTop + titleGap + plotHeight + 64
	seg := 430
	start := float64(width-seg*len(d.Legend))/2 + 40
	for j, it := range d.Legend {
		lx := start + float64(j*seg)
		out = append(out, fmt.Sprintf(`  <line x1="%.0f" y1="%d" x2="%.0f" y2="%d" stroke="%s" stroke-width="4" stroke-linecap="round"/>`, lx, legY-5, lx+34, legY-5, it.Color))
		out = append(out, fmt.Sprintf(`  <circle cx="%.0f" cy="%d" r="4.5" fill="%s"/>`, lx+17, legY-5, it.Color))
		out = append(out, fmt.Sprintf(`  <text x="%.0f" y="%d" class="leg">%s</text>`, lx+46, legY, esc(it.Label)))
	}
	out = append(out, fmt.Sprintf(`  <text x="%d" y="%d" class="foot">%s</text>`, marginLeft, height-20, esc(d.Footer)))
	out = append(out, "</svg>\n")
	return []byte(strings.Join(out, "\n")), nil
}

type point struct{ x, y float64 }

func panelSVG(out *[]string, i int, p Panel) {
	px := panelX(i)
	x0 := px + plotPadLeft
	x1 := px + panelWidth - plotPadRight
	y0 := panelsTop + titleGap
	y1 := y0 + plotHeight
	plotW := x1 - x0
	xlo, xhi := p.XVals[0], p.XVals[len(p.XVals)-1]
	x := func(v float64) float64 { return float64(x0) + (v-xlo)/(xhi-xlo)*float64(plotW) }
	y := func(v float64) float64 { return float64(y1) - (v/p.YMax)*plotHeight }
	*out = append(*out, fmt.Sprintf(`  <text x="%d" y="%d" class="pt">%s</text>`, px, panelsTop+14, esc(p.Title)))
	*out = append(*out, fmt.Sprintf(`  <text x="%d" y="%d" class="pst">%s</text>`, px, panelsTop+34, esc(p.Subtitle)))
	for _, tk := range p.YTicks {
		gy := y(tk.V)
		*out = append(*out, fmt.Sprintf(`  <line x1="%d" y1="%s" x2="%d" y2="%s" stroke="%s" stroke-width="1"/>`, x0, f1(gy), x1, f1(gy), Grid), fmt.Sprintf(`  <text x="%d" y="%s" class="ytk" text-anchor="end">%s</text>`, x0-6, f1(gy+4), esc(tk.T)))
	}
	*out = append(*out, fmt.Sprintf(`  <line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.5"/>`, x0, y0, x0, y1, Axis), fmt.Sprintf(`  <line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="1.5"/>`, x0, y1, x1, y1, Axis))
	base := make([]point, len(p.XVals))
	fak := make([]point, len(p.XVals))
	for j, v := range p.XVals {
		base[j] = point{x(v), y(p.Baseline[j])}
		fak[j] = point{x(v), y(p.FAK[j])}
	}
	pts := points(base)
	rev := make([]point, len(fak))
	for j := range fak {
		rev[j] = fak[len(fak)-1-j]
	}
	*out = append(*out, fmt.Sprintf(`  <polygon points="%s %s" fill="%s" opacity="0.65"/>`, pts, points(rev), BaseFill))
	*out = append(*out, fmt.Sprintf(`  <polygon points="%s %s,%s %s,%s" fill="%s" opacity="0.7"/>`, points(fak), f1(fak[len(fak)-1].x), f1(float64(y1)), f1(fak[0].x), f1(float64(y1)), FAKFill))
	*out = append(*out, fmt.Sprintf(`  <polyline points="%s" fill="none" stroke="%s" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>`, points(base), Base), fmt.Sprintf(`  <polyline points="%s" fill="none" stroke="%s" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>`, points(fak), FAK))
	ai := len(p.XVals) - 1
	if p.AnchorIdx != nil {
		ai = *p.AnchorIdx
	}
	for _, q := range []struct {
		ps []point
		c  string
	}{{base, Base}, {fak, FAK}} {
		*out = append(*out, fmt.Sprintf(`  <circle cx="%s" cy="%s" r="5" fill="%s"/>`, f1(q.ps[ai].x), f1(q.ps[ai].y), q.c))
	}
	for j, t := range p.XTicks {
		*out = append(*out, fmt.Sprintf(`  <text x="%s" y="%d" class="axt" text-anchor="middle">%s</text>`, f1(x(p.XVals[j])), y1+18, esc(t)))
	}
	*out = append(*out, fmt.Sprintf(`  <text x="%s" y="%d" class="axl" text-anchor="middle">%s</text>`, f1(float64(x0+x1)/2), y1+38, esc(p.XLabel)))
	ylx := px + 12
	yly := float64(y0+y1) / 2
	*out = append(*out, fmt.Sprintf(`  <text x="%d" y="%s" class="axl" text-anchor="middle" transform="rotate(-90 %d %s)">%s</text>`, ylx, f1(yly), ylx, f1(yly), esc(p.YLabel)))
	mx := float64(x0) + float64(plotW)*.55
	my := float64(y0) + plotHeight*.60
	*out = append(*out, fmt.Sprintf(`  <text x="%s" y="%s" class="mult" text-anchor="middle">%s</text>`, f1(mx), f1(my), esc(p.Mult)), fmt.Sprintf(`  <text x="%s" y="%s" class="msub" text-anchor="middle">%s</text>`, f1(mx), f1(my+20), esc(p.MultSub)), fmt.Sprintf(`  <text x="%s" y="%s" class="mfoot" text-anchor="middle">%s</text>`, f1(mx), f1(my+40), esc(p.MultFoot)))
}

func points(ps []point) string {
	var b bytes.Buffer
	for i, p := range ps {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%s,%s", f1(p.x), f1(p.y))
	}
	return b.String()
}

func OutputPath(dataPath string, d Data) string {
	out := d.Meta.OutSVG
	if out == "" {
		out = DefaultOut
	}
	if filepath.IsAbs(out) {
		return out
	}
	root := filepath.Dir(filepath.Dir(dataPath))
	return filepath.Join(root, filepath.FromSlash(out))
}

func Generate(dataPath string, check bool) (string, error) {
	d, err := Load(dataPath)
	if err != nil {
		return "", err
	}
	svg, err := Render(d)
	if err != nil {
		return "", err
	}
	out := OutputPath(dataPath, d)
	if check {
		got, err := os.ReadFile(out)
		if err != nil {
			return out, fmt.Errorf("turn-tax visual drift: %w", err)
		}
		if !bytes.Equal(got, svg) {
			return out, fmt.Errorf("turn-tax visual drift: %s is stale", out)
		}
		return out, nil
	}
	if err := os.WriteFile(out, svg, 0o644); err != nil {
		return out, err
	}
	return out, nil
}
