package launchvisual

import (
	"errors"
	"fmt"
	"math"
	"path"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Canvas and layout, in SVG user units. Every label is placed in a slot whose width is
// known here, and fitText refuses a label that would not fit, so no text overlaps.
const (
	canvasW        = 900
	marginX        = 40.0
	bodyTop        = 74.0
	rowH           = 236.0 // one row of panels, top edge to bottom label
	gapX           = 16.0
	gapY           = 14.0
	legendGap      = 22.0 // body bottom to the legend line
	footerGap      = 22.0 // legend line to the first footer line
	footerStep     = 11.0
	maxFooterLines = 16
	footerPadBot   = 14.0 // room under the last footer line so its descenders stay on the canvas
	footerChars    = 130
	plotInsetX     = 6.0
	axisGutter     = 36.0 // room for tick labels left of a panel that draws the axis
	plotPadTop     = 44.0 // cell top to plot top: title, subtitle, unit label
	plotPadBot     = 44.0 // plot bottom to cell bottom: series label and optional second line
	headroom       = 26.0 // room above the tallest bar for value and ratio labels
	barMaxWidth    = 72.0
	barSlotShare   = 0.56
	maxAxisSteps   = 5 // a shared axis has at most this many ticks above zero
)

// Character advance estimates for Helvetica at the sizes used, in px.
const (
	charsKicker   = 7.0
	charsHeader   = 10.8 // 18 px bold
	charsSubhead  = 6.4  // 12 px
	charsTitle    = 7.6  // 13 px bold
	charsRegular  = 5.8  // 10.5 px
	charsValue    = 6.2  // 11 px
	charsFooter   = 5.4  // 10 px
	legendSwatch  = 28.0
	legendGapItem = 20.0
)

const (
	colorBG    = "#ffffff"
	colorInk   = "#111111"
	colorSub   = "#444444"
	colorAxis  = "#333333"
	colorGrid  = "#e5e7eb"
	colorRef   = "#8b949e"
	colorFAK   = "#1f6feb"
	colorBase  = "#6e7781"
	colorBox2  = "#9a6700"
	colorPale  = "#fbf3dc"
	colorOptIn = "#ddf4ff"
	colorLoss  = "#b42318"
	hatchFill  = "url(#lv-hatch)"
)

// Rendered is one generated SVG and its repo-relative path.
type Rendered struct {
	Rel string
	SVG []byte
}

// ReadFunc reads a repo-relative, slash-separated path.
type ReadFunc func(rel string) ([]byte, error)

type rect struct{ x, y, w, h float64 }

type point struct{ x, y float64 }

// axisScale is the shared scale of one axis in one chart: every bar on it is value/top of the
// plot's usable height, and ticks are multiples of step up to top.
type axisScale struct {
	unit string
	top  float64
	step float64
}

// legendItem is one legend entry: a reference line sample, or a swatch for a fill kind.
type legendItem struct {
	kind string // "ref", "loss", "box2" or "optin"
	text string
}

// Render produces every chart in the data file. Each drawn number comes from an index row
// (value, display, ratio, build id), and each bar height is that row's value on its axis.
func Render(read ReadFunc, dataRel, indexRel string) ([]Rendered, ChartFile, Index, error) {
	dataBytes, err := read(dataRel)
	if err != nil {
		return nil, ChartFile{}, Index{}, fmt.Errorf("read %s: %w", dataRel, err)
	}
	cf, err := ParseChartFile(dataBytes)
	if err != nil {
		return nil, ChartFile{}, Index{}, err
	}
	indexBytes, err := read(indexRel)
	if err != nil {
		return nil, cf, Index{}, fmt.Errorf("read %s: %w", indexRel, err)
	}
	ix, err := ParseIndex(indexBytes)
	if err != nil {
		return nil, cf, Index{}, err
	}
	byID := ix.ByID()
	dir := path.Dir(dataRel)
	out := make([]Rendered, 0, len(cf.Charts))
	for _, c := range cf.Charts {
		svg, err := renderChart(c, byID)
		if err != nil {
			return nil, cf, ix, fmt.Errorf("chart %s: %w", c.ID, err)
		}
		out = append(out, Rendered{Rel: path.Join(dir, c.File), SVG: svg})
	}
	return out, cf, ix, nil
}

// renderChart lays the panels out in rows, draws each axis once per row, and appends the legend,
// the data-file footers, and one caveat per drawn row.
func renderChart(c Chart, byID map[string]Row) ([]byte, error) {
	if err := fitText(c.Title, canvasW-2*marginX, charsHeader); err != nil {
		return nil, err
	}
	if err := fitText(c.Subtitle, canvasW-2*marginX, charsSubhead); err != nil {
		return nil, err
	}
	if err := fitText(c.Kicker, canvasW-2*marginX, charsKicker); err != nil {
		return nil, err
	}
	scales, err := chartScales(c, byID)
	if err != nil {
		return nil, err
	}

	cells, rows, err := placePanels(c)
	if err != nil {
		return nil, err
	}
	bodyBot := bodyTop + float64(rows)*rowH + float64(rows-1)*gapY

	var body []string
	var legend []legendItem
	var caveats []string
	seenCaveat := map[string]bool{}
	note := func(r Row) {
		if r.Caveat != "" && !seenCaveat[r.Caveat] {
			seenCaveat[r.Caveat] = true
			caveats = append(caveats, r.Caveat)
		}
	}
	ticked := map[string]bool{} // "row/axis": the axis labels are drawn once per panel row
	for i, p := range c.Panels {
		cell := cells[i]
		key := fmt.Sprintf("%.0f/%s", cell.y, p.Axis)
		drawTicks := !ticked[key]
		ticked[key] = true
		frag, items, err := renderPanel(cell, p, scales[p.Axis], drawTicks, byID, note)
		if err != nil {
			return nil, fmt.Errorf("panel %q: %w", p.Title, err)
		}
		body = append(body, frag...)
		legend = appendLegend(legend, items...)
	}

	footer := footerLines(append(append([]string(nil), c.Footers...), caveats...))
	if len(footer) > maxFooterLines {
		return nil, fmt.Errorf("footer needs %d lines, the limit is %d", len(footer), maxFooterLines)
	}
	legendY := bodyBot + legendGap
	legendLine, err := legendSVG(legend, legendY)
	if err != nil {
		return nil, err
	}
	footerY := legendY + footerGap
	height := legendY + 14
	if len(footer) > 0 {
		height = footerY + float64(len(footer)-1)*footerStep + footerPadBot
	}
	height = math.Ceil(height)

	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%.0f" viewBox="0 0 %d %.0f" role="img" aria-labelledby="lv-title lv-desc" font-family="Helvetica, Arial, sans-serif">`+"\n", canvasW, height, canvasW, height)
	fmt.Fprintf(&b, "  <title id=\"lv-title\">%s</title>\n", esc(c.Title))
	fmt.Fprintf(&b, "  <desc id=\"lv-desc\">%s</desc>\n", esc(c.Subtitle))
	fmt.Fprintf(&b, "  <defs><pattern id=\"lv-hatch\" width=\"6\" height=\"6\" patternUnits=\"userSpaceOnUse\" patternTransform=\"rotate(45)\"><rect width=\"6\" height=\"6\" fill=\"%s\"/><line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"6\" stroke=\"%s\" stroke-width=\"2\"/></pattern></defs>\n", colorPale, colorBox2)
	fmt.Fprintf(&b, "  <rect width=\"%d\" height=\"%.0f\" fill=\"%s\"/>\n", canvasW, height, colorBG)
	writeText(&b, marginX, 22, 11, "bold", colorFAK, "start", c.Kicker)
	writeText(&b, marginX, 44, 18, "bold", colorInk, "start", c.Title)
	writeText(&b, marginX, 62, 12, "", colorSub, "start", c.Subtitle)
	for _, l := range body {
		b.WriteString(l)
		b.WriteString("\n")
	}
	b.WriteString(legendLine)
	for i, f := range footer {
		writeText(&b, marginX, footerY+float64(i)*footerStep, 10, "", colorSub, "start", f)
	}
	b.WriteString("</svg>\n")
	return []byte(b.String()), nil
}

// placePanels assigns each panel a cell: panels fill the grid row-major, a span takes adjacent
// columns, and a panel that does not fit on the current row starts the next one.
func placePanels(c Chart) ([]rect, int, error) {
	cols := c.Columns
	cellW := (canvasW - 2*marginX - float64(cols-1)*gapX) / float64(cols)
	cells := make([]rect, 0, len(c.Panels))
	col, row := 0, 0
	for _, p := range c.Panels {
		span := p.Span
		if span == 0 {
			span = 1
		}
		if col+span > cols {
			col = 0
			row++
		}
		cells = append(cells, rect{
			x: marginX + float64(col)*(cellW+gapX),
			y: bodyTop + float64(row)*(rowH+gapY),
			w: float64(span)*cellW + float64(span-1)*gapX,
			h: rowH,
		})
		col += span
	}
	return cells, row + 1, nil
}

// chartScales finds each axis's top and tick step from the largest value on it. Every row on an
// axis must share one unit, so a single scale is never drawn across different quantities.
func chartScales(c Chart, byID map[string]Row) (map[string]axisScale, error) {
	maxV := map[string]float64{}
	unit := map[string]string{}
	for _, p := range c.Panels {
		ids := make([]string, 0, len(p.Series)+len(p.References))
		for _, s := range p.Series {
			ids = append(ids, s.Row)
		}
		for _, r := range p.References {
			ids = append(ids, r.Row)
		}
		for _, id := range ids {
			r, err := lookupRow(byID, id)
			if err != nil {
				return nil, err
			}
			if u, ok := unit[p.Axis]; ok && u != r.Unit {
				return nil, fmt.Errorf("axis %q mixes units %q and %q", p.Axis, u, r.Unit)
			}
			unit[p.Axis] = r.Unit
			maxV[p.Axis] = math.Max(maxV[p.Axis], r.Value)
		}
	}
	out := make(map[string]axisScale, len(maxV))
	for axis, m := range maxV {
		if m <= 0 {
			return nil, fmt.Errorf("axis %q has no positive value to scale against", axis)
		}
		top, step := niceScale(m)
		out[axis] = axisScale{unit: unit[axis], top: top, step: step}
	}
	return out, nil
}

// niceScale rounds the largest value up to a multiple of a 1, 2, 2.5, 5 or 10 times a power of
// ten, choosing the smallest step that keeps the axis within maxAxisSteps.
func niceScale(maxV float64) (top, step float64) {
	mag := math.Pow(10, math.Floor(math.Log10(maxV)))
	for _, m := range []float64{1, 2, 2.5, 5, 10} {
		s := m * mag
		if maxV/s <= maxAxisSteps {
			return math.Ceil(maxV/s-1e-9) * s, s
		}
	}
	return math.Ceil(maxV/(10*mag)-1e-9) * 10 * mag, 10 * mag
}

// ticks lists 0, step, 2*step ... top for an axis.
func (a axisScale) ticks() []float64 {
	n := int(math.Round(a.top / a.step))
	out := make([]float64, 0, n+1)
	for k := 0; k <= n; k++ {
		out = append(out, float64(k)*a.step)
	}
	return out
}

// renderPanel draws one plot inside cell on its axis scale. drawTicks is true for the first panel
// of the axis in its row, which carries the tick labels; every panel carries the gridlines.
func renderPanel(cell rect, p Panel, sc axisScale, drawTicks bool, byID map[string]Row, note func(Row)) ([]string, []legendItem, error) {
	if err := fitText(p.Title, cell.w, charsTitle); err != nil {
		return nil, nil, err
	}
	if err := fitText(p.Subtitle, cell.w, charsRegular); err != nil {
		return nil, nil, err
	}
	if !validStyles[p.Style] {
		return nil, nil, fmt.Errorf("bad style %q", p.Style)
	}

	series := make([]Row, len(p.Series))
	for i, s := range p.Series {
		r, err := lookupRow(byID, s.Row)
		if err != nil {
			return nil, nil, err
		}
		series[i] = r
	}
	refs := make([]Row, len(p.References))
	for i, ref := range p.References {
		r, err := lookupRow(byID, ref.Row)
		if err != nil {
			return nil, nil, err
		}
		refs[i] = r
	}

	gutter := plotInsetX
	if drawTicks {
		gutter = axisGutter
	}
	plotL := cell.x + gutter
	plotR := cell.x + cell.w - plotInsetX
	plotB := cell.y + cell.h - plotPadBot
	plotT := cell.y + plotPadTop
	usable := plotB - plotT - headroom
	if usable <= 0 {
		return nil, nil, errors.New("panel too short for its labels")
	}
	height := func(v float64) float64 { return v / sc.top * usable }

	var out []string
	var legend []legendItem
	out = append(out, svgText(cell.x, cell.y+14, 13, "bold", colorInk, "start", p.Title))
	out = append(out, svgText(cell.x, cell.y+28, 10.5, "", colorSub, "start", p.Subtitle))
	for _, t := range sc.ticks() {
		y := plotB - height(t)
		if t > 0 {
			out = append(out, svgLine(plotL, y, plotR, y, colorGrid, 1, ""))
		}
		if drawTicks {
			out = append(out, svgText(cell.x+axisGutter-5, y+3.5, 10, "", colorSub, "end", formatTick(t)))
		}
	}
	if drawTicks {
		out = append(out, svgText(cell.x+axisGutter-5, plotT-4, 10, "", colorSub, "end", sc.unit))
	}
	out = append(out, svgLine(plotL, plotB, plotR, plotB, colorAxis, 1, ""))

	for i, ref := range p.References {
		r := refs[i]
		y := plotB - height(r.Value)
		out = append(out, svgLine(plotL, y, plotR, y, colorRef, 1.5, "6 4"))
		legend = append(legend, legendItem{kind: "ref", text: ref.Label + " " + r.Display()})
		note(r)
	}

	slot := (plotR - plotL) / float64(len(p.Series))
	barW := math.Min(slot*barSlotShare, barMaxWidth)
	var tops []point
	for i, s := range p.Series {
		r := series[i]
		sub := s.Sub
		if s.Build {
			if r.FakCommit == "" {
				return nil, nil, fmt.Errorf("series %q asks for a build id but row %s has none", s.Label, s.Row)
			}
			sub = "build " + r.FakCommit
		}
		if err := fitText(s.Label, slot, charsRegular); err != nil {
			return nil, nil, err
		}
		if err := fitText(sub, slot, charsFooter); err != nil {
			return nil, nil, err
		}
		if err := fitText(r.Display(), slot, charsValue); err != nil {
			return nil, nil, err
		}
		cx := plotL + (float64(i)+0.5)*slot
		h := height(r.Value)
		top := plotB - h
		fill, stroke, dash := seriesPaint(s.Tone, r.Kind == "loss")
		out = append(out, svgBar(cx-barW/2, top, barW, h, fill, stroke, dash))
		out = append(out, svgText(cx, top-6, 11, "bold", colorInk, "middle", r.Display()))
		if rl := r.RatioLabel(); rl != "" {
			ratioColour := colorFAK
			if r.Kind == "loss" {
				ratioColour = colorLoss
			}
			out = append(out, svgText(cx, top-19, 10.5, "bold", ratioColour, "middle", rl))
		}
		out = append(out, svgText(cx, plotB+14, 10.5, "", colorSub, "middle", s.Label))
		if sub != "" {
			out = append(out, svgText(cx, plotB+28, 10, "", colorSub, "middle", sub))
		}
		tops = append(tops, point{cx, top})
		note(r)
		if r.Kind == "loss" {
			legend = appendLegend(legend, legendItem{kind: "loss", text: "loss"})
		}
		switch s.Tone {
		case "box2":
			legend = appendLegend(legend, legendItem{kind: "box2", text: "second box, compared within it"})
		case "optin":
			legend = appendLegend(legend, legendItem{kind: "optin", text: "opt-in, off by default"})
		case "unmatched":
			legend = appendLegend(legend, legendItem{kind: "unmatched", text: "unmatched reference, not measured on the same box"})
		}
	}
	if p.Style == "staircase" {
		out = append(out, svgPolyline(tops, colorAxis))
	}
	return out, legend, nil
}

// formatTick prints an axis tick without trailing zeros.
func formatTick(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }

func seriesPaint(tone string, loss bool) (fill, stroke, dash string) {
	switch {
	case loss:
		return colorLoss, "", ""
	case tone == "optin":
		return colorOptIn, colorFAK, "4 3"
	case tone == "box2":
		return hatchFill, colorBox2, ""
	case tone == "unmatched":
		return colorBG, colorBase, "3 2"
	case tone == "fak":
		return colorFAK, "", ""
	default:
		return colorBase, "", ""
	}
}

// footerLines packs the footers and the drawn rows' caveats into lines. Each text is split into
// sentences and an identical sentence is kept once. Sentences are packed whole, so a caveat is
// never broken across a line shared with unrelated text; a sentence longer than one line wraps
// on lines of its own.
func footerLines(texts []string) []string {
	var lines []string
	cur := ""
	flush := func() {
		if cur != "" {
			lines = append(lines, cur)
			cur = ""
		}
	}
	seen := map[string]bool{}
	for _, t := range texts {
		for _, s := range sentences(t) {
			if seen[s] {
				continue
			}
			seen[s] = true
			if utf8.RuneCountInString(s) > footerChars {
				flush()
				lines = append(lines, wrapLines(s, footerChars)...)
				continue
			}
			if cur != "" && utf8.RuneCountInString(cur)+1+utf8.RuneCountInString(s) > footerChars {
				flush()
			}
			if cur == "" {
				cur = s
			} else {
				if !strings.HasSuffix(cur, ".") {
					cur += "."
				}
				cur += " " + s
			}
		}
	}
	flush()
	return lines
}

// sentences splits text at sentence ends, keeping each full stop with its sentence.
func sentences(s string) []string {
	parts := strings.Split(strings.TrimSpace(s), ". ")
	out := make([]string, 0, len(parts))
	for i, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if i < len(parts)-1 {
			p += "."
		}
		out = append(out, p)
	}
	return out
}

func lookupRow(byID map[string]Row, id string) (Row, error) {
	r, ok := byID[id]
	if !ok {
		return Row{}, fmt.Errorf("unknown headline row %q", id)
	}
	return r, nil
}

// legendSVG lays the legend out on one line at y and refuses to overflow the canvas.
func legendSVG(items []legendItem, y float64) (string, error) {
	if len(items) == 0 {
		return "", nil
	}
	var b strings.Builder
	x := marginX
	for _, it := range items {
		width := legendSwatch + float64(utf8.RuneCountInString(it.text))*charsRegular + legendGapItem
		switch it.kind {
		case "ref":
			b.WriteString(svgLine(x, y-4, x+22, y-4, colorRef, 1.5, "6 4") + "\n")
		case "box2":
			b.WriteString(svgSwatch(x, y-9, 12, 10, hatchFill, colorBox2, "") + "\n")
		case "optin":
			b.WriteString(svgSwatch(x, y-9, 12, 10, colorOptIn, colorFAK, "4 3") + "\n")
		case "unmatched":
			b.WriteString(svgSwatch(x, y-9, 12, 10, colorBG, colorBase, "3 2") + "\n")
		default:
			b.WriteString(svgSwatch(x, y-9, 12, 10, colorLoss, "", "") + "\n")
		}
		b.WriteString(svgText(x+legendSwatch-6, y, 11, "", colorSub, "start", it.text) + "\n")
		x += width
	}
	if x-legendGapItem > canvasW-marginX {
		return "", errors.New("legend does not fit on one line")
	}
	return b.String(), nil
}

// appendLegend adds items once each, keeping first-seen order.
func appendLegend(legend []legendItem, items ...legendItem) []legendItem {
	for _, it := range items {
		dup := false
		for _, have := range legend {
			if have == it {
				dup = true
				break
			}
		}
		if !dup {
			legend = append(legend, it)
		}
	}
	return legend
}

// fitText refuses a label that would run past its slot at the given character advance.
func fitText(s string, maxPx, charPx float64) error {
	if float64(utf8.RuneCountInString(s))*charPx > maxPx {
		return fmt.Errorf("text %q does not fit in %.0f px", s, maxPx)
	}
	return nil
}

// wrapLines breaks s on spaces into lines of at most width runes.
func wrapLines(s string, width int) []string {
	var lines []string
	cur := ""
	for _, w := range strings.Fields(s) {
		switch {
		case cur == "":
			cur = w
		case utf8.RuneCountInString(cur)+1+utf8.RuneCountInString(w) > width:
			lines = append(lines, cur)
			cur = w
		default:
			cur += " " + w
		}
	}
	if cur != "" {
		lines = append(lines, cur)
	}
	return lines
}

// svgBar is one data bar. Its class lets a test pick the bars out of the drawing.
func svgBar(x, y, w, h float64, fill, stroke, dash string) string {
	s := fmt.Sprintf(`  <rect class="lv-bar" x="%s" y="%s" width="%s" height="%s" fill="%s"`, f2(x), f2(y), f2(w), f2(h), fill)
	if stroke != "" {
		s += fmt.Sprintf(` stroke="%s" stroke-width="1.5"`, stroke)
	}
	if dash != "" {
		s += fmt.Sprintf(` stroke-dasharray="%s"`, dash)
	}
	return s + "/>"
}

// svgSwatch is a legend colour sample.
func svgSwatch(x, y, w, h float64, fill, stroke, dash string) string {
	s := fmt.Sprintf(`  <rect x="%s" y="%s" width="%s" height="%s" fill="%s"`, f2(x), f2(y), f2(w), f2(h), fill)
	if stroke != "" {
		s += fmt.Sprintf(` stroke="%s" stroke-width="1.5"`, stroke)
	}
	if dash != "" {
		s += fmt.Sprintf(` stroke-dasharray="%s"`, dash)
	}
	return s + "/>"
}

func svgLine(x1, y1, x2, y2 float64, stroke string, width float64, dash string) string {
	s := fmt.Sprintf(`  <line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"`, f2(x1), f2(y1), f2(x2), f2(y2), stroke, f2(width))
	if dash != "" {
		s += fmt.Sprintf(` stroke-dasharray="%s"`, dash)
	}
	return s + "/>"
}

func svgPolyline(ps []point, stroke string) string {
	var pts []string
	for _, p := range ps {
		pts = append(pts, f2(p.x)+","+f2(p.y))
	}
	return fmt.Sprintf(`  <polyline points="%s" fill="none" stroke="%s" stroke-width="1.5"/>`, strings.Join(pts, " "), stroke)
}

func svgText(x, y, size float64, weight, fill, anchor, s string) string {
	w := ""
	if weight != "" {
		w = ` font-weight="` + weight + `"`
	}
	return fmt.Sprintf(`  <text x="%s" y="%s" font-size="%s"%s fill="%s" text-anchor="%s">%s</text>`, f2(x), f2(y), f2(size), w, fill, anchor, esc(s))
}

func writeText(b *strings.Builder, x, y, size float64, weight, fill, anchor, s string) {
	b.WriteString(svgText(x, y, size, weight, fill, anchor, s))
	b.WriteString("\n")
}

func f2(v float64) string { return strconv.FormatFloat(v, 'f', 2, 64) }

var xmlEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;")

func esc(s string) string { return xmlEscaper.Replace(s) }
