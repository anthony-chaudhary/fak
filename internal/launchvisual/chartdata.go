package launchvisual

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// ChartSchema names the figure data layout. The file holds structure only: chart ids,
// panels, axes, ordered headline ids, label text, footers, and the doc-check policy. Numbers
// come from the index at render time, never from this file.
const ChartSchema = "fak.launchvisual.charts.v1"

// DefaultData and DefaultIndex are repo-root-relative locations used when a flag is unset.
const (
	DefaultData  = "docs/launch/assets/inkling2-charts.data.json"
	DefaultIndex = "docs/benchmarks/published/HEADLINES.json"
)

// defaultDocs is the launch doc checked when the data file names none.
var defaultDocs = []string{"docs/launch/inkling2-launch-2026-11-24.md"}

// ChartFile is the decoded figure data file.
type ChartFile struct {
	Schema           string          `json:"schema"`
	Charts           []Chart         `json:"charts"`
	Docs             []string        `json:"docs"`
	AllowedNumbers   []AllowedNumber `json:"allowed_numbers"`
	SupersededTokens []string        `json:"superseded_tokens"`
}

// AllowedNumber exempts one decimal token in the docs for a non-index fact (for example
// a build time). The reason is kept in the file so the exemption is reviewable.
type AllowedNumber struct {
	Token  string `json:"token"`
	Reason string `json:"reason"`
}

// Chart is one output SVG. Columns is the grid width; panels fill it row-major, and a panel
// with span > 1 takes that many adjacent columns.
type Chart struct {
	ID       string   `json:"id"`
	File     string   `json:"file"`
	Kicker   string   `json:"kicker"`
	Title    string   `json:"title"`
	Subtitle string   `json:"subtitle"`
	Columns  int      `json:"columns"`
	Panels   []Panel  `json:"panels"`
	Footers  []string `json:"footers"`
}

// Panel is one plot. Panels that name the same Axis within a chart share one scale, drawn
// once per row with tick labels: every bar on that axis is drawn at value/top of the axis.
// Style "bars" draws independent bars; "staircase" also joins the bar tops so the steps read
// as one climb. References are dashed horizontal lines at a row's value.
type Panel struct {
	Title      string      `json:"title"`
	Subtitle   string      `json:"subtitle"`
	Style      string      `json:"style"`
	Axis       string      `json:"axis"`
	Span       int         `json:"span"`
	Series     []Series    `json:"series"`
	References []Reference `json:"references"`
}

// Series is one bar, named by its index row. Tone picks the fill: "box2" is hatched (a second
// box, compared within that box), "optin" is outlined and dashed (opt-in, not the default). A
// loss row always draws in the loss colour. Sub is a second line under the label; Build
// replaces it with the row's build id from the index.
type Series struct {
	Row   string `json:"row"`
	Label string `json:"label"`
	Tone  string `json:"tone"`
	Sub   string `json:"sub"`
	Build bool   `json:"build"`
}

// Reference is a dashed line at a row's value.
type Reference struct {
	Row   string `json:"row"`
	Label string `json:"label"`
}

var (
	decimalInText = regexp.MustCompile(`\d\.\d`)
	chartFileName = regexp.MustCompile(`^[a-z0-9][a-z0-9.-]*\.svg$`)
	axisName      = regexp.MustCompile(`^[a-z][a-z0-9-]*$`)
	validTones    = map[string]bool{"": true, "fak": true, "base": true, "box2": true, "optin": true, "unmatched": true}
	validStyles   = map[string]bool{"bars": true, "staircase": true}
)

// ParseChartFile strictly decodes and validates a figure data file.
func ParseChartFile(b []byte) (ChartFile, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var cf ChartFile
	if err := dec.Decode(&cf); err != nil {
		return ChartFile{}, fmt.Errorf("launch chart data: %w", err)
	}
	if cf.Schema != ChartSchema {
		return ChartFile{}, fmt.Errorf("launch chart data: schema %q, want %q", cf.Schema, ChartSchema)
	}
	if len(cf.Charts) == 0 {
		return ChartFile{}, errors.New("launch chart data: no charts")
	}
	if len(cf.Docs) == 0 {
		cf.Docs = defaultDocs
	}
	seen := map[string]bool{}
	for _, c := range cf.Charts {
		if err := c.validate(); err != nil {
			return ChartFile{}, fmt.Errorf("launch chart %q: %w", c.ID, err)
		}
		if seen[c.ID] || seen[c.File] {
			return ChartFile{}, fmt.Errorf("launch chart %q: duplicate id or file", c.ID)
		}
		seen[c.ID], seen[c.File] = true, true
	}
	return cf, nil
}

func (c Chart) validate() error {
	if c.ID == "" || !chartFileName.MatchString(c.File) {
		return fmt.Errorf("id and an .svg file name are required, got file %q", c.File)
	}
	if c.Columns < 1 || c.Columns > 6 {
		return fmt.Errorf("columns %d outside 1..6", c.Columns)
	}
	if len(c.Panels) == 0 || len(c.Panels) > 8 {
		return fmt.Errorf("%d panels outside 1..8", len(c.Panels))
	}
	texts := append([]string{c.Kicker, c.Title, c.Subtitle}, c.Footers...)
	for _, p := range c.Panels {
		if !validStyles[p.Style] {
			return fmt.Errorf("panel %q: bad style %q", p.Title, p.Style)
		}
		if !axisName.MatchString(p.Axis) {
			return fmt.Errorf("panel %q: axis %q must be a lowercase name", p.Title, p.Axis)
		}
		if p.Span < 0 || p.Span > c.Columns {
			return fmt.Errorf("panel %q: span %d outside 0..%d", p.Title, p.Span, c.Columns)
		}
		if len(p.Series) == 0 || len(p.Series) > 6 || len(p.References) > 2 {
			return fmt.Errorf("panel %q: needs 1..6 series and at most 2 references", p.Title)
		}
		texts = append(texts, p.Title, p.Subtitle)
		for _, s := range p.Series {
			if s.Row == "" || s.Label == "" || !validTones[s.Tone] {
				return fmt.Errorf("panel %q: series needs row, label, and a known tone", p.Title)
			}
			if s.Build && s.Sub != "" {
				return fmt.Errorf("panel %q: series %q sets both build and sub", p.Title, s.Label)
			}
			texts = append(texts, s.Label, s.Sub)
		}
		for _, r := range p.References {
			if r.Row == "" || r.Label == "" {
				return fmt.Errorf("panel %q: reference needs row and label", p.Title)
			}
			texts = append(texts, r.Label)
		}
	}
	for _, t := range texts {
		if decimalInText.MatchString(t) {
			return fmt.Errorf("label %q carries a decimal number; numbers come from the index", t)
		}
	}
	return nil
}

// RowIDs lists every headline row a chart draws, series and references, in first-seen order.
func (cf ChartFile) RowIDs() []string {
	var ids []string
	seen := map[string]bool{}
	add := func(id string) {
		if !seen[id] {
			seen[id] = true
			ids = append(ids, id)
		}
	}
	for _, c := range cf.Charts {
		for _, p := range c.Panels {
			for _, s := range p.Series {
				add(s.Row)
			}
			for _, r := range p.References {
				add(r.Row)
			}
		}
	}
	return ids
}

// allowedSet indexes allowed_numbers tokens for the doc check.
func allowedSet(cf ChartFile) map[string]bool {
	m := make(map[string]bool, len(cf.AllowedNumbers))
	for _, a := range cf.AllowedNumbers {
		if strings.TrimSpace(a.Token) != "" {
			m[a.Token] = true
		}
	}
	return m
}
