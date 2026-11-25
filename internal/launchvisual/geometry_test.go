package launchvisual

import (
	"bytes"
	"encoding/xml"
	"io"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

// glyphAdvance estimates a Helvetica character advance as a fraction of the font size; the
// same estimate sizes the labels in render.go, so a label that fits its slot here fits the eye.
const glyphAdvance = 0.6

// ascent and descent bound a line above and below its baseline, as fractions of the font size;
// capitals and digits stand near 0.72 em, so the ascent is the cap height, not the full em.
const (
	ascent  = 0.75
	descent = 0.25
)

// svgLabel is one <text> element with its estimated box.
type svgLabel struct {
	text                     string
	left, right, top, bottom float64
}

// parseLabels reads the viewBox and every text element of an SVG and estimates each label's box.
func parseLabels(t *testing.T, b []byte) (w, h float64, labels []svgLabel) {
	t.Helper()
	dec := xml.NewDecoder(bytes.NewReader(b))
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("parse svg: %v", err)
		}
		se, ok := tok.(xml.StartElement)
		if !ok {
			continue
		}
		switch se.Name.Local {
		case "svg":
			vb, _ := attrOf(se, "viewBox")
			f := strings.Fields(vb)
			if len(f) != 4 {
				t.Fatalf("svg has no 4-part viewBox: %q", vb)
			}
			w, h = mustFloat(t, f[2]), mustFloat(t, f[3])
		case "text":
			x := mustFloat(t, attrMust(t, se, "x"))
			y := mustFloat(t, attrMust(t, se, "y"))
			size := mustFloat(t, attrMust(t, se, "font-size"))
			anchor, _ := attrOf(se, "text-anchor")
			var sb strings.Builder
			for {
				inner, err := dec.Token()
				if err != nil {
					t.Fatalf("text element: %v", err)
				}
				if cd, ok := inner.(xml.CharData); ok {
					sb.Write(cd)
				}
				if _, ok := inner.(xml.EndElement); ok {
					break
				}
			}
			text := sb.String()
			width := glyphAdvance * size * float64(utf8.RuneCountInString(text))
			l := svgLabel{text: text, top: y - ascent*size, bottom: y + descent*size}
			switch anchor {
			case "middle":
				l.left, l.right = x-width/2, x+width/2
			case "end":
				l.left, l.right = x-width, x
			default:
				l.left, l.right = x, x+width
			}
			labels = append(labels, l)
		}
	}
	if w == 0 || h == 0 {
		t.Fatal("svg has no viewBox size")
	}
	return w, h, labels
}

func attrOf(se xml.StartElement, name string) (string, bool) {
	for _, a := range se.Attr {
		if a.Name.Local == name {
			return a.Value, true
		}
	}
	return "", false
}

func attrMust(t *testing.T, se xml.StartElement, name string) string {
	t.Helper()
	v, ok := attrOf(se, name)
	if !ok {
		t.Fatalf("text element lacks %s", name)
	}
	return v
}

func mustFloat(t *testing.T, s string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("number %q: %v", s, err)
	}
	return v
}

func overlaps(a, b svgLabel) bool {
	return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom
}

// TestChartLabelsFitInsideViewBoxAndDoNotOverlap checks every chart the data file draws:
// each label's estimated box lies inside the viewBox and no two label boxes overlap.
func TestChartLabelsFitInsideViewBoxAndDoNotOverlap(t *testing.T) {
	rendered, cf, _, err := Render(repoReader(), DefaultData, DefaultIndex)
	if err != nil {
		t.Fatal(err)
	}
	if len(rendered) != len(cf.Charts) || len(rendered) == 0 {
		t.Fatalf("rendered %d charts for %d in the data file", len(rendered), len(cf.Charts))
	}
	for _, r := range rendered {
		w, h, labels := parseLabels(t, r.SVG)
		if len(labels) == 0 {
			t.Fatalf("%s has no labels", r.Rel)
		}
		for _, l := range labels {
			if l.left < 0 || l.top < 0 || l.right > w || l.bottom > h {
				t.Errorf("%s: label %q box [%.1f,%.1f]x[%.1f,%.1f] leaves the %.0fx%.0f viewBox", r.Rel, l.text, l.left, l.right, l.top, l.bottom, w, h)
			}
		}
		for i := range labels {
			for j := i + 1; j < len(labels); j++ {
				if overlaps(labels[i], labels[j]) {
					t.Errorf("%s: labels %q and %q overlap", r.Rel, labels[i].text, labels[j].text)
				}
			}
		}
	}
}
