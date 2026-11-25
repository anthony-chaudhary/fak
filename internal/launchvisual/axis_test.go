package launchvisual

import (
	"bytes"
	"encoding/xml"
	"io"
	"math"
	"strconv"
	"testing"
)

// barHeights returns the heights of the data bars in drawing order. Bars carry the lv-bar class;
// legend swatches and gridlines do not.
func barHeights(t *testing.T, svg []byte) []float64 {
	t.Helper()
	dec := xml.NewDecoder(bytes.NewReader(svg))
	var out []float64
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("parse svg: %v", err)
		}
		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "rect" {
			continue
		}
		if cls, _ := attrOf(se, "class"); cls != "lv-bar" {
			continue
		}
		h, err := strconv.ParseFloat(attrMust(t, se, "height"), 64)
		if err != nil {
			t.Fatalf("bar height: %v", err)
		}
		out = append(out, h)
	}
	return out
}

// TestBarsOnOneAxisHaveHeightsProportionalToValues draws three bars from two panels on one axis:
// each bar's height must be its value over the axis top, so the heights keep the value ratios
// across panels and a repeated row draws the same height.
func TestBarsOnOneAxisHaveHeightsProportionalToValues(t *testing.T) {
	byID := goldenIndex(t).ByID()
	c := Chart{
		ID: "axis", File: "axis.svg", Kicker: "K", Title: "T", Subtitle: "S", Columns: 2,
		Panels: []Panel{
			{Title: "A", Subtitle: "a", Style: "bars", Axis: "decode", Series: []Series{
				{Row: "spec.plain.tok_s", Label: "plain"},
				{Row: "spec.d2.tok_s", Label: "d2"},
			}},
			{Title: "B", Subtitle: "b", Style: "bars", Axis: "decode", Series: []Series{
				{Row: "spec.d4.tok_s", Label: "d4"},
				{Row: "spec.plain.tok_s", Label: "plain again"},
			}},
		},
	}
	svg, err := renderChart(c, byID)
	if err != nil {
		t.Fatal(err)
	}
	h := barHeights(t, svg)
	if len(h) != 4 {
		t.Fatalf("want 4 bars, got %d", len(h))
	}
	plain, d2, d4 := byID["spec.plain.tok_s"].Value, byID["spec.d2.tok_s"].Value, byID["spec.d4.tok_s"].Value
	if want := plain / d2; math.Abs(h[0]/h[1]-want) > 1e-3*want {
		t.Errorf("plain/d2 height ratio %.6f, value ratio %.6f", h[0]/h[1], want)
	}
	if want := d4 / d2; math.Abs(h[2]/h[1]-want) > 1e-3*want {
		t.Errorf("d4/d2 height ratio %.6f, value ratio %.6f", h[2]/h[1], want)
	}
	if math.Abs(h[0]-h[3]) > 1e-9 {
		t.Errorf("same row drew heights %v and %v on one axis", h[0], h[3])
	}
}

func TestAxisRefusesMixedUnits(t *testing.T) {
	byID := map[string]Row{
		"a.tok": {ID: "a.tok", Unit: "tok/s", Value: 10, Samples: []float64{10}, Agg: "scalar", DisplayDecimals: 1},
		"b.ms":  {ID: "b.ms", Unit: "ms", Value: 20, Samples: []float64{20}, Agg: "scalar", DisplayDecimals: 1},
	}
	c := Chart{ID: "mix", File: "mix.svg", Columns: 1, Panels: []Panel{{
		Title: "M", Style: "bars", Axis: "decode",
		Series: []Series{{Row: "a.tok", Label: "a"}, {Row: "b.ms", Label: "b"}},
	}}}
	if _, err := renderChart(c, byID); err == nil {
		t.Fatal("one axis mixing tok/s and ms rendered")
	}
}

func TestNiceScaleKeepsTicksFew(t *testing.T) {
	cases := map[float64]float64{180.9: 200, 783.7: 800, 5106: 6000, 6.32: 8}
	for v, want := range cases {
		top, step := niceScale(v)
		if top != want || top < v {
			t.Errorf("niceScale(%v) top %v step %v, want top %v", v, top, step, want)
		}
		if n := top / step; n > maxAxisSteps {
			t.Errorf("niceScale(%v) has %v ticks, limit %d", v, n, maxAxisSteps)
		}
	}
}

func TestDocCheckNumericSlashPairIsChecked(t *testing.T) {
	vs := docViolations(t, "Decode 117.16/110.67 tok/s.\n")
	tokens := map[string]bool{}
	for _, v := range vs {
		tokens[v.Token] = true
	}
	if !tokens["117.16"] || !tokens["110.67"] {
		t.Fatalf("numeric slash pair not checked: %+v", vs)
	}
}

func TestDocCheckFlagsIntegerRatio(t *testing.T) {
	vs := docViolations(t, "A 3x gain on this card.\n")
	if len(vs) != 1 || vs[0].Token != "3x" {
		t.Fatalf("integer ratio not flagged: %+v", vs)
	}
}

func TestRoundHalfUpMatchesPublisher(t *testing.T) {
	cases := []struct {
		v    float64
		d    int
		want float64
	}{
		{1.005, 2, 1.01},
		{0.125, 2, 0.13},
		{-1.005, 2, -1.01},
		{1.4, 0, 1},
	}
	for _, c := range cases {
		if got := RoundHalfUp(c.v, c.d); math.Abs(got-c.want) > 1e-12 {
			t.Errorf("RoundHalfUp(%v,%d)=%v want %v", c.v, c.d, got, c.want)
		}
	}
}
