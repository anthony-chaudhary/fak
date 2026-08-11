package turntaxvisual

import (
	"bytes"
	"encoding/xml"
	"os"
	"strings"
	"testing"
)

func TestRenderMatchesRetiredGeneratorArtifact(t *testing.T) {
	dataPath := "../../tools/hero_turntax.data.json"
	d, err := Load(dataPath)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Render(d)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("../../visuals/60-hero-turntax-curves.svg")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("Go renderer drifted from the checked-in Python-generated witness: got %d bytes, want %d", len(got), len(want))
	}
	var doc any
	if err := xml.Unmarshal(got, &doc); err != nil {
		t.Fatalf("rendered SVG is not well-formed XML: %v", err)
	}
}

func TestRenderPreservesHonestyAndCurveContract(t *testing.T) {
	d, err := Load("../../tools/hero_turntax.data.json")
	if err != nil {
		t.Fatal(err)
	}
	gotBytes, err := Render(d)
	if err != nil {
		t.Fatal(err)
	}
	got := string(gotBytes)
	if len(d.Panels) != 3 {
		t.Fatalf("panels = %d, want 3", len(d.Panels))
	}
	if n := strings.Count(got, "<polyline"); n < 2*len(d.Panels) {
		t.Fatalf("polylines = %d, want at least %d", n, 2*len(d.Panels))
	}
	for _, p := range d.Panels {
		for _, text := range []string{p.Title, p.Mult} {
			if !strings.Contains(got, text) {
				t.Errorf("render missing %q", text)
			}
		}
	}
	for _, text := range []string{Base, FAK, "MODELED", "643", "9.7×", "tuned warm-cache SOTA", "don't lead with"} {
		if !strings.Contains(got, text) {
			t.Errorf("render missing parity/honesty marker %q", text)
		}
	}
	if d.Panels[2].Mult != "4.1×" || !strings.Contains(d.Panels[2].MultFoot, "60×") {
		t.Fatalf("tuned-SOTA headline contract changed: %+v", d.Panels[2])
	}
	for _, item := range d.Legend {
		if !strings.Contains(got, item.Label) {
			t.Errorf("render missing legend label %q", item.Label)
		}
	}
	if !strings.Contains(got, d.Footer[:24]) {
		t.Error("render missing footer")
	}
}

func TestGenerateCheckDetectsDrift(t *testing.T) {
	data, err := os.ReadFile("../../tools/hero_turntax.data.json")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	dataPath := dir + "/data.json"
	// Keep this focused on the public Generate seam by rewriting only out_svg.
	data = bytes.Replace(data, []byte(`"out_svg": "visuals/60-hero-turntax-curves.svg"`), []byte(`"out_svg": "out.svg"`), 1)
	if err := os.WriteFile(dataPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Generate(dataPath, true); err == nil || !strings.Contains(err.Error(), "drift") {
		t.Fatalf("missing artifact check error = %v", err)
	}
	if _, err := Generate(dataPath, false); err != nil {
		t.Fatal(err)
	}
	if _, err := Generate(dataPath, true); err != nil {
		t.Fatalf("fresh artifact rejected: %v", err)
	}
}
