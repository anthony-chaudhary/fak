package launchvisual

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	fixtureData  = "internal/launchvisual/testdata/charts.data.json"
	fixtureIndex = "internal/launchvisual/testdata/headlines.golden.json"
)

// repoReader reads from the repository root, two levels above this package.
func repoReader() ReadFunc { return DiskReader(filepath.Join("..", "..")) }

// overlay serves override bytes for the given paths and the disk for everything else.
func overlay(over map[string][]byte) ReadFunc {
	base := repoReader()
	return func(rel string) ([]byte, error) {
		if b, ok := over[rel]; ok {
			return b, nil
		}
		return base(rel)
	}
}

func renderFixture(t *testing.T) map[string]string {
	t.Helper()
	rendered, _, _, err := Render(repoReader(), fixtureData, fixtureIndex)
	if err != nil {
		t.Fatalf("render fixture: %v", err)
	}
	out := map[string]string{}
	for _, r := range rendered {
		out[filepath.Base(r.Rel)] = string(r.SVG)
	}
	return out
}

func TestCommittedFixtureSVGsAreCurrent(t *testing.T) {
	rep, err := Verify(repoReader(), fixtureData, fixtureIndex)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Drift) != 0 {
		t.Fatalf("fixture SVGs drifted: %v", rep.Drift)
	}
	if len(rep.Checked) != 2 {
		t.Fatalf("want 2 charts checked, got %v", rep.Checked)
	}
}

func TestRenderIsDeterministic(t *testing.T) {
	a, _, _, err := Render(repoReader(), fixtureData, fixtureIndex)
	if err != nil {
		t.Fatal(err)
	}
	b, _, _, err := Render(repoReader(), fixtureData, fixtureIndex)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if !bytes.Equal(a[i].SVG, b[i].SVG) {
			t.Fatalf("render of %s differs between runs", a[i].Rel)
		}
	}
}

func TestRenderDrawsIndexDisplayValuesAndRatio(t *testing.T) {
	svg := renderFixture(t)["inkling2-decode-comparison.svg"]
	for _, want := range []string{">171.5<", ">180.9<", ">81.0<", ">118.1<", ">1.43x<"} {
		if !strings.Contains(svg, want) {
			t.Errorf("comparison SVG lacks %s", want)
		}
	}
	if n := strings.Count(svg, "x</text>"); n != 2 {
		t.Errorf("ratio label drawn %d times; want exactly the two speculative rows, which carry ratios", n)
	}
	if !strings.Contains(svg, ">1.51x<") {
		t.Error("spec.d4 ratio label missing")
	}
}

func TestLossRowUsesLossColourAndCaveatFooter(t *testing.T) {
	svg := renderFixture(t)["inkling2-decode-climb.svg"]
	if !strings.Contains(svg, `fill="`+colorLoss+`"`) {
		t.Error("loss row not drawn in the loss colour")
	}
	if !strings.Contains(svg, "below OSS llama-server") {
		t.Error("loss row caveat footer missing")
	}
	comparison := renderFixture(t)["inkling2-decode-comparison.svg"]
	if !strings.Contains(comparison, "not bit-exact") {
		t.Error("opt-in caveat footer missing from comparison chart")
	}
}

func TestFitTextRefusesOverflow(t *testing.T) {
	if err := fitText("abcdefghijk", 10, 5.8); err == nil {
		t.Error("overflowing label accepted")
	}
	if err := fitText("ab", 12, 5.8); err != nil {
		t.Errorf("fitting label refused: %v", err)
	}
}

func TestChartDataRefusesDecimalLabelAndUnknownField(t *testing.T) {
	base, err := os.ReadFile(filepath.Join("..", "..", fixtureData))
	if err != nil {
		t.Fatal(err)
	}
	withDecimal := strings.Replace(string(base), `"title": "Decode fixture, tok/s"`, `"title": "Decode 118.1 fixture"`, 1)
	if _, err := ParseChartFile([]byte(withDecimal)); err == nil {
		t.Error("decimal number in a chart label accepted")
	}
	withField := strings.Replace(string(base), `"columns": 3,`, `"columns": 3, "numbers": [1.5],`, 1)
	if _, err := ParseChartFile([]byte(withField)); err == nil {
		t.Error("unknown field in the data file accepted")
	}
}

func TestRenderRefusesUnknownRowID(t *testing.T) {
	base, err := os.ReadFile(filepath.Join("..", "..", fixtureData))
	if err != nil {
		t.Fatal(err)
	}
	bad := strings.Replace(string(base), "spec.plain.tok_s", "spec.nope.tok_s", 1)
	read := overlay(map[string][]byte{fixtureData: []byte(bad)})
	if _, _, _, err := Render(read, fixtureData, fixtureIndex); err == nil {
		t.Error("chart referencing an unknown row rendered")
	}
}

func TestGenerateSVGDriftIsReported(t *testing.T) {
	svgRel := "internal/launchvisual/testdata/inkling2-decode-comparison.svg"
	read := overlay(map[string][]byte{svgRel: []byte("<svg>stale</svg>")})
	rep, err := Verify(read, fixtureData, fixtureIndex)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Drift) != 1 || rep.Drift[0] != svgRel {
		t.Fatalf("drift %v, want only %s", rep.Drift, svgRel)
	}
	if err := rep.Err(); err == nil || !strings.Contains(err.Error(), "launch visual drift") {
		t.Fatalf("drift error %v does not name the drift", err)
	}
}

func TestFindRootFindsGoMod(t *testing.T) {
	root, err := FindRoot(filepath.Join("..", "..", "internal", "launchvisual"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		t.Fatalf("root %s has no go.mod: %v", root, err)
	}
}

func TestFooterLinesDedupesAndKeepsSentencesWhole(t *testing.T) {
	got := footerLines([]string{"Model line. Shared caveat.", "Shared caveat. Own caveat."})
	want := []string{"Model line. Shared caveat. Own caveat."}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("footerLines = %q, want %q", got, want)
	}
	long := strings.Repeat("word ", 40) + "end."
	lines := footerLines([]string{"Short. " + long})
	if len(lines) < 3 || lines[0] != "Short." {
		t.Fatalf("long sentence not isolated after a short one: %q", lines)
	}
}
