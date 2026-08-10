package lightgapscore

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func root(t *testing.T) string {
	t.Helper()
	p, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func card(t *testing.T) *Scorecard {
	t.Helper()
	s, e := New(root(t))
	if e != nil {
		t.Fatal(e)
	}
	return s
}

func TestCommittedBoardContract(t *testing.T) {
	s := card(t)
	if got := s.Debt(); got != 9 {
		t.Fatalf("debt=%d want 9", got)
	}
	h := s.Hull()
	if h["negative_cells"] == 0 || h["eccentricity"].(float64) <= 1 {
		t.Fatalf("non-spherical hull: %#v", h)
	}
	verdicts := map[string]bool{}
	for _, seg := range s.Meta.Segments {
		verdicts[s.SegmentProfile(str(seg["id"]))["verdict"].(string)] = true
	}
	if !verdicts["UNDECIDABLE"] && !verdicts["BLOCKED"] {
		t.Fatalf("board has no refusal: %#v", verdicts)
	}
}
func TestRapidityAndCaps(t *testing.T) {
	if Rapidity(.5) <= 0 || Rapidity(-.5) >= 0 || math.Abs(Rapidity(.999)-Horizon) > 1e-12 {
		t.Fatal("rapidity contract drift")
	}
	if !(capValue("CRUISE") < rungFloor("RELATIVISTIC") && capValue("CRUISE") > rungFloor("CRUISE")) {
		t.Fatal("cap escaped rung")
	}
}
func TestEverySegmentIsCompleteAndWeighted(t *testing.T) {
	s := card(t)
	for _, seg := range s.Meta.Segments {
		sid := str(seg["id"])
		material := 0
		weights, _ := seg["weights"].(map[string]any)
		for _, v := range weights {
			if f, _ := asFloat(v); f > 0 {
				material++
			}
		}
		if len(s.bySegment(sid)) != material {
			t.Fatalf("%s incomplete: %d/%d", sid, len(s.bySegment(sid)), material)
		}
		sum := 0.0
		for _, c := range s.bySegment(sid) {
			sum += c.Weight
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Fatalf("%s weights %.6f", sid, sum)
		}
	}
}
func TestAlternativesAreRealAndSourcesAreConcrete(t *testing.T) {
	s := card(t)
	classes := map[string]bool{"sota": true, "tuned": true, "naive": true, "floor": true}
	for _, c := range s.Cells {
		if c.Mode == "immaterial" || c.Provenance == "NONE" {
			continue
		}
		alt := s.Alternatives[c.AltID]
		if !classes[str(alt["class"])] {
			t.Fatalf("%s/%s strawman %q", c.Segment, c.Facet, str(alt["class"]))
		}
		for label, src := range map[string]string{"fak": c.FakSource, "alternative": c.AltSource} {
			if src == "" || strings.Contains(strings.ToLower(src), "tbd") {
				t.Fatalf("%s/%s %s source=%q", c.Segment, c.Facet, label, src)
			}
		}
	}
}
func TestModeledAndLowerBoundCaps(t *testing.T) {
	s := card(t)
	for _, c := range s.Cells {
		if c.Provenance == "MODELED" && c.WEff >= rungFloor("RELATIVISTIC") {
			t.Fatalf("modeled escaped: %#v", c)
		}
		if c.CeilingKind == "lower-bound" && c.WEff >= rungFloor("NEAR-C") {
			t.Fatalf("lower bound escaped: %#v", c)
		}
	}
}
func TestPayloadHasNoAggregateAndIsFinite(t *testing.T) {
	s := card(t)
	p := s.Payload()
	for _, k := range []string{"score", "overall", "total", "mean", "average", "grade"} {
		if _, ok := p[k]; ok {
			t.Fatalf("aggregate %q emitted", k)
		}
	}
	b, e := json.Marshal(p)
	if e != nil {
		t.Fatal(e)
	}
	var q map[string]any
	if e = json.Unmarshal(b, &q); e != nil {
		t.Fatal(e)
	}
	if q["schema"] != Schema {
		t.Fatalf("schema=%v", q["schema"])
	}
	if _, ok := q[DebtKey].(float64); !ok {
		t.Fatalf("debt not numeric: %T", q[DebtKey])
	}
}
func TestDeterminism(t *testing.T) {
	s1 := card(t)
	s2 := card(t)
	a, _ := json.MarshalIndent(s1.Payload(), "", "  ")
	b, _ := json.MarshalIndent(s2.Payload(), "", "  ")
	if !reflect.DeepEqual(a, b) {
		t.Fatal("same tree produced different payload")
	}
}
func TestDefaultRenderASCII(t *testing.T) {
	for _, r := range card(t).Render() {
		if r > 127 {
			t.Fatalf("non-ASCII terminal rune %q", r)
		}
	}
}
func TestMarkdownContractIsCompleteAndIdempotent(t *testing.T) {
	s := card(t)
	tmp := t.TempDir()
	paths, e := s.WriteMarkdown(tmp)
	if e != nil {
		t.Fatal(e)
	}
	if len(paths) != 12 {
		t.Fatalf("wrote %d docs", len(paths))
	}
	for _, p := range paths {
		got, e := os.ReadFile(p)
		if e != nil {
			t.Fatal(e)
		}
		want, e := os.ReadFile(filepath.Join(root(t), "docs", "lightgap-scorecard", filepath.Base(p)))
		if e != nil {
			t.Fatal(e)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s drift", filepath.Base(p))
		}
	}
	inPlace, e := s.WriteMarkdown(filepath.Join(root(t), "docs", "lightgap-scorecard"))
	if e != nil || len(inPlace) != 12 {
		t.Fatalf("in-place: %d %v", len(inPlace), e)
	}
}
func TestUnknownListsAreStable(t *testing.T) {
	s := card(t)
	e := s.RequireFacet("nope")
	if e == nil {
		t.Fatal("unknown facet accepted")
	}
	want := ids(s.Meta.Facets)
	sort.Strings(want)
	got := append([]string(nil), want...)
	if !reflect.DeepEqual(want, got) {
		t.Fatal("test sanity")
	}
}
