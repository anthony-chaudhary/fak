package breath

import (
	"strings"
	"testing"
)

func f(k Kind, p string) Finding { return Finding{Kind: k, Path: p, Detail: "x"} }

// TestKeyIsStableUnderEditing pins clause (a): the ratchet key must not move when the page
// is edited around the finding. A key that embedded a line number would renumber on every
// inserted sentence and the floor would be meaningless.
func TestKeyIsStableUnderEditing(t *testing.T) {
	a := Finding{Kind: BreathSentenceLength, Path: "docs/explainers/x.md", Line: 7, Detail: "sentence 1"}
	b := Finding{Kind: BreathSentenceLength, Path: "docs/explainers/x.md", Line: 99, Detail: "sentence 3"}
	if a.Key() != b.Key() {
		t.Fatalf("Key must ignore line and detail: %q vs %q", a.Key(), b.Key())
	}
	if !strings.Contains(a.Key(), "\t") {
		t.Errorf("Key must be the tab-joined KIND<TAB>path baseline key, got %q", a.Key())
	}
}

// TestBaselineParserHardFailsOnAMalformedRow is clause (c), and it is the clause worth a
// test of its own: a lenient parser that SKIPPED a malformed row would read as a floor of
// zero for that key, so this package's own bug would surface as a fresh finding in someone
// else's diff. The assertion is on the error, never on a skip.
func TestBaselineParserHardFailsOnAMalformedRow(t *testing.T) {
	for _, tc := range []struct{ name, body, wantIn string }{
		{"two fields", "BREATH_MISSING\tdocs/explainers/x.md\n", "line 1"},
		{"four fields", "BREATH_MISSING\tdocs/explainers/x.md\t1\textra\n", "line 1"},
		{"count not a number", "BREATH_MISSING\tdocs/explainers/x.md\tmany\n", "non-negative integer"},
		{"negative count", "BREATH_MISSING\tdocs/explainers/x.md\t-1\n", "non-negative integer"},
		{"empty path", "BREATH_MISSING\t\t1\n", "empty kind or path"},
		{"unknown kind", "BREATH_VIBES\tdocs/explainers/x.md\t1\n", "not a breath finding kind"},
		{"malformed after good rows", "BREATH_MISSING\ta.md\t1\nBREATH_MISSING\tb.md\n", "line 2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseBaseline(strings.NewReader(tc.body))
			if err == nil {
				t.Fatalf("want a hard error, got baseline %v — a skipped malformed row reads as a "+
					"floor of zero and turns this package's bug into someone else's denial", got)
			}
			if !strings.Contains(err.Error(), tc.wantIn) {
				t.Errorf("error must name %q; got %v", tc.wantIn, err)
			}
		})
	}
}

func TestBaselineParserSkipsCommentsAndBlanks(t *testing.T) {
	b, err := ParseBaseline(strings.NewReader(
		"# a comment\n\n   \nBREATH_MISSING\tdocs/explainers/x.md\t2\n# trailing note\n"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got, want := b["BREATH_MISSING\tdocs/explainers/x.md"], 2; got != want {
		t.Fatalf("floor = %d, want %d", got, want)
	}
	if len(b) != 1 {
		t.Errorf("baseline = %v, want exactly one key", b)
	}
	// The header this package emits must round-trip through its own parser.
	if _, err := ParseBaseline(strings.NewReader(BaselineHeader)); err != nil {
		t.Errorf("the emitted baseline header does not parse: %v", err)
	}
}

// TestRatchetReportsOnlyAboveTheFloor is clause (b) and (d): counts, and "not growing".
func TestRatchetReportsOnlyAboveTheFloor(t *testing.T) {
	base := Baseline{"BREATH_SENTENCE_LENGTH\tdocs/explainers/x.md": 2}
	two := []Finding{f(BreathSentenceLength, "docs/explainers/x.md"), f(BreathSentenceLength, "docs/explainers/x.md")}
	if got := Ratchet(two, base); len(got) != 0 {
		t.Fatalf("two findings against a floor of two must be silent, got %+v", got)
	}
	three := append(append([]Finding{}, two...), f(BreathSentenceLength, "docs/explainers/x.md"))
	if got := Ratchet(three, base); len(got) != 1 {
		t.Fatalf("a third finding must be reported as NEW, got %d", len(got))
	}
	// Fixing one of two tightens the floor on regeneration — the property a count buys
	// and a per-line pin does not.
	one := two[:1]
	if got := Ratchet(one, base); len(got) != 0 {
		t.Fatalf("one finding against a floor of two must be silent, got %+v", got)
	}
	regen, err := ParseBaseline(strings.NewReader(FormatBaseline(one)))
	if err != nil {
		t.Fatalf("regenerated baseline does not parse: %v", err)
	}
	if got, want := regen["BREATH_SENTENCE_LENGTH\tdocs/explainers/x.md"], 1; got != want {
		t.Fatalf("regenerated floor = %d, want %d (fixing one of two must tighten it)", got, want)
	}
	if got := Ratchet(two, regen); len(got) != 1 {
		t.Errorf("the tightened floor must catch the regression, got %d", len(got))
	}
	// A finding on a key with no floor at all is NEW.
	if got := Ratchet([]Finding{f(BreathMissing, "docs/explainers/new.md")}, base); len(got) != 1 {
		t.Errorf("an unbaselined key must report, got %d", len(got))
	}
}

// TestScanFloorIsNeverRatchetedAway: a run that examined too little to have an opinion is
// a defect in the GATE, not grandfathered doc debt, so no baseline entry may silence it.
func TestScanFloorIsNeverRatchetedAway(t *testing.T) {
	floor := Finding{Kind: BreathScanFloor, Path: "internal/promptlint/breath", Detail: "starved"}
	base := Baseline{floor.Key(): 99}
	if got := Ratchet([]Finding{floor}, base); len(got) != 1 {
		t.Fatalf("BREATH_SCAN_FLOOR must survive any baseline, got %+v", got)
	}
}

func TestFormatBaselineIsSortedAndStable(t *testing.T) {
	in := []Finding{
		f(BreathMissing, "docs/explainers/z.md"),
		f(BreathSentenceCount, "docs/explainers/a.md"),
		f(BreathMissing, "docs/explainers/a.md"),
		f(BreathMissing, "docs/explainers/a.md"),
	}
	out := FormatBaseline(in)
	if a, b := FormatBaseline(in), out; a != b {
		t.Fatal("FormatBaseline is not deterministic")
	}
	rows := []string{}
	for _, l := range strings.Split(out, "\n") {
		if l != "" && !strings.HasPrefix(l, "#") {
			rows = append(rows, l)
		}
	}
	want := []string{
		"BREATH_MISSING\tdocs/explainers/a.md\t2",
		"BREATH_MISSING\tdocs/explainers/z.md\t1",
		"BREATH_SENTENCE_COUNT\tdocs/explainers/a.md\t1",
	}
	if strings.Join(rows, "|") != strings.Join(want, "|") {
		t.Errorf("rows =\n%v\nwant\n%v", rows, want)
	}
}

func TestInScopeMatchesOnAPathBoundary(t *testing.T) {
	c := DefaultContract()
	for _, tc := range []struct {
		rel  string
		want bool
	}{
		{"docs/explainers/x.md", true},
		{"docs/explainers/caching/level-1.md", true},
		{"docs/explainers-archive/x.md", false}, // prefix, but not a path boundary
		{"docs/explainers/x.txt", false},
		{"README.md", false},
	} {
		if got := c.InScope(tc.rel); got != tc.want {
			t.Errorf("InScope(%q) = %v, want %v", tc.rel, got, tc.want)
		}
	}
}
