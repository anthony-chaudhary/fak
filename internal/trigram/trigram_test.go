package trigram

import (
	"reflect"
	"sort"
	"testing"
)

func buildIndex() *Index {
	ix := &Index{}
	ix.Add("a", "auth.go", "func authenticate(user string) error {\n\treturn verifyToken(user)\n}")
	ix.Add("b", "log.go", "func rotateLog(path string) {\n\tcompress(path)\n}")
	ix.Add("c", "dispatch.go", "func dispatchWave(w Wave) {\n\troute(w)\n}")
	ix.Add("d", "empty.go", "package x\n")
	return ix
}

func ids(rs []Result) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	sort.Strings(out)
	return out
}

// TestSearchLiteral verifies the candidate pre-filter is a SUPERSET of the exact
// matches and the verify pass narrows it to the truth.
func TestSearchLiteral(t *testing.T) {
	ix := buildIndex()
	hits := ix.Search("verifyToken")
	if got := ids(hits); !reflect.DeepEqual(got, []string{"a"}) {
		t.Fatalf("Search(verifyToken) = %v, want [a]", got)
	}
	if len(hits) == 1 && !reflect.DeepEqual(hits[0].Lines, []int{2}) {
		t.Errorf("verifyToken lines = %v, want [2]", hits[0].Lines)
	}

	// The candidate set must contain every true match (soundness).
	cand := ix.Candidates("verifyToken")
	found := false
	for _, id := range cand {
		if ix.docs[id].id == "a" {
			found = true
		}
	}
	if !found {
		t.Errorf("candidate set %v omitted the true match doc a", cand)
	}
}

// TestFreqZeroShortCircuit is the Code Search optimization: a literal carrying a
// trigram that appears in no document yields zero candidates without scanning.
func TestFreqZeroShortCircuit(t *testing.T) {
	ix := buildIndex()
	if cand := ix.Candidates("zzzQQQxyz"); cand != nil {
		t.Errorf("absent literal returned candidates %v, want nil (short-circuit)", cand)
	}
	if hits := ix.Search("zzzQQQxyz"); len(hits) != 0 {
		t.Errorf("absent literal matched %v, want none", hits)
	}
}

// TestShortLiteralFallsBack: a <3-rune literal can't be indexed, so every doc is a
// candidate and the verify pass still returns the exact matches.
func TestShortLiteralFallsBack(t *testing.T) {
	ix := buildIndex()
	if got := len(ix.Candidates("fn")); got != ix.DocCount() {
		t.Errorf("short literal candidates = %d, want all %d docs", got, ix.DocCount())
	}
	hits := ix.Search("w)") // appears only in dispatch.go: route(w)
	if got := ids(hits); !reflect.DeepEqual(got, []string{"c"}) {
		t.Errorf("Search(w)) = %v, want [c]", got)
	}
}

// TestSearchRegexp narrows by required literals then verifies with the compiled
// regexp, and stays sound when the pattern has no required literal.
func TestSearchRegexp(t *testing.T) {
	ix := buildIndex()

	// `authenticate` is a required literal of this pattern -> only doc a is a
	// candidate, and it matches.
	hits, err := ix.SearchRegexp(`authenticate\(`)
	if err != nil {
		t.Fatalf("SearchRegexp: %v", err)
	}
	if got := ids(hits); !reflect.DeepEqual(got, []string{"a"}) {
		t.Fatalf("regexp authenticate\\( = %v, want [a]", got)
	}

	// `func\s+\w+` has no required literal >=3 runes except "func"; every func-doc
	// must be returned (soundness across a, b, c).
	hits, err = ix.SearchRegexp(`func\s+\w+`)
	if err != nil {
		t.Fatalf("SearchRegexp func: %v", err)
	}
	if got := ids(hits); !reflect.DeepEqual(got, []string{"a", "b", "c"}) {
		t.Errorf("regexp func = %v, want [a b c]", got)
	}
}

// TestRequiredLiteralsSoundness pins the extractor: it emits a literal only when
// every match must contain it, and never from an alternation or a starred group.
func TestRequiredLiteralsSoundness(t *testing.T) {
	cases := []struct {
		pattern string
		want    []string
	}{
		{`authenticate\(`, []string{"authenticate("}}, // escaped paren folds into the literal run
		{`foo(bar)?`, []string{"foo"}},                // bar is optional -> not required
		{`(cat|dog)house`, []string{"house"}},         // alternation contributes nothing
		{`ab*cdef`, []string{"cdef"}},                 // b* breaks "ab", cdef survives
		{`ident+`, []string{"ident"}},                 // x+ requires at least one -> "ident" required
		{`.*`, nil},                                   // no required literal
	}
	for _, c := range cases {
		got := requiredLiterals(c.pattern)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("requiredLiterals(%q) = %v, want %v", c.pattern, got, c.want)
		}
	}
}

// TestRegexpBruteForceStillCorrect: a wildcard-led pattern with no usable literal
// falls back to scanning every doc and still finds the true match.
func TestRegexpBruteForceStillCorrect(t *testing.T) {
	ix := buildIndex()
	hits, err := ix.SearchRegexp(`ro.te`) // matches "rotate"? no — matches "route" in c
	if err != nil {
		t.Fatalf("SearchRegexp: %v", err)
	}
	// ro.te matches "route" (dispatch.go, doc c). rotateLog is "rotate" (ro t a te? no)
	if got := ids(hits); !reflect.DeepEqual(got, []string{"c"}) {
		t.Errorf("regexp ro.te = %v, want [c]", got)
	}
}
