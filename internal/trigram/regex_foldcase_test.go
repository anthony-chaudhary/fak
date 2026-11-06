package trigram

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestSearchRegexpMandatoryHintsPreserveMatches(t *testing.T) {
	t.Parallel()
	ix := &Index{}
	ix.Add("lower", "lower.go", "header\nfoo\nprefixpost\nkelvin\n")
	ix.Add("mixed", "mixed.go", "header\nFoO\npreFIXpost\nKELVIN\n")
	ix.Add("upper", "upper.go", "header\nFOO\nPREFIXPOST\nKELVIN\n")
	ix.Add("decoy", "decoy.go", "header\nfood\npreFixPOST\nkelvins\n")

	for _, tc := range []struct {
		pattern      string
		wantIDs      []string
		wantLine     int
		wantLiterals []string
	}{
		{`(?i)^foo$`, []string{"lower", "mixed", "upper"}, 2, nil},
		{`^pre(?i:fix)post$`, []string{"lower", "mixed"}, 3, []string{"pre", "post"}},
		{`(?i)^kelvin$`, []string{"lower", "mixed", "upper"}, 4, nil},
		{`^foo$`, []string{"lower"}, 2, []string{"foo"}},
	} {
		t.Run(tc.pattern, func(t *testing.T) {
			hits, err := ix.SearchRegexp(tc.pattern)
			if err != nil {
				t.Fatalf("SearchRegexp(%q): %v", tc.pattern, err)
			}
			if got := ids(hits); !reflect.DeepEqual(got, tc.wantIDs) {
				t.Fatalf("SearchRegexp(%q) IDs = %v, want %v", tc.pattern, got, tc.wantIDs)
			}
			for _, hit := range hits {
				if hit.Path != hit.ID+".go" || !reflect.DeepEqual(hit.Lines, []int{tc.wantLine}) {
					t.Errorf("SearchRegexp(%q) hit = %+v, want original path and line %d", tc.pattern, hit, tc.wantLine)
				}
			}
			if got := requiredLiterals(tc.pattern); !reflect.DeepEqual(got, tc.wantLiterals) {
				t.Errorf("requiredLiterals(%q) = %v, want %v", tc.pattern, got, tc.wantLiterals)
			}
		})
	}

	// The old hint abcFOO happens to select target via rare bcF/cFO postings.
	// Dropping only the folded suffix leaves abc, which wrongly excludes it.
	plus := &Index{}
	plus.Add("target", "target.go", "abbbcfoo\nbcFO")
	plus.Add("decoy-a", "decoy-a.go", "abc\nFOO")
	plus.Add("decoy-b", "decoy-b.go", "abc\nFOO")
	for _, tc := range []struct {
		pattern      string
		wantLiterals []string
	}{
		{`^ab+c(?i:foo)$`, nil},
		{`^(ab+)+c(?i:foo)$`, nil},
		{`^ab+cfoo$`, []string{"cfoo"}},
		{`^(abbbc)+(?i:foo)$`, []string{"abbbc"}},
	} {
		t.Run(tc.pattern, func(t *testing.T) {
			hits, err := plus.SearchRegexp(tc.pattern)
			if err != nil {
				t.Fatalf("SearchRegexp(%q): %v", tc.pattern, err)
			}
			want := []Result{{ID: "target", Path: "target.go", Lines: []int{1}}}
			if !reflect.DeepEqual(hits, want) {
				t.Fatalf("SearchRegexp(%q) = %+v, want %+v", tc.pattern, hits, want)
			}
			if got := requiredLiterals(tc.pattern); !reflect.DeepEqual(got, tc.wantLiterals) {
				t.Errorf("requiredLiterals(%q) = %v, want %v", tc.pattern, got, tc.wantLiterals)
			}
		})
	}
	t.Run("top-level plus", func(t *testing.T) {
		const pattern = `^(abc)+$`
		hits, err := plus.SearchRegexp(pattern)
		want := []Result{
			{ID: "decoy-a", Path: "decoy-a.go", Lines: []int{1}},
			{ID: "decoy-b", Path: "decoy-b.go", Lines: []int{1}},
		}
		if err != nil || !reflect.DeepEqual(hits, want) {
			t.Fatalf("SearchRegexp(%q) = %+v, %v; want %+v", pattern, hits, err, want)
		}
		if got := requiredLiterals(pattern); !reflect.DeepEqual(got, []string{"abc"}) {
			t.Errorf("requiredLiterals(%q) = %v, want [abc]", pattern, got)
		}
	})
}
