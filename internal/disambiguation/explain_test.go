package disambiguation

import (
	"strings"
	"testing"
)

func TestExplainGoldenIncludesHumanDistinctionFields(t *testing.T) {
	result, err := Resolve("fused agent kernel")
	if err != nil {
		t.Fatal(err)
	}
	got := Explain(result)
	wants := []string{
		"agent kernel\n",
		"Meaning: The fak management boundary",
		"Matched alias: fused agent kernel",
		"Not to confuse with:\n- compute kernel -",
		"Owner: kernel (lane kernel)",
		"Freshness: fresh (SOURCE_CURRENT)",
		"Sources:\n- document: README.md#how-it-works @",
	}
	for _, want := range wants {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in render:\n%s", want, got)
		}
	}
}
