package issuefanout

import "testing"

func TestFileLiveRefusesMissingProjectDenominator(t *testing.T) {
	p, err := Build(Input{Title: "x", Leaf: "x", SpineRef: "abc", Max: MinFanout})
	if err != nil {
		t.Fatal(err)
	}
	_, err = FileLive(p, nil, LiveOptions{Runner: func([]string) (string, string, bool) { t.Fatal("runner called"); return "", "", false }})
	if err == nil {
		t.Fatal("missing denominator accepted")
	}
}
