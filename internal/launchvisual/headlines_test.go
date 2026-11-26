package launchvisual

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func loadGoldenBytes(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/headlines.golden.json")
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// mutate replaces the first occurrence of old in the golden index, failing if it is absent
// so a fixture edit cannot silently turn a negative test into a positive one.
func mutate(t *testing.T, old, repl string) []byte {
	t.Helper()
	b := loadGoldenBytes(t)
	if !strings.Contains(string(b), old) {
		t.Fatalf("fixture lacks %q", old)
	}
	return []byte(strings.Replace(string(b), old, repl, 1))
}

// goldenRowCount counts the golden index rows straight from the JSON, so the
// assertion does not reuse ParseIndex to decide what the golden holds.
func goldenRowCount(t *testing.T) int {
	t.Helper()
	var raw struct {
		Rows []json.RawMessage `json:"rows"`
	}
	if err := json.Unmarshal(loadGoldenBytes(t), &raw); err != nil {
		t.Fatal(err)
	}
	return len(raw.Rows)
}

func TestParseGoldenIndex(t *testing.T) {
	ix, err := ParseIndex(loadGoldenBytes(t))
	if err != nil {
		t.Fatalf("golden index refused: %v", err)
	}
	if len(ix.Rows) != goldenRowCount(t) {
		t.Fatalf("want %d rows, got %d", goldenRowCount(t), len(ix.Rows))
	}
}

func TestParseIndexRefusals(t *testing.T) {
	cases := map[string][]byte{
		"unknown field": mutate(t, `"generator":"benchpub/headlines/1"`, `"generator":"benchpub/headlines/1","stray":1`),
		"dotted quad":   mutate(t, `"generator":"benchpub/headlines/1"`, `"generator":"benchpub/1 `+"10"+`.0.0.`+"1"+`"`),
		"missing key":   mutate(t, `,"caveat":""}`, `}`),
		"caveat empty on opt-in row": mutate(t,
			`"caveat":"opt-in; not bit-exact (oracle diverges at token 104 or 111)"`, `"caveat":""`),
		"ratio off by a hundredth": mutate(t, `"ratio":1.43`, `"ratio":1.44`),
		"mean does not match":      mutate(t, `"value":171.535`, `"value":171.6`),
		"row order":                mutate(t, `"id":"batched.c24_f16.agg_tok_s"`, `"id":"zzz.c24_f16.agg_tok_s"`),
		"bad kind":                 mutate(t, `"kind":"loss"`, `"kind":"lost"`),
		"display decimals range":   mutate(t, `"display_decimals":2,"comparator_id":"oss`, `"display_decimals":4,"comparator_id":"oss`),
	}
	for name, b := range cases {
		if _, err := ParseIndex(b); err == nil {
			t.Errorf("%s: want refusal, got none", name)
		}
	}
}

func TestDisplayValueRoundsHalfUp(t *testing.T) {
	cases := []struct {
		v    float64
		d    int
		want string
	}{
		{171.535, 1, "171.5"},
		{333.65, 1, "333.7"},
		{80.98, 1, "81.0"},
		{180.92, 1, "180.9"},
		{0.125, 2, "0.13"},
		{2.5, 0, "3"},
		{205.67, 2, "205.67"},
		{6.32, 2, "6.32"},
	}
	for _, c := range cases {
		if got := DisplayValue(c.v, c.d); got != c.want {
			t.Errorf("DisplayValue(%v,%d)=%q want %q", c.v, c.d, got, c.want)
		}
	}
}

func TestRatioLabelOnlyForNonNullRatio(t *testing.T) {
	ix, err := ParseIndex(loadGoldenBytes(t))
	if err != nil {
		t.Fatal(err)
	}
	byID := ix.ByID()
	if got := byID["spec.d2.tok_s"].RatioLabel(); got != "1.43x" {
		t.Errorf("spec.d2 ratio label %q", got)
	}
	if got := byID["spec.plain.tok_s"].RatioLabel(); got != "" {
		t.Errorf("spec.plain has no ratio, got label %q", got)
	}
}
