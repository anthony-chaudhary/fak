package launchvisual

import (
	"strings"
	"testing"
)

func realIndex(t *testing.T) Index {
	t.Helper()
	b, err := repoReader()(DefaultIndex)
	if err != nil {
		t.Fatal(err)
	}
	ix, err := ParseIndex(b)
	if err != nil {
		t.Fatal(err)
	}
	return ix
}

// runDocCheck checks one in-memory doc against the real index with cf's policy.
func runDocCheck(t *testing.T, cf ChartFile, doc string) []Violation {
	t.Helper()
	cf.Docs = []string{"doc.md"}
	read := func(rel string) ([]byte, error) { return []byte(doc), nil }
	vs, err := DocCheck(read, cf, realIndex(t))
	if err != nil {
		t.Fatal(err)
	}
	return vs
}

func chartDrawing(rows ...string) ChartFile {
	p := Panel{Style: "bars"}
	for _, r := range rows {
		p.Series = append(p.Series, Series{Row: r, Label: "x"})
	}
	return ChartFile{Charts: []Chart{{ID: "c", File: "c.svg", Columns: 1, Panels: []Panel{p}}}}
}

func TestDocCheckRejectsReceiptSampleDisplayInProse(t *testing.T) {
	vs := runDocCheck(t, ChartFile{}, "Two runs at 116.9 and 116.1 tok/s.\n")
	if len(vs) != 2 || vs[0].Token != "116.9" || vs[1].Token != "116.1" {
		t.Fatalf("sample displays in prose not refused: %+v", vs)
	}
}

func TestDocCheckAcceptsReceiptSampleDisplayOnCitingRow(t *testing.T) {
	cited := "| fresh | `hl:freshuser.run_decode_tok_s` 116.5 (runs 116.9 and 116.1) |\n"
	if vs := runDocCheck(t, ChartFile{}, cited); len(vs) != 0 {
		t.Errorf("sample displays on the citing row rejected: %+v", vs)
	}
	other := "| fresh | `hl:freshuser.first_token_s` 13.1 (116.9) |\n"
	if vs := runDocCheck(t, ChartFile{}, other); len(vs) != 1 || vs[0].Token != "116.9" {
		t.Errorf("sample display on a row citing another id accepted: %+v", vs)
	}
}

func TestDocCheckUnitRatioNeedsRowAndSuffix(t *testing.T) {
	if vs := runDocCheck(t, ChartFile{}, "Per prompt, 1.09x to 1.35x.\n"); len(vs) != 0 {
		t.Errorf("unit-x index row labels rejected: %+v", vs)
	}
	if vs := runDocCheck(t, ChartFile{}, "Per prompt, 1.10x.\n"); len(vs) != 1 || vs[0].Token != "1.10x" {
		t.Errorf("untraced per-prompt ratio accepted: %+v", vs)
	}
}

func TestDocCheckIntegersNeedUnitOrTableRow(t *testing.T) {
	if vs := runDocCheck(t, ChartFile{}, "The build took 507 s.\n"); len(vs) != 0 {
		t.Errorf("integer display with unit rejected: %+v", vs)
	}
	if vs := runDocCheck(t, ChartFile{}, "The build took 508 s.\n"); len(vs) != 1 || vs[0].Token != "508" {
		t.Errorf("untraced integer with unit accepted: %+v", vs)
	}
	if vs := runDocCheck(t, ChartFile{}, "Twenty-four streams and 127 steps are prose.\n"); len(vs) != 0 {
		t.Errorf("unitless prose integer checked: %+v", vs)
	}
	vs := runDocCheck(t, ChartFile{}, "| c | `hl:samewindow.fak.decode_tok_s` 118.1 | 5 |\n")
	if len(vs) != 1 || vs[0].Token != "5" {
		t.Errorf("unlisted integer in a table row accepted: %+v", vs)
	}
}

func TestDocCheckMasksOnlyNonProseSpans(t *testing.T) {
	cases := map[string]string{
		"rate of 180.8/s":                         "180.8",
		"pair 117.16/110.67 tok/s":                "117.16",
		"ratio 1.46x/2.1 here":                    "2.1",
		"path docs/launch/9.99 stays masked":      "",
		"link [see](docs/notes/7.77.md) only":     "",
		"code `fak run 4.44` is masked, 5.55 not": "5.55",
	}
	for doc, want := range cases {
		vs := runDocCheck(t, ChartFile{}, doc+"\n")
		if want == "" {
			if len(vs) != 0 {
				t.Errorf("%q: masked span flagged: %+v", doc, vs)
			}
			continue
		}
		found := false
		for _, v := range vs {
			if v.Token == want {
				found = true
			}
		}
		if !found {
			t.Errorf("%q: %s not checked: %+v", doc, want, vs)
		}
	}
}

func TestDocCheckRequiresChartedRowCitation(t *testing.T) {
	cf := chartDrawing("samewindow.fak.decode_tok_s")
	vs := runDocCheck(t, cf, "No table here.\n")
	if len(vs) != 1 || vs[0].Token != "hl:samewindow.fak.decode_tok_s" {
		t.Fatalf("uncited charted row not flagged: %+v", vs)
	}
	cited := "| fak | `hl:samewindow.fak.decode_tok_s` 118.1 |\n"
	if vs := runDocCheck(t, cf, cited); len(vs) != 0 {
		t.Errorf("cited charted row flagged: %+v", vs)
	}
}

func TestDocCheckRequiresReferenceRowCitation(t *testing.T) {
	cf := ChartFile{Charts: []Chart{{ID: "c", File: "c.svg", Columns: 1, Panels: []Panel{{
		Style:      "bars",
		Series:     []Series{{Row: "fork.fak.decode_tok_s", Label: "fak"}},
		References: []Reference{{Row: "fork.realugbun.decode_tok_s", Label: "fork"}},
	}}}}}
	vs := runDocCheck(t, cf, "| fak | `hl:fork.fak.decode_tok_s` 119.9 |\n")
	if len(vs) != 1 || vs[0].Token != "hl:fork.realugbun.decode_tok_s" {
		t.Fatalf("uncited reference row not flagged: %+v", vs)
	}
}

func TestDocCheckSupersededTokenOnlyOnSupersededLine(t *testing.T) {
	cf := ChartFile{SupersededTokens: []string{"151.7"}}
	if vs := runDocCheck(t, cf, "Speculative decode reaches 151.7 tok/s.\n"); len(vs) != 1 || vs[0].Token != "151.7" {
		t.Errorf("bare superseded token not flagged: %+v", vs)
	}
	if vs := runDocCheck(t, cf, "Old speculative figure 151.7 is superseded by the new receipt.\n"); len(vs) != 0 {
		t.Errorf("superseded token on a superseded line flagged: %+v", vs)
	}
}

func TestChartFileRowIDsIncludesReferences(t *testing.T) {
	cf := ChartFile{Charts: []Chart{{Panels: []Panel{
		{Series: []Series{{Row: "a.b.c"}}, References: []Reference{{Row: "d.e.f"}}},
		{Series: []Series{{Row: "a.b.c"}}},
	}}}}
	got := strings.Join(cf.RowIDs(), ",")
	if got != "a.b.c,d.e.f" {
		t.Fatalf("RowIDs = %q, want a.b.c,d.e.f", got)
	}
}
