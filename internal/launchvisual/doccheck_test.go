package launchvisual

import (
	"fmt"
	"strings"
	"testing"
)

func goldenIndex(t *testing.T) Index {
	t.Helper()
	ix, err := ParseIndex(loadGoldenBytes(t))
	if err != nil {
		t.Fatal(err)
	}
	return ix
}

// docViolations runs DocCheck on one in-memory doc with the given allowed-number list.
func docViolations(t *testing.T, doc string, allowed ...AllowedNumber) []Violation {
	t.Helper()
	cf := ChartFile{Schema: ChartSchema, Docs: []string{"doc.md"}, AllowedNumbers: allowed}
	read := func(rel string) ([]byte, error) {
		if rel != "doc.md" {
			return nil, fmt.Errorf("unexpected read %s", rel)
		}
		return []byte(doc), nil
	}
	vs, err := DocCheck(read, cf, goldenIndex(t))
	if err != nil {
		t.Fatal(err)
	}
	return vs
}

func TestDocCheckFixturePasses(t *testing.T) {
	rep, err := Verify(repoReader(), fixtureData, fixtureIndex)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Violations) != 0 {
		t.Fatalf("fixture doc has violations: %+v", rep.Violations)
	}
}

func TestDocCheckFlagsSupersededNumber(t *testing.T) {
	vs := docViolations(t, "Decode reaches 999.9 tok/s.\n")
	if len(vs) != 1 || vs[0].Token != "999.9" || vs[0].Line != 1 {
		t.Fatalf("want one violation for 999.9 on line 1, got %+v", vs)
	}
}

func TestDocCheckRatioNeedsNxSuffix(t *testing.T) {
	if vs := docViolations(t, "A 1.43x ratio.\n"); len(vs) != 0 {
		t.Errorf("headline ratio with x suffix rejected: %+v", vs)
	}
	if vs := docViolations(t, "A 1.43 ratio.\n"); len(vs) != 1 {
		t.Errorf("bare 1.43 accepted as a ratio: %+v", vs)
	}
}

func TestDocCheckTableCellMustShowRowValue(t *testing.T) {
	good := "| stock | `hl:samewindow.stock.decode_tok_s` 81.0 |\n"
	bad := "| stock | `hl:samewindow.stock.decode_tok_s` 80.98 |\n"
	if vs := docViolations(t, good); len(vs) != 0 {
		t.Errorf("matching hl cell rejected: %+v", vs)
	}
	found := false
	for _, v := range docViolations(t, bad) {
		if strings.Contains(v.Reason, "must show 81.0") {
			found = true
		}
	}
	if !found {
		t.Error("hl cell with a stale value accepted")
	}
}

func TestDocCheckUnknownHLRowRefused(t *testing.T) {
	vs := docViolations(t, "| x | `hl:nope.row.tok_s` 1 |\n")
	if len(vs) == 0 || !strings.Contains(vs[0].Reason, "unknown headline row") {
		t.Fatalf("unknown hl id not refused: %+v", vs)
	}
}

func TestDocCheckSkipsURLsPathsVersionsAndFences(t *testing.T) {
	doc := strings.Join([]string{
		"See https://example.com/9.99 and docs/notes/1.25.md and v1.2.3 for details.",
		"```",
		"a fenced figure 7.77 is not prose",
		"```",
	}, "\n")
	if vs := docViolations(t, doc); len(vs) != 0 {
		t.Errorf("non-prose numbers flagged: %+v", vs)
	}
}

func TestDocCheckAllowedNumberPasses(t *testing.T) {
	doc := "The build took 12.5 seconds.\n"
	if vs := docViolations(t, doc); len(vs) != 1 {
		t.Fatalf("unlisted build seconds should be flagged, got %+v", vs)
	}
	if vs := docViolations(t, doc, AllowedNumber{Token: "12.5", Reason: "build seconds"}); len(vs) != 0 {
		t.Errorf("allowed number flagged: %+v", vs)
	}
}
