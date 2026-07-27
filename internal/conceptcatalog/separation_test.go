package conceptcatalog

import (
	"encoding/json"
	"strings"
	"testing"
)

const twinFile = `{
  "note": "keep me",
  "rows": [
    {
      "id": "cache-a",
      "canonical": "Cache A",
      "distinct_from": [
        "cache-b"
      ],
      "aliases": ["A"]
    },
    {"id": "cache-b", "canonical": "Cache B", "distinct_from": ["cache-a"]}
  ]
}
`

func TestBackReferenceKeepsLayoutAndEveryOtherByte(t *testing.T) {
	out, changed, err := addBackReference([]byte(twinFile), "cache-a", "cache-c")
	if err != nil || !changed {
		t.Fatalf("multi-line row: changed=%v err=%v", changed, err)
	}
	// The array it edited keeps one element per line; nothing else moves.
	if !strings.Contains(string(out), "\"distinct_from\": [\n        \"cache-b\",\n        \"cache-c\"\n      ]") {
		t.Fatalf("multi-line layout not preserved:\n%s", out)
	}
	if !strings.Contains(string(out), `"note": "keep me"`) || !strings.Contains(string(out), `"aliases": ["A"]`) {
		t.Fatalf("unrelated bytes were rewritten:\n%s", out)
	}
	// A single-line array stays a single line.
	out2, changed, err := addBackReference(out, "cache-b", "cache-c")
	if err != nil || !changed {
		t.Fatalf("single-line row: changed=%v err=%v", changed, err)
	}
	if !strings.Contains(string(out2), `"distinct_from": ["cache-a", "cache-c"]`) {
		t.Fatalf("single-line layout not preserved:\n%s", out2)
	}
	var check struct {
		Rows []Row `json:"rows"`
	}
	if err = json.Unmarshal(out2, &check); err != nil {
		t.Fatalf("result is not valid JSON: %v", err)
	}
	if len(check.Rows) != 2 || len(check.Rows[0].DistinctFrom) != 2 || len(check.Rows[1].DistinctFrom) != 2 {
		t.Fatalf("references not added: %+v", check.Rows)
	}
}

func TestBackReferenceIsIdempotentAndRefusesUnknownRows(t *testing.T) {
	// Drawing the same boundary twice must not duplicate it - re-running the
	// authoring path over a catalog that already has the reverse half is a no-op.
	out, changed, err := addBackReference([]byte(twinFile), "cache-a", "cache-b")
	if err != nil || changed || string(out) != twinFile {
		t.Fatalf("existing reference re-added: changed=%v err=%v", changed, err)
	}
	if _, _, err = addBackReference([]byte(twinFile), "cache-z", "cache-c"); err == nil {
		t.Fatal("want a refusal for a row this file does not hold")
	}
	noField := []byte(`{"rows": [{"id": "cache-a", "canonical": "Cache A"}]}`)
	if _, _, err = addBackReference(noField, "cache-a", "cache-c"); err == nil {
		t.Fatal("want a refusal when the row has no distinct_from array")
	}
}

func TestUnseparatedPairsAreReportedFromEitherEnd(t *testing.T) {
	var snap shadowSnapshot
	raw := `{"corpus":{"separation":{"unseparated":[
	  {"a":"cache-a","b":"cache-gap","kind":"near","why":"2 edit(s) apart","state":"one_sided"},
	  {"a":"lease-x","b":"lease-y","kind":"near","why":"1 edit apart","state":"undrawn"}]}}}`
	if err := json.Unmarshal([]byte(raw), &snap); err != nil {
		t.Fatal(err)
	}
	// The pair is the author's to pay from EITHER end - the discovery order of a
	// pair is alphabetical, not a statement about which row is new.
	for _, id := range []string{"cache-a", "cache-gap"} {
		miss := snap.unseparatedFor(id)
		if len(miss) != 1 {
			t.Fatalf("%s: want 1 unseparated twin, got %d", id, len(miss))
		}
		other, ok := miss[0].Other(id)
		if !ok || other == id {
			t.Fatalf("%s: bad far end %q", id, other)
		}
	}
	// Debt between concepts the author never touched is not theirs to pay.
	if miss := snap.unseparatedFor("cache-b"); len(miss) != 0 {
		t.Fatalf("unrelated debt charged to the author: %+v", miss)
	}
}
