package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/disambiguation"
)

func TestDisambiguationSearchReturnsRankedJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runDisambiguation(&out, &errb, []string{"search", "compute k", "--json"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	var got disambiguation.SearchResponse
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != disambiguation.SearchSchemaVersion || got.Verdict != disambiguation.SearchVerdictPrefix {
		t.Fatalf("response = %+v", got)
	}
	if len(got.Groups.Exact) != 0 || len(got.Groups.Alias) != 0 || len(got.Groups.Prefix) == 0 {
		t.Fatalf("ranked groups = %+v", got.Groups)
	}
}

func TestDisambiguationSearchHumanOutputNamesMatchClass(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runDisambiguation(&out, &errb, []string{"search", "fused agent kernel"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	for _, want := range []string{"alias: fused agent kernel", "alias:", "fused agent kernel -> agent kernel"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in:\n%s", want, out.String())
		}
	}
}

func TestDisambiguationSearchReturnsTypedAmbiguity(t *testing.T) {
	var out, errb bytes.Buffer
	code := runDisambiguation(&out, &errb, []string{"search", "kernel", "--json"})
	if code != 3 {
		t.Fatalf("code=%d, want ambiguity exit 3; err=%s out=%s", code, errb.String(), out.String())
	}
	var got disambiguation.SearchResponse
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != disambiguation.SearchVerdictAmbiguous || len(got.Groups.Exact) < 2 {
		t.Fatalf("response = %+v", got)
	}
}
