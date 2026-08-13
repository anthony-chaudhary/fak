package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRunMicroCollapseMeasuresSavedParentContext(t *testing.T) {
	r, err := runMicroCollapse(3, 2048)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "PASS" || r.Allowed != 3 || r.Denied != 0 || r.Errored != 0 || r.JournalRows != 3 {
		t.Fatalf("receipt=%+v", r)
	}
	if r.IntermediateTokens <= r.FoldedTokens || r.SavedTokens != r.IntermediateTokens-r.FoldedTokens {
		t.Fatalf("context accounting=%+v", r)
	}
}

func TestMicroCollapseJSONIsCapturedReceipt(t *testing.T) {
	r, err := runMicroCollapse(2, 1024)
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(r); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"schema":"fak-micro-collapse/1"`, `"verdict":"PASS"`, `"saved_tokens":`} {
		if !strings.Contains(b.String(), want) {
			t.Fatalf("receipt missing %s: %s", want, b.String())
		}
	}
}
