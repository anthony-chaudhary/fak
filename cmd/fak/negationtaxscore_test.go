package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestNegationTaxScoreJSONAndRatchet(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegationTaxScore(&out, &errb, []string{"--json"})
	if code != 0 && code != 1 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	var got map[string]any
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	corpus := got["corpus"].(map[string]any)
	if _, ok := corpus["negation_tax_debt"]; !ok {
		t.Fatalf("payload=%s", out.String())
	}
	base := filepath.Join(t.TempDir(), "base.json")
	if err := os.WriteFile(base, out.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	if code := runNegationTaxScore(&out, &errb, []string{"--ratchet", base}); code == 2 {
		t.Fatalf("ratchet usage failure: %s", errb.String())
	}
}
