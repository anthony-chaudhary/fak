package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestDisambiguationExplainHumanOutput(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runDisambiguation(&out, &errb, []string{"explain", "fused agent kernel"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	for _, want := range []string{"agent kernel", "Meaning:", "Not to confuse with:", "Owner:", "Freshness:", "Sources:"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in:\n%s", want, out.String())
		}
	}
}

func TestDisambiguationExplainKeepsStructuredQuerySeparate(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runDisambiguation(&out, &errb, []string{"query", "agent kernel", "--json"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), `"schema": "fak-disambiguation-query/1"`) {
		t.Fatalf("structured query contract changed: %s", out.String())
	}
}
