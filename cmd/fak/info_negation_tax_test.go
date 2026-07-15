package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestInfoNegationTaxPane(t *testing.T) {
	var out, errb bytes.Buffer
	code := runInfo(&out, &errb, []string{"--negation-tax", "--negation-tax-top", "0"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "negation-tax debt=") {
		t.Fatalf("out=%q", out.String())
	}
}
