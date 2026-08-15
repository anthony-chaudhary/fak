package main

import (
	"os/exec"
	"strings"
	"testing"
)

func TestSelfcheck(t *testing.T) {
	out, err := exec.Command("go", "run", ".", "-selfcheck").CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	for _, want := range []string{"CODING fit: choose ponytail@r8", "LEGAL REVIEW fit: choose legal-review-harness@r4", "PONYTAIL for legal: refuse", "not legal certification", "SELFCHECK PASS"} {
		if !strings.Contains(string(out), want) {
			t.Fatalf("missing %q\n%s", want, out)
		}
	}
}
