package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestHarnessReleaseWitnessRequiresInputs(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runHarnessRelease(&out, &errOut, []string{"witness"}); code != 1 {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errOut.String())
	}
	if !strings.Contains(errOut.String(), "required") {
		t.Fatalf("err=%s", errOut.String())
	}
}
