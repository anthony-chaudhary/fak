package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestExampleOutputMatchesSelfcheck(t *testing.T) {
	var got, errw bytes.Buffer
	if code := run(&got, &errw, []string{"-selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errw.String())
	}
	doc, err := os.ReadFile("EXAMPLE-OUTPUT.md")
	if err != nil {
		t.Fatal(err)
	}
	wantBlock := "```text\n" + got.String() + "```"
	if !strings.Contains(string(doc), wantBlock) {
		t.Fatalf("EXAMPLE-OUTPUT.md does not contain the captured selfcheck output:\n%s", got.String())
	}
}
