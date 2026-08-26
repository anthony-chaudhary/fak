package main

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
)

func TestCapturedSelfcheckMatchesExampleOutput(t *testing.T) {
	var got bytes.Buffer
	if err := runCapturedSelfcheck(context.Background(), &got); err != nil {
		t.Fatal(err)
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
