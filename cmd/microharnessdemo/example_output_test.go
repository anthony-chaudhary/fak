package main

import (
	"bytes"
	"context"
	"os"
	"strings"
	"testing"
)

func TestExampleOutputMatchesSelfcheck(t *testing.T) {
	r, err := run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := check(r); err != nil {
		t.Fatal(err)
	}
	var got bytes.Buffer
	render(&got, r)
	doc, err := os.ReadFile("EXAMPLE-OUTPUT.md")
	if err != nil {
		t.Fatal(err)
	}
	wantBlock := "```text\n" + got.String() + "```"
	if !strings.Contains(string(doc), wantBlock) {
		t.Fatalf("EXAMPLE-OUTPUT.md does not contain the captured selfcheck output:\n%s", got.String())
	}
}
