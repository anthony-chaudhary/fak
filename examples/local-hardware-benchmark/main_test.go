package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/localbench"
)

func TestExampleDelegatesToPromotedImplementation(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if err := localbench.RunCLI([]string{"--help"}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stdout.String(), "fak bench local run") {
		t.Fatalf("promoted help missing first-class command: %s", stdout.String())
	}
}
