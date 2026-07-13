package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
	"strings"
	"testing"
)

func TestSessionLogPrintsHashesAndKinds(t *testing.T) {
	l := sessionledger.Memory()
	e, err := l.Append("trace", "user", json.RawMessage(`{"x":1}`))
	if err != nil {
		t.Fatal(err)
	}
	entries, err := l.Chain("trace")
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	renderSessionLog(&out, entries, false)
	if !strings.Contains(out.String(), string(e.Hash)+" user") {
		t.Fatalf("output %q", out.String())
	}
}
