package main

import (
	"strings"
	"testing"
)

func TestAllVerbHelpListsServer(t *testing.T) {
	var out strings.Builder
	usageAllVerbs(&out)
	if !strings.Contains(out.String(), "fak server <init|up|status|down>") {
		t.Fatalf("fak help --all omits the live server verb")
	}
}
