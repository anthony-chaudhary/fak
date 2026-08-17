package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestUsageAllVerbsExcludesMovedDevCommands(t *testing.T) {
	var out bytes.Buffer
	usageAllVerbs(&out)
	if strings.Contains(out.String(), "  fak buildcheck ") {
		t.Fatalf("runtime help still advertises moved buildcheck:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "  fak agent ") {
		t.Fatalf("runtime help lost runtime command agent:\n%s", out.String())
	}
}

func TestSuggestVerbSpellingHandsMovedCommandsToDevNamespace(t *testing.T) {
	if got := suggestVerbSpelling("buildcheck"); got != "dev buildcheck" {
		t.Fatalf("suggestVerbSpelling(buildcheck) = %q, want dev buildcheck", got)
	}
	if got := suggestVerbSpelling("guardd"); got != "manage" {
		t.Fatalf("suggestVerbSpelling(guardd) = %q, want manage", got)
	}
}
