package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestPrintGuardCapabilitiesNoteMentionsMCPToolWhenRegistered(t *testing.T) {
	var buf bytes.Buffer
	printGuardCapabilitiesNote(&buf, guardMCPInstall{Applied: true, URL: "http://127.0.0.1:4567/mcp"})
	out := buf.String()
	for _, want := range []string{"fak-dev capabilities", "fak_capabilities", "intent `WIP`", "fak wip queue --json"} {
		if !strings.Contains(out, want) {
			t.Fatalf("capabilities note missing %q: %q", want, out)
		}
	}
}

func TestPrintGuardCapabilitiesNoteOmitsMCPToolWhenNotRegistered(t *testing.T) {
	var buf bytes.Buffer
	printGuardCapabilitiesNote(&buf, guardMCPInstall{Applied: false})
	out := buf.String()
	for _, want := range []string{"fak-dev capabilities", "fak-dev capabilities WIP", "fak wip queue --json"} {
		if !strings.Contains(out, want) {
			t.Fatalf("capabilities note missing %q: %q", want, out)
		}
	}
	if strings.Contains(out, "fak_capabilities MCP tool") {
		t.Fatalf("capabilities note should not advertise the MCP tool when MCP registration was not applied: %q", out)
	}
}
