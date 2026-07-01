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
	if !strings.Contains(out, "fak capabilities") || !strings.Contains(out, "fak_capabilities") {
		t.Fatalf("capabilities note missing CLI/MCP mention: %q", out)
	}
}

func TestPrintGuardCapabilitiesNoteOmitsMCPToolWhenNotRegistered(t *testing.T) {
	var buf bytes.Buffer
	printGuardCapabilitiesNote(&buf, guardMCPInstall{Applied: false})
	out := buf.String()
	if !strings.Contains(out, "fak capabilities") {
		t.Fatalf("capabilities note missing CLI mention: %q", out)
	}
	if strings.Contains(out, "fak_capabilities MCP tool") {
		t.Fatalf("capabilities note should not advertise the MCP tool when MCP registration was not applied: %q", out)
	}
}
