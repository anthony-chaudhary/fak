package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/codexmcpdiag"
	"strings"
	"testing"
)

func TestDoctorCodexMCPWarningRedactsEvidence(t *testing.T) {
	old := readCodexMCPEvents
	defer func() { readCodexMCPEvents = old }()
	readCodexMCPEvents = func(_ string, n []string) ([]codexmcpdiag.Event, error) {
		var e []codexmcpdiag.Event
		for _, s := range n {
			e = append(e, codexmcpdiag.Event{Level: "INFO", Target: "mcp", Body: s + " initialized token=SECRET C:\\private\\x"})
		}
		return e, nil
	}
	// Exercise the pure rendering contract used by the command, since cmd uses process stdout.
	r := codexmcpdiag.Classify([]string{"codex_apps", "dos", "fak", "openaiDeveloperDocs"}, mustEvents(t))
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(r); err != nil {
		t.Fatal(err)
	}
	got := b.String()
	if !strings.Contains(got, codexmcpdiag.VerdictFalseNegative) || strings.Contains(got, "SECRET") || strings.Contains(got, "private") {
		t.Fatalf("unsafe output: %s", got)
	}
}
func mustEvents(t *testing.T) []codexmcpdiag.Event {
	t.Helper()
	e, err := readCodexMCPEvents("ignored", []string{"codex_apps", "dos", "fak", "openaiDeveloperDocs"})
	if err != nil {
		t.Fatal(err)
	}
	return e
}
