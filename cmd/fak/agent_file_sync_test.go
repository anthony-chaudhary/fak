package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestAgentFileSync(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	a := write("AGENTS.md", "rules\n")
	b := write("CLAUDE.md", "rules\n")
	c := write("GEMINI.md", "rules\r\n") // one CRLF byte difference is drift
	d := write("OTHER.md", "rules\n")

	drift, err := agentFileSyncDrift([]string{a, b, d})
	if err != nil || len(drift) != 0 {
		t.Fatalf("identical files reported drift %v, err %v", drift, err)
	}
	drift, err = agentFileSyncDrift([]string{a, b, c, d})
	if err != nil || !slices.Equal(drift, []string{c}) {
		t.Fatalf("drift = %v, err %v; want only %s", drift, err, c)
	}
	if _, err := agentFileSyncDrift([]string{a, filepath.Join(dir, "missing.md")}); err == nil {
		t.Fatal("a missing mirror must be an error, not silent agreement")
	}
}
