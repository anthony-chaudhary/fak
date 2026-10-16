// fak-test:runtime fast est=1s
package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeProse(t *testing.T, root, rel, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(root, rel), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func totalLine(t *testing.T, out string) string {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(out), "\n")
	last := lines[len(lines)-1]
	if !strings.HasPrefix(last, "# TOTAL ") {
		t.Fatalf("last line = %q, want # TOTAL summary", last)
	}
	return last
}

func TestRunReportsMechanicalReframe(t *testing.T) {
	root := t.TempDir()
	writeProse(t, root, "steer.md", "Do not forget to verify.\n")
	writeProse(t, root, "clean.md", "Verify every change before landing.\n")
	var out bytes.Buffer
	if code := run(root, []string{"steer.md", "clean.md"}, &out); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if got := totalLine(t, out.String()); got != "# TOTAL mechanical=1 judgement=0 across 2 path(s)" {
		t.Fatalf("total = %q", got)
	}
	if !strings.Contains(out.String(), "== steer.md ") || strings.Contains(out.String(), "== clean.md ") {
		t.Fatalf("only the negative file gets a section:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "] MECH\n") {
		t.Fatalf("mechanical finding not rendered:\n%s", out.String())
	}
}

func TestRunMissingPathCountsButScoresNothing(t *testing.T) {
	var out bytes.Buffer
	if code := run(t.TempDir(), []string{"absent.md"}, &out); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if got := totalLine(t, out.String()); got != "# TOTAL mechanical=0 judgement=0 across 1 path(s)" {
		t.Fatalf("total = %q", got)
	}
}

func TestRunEmptyRootUsesDefaultCorpus(t *testing.T) {
	var out bytes.Buffer
	if code := run(t.TempDir(), nil, &out); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.HasPrefix(out.String(), "# default corpus: 0 file(s)\n") {
		t.Fatalf("stdout = %q, want default-corpus header", out.String())
	}
	if got := totalLine(t, out.String()); got != "# TOTAL mechanical=0 judgement=0 across 0 path(s)" {
		t.Fatalf("total = %q", got)
	}
}
