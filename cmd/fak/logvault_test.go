package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/logvault"
)

// TestLogvaultDuReportsFootprint drives `fak logvault du` against a captured
// fixture vault and asserts the per-source footprint + capture-lag line and the
// WITNESSED total surface — the "is my backup current?" answer (#2455).
func TestLogvaultDuReportsFootprint(t *testing.T) {
	srcDir := t.TempDir()
	if err := writeFileLV(filepath.Join(srcDir, "loops.jsonl"), "row1\nrow2\n"); err != nil {
		t.Fatal(err)
	}
	vaultDir := t.TempDir()
	v := &logvault.Vault{Dir: vaultDir, Sources: []logvault.Source{{ID: "s", Root: srcDir}}}
	if _, err := v.Capture(); err != nil {
		t.Fatalf("capture: %v", err)
	}

	var out, errOut bytes.Buffer
	code := runLogvault(&out, &errOut, []string{"du", "-vault", vaultDir, "-repo", srcDir})
	if code != 0 {
		t.Fatalf("du exit = %d, stderr=%q", code, errOut.String())
	}
	got := out.String()
	if !strings.Contains(got, "  s ") {
		t.Fatalf("du output missing source line:\n%s", got)
	}
	if !strings.Contains(got, "files=1") || !strings.Contains(got, "bytes=10B") {
		t.Fatalf("du output missing witnessed footprint (want files=1 bytes=10B):\n%s", got)
	}
	if !strings.Contains(got, "last-capture=") || strings.Contains(got, "last-capture=never") {
		t.Fatalf("du output missing a fresh last-capture age:\n%s", got)
	}
	if !strings.Contains(got, "WITNESSED") {
		t.Fatalf("du total line must disclose the WITNESSED basis:\n%s", got)
	}
}

// TestLogvaultDuEmptyVaultIsValidEmpty proves an un-captured vault is the
// valid-empty posture (header + zero total), never an error.
func TestLogvaultDuEmptyVaultIsValidEmpty(t *testing.T) {
	vaultDir := t.TempDir()
	var out, errOut bytes.Buffer
	code := runLogvault(&out, &errOut, []string{"du", "-vault", vaultDir, "-repo", t.TempDir()})
	if code != 0 {
		t.Fatalf("du on empty vault exit = %d, stderr=%q", code, errOut.String())
	}
	if got := out.String(); !strings.Contains(got, "TOTAL bytes=0B") {
		t.Fatalf("empty vault du = %q, want a zero total", got)
	}
}

func writeFileLV(path, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0o644)
}
