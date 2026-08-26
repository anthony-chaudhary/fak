package democapture

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMatchMarkdown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "EXAMPLE-OUTPUT.md")
	const doc = "# Capture\n\n<!-- BEGIN SELFCHECK OUTPUT -->\n```text\nPASS\nline two\n```\n<!-- END SELFCHECK OUTPUT -->\n"
	if err := os.WriteFile(path, []byte(doc), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := MatchMarkdown(path, []byte("PASS\r\nline two\r\n")); err != nil {
		t.Fatal(err)
	}
	if err := MatchMarkdown(path, []byte("FAIL\n")); err == nil {
		t.Fatal("drifted output matched")
	}
}
