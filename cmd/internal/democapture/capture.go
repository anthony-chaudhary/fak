// Package democapture compares a demo's real selfcheck bytes with the output
// published in its adopter-facing EXAMPLE-OUTPUT.md.
package democapture

import (
	"bytes"
	"fmt"
	"os"
)

const (
	captureBegin = "<!-- BEGIN SELFCHECK OUTPUT -->\n```text\n"
	captureEnd   = "```\n<!-- END SELFCHECK OUTPUT -->"
)

// MatchMarkdown returns nil when got is byte-for-byte identical to the captured
// selfcheck block in path. Line endings are normalized so the same committed
// capture works in Windows and Unix checkouts.
func MatchMarkdown(path string, got []byte) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read captured output: %w", err)
	}
	raw = normalizeLineEndings(raw)
	got = normalizeLineEndings(got)

	start := bytes.Index(raw, []byte(captureBegin))
	if start < 0 {
		return fmt.Errorf("captured output is missing begin marker %q", captureBegin)
	}
	start += len(captureBegin)
	end := bytes.Index(raw[start:], []byte(captureEnd))
	if end < 0 {
		return fmt.Errorf("captured output is missing end marker %q", captureEnd)
	}
	want := raw[start : start+end]
	if !bytes.Equal(got, want) {
		return fmt.Errorf("selfcheck output drifted from %s\nwant:\n%s\ngot:\n%s", path, want, got)
	}
	return nil
}

func normalizeLineEndings(in []byte) []byte {
	return bytes.ReplaceAll(in, []byte("\r\n"), []byte("\n"))
}
