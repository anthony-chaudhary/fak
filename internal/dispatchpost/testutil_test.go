package dispatchpost

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// chdir changes the working directory for the test and restores it on cleanup.
// Mirrors the helper in internal/scoreboard and internal/benchpost so the
// .env.slack.local walk-up resolves against a controlled temp dir.
func chdir(t *testing.T, dir string) {
	t.Helper()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(prev) })
}

// writeFile writes name under dir with the given content.
func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// durationFromMS builds a time.Duration from milliseconds (the unit Result carries),
// keeping the table-test free of the time package import.
func durationFromMS(ms int64) time.Duration { return time.Duration(ms) * time.Millisecond }

// flattenBlocks pulls every mrkdwn text string out of a Block Kit payload so a test
// can assert a fact appears somewhere in the rendered blocks without coupling to the
// nesting shape.
func flattenBlocks(blocks []any) string {
	var out string
	for _, b := range blocks {
		m, ok := b.(map[string]any)
		if !ok {
			continue
		}
		if txt, ok := m["text"].(map[string]any); ok {
			if s, ok := txt["text"].(string); ok {
				out += s + "\n"
			}
		}
		if els, ok := m["elements"].([]any); ok {
			for _, e := range els {
				if em, ok := e.(map[string]any); ok {
					if s, ok := em["text"].(string); ok {
						out += s + "\n"
					}
				}
			}
		}
	}
	return fmt.Sprint(out)
}
