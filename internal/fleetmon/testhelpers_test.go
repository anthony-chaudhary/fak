package fleetmon

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// tsAt renders an RFC3339 timestamp n minutes before base — a helper for building
// transcripts whose last line is a known age.
func tsAt(base time.Time, minutesAgo float64) string {
	return base.Add(-time.Duration(minutesAgo * float64(time.Minute))).UTC().Format(time.RFC3339Nano)
}

// jline marshals one transcript record to a JSONL line.
func jline(t *testing.T, rec map[string]any) string {
	t.Helper()
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("marshal transcript line: %v", err)
	}
	return string(b) + "\n"
}

// assistantText builds an assistant record that ends the turn with a text block —
// i.e. a final report.
func assistantText(t *testing.T, ts, text, stopReason string) string {
	return jline(t, map[string]any{
		"type":      "assistant",
		"timestamp": ts,
		"message": map[string]any{
			"role":        "assistant",
			"stop_reason": stopReason,
			"content":     []any{map[string]any{"type": "text", "text": text}},
		},
	})
}

// assistantToolUse builds an assistant record whose last block is a tool_use
// (mid-work, no final report).
func assistantToolUse(t *testing.T, ts, tool string, input map[string]any) string {
	return jline(t, map[string]any{
		"type":      "assistant",
		"timestamp": ts,
		"message": map[string]any{
			"role":        "assistant",
			"stop_reason": "tool_use",
			"content":     []any{map[string]any{"type": "tool_use", "name": tool, "input": input}},
		},
	})
}

// userToolResult builds a user record carrying a tool_result (a text blob, e.g.
// an error message the blocker scan reads).
func userToolResult(t *testing.T, ts, content string, isError bool) string {
	return jline(t, map[string]any{
		"type":      "user",
		"timestamp": ts,
		"message": map[string]any{
			"role":    "user",
			"content": []any{map[string]any{"type": "tool_result", "content": content, "is_error": isError}},
		},
	})
}

// writeTranscript writes the concatenated lines to a temp .jsonl file and returns
// its path.
func writeTranscript(t *testing.T, lines ...string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "session.jsonl")
	var buf []byte
	for _, l := range lines {
		buf = append(buf, l...)
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}
	return path
}

// proc builds one relations-snapshot process row.
func proc(pid, ppid int, name, cmd string, ageSec int, start string) procguard.Proc {
	p := procguard.Proc{PID: pid, Name: name, Cmdline: cmd, Start: start}
	if ppid != 0 {
		pp := ppid
		p.PPID = &pp
	}
	a := ageSec
	p.AgeSec = &a
	return p
}

func fptr(f float64) *float64 { return &f }
func iptr(n int) *int         { return &n }
func bptr(b bool) *bool       { return &b }
