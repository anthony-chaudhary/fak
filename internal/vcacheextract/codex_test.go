package vcacheextract

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSanitizeTokenCountRow(t *testing.T) {
	row := map[string]any{
		"type": "event_msg",
		"payload": map[string]any{
			"type": "token_count",
			"info": map[string]any{
				"last_token_usage": map[string]any{
					"input_tokens":        2006,
					"cached_input_tokens": 1920,
					"output_tokens":       12,
				},
				"prompt_text": "must not be copied",
			},
		},
		"extra": "must not be copied",
	}
	got, ok := SanitizeRow(row)
	if !ok {
		t.Fatal("SanitizeRow returned false")
	}
	payload := got["payload"].(map[string]any)
	info := payload["info"].(map[string]any)
	usage := info["last_token_usage"].(map[string]any)
	if usage["input_tokens"] != 2006 || usage["cached_input_tokens"] != 1920 {
		t.Fatalf("usage=%v", usage)
	}
	if _, ok := usage["output_tokens"]; ok {
		t.Fatalf("output tokens were not stripped: %v", usage)
	}
	if _, ok := got["extra"]; ok {
		t.Fatalf("extra field was not stripped: %v", got)
	}
}

func TestSanitizeTurnCompleted(t *testing.T) {
	got, ok := SanitizeRow(map[string]any{
		"type": "turn.completed",
		"usage": map[string]any{
			"input_tokens":        24763,
			"cached_input_tokens": 24448,
			"output_tokens":       122,
		},
	})
	if !ok {
		t.Fatal("SanitizeRow returned false")
	}
	usage := got["usage"].(map[string]any)
	if usage["input_tokens"] != 24763 || usage["cached_input_tokens"] != 24448 {
		t.Fatalf("usage=%v", usage)
	}
}

func TestExtractRowsExplicitSession(t *testing.T) {
	dir := t.TempDir()
	raw := filepath.Join(dir, "raw.jsonl")
	out := filepath.Join(dir, "sanitized.jsonl")
	writeLines(t, raw,
		map[string]any{"type": "response_item", "payload": map[string]any{"content": "drop"}},
		map[string]any{"type": "event_msg", "payload": map[string]any{"type": "token_count", "info": map[string]any{"last_token_usage": map[string]any{"input_tokens": 100, "cached_input_tokens": 50}}}},
		map[string]any{"type": "turn.completed", "usage": map[string]any{"input_tokens": 200, "cached_input_tokens": 180}},
	)
	rows, err := ExtractRows(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows=%d, want 2", len(rows))
	}
	if err := WriteRows(out, rows, nil); err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"cached_input_tokens":50`) || !strings.Contains(string(body), `"cached_input_tokens":180`) {
		t.Fatalf("output missing sanitized counters:\n%s", body)
	}
}

func TestTurnsFromRowsKeepsOnlyTokenCounters(t *testing.T) {
	rows := []map[string]any{
		{
			"type": "event_msg",
			"payload": map[string]any{
				"type": "token_count",
				"info": map[string]any{
					"last_token_usage": map[string]any{
						"input_tokens":        100,
						"cached_input_tokens": 80,
					},
				},
			},
		},
		{
			"type": "turn.completed",
			"usage": map[string]any{
				"input_tokens":        250,
				"cached_input_tokens": 200,
			},
		},
	}
	turns := TurnsFromRows(rows, "codex-thread")
	if len(turns) != 2 {
		t.Fatalf("turns=%d, want 2", len(turns))
	}
	if turns[0].Family != "codex-thread" || turns[0].InputTokens != 20 || turns[0].CacheRead != 80 {
		t.Fatalf("turn[0]=%+v, want family plus uncached/cache split", turns[0])
	}
	if turns[1].UnixMillis != 1 || turns[1].InputTokens != 50 || turns[1].CacheRead != 200 {
		t.Fatalf("turn[1]=%+v, want sequence clock and uncached/cache split", turns[1])
	}
}

func TestFindSessionByThreadID(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionDir := filepath.Join(home, "sessions", "2026", "06")
	if err := os.MkdirAll(sessionDir, 0o755); err != nil {
		t.Fatal(err)
	}
	raw := filepath.Join(sessionDir, "thread-abc123.jsonl")
	writeLines(t, raw, map[string]any{"type": "turn.completed", "usage": map[string]any{"input_tokens": 10, "cached_input_tokens": 9}})
	got, err := FindSession(home, "abc123")
	if err != nil {
		t.Fatal(err)
	}
	if got != raw {
		t.Fatalf("FindSession=%q, want %q", got, raw)
	}
}

func TestFindLatestSessionSince(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionDir := filepath.Join(home, "sessions", "2026", "07")
	if err := os.MkdirAll(sessionDir, 0o755); err != nil {
		t.Fatal(err)
	}
	oldPath := filepath.Join(sessionDir, "rollout-old.jsonl")
	newPath := filepath.Join(sessionDir, "rollout-new.jsonl")
	writeLines(t, oldPath, map[string]any{"type": "turn.completed", "usage": map[string]any{"input_tokens": 10}})
	writeLines(t, newPath, map[string]any{"type": "turn.completed", "usage": map[string]any{"input_tokens": 20}})
	oldTime := time.Date(2026, 7, 1, 10, 0, 0, 0, time.UTC)
	newTime := oldTime.Add(2 * time.Minute)
	if err := os.Chtimes(oldPath, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(newPath, newTime, newTime); err != nil {
		t.Fatal(err)
	}

	got, err := FindLatestSession(home, oldTime.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if got != newPath {
		t.Fatalf("FindLatestSession=%q, want %q", got, newPath)
	}
	if _, err := FindLatestSession(home, newTime.Add(time.Second)); err == nil {
		t.Fatalf("FindLatestSession after newest file unexpectedly succeeded")
	}
}

func TestNoTokenRows(t *testing.T) {
	raw := filepath.Join(t.TempDir(), "raw.jsonl")
	writeLines(t, raw, map[string]any{"type": "response_item"})
	rows, err := ExtractRows(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 0 {
		t.Fatalf("rows=%d, want 0", len(rows))
	}
}

func writeLines(t *testing.T, path string, rows ...map[string]any) {
	t.Helper()
	var b strings.Builder
	for _, row := range rows {
		data, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(data)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
