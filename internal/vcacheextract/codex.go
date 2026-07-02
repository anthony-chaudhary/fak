// Package vcacheextract sanitizes Codex session JSONL token telemetry.
package vcacheextract

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

const Schema = "fak-vcache-codex-session-extract/1"

// AsNonnegativeInt coerces numeric JSON values to a nonnegative integer.
func AsNonnegativeInt(value any) int {
	var n int64
	switch v := value.(type) {
	case nil:
		return 0
	case int:
		n = int64(v)
	case int64:
		n = v
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0
		}
		n = int64(v)
	case json.Number:
		i, err := v.Int64()
		if err != nil {
			f, ferr := v.Float64()
			if ferr != nil {
				return 0
			}
			i = int64(f)
		}
		n = i
	case string:
		i, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
		if err != nil {
			return 0
		}
		n = i
	default:
		return 0
	}
	if n < 0 {
		return 0
	}
	if n > int64(^uint(0)>>1) {
		return int(^uint(0) >> 1)
	}
	return int(n)
}

// UsagePair extracts total input tokens and cached input tokens from a provider usage object.
func UsagePair(usage any) (total, cached int, ok bool) {
	u, ok := usage.(map[string]any)
	if !ok {
		return 0, 0, false
	}
	total = AsNonnegativeInt(firstPresent(u, "input_tokens", "prompt_tokens"))
	cached = AsNonnegativeInt(firstPresent(u, "cached_input_tokens", "cached_tokens"))
	if cached == 0 {
		if details, ok := u["input_tokens_details"].(map[string]any); ok {
			cached = AsNonnegativeInt(details["cached_tokens"])
		}
	}
	if cached == 0 {
		if details, ok := u["prompt_tokens_details"].(map[string]any); ok {
			cached = AsNonnegativeInt(details["cached_tokens"])
		}
	}
	if total == 0 && cached == 0 {
		return 0, 0, false
	}
	if total > 0 && cached > total {
		cached = total
	}
	return total, cached, true
}

func firstPresent(m map[string]any, keys ...string) any {
	for _, k := range keys {
		if v, ok := m[k]; ok && v != nil {
			return v
		}
	}
	return nil
}

// SanitizeRow drops all prompt/tool/response content and keeps only token counters.
func SanitizeRow(row any) (map[string]any, bool) {
	r, ok := row.(map[string]any)
	if !ok {
		return nil, false
	}
	if r["type"] == "event_msg" {
		payload, _ := r["payload"].(map[string]any)
		if payload != nil && payload["type"] == "token_count" {
			info, _ := payload["info"].(map[string]any)
			var last any
			if info != nil {
				last = info["last_token_usage"]
			}
			if total, cached, ok := UsagePair(last); ok {
				return map[string]any{
					"type": "event_msg",
					"payload": map[string]any{
						"type": "token_count",
						"info": map[string]any{
							"last_token_usage": map[string]any{
								"input_tokens":        total,
								"cached_input_tokens": cached,
							},
						},
					},
				}, true
			}
		}
	}
	if r["type"] == "turn.completed" {
		if total, cached, ok := UsagePair(r["usage"]); ok {
			return map[string]any{
				"type": "turn.completed",
				"usage": map[string]any{
					"input_tokens":        total,
					"cached_input_tokens": cached,
				},
			}, true
		}
	}
	return nil, false
}

// CodexHome resolves CODEX_HOME or ~/.codex.
func CodexHome(env map[string]string) string {
	if configured := strings.TrimSpace(env["CODEX_HOME"]); configured != "" {
		return configured
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".codex")
	}
	return ".codex"
}

// FindSession returns the newest session JSONL matching threadID.
func FindSession(home, threadID string) (string, error) {
	threadID = strings.TrimSpace(threadID)
	if threadID == "" {
		return "", errors.New("CODEX_THREAD_ID is empty")
	}
	sessions := filepath.Join(home, "sessions")
	if st, err := os.Stat(sessions); err != nil || !st.IsDir() {
		return "", fmt.Errorf("Codex sessions directory not found: %s", sessions)
	}
	type candidate struct {
		path string
		mod  int64
	}
	var candidates []candidate
	err := filepath.WalkDir(sessions, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		name := d.Name()
		if !strings.HasSuffix(name, ".jsonl") || !strings.Contains(name, threadID) {
			return nil
		}
		if info, statErr := d.Info(); statErr == nil {
			candidates = append(candidates, candidate{path: path, mod: info.ModTime().UnixNano()})
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].mod != candidates[j].mod {
			return candidates[i].mod > candidates[j].mod
		}
		return candidates[i].path < candidates[j].path
	})
	if len(candidates) == 0 {
		return "", fmt.Errorf("no Codex session JSONL matched thread id %q under %s", threadID, sessions)
	}
	return candidates[0].path, nil
}

// FindLatestSession returns the newest Codex session JSONL under home/sessions. When
// since is non-zero, candidates older than since are ignored. This supports post-run
// launchers that do not know Codex's thread id yet but can bound discovery to files
// touched during the child process lifetime.
func FindLatestSession(home string, since time.Time) (string, error) {
	sessions := filepath.Join(home, "sessions")
	if st, err := os.Stat(sessions); err != nil || !st.IsDir() {
		return "", fmt.Errorf("Codex sessions directory not found: %s", sessions)
	}
	type candidate struct {
		path string
		mod  time.Time
	}
	var candidates []candidate
	err := filepath.WalkDir(sessions, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(d.Name(), ".jsonl") {
			return nil
		}
		info, statErr := d.Info()
		if statErr != nil {
			return nil
		}
		if !since.IsZero() && info.ModTime().Before(since) {
			return nil
		}
		candidates = append(candidates, candidate{path: path, mod: info.ModTime()})
		return nil
	})
	if err != nil {
		return "", err
	}
	sort.Slice(candidates, func(i, j int) bool {
		if !candidates[i].mod.Equal(candidates[j].mod) {
			return candidates[i].mod.After(candidates[j].mod)
		}
		return candidates[i].path < candidates[j].path
	})
	if len(candidates) == 0 {
		if since.IsZero() {
			return "", fmt.Errorf("no Codex session JSONL found under %s", sessions)
		}
		return "", fmt.Errorf("no Codex session JSONL newer than %s under %s", since.Format(time.RFC3339), sessions)
	}
	return candidates[0].path, nil
}

// ExtractRows reads a JSONL session and returns sanitized token rows.
func ExtractRows(path string) ([]map[string]any, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []map[string]any
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var raw any
		dec := json.NewDecoder(strings.NewReader(line))
		dec.UseNumber()
		if err := dec.Decode(&raw); err != nil {
			return nil, fmt.Errorf("%s:%d: invalid JSON: %w", path, lineNo, err)
		}
		if sanitized, ok := SanitizeRow(raw); ok {
			rows = append(rows, sanitized)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

// TurnsFromRows converts sanitized Codex token rows into the observed-turn shape
// `fak vcache score` already consumes from its default snapshot. The conversion keeps
// only token counters and a caller-supplied family label; prompts, tool arguments,
// tool outputs, model text, and response text are never present in sanitized rows.
func TurnsFromRows(rows []map[string]any, family string) []vcacheobserve.Turn {
	family = strings.TrimSpace(family)
	if family == "" {
		family = "codex"
	}
	turns := make([]vcacheobserve.Turn, 0, len(rows))
	for i, row := range rows {
		total, cached, ok := sanitizedUsagePair(row)
		if !ok {
			continue
		}
		turns = append(turns, vcacheobserve.Turn{
			Family:      family,
			UnixMillis:  int64(i),
			InputTokens: int64(total - cached),
			CacheRead:   int64(cached),
		})
	}
	return turns
}

func sanitizedUsagePair(row map[string]any) (total, cached int, ok bool) {
	if row == nil {
		return 0, 0, false
	}
	if row["type"] == "turn.completed" {
		return UsagePair(row["usage"])
	}
	if row["type"] == "event_msg" {
		payload, _ := row["payload"].(map[string]any)
		if payload == nil || payload["type"] != "token_count" {
			return 0, 0, false
		}
		info, _ := payload["info"].(map[string]any)
		if info == nil {
			return 0, 0, false
		}
		return UsagePair(info["last_token_usage"])
	}
	return 0, 0, false
}

// WriteRows writes sanitized JSONL rows to path or stdout when path is "-".
func WriteRows(path string, rows []map[string]any, stdout io.Writer) error {
	if path == "-" {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		for _, row := range rows {
			if err := enc.Encode(row); err != nil {
				return err
			}
		}
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}
