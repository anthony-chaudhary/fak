package archreport

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const UsageSchema = "fak-architecture-usage/1"

// Usage records one privacy-safe architecture command outcome. It deliberately omits
// workspace paths, hostnames, usernames, leaf names, and error text.
type Usage struct {
	Schema      string `json:"schema"`
	At          string `json:"at"`
	Mode        string `json:"mode"`
	Format      string `json:"format"`
	Outcome     string `json:"outcome"`
	Diagnostics int    `json:"diagnostics,omitempty"`
	Violations  int    `json:"violations,omitempty"`
}

type UsageWeek struct {
	Week        string `json:"week"`
	Invocations int    `json:"invocations"`
	Full        int    `json:"full"`
	Scoped      int    `json:"scoped"`
	Text        int    `json:"text"`
	JSON        int    `json:"json"`
	OK          int    `json:"ok"`
	Error       int    `json:"error"`
}

// UsagePath returns the durable per-user ledger path. FAK_ARCHITECTURE_USAGE_FILE is
// an explicit test/operator override; "off" disables recording.
func UsagePath() (string, error) {
	if override := strings.TrimSpace(os.Getenv("FAK_ARCHITECTURE_USAGE_FILE")); override != "" {
		if strings.EqualFold(override, "off") {
			return "", nil
		}
		return override, nil
	}
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("locate user cache for architecture usage ledger: %w", err)
	}
	return filepath.Join(cache, "fak", "architecture-usage.jsonl"), nil
}

func AppendUsage(path string, row Usage) error {
	if path == "" {
		return nil
	}
	if row.Schema == "" {
		row.Schema = UsageSchema
	}
	if _, err := time.Parse(time.RFC3339, row.At); err != nil {
		return fmt.Errorf("architecture usage timestamp: %w", err)
	}
	if !oneOf(row.Mode, "full", "scoped") || !oneOf(row.Format, "text", "json") || !oneOf(row.Outcome, "ok", "error") {
		return fmt.Errorf("architecture usage row has invalid mode/format/outcome")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create architecture usage ledger directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open architecture usage ledger: %w", err)
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(row); err != nil {
		return fmt.Errorf("append architecture usage ledger: %w", err)
	}
	return f.Sync()
}

func FoldUsage(path string) ([]UsageWeek, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("open architecture usage ledger: %w", err)
	}
	defer f.Close()
	weeks := map[string]*UsageWeek{}
	scanner := bufio.NewScanner(f)
	for line := 1; scanner.Scan(); line++ {
		var row Usage
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("read architecture usage ledger line %d: %w", line, err)
		}
		at, err := time.Parse(time.RFC3339, row.At)
		if err != nil {
			return nil, fmt.Errorf("read architecture usage ledger line %d timestamp: %w", line, err)
		}
		year, week := at.ISOWeek()
		key := fmt.Sprintf("%04d-W%02d", year, week)
		if weeks[key] == nil {
			weeks[key] = &UsageWeek{Week: key}
		}
		w := weeks[key]
		w.Invocations++
		switch row.Mode {
		case "full":
			w.Full++
		case "scoped":
			w.Scoped++
		}
		switch row.Format {
		case "text":
			w.Text++
		case "json":
			w.JSON++
		}
		switch row.Outcome {
		case "ok":
			w.OK++
		case "error":
			w.Error++
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan architecture usage ledger: %w", err)
	}
	keys := make([]string, 0, len(weeks))
	for key := range weeks {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]UsageWeek, 0, len(keys))
	for _, key := range keys {
		out = append(out, *weeks[key])
	}
	return out, nil
}

func oneOf(got string, want ...string) bool {
	for _, value := range want {
		if got == value {
			return true
		}
	}
	return false
}
