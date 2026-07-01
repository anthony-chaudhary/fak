package taskdecision

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	Schema             = "fak.task-decision.v1"
	DefaultReloadLimit = 8
)

var safeTaskIDRE = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

type Entry struct {
	Schema      string   `json:"schema"`
	TaskID      string   `json:"task_id"`
	Decision    string   `json:"decision"`
	Rationale   string   `json:"rationale"`
	EvidenceRef string   `json:"evidence_ref"`
	OpenThreads []string `json:"open_threads,omitempty"`
	UnixNano    int64    `json:"unix_nano,omitempty"`
}

func DefaultPath(root, taskID string) string {
	root = strings.TrimSpace(root)
	if root == "" {
		root = "."
	}
	return filepath.Join(root, ".fak", "task-decisions", Slug(taskID)+".jsonl")
}

func Slug(taskID string) string {
	s := strings.Trim(safeTaskIDRE.ReplaceAllString(strings.TrimSpace(taskID), "-"), "-._")
	if s == "" {
		return "task"
	}
	if len(s) > 80 {
		s = s[:80]
	}
	return s
}

func Append(path string, entry Entry) error {
	entry = Normalize(entry)
	if err := Validate(entry); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	return enc.Encode(entry)
}

func Load(path, taskID string, limit int) ([]Entry, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	taskID = strings.TrimSpace(taskID)
	var out []Entry
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var entry Entry
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			return nil, fmt.Errorf("parse decision log %s: %w", path, err)
		}
		entry = Normalize(entry)
		if taskID != "" && entry.TaskID != taskID {
			continue
		}
		if err := Validate(entry); err != nil {
			return nil, fmt.Errorf("invalid decision log entry in %s: %w", path, err)
		}
		out = append(out, entry)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if limit <= 0 || len(out) <= limit {
		return out, nil
	}
	return append([]Entry(nil), out[len(out)-limit:]...), nil
}

func Normalize(entry Entry) Entry {
	entry.Schema = strings.TrimSpace(entry.Schema)
	if entry.Schema == "" {
		entry.Schema = Schema
	}
	entry.TaskID = strings.TrimSpace(entry.TaskID)
	entry.Decision = strings.TrimSpace(entry.Decision)
	entry.Rationale = strings.TrimSpace(entry.Rationale)
	entry.EvidenceRef = strings.TrimSpace(entry.EvidenceRef)
	entry.OpenThreads = compactStrings(entry.OpenThreads)
	return entry
}

func Validate(entry Entry) error {
	if entry.Schema != Schema {
		return fmt.Errorf("schema = %q, want %q", entry.Schema, Schema)
	}
	if entry.TaskID == "" {
		return fmt.Errorf("task_id is required")
	}
	if entry.Decision == "" {
		return fmt.Errorf("decision is required")
	}
	if entry.Rationale == "" {
		return fmt.Errorf("rationale is required")
	}
	if entry.EvidenceRef == "" {
		return fmt.Errorf("evidence_ref is required")
	}
	return nil
}

func Render(entries []Entry) string {
	if len(entries) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Task decision log (bounded reload):")
	for _, entry := range entries {
		b.WriteString("\n- Decision: ")
		b.WriteString(entry.Decision)
		b.WriteString("\n  Rationale: ")
		b.WriteString(entry.Rationale)
		b.WriteString("\n  Evidence: ")
		b.WriteString(entry.EvidenceRef)
		if len(entry.OpenThreads) > 0 {
			b.WriteString("\n  Open threads: ")
			b.WriteString(strings.Join(entry.OpenThreads, "; "))
		}
	}
	return b.String()
}

func compactStrings(in []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, item := range in {
		item = strings.TrimSpace(item)
		if item == "" || seen[item] {
			continue
		}
		seen[item] = true
		out = append(out, item)
	}
	return out
}
