// Package sessionmine normalizes local agent histories into privacy-safe workflow metrics.
package sessionmine

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const Schema = "fak-session-mine/1"

type Options struct {
	CodexRoot  string
	ClaudeRoot string
	Since      time.Time
	MinSupport int
	Limit      int
}

type Report struct {
	Schema     string      `json:"schema"`
	Generated  string      `json:"generated_at"`
	Inputs     InputStats  `json:"inputs"`
	Metrics    Metrics     `json:"metrics"`
	Sessions   []Session   `json:"sessions"`
	Candidates []Candidate `json:"candidates"`
}

type InputStats struct {
	CodexRoot      string `json:"codex_root"`
	ClaudeRoot     string `json:"claude_root"`
	FilesScanned   int    `json:"files_scanned"`
	FilesRejected  int    `json:"files_rejected"`
	MalformedLines int    `json:"malformed_lines"`
}

type Metrics struct {
	Sessions       int            `json:"sessions"`
	DurationsMS    int64          `json:"durations_ms"`
	P50DurationMS  int64          `json:"p50_duration_ms,omitempty"`
	P95DurationMS  int64          `json:"p95_duration_ms,omitempty"`
	ByProvider     map[string]int `json:"by_provider"`
	ToolCalls      int            `json:"tool_calls"`
	ToolResults    int            `json:"tool_results"`
	ToolErrors     int            `json:"tool_errors"`
	UserTurns      int            `json:"user_turns"`
	AssistantTurns int            `json:"assistant_turns"`
	CandidateCount int            `json:"candidate_count"`
}

type Session struct {
	ID             string   `json:"id"`
	Provider       string   `json:"provider"`
	StartedAt      string   `json:"started_at,omitempty"`
	EndedAt        string   `json:"ended_at,omitempty"`
	DurationMS     int64    `json:"duration_ms,omitempty"`
	UserTurns      int      `json:"user_turns"`
	AssistantTurns int      `json:"assistant_turns"`
	ToolCalls      int      `json:"tool_calls"`
	ToolResults    int      `json:"tool_results"`
	ToolErrors     int      `json:"tool_errors"`
	Trajectory     []string `json:"trajectory"`
}

type Candidate struct {
	Rank             int      `json:"rank"`
	Fingerprint      string   `json:"fingerprint"`
	Trajectory       []string `json:"trajectory"`
	Occurrences      int      `json:"occurrences"`
	SessionSupport   int      `json:"session_support"`
	ProviderSupport  int      `json:"provider_support"`
	DeterminismScore int      `json:"determinism_score"`
	SuggestedLeaf    string   `json:"suggested_leaf"`
	Witness          string   `json:"witness"`
}

type rawLine map[string]any

type aggregate struct {
	trajectory  []string
	occurrences int
	sessions    map[string]struct{}
	providers   map[string]struct{}
}

func Mine(opts Options) (Report, error) {
	if opts.MinSupport <= 0 {
		opts.MinSupport = 2
	}
	if opts.Limit <= 0 {
		opts.Limit = 25
	}
	report := Report{Schema: Schema, Generated: generatedAt(opts.Since), Inputs: InputStats{CodexRoot: sourceLabel(opts.CodexRoot, "codex"), ClaudeRoot: sourceLabel(opts.ClaudeRoot, "claude")}, Metrics: Metrics{ByProvider: map[string]int{}}}
	var sessions []Session
	for _, source := range []struct{ provider, root string }{{"codex", opts.CodexRoot}, {"claude", opts.ClaudeRoot}} {
		if strings.TrimSpace(source.root) == "" {
			continue
		}
		err := filepath.WalkDir(source.root, func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				report.Inputs.FilesRejected++
				return nil
			}
			if entry.IsDir() || !strings.EqualFold(filepath.Ext(path), ".jsonl") {
				return nil
			}
			info, err := entry.Info()
			if err != nil {
				report.Inputs.FilesRejected++
				return nil
			}
			if !opts.Since.IsZero() && info.ModTime().Before(opts.Since) {
				return nil
			}
			report.Inputs.FilesScanned++
			s, malformed, err := parseFile(path, source.provider)
			report.Inputs.MalformedLines += malformed
			if err != nil {
				report.Inputs.FilesRejected++
				return nil
			}
			if s.ToolCalls+s.UserTurns+s.AssistantTurns > 0 {
				sessions = append(sessions, s)
			}
			return nil
		})
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return Report{}, fmt.Errorf("walk %s sessions: %w", source.provider, err)
		}
	}
	sort.Slice(sessions, func(i, j int) bool {
		if sessions[i].Provider != sessions[j].Provider {
			return sessions[i].Provider < sessions[j].Provider
		}
		return sessions[i].ID < sessions[j].ID
	})
	report.Sessions = sessions
	aggs := map[string]*aggregate{}
	durations := make([]int64, 0, len(sessions))
	for _, s := range sessions {
		report.Metrics.Sessions++
		report.Metrics.ByProvider[s.Provider]++
		report.Metrics.ToolCalls += s.ToolCalls
		report.Metrics.ToolResults += s.ToolResults
		report.Metrics.ToolErrors += s.ToolErrors
		report.Metrics.UserTurns += s.UserTurns
		report.Metrics.AssistantTurns += s.AssistantTurns
		if s.DurationMS > 0 {
			report.Metrics.DurationsMS += s.DurationMS
			durations = append(durations, s.DurationMS)
		}
		seen := map[string]struct{}{}
		for n := 2; n <= 4; n++ {
			for i := 0; i+n <= len(s.Trajectory); i++ {
				seq := append([]string(nil), s.Trajectory[i:i+n]...)
				key := strings.Join(seq, "\x1f")
				a := aggs[key]
				if a == nil {
					a = &aggregate{trajectory: seq, sessions: map[string]struct{}{}, providers: map[string]struct{}{}}
					aggs[key] = a
				}
				a.occurrences++
				a.providers[s.Provider] = struct{}{}
				if _, ok := seen[key]; !ok {
					a.sessions[s.ID] = struct{}{}
					seen[key] = struct{}{}
				}
			}
		}
	}
	sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
	if len(durations) > 0 {
		report.Metrics.P50DurationMS = percentile(durations, 50)
		report.Metrics.P95DurationMS = percentile(durations, 95)
	}
	for key, a := range aggs {
		if len(a.sessions) < opts.MinSupport {
			continue
		}
		sum := sha256.Sum256([]byte(key))
		score := len(a.sessions)*100 + len(a.providers)*25 + a.occurrences*5 + len(a.trajectory)
		report.Candidates = append(report.Candidates, Candidate{Fingerprint: hex.EncodeToString(sum[:6]), Trajectory: a.trajectory, Occurrences: a.occurrences, SessionSupport: len(a.sessions), ProviderSupport: len(a.providers), DeterminismScore: score, SuggestedLeaf: suggestLeaf(a.trajectory), Witness: "implement a Go leaf/verb and replay this normalized trajectory against captured fixtures"})
	}
	sort.Slice(report.Candidates, func(i, j int) bool {
		a, b := report.Candidates[i], report.Candidates[j]
		if a.DeterminismScore != b.DeterminismScore {
			return a.DeterminismScore > b.DeterminismScore
		}
		return a.Fingerprint < b.Fingerprint
	})
	if len(report.Candidates) > opts.Limit {
		report.Candidates = report.Candidates[:opts.Limit]
	}
	for i := range report.Candidates {
		report.Candidates[i].Rank = i + 1
	}
	report.Metrics.CandidateCount = len(report.Candidates)
	return report, nil
}

func generatedAt(since time.Time) string {
	if since.IsZero() {
		return "all"
	}
	return since.UTC().Format(time.RFC3339)
}
func percentile(sorted []int64, pct int) int64 {
	index := (len(sorted)*pct+99)/100 - 1
	if index < 0 {
		index = 0
	}
	return sorted[index]
}

func parseFile(path, provider string) (Session, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return Session{}, 0, err
	}
	defer f.Close()
	h := sha256.Sum256([]byte(provider + "\x00" + filepath.Clean(path)))
	s := Session{ID: provider + "-" + hex.EncodeToString(h[:6]), Provider: provider}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	malformed := 0
	var first, last time.Time
	for scanner.Scan() {
		var row rawLine
		if json.Unmarshal(scanner.Bytes(), &row) != nil {
			malformed++
			continue
		}
		ts := parseTime(text(row["timestamp"]))
		if !ts.IsZero() {
			if first.IsZero() || ts.Before(first) {
				first = ts
			}
			if last.IsZero() || ts.After(last) {
				last = ts
			}
		}
		if provider == "codex" {
			parseCodex(row, &s)
		} else {
			parseClaude(row, &s)
		}
	}
	if err := scanner.Err(); err != nil {
		return Session{}, malformed, err
	}
	if !first.IsZero() {
		s.StartedAt = first.UTC().Format(time.RFC3339)
		s.EndedAt = last.UTC().Format(time.RFC3339)
		s.DurationMS = last.Sub(first).Milliseconds()
	}
	return s, malformed, nil
}

func parseCodex(row rawLine, s *Session) {
	typ := text(row["type"])
	p, _ := row["payload"].(map[string]any)
	if typ != "response_item" || p == nil {
		return
	}
	switch text(p["type"]) {
	case "message":
		role := text(p["role"])
		if role == "user" {
			s.UserTurns++
		} else if role == "assistant" {
			s.AssistantTurns++
		}
	case "function_call", "custom_tool_call":
		name := normalizeTool(text(p["name"]))
		if name == "shell_command" {
			name = classifyCodexShell(text(p["arguments"]))
		}
		if name != "" {
			s.ToolCalls++
			s.Trajectory = append(s.Trajectory, name)
		}
	case "function_call_output", "custom_tool_call_output":
		s.ToolResults++
		if outputLooksError(p["output"]) {
			s.ToolErrors++
		}
	}
}
func parseClaude(row rawLine, s *Session) {
	typ := text(row["type"])
	if typ == "user" {
		s.UserTurns++
	}
	if typ != "assistant" && typ != "user" {
		return
	}
	msg, _ := row["message"].(map[string]any)
	if typ == "assistant" {
		s.AssistantTurns++
	}
	content, _ := msg["content"].([]any)
	for _, v := range content {
		item, _ := v.(map[string]any)
		switch text(item["type"]) {
		case "tool_use":
			name := normalizeTool(text(item["name"]))
			if name == "shell_command" {
				name = classifyClaudeShell(item["input"])
			}
			if name != "" {
				s.ToolCalls++
				s.Trajectory = append(s.Trajectory, name)
			}
		case "tool_result":
			s.ToolResults++
			if b, ok := item["is_error"].(bool); ok && b {
				s.ToolErrors++
			}
		}
	}
}
func classifyCodexShell(arguments string) string {
	var envelope struct {
		Command string `json:"command"`
	}
	if json.Unmarshal([]byte(arguments), &envelope) != nil {
		return "shell_command"
	}
	return classifyShell(envelope.Command)
}

func classifyClaudeShell(input any) string {
	envelope, ok := input.(map[string]any)
	if !ok {
		return "shell_command"
	}
	return classifyShell(text(envelope["command"]))
}

// classifyShell emits only a bounded operation family. Arguments, paths, URLs,
// issue text, and other transcript content never enter the report.
func classifyShell(command string) string {
	v := strings.ToLower(strings.TrimSpace(command))
	for _, rule := range []struct {
		need  []string
		label string
	}{
		{[]string{"git", "status"}, "git_status"},
		{[]string{"git", "diff"}, "git_diff"},
		{[]string{"git", "log"}, "git_log"},
		{[]string{"git", "commit"}, "git_commit"},
		{[]string{"git", "push"}, "git_push"},
		{[]string{"git", "fetch"}, "git_fetch"},
		{[]string{"git", "merge"}, "git_merge"},
		{[]string{"go", "test"}, "go_test"},
		{[]string{"go", "build"}, "go_build"},
		{[]string{"go", "vet"}, "go_vet"},
		{[]string{"fak", "validate"}, "fak_validate"},
		{[]string{"fak", "buildcheck"}, "fak_buildcheck"},
		{[]string{"fak", "commit"}, "fak_commit"},
		{[]string{"fak", "sweep"}, "fak_sweep"},
		{[]string{"gh", "issue"}, "github_issue"},
		{[]string{"rg"}, "search_text"},
		{[]string{"select-string"}, "search_text"},
		{[]string{"get-content"}, "read_file"},
		{[]string{"cat"}, "read_file"},
		{[]string{"get-childitem"}, "find_files"},
		{[]string{"ls"}, "find_files"},
	} {
		matched := true
		for _, token := range rule.need {
			if !containsCommandToken(v, token) {
				matched = false
				break
			}
		}
		if matched {
			return rule.label
		}
	}
	return "shell_command"
}

func containsCommandToken(command, token string) bool {
	for _, field := range strings.FieldsFunc(command, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' || r == '-')
	}) {
		if field == token {
			return true
		}
	}
	return false
}

func normalizeTool(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	v = strings.TrimPrefix(v, "mcp__")
	v = strings.ReplaceAll(v, "-", "_")
	aliases := map[string]string{
		"bash": "shell_command", "shell": "shell_command", "powershell": "shell_command",
		"read": "read_file", "write": "write_file", "edit": "edit_file", "multiedit": "edit_file",
		"glob": "find_files", "grep": "search_text", "websearch": "web_search", "webfetch": "web_fetch",
		"todowrite": "update_plan", "askuserquestion": "request_user_input",
	}
	if canonical, ok := aliases[v]; ok {
		return canonical
	}
	allow := map[string]struct{}{
		"shell_command": {}, "read_file": {}, "write_file": {}, "edit_file": {}, "find_files": {},
		"search_text": {}, "web_search": {}, "web_fetch": {}, "update_plan": {}, "request_user_input": {},
		"view_image": {}, "list_mcp_resources": {}, "read_mcp_resource": {}, "tool_search": {},
	}
	if _, ok := allow[v]; ok {
		return v
	}
	if v == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(v))
	return "other_tool_" + hex.EncodeToString(sum[:3])
}
func outputLooksError(v any) bool {
	x := strings.ToLower(text(v))
	return strings.Contains(x, "error") || strings.Contains(x, "failed")
}
func text(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}
func parseTime(v string) time.Time { t, _ := time.Parse(time.RFC3339Nano, v); return t }
func sourceLabel(v, provider string) string {
	if strings.TrimSpace(v) == "" {
		return "disabled"
	}
	return provider + "-sessions"
}

func cleanRoot(v string) string {
	if v == "" {
		return ""
	}
	return filepath.Clean(v)
}
func suggestLeaf(seq []string) string {
	parts := make([]string, 0, len(seq))
	for _, v := range seq {
		v = strings.Trim(v, "_ ")
		if v != "" && (len(parts) == 0 || parts[len(parts)-1] != v) {
			parts = append(parts, v)
		}
	}
	x := strings.Join(parts, "-")
	if len(x) > 48 {
		x = x[:48]
	}
	return strings.Trim(x, "-")
}

func WriteJSON(w io.Writer, report Report) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}
