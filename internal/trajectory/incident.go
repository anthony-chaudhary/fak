package trajectory

import (
	"bufio"
	"bytes"
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

const (
	IncidentDefaultMaxFiles        = 2000
	IncidentDefaultMaxBytesPerFile = int64(12 << 20)
	IncidentDefaultMaxBytesTotal   = int64(128 << 20)
	IncidentDefaultMaxDuration     = 10 * time.Second
)

type IncidentOptions struct {
	Root            string
	Tag             string
	Since           time.Time
	Until           time.Time
	Restart         time.Time
	PromptSHA256    string
	MaxFiles        int
	MaxBytesPerFile int64
	MaxBytesTotal   int64
	MaxDuration     time.Duration
	Now             time.Time
}

type IncidentSession struct {
	SessionID  string    `json:"session_id"`
	Start      time.Time `json:"start"`
	Source     string    `json:"source"`
	ParentID   string    `json:"parent_id,omitempty"`
	RootID     string    `json:"root_id"`
	Boundary   string    `json:"boundary,omitempty"`
	SourcePath string    `json:"source_path"`
}

type IncidentCount struct {
	Key   string `json:"key"`
	Count int    `json:"count"`
}

type IncidentLimits struct {
	MaxFiles        int   `json:"max_files"`
	MaxBytesPerFile int64 `json:"max_bytes_per_file"`
	MaxBytesTotal   int64 `json:"max_bytes_total"`
	MaxDurationMS   int64 `json:"max_duration_ms"`
	FilesScanned    int   `json:"files_scanned"`
	BytesScanned    int64 `json:"bytes_scanned"`
	FilesSkipped    int   `json:"files_skipped"`
	Truncated       bool  `json:"truncated"`
}

type IncidentPacket struct {
	Schema       string            `json:"schema"`
	Root         string            `json:"root"`
	Tag          string            `json:"tag,omitempty"`
	PromptSHA256 string            `json:"prompt_sha256,omitempty"`
	Since        *time.Time        `json:"since,omitempty"`
	Until        *time.Time        `json:"until,omitempty"`
	Restart      *time.Time        `json:"restart,omitempty"`
	Sessions     []IncidentSession `json:"sessions"`
	BySource     []IncidentCount   `json:"by_source"`
	ByRoot       []IncidentCount   `json:"by_root"`
	ByBoundary   []IncidentCount   `json:"by_boundary,omitempty"`
	Limits       IncidentLimits    `json:"limits"`
}

type incidentMeta struct {
	ID       string
	Start    time.Time
	Source   string
	ParentID string
	Prompt   string
	Path     string
}

func RunIncident(opts IncidentOptions) (IncidentPacket, error) {
	if strings.TrimSpace(opts.Root) == "" || (strings.TrimSpace(opts.Tag) == "" && strings.TrimSpace(opts.PromptSHA256) == "") {
		return IncidentPacket{}, errors.New("trajectory incident: root and tag or prompt sha256 are required")
	}
	if opts.PromptSHA256 != "" {
		raw, err := hex.DecodeString(opts.PromptSHA256)
		if err != nil || len(raw) != sha256.Size {
			return IncidentPacket{}, errors.New("trajectory incident: prompt sha256 must be 64 hexadecimal characters")
		}
		opts.PromptSHA256 = strings.ToLower(opts.PromptSHA256)
	}
	if opts.MaxFiles <= 0 {
		opts.MaxFiles = IncidentDefaultMaxFiles
	}
	if opts.MaxBytesPerFile <= 0 {
		opts.MaxBytesPerFile = IncidentDefaultMaxBytesPerFile
	}
	if opts.MaxBytesTotal <= 0 {
		opts.MaxBytesTotal = IncidentDefaultMaxBytesTotal
	}
	if opts.MaxDuration <= 0 {
		opts.MaxDuration = IncidentDefaultMaxDuration
	}
	if opts.Now.IsZero() {
		opts.Now = time.Now()
	}
	deadline := time.Now().Add(opts.MaxDuration)
	packet := IncidentPacket{Schema: "fak-trajectory-incident/1", Root: filepath.Clean(opts.Root), Tag: opts.Tag, PromptSHA256: opts.PromptSHA256}
	if !opts.Since.IsZero() {
		t := opts.Since.UTC()
		packet.Since = &t
	}
	if !opts.Until.IsZero() {
		t := opts.Until.UTC()
		packet.Until = &t
	}
	if !opts.Restart.IsZero() {
		t := opts.Restart.UTC()
		packet.Restart = &t
	}
	packet.Limits = IncidentLimits{MaxFiles: opts.MaxFiles, MaxBytesPerFile: opts.MaxBytesPerFile, MaxBytesTotal: opts.MaxBytesTotal, MaxDurationMS: opts.MaxDuration.Milliseconds()}

	all := map[string]incidentMeta{}
	err := filepath.WalkDir(opts.Root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			packet.Limits.FilesSkipped++
			return nil
		}
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".jsonl") {
			return nil
		}
		if packet.Limits.FilesScanned >= opts.MaxFiles || packet.Limits.BytesScanned >= opts.MaxBytesTotal || time.Now().After(deadline) {
			packet.Limits.Truncated = true
			return fs.SkipAll
		}
		remaining := opts.MaxBytesTotal - packet.Limits.BytesScanned
		limit := min(opts.MaxBytesPerFile, remaining)
		meta, used, truncated, parseErr := readIncidentSession(path, limit)
		packet.Limits.FilesScanned++
		packet.Limits.BytesScanned += used
		packet.Limits.Truncated = packet.Limits.Truncated || truncated
		if parseErr != nil {
			packet.Limits.FilesSkipped++
			return nil
		}
		if meta.ID != "" {
			all[meta.ID] = meta
		}
		return nil
	})
	if err != nil {
		return IncidentPacket{}, fmt.Errorf("trajectory incident: scan %s: %w", opts.Root, err)
	}

	for _, meta := range all {
		if meta.Start.IsZero() || (!opts.Since.IsZero() && meta.Start.Before(opts.Since)) || (!opts.Until.IsZero() && meta.Start.After(opts.Until)) || !launchIdentityMatches(meta.Prompt, opts.Tag, opts.PromptSHA256) {
			continue
		}
		row := IncidentSession{SessionID: meta.ID, Start: meta.Start.UTC(), Source: meta.Source, ParentID: meta.ParentID, RootID: incidentRootID(meta, all), SourcePath: meta.Path}
		if !opts.Restart.IsZero() {
			if meta.Start.Before(opts.Restart) {
				row.Boundary = "before_restart"
			} else {
				row.Boundary = "after_restart"
			}
		}
		packet.Sessions = append(packet.Sessions, row)
	}
	sort.Slice(packet.Sessions, func(i, j int) bool {
		if !packet.Sessions[i].Start.Equal(packet.Sessions[j].Start) {
			return packet.Sessions[i].Start.Before(packet.Sessions[j].Start)
		}
		return packet.Sessions[i].SessionID < packet.Sessions[j].SessionID
	})
	packet.BySource = incidentCounts(packet.Sessions, func(s IncidentSession) string { return s.Source })
	packet.ByRoot = incidentCounts(packet.Sessions, func(s IncidentSession) string { return s.RootID })
	if !opts.Restart.IsZero() {
		packet.ByBoundary = incidentCounts(packet.Sessions, func(s IncidentSession) string { return s.Boundary })
	}
	return packet, nil
}

func readIncidentSession(path string, limit int64) (incidentMeta, int64, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return incidentMeta{}, 0, false, err
	}
	defer f.Close()
	reader := bufio.NewReader(io.LimitReader(f, limit+1))
	var meta incidentMeta
	meta.Path = filepath.ToSlash(path)
	var used int64
	for {
		line, readErr := reader.ReadBytes('\n')
		used += int64(len(line))
		if used > limit {
			return meta, limit, true, nil
		}
		if len(bytes.TrimSpace(line)) > 0 {
			var row struct {
				Timestamp time.Time       `json:"timestamp"`
				Type      string          `json:"type"`
				Payload   json.RawMessage `json:"payload"`
			}
			if json.Unmarshal(line, &row) == nil {
				switch row.Type {
				case "session_meta":
					var payload struct {
						ID         string          `json:"id"`
						Timestamp  time.Time       `json:"timestamp"`
						Originator string          `json:"originator"`
						Source     json.RawMessage `json:"source"`
					}
					if json.Unmarshal(row.Payload, &payload) == nil {
						meta.ID, meta.Start = payload.ID, payload.Timestamp
						if meta.Start.IsZero() {
							meta.Start = row.Timestamp
						}
						meta.Source, meta.ParentID = incidentSource(payload.Originator, payload.Source)
					}
				case "response_item":
					if meta.Prompt == "" {
						meta.Prompt = incidentUserPrompt(row.Payload)
					}
				}
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return meta, used, false, readErr
		}
	}
	return meta, used, false, nil
}

func incidentSource(originator string, raw json.RawMessage) (string, string) {
	var source string
	if json.Unmarshal(raw, &source) == nil && source != "" {
		return source, ""
	}
	var nested struct {
		Subagent struct {
			ThreadSpawn struct {
				ParentThreadID string `json:"parent_thread_id"`
			} `json:"thread_spawn"`
		} `json:"subagent"`
	}
	if json.Unmarshal(raw, &nested) == nil && nested.Subagent.ThreadSpawn.ParentThreadID != "" {
		return "subagent", nested.Subagent.ThreadSpawn.ParentThreadID
	}
	if strings.Contains(originator, "exec") {
		return "exec", ""
	}
	return "cli", ""
}

func incidentUserPrompt(raw json.RawMessage) string {
	var payload struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if json.Unmarshal(raw, &payload) != nil || payload.Type != "message" || payload.Role != "user" {
		return ""
	}
	var parts []string
	for _, item := range payload.Content {
		if item.Type == "input_text" {
			parts = append(parts, item.Text)
		}
	}
	text := strings.TrimSpace(strings.Join(parts, "\n"))
	if strings.HasPrefix(text, "<environment_context>") {
		return ""
	}
	return text
}

func launchIdentityMatches(prompt, tag, promptSHA256 string) bool {
	if prompt == "" {
		return false
	}
	if promptSHA256 != "" {
		sum := sha256.Sum256([]byte(strings.ReplaceAll(prompt, "\r\n", "\n")))
		if hex.EncodeToString(sum[:]) != promptSHA256 {
			return false
		}
	}
	if tag == "" {
		return true
	}
	for _, line := range strings.Split(prompt, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return strings.Contains(line, tag)
		}
	}
	return false
}

func incidentRootID(meta incidentMeta, all map[string]incidentMeta) string {
	root, seen := meta.ID, map[string]bool{meta.ID: true}
	parent := meta.ParentID
	for parent != "" && !seen[parent] {
		seen[parent] = true
		root = parent
		next, ok := all[parent]
		if !ok {
			break
		}
		parent = next.ParentID
	}
	return root
}

func incidentCounts(rows []IncidentSession, key func(IncidentSession) string) []IncidentCount {
	counts := map[string]int{}
	for _, row := range rows {
		counts[key(row)]++
	}
	out := make([]IncidentCount, 0, len(counts))
	for k, count := range counts {
		out = append(out, IncidentCount{Key: k, Count: count})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}
