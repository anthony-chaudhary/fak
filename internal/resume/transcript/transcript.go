// Package transcript is the ONE Claude Code session-transcript record model the
// resume-family tools share. The three Python fleet tools this family ports
// (resume_sweep.py, stopped_sessions.py, fleet_resume_watchdog.py) each re-implemented
// JSONL parsing, message-text extraction, and role resolution; this package factors
// that into a single place so the sweep, the stopped-session classifier, and the
// watchdog read a transcript record identically.
//
// Parsing is best-effort over real data: a malformed line is skipped, never fatal
// (transcripts are appended live and tails get truncated). Text extraction never
// interprets content — it only flattens the string-or-block-list shapes the harness
// writes. Stdlib-only; file I/O is limited to the two Load helpers the shells use.
package transcript

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"strings"
	"time"
)

// Message is the message envelope of a user/assistant record.
type Message struct {
	Role    string          `json:"role"`
	Model   string          `json:"model"`
	Content json.RawMessage `json:"content"`
}

// Record is one transcript JSONL line — only the fields the resume family reads.
type Record struct {
	Type       string   `json:"type"`
	Timestamp  string   `json:"timestamp"`
	UUID       string   `json:"uuid"`
	IsAPIError bool     `json:"isApiErrorMessage"`
	SessionID  string   `json:"sessionId"`
	Cwd        string   `json:"cwd"`
	GitBranch  string   `json:"gitBranch"`
	Version    string   `json:"version"`
	Message    *Message `json:"message"`
}

// block is one typed content block; Content nests for tool_result.
type block struct {
	Type    string          `json:"type"`
	Text    string          `json:"text"`
	Name    string          `json:"name"`
	Content json.RawMessage `json:"content"`
}

// Role is the record's conversational role: the message envelope's role when
// present, else the record type (the same fallback the Python tools applied).
func (r Record) Role() string {
	if r.Message != nil && r.Message.Role != "" {
		return r.Message.Role
	}
	return r.Type
}

// IsSynthetic reports a harness-injected turn (model "<synthetic>") — the shape a
// usage-limit banner is recorded as.
func (r Record) IsSynthetic() bool {
	return r.Message != nil && r.Message.Model == "<synthetic>"
}

// Text is the record's human text: a bare string content verbatim, or the text
// blocks of a block-list joined with newlines. Tool results are NOT included (the
// sweep/watchdog reading: classification must key on what the model said, not on
// tool output).
func (r Record) Text() string {
	if r.Message == nil {
		return ""
	}
	if s, ok := contentString(r.Message.Content); ok {
		return s
	}
	var parts []string
	for _, b := range contentBlocks(r.Message.Content) {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// TextWithToolResults is the wider extraction the stopped-session classifier uses:
// text blocks plus tool_result contents (recursively flattened), joined with spaces —
// a session parked on a background task often says so only in a tool result.
func (r Record) TextWithToolResults() string {
	if r.Message == nil {
		return ""
	}
	return flattenContent(r.Message.Content)
}

func flattenContent(raw json.RawMessage) string {
	if s, ok := contentString(raw); ok {
		return s
	}
	var parts []string
	for _, b := range contentBlocks(raw) {
		switch b.Type {
		case "text":
			if b.Text != "" {
				parts = append(parts, b.Text)
			}
		case "tool_result":
			if s := flattenContent(b.Content); s != "" {
				parts = append(parts, s)
			}
		}
	}
	return strings.Join(parts, " ")
}

// LastToolUseName is the name of the LAST tool_use block in the record's content, or
// "" — the signal the mid-tool detector pairs against a following tool_result.
func (r Record) LastToolUseName() string {
	if r.Message == nil {
		return ""
	}
	blocks := contentBlocks(r.Message.Content)
	for i := len(blocks) - 1; i >= 0; i-- {
		if blocks[i].Type == "tool_use" {
			return blocks[i].Name
		}
	}
	return ""
}

// HasToolResult reports whether the record's content carries a tool_result block —
// the turn shape that clears a pending tool_use.
func (r Record) HasToolResult() bool {
	if r.Message == nil {
		return false
	}
	for _, b := range contentBlocks(r.Message.Content) {
		if b.Type == "tool_result" {
			return true
		}
	}
	return false
}

func contentString(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, true
	}
	return "", false
}

func contentBlocks(raw json.RawMessage) []block {
	if len(raw) == 0 {
		return nil
	}
	var blocks []block
	if json.Unmarshal(raw, &blocks) != nil {
		return nil
	}
	return blocks
}

// Parse streams JSONL records from r, skipping blank and malformed lines. The buffer
// admits the multi-megabyte tool-result lines real transcripts carry.
func Parse(r io.Reader) []Record {
	var out []Record
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec Record
		if json.Unmarshal(line, &rec) != nil {
			continue
		}
		out = append(out, rec)
	}
	return out
}

// LoadFile parses a whole transcript file; a missing/unreadable file yields nil
// (fail-open, matching every Python reader in the family).
func LoadFile(path string) []Record {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	return Parse(f)
}

// LoadFileTail parses only the last tailBytes of a transcript — the cheap read the
// stopped-session classifier uses on large files. The first (possibly truncated)
// line falls out naturally as a malformed skip. tailBytes <= 0 reads the whole file.
func LoadFileTail(path string, tailBytes int64) []Record {
	if tailBytes <= 0 {
		return LoadFile(path)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	if fi, err := f.Stat(); err == nil && fi.Size() > tailBytes {
		if _, err := f.Seek(fi.Size()-tailBytes, io.SeekStart); err != nil {
			return nil
		}
	}
	return Parse(f)
}

// LastTimestamp is the timestamp string of the LAST record that carries one, or "".
func LastTimestamp(recs []Record) string {
	for i := len(recs) - 1; i >= 0; i-- {
		if recs[i].Timestamp != "" {
			return recs[i].Timestamp
		}
	}
	return ""
}

// UUIDSet is the set of record uuids present — the superset-resolution input.
func UUIDSet(recs []Record) map[string]bool {
	out := make(map[string]bool, len(recs))
	for _, r := range recs {
		if r.UUID != "" {
			out[r.UUID] = true
		}
	}
	return out
}

// TerminalText is the text of the transcript's TERMINAL user/assistant record — the
// last real turn, ignoring trailing control/metadata records (mode/last-prompt/
// summary etc.). Classification must read only this: a banner that sits turns back
// is not the session's current outcome.
func TerminalText(recs []Record) string {
	for i := len(recs) - 1; i >= 0; i-- {
		role := recs[i].Role()
		if role == "user" || role == "assistant" ||
			recs[i].Type == "user" || recs[i].Type == "assistant" {
			return recs[i].Text()
		}
	}
	return ""
}

// ParseTime parses a transcript ISO-8601 timestamp ("2026-06-23T10:05:00Z" or with a
// numeric offset / fractional seconds). The zero time on failure.
func ParseTime(ts string) time.Time {
	if ts == "" {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339} {
		if t, err := time.Parse(layout, ts); err == nil {
			return t
		}
	}
	return time.Time{}
}
