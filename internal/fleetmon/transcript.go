package fleetmon

import (
	"bufio"
	"encoding/json"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
)

// TranscriptSignal is the evidence a fleet monitor reads out of one worker's
// Claude Code JSONL transcript. It is derived ONLY from the transcript bytes on
// disk — never from a terminal render and never from the worker's own status —
// so it is a witness, not a self-report.
type TranscriptSignal struct {
	Path            string    `json:"path,omitempty"`
	Exists          bool      `json:"exists"`
	Error           string    `json:"error,omitempty"`
	Lines           int       `json:"lines"`
	LastTimestamp   time.Time `json:"-"`
	HasTimestamp    bool      `json:"-"`
	LastTimestampS  string    `json:"last_timestamp,omitempty"`
	LastRole        string    `json:"last_role,omitempty"` // "assistant" | "user" | ""
	LastStopReason  string    `json:"last_stop_reason,omitempty"`
	FinalReport     bool      `json:"final_report"` // session ended on an assistant text turn (agent stopped, produced a report)
	FinalReportText string    `json:"final_report_text,omitempty"`
	Blocker         string    `json:"blocker,omitempty"`           // human blocker string from the recent tail ("" if none)
	BlockerKind     string    `json:"blocker_kind,omitempty"`      // auth | rate | credit | access | ""
	Interrupted     bool      `json:"interrupted"`                 // the recent tail has an interrupted tool turn
	ChangedFiles    []string  `json:"changed_files,omitempty"`     // paths touched by Edit/Write/NotebookEdit (evidence)
	WitnessCommands []string  `json:"witness_commands,omitempty"`  // test/build/commit commands seen (evidence)
	LastToolUse     string    `json:"last_tool_use,omitempty"`     // name of the last tool_use block
	LastBashCommand string    `json:"last_bash_command,omitempty"` // command text of the last Bash tool_use (child-command correlation)
}

// tailScanLines bounds how many trailing records the blocker scan inspects, so a
// long transcript's early, resolved errors do not read as a current blocker.
const tailScanLines = 40

// ReadTranscript reads a transcript file and folds it into a TranscriptSignal.
// A missing file yields Exists=false (not an error) — a worker may not have
// written a transcript yet. The whole file is streamed once; only bounded state
// is kept, so a large transcript costs no more memory than a small one.
func ReadTranscript(path string) TranscriptSignal {
	sig := TranscriptSignal{Path: path}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return sig
		}
		sig.Error = err.Error()
		return sig
	}
	defer f.Close()
	sig.Exists = true

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 64*1024*1024)

	changed := map[string]bool{}
	witness := map[string]bool{}
	var tail []tRecord
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r tRecord
		if json.Unmarshal([]byte(line), &r); r.Type == "" {
			continue
		}
		sig.Lines++
		if r.Timestamp != "" {
			if ts, ok := parseTS(r.Timestamp); ok {
				sig.LastTimestamp, sig.HasTimestamp, sig.LastTimestampS = ts, true, r.Timestamp
			}
		}
		collectEvidence(r, changed, witness, &sig)
		tail = append(tail, r)
		if len(tail) > tailScanLines {
			tail = tail[1:]
		}
	}
	if err := sc.Err(); err != nil {
		sig.Error = err.Error()
	}

	sig.ChangedFiles = sortedKeys(changed)
	sig.WitnessCommands = sortedKeys(witness)
	classifyTail(tail, &sig)
	return sig
}

// collectEvidence accumulates the file-changing and witness-command evidence
// from one assistant record's tool_use blocks.
func collectEvidence(r tRecord, changed, witness map[string]bool, sig *TranscriptSignal) {
	if r.Type != "assistant" {
		return
	}
	for _, b := range r.blocks() {
		if b.Type != "tool_use" {
			continue
		}
		sig.LastToolUse = b.Name
		switch b.Name {
		case "Edit", "Write", "MultiEdit", "NotebookEdit":
			if p := firstString(b.Input, "file_path", "notebook_path"); p != "" {
				changed[p] = true
			}
		case "Bash", "PowerShell":
			cmd := firstString(b.Input, "command")
			sig.LastBashCommand = cmd
			if isWitnessCommand(cmd) {
				witness[collapseWS(cmd)] = true
			}
		}
	}
}

// classifyTail derives the final-report signal and any current blocker from the
// bounded trailing window of records.
func classifyTail(tail []tRecord, sig *TranscriptSignal) {
	// Final report: the LAST assistant turn stopped with an end_turn and carried a
	// text block (the agent produced an answer and stopped) rather than issuing a
	// tool_use (mid-work) — and no user/tool turn followed it (the session did not
	// resume). We walk from the end for the last assistant record.
	for i := len(tail) - 1; i >= 0; i-- {
		r := tail[i]
		if r.Type == "assistant" {
			sig.LastRole = "assistant"
			sig.LastStopReason = r.Message.StopReason
			text, hasText, hasTool := lastText(r)
			// A trailing assistant text turn with no tool_use after it is a final
			// report. end_turn confirms the model chose to stop; when the field is
			// absent (older transcripts) a text-only trailing turn still counts.
			if hasText && !hasTool && (r.Message.StopReason == "" || r.Message.StopReason == "end_turn") {
				sig.FinalReport = true
				sig.FinalReportText = trimReport(text)
			}
			break
		}
		if r.Type == "user" {
			sig.LastRole = "user" // a tool_result or a follow-up prompt — work in flight
			break
		}
	}
	// Blocker: scan the recent tail's text and tool_result content for an auth or
	// rate-limit signature. Only the tail is scanned so an early, since-resolved
	// error does not read as a live blocker.
	for i := len(tail) - 1; i >= 0; i-- {
		if r := tail[i]; r.Interrupted() {
			sig.Interrupted = true
		}
		kind, reason := blockerIn(tail[i])
		if kind != "" {
			sig.BlockerKind, sig.Blocker = kind, reason
			break
		}
	}
}

// --- blocker taxonomy ----------------------------------------------------- //

var (
	reAuth   = regexp.MustCompile(`(?i)please run /login|invalid api key|authentication_error|oauth token (?:has )?expired|not authenticated|login required|invalid bearer token`)
	reAccess = regexp.MustCompile(`(?i)subscription access.*(?:disabled|not enabled)|organization has disabled|use an anthropic api key instead`)
	reCredit = regexp.MustCompile(`(?i)credit balance is too low|insufficient credit|billing`)
	reRate   = regexp.MustCompile(`(?i)rate[ _-]?limit|429|usage limit reached|overloaded_error|please try again later|too many requests|quota exceeded`)
)

// blockerIn classifies the text of one record into (kind, human reason). The
// order is credit -> access -> auth -> rate so the most specific wins.
func blockerIn(r tRecord) (string, string) {
	text := r.text()
	if text == "" {
		return "", ""
	}
	switch {
	case reCredit.MatchString(text):
		return "credit", "credit balance too low"
	case reAccess.MatchString(text):
		return "access", "subscription access disabled"
	case reAuth.MatchString(text):
		return "auth", "auth/login required"
	case reRate.MatchString(text):
		return "rate", "rate/usage limit hit"
	}
	return "", ""
}

var witnessRE = regexp.MustCompile(`(?i)\b(go test|make ci|make test|make test-fast|make test-race|\./test\.ps1|test\.ps1|pytest|npm (?:run )?test|cargo test|go build|go vet|dos verify|dos commit-audit|dos test-witness|git commit|fak commit|fak sweep)\b`)

// isWitnessCommand reports whether a shell command is a proof/witness step — a
// test, a build/vet, or a commit — the kind of command that makes a patch
// "witnessed" rather than merely claimed.
func isWitnessCommand(cmd string) bool {
	return witnessRE.MatchString(cmd)
}

// --- transcript record shape ---------------------------------------------- //
//
// Only the fields the liveness/outcome fold needs are modeled; the cost-and-
// tokens audit lives in internal/sessionaudit and reads the same files for a
// different purpose.

type tRecord struct {
	Type                 string `json:"type"`
	Timestamp            string `json:"timestamp"`
	InterruptedMessageID string `json:"interruptedMessageId"`
	Message              tMsg   `json:"message"`
}

type tMsg struct {
	Role       string          `json:"role"`
	StopReason string          `json:"stop_reason"`
	Content    json.RawMessage `json:"content"`
}

type tBlock struct {
	Type    string          `json:"type"`
	Name    string          `json:"name"`
	Text    string          `json:"text"`
	Input   json.RawMessage `json:"input"`
	Content json.RawMessage `json:"content"`
	IsError bool            `json:"is_error"`
}

func (r tRecord) blocks() []tBlock {
	if len(r.Message.Content) == 0 {
		return nil
	}
	var blocks []tBlock
	if json.Unmarshal(r.Message.Content, &blocks) == nil {
		return blocks
	}
	return nil
}

func (r tRecord) Interrupted() bool {
	if r.InterruptedMessageID != "" || r.Message.StopReason == "interrupted" {
		return true
	}
	for _, b := range r.blocks() {
		if b.IsError {
			return true
		}
	}
	return false
}

// text returns all human-readable text in a record: assistant text blocks, a
// bare string user message, and tool_result content (where a blocker surfaces).
func (r tRecord) text() string {
	var parts []string
	if len(r.Message.Content) > 0 {
		var s string
		if json.Unmarshal(r.Message.Content, &s) == nil {
			parts = append(parts, s)
		}
		for _, b := range r.blocks() {
			if b.Text != "" {
				parts = append(parts, b.Text)
			}
			if len(b.Content) > 0 {
				parts = append(parts, rawText(b.Content))
			}
		}
	}
	return strings.Join(parts, "\n")
}

// lastText returns the concatenated text of an assistant record, whether it has
// any text block, and whether it has any tool_use block.
func lastText(r tRecord) (text string, hasText, hasTool bool) {
	var parts []string
	for _, b := range r.blocks() {
		switch b.Type {
		case "text":
			if b.Text != "" {
				parts = append(parts, b.Text)
				hasText = true
			}
		case "tool_use":
			hasTool = true
		}
	}
	return strings.Join(parts, "\n"), hasText, hasTool
}

// --- small helpers -------------------------------------------------------- //

func parseTS(s string) (time.Time, bool) {
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05.000Z07:00"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), true
		}
	}
	return time.Time{}, false
}

func firstString(raw json.RawMessage, keys ...string) string {
	if len(raw) == 0 {
		return ""
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return ""
	}
	for _, k := range keys {
		if v, ok := obj[k]; ok {
			var s string
			if json.Unmarshal(v, &s) == nil && s != "" {
				return s
			}
		}
	}
	return ""
}

func rawText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []tBlock
	if json.Unmarshal(raw, &blocks) == nil {
		var parts []string
		for _, b := range blocks {
			if b.Text != "" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

func trimReport(s string) string {
	s = strings.TrimSpace(s)
	const max = 600
	if len(s) > max {
		return s[:max] + "…"
	}
	return s
}

func collapseWS(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func sortedKeys(m map[string]bool) []string {
	if len(m) == 0 {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
