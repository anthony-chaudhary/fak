package main

import (
	"bytes"
	"encoding/json"
	"os"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/headlesslint"
	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
	"github.com/anthony-chaudhary/fak/internal/shelltoken"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// guardLeftoversSignal is the observe-only Stop-hook reading for the end-of-run
// leftovers doctrine. It deliberately carries no stop disposition: #4385 starts as
// a fail-open sensor and must soak before any promotion can change Stop behavior.
//
// FilingKnown is the field that keeps the sensor honest (#5425): when the transcript
// could not be read (or only a bounded tail was read and it showed nothing), IssuesFiled
// stays 0 but FilingKnown is false — "cannot say", not "filed nothing".
type guardLeftoversSignal struct {
	LeftoversUnfiled bool
	FilingUnknown    bool
	Narrated         int
	IssuesFiled      int
	FilingKnown      bool
	FilingSource     string
}

// scanGuardLeftovers folds the final assistant turn against issue-creation calls made
// over the whole run. Tool inputs are used rather than assistant prose, so merely saying
// "filed an issue" cannot satisfy the sensor.
func scanGuardLeftovers(transcriptPath, finalTurn string) guardLeftoversSignal {
	return foldGuardLeftovers(guardIssuesFiledEvidence(transcriptPath), finalTurn)
}

// foldGuardLeftovers is the shared tail of the two readings: evidence in, signal out.
func foldGuardLeftovers(filed headlesslint.IssuesFiledEvidence, finalTurn string) guardLeftoversSignal {
	rep := headlesslint.ScanLeftoversEvidence(finalTurn, filed, false)
	count, known := rep.FiledCount()
	return guardLeftoversSignal{
		LeftoversUnfiled: rep.Refused(),
		FilingUnknown:    rep.Undecided(),
		Narrated:         rep.Narrated,
		IssuesFiled:      count,
		FilingKnown:      known,
		FilingSource:     rep.IssuesFiledSource,
	}
}

// guardIssuesFiledEvidence counts the run's issue filings from the WHOLE transcript at
// path. An empty path, an unreadable file, or a file nothing parses out of yields the
// unknown reading rather than a zero: the fold must be able to tell "this run filed
// nothing" from "there was nothing to look at".
func guardIssuesFiledEvidence(transcriptPath string) headlesslint.IssuesFiledEvidence {
	path := strings.TrimSpace(transcriptPath)
	if path == "" {
		return headlesslint.UnknownIssuesFiled()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return headlesslint.UnknownIssuesFiled()
	}
	rollout := toolproc.IngestRolloutEvidence(bytes.NewReader(data))
	if rollout.Complete {
		count := 0
		for _, call := range rollout.Calls {
			var n int
			var known bool
			switch call.Kind {
			case toolproc.RolloutKindFunctionCall:
				n, known = guardIssueCreatesInToolUse(transcript.ToolUse{Name: call.Name, Input: json.RawMessage(call.Arguments)})
			case toolproc.RolloutKindLocalShell:
				n, known = guardIssueCreatesInArgv(call.Command, true)
			case toolproc.RolloutKindCustomTool:
				// Raw executable inputs (notably functions.exec JavaScript) are
				// opaque. A patch's literal body is not an execution input.
				known = strings.TrimPrefix(strings.ToLower(call.Name), "functions.") == "apply_patch"
			}
			if !known {
				return headlesslint.UnknownIssuesFiled()
			}
			count += n
		}
		return headlesslint.WitnessedIssuesFiled(count)
	}
	// Keep the shared Claude parser best-effort, but never promote rows it
	// discards to complete evidence. Foreign record types fail closed below.
	var records []transcript.Record
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		if line = bytes.TrimSpace(line); len(line) == 0 {
			continue
		}
		var rec transcript.Record
		if json.Unmarshal(line, &rec) != nil {
			return headlesslint.UnknownIssuesFiled()
		}
		records = append(records, rec)
	}
	return guardIssuesFiledEvidenceFromRecords(records, false)
}

// guardIssuesFiledEvidenceFromRecords is the Claude-only count over records a caller
// already holds — the Stop hook reads a bounded tail once and must not pay for a second read.
// truncated says those records are only the tail of a longer run, which makes the count
// a LOWER bound: positive still proves a filing, but zero proves nothing, so it resolves
// to unknown (headlesslint.WitnessedIssuesFiledTail).
func guardIssuesFiledEvidenceFromRecords(records []transcript.Record, truncated bool) headlesslint.IssuesFiledEvidence {
	if len(records) == 0 {
		return headlesslint.UnknownIssuesFiled()
	}
	count := 0
	for _, rec := range records {
		if !guardClaudeRecordSupported(rec) {
			return headlesslint.UnknownIssuesFiled()
		}
		for _, use := range rec.ToolUses() {
			n, known := guardIssueCreatesInToolUse(use)
			if !known {
				return headlesslint.UnknownIssuesFiled()
			}
			count += n
		}
	}
	if truncated {
		return headlesslint.WitnessedIssuesFiledTail(count)
	}
	return headlesslint.WitnessedIssuesFiled(count)
}

func guardClaudeRecordSupported(rec transcript.Record) bool {
	switch rec.Type {
	case "user", "assistant":
		if rec.Message == nil {
			return false
		}
	case "system", "summary", "progress", "file-history-snapshot", "queue-operation", "last-prompt", "mode":
		// Ordinary Claude bookkeeping can have no Message at all.
	default:
		// Codex payloads have already been discarded by transcript.Record.
		return false
	}
	if rec.Message == nil {
		return true
	}
	if bytes.Equal(bytes.TrimSpace(rec.Message.Content), []byte("null")) {
		return false
	}
	var text string
	if json.Unmarshal(rec.Message.Content, &text) == nil {
		return true
	}
	// Match the shared parser's block schema so its second decode cannot
	// discard a malformed text block and silently lose neighboring tool uses.
	var blocks []struct {
		Type    string          `json:"type"`
		Text    string          `json:"text"`
		Name    string          `json:"name"`
		Input   json.RawMessage `json:"input"`
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(rec.Message.Content, &blocks) != nil || blocks == nil {
		return false
	}
	for _, block := range blocks {
		switch block.Type {
		case "text", "tool_result", "thinking", "redacted_thinking", "image", "document":
		case "tool_use":
			if strings.TrimSpace(block.Name) == "" {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// Count only attempted invocations in supported input schemas, never output or
// recursively discovered prose. Unknown execution wrappers invalidate coverage.
func guardIssueCreatesInToolUse(use transcript.ToolUse) (int, bool) {
	var input map[string]json.RawMessage
	if json.Unmarshal(use.Input, &input) != nil || input == nil {
		return 0, false
	}
	if guardDirectIssueCreateTool(use.Name) {
		return 1, true
	}
	name := strings.TrimPrefix(strings.ToLower(use.Name), "functions.")
	switch name {
	case "multi_tool_use.parallel":
		var nested []struct {
			Name  string          `json:"recipient_name"`
			Input json.RawMessage `json:"parameters"`
		}
		if json.Unmarshal(input["tool_uses"], &nested) != nil || nested == nil {
			return 0, false
		}
		total := 0
		for _, call := range nested {
			n, known := guardIssueCreatesInToolUse(transcript.ToolUse{Name: call.Name, Input: call.Input})
			if !known {
				return 0, false
			}
			total += n
		}
		return total, true
	case "bash", "sh", "shell", "local_shell", "container.exec", "exec_command", "shell_command", "powershell", "pwsh":
		var command json.RawMessage
		for _, key := range []string{"command", "cmd", "script"} {
			if value, ok := input[key]; ok {
				if command != nil {
					return 0, false
				}
				command = value
			}
		}
		var text string
		if json.Unmarshal(command, &text) == nil {
			return guardIssueCreatesInCommand(text)
		}
		var argv []string
		if json.Unmarshal(command, &argv) == nil {
			return guardIssueCreatesInArgv(argv, true)
		}
		return 0, false
	case "apply_patch", "update_plan", "read", "write", "edit", "multiedit", "glob", "grep", "ls", "webfetch", "websearch", "todowrite":
		return 0, true
	default:
		return 0, false
	}
}

var guardCommandWord = regexp.MustCompile(`"[^"\r\n]*"|'[^'\r\n]*'|[^\s"']+`)

func guardIssueCreatesInCommand(command string) (int, bool) {
	// Do not evaluate expansions, script blocks, or nested interpreters. Even
	// a create-looking fragment inside such input cannot establish coverage.
	if strings.ContainsAny(command, "$`{}()") {
		return 0, false
	}
	segments, ok := vdso.SplitCompoundCommand(command)
	if !ok || len(segments) == 0 {
		return 0, false
	}
	total := 0
	for _, segment := range segments {
		var argv []string
		end := 0
		for _, span := range guardCommandWord.FindAllStringIndex(segment.RawCmd, -1) {
			if strings.TrimSpace(segment.RawCmd[end:span[0]]) != "" || (end > 0 && end == span[0]) {
				return 0, false
			}
			word := segment.RawCmd[span[0]:span[1]]
			if word[0] == '\'' || word[0] == '"' {
				word = word[1 : len(word)-1]
			} else if strings.ContainsAny(word, "*?[]#") || strings.HasSuffix(word, `\`) {
				return 0, false
			}
			argv = append(argv, word)
			end = span[1]
		}
		if strings.TrimSpace(segment.RawCmd[end:]) != "" {
			return 0, false
		}
		n, known := guardIssueCreatesInArgv(argv, false)
		if !known {
			return 0, false
		}
		total += n
	}
	return total, true
}

func guardIssueCreatesInArgv(argv []string, unwrap bool) (int, bool) {
	if len(argv) == 0 {
		return 0, false
	}
	program := shelltoken.ProgramBasename(argv[0])
	switch program {
	case "bash", "sh", "zsh", "dash", "pwsh", "powershell":
		if unwrap && len(argv) == 3 {
			switch strings.ToLower(argv[1]) {
			case "-c", "-lc", "-command":
				return guardIssueCreatesInCommand(argv[2])
			}
		}
		return 0, false
	case "go":
		if len(argv) < 3 || argv[1] != "run" {
			return 0, false
		}
		target := strings.TrimSuffix(strings.ReplaceAll(argv[2], `\`, "/"), "/")
		if target != "cmd/fak-dev" && !strings.HasSuffix(target, "/cmd/fak-dev") {
			return 0, false
		}
		argv = argv[2:]
	case "gh", "fak", "fak-dev":
	case "git":
		return 0, len(argv) > 1 && argv[1] == "status"
	case "echo", "write-output":
		return 0, true
	default:
		return 0, false
	}
	if len(argv) < 3 || argv[1] != "issue" {
		return 0, false
	}
	switch argv[2] {
	case "create":
		return 1, true
	case "list", "view", "search", "comment", "edit", "close", "reopen", "status":
		return 0, true
	default:
		return 0, false
	}
}

// guardNotAFilingWord names the verbs and nouns that make an issue-shaped tool name
// something OTHER than filing a new issue — commenting on one, listing them, closing
// one. They exist to bias the count DOWN: `create_issue_comment` reads as "create" plus
// "issue" but files nothing, and a count that runs high is worse than one that runs low,
// because the whole value of an evidence-derived number is that it can be trusted.
var guardNotAFilingWord = map[string]bool{
	"comment": true, "comments": true, "commented": true,
	"list": true, "search": true, "get": true, "read": true, "view": true, "fetch": true,
	"close": true, "closed": true, "reopen": true, "update": true, "edit": true, "delete": true,
	"label": true, "assign": true, "link": true, "transfer": true, "template": true, "templates": true,
}

func guardDirectIssueCreateTool(name string) bool {
	normalized := strings.NewReplacer("_", " ", "-", " ", ".", " ", "/", " ", ":", " ").Replace(strings.ToLower(name))
	words := strings.Fields(normalized)
	hasIssue, hasCreate := false, false
	for _, word := range words {
		if guardNotAFilingWord[word] {
			return false
		}
		switch word {
		case "issue", "issues":
			hasIssue = true
		case "create":
			hasCreate = true
		case "createissue", "createissues":
			// A camelCase name collapses to one word once separators are stripped;
			// `createIssue` is still unambiguously a filing.
			hasIssue, hasCreate = true, true
		}
	}
	return hasIssue && hasCreate
}
