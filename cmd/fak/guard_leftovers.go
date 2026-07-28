package main

import (
	"encoding/json"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/headlesslint"
	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
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
	records := transcript.LoadFile(path)
	if len(records) == 0 {
		return headlesslint.UnknownIssuesFiled()
	}
	return headlesslint.WitnessedIssuesFiled(guardIssuesFiled(records))
}

// guardIssuesFiledEvidenceFromRecords is the same count over records a caller already
// holds — the Stop hook reads a bounded tail once and must not pay for a second read.
// truncated says those records are only the tail of a longer run, which makes the count
// a LOWER bound: positive still proves a filing, but zero proves nothing, so it resolves
// to unknown (headlesslint.WitnessedIssuesFiledTail).
func guardIssuesFiledEvidenceFromRecords(records []transcript.Record, truncated bool) headlesslint.IssuesFiledEvidence {
	if len(records) == 0 {
		return headlesslint.UnknownIssuesFiled()
	}
	count := guardIssuesFiled(records)
	if truncated {
		return headlesslint.WitnessedIssuesFiledTail(count)
	}
	return headlesslint.WitnessedIssuesFiled(count)
}

func guardIssuesFiled(records []transcript.Record) int {
	count := 0
	for _, rec := range records {
		for _, use := range rec.ToolUses() {
			count += guardIssueCreatesInToolUse(use)
		}
	}
	return count
}

var guardIssueCreateCommand = regexp.MustCompile(`(?i)(?:^|[\s;&|])(?:[^\s;&|]+[\\/])?(?:fak(?:\.exe)?\s+issue|gh(?:\.exe)?\s+issue)\s+create\b`)

func guardIssueCreatesInToolUse(use transcript.ToolUse) int {
	if guardDirectIssueCreateTool(use.Name) {
		return 1
	}
	var input any
	if len(use.Input) == 0 || json.Unmarshal(use.Input, &input) != nil {
		return 0
	}
	return guardIssueCreatesInInput(input)
}

// guardIssueCreatesInInput recognizes shell command fields and nested native-tool
// requests (for example multi_tool_use.parallel's recipient_name/parameters objects).
// It intentionally ignores arbitrary prose-valued fields such as titles and bodies.
func guardIssueCreatesInInput(v any) int {
	switch x := v.(type) {
	case []any:
		total := 0
		for _, item := range x {
			total += guardIssueCreatesInInput(item)
		}
		return total
	case map[string]any:
		total := 0
		for key, value := range x {
			switch strings.ToLower(key) {
			case "recipient_name", "tool_name":
				if name, ok := value.(string); ok && guardDirectIssueCreateTool(name) {
					total++
				}
			case "command", "cmd", "script":
				if command, ok := value.(string); ok {
					total += len(guardIssueCreateCommand.FindAllStringIndex(command, -1))
				}
			default:
				total += guardIssueCreatesInInput(value)
			}
		}
		return total
	default:
		return 0
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
