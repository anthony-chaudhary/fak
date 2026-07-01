package fleetmon

import (
	"bufio"
	"encoding/json"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// RunLedgerSchema tags each run-ledger row.
const RunLedgerSchema = "fak-fleet-runledger/1"

// Outcome is the closed vocabulary of witnessed run outcomes. Every completed
// worker resolves to exactly one; a row with an out-of-vocabulary outcome is a
// validator defect.
type Outcome string

const (
	OutcomePatchWitness    Outcome = "patch-with-witness"  // changed files AND a captured witness (test/build/commit)
	OutcomeBlockedScoped   Outcome = "blocked-scoped"      // a final report that is blocked/scoped, with the smallest follow-up
	OutcomeReadOnlyAudit   Outcome = "read-only-audit"     // a final report with no file changes (an audit/scoped read)
	OutcomeCrashedNoFinal  Outcome = "crashed-no-final"    // the worker process is gone and left no final report
	OutcomeStaleIncomplete Outcome = "stale-incomplete"    // idle or busy but no final report — NOT complete
	OutcomeSuperseded      Outcome = "superseded"          // a replacement session took over this issue
)

var validOutcomes = map[Outcome]bool{
	OutcomePatchWitness:    true,
	OutcomeBlockedScoped:   true,
	OutcomeReadOnlyAudit:   true,
	OutcomeCrashedNoFinal:  true,
	OutcomeStaleIncomplete: true,
	OutcomeSuperseded:      true,
}

// IsValidOutcome reports membership in the closed outcome vocabulary.
func IsValidOutcome(o Outcome) bool { return validOutcomes[o] }

// Completed reports whether an outcome represents a finished, accounted-for
// worker (as opposed to one still in flight). superseded/crashed/stale are all
// accounted for; only a worker with no row at all is "unaccounted".
func (o Outcome) Completed() bool { return validOutcomes[o] }

// LedgerRow is one durable, append-only witnessed outcome record for one worker.
// The evidence fields are what make the ledger a WITNESS rather than a checklist:
// a patch-with-witness row must carry the changed files and the witness command,
// a blocked row must carry the blocker or the follow-up.
type LedgerRow struct {
	Schema       string   `json:"schema"`
	RunID        string   `json:"run_id,omitempty"`
	Issue        int      `json:"issue"`
	Session      string   `json:"session"`
	Outcome      string   `json:"outcome"`
	ChangedFiles []string `json:"changed_files,omitempty"`
	Witness      string   `json:"witness,omitempty"`
	Blocker      string   `json:"blocker,omitempty"`
	FollowUp     string   `json:"follow_up,omitempty"`
	Supersedes   string   `json:"supersedes,omitempty"`    // a session id this row's session replaced
	SupersededBy string   `json:"superseded_by,omitempty"` // the session id that replaced this row's session
	FinalReport  string   `json:"final_report,omitempty"`  // the extracted final-report text (trimmed)
	RecordedAt   string   `json:"recorded_at"`
}

// FoldInput is the per-worker evidence the fold classifies into a ledger row.
type FoldInput struct {
	RunID      string
	Worker     PlanWorker
	Transcript TranscriptSignal
	PIDAlive   *bool  // nil = unknown; false = the worker process is gone
	Idle       bool   // the transcript is not advancing (a settled worker)
	Superseded bool   // a replacement session exists for this issue
	SupersededBy string
	Now        time.Time
}

// FoldWorker folds one worker's transcript + liveness into a witnessed ledger
// row. It never marks a worker complete just because it is idle: an idle worker
// with no final report is stale-incomplete, not done.
func FoldWorker(in FoldInput) LedgerRow {
	t := in.Transcript
	row := LedgerRow{
		Schema:      RunLedgerSchema,
		RunID:       in.RunID,
		Issue:       in.Worker.Issue,
		Session:     in.Worker.Session,
		RecordedAt:  in.Now.UTC().Format(time.RFC3339),
		FinalReport: t.FinalReportText,
	}
	if in.Worker.ReplacementOf != "" {
		row.Supersedes = in.Worker.ReplacementOf
	}

	switch {
	case in.Superseded:
		row.Outcome = string(OutcomeSuperseded)
		row.SupersededBy = in.SupersededBy
		row.FollowUp = "replaced by " + in.SupersededBy

	case !t.FinalReport:
		if in.PIDAlive != nil && !*in.PIDAlive {
			row.Outcome = string(OutcomeCrashedNoFinal)
			row.FollowUp = "worker process gone with no final report; re-dispatch or inspect the transcript tail"
		} else {
			row.Outcome = string(OutcomeStaleIncomplete)
			row.FollowUp = "idle/busy without a final report; not complete — monitor or replace"
		}

	default: // a final report is present — classify by the witnessed evidence
		blocked, followUp := blockedFinal(t)
		row.Witness = strings.Join(t.WitnessCommands, "; ")
		switch {
		case len(t.ChangedFiles) > 0 && len(t.WitnessCommands) > 0 && !blocked:
			row.Outcome = string(OutcomePatchWitness)
			row.ChangedFiles = t.ChangedFiles
		case blocked:
			row.Outcome = string(OutcomeBlockedScoped)
			row.ChangedFiles = t.ChangedFiles
			row.Blocker = firstNonEmpty(t.Blocker, "reported blocked/scoped in final report")
			row.FollowUp = followUp
		case len(t.ChangedFiles) == 0:
			row.Outcome = string(OutcomeReadOnlyAudit)
		default:
			// Changed files but no captured witness: a claim, not a proof.
			row.Outcome = string(OutcomeBlockedScoped)
			row.ChangedFiles = t.ChangedFiles
			row.FollowUp = "changed files but no witness command captured — add a test/build/commit witness"
		}
	}
	return row
}

// blockedFinal inspects a final report's text for a blocked/scoped signal and
// extracts the smallest follow-up sentence.
func blockedFinal(t TranscriptSignal) (bool, string) {
	if t.Blocker != "" {
		return true, "clear the blocker: " + t.Blocker
	}
	text := t.FinalReportText
	if text == "" {
		return false, ""
	}
	if blockedTextRE.MatchString(text) {
		return true, smallestFollowUp(text)
	}
	return false, ""
}

var blockedTextRE = regexp.MustCompile(`(?i)\bnot yet\b|\bblocked\b|\bcannot\b|\bcan't\b|\bunable to\b|\bfollow[- ]up\b|\bneeds? (?:a )?(?:follow|human|decision)\b|\bout of scope\b|\bwaiting on\b`)

var followUpRE = regexp.MustCompile(`(?i)(?:follow[- ]up|next step|todo|remaining|still needs?)[:\s-]+([^\n.]+)`)

func smallestFollowUp(text string) string {
	if m := followUpRE.FindStringSubmatch(text); m != nil {
		return strings.TrimSpace(m[1])
	}
	// Fall back to the first blocked-signal sentence.
	for _, line := range strings.Split(text, "\n") {
		if blockedTextRE.MatchString(line) {
			return trimReport(strings.TrimSpace(line))
		}
	}
	return "resolve the reported blocker"
}

// --- ledger I/O (mirrors internal/nightrun's JSONL shape) ----------------- //

// ParseLedger reads an append-only JSONL run ledger, tolerating blank lines and
// skipping any line that is not a complete row (so a hand-edit cannot crash the
// reader). Rows are returned in file order.
func ParseLedger(content string) []LedgerRow {
	var rows []LedgerRow
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row LedgerRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Session == "" && row.Issue == 0 {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// AppendLedgerLine renders the JSONL line for a row (no trailing newline); the
// caller appends the newline. Keeping the rendering pure makes the writer
// testable without touching disk.
func AppendLedgerLine(row LedgerRow) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// LedgerDefect is one reason a ledger row failed validation, keyed to its
// 1-based line so a CI report points at the exact row to fix.
type LedgerDefect struct {
	Line    int    `json:"line"`
	Session string `json:"session"`
	Outcome string `json:"outcome"`
	Reason  string `json:"reason"`
}

// ValidateLedger checks the witnessed-evidence invariants: an outcome in the
// closed vocabulary, and the evidence a given outcome REQUIRES to be honest —
// a patch-with-witness must carry changed files + a witness; a blocked row must
// carry a blocker or a follow-up; a superseded row must name its successor.
func ValidateLedger(rows []LedgerRow) []LedgerDefect {
	var defects []LedgerDefect
	for i, r := range rows {
		line := i + 1
		add := func(reason string) {
			defects = append(defects, LedgerDefect{Line: line, Session: r.Session, Outcome: r.Outcome, Reason: reason})
		}
		if !IsValidOutcome(Outcome(r.Outcome)) {
			add("off-schema outcome (not in the closed run-outcome vocabulary)")
			continue
		}
		switch Outcome(r.Outcome) {
		case OutcomePatchWitness:
			if len(r.ChangedFiles) == 0 {
				add("patch-with-witness row has no changed_files")
			}
			if strings.TrimSpace(r.Witness) == "" {
				add("patch-with-witness row has no witness")
			}
		case OutcomeBlockedScoped:
			if strings.TrimSpace(r.Blocker) == "" && strings.TrimSpace(r.FollowUp) == "" {
				add("blocked-scoped row needs a blocker or a follow-up")
			}
		case OutcomeSuperseded:
			if strings.TrimSpace(r.SupersededBy) == "" {
				add("superseded row must name superseded_by")
			}
		}
	}
	return defects
}

// --- markdown fold summary ------------------------------------------------ //

// RunLedgerSummary is the aggregate the fold emits alongside the JSONL rows.
type RunLedgerSummary struct {
	Schema     string           `json:"schema"`
	RunID      string           `json:"run_id,omitempty"`
	Total      int              `json:"total"`
	ByOutcome  map[Outcome]int  `json:"by_outcome"`
	Rows       []LedgerRow      `json:"rows"`
	Defects    []LedgerDefect   `json:"defects,omitempty"`
}

// Summarize folds rows into a run summary with an outcome histogram and any
// validator defects.
func Summarize(runID string, rows []LedgerRow) RunLedgerSummary {
	byOutcome := map[Outcome]int{}
	for _, r := range rows {
		byOutcome[Outcome(r.Outcome)]++
	}
	return RunLedgerSummary{
		Schema:    RunLedgerSchema,
		RunID:     runID,
		Total:     len(rows),
		ByOutcome: byOutcome,
		Rows:      rows,
		Defects:   ValidateLedger(rows),
	}
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// itoa keeps a local, allocation-free int->string for the renderers.
func itoa(n int) string { return strconv.Itoa(n) }
