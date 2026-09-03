package sessionrecovery

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const SummarySchema = "fak.session_recovery.summary.v2"

type Summary struct {
	Schema      string        `json:"schema"`
	Mode        string        `json:"mode"`
	StartedAt   string        `json:"started_at"`
	FinishedAt  string        `json:"finished_at"`
	WitnessPath string        `json:"witness_path"`
	Counts      SummaryCounts `json:"counts"`
	Results     []Result      `json:"results"`
}

type SummaryCounts struct {
	Discovered        int `json:"discovered"`
	Selected          int `json:"selected"`
	AlreadyActive     int `json:"already_active"`
	AlreadyReceipted  int `json:"already_receipted"`
	Launched          int `json:"launched"`
	Attempted         int `json:"attempted"`
	ProvenActive      int `json:"proven_active"`
	FailedAndReaped   int `json:"failed_and_reaped"`
	OperatorOwnedLive int `json:"operator_owned_live"`
	Active            int `json:"active"`
	Productive        int `json:"productive"`
	Completed         int `json:"completed"`
	Failed            int `json:"failed"`
	LaunchedUnproven  int `json:"launched_unproven"`
	ExactCardinality  int `json:"exact_cardinality"`
	CardinalityFailed int `json:"cardinality_failed"`
	Probe             int `json:"probe"`
	Substantive       int `json:"substantive"`
	Live              int `json:"live"`
	IdentityBlocked   int `json:"identity_blocked"`
	WaitingReset      int `json:"waiting_reset"`
}

type Result struct {
	ThreadID             string   `json:"thread_id"`
	CWD                  string   `json:"cwd,omitempty"`
	Source               string   `json:"source,omitempty"`
	Provider             string   `json:"provider,omitempty"`
	Harness              string   `json:"harness,omitempty"`
	HarnessSource        string   `json:"harness_source,omitempty"`
	Category             string   `json:"category,omitempty"`
	Action               string   `json:"action,omitempty"`
	Status               string   `json:"status"`
	Reason               string   `json:"reason,omitempty"`
	SelectionReason      string   `json:"selection_reason,omitempty"`
	ReceiptPath          string   `json:"receipt_path,omitempty"`
	GuardedProcessTrees  int      `json:"guarded_process_trees"`
	Remediation          string   `json:"remediation,omitempty"`
	Argv                 []string `json:"argv,omitempty"`
	LaunchedAt           string   `json:"launched_at,omitempty"`
	BaselineCursor       string   `json:"baseline_cursor,omitempty"`
	BaselineAt           string   `json:"baseline_at,omitempty"`
	PostCursor           string   `json:"post_cursor,omitempty"`
	PostAt               string   `json:"post_at,omitempty"`
	Advanced             bool     `json:"advanced"`
	ProgressEvidence     string   `json:"progress_evidence,omitempty"`
	LaunchIdentity       string   `json:"launch_identity,omitempty"`
	HostHandles          []string `json:"host_handles,omitempty"`
	IdentityProvenance   string   `json:"identity_provenance,omitempty"`
	QualifyingEvidenceAt string   `json:"qualifying_evidence_at,omitempty"`
}

func NewSummary(mode string, report InventoryReport, requests []Request, now time.Time) Summary {
	results := make([]Result, 0, len(requests))
	selected := 0
	for _, req := range requests {
		if req.Status == "candidate" {
			selected++
		}
		result := Result{
			ThreadID: req.ThreadID, CWD: req.CWD, Source: req.Source,
			Provider: req.Provider, Harness: req.Harness, HarnessSource: req.HarnessSource, Category: req.Category, Action: req.Action,
			Status: req.Status, Reason: req.Reason, ReceiptPath: req.ReceiptPath,
			Argv: append([]string(nil), req.Argv...), HostHandles: append([]string(nil), req.HostHandles...), IdentityProvenance: req.IdentityProvenance, QualifyingEvidenceAt: req.QualifyingEvidenceAt,
		}
		if baseline := lookupReportRow(report, req.ThreadID); baseline != nil {
			result.BaselineCursor, result.BaselineAt = progressCursor(baseline)
		}
		if req.Status == "candidate" {
			result.SelectionReason = "crashed session has an in-progress turn and no live process tree"
		} else if req.Reason != "" {
			result.SelectionReason = req.Reason
		}
		result.Remediation = Remediation(result)
		results = append(results, result)
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].ThreadID < results[j].ThreadID })
	stamp := now.UTC().Format(time.RFC3339Nano)
	summary := Summary{Schema: SummarySchema, Mode: mode, StartedAt: stamp, FinishedAt: stamp, Results: results}
	summary.Counts.Discovered = len(report.Sessions)
	summary.Counts.Selected = selected
	for _, session := range report.Sessions {
		if session.Thread != nil && guardedTreeCount(session) == 1 {
			summary.Counts.AlreadyActive++
		}
	}
	summary.Recount()
	return summary
}

func (s *Summary) Recount() {
	discovered, selected, alreadyActive := s.Counts.Discovered, s.Counts.Selected, s.Counts.AlreadyActive
	s.Counts = SummaryCounts{Discovered: discovered, Selected: selected, AlreadyActive: alreadyActive}
	for _, result := range s.Results {
		switch result.Category {
		case CategoryProbe:
			s.Counts.Probe++
		case CategoryLive:
			s.Counts.Live++
		case CategoryIdentityBlocked:
			s.Counts.IdentityBlocked++
		case CategorySubstantive:
			s.Counts.Substantive++
		}
		switch result.Status {
		case "launch_intent", "launched_unproven", "active", "productive", "completed", "cardinality_failed", "launch_failed", "verification_failed", "failed_and_reaped", "reap_failed":
			s.Counts.Attempted++
		}
		switch result.Status {
		case "already_receipted":
			s.Counts.AlreadyReceipted++
		case "launched_unproven":
			s.Counts.Launched++
			s.Counts.LaunchedUnproven++
		case "active":
			s.Counts.Launched++
			s.Counts.Active++
			if result.Advanced {
				s.Counts.ProvenActive++
			} else {
				s.Counts.OperatorOwnedLive++
			}
		case "productive":
			s.Counts.Launched++
			s.Counts.Productive++
		case "completed":
			s.Counts.Launched++
			s.Counts.Completed++
		case "cardinality_failed":
			s.Counts.Launched++
			s.Counts.CardinalityFailed++
			s.Counts.Failed++
		case "failed_and_reaped":
			s.Counts.Failed++
			s.Counts.FailedAndReaped++
		case "reap_failed":
			s.Counts.Failed++
			s.Counts.OperatorOwnedLive++
		case "launch_failed", "receipt_failed", "verification_failed":
			s.Counts.Failed++
		case "identity_blocked":
			// Identity-blocked is fully accounted, but not a launch failure: the
			// operator action is login/exact mapping rather than a guessed resume.
		case "waiting_reset":
			s.Counts.WaitingReset++
		}
		if result.GuardedProcessTrees == 1 {
			s.Counts.ExactCardinality++
		}
	}
}

func Observe(before InventoryReport, after InventoryReport, prior Result) Result {
	result := prior
	baseline := lookupReportRow(before, prior.ThreadID)
	current := lookupReportRow(after, prior.ThreadID)
	if result.BaselineCursor == "" && result.BaselineAt == "" {
		result.BaselineCursor, result.BaselineAt = progressCursor(baseline)
	}
	if current == nil {
		result.Status = "launched_unproven"
		result.Reason = "sessiondiag has not observed the resumed thread"
		result.Remediation = Remediation(result)
		return result
	}
	result.PostCursor, result.PostAt = progressCursor(current)
	result.Advanced = cursorAdvanced(result.BaselineCursor, result.BaselineAt, result.PostCursor, result.PostAt, result.LaunchedAt)
	if result.Advanced {
		if result.Provider == ProviderClaude {
			result.ProgressEvidence = "claude_assistant_transcript_advanced"
		} else {
			result.ProgressEvidence = "codex_thread_turn_advanced"
		}
	}
	count := guardedTreeCount(*current)
	result.GuardedProcessTrees = count
	if result.Provider != ProviderClaude && count > 1 {
		result.Status = "cardinality_failed"
		result.Reason = fmt.Sprintf("expected exactly one guarded process tree; observed %d", count)
		result.Remediation = Remediation(result)
		return result
	}
	if terminalTurn(current.LatestTurn) && result.Advanced {
		if strings.EqualFold(strings.TrimSpace(current.LatestTurn.Status), "completed") {
			result.Status = "completed"
			result.Reason = "resumed turn completed"
			result.Remediation = ""
		} else {
			result.Status = "verification_failed"
			result.Reason = "resumed turn ended with status " + current.LatestTurn.Status
			result.Remediation = Remediation(result)
		}
		return result
	}
	if result.Advanced {
		result.Status = "productive"
		result.Reason = "provider transcript/thread advanced after launch"
		result.Remediation = ""
		return result
	}
	result.Status = "launched_unproven"
	if count == 1 {
		result.Reason = "wrapper/process is visible but provider transcript/thread has not advanced"
	} else {
		result.Reason = "provider transcript/thread has not advanced after launch"
	}
	result.Remediation = Remediation(result)
	return result
}

func terminalTurn(turn *Turn) bool {
	if turn == nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(turn.Status)) {
	case "completed", "failed", "cancelled", "canceled", "interrupted":
		return true
	default:
		return false
	}
}

func lookupReportRow(report InventoryReport, id string) *Session {
	for i := range report.Sessions {
		if report.Sessions[i].Thread != nil && report.Sessions[i].Thread.ID == id {
			return &report.Sessions[i]
		}
	}
	return nil
}

func progressCursor(session *Session) (string, string) {
	if session == nil {
		return "", ""
	}
	if session.Cursor != "" || session.CursorAt != "" {
		return session.Cursor, session.CursorAt
	}
	if session.LatestTurn == nil {
		return "", ""
	}
	turn := session.LatestTurn
	// A resumed Codex turn can finish without creating a second turn row. Bind
	// the cursor to both identity and terminal state, and use completed_at when
	// present, so that state transition is witnessed without treating a newly
	// visible but idle wrapper as progress.
	cursor := strings.Join([]string{turn.ID, turn.Status, turn.CompletedAt}, "|")
	at := firstNonBlank(turn.CompletedAt, turn.StartedAt)
	return cursor, at
}

func cursorAdvanced(beforeCursor, beforeAt, afterCursor, afterAt, launchedAt string) bool {
	if afterCursor == "" && afterAt == "" {
		return false
	}
	if afterCursor == beforeCursor && afterAt == beforeAt {
		return false
	}
	afterTime, afterOK := parseTime(afterAt)
	beforeTime, beforeOK := parseTime(beforeAt)
	launchTime, launchOK := parseTime(launchedAt)
	if !afterOK {
		return false
	}
	if beforeOK && !afterTime.After(beforeTime) {
		return false
	}
	if launchOK && !afterTime.After(launchTime) {
		return false
	}
	return true
}

func guardedTreeCount(session Session) int {
	count := 0
	for _, tree := range session.ProcessTrees {
		if tree.HasGuard {
			count++
		}
	}
	return count
}

func TerminalStatus(status string) bool {
	switch status {
	case "productive", "completed", "cardinality_failed", "launch_failed", "receipt_failed", "verification_failed", "failed_and_reaped", "reap_failed", "already_receipted":
		return true
	default:
		return false
	}
}

func Remediation(result Result) string {
	switch result.Status {
	case "candidate":
		if result.ThreadID != "" {
			return "fak session recover --thread " + result.ThreadID + " --live"
		}
	case "already_receipted":
		return "fak-dev sessiondiag --inventory --json --since 24h"
	case "launched_unproven":
		return "fak session recover --thread " + result.ThreadID + " --json"
	case "cardinality_failed":
		return "fak session audit actions --here"
	case "launch_failed", "receipt_failed", "verification_failed", "failed_and_reaped", "reap_failed", "refused":
		if result.ThreadID != "" {
			return "fak session recover --thread " + result.ThreadID + " --json"
		}
	}
	return ""
}

// SummaryPath allocates one durable run-witness path. Nanosecond precision makes
// parallel recovery previews distinct without relying on process-global state.
func SummaryPath(dir string, started time.Time) string {
	return filepath.Join(dir, "run-"+started.UTC().Format("20060102T150405.000000000Z")+".json")
}

// WriteSummary atomically persists the complete cohort/launch/progress witness.
// Preview runs write it too: discovery evidence must survive even when no launch
// is authorized.
func WriteSummary(path string, summary Summary) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("session recovery witness path is empty")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".session-recovery-run-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(summary); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
