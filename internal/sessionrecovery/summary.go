package sessionrecovery

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

const SummarySchema = "fak.session_recovery.summary.v1"

type Summary struct {
	Schema     string        `json:"schema"`
	Mode       string        `json:"mode"`
	StartedAt  string        `json:"started_at"`
	FinishedAt string        `json:"finished_at"`
	Counts     SummaryCounts `json:"counts"`
	Results    []Result      `json:"results"`
}

type SummaryCounts struct {
	Discovered        int `json:"discovered"`
	Selected          int `json:"selected"`
	AlreadyActive     int `json:"already_active"`
	AlreadyReceipted  int `json:"already_receipted"`
	Launched          int `json:"launched"`
	Active            int `json:"active"`
	Productive        int `json:"productive"`
	Completed         int `json:"completed"`
	Failed            int `json:"failed"`
	LaunchedUnproven  int `json:"launched_unproven"`
	ExactCardinality  int `json:"exact_cardinality"`
	CardinalityFailed int `json:"cardinality_failed"`
}

type Result struct {
	ThreadID            string   `json:"thread_id"`
	CWD                 string   `json:"cwd,omitempty"`
	Source              string   `json:"source,omitempty"`
	Status              string   `json:"status"`
	Reason              string   `json:"reason,omitempty"`
	SelectionReason     string   `json:"selection_reason,omitempty"`
	ReceiptPath         string   `json:"receipt_path,omitempty"`
	GuardedProcessTrees int      `json:"guarded_process_trees"`
	Remediation         string   `json:"remediation,omitempty"`
	Argv                []string `json:"argv,omitempty"`
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
			Status: req.Status, Reason: req.Reason, ReceiptPath: req.ReceiptPath,
			Argv: append([]string(nil), req.Argv...),
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
		switch result.Status {
		case "already_receipted":
			s.Counts.AlreadyReceipted++
		case "launched_unproven":
			s.Counts.Launched++
			s.Counts.LaunchedUnproven++
		case "active":
			s.Counts.Launched++
			s.Counts.Active++
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
		case "launch_failed", "receipt_failed", "verification_failed":
			s.Counts.Failed++
		}
		if result.GuardedProcessTrees == 1 {
			s.Counts.ExactCardinality++
		}
	}
}

func Observe(before InventoryReport, after InventoryReport, prior Result) Result {
	result := prior
	var baseline, current *Session
	for i := range before.Sessions {
		if before.Sessions[i].Thread != nil && before.Sessions[i].Thread.ID == prior.ThreadID {
			baseline = &before.Sessions[i]
		}
	}
	for i := range after.Sessions {
		if after.Sessions[i].Thread != nil && after.Sessions[i].Thread.ID == prior.ThreadID {
			current = &after.Sessions[i]
		}
	}
	if current == nil {
		result.Status = "launched_unproven"
		result.Reason = "sessiondiag has not observed the resumed thread"
		result.Remediation = Remediation(result)
		return result
	}
	count := guardedTreeCount(*current)
	result.GuardedProcessTrees = count
	if count > 1 {
		result.Status = "cardinality_failed"
		result.Reason = fmt.Sprintf("expected exactly one guarded process tree; observed %d", count)
		result.Remediation = Remediation(result)
		return result
	}
	if terminalTurn(current.LatestTurn) && newerTurn(baseline, current) {
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
	if count == 0 || current.GuardReceipt == nil {
		result.Status = "launched_unproven"
		result.Reason = "no guarded process tree and guard receipt observed yet"
		result.Remediation = Remediation(result)
		return result
	}
	if newerTurn(baseline, current) {
		result.Status = "productive"
		result.Reason = "exactly one guarded process tree and a fresh turn were observed"
		result.Remediation = ""
		return result
	}
	result.Status = "active"
	result.Reason = "exactly one guarded process tree is active; waiting for a fresh turn"
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

func newerTurn(before, after *Session) bool {
	if after == nil || after.LatestTurn == nil {
		return false
	}
	return before == nil || before.LatestTurn == nil || after.LatestTurn.StartedAt > before.LatestTurn.StartedAt
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
	case "productive", "completed", "cardinality_failed", "launch_failed", "receipt_failed", "verification_failed", "already_receipted":
		return true
	default:
		return false
	}
}

func Remediation(result Result) string {
	switch result.Status {
	case "candidate":
		if result.ThreadID != "" {
			return "fak session recover --thread " + result.ThreadID + " --apply"
		}
	case "already_receipted":
		return "fak-dev sessiondiag --inventory --json --since 24h"
	case "launched_unproven", "active":
		return "fak session recover --thread " + result.ThreadID + " --json"
	case "cardinality_failed":
		return "fak session audit actions --here"
	case "launch_failed", "receipt_failed", "verification_failed", "refused":
		if result.ThreadID != "" {
			return "fak session recover --thread " + result.ThreadID + " --json"
		}
	}
	return ""
}
