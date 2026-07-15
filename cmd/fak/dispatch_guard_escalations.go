package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// humanBlockedGuardEscalations folds each session to its latest Stop row and projects
// only genuine HUMAN_RESIDUAL outcomes into the dispatcher's existing closed token.
// Session ids are opaque, so Number stays zero and Title carries the stable identity.
func humanBlockedGuardEscalations(ledgerPath string) []dispatchtick.SkippedIssue {
	ledgerPath = strings.TrimSpace(ledgerPath)
	if ledgerPath == "" {
		return nil
	}
	b, err := os.ReadFile(ledgerPath)
	if err != nil {
		return nil
	}
	rows := jsonlledger.Parse(string(b), func(r guardStopRecord) bool { return r.Schema == guardStopRecordSchema })
	latest := map[string]guardStopRecord{}
	order := []string{}
	for _, row := range rows {
		session := strings.TrimSpace(row.Session)
		if session == "" {
			continue
		}
		if _, seen := latest[session]; !seen {
			order = append(order, session)
		}
		latest[session] = row
	}
	out := make([]dispatchtick.SkippedIssue, 0, len(order))
	for _, session := range order {
		row := latest[session]
		if guardStopDisposition(row.Disposition) != stopDispOperatorDirectedEscalate {
			continue
		}
		out = append(out, dispatchtick.SkippedIssue{
			Title:      "guard session " + session,
			Reason:     reasonBlockedByHuman,
			NextAction: "operator must resolve the HUMAN_RESIDUAL escalation before dispatch resumes",
			WorkUnit:   "session",
		})
	}
	return out
}

func mergeHumanBlockedSkipped(routerRows, guardRows []dispatchtick.SkippedIssue) []dispatchtick.SkippedIssue {
	out := make([]dispatchtick.SkippedIssue, 0, len(routerRows)+len(guardRows))
	seen := map[string]bool{}
	add := func(row dispatchtick.SkippedIssue) {
		key := strings.ToLower(strings.TrimSpace(row.Title))
		if row.Number != 0 {
			key = fmt.Sprintf("issue:%d", row.Number)
		}
		if key == "" || seen[key] {
			return
		}
		seen[key] = true
		out = append(out, row)
	}
	for _, row := range routerRows {
		add(row)
	}
	for _, row := range guardRows {
		add(row)
	}
	return out
}
