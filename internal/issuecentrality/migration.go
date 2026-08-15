package issuecentrality

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

const MigrationSchema = "fak-issue-centrality-migration/1"

type Selection struct {
	Number     int    `json:"number"`
	Centrality string `json:"centrality"`
	Evidence   string `json:"evidence"`
	P1         string `json:"p1"`
	P2         string `json:"p2"`
	P3         string `json:"p3"`
	P4         string `json:"p4"`
}

type Patch struct {
	Number       int    `json:"number"`
	Title        string `json:"title,omitempty"`
	OriginalBody string `json:"original_body"`
	NewBody      string `json:"new_body"`
	Centrality   string `json:"centrality"`
	Evidence     string `json:"evidence"`
}

type MigrationPreview struct {
	Schema  string  `json:"schema"`
	Mode    string  `json:"mode"`
	Patches []Patch `json:"patches"`
}

func PreviewMigration(issues []Issue, selections []Selection) (MigrationPreview, error) {
	plan := MigrationPreview{Schema: MigrationSchema, Mode: "preview", Patches: []Patch{}}
	byNumber := make(map[int]Issue, len(issues))
	for _, issue := range issues {
		byNumber[issue.Number] = issue
	}
	seen := make(map[int]bool, len(selections))
	ordered := append([]Selection(nil), selections...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Number < ordered[j].Number })
	for _, selection := range ordered {
		if selection.Number <= 0 {
			return plan, fmt.Errorf("selection issue number must be positive")
		}
		if seen[selection.Number] {
			return plan, fmt.Errorf("duplicate selection #%d", selection.Number)
		}
		seen[selection.Number] = true
		issue, ok := byNumber[selection.Number]
		if !ok {
			return plan, fmt.Errorf("selected issue #%d is not in the audited input", selection.Number)
		}
		if strings.TrimSpace(selection.Evidence) == "" {
			return plan, fmt.Errorf("selection #%d requires evidence", selection.Number)
		}
		frameText := selectionBlock(selection)
		frame := issuepolicy.AssessProblemFrame(issuepolicy.IssueDraft{Number: issue.Number, Title: issue.Title, Body: frameText})
		if !frame.Ready {
			return plan, fmt.Errorf("selection #%d is invalid: %s", selection.Number, strings.Join(frame.Reasons, ","))
		}
		current := issuepolicy.AssessProblemFrame(issuepolicy.IssueDraft{Number: issue.Number, Title: issue.Title, Body: issue.Body})
		if current.Enforced {
			return plan, fmt.Errorf("selection #%d already declares a problem frame; edit it deliberately instead of appending a second frame", selection.Number)
		}
		newBody := strings.TrimRight(issue.Body, "\r\n") + "\n\n" + frameText
		plan.Patches = append(plan.Patches, Patch{Number: issue.Number, Title: issue.Title, OriginalBody: issue.Body, NewBody: newBody, Centrality: frame.Centrality, Evidence: strings.TrimSpace(selection.Evidence)})
	}
	return plan, nil
}

func selectionBlock(selection Selection) string {
	return fmt.Sprintf("## Problem frame\n- Centrality: %s\n- P1 Context: %s\n- P2 Net value: %s\n- P3 Adaptation: %s\n- P4 Operations: %s\n\nMigration evidence: %s\n", strings.TrimSpace(selection.Centrality), strings.TrimSpace(selection.P1), strings.TrimSpace(selection.P2), strings.TrimSpace(selection.P3), strings.TrimSpace(selection.P4), strings.TrimSpace(selection.Evidence))
}
