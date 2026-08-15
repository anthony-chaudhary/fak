// Package issuecentrality audits canonical problem-frame coverage in an issue portfolio.
package issuecentrality

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

const Schema = "fak-issue-centrality-audit/1"

type Issue struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
}

type Counts struct {
	Core         int `json:"core"`
	Enabling     int `json:"enabling"`
	Stewardship  int `json:"stewardship"`
	Peripheral   int `json:"peripheral"`
	Unknown      int `json:"unknown"`
	Valid        int `json:"valid"`
	Invalid      int `json:"invalid"`
	Unclassified int `json:"unclassified"`
	ProblemFrame int `json:"complete_problem_frame"`
}

type Finding struct {
	Number        int      `json:"number"`
	Title         string   `json:"title,omitempty"`
	Status        string   `json:"status"`
	Centrality    string   `json:"centrality"`
	Target        string   `json:"centrality_target,omitempty"`
	Reasons       []string `json:"reasons"`
	RepairActions []string `json:"repair_actions"`
}

type Report struct {
	Schema      string    `json:"schema"`
	Scope       string    `json:"scope"`
	Provenance  string    `json:"provenance"`
	CollectedAt string    `json:"collected_at,omitempty"`
	Total       int       `json:"total"`
	Classified  int       `json:"classified"`
	CoveragePct float64   `json:"coverage_pct"`
	Counts      Counts    `json:"counts"`
	Findings    []Finding `json:"findings"`
	Errors      []string  `json:"errors"`
}

func Audit(issues []Issue, scope, provenance string, collectedAt time.Time, collectionErrors []string) Report {
	r := Report{Schema: Schema, Scope: scope, Provenance: provenance, Total: len(issues), Errors: append([]string(nil), collectionErrors...)}
	if !collectedAt.IsZero() {
		r.CollectedAt = collectedAt.UTC().Format(time.RFC3339)
	}
	for _, issue := range issues {
		frame := issuepolicy.AssessProblemFrame(issuepolicy.IssueDraft{Number: issue.Number, Title: issue.Title, Body: issue.Body})
		status := "invalid"
		switch {
		case !frame.Enforced || frame.Centrality == issuepolicy.CentralityUnclassified:
			status = "unclassified"
			r.Counts.Unclassified++
		case frame.Ready:
			status = "valid"
			r.Counts.Valid++
		default:
			r.Counts.Invalid++
		}
		reasons := append([]string(nil), frame.Reasons...)
		repairs := append([]string(nil), frame.RepairActions...)
		if status == "unclassified" {
			reasons = []string{"problem_frame_unclassified"}
			repairs = []string{"classify from issue evidence, then declare Centrality and P1-P4; do not infer from title, path, labels, or parent epic"}
		}
		r.Findings = append(r.Findings, Finding{Number: issue.Number, Title: issue.Title, Status: status, Centrality: frame.Centrality, Target: frame.CentralityTarget, Reasons: reasons, RepairActions: repairs})
		switch frame.Centrality {
		case issuepolicy.CentralityCore:
			r.Counts.Core++
		case issuepolicy.CentralityEnabling:
			r.Counts.Enabling++
		case issuepolicy.CentralityStewardship:
			r.Counts.Stewardship++
		case issuepolicy.CentralityPeripheral:
			r.Counts.Peripheral++
		default:
			r.Counts.Unknown++
		}
		if frame.Centrality != issuepolicy.CentralityUnclassified {
			r.Classified++
		}
		if frame.Enforced && frame.Ready {
			r.Counts.ProblemFrame++
		}
	}
	if r.Total > 0 {
		r.CoveragePct = float64(r.Classified) * 100 / float64(r.Total)
	}
	sort.SliceStable(r.Findings, func(i, j int) bool { return r.Findings[i].Number < r.Findings[j].Number })
	sort.Strings(r.Errors)
	if r.Findings == nil {
		r.Findings = []Finding{}
	}
	if r.Errors == nil {
		r.Errors = []string{}
	}
	return r
}

func Decode(data []byte) ([]Issue, error) {
	var issues []Issue
	if err := json.Unmarshal(data, &issues); err != nil {
		return nil, fmt.Errorf("decode issue portfolio: %w", err)
	}
	return issues, nil
}

func (r Report) Text() string {
	var b strings.Builder
	fmt.Fprintf(&b, "CENTRALITY COVERAGE %.1f%% (%d/%d)\nCore %d | Enabling %d | Stewardship %d | Peripheral %d | Unknown %d\nFrame status: valid %d | invalid %d | unclassified %d\nComplete P1-P4 frame %d/%d\n", r.CoveragePct, r.Classified, r.Total, r.Counts.Core, r.Counts.Enabling, r.Counts.Stewardship, r.Counts.Peripheral, r.Counts.Unknown, r.Counts.Valid, r.Counts.Invalid, r.Counts.Unclassified, r.Counts.ProblemFrame, r.Total)
	for _, finding := range r.Findings {
		target := ""
		if finding.Target != "" {
			target = "(" + finding.Target + ")"
		}
		fmt.Fprintf(&b, "#%d [%s] centrality=%s%s", finding.Number, finding.Status, finding.Centrality, target)
		if len(finding.Reasons) > 0 {
			fmt.Fprintf(&b, " reasons=%s", strings.Join(finding.Reasons, ","))
		}
		b.WriteByte('\n')
		for _, repair := range finding.RepairActions {
			fmt.Fprintf(&b, "  repair: %s\n", repair)
		}
	}
	fmt.Fprintf(&b, "Scope: %s\nProvenance: %s\nCollection errors: %d\n", r.Scope, r.Provenance, len(r.Errors))
	return b.String()
}
