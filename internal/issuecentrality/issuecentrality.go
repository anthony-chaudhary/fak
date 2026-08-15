// Package issuecentrality audits canonical problem-frame coverage in an issue portfolio.
package issuecentrality

import (
	"encoding/json"
	"fmt"
	"sort"
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
	ProblemFrame int `json:"complete_problem_frame"`
}

type Report struct {
	Schema      string   `json:"schema"`
	Scope       string   `json:"scope"`
	Provenance  string   `json:"provenance"`
	CollectedAt string   `json:"collected_at,omitempty"`
	Total       int      `json:"total"`
	Classified  int      `json:"classified"`
	CoveragePct float64  `json:"coverage_pct"`
	Counts      Counts   `json:"counts"`
	Errors      []string `json:"errors"`
}

func Audit(issues []Issue, scope, provenance string, collectedAt time.Time, collectionErrors []string) Report {
	r := Report{Schema: Schema, Scope: scope, Provenance: provenance, Total: len(issues), Errors: append([]string(nil), collectionErrors...)}
	if !collectedAt.IsZero() {
		r.CollectedAt = collectedAt.UTC().Format(time.RFC3339)
	}
	for _, issue := range issues {
		frame := issuepolicy.AssessProblemFrame(issuepolicy.IssueDraft{Number: issue.Number, Title: issue.Title, Body: issue.Body})
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
	sort.Strings(r.Errors)
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
	return fmt.Sprintf("CENTRALITY COVERAGE %.1f%% (%d/%d)\nCore %d | Enabling %d | Stewardship %d | Peripheral %d | Unknown %d\nComplete P1-P4 frame %d/%d\nScope: %s\nProvenance: %s\nCollection errors: %d\n", r.CoveragePct, r.Classified, r.Total, r.Counts.Core, r.Counts.Enabling, r.Counts.Stewardship, r.Counts.Peripheral, r.Counts.Unknown, r.Counts.ProblemFrame, r.Total, r.Scope, r.Provenance, len(r.Errors))
}
