package issuepolicy

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

const (
	ProjectWorkValid      = "valid"
	ProjectWorkUndeclared = "undeclared"
	ProjectWorkInvalid    = "invalid"
)

type ProjectWorkReadout struct {
	Status             string   `json:"status"`
	EstimatePoints     float64  `json:"estimate_points,omitempty"`
	EstimateClass      string   `json:"estimate_class,omitempty"`
	Parent             string   `json:"parent,omitempty"`
	ParentBaseline     float64  `json:"parent_baseline_points,omitempty"`
	Contribution       float64  `json:"contribution_points,omitempty"`
	ContributionShare  float64  `json:"contribution_share,omitempty"`
	CompletionStandard string   `json:"completion_standard,omitempty"`
	ProductionCredit   bool     `json:"production_credit"`
	Invalid            []string `json:"invalid,omitempty"`
	Repair             []string `json:"repair,omitempty"`
}

var estimateRE = regexp.MustCompile(`(?i)(?:estimate\s*:\s*)?([0-9]+(?:\.[0-9]+)?)\s*points?(?:\s*\(([^)]*)\))?`)
var contributionRE = regexp.MustCompile(`(?i)(?:contribution\s*:\s*)?([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*points?`)
var parentIssueRE = regexp.MustCompile(`#([0-9]+)`)

func projectWork(c Candidate) ProjectWorkReadout {
	out := ProjectWorkReadout{Status: ProjectWorkUndeclared, CompletionStandard: normalizeCompletionStandard(c.CompletionStandard)}
	if strings.TrimSpace(c.WorkEstimate) == "" && strings.TrimSpace(c.ScopeContribution) == "" && out.CompletionStandard == "" {
		return out
	}
	if m := estimateRE.FindStringSubmatch(c.WorkEstimate); m != nil {
		out.EstimatePoints, _ = strconv.ParseFloat(m[1], 64)
		out.EstimateClass = strings.TrimSpace(m[2])
	} else {
		out.Invalid = append(out.Invalid, "work estimate must contain 'N points' (for example: Estimate: 5 points (medium))")
		out.Repair = append(out.Repair, "add ## Work estimate with a positive point estimate and uncertainty")
	}
	if out.EstimatePoints <= 0 {
		out.Invalid = append(out.Invalid, "work estimate must be greater than zero")
	}
	if m := contributionRE.FindStringSubmatch(c.ScopeContribution); m != nil {
		out.Contribution, _ = strconv.ParseFloat(m[1], 64)
		out.ParentBaseline, _ = strconv.ParseFloat(m[2], 64)
		if out.ParentBaseline > 0 {
			out.ContributionShare = out.Contribution / out.ParentBaseline
		}
	} else {
		out.Invalid = append(out.Invalid, "scope contribution must contain 'N/M points'")
		out.Repair = append(out.Repair, "add ## Overall completion contribution with contribution/parent-baseline points")
	}
	if m := parentIssueRE.FindStringSubmatch(c.ParentRef); m != nil {
		out.Parent = "#" + m[1]
	} else {
		out.Invalid = append(out.Invalid, "parent context must name a #N parent for denominator binding")
	}
	if out.Contribution <= 0 || out.ParentBaseline <= 0 || out.Contribution > out.ParentBaseline {
		out.Invalid = append(out.Invalid, "contribution and parent baseline must be positive and contribution cannot exceed baseline")
	}
	if out.EstimatePoints > 0 && out.Contribution > 0 && out.EstimatePoints != out.Contribution {
		out.Invalid = append(out.Invalid, fmt.Sprintf("estimate %.g points does not match contribution %.g points", out.EstimatePoints, out.Contribution))
	}
	if out.CompletionStandard == "" {
		out.Invalid = append(out.Invalid, "completion standard is missing")
		out.Repair = append(out.Repair, "add ## Completion standard; unqualified authoring defaults to production")
	}
	switch out.CompletionStandard {
	case "production", "research", "experiment", "prototype", "demo", "development", "dev", "integrated", "staging":
	default:
		if out.CompletionStandard != "" {
			out.Invalid = append(out.Invalid, "completion standard must be production, integrated, staging, development, demo, prototype, experiment, or research")
		}
	}
	out.ProductionCredit = out.CompletionStandard == "production" && len(out.Invalid) == 0
	if len(out.Invalid) > 0 {
		out.Status = ProjectWorkInvalid
	} else {
		out.Status = ProjectWorkValid
	}
	return out
}
