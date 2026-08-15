// Package stackpreflight binds assembly, workload fitness, and exact support evidence before launch.
package stackpreflight

import (
	"fmt"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stackresolve"
	"github.com/anthony-chaudhary/fak/internal/supportgraph"
	"github.com/anthony-chaudhary/fak/internal/workloadfit"
)

const Schema = "fak-stack-preflight/1"

type Input struct {
	Stack          stackresolve.Receipt
	Fitness        workloadfit.Assessment
	Graph          supportgraph.Graph
	Tuple          supportgraph.Tuple
	AsOf           time.Time
	CapacityTarget string
}

type Alternative struct {
	Rank   int    `json:"rank"`
	Action string `json:"action"`
	Impact string `json:"impact,omitempty"`
}

type Result struct {
	Schema         string              `json:"schema"`
	Status         string              `json:"status"`
	Required       []string            `json:"required_baseline,omitempty"`
	Recommended    []string            `json:"recommended_baseline,omitempty"`
	Warnings       []string            `json:"warnings,omitempty"`
	Blockers       []string            `json:"blockers,omitempty"`
	Alternatives   []Alternative       `json:"alternatives,omitempty"`
	Support        supportgraph.Result `json:"support"`
	CapacityTarget string              `json:"capacity_target,omitempty"`
}

func Run(in Input) Result {
	result := Result{Schema: Schema, Status: "allow", CapacityTarget: in.CapacityTarget}
	if in.Stack.Status != "allow" {
		result.Status = "refuse"
		result.Blockers = append(result.Blockers, "assembly: "+stackConflict(in.Stack))
		return finish(result)
	}
	if in.Fitness.Status != "fit" {
		result.Status = "refuse"
		for _, finding := range in.Fitness.Findings {
			if finding.State != workloadfit.Met && finding.Class != workloadfit.Preference {
				result.Blockers = append(result.Blockers, fmt.Sprintf("fitness:%s:%s", finding.Requirement, finding.State))
			}
		}
		return finish(result)
	}

	result.Support = supportgraph.Query(in.Graph, in.Tuple, in.AsOf)
	result.Required = append(result.Required, result.Support.Required...)
	result.Recommended = append(result.Recommended, result.Support.Recommended...)
	switch result.Support.State {
	case supportgraph.Supported:
		for _, baseline := range result.Recommended {
			result.Warnings = append(result.Warnings, "recommended, not required: "+baseline)
		}
	case supportgraph.Unknown:
		result.Status = "refuse"
		result.Blockers = append(result.Blockers, "support: exact tuple is not evaluated")
		result.Alternatives = append(result.Alternatives, Alternative{Rank: 1, Action: "collect an exact-tuple witness", Impact: "support remains unknown until witnessed"})
	case supportgraph.Stale:
		result.Status = "refuse"
		result.Blockers = append(result.Blockers, "support: evidence is stale")
		result.Alternatives = append(result.Alternatives, Alternative{Rank: 1, Action: "refresh the expired witness", Impact: "revalidates the exact tuple"})
	case supportgraph.Unsupported:
		result.Status = "refuse"
		result.Blockers = append(result.Blockers, "support: exact tuple is unsupported")
		if result.Support.Fallback != "" {
			result.Alternatives = append(result.Alternatives, Alternative{Rank: 1, Action: "select " + result.Support.Fallback, Impact: result.Support.Penalty})
		}
	case supportgraph.Conflict:
		result.Status = "refuse"
		result.Blockers = append(result.Blockers, "support: equal-tier evidence conflicts")
	default:
		result.Status = "refuse"
		result.Blockers = append(result.Blockers, "support: invalid state")
	}
	return finish(result)
}

func stackConflict(receipt stackresolve.Receipt) string {
	if receipt.Conflict == nil {
		return "unresolved"
	}
	return receipt.Conflict.Code + ":" + receipt.Conflict.Wanted
}

func finish(result Result) Result {
	sort.Strings(result.Required)
	sort.Strings(result.Recommended)
	sort.Strings(result.Warnings)
	sort.Strings(result.Blockers)
	sort.SliceStable(result.Alternatives, func(i, j int) bool { return result.Alternatives[i].Rank < result.Alternatives[j].Rank })
	return result
}
