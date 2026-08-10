package archreport

import (
	"encoding/json"
	"sort"
)

const DiffSchema = "fak-architecture-diff/1"

type TierChange struct {
	Leaf       string `json:"leaf"`
	Before     int    `json:"before"`
	BeforeName string `json:"before_name"`
	After      int    `json:"after"`
	AfterName  string `json:"after_name"`
}

type EdgeChange struct {
	From string `json:"from"`
	To   string `json:"to"`
}

type ReportDiff struct {
	Schema               string       `json:"schema"`
	Verdict              string       `json:"verdict"`
	AddedLeaves          []string     `json:"added_leaves,omitempty"`
	RemovedLeaves        []string     `json:"removed_leaves,omitempty"`
	TierChanges          []TierChange `json:"tier_changes,omitempty"`
	AddedEdges           []EdgeChange `json:"added_edges,omitempty"`
	RemovedEdges         []EdgeChange `json:"removed_edges,omitempty"`
	IntroducedViolations []string     `json:"introduced_violations,omitempty"`
	ResolvedViolations   []string     `json:"resolved_violations,omitempty"`
}

func Diff(before, after Report) ReportDiff {
	out := ReportDiff{Schema: DiffSchema, Verdict: "clean"}
	beforeLeaves, afterLeaves := leafIndex(before), leafIndex(after)
	for name, a := range afterLeaves {
		b, ok := beforeLeaves[name]
		if !ok {
			out.AddedLeaves = append(out.AddedLeaves, name)
			continue
		}
		if b.DeclaredTier != a.DeclaredTier {
			out.TierChanges = append(out.TierChanges, TierChange{Leaf: name, Before: b.DeclaredTier, BeforeName: b.DeclaredTierName, After: a.DeclaredTier, AfterName: a.DeclaredTierName})
		}
	}
	for name := range beforeLeaves {
		if _, ok := afterLeaves[name]; !ok {
			out.RemovedLeaves = append(out.RemovedLeaves, name)
		}
	}
	beforeEdges, afterEdges := edgeSet(before), edgeSet(after)
	for key, edge := range afterEdges {
		if _, ok := beforeEdges[key]; !ok {
			out.AddedEdges = append(out.AddedEdges, edge)
		}
	}
	for key, edge := range beforeEdges {
		if _, ok := afterEdges[key]; !ok {
			out.RemovedEdges = append(out.RemovedEdges, edge)
		}
	}
	beforeViolations, afterViolations := violationSet(before), violationSet(after)
	for edge := range afterViolations {
		if !beforeViolations[edge] {
			out.IntroducedViolations = append(out.IntroducedViolations, edge)
		}
	}
	for edge := range beforeViolations {
		if !afterViolations[edge] {
			out.ResolvedViolations = append(out.ResolvedViolations, edge)
		}
	}
	sort.Strings(out.AddedLeaves)
	sort.Strings(out.RemovedLeaves)
	sort.Slice(out.TierChanges, func(i, j int) bool { return out.TierChanges[i].Leaf < out.TierChanges[j].Leaf })
	sortEdges(out.AddedEdges)
	sortEdges(out.RemovedEdges)
	sort.Strings(out.IntroducedViolations)
	sort.Strings(out.ResolvedViolations)
	if len(out.IntroducedViolations) > 0 {
		out.Verdict = "regression"
	}
	return out
}

func (d ReportDiff) Changes() int {
	return len(d.AddedLeaves) + len(d.RemovedLeaves) + len(d.TierChanges) + len(d.AddedEdges) + len(d.RemovedEdges) + len(d.IntroducedViolations) + len(d.ResolvedViolations)
}
func (d ReportDiff) JSON() ([]byte, error) { return json.MarshalIndent(d, "", "  ") }
func leafIndex(r Report) map[string]Leaf {
	out := map[string]Leaf{}
	for _, l := range r.Leaves {
		out[l.Name] = l
	}
	return out
}
func edgeSet(r Report) map[string]EdgeChange {
	out := map[string]EdgeChange{}
	for _, l := range r.Leaves {
		for _, to := range l.Dependencies {
			e := EdgeChange{From: l.Name, To: to}
			out[l.Name+"\x00"+to] = e
		}
	}
	return out
}
func violationSet(r Report) map[string]bool {
	out := map[string]bool{}
	for _, l := range r.Leaves {
		for _, e := range l.Violations {
			out[e] = true
		}
	}
	return out
}
func sortEdges(edges []EdgeChange) {
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].From != edges[j].From {
			return edges[i].From < edges[j].From
		}
		return edges[i].To < edges[j].To
	})
}
