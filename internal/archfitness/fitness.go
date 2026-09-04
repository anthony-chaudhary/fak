// Package archfitness scores composition architecture debt, distinct from file-size quality checks.
package archfitness

import (
	"bytes"
	"encoding/json"
	"sort"
)

// Invariant: architecture fitness score is deterministic and monotone as debt findings decrease.
// Guard: fail-closed on unknown severities by treating unrecognised or unset severities as hard debt.

// Finding represents an individual architecture defect, debt item, or boundary violation.
type Finding struct {
	Dimension, Severity, File, Symbol, Reason, Issue, Owner, Expiry string `json:",omitempty"`
}

// Input contains collections of findings categorized across the architecture dimensions.
type Input struct {
	ForbiddenImports           []Finding
	FrozenSeamChurn            []Finding
	FamilySwitches             []Finding
	CrossPlaneAmplification    []Finding
	BespokeBranches            []Finding
	AmbiguousResources         []Finding
	MissingCompositionFixtures []Finding
	MissingCausalProjection    []Finding
	UnversionedSchemas         []Finding
	DynamicHotPath             []Finding
	PrivacyCardinality         []Finding
	StaleExceptions            []Finding
}

// Report captures the computed fitness score, debt tallies, and ordered findings work list.
type Report struct {
	Schema     string         `json:"schema"`
	Score      int            `json:"score"`
	HardDebt   int            `json:"hard_debt"`
	Dimensions map[string]int `json:"dimensions"`
	Work       []Finding      `json:"work"`
}

// Analyze aggregates findings across declared dimensions, computes total hard debt and the 0-100 score,
// and sorts the work list deterministically by dimension, file, and symbol.
func Analyze(in Input) Report {
	sets := map[string][]Finding{
		"dependency_dag":       in.ForbiddenImports,
		"frozen_seams":         in.FrozenSeamChurn,
		"family_switches":      in.FamilySwitches,
		"change_amplification": in.CrossPlaneAmplification,
		"descriptor_coverage":  in.BespokeBranches,
		"resource_ownership":   in.AmbiguousResources,
		"composition_fixtures": in.MissingCompositionFixtures,
		"causal_evidence":      in.MissingCausalProjection,
		"schema_migration":     in.UnversionedSchemas,
		"hot_path_scaling":     in.DynamicHotPath,
		"privacy_cardinality":  in.PrivacyCardinality,
		"stale_exceptions":     in.StaleExceptions,
	}
	r := Report{Schema: "fak.architecture-fitness/1", Score: 100, Dimensions: map[string]int{}}
	for dim, fs := range sets {
		r.Dimensions[dim] = len(fs)
		for _, f := range fs {
			f.Dimension = dim
			if f.Severity != "soft" {
				f.Severity = "hard"
			}
			r.Work = append(r.Work, f)
			if f.Severity == "hard" {
				r.HardDebt++
			}
		}
	}
	sort.Slice(r.Work, func(i, j int) bool {
		a, b := r.Work[i], r.Work[j]
		if a.Dimension != b.Dimension {
			return a.Dimension < b.Dimension
		}
		if a.File != b.File {
			return a.File < b.File
		}
		return a.Symbol < b.Symbol
	})
	r.Score -= r.HardDebt * 5
	if r.Score < 0 {
		r.Score = 0
	}
	return r
}

// JSON serializes the architecture fitness report into indented JSON formatting.
func JSON(r Report) ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

// WorkList formats the findings in the report into a human-readable list of actionable items.
func WorkList(r Report) string {
	var b bytes.Buffer
	for _, f := range r.Work {
		b.WriteString(f.Severity + " " + f.Dimension + " " + f.File + ":" + f.Symbol + " — " + f.Reason + "\n")
	}
	return b.String()
}

// Ratchet returns true if current report has no more hard debt than baseline report.
func Ratchet(baseline, current Report) bool { return current.HardDebt <= baseline.HardDebt }
