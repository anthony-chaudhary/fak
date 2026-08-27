// Package archfitness scores composition architecture debt, distinct from file-size quality checks.
package archfitness

import (
	"bytes"
	"encoding/json"
	"sort"
)

type Finding struct {
	Dimension, Severity, File, Symbol, Reason, Issue, Owner, Expiry string `json:",omitempty"`
}
type Input struct{ ForbiddenImports, FrozenSeamChurn, FamilySwitches, CrossPlaneAmplification, BespokeBranches, AmbiguousResources, MissingCompositionFixtures, MissingCausalProjection, UnversionedSchemas, DynamicHotPath, PrivacyCardinality, StaleExceptions []Finding }
type Report struct {
	Schema     string         `json:"schema"`
	Score      int            `json:"score"`
	HardDebt   int            `json:"hard_debt"`
	Dimensions map[string]int `json:"dimensions"`
	Work       []Finding      `json:"work"`
}

func Analyze(in Input) Report {
	sets := map[string][]Finding{"dependency_dag": in.ForbiddenImports, "frozen_seams": in.FrozenSeamChurn, "family_switches": in.FamilySwitches, "change_amplification": in.CrossPlaneAmplification, "descriptor_coverage": in.BespokeBranches, "resource_ownership": in.AmbiguousResources, "composition_fixtures": in.MissingCompositionFixtures, "causal_evidence": in.MissingCausalProjection, "schema_migration": in.UnversionedSchemas, "hot_path_scaling": in.DynamicHotPath, "privacy_cardinality": in.PrivacyCardinality, "stale_exceptions": in.StaleExceptions}
	r := Report{Schema: "fak.architecture-fitness/1", Score: 100, Dimensions: map[string]int{}}
	for dim, fs := range sets {
		r.Dimensions[dim] = len(fs)
		for _, f := range fs {
			f.Dimension = dim
			if f.Severity == "" {
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
func JSON(r Report) ([]byte, error) { return json.MarshalIndent(r, "", "  ") }
func WorkList(r Report) string {
	var b bytes.Buffer
	for _, f := range r.Work {
		b.WriteString(f.Severity + " " + f.Dimension + " " + f.File + ":" + f.Symbol + " — " + f.Reason + "\n")
	}
	return b.String()
}
func Ratchet(baseline, current Report) bool { return current.HardDebt <= baseline.HardDebt }
