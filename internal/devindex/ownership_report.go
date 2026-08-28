package devindex

import (
	"encoding/json"
	"fmt"
)

// OwnershipReport is the complete machine-readable #6020 boundary witness.
type OwnershipReport struct {
	Schema     string             `json:"schema"`
	Commands   []CommandOwnership `json:"commands"`
	Packages   []PackageOwnership `json:"packages"`
	Graph      GraphReport        `json:"graph"`
	Extraction ExtractionReport   `json:"remaining_tier_dev_extraction"`
}

// BuildOwnershipReport loads the authoritative command catalog and Go import
// graph from root. It reports current leaks rather than treating them as an
// error; the migration ratchet decides when the required leak count becomes 0.
func BuildOwnershipReport(root, pattern, importRoot string) (OwnershipReport, error) {
	cat, err := Load(root)
	if err != nil {
		return OwnershipReport{}, err
	}
	verbs := OwnershipVerbs(cat.Verbs())
	commands := CommandOwnerships(verbs)
	if problems := ValidateCommandOwnership(verbs, commands); len(problems) != 0 {
		return OwnershipReport{}, fmt.Errorf("invalid command ownership: %v", problems)
	}
	nodes, err := LoadImportGraph(root, pattern)
	if err != nil {
		return OwnershipReport{}, err
	}
	extraction, err := BuildRemainingExtractionReport(root, nodes)
	if err != nil {
		return OwnershipReport{}, fmt.Errorf("remaining TierDev extraction plan: %w", err)
	}
	return OwnershipReport{
		Schema:     "fak-command-ownership/2",
		Commands:   commands,
		Packages:   append([]PackageOwnership(nil), DevOnlyPackages...),
		Graph:      BuildGraphReport(importRoot, nodes, DevOnlyPackages),
		Extraction: extraction,
	}, nil
}

// MarshalOwnershipReport emits a stable indented JSON artifact.
func MarshalOwnershipReport(report OwnershipReport) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}
