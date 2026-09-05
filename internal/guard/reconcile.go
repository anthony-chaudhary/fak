package guard

import (
	"fmt"
	"sort"
	"strings"
)

// ToolCatalog represents the advertised tools and metadata reported by a harness.
type ToolCatalog struct {
	Version string   `json:"version,omitempty"`
	Harness string   `json:"harness,omitempty"`
	Tools   []string `json:"tools,omitempty"`
}

// CapabilityProfile defines the standard capability profile and allowed tools.
type CapabilityProfile struct {
	Name         string            `json:"name,omitempty"`
	Version      string            `json:"version,omitempty"`
	AllowedTools []string          `json:"allowed_tools,omitempty"`
	KnownAliases map[string]string `json:"known_aliases,omitempty"`
}

// Allows reports whether the given tool is permitted by the profile,
// either directly in AllowedTools or via a known alias to an allowed tool.
func (p CapabilityProfile) Allows(tool string) bool {
	for _, allowed := range p.AllowedTools {
		if allowed == tool {
			return true
		}
	}
	if target, ok := p.KnownAliases[tool]; ok {
		for _, allowed := range p.AllowedTools {
			if allowed == target {
				return true
			}
		}
	}
	return false
}

// ReconciliationResult holds the classification of advertised tools against a capability profile.
type ReconciliationResult struct {
	Recognized     []string          `json:"recognized,omitempty"`
	DriftedAliases map[string]string `json:"drifted_aliases,omitempty"`
	Unknown        []string          `json:"unknown,omitempty"`
	Remedies       []string          `json:"remedies,omitempty"`
	Warning        string            `json:"warning,omitempty"`
}

// ReconcileCatalog compares advertised harness tools with a selected capability profile.
// It classifies standard capabilities, aliases, and unknown names.
// For each unknown tool, it generates an exact copyable option: "--allow-tool <tool>".
//
// Invariant: No model or agent-controlled prompt can turn a remedy into authority;
// remedies are strictly operator-facing recommendations. Neither ReconcileCatalog nor
// ReconciliationResult grants authority or modifies the active CapabilityProfile.
func ReconcileCatalog(catalog ToolCatalog, profile CapabilityProfile) ReconciliationResult {
	res := ReconciliationResult{
		Recognized:     make([]string, 0),
		DriftedAliases: make(map[string]string),
		Unknown:        make([]string, 0),
		Remedies:       make([]string, 0),
	}

	allowed := make(map[string]struct{}, len(profile.AllowedTools))
	for _, t := range profile.AllowedTools {
		allowed[t] = struct{}{}
	}

	seen := make(map[string]struct{}, len(catalog.Tools))
	for _, tool := range catalog.Tools {
		if _, exists := seen[tool]; exists {
			continue
		}
		seen[tool] = struct{}{}

		if _, ok := allowed[tool]; ok {
			res.Recognized = append(res.Recognized, tool)
			continue
		}

		if target, isAlias := profile.KnownAliases[tool]; isAlias {
			res.Recognized = append(res.Recognized, tool)
			res.DriftedAliases[tool] = target
			continue
		}

		res.Unknown = append(res.Unknown, tool)
		res.Remedies = append(res.Remedies, "--allow-tool "+tool)
	}

	var warnings []string
	if profile.Version != "" && catalog.Version != profile.Version {
		warnings = append(warnings, fmt.Sprintf("catalog version %q drifts from profile version %q", catalog.Version, profile.Version))
	}

	if len(res.DriftedAliases) > 0 {
		var aliases []string
		for k, v := range res.DriftedAliases {
			aliases = append(aliases, fmt.Sprintf("%s -> %s", k, v))
		}
		sort.Strings(aliases)
		warnings = append(warnings, fmt.Sprintf("advertised tool names drift from standard profile: %s", strings.Join(aliases, ", ")))
	}

	if len(res.Unknown) > 0 {
		warnings = append(warnings, fmt.Sprintf("unrecognized advertised tools: %s", strings.Join(res.Unknown, ", ")))
	}

	if len(warnings) > 0 {
		res.Warning = strings.Join(warnings, "; ")
	}

	return res
}
