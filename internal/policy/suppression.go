package policy

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// SuppressionRefusal is a closed vocabulary for suppressions that fail closed.
type SuppressionRefusal string

const (
	SuppressionUnknownRule   SuppressionRefusal = "UNKNOWN_RULE"
	SuppressionMissingReason SuppressionRefusal = "MISSING_REASON"
	SuppressionMalformed     SuppressionRefusal = "MALFORMED_SUPPRESSION"
)

// Finding identifies one policy finding at an exact, caller-defined scope.
type Finding struct {
	Rule  string `json:"rule"`
	Scope string `json:"scope"`
}

// Suppression is an auditable policy exception. Rule, Scope, and Reason are required.
type Suppression struct {
	Rule   string `json:"rule"`
	Scope  string `json:"scope"`
	Reason string `json:"reason"`
}

// SuppressionRecord preserves every requested exception, including invalid and stale ones.
type SuppressionRecord struct {
	Suppression
	Status  string             `json:"status"` // used, unused, or refused
	Used    bool               `json:"used"`
	Refusal SuppressionRefusal `json:"refusal,omitempty"`
}

// SuppressionResult is the stable machine-readable suppression receipt.
type SuppressionResult struct {
	Schema       string              `json:"schema"`
	Findings     []Finding           `json:"findings"`
	Remaining    []Finding           `json:"remaining"`
	Suppressions []SuppressionRecord `json:"suppressions"`
}

// ApplySuppressions removes only findings with an exact rule-and-scope match. Invalid
// exceptions remain observable and never suppress an underlying finding.
func ApplySuppressions(inventory []string, findings []Finding, requested []Suppression) SuppressionResult {
	known := make(map[string]struct{}, len(inventory))
	for _, rule := range inventory {
		if rule != "" {
			known[rule] = struct{}{}
		}
	}
	result := SuppressionResult{Schema: "fak-policy-suppressions/1", Findings: append([]Finding(nil), findings...)}
	suppressed := make([]bool, len(findings))
	for _, s := range requested {
		rec := SuppressionRecord{Suppression: s, Status: "unused"}
		switch {
		case strings.TrimSpace(s.Rule) == "" || strings.TrimSpace(s.Scope) == "":
			rec.Status, rec.Refusal = "refused", SuppressionMalformed
		case strings.TrimSpace(s.Reason) == "":
			rec.Status, rec.Refusal = "refused", SuppressionMissingReason
		default:
			if _, ok := known[s.Rule]; !ok {
				rec.Status, rec.Refusal = "refused", SuppressionUnknownRule
			} else {
				for i, f := range findings {
					if !suppressed[i] && f.Rule == s.Rule && f.Scope == s.Scope {
						suppressed[i], rec.Used, rec.Status = true, true, "used"
					}
				}
			}
		}
		result.Suppressions = append(result.Suppressions, rec)
	}
	for i, f := range findings {
		if !suppressed[i] {
			result.Remaining = append(result.Remaining, f)
		}
	}
	return result
}

// JSON returns the canonical machine-readable receipt.
func (r SuppressionResult) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

// Text renders honored, stale, and refused exceptions; none disappear from human output.
func (r SuppressionResult) Text() string {
	var lines []string
	for _, s := range r.Suppressions {
		line := fmt.Sprintf("%s rule=%s scope=%s reason=%q", strings.ToUpper(s.Status), s.Rule, s.Scope, s.Reason)
		if s.Refusal != "" {
			line += " refusal=" + string(s.Refusal)
		}
		lines = append(lines, line)
	}
	sort.Strings(lines)
	return strings.Join(lines, "\n")
}
