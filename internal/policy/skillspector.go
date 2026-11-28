// Conceptual INSPIRE-ONLY seam derived from NVIDIA/SkillSpector
// (e48d771a26f7398b36256ceb8d8f94adce3d5c44, Apache-2.0).
// No Python scanner runtime is ported and no completeness claim is made over
// upstream detection rules.
package policy

import (
	"fmt"
	"strings"
)

// SkillFindingCategory classifies static analysis findings on an agent skill.
type SkillFindingCategory string

const (
	SkillFindingExec         SkillFindingCategory = "arbitrary_exec"
	SkillFindingExfil        SkillFindingCategory = "data_exfiltration"
	SkillFindingObfuscation  SkillFindingCategory = "obfuscated_payload"
	SkillFindingPrivilege    SkillFindingCategory = "privilege_escalation"
	SkillFindingPromptInject SkillFindingCategory = "prompt_injection"
)

// SkillSeverity orders finding severity from info (0) to critical (4).
type SkillSeverity int

const (
	SeverityInfo SkillSeverity = iota
	SeverityLow
	SeverityMedium
	SeverityHigh
	SeverityCritical
)

// SkillFinding records one static-analysis finding on a skill artifact.
type SkillFinding struct {
	RuleID   string               `json:"rule_id"`
	Category SkillFindingCategory `json:"category"`
	Severity SkillSeverity        `json:"severity"`
	Summary  string               `json:"summary"`
}

// SkillScanReport holds the pre-admission static scan verdict for a skill.
type SkillScanReport struct {
	SkillName string         `json:"skill_name"`
	Scanned   bool           `json:"scanned"`
	Findings  []SkillFinding `json:"findings,omitempty"`
}

// SkillScanGate configures the fail-closed admission threshold for scanned skills.
type SkillScanGate struct {
	MaxAllowedSeverity SkillSeverity          `json:"max_allowed_severity"`
	BlockedCategories  []SkillFindingCategory `json:"blocked_categories,omitempty"`
}

// DefaultSkillScanGate denies high/critical findings and exfiltration/obfuscation/prompt-injection.
func DefaultSkillScanGate() SkillScanGate {
	return SkillScanGate{
		MaxAllowedSeverity: SeverityMedium,
		BlockedCategories: []SkillFindingCategory{
			SkillFindingExfil,
			SkillFindingObfuscation,
			SkillFindingPromptInject,
		},
	}
}

// SkillAdmissionDecision is the combined static-scan + SkillSpan compilation verdict.
type SkillAdmissionDecision struct {
	Admitted bool      `json:"admitted"`
	Reason   string    `json:"reason,omitempty"`
	Span     SkillSpan `json:"span,omitempty"`
}

// EvaluateSkillScanGate enforces that a skill carries a passing scan report
// before compiling its frontmatter into a default-deny SkillSpan.
func EvaluateSkillScanGate(fm SkillFrontmatter, report SkillScanReport, gate SkillScanGate) SkillAdmissionDecision {
	if !report.Scanned {
		return SkillAdmissionDecision{Admitted: false, Reason: "skill has not been scanned"}
	}
	if strings.TrimSpace(fm.Name) == "" || report.SkillName != fm.Name {
		return SkillAdmissionDecision{Admitted: false, Reason: "scan report skill_name mismatch"}
	}
	blocked := make(map[SkillFindingCategory]struct{}, len(gate.BlockedCategories))
	for _, c := range gate.BlockedCategories {
		blocked[c] = struct{}{}
	}
	for _, f := range report.Findings {
		if _, isBlocked := blocked[f.Category]; isBlocked {
			return SkillAdmissionDecision{
				Admitted: false,
				Reason:   fmt.Sprintf("blocked finding category %s (rule %s)", f.Category, f.RuleID),
			}
		}
		if f.Severity > gate.MaxAllowedSeverity {
			return SkillAdmissionDecision{
				Admitted: false,
				Reason:   fmt.Sprintf("finding %s severity %d exceeds max %d", f.RuleID, f.Severity, gate.MaxAllowedSeverity),
			}
		}
	}
	span, err := CompileSkillSpan(fm)
	if err != nil {
		return SkillAdmissionDecision{Admitted: false, Reason: err.Error()}
	}
	return SkillAdmissionDecision{Admitted: true, Span: span}
}
