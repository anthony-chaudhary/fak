package policy

import "testing"

// fak-test:runtime fast est=5ms lane=default
func TestSkillSpectorGateAdmission(t *testing.T) {
	fm := SkillFrontmatter{
		Name:         "safe-reader",
		Description:  "Reads workspace files",
		AllowedTools: []string{"read_file"},
	}
	gate := DefaultSkillScanGate()

	// Unscanned skill is refused fail-closed.
	unscanned := EvaluateSkillScanGate(fm, SkillScanReport{SkillName: "safe-reader", Scanned: false}, gate)
	if unscanned.Admitted {
		t.Fatal("expected unscanned skill to be denied")
	}

	// Blocked category (exfiltration) even at low severity is refused.
	exfil := EvaluateSkillScanGate(fm, SkillScanReport{
		SkillName: "safe-reader",
		Scanned:   true,
		Findings:  []SkillFinding{{RuleID: "EXFIL-1", Category: SkillFindingExfil, Severity: SeverityLow}},
	}, gate)
	if exfil.Admitted {
		t.Fatal("expected exfiltration finding to be denied")
	}

	// High severity finding above MaxAllowedSeverity is refused.
	high := EvaluateSkillScanGate(fm, SkillScanReport{
		SkillName: "safe-reader",
		Scanned:   true,
		Findings:  []SkillFinding{{RuleID: "EXEC-9", Category: SkillFindingExec, Severity: SeverityHigh}},
	}, gate)
	if high.Admitted {
		t.Fatal("expected high severity finding to be denied")
	}

	// Clean scanned skill is admitted with compiled SkillSpan.
	clean := EvaluateSkillScanGate(fm, SkillScanReport{
		SkillName: "safe-reader",
		Scanned:   true,
		Findings:  []SkillFinding{{RuleID: "INFO-1", Category: SkillFindingExec, Severity: SeverityLow}},
	}, gate)
	if !clean.Admitted || clean.Span.Name != "safe-reader" {
		t.Fatalf("expected admission, got %+v", clean)
	}
}
