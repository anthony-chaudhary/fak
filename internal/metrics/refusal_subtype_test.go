package metrics

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestRefusalSubtypeClassification(t *testing.T) {
	tests := []struct {
		name     string
		reason   string
		meta     map[string]string
		expected string
	}{
		// Non-TRUST_VIOLATION reasons -> empty string (cardinality control)
		{"policy_block", "POLICY_BLOCK", map[string]string{"subsystem": "ctxmmu"}, ""},
		{"rate_limited", "RATE_LIMITED", map[string]string{"by": "ifc-sink"}, ""},
		{"malformed", "MALFORMED", map[string]string{"witness": "refuted"}, ""},
		{"allow", "ALLOW", map[string]string{"residency": "mismatch"}, ""},
		{"empty_reason", "", map[string]string{"quarantine_id": "qid-1"}, ""},
		{"none", "NONE", map[string]string{"admit": "quarantined"}, ""},

		// TRUST_VIOLATION normalization
		{"trust_violation_exact", "TRUST_VIOLATION", nil, RefusalSubtypeUnspecified},
		{"trust_violation_lower", "trust_violation", nil, RefusalSubtypeUnspecified},
		{"trust_violation_spaced", "  TRUST_VIOLATION  ", nil, RefusalSubtypeUnspecified},
		{"trustviolation_no_underscore", "TRUSTVIOLATION", nil, RefusalSubtypeUnspecified},
		{"trustviolation_mixed_case", "TrustViolation", nil, RefusalSubtypeUnspecified},

		// Unspecified fallback
		{"nil_meta", "TRUST_VIOLATION", nil, RefusalSubtypeUnspecified},
		{"empty_meta", "TRUST_VIOLATION", map[string]string{}, RefusalSubtypeUnspecified},
		{"unrelated_meta", "TRUST_VIOLATION", map[string]string{"foo": "bar", "trace_id": "123"}, RefusalSubtypeUnspecified},

		// injection_quarantine triggers
		{"injection_quarantine_id", "TRUST_VIOLATION", map[string]string{"quarantine_id": "q-100"}, RefusalSubtypeInjectionQuarantine},
		{"injection_subsystem_ctxmmu", "TRUST_VIOLATION", map[string]string{"subsystem": "ctxmmu"}, RefusalSubtypeInjectionQuarantine},
		{"injection_subsystem_wirescreen", "TRUST_VIOLATION", map[string]string{"subsystem": "wirescreen"}, RefusalSubtypeInjectionQuarantine},
		{"injection_by_ctxmmu", "TRUST_VIOLATION", map[string]string{"by": "ctxmmu"}, RefusalSubtypeInjectionQuarantine},
		{"injection_by_wirescreen", "TRUST_VIOLATION", map[string]string{"by": "wirescreen"}, RefusalSubtypeInjectionQuarantine},
		{"injection_detector_marker", "TRUST_VIOLATION", map[string]string{"detector": "injection_marker"}, RefusalSubtypeInjectionQuarantine},
		{"injection_admit_quarantined", "TRUST_VIOLATION", map[string]string{"admit": "quarantined"}, RefusalSubtypeInjectionQuarantine},
		{"injection_claim_contains", "TRUST_VIOLATION", map[string]string{"claim": "detected prompt_injection in body"}, RefusalSubtypeInjectionQuarantine},
		{"injection_quarantine_reason_contains", "TRUST_VIOLATION", map[string]string{"quarantine_reason": "injection detected"}, RefusalSubtypeInjectionQuarantine},

		// ifc_sink triggers
		{"ifc_subsystem_ifc_sink", "TRUST_VIOLATION", map[string]string{"subsystem": "ifc-sink"}, RefusalSubtypeIFCSink},
		{"ifc_by_ifc_sink", "TRUST_VIOLATION", map[string]string{"by": "ifc-sink"}, RefusalSubtypeIFCSink},
		{"ifc_sink_not_empty", "TRUST_VIOLATION", map[string]string{"ifc_sink": "network_egress"}, RefusalSubtypeIFCSink},
		{"ifc_deny_rule_taint", "TRUST_VIOLATION", map[string]string{"deny_rule": "ifc_taint_egress"}, RefusalSubtypeIFCSink},
		{"ifc_flow_not_empty", "TRUST_VIOLATION", map[string]string{"ifc_flow": "confidential->public"}, RefusalSubtypeIFCSink},

		// witness_refuted triggers
		{"witness_subsystem_witness", "TRUST_VIOLATION", map[string]string{"subsystem": "witness"}, RefusalSubtypeWitnessRefuted},
		{"witness_by_witness", "TRUST_VIOLATION", map[string]string{"by": "witness"}, RefusalSubtypeWitnessRefuted},
		{"witness_refuted_exact", "TRUST_VIOLATION", map[string]string{"witness": "refuted"}, RefusalSubtypeWitnessRefuted},
		{"witness_status_refuted", "TRUST_VIOLATION", map[string]string{"witness_status": "refuted"}, RefusalSubtypeWitnessRefuted},
		{"witness_contains_refuted", "TRUST_VIOLATION", map[string]string{"witness": "claim_refuted_by_oracle"}, RefusalSubtypeWitnessRefuted},
		{"claim_contains_refuted", "TRUST_VIOLATION", map[string]string{"claim": "diff refuted by git log"}, RefusalSubtypeWitnessRefuted},

		// residency_mismatch triggers
		{"residency_subsystem_residency", "TRUST_VIOLATION", map[string]string{"subsystem": "residency"}, RefusalSubtypeResidencyMismatch},
		{"residency_subsystem_engine_residency", "TRUST_VIOLATION", map[string]string{"subsystem": "engine-residency"}, RefusalSubtypeResidencyMismatch},
		{"residency_by_residency", "TRUST_VIOLATION", map[string]string{"by": "residency"}, RefusalSubtypeResidencyMismatch},
		{"residency_by_engine_residency", "TRUST_VIOLATION", map[string]string{"by": "engine-residency"}, RefusalSubtypeResidencyMismatch},
		{"residency_not_empty", "TRUST_VIOLATION", map[string]string{"residency": "mismatch"}, RefusalSubtypeResidencyMismatch},
		{"residency_mismatch_not_empty", "TRUST_VIOLATION", map[string]string{"residency_mismatch": "tier_1_required"}, RefusalSubtypeResidencyMismatch},

		// Precedence check (injection > ifc_sink > witness > residency)
		{"precedence_injection_over_ifc", "TRUST_VIOLATION", map[string]string{"quarantine_id": "qid-1", "ifc_sink": "true"}, RefusalSubtypeInjectionQuarantine},
		{"precedence_ifc_over_witness", "TRUST_VIOLATION", map[string]string{"ifc_sink": "egress", "witness": "refuted"}, RefusalSubtypeIFCSink},
		{"precedence_witness_over_residency", "TRUST_VIOLATION", map[string]string{"witness": "refuted", "residency": "mismatch"}, RefusalSubtypeWitnessRefuted},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyRefusalSubtype(tc.reason, tc.meta)
			if got != tc.expected {
				t.Errorf("ClassifyRefusalSubtype(%q, %v) = %q, want %q", tc.reason, tc.meta, got, tc.expected)
			}
		})
	}
}

func TestRefusalSubtypeLabel(t *testing.T) {
	lbl := RefusalSubtypeLabel("TRUST_VIOLATION", map[string]string{"subsystem": "ctxmmu"})
	if lbl.Name != "refusal_subtype" {
		t.Errorf("lbl.Name = %q, want refusal_subtype", lbl.Name)
	}
	if lbl.Value != RefusalSubtypeInjectionQuarantine {
		t.Errorf("lbl.Value = %q, want %q", lbl.Value, RefusalSubtypeInjectionQuarantine)
	}

	lblOther := RefusalSubtypeLabel("POLICY_BLOCK", map[string]string{"subsystem": "ctxmmu"})
	if lblOther.Value != "" {
		t.Errorf("lblOther.Value = %q, want empty string", lblOther.Value)
	}
}

func TestVerdictRefusalSubtype(t *testing.T) {
	// abi.Verdict with ReasonTrustViolation and meta
	v1 := abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		Meta:   map[string]string{"deny_rule": "ifc_taint_egress"},
	}
	if got := VerdictRefusalSubtype(v1); got != RefusalSubtypeIFCSink {
		t.Errorf("VerdictRefusalSubtype(v1) = %q, want %q", got, RefusalSubtypeIFCSink)
	}

	// abi.Verdict with ReasonTrustViolation and By="ctxmmu"
	v2 := abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "ctxmmu",
	}
	if got := VerdictRefusalSubtype(v2); got != RefusalSubtypeInjectionQuarantine {
		t.Errorf("VerdictRefusalSubtype(v2) = %q, want %q", got, RefusalSubtypeInjectionQuarantine)
	}

	// abi.Verdict with ReasonTrustViolation and WitnessPayload containing "refuted"
	v3 := abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  abi.ReasonTrustViolation,
		Payload: abi.WitnessPayload{Claim: "claim refuted"},
	}
	if got := VerdictRefusalSubtype(v3); got != RefusalSubtypeWitnessRefuted {
		t.Errorf("VerdictRefusalSubtype(v3) = %q, want %q", got, RefusalSubtypeWitnessRefuted)
	}

	// abi.Verdict with ReasonTrustViolation and By="residency"
	v4 := abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "residency",
	}
	if got := VerdictRefusalSubtype(v4); got != RefusalSubtypeResidencyMismatch {
		t.Errorf("VerdictRefusalSubtype(v4) = %q, want %q", got, RefusalSubtypeResidencyMismatch)
	}

	// abi.Verdict with ReasonTrustViolation and no triggers -> unspecified
	v5 := abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
	}
	if got := VerdictRefusalSubtype(v5); got != RefusalSubtypeUnspecified {
		t.Errorf("VerdictRefusalSubtype(v5) = %q, want %q", got, RefusalSubtypeUnspecified)
	}

	// Non-TRUST_VIOLATION verdict -> ""
	v6 := abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonPolicyBlock,
		By:     "ctxmmu",
	}
	if got := VerdictRefusalSubtype(v6); got != "" {
		t.Errorf("VerdictRefusalSubtype(v6) = %q, want empty string", got)
	}
}
