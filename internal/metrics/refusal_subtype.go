package metrics

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Granular refusal subtypes for TRUST_VIOLATION metrics.
const (
	RefusalSubtypeInjectionQuarantine = "injection_quarantine"
	RefusalSubtypeIFCSink             = "ifc_sink"
	RefusalSubtypeWitnessRefuted      = "witness_refuted"
	RefusalSubtypeResidencyMismatch   = "residency_mismatch"
	RefusalSubtypeUnspecified         = "unspecified"
)

// ClassifyRefusalSubtype classifies a refusal reason and its metadata into a granular subtype.
// If reason is not TRUST_VIOLATION (or TRUSTVIOLATION), an empty string is returned for cardinality control.
func ClassifyRefusalSubtype(reason string, meta map[string]string) string {
	norm := strings.ToUpper(strings.TrimSpace(reason))
	if norm != "TRUST_VIOLATION" && norm != "TRUSTVIOLATION" {
		return ""
	}
	if meta == nil {
		return RefusalSubtypeUnspecified
	}
	if meta["quarantine_id"] != "" ||
		meta["subsystem"] == "ctxmmu" ||
		meta["subsystem"] == "wirescreen" ||
		meta["by"] == "ctxmmu" ||
		meta["by"] == "wirescreen" ||
		meta["detector"] == "injection_marker" ||
		meta["admit"] == "quarantined" ||
		strings.Contains(meta["claim"], "injection") ||
		strings.Contains(meta["quarantine_reason"], "injection") {
		return RefusalSubtypeInjectionQuarantine
	}
	if meta["subsystem"] == "ifc-sink" ||
		meta["by"] == "ifc-sink" ||
		meta["ifc_sink"] != "" ||
		meta["deny_rule"] == "ifc_taint_egress" ||
		meta["ifc_flow"] != "" {
		return RefusalSubtypeIFCSink
	}
	if meta["subsystem"] == "witness" ||
		meta["by"] == "witness" ||
		meta["witness"] == "refuted" ||
		meta["witness_status"] == "refuted" ||
		strings.Contains(meta["witness"], "refuted") ||
		strings.Contains(meta["claim"], "refuted") {
		return RefusalSubtypeWitnessRefuted
	}
	if meta["subsystem"] == "residency" ||
		meta["subsystem"] == "engine-residency" ||
		meta["by"] == "residency" ||
		meta["by"] == "engine-residency" ||
		meta["residency"] != "" ||
		meta["residency_mismatch"] != "" {
		return RefusalSubtypeResidencyMismatch
	}
	return RefusalSubtypeUnspecified
}

// RefusalSubtypeLabel returns an OpenMetricLabel for the refusal_subtype dimension.
func RefusalSubtypeLabel(reason string, meta map[string]string) OpenMetricLabel {
	return OpenMetricLabel{Name: "refusal_subtype", Value: ClassifyRefusalSubtype(reason, meta)}
}

// VerdictRefusalSubtype classifies the refusal subtype from an abi.Verdict.
func VerdictRefusalSubtype(v abi.Verdict) string {
	reason := ""
	if v.Reason != abi.ReasonNone {
		reason = abi.ReasonName(v.Reason)
	}
	if reason == "" && v.Meta != nil {
		reason = v.Meta["reason"]
	}
	meta := v.Meta
	if v.By != "" {
		if meta == nil {
			meta = map[string]string{"by": v.By}
		} else if _, ok := meta["by"]; !ok {
			combined := make(map[string]string, len(meta)+1)
			for k, val := range meta {
				combined[k] = val
			}
			combined["by"] = v.By
			meta = combined
		}
	}
	if wp, ok := v.Payload.(abi.WitnessPayload); ok && wp.Claim != "" {
		if meta == nil {
			meta = map[string]string{"claim": wp.Claim}
		} else if _, ok := meta["claim"]; !ok {
			combined := make(map[string]string, len(meta)+1)
			for k, val := range meta {
				combined[k] = val
			}
			combined["claim"] = wp.Claim
			meta = combined
		}
	}
	return ClassifyRefusalSubtype(reason, meta)
}
