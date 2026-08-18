package disambiguation

import "errors"

const PolicySourceSelfTestSchemaVersion = "fak-disambiguation-policy-source-self-test/1"

type PolicySourceResolution struct {
	Term          string `json:"term"`
	CanonicalTerm string `json:"canonical_term"`
	SourcePath    string `json:"source_path"`
	OwnerLeaf     string `json:"owner_leaf"`
}

type PolicySourceSelfTestReport struct {
	Schema                     string                   `json:"schema"`
	IndexVersion               string                   `json:"index_version"`
	Resolutions                []PolicySourceResolution `json:"resolutions"`
	StructuralBeforeModel      bool                     `json:"structural_before_model"`
	CapabilityNotVerdict       bool                     `json:"capability_not_verdict"`
	IncompatibleReasonRejected bool                     `json:"incompatible_reason_rejected"`
}

func RunPolicySourceSelfTest() (PolicySourceSelfTestReport, error) {
	terms := []string{"policy declaration", "capability floor", "adjudication verdict", "structural preflight", "model-mediated check", "ABI refusal reason"}
	report := PolicySourceSelfTestReport{Schema: PolicySourceSelfTestSchemaVersion, IndexVersion: PublicIndexVersion}
	for _, term := range terms {
		resolved, err := Resolve(term)
		if err != nil {
			return report, err
		}
		if len(resolved.Entry.Sources) == 0 {
			return report, errors.New("policy source entry has no public source")
		}
		report.Resolutions = append(report.Resolutions, PolicySourceResolution{
			Term: term, CanonicalTerm: resolved.Entry.Identity.CanonicalTerm,
			SourcePath: resolved.Entry.Sources[0].Locator, OwnerLeaf: resolved.Entry.Owner.Leaf,
		})
	}
	structural, _ := Query("structural preflight")
	capability, _ := Query("capability floor")
	verdict, _ := Query("adjudication verdict")
	report.StructuralBeforeModel = hasForbiddenContrast(structural.Entry, "model-mediated check")
	report.CapabilityNotVerdict = hasForbiddenContrast(capability.Entry, verdict.Entry.Identity.CanonicalTerm)

	collision := []VocabularyTerm{
		{Code: "POLICY_BLOCK", Kind: VocabularyReason, Package: "abi", Symbol: "ReasonPolicyBlock", CanonicalMeaning: "policy-refusal", SourcePath: "internal/abi/types.go"},
		{Code: "POLICY_BLOCK", Kind: VocabularyVerdict, Package: "policy", Symbol: "VerdictPolicyBlock", CanonicalMeaning: "policy-outcome", SourcePath: "internal/policy/posture.go"},
	}
	report.IncompatibleReasonRejected = errors.Is(ValidateVocabulary(collision), ErrVocabularyCollision)
	if !report.StructuralBeforeModel || !report.CapabilityNotVerdict || !report.IncompatibleReasonRejected {
		return report, errors.New("policy terminology invariants failed")
	}
	return report, nil
}

func hasForbiddenContrast(entry Entry, canonicalTerm string) bool {
	for _, contrast := range entry.Contrasts {
		if contrast.CanonicalTerm == canonicalTerm && contrast.ForbiddenConflation != nil && *contrast.ForbiddenConflation {
			return true
		}
	}
	return false
}
