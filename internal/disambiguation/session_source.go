package disambiguation

import (
	"errors"
	"fmt"
)

const SessionSourceSelfTestSchemaVersion = "fak-disambiguation-session-source-self-test/1"

// SessionSourceResolution proves one public term resolves to its intended
// canonical identity rather than a similarly named session mechanism.
type SessionSourceResolution struct {
	Input         string `json:"input"`
	CanonicalTerm string `json:"canonical_term"`
	SourcePath    string `json:"source_path"`
}

// SessionSourceSelfTestReport is the public fixture witness for #6314.
type SessionSourceSelfTestReport struct {
	Schema                         string                    `json:"schema"`
	IndexVersion                   string                    `json:"index_version"`
	Resolutions                    []SessionSourceResolution `json:"resolutions"`
	ResumeRecoveryConflation       bool                      `json:"resume_recovery_conflation_rejected"`
	CompactionCheckpointConflation bool                      `json:"compaction_checkpoint_conflation_rejected"`
}

// RunSessionSourceSelfTest resolves every session-family alias and verifies the
// two required forbidden distinctions are present in both directions.
func RunSessionSourceSelfTest() (SessionSourceSelfTestReport, error) {
	terms := []string{"session", "resume", "recovery", "compaction", "checkpoint"}
	report := SessionSourceSelfTestReport{Schema: SessionSourceSelfTestSchemaVersion, IndexVersion: PublicIndexVersion, Resolutions: make([]SessionSourceResolution, 0, len(terms))}
	for _, term := range terms {
		resolved, err := Resolve(term)
		if err != nil {
			return SessionSourceSelfTestReport{}, fmt.Errorf("resolve %q: %w", term, err)
		}
		if len(resolved.Entry.Sources) == 0 {
			return SessionSourceSelfTestReport{}, fmt.Errorf("resolve %q returned no public source", term)
		}
		report.Resolutions = append(report.Resolutions, SessionSourceResolution{Input: term, CanonicalTerm: resolved.Entry.Identity.CanonicalTerm, SourcePath: resolved.Entry.Sources[0].Locator})
	}
	var err error
	report.ResumeRecoveryConflation, err = requiredForbiddenPair("session resume", "session recovery")
	if err != nil {
		return SessionSourceSelfTestReport{}, err
	}
	report.CompactionCheckpointConflation, err = requiredForbiddenPair("context compaction", "recovery checkpoint")
	if err != nil {
		return SessionSourceSelfTestReport{}, err
	}
	return report, nil
}

func requiredForbiddenPair(leftTerm, rightTerm string) (bool, error) {
	left, err := Query(leftTerm)
	if err != nil {
		return false, err
	}
	right, err := Query(rightTerm)
	if err != nil {
		return false, err
	}
	leftContrast, leftOK := contrastTo(left.Entry, rightTerm)
	rightContrast, rightOK := contrastTo(right.Entry, leftTerm)
	if !leftOK || !rightOK || leftContrast.RequiredPair == nil || rightContrast.RequiredPair == nil || leftContrast.ForbiddenConflation == nil || rightContrast.ForbiddenConflation == nil {
		return false, errors.New("required forbidden contrast is incomplete")
	}
	if !*leftContrast.RequiredPair || !*rightContrast.RequiredPair || !*leftContrast.ForbiddenConflation || !*rightContrast.ForbiddenConflation {
		return false, errors.New("required forbidden contrast is not enforced")
	}
	return true, nil
}
