package modelscore

import (
	"fmt"
	"strings"
)

type SubmissionTruthTier string

const (
	TierOfficialSubmission     SubmissionTruthTier = "official-submission"
	TierAuthorPublishedPosthoc SubmissionTruthTier = "author-published-posthoc"
	TierReconstructedFromBlog  SubmissionTruthTier = "reconstructed-from-blog"
	TierUnavailable            SubmissionTruthTier = "unavailable"
)

func (t SubmissionTruthTier) Valid() bool {
	switch t {
	case TierOfficialSubmission, TierAuthorPublishedPosthoc, TierReconstructedFromBlog, TierUnavailable:
		return true
	}
	return false
}

// InferredSubmissionTruthTier maps the existing source-confidence vocabulary
// onto the stricter submission-truth tier for legacy benchmark fixtures.
func InferredSubmissionTruthTier(sourceType string) SubmissionTruthTier {
	switch strings.ToLower(strings.TrimSpace(sourceType)) {
	case "official":
		return TierOfficialSubmission
	case "vendor_reported":
		return TierAuthorPublishedPosthoc
	case "community":
		return TierReconstructedFromBlog
	default:
		return TierUnavailable
	}
}

// ValidateSubmissionTruth enforces evidence requirements that grow stricter as
// provenance weakens. Official results require the submission artifact; posthoc
// results require an author source; reconstructions require both the source and
// a reproducible reconstruction witness. Unavailable carries no numeric truth.
func (p Provenance) ValidateSubmissionTruth() error {
	if !p.SubmissionTruthTier.Valid() {
		return fmt.Errorf("invalid submission truth tier %q", p.SubmissionTruthTier)
	}
	source := strings.TrimSpace(p.SubmissionSource)
	switch p.SubmissionTruthTier {
	case TierOfficialSubmission:
		if source == "" {
			return fmt.Errorf("official-submission requires submission_source")
		}
	case TierAuthorPublishedPosthoc:
		if source == "" || strings.TrimSpace(p.AuthorSource) == "" {
			return fmt.Errorf("author-published-posthoc requires submission_source and author_source")
		}
	case TierReconstructedFromBlog:
		if source == "" || strings.TrimSpace(p.AuthorSource) == "" || strings.TrimSpace(p.ReconstructionWitness) == "" {
			return fmt.Errorf("reconstructed-from-blog requires submission_source, author_source, and reconstruction_witness")
		}
	case TierUnavailable:
		if source != "" || strings.TrimSpace(p.AuthorSource) != "" || strings.TrimSpace(p.ReconstructionWitness) != "" {
			return fmt.Errorf("unavailable must not claim submission truth fields")
		}
	}
	return nil
}
