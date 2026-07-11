package modelscore

import "testing"

func TestSubmissionTruthTierRequirements(t *testing.T) {
	tests := []struct {
		name string
		p    Provenance
		ok   bool
	}{
		{"official", Provenance{SubmissionTruthTier: TierOfficialSubmission, SubmissionSource: "official.json"}, true},
		{"official missing", Provenance{SubmissionTruthTier: TierOfficialSubmission}, false},
		{"posthoc", Provenance{SubmissionTruthTier: TierAuthorPublishedPosthoc, SubmissionSource: "result", AuthorSource: "author post"}, true},
		{"posthoc missing author", Provenance{SubmissionTruthTier: TierAuthorPublishedPosthoc, SubmissionSource: "result"}, false},
		{"reconstructed", Provenance{SubmissionTruthTier: TierReconstructedFromBlog, SubmissionSource: "result", AuthorSource: "blog", ReconstructionWitness: "script+hash"}, true},
		{"reconstructed missing witness", Provenance{SubmissionTruthTier: TierReconstructedFromBlog, SubmissionSource: "result", AuthorSource: "blog"}, false},
		{"unavailable", Provenance{SubmissionTruthTier: TierUnavailable}, true},
		{"unavailable overclaims", Provenance{SubmissionTruthTier: TierUnavailable, SubmissionSource: "claimed"}, false},
		{"unknown", Provenance{SubmissionTruthTier: "trust-me"}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.p.ValidateSubmissionTruth()
			if (err == nil) != tc.ok {
				t.Fatalf("err=%v ok=%v", err, tc.ok)
			}
		})
	}
}
