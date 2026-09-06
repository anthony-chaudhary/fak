package memq

import (
	"crypto/sha256"
	"encoding/hex"
)

const (
	ArmRawDump        = "raw_dump"
	ArmRelevanceOnly  = "relevance_only"
	ArmVerifiedRecall = "verified_recall"

	VerdictClean       = "CLEAN"
	VerdictStaleLeaked = "STALE_LEAKED"

	VerdictMatrixAdmitted = "ADMITTED"
	VerdictMatrixRefused  = "REFUSED"
)

// RecallCandidateNote is one candidate note in the precision fixture inventory.
type RecallCandidateNote struct {
	ID       string `json:"id"`
	Content  string `json:"content"`
	IsStale  bool   `json:"is_stale"`
	Relevant bool   `json:"relevant"`
}

// RecallPrecisionArmReceipt is one arm's outcome in the recall precision matrix.
// It contains counts, bytes, and verdicts, but strictly NO raw note bodies.
type RecallPrecisionArmReceipt struct {
	Arm                  string   `json:"arm"`
	CandidateDenominator int      `json:"candidate_denominator"`
	Fresh                int      `json:"fresh"`
	Stale                int      `json:"stale"`
	Rendered             int      `json:"rendered"`
	Withheld             int      `json:"withheld"`
	FreshRendered        int      `json:"fresh_rendered"`
	StaleRendered        int      `json:"stale_rendered"`
	FreshWithheld        int      `json:"fresh_withheld"`
	StaleWithheld        int      `json:"stale_withheld"`
	RenderedBytes        int      `json:"rendered_bytes"`
	WithheldBytes        int      `json:"withheld_bytes"`
	TotalBytes           int      `json:"total_bytes"`
	Precision            float64  `json:"precision"`
	Verdict              string   `json:"verdict"`
	RenderedIDs          []string `json:"rendered_ids"`
	WithheldIDs          []string `json:"withheld_ids"`
}

// RecallPrecisionMatrixReceipt is the payload-free receipt for the recall precision matrix.
type RecallPrecisionMatrixReceipt struct {
	Schema               string                      `json:"schema"`
	Verdict              string                      `json:"verdict"`
	CandidateDenominator int                         `json:"candidate_denominator"`
	Fresh                int                         `json:"fresh"`
	Stale                int                         `json:"stale"`
	TotalBytes           int                         `json:"total_bytes"`
	FixtureDigests       map[string]string           `json:"fixture_digests"`
	Verdicts             map[string]string           `json:"verdicts"`
	Arms                 []RecallPrecisionArmReceipt `json:"arms"`
}

// RecallPrecisionFixture defines the ordered candidate notes and expected outcome.
type RecallPrecisionFixture struct {
	Candidates []RecallCandidateNote
}

// EvaluateRecallPrecisionMatrix evaluates arms A, B, and C against the fixture.
func EvaluateRecallPrecisionMatrix(fixture RecallPrecisionFixture) RecallPrecisionMatrixReceipt {
	freshCount := 0
	staleCount := 0
	totalBytes := 0
	candidateBytes := make(map[string]int)
	fixtureDigests := make(map[string]string)

	for _, c := range fixture.Candidates {
		b := len([]byte(c.Content))
		candidateBytes[c.ID] = b
		totalBytes += b
		if c.IsStale {
			staleCount++
		} else {
			freshCount++
		}
		h := sha256.Sum256([]byte(c.Content))
		fixtureDigests[c.ID] = hex.EncodeToString(h[:8])
	}

	n := len(fixture.Candidates)

	// Arm A: Raw memory dump (all candidates rendered)
	armA := RecallPrecisionArmReceipt{
		Arm:                  ArmRawDump,
		CandidateDenominator: n,
		Fresh:                freshCount,
		Stale:                staleCount,
		TotalBytes:           totalBytes,
		RenderedIDs:          make([]string, 0, n),
		WithheldIDs:          make([]string, 0),
	}
	for _, c := range fixture.Candidates {
		b := candidateBytes[c.ID]
		armA.Rendered++
		armA.RenderedBytes += b
		armA.RenderedIDs = append(armA.RenderedIDs, c.ID)
		if c.IsStale {
			armA.StaleRendered++
		} else {
			armA.FreshRendered++
		}
	}
	if armA.Rendered > 0 {
		armA.Precision = float64(armA.FreshRendered) / float64(armA.Rendered)
	}
	if armA.StaleRendered == 0 {
		armA.Verdict = VerdictClean
	} else {
		armA.Verdict = VerdictStaleLeaked
	}

	// Arm B: Relevance-only tuned dump without verification (relevant candidates rendered regardless of staleness)
	armB := RecallPrecisionArmReceipt{
		Arm:                  ArmRelevanceOnly,
		CandidateDenominator: n,
		Fresh:                freshCount,
		Stale:                staleCount,
		TotalBytes:           totalBytes,
		RenderedIDs:          make([]string, 0),
		WithheldIDs:          make([]string, 0),
	}
	for _, c := range fixture.Candidates {
		b := candidateBytes[c.ID]
		if c.Relevant {
			armB.Rendered++
			armB.RenderedBytes += b
			armB.RenderedIDs = append(armB.RenderedIDs, c.ID)
			if c.IsStale {
				armB.StaleRendered++
			} else {
				armB.FreshRendered++
			}
		} else {
			armB.Withheld++
			armB.WithheldBytes += b
			armB.WithheldIDs = append(armB.WithheldIDs, c.ID)
			if c.IsStale {
				armB.StaleWithheld++
			} else {
				armB.FreshWithheld++
			}
		}
	}
	if armB.Rendered > 0 {
		armB.Precision = float64(armB.FreshRendered) / float64(armB.Rendered)
	}
	if armB.StaleRendered == 0 {
		armB.Verdict = VerdictClean
	} else {
		armB.Verdict = VerdictStaleLeaked
	}

	// Arm C: Verified recall (only relevant and fresh candidates rendered; stale withheld)
	armC := RecallPrecisionArmReceipt{
		Arm:                  ArmVerifiedRecall,
		CandidateDenominator: n,
		Fresh:                freshCount,
		Stale:                staleCount,
		TotalBytes:           totalBytes,
		RenderedIDs:          make([]string, 0),
		WithheldIDs:          make([]string, 0),
	}
	for _, c := range fixture.Candidates {
		b := candidateBytes[c.ID]
		if c.Relevant && !c.IsStale {
			armC.Rendered++
			armC.RenderedBytes += b
			armC.RenderedIDs = append(armC.RenderedIDs, c.ID)
			armC.FreshRendered++
		} else {
			armC.Withheld++
			armC.WithheldBytes += b
			armC.WithheldIDs = append(armC.WithheldIDs, c.ID)
			if c.IsStale {
				armC.StaleWithheld++
			} else {
				armC.FreshWithheld++
			}
		}
	}
	if armC.Rendered > 0 {
		armC.Precision = float64(armC.FreshRendered) / float64(armC.Rendered)
	} else {
		armC.Precision = 1.0
	}
	if armC.StaleRendered == 0 {
		armC.Verdict = VerdictClean
	} else {
		armC.Verdict = VerdictStaleLeaked
	}

	matrixVerdict := VerdictMatrixAdmitted
	if armC.StaleRendered > 0 || (armA.StaleRendered == 0 && armB.StaleRendered == 0) {
		matrixVerdict = VerdictMatrixRefused
	}

	return RecallPrecisionMatrixReceipt{
		Schema:               "fak.memq.recall_value_precision.v1",
		Verdict:              matrixVerdict,
		CandidateDenominator: n,
		Fresh:                freshCount,
		Stale:                staleCount,
		TotalBytes:           totalBytes,
		FixtureDigests:       fixtureDigests,
		Verdicts: map[string]string{
			ArmRawDump:        armA.Verdict,
			ArmRelevanceOnly:  armB.Verdict,
			ArmVerifiedRecall: armC.Verdict,
		},
		Arms: []RecallPrecisionArmReceipt{armA, armB, armC},
	}
}
