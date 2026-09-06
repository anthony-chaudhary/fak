package memq

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func samplePrecisionFixture() RecallPrecisionFixture {
	return RecallPrecisionFixture{
		Candidates: []RecallCandidateNote{
			{
				ID:       "note-01-fresh-active",
				Content:  "The test suite runs with go test ./internal/memq -v and requires Go 1.22+",
				IsStale:  false,
				Relevant: true,
			},
			{
				ID:       "note-02-stale-planted",
				Content:  "DEPRECATED: Old test runner used python scripts/run_tests.py at commit 0000000",
				IsStale:  true,
				Relevant: true,
			},
			{
				ID:       "note-03-fresh-irrelevant",
				Content:  "The UI color palette uses slate and emerald for dark mode backgrounds",
				IsStale:  false,
				Relevant: false,
			},
			{
				ID:       "note-04-stale-irrelevant",
				Content:  "Old SVN repository mirror was hosted at svn.internal.archive.org/repo",
				IsStale:  true,
				Relevant: false,
			},
		},
	}
}

// TestRecallPrecisionMatrix asserts the A/B/C matrix requirements from #9797:
// - Arm C renders zero planted stale notes.
// - At least one non-verified arm renders the planted stale note.
// - Every arm conserves fresh + stale = candidate denominator.
// - Rendered-byte counters reconcile exactly (rendered + withheld = total).
// - The receipt contains arm names, counts, bytes, verdicts, fixture digests, but no note bodies.
// - JSON/order is deterministic and the test performs no model or network calls.
func TestRecallPrecisionMatrix(t *testing.T) {
	fixture := samplePrecisionFixture()
	receipt := EvaluateRecallPrecisionMatrix(fixture)

	if receipt.Verdict != VerdictMatrixAdmitted {
		t.Fatalf("expected matrix verdict %s, got %s", VerdictMatrixAdmitted, receipt.Verdict)
	}

	if receipt.CandidateDenominator != len(fixture.Candidates) {
		t.Fatalf("expected denominator %d, got %d", len(fixture.Candidates), receipt.CandidateDenominator)
	}

	// 1. Arm C renders zero planted stale notes.
	armC := receipt.Arms[2]
	if armC.Arm != ArmVerifiedRecall {
		t.Fatalf("expected arm 2 to be %s, got %s", ArmVerifiedRecall, armC.Arm)
	}
	if armC.StaleRendered != 0 {
		t.Fatalf("arm C rendered %d stale notes; expected 0", armC.StaleRendered)
	}
	if armC.Verdict != VerdictClean {
		t.Fatalf("expected arm C verdict %s, got %s", VerdictClean, armC.Verdict)
	}

	// 2. At least one non-verified arm renders the planted stale note.
	armA := receipt.Arms[0]
	armB := receipt.Arms[1]
	if armA.StaleRendered == 0 && armB.StaleRendered == 0 {
		t.Fatalf("expected at least one non-verified arm to render planted stale notes (got armA: %d, armB: %d)", armA.StaleRendered, armB.StaleRendered)
	}

	// 3. Every arm conserves fresh + stale = candidate denominator.
	for _, arm := range receipt.Arms {
		if arm.Fresh+arm.Stale != arm.CandidateDenominator {
			t.Fatalf("arm %s did not conserve fresh+stale: %d + %d != %d",
				arm.Arm, arm.Fresh, arm.Stale, arm.CandidateDenominator)
		}
		if arm.Rendered+arm.Withheld != arm.CandidateDenominator {
			t.Fatalf("arm %s did not conserve rendered+withheld: %d + %d != %d",
				arm.Arm, arm.Rendered, arm.Withheld, arm.CandidateDenominator)
		}
		// 4. Rendered-byte counters reconcile exactly.
		if arm.RenderedBytes+arm.WithheldBytes != arm.TotalBytes {
			t.Fatalf("arm %s byte counters do not reconcile: %d + %d != %d",
				arm.Arm, arm.RenderedBytes, arm.WithheldBytes, arm.TotalBytes)
		}
		if arm.TotalBytes != receipt.TotalBytes {
			t.Fatalf("arm %s total bytes %d != receipt total bytes %d",
				arm.Arm, arm.TotalBytes, receipt.TotalBytes)
		}
	}

	// 5. Receipt contains no raw note bodies (payload-free).
	rawJSON, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	jsonStr := string(rawJSON)
	for _, note := range fixture.Candidates {
		if strings.Contains(jsonStr, note.Content) {
			t.Fatalf("receipt leaks note body for %s: %q", note.ID, note.Content)
		}
	}

	// 6. JSON serialization is deterministic.
	rawJSON2, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("marshal receipt second time: %v", err)
	}
	if !bytes.Equal(rawJSON, rawJSON2) {
		t.Fatalf("receipt JSON serialization is not deterministic")
	}

	// 7. Digests exist for all fixtures.
	for _, note := range fixture.Candidates {
		if _, ok := receipt.FixtureDigests[note.ID]; !ok {
			t.Fatalf("missing fixture digest for note %s", note.ID)
		}
	}
}

// TestRecallPrecisionMatrix_VacuousRefusal tests that if no arm leaks stale notes,
// the matrix refuses admission (prevents a vacuous corpus where no stale note was tested).
func TestRecallPrecisionMatrix_VacuousRefusal(t *testing.T) {
	fixture := RecallPrecisionFixture{
		Candidates: []RecallCandidateNote{
			{
				ID:       "fresh-only-1",
				Content:  "Clean fresh note 1",
				IsStale:  false,
				Relevant: true,
			},
			{
				ID:       "fresh-only-2",
				Content:  "Clean fresh note 2",
				IsStale:  false,
				Relevant: false,
			},
		},
	}
	receipt := EvaluateRecallPrecisionMatrix(fixture)
	if receipt.Verdict != VerdictMatrixRefused {
		t.Fatalf("expected vacuous corpus to yield %s, got %s", VerdictMatrixRefused, receipt.Verdict)
	}
}
