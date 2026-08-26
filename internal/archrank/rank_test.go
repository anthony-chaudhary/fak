package archrank

import (
	"math"
	"strings"
	"testing"
)

func TestFixtureRanksSyntheticControlsAndLeavesHypothesesUnranked(t *testing.T) {
	dataset, err := LoadFile("testdata/candidates.json")
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}

	result, err := Rank(*dataset)
	if err != nil {
		t.Fatalf("Rank: %v", err)
	}
	if got, want := len(result.Groups), 1; got != want {
		t.Fatalf("ranked groups = %d, want %d", got, want)
	}
	rows := result.Groups[0].Rows
	if got, want := len(rows), 2; got != want {
		t.Fatalf("ranked rows = %d, want %d", got, want)
	}
	if rows[0].ID != "synthetic-control-b" || rows[0].Rank != 1 || rows[0].ActiveBytes != 100 {
		t.Fatalf("first row = %+v, want control B at rank 1 with 100 active bytes", rows[0])
	}
	if rows[1].ID != "synthetic-control-a" || rows[1].Rank != 2 || rows[1].ActiveBytes != 120 {
		t.Fatalf("second row = %+v, want control A at rank 2 with 120 active bytes", rows[1])
	}
	if diff := math.Abs(rows[0].QualityPerActiveByte - 0.0075); diff > 1e-15 {
		t.Fatalf("control B score = %.17g, want 0.0075", rows[0].QualityPerActiveByte)
	}
	if diff := math.Abs(rows[1].QualityPerActiveByte - (0.8 / 120.0)); diff > 1e-15 {
		t.Fatalf("control A score = %.17g, want %.17g", rows[1].QualityPerActiveByte, 0.8/120.0)
	}

	if got, want := len(result.Unranked), 6; got != want {
		t.Fatalf("unranked rows = %d, want %d", got, want)
	}
	for _, row := range result.Unranked {
		if !strings.HasPrefix(row.ID, "hypothesis-") {
			t.Errorf("unexpected unranked row: %+v", row)
		}
		if row.Reason != "estimated row: ranking requires accepted measured evidence" {
			t.Errorf("reason for %q = %q", row.ID, row.Reason)
		}
	}
}

func TestRankRequiresExactComparabilityKey(t *testing.T) {
	base := measuredCandidate("base")
	byEnvelope := measuredCandidate("different-envelope")
	byEnvelope.EnvelopeID = "another-envelope"
	byMetric := measuredCandidate("different-metric")
	byMetric.QualityMetric = "another-metric"
	bySource := measuredCandidate("different-source")
	bySource.QualitySourceKind = "another-source"

	dataset := validDataset(base, byEnvelope, byMetric, bySource)
	result, err := Rank(dataset)
	if err != nil {
		t.Fatalf("Rank: %v", err)
	}
	if len(result.Groups) != 0 {
		t.Fatalf("ranked groups = %+v, want none", result.Groups)
	}
	if got, want := len(result.Unranked), 4; got != want {
		t.Fatalf("unranked rows = %d, want %d", got, want)
	}
	for _, row := range result.Unranked {
		if !strings.Contains(row.Reason, "unmatched comparability key") ||
			!strings.Contains(row.Reason, "envelope_id=") ||
			!strings.Contains(row.Reason, "quality_metric=") ||
			!strings.Contains(row.Reason, "quality_source_kind=") {
			t.Errorf("reason for %q does not expose the exact key: %q", row.ID, row.Reason)
		}
	}
}

func TestDeterministicTieBreakUsesID(t *testing.T) {
	zeta := measuredCandidate("zeta")
	alpha := measuredCandidate("alpha")
	result, err := Rank(validDataset(zeta, alpha))
	if err != nil {
		t.Fatalf("Rank: %v", err)
	}
	if got := result.Groups[0].Rows[0].ID; got != "alpha" {
		t.Fatalf("first tied row = %q, want alpha", got)
	}
}

func TestValidationRejectsFormulaAndProvenanceDrift(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Dataset)
		want string
	}{
		{
			name: "active byte formula",
			edit: func(dataset *Dataset) { dataset.Formulas.ActiveBytes = "active_bytes = active_weight_bytes" },
			want: "formulas.active_bytes",
		},
		{
			name: "score formula",
			edit: func(dataset *Dataset) { dataset.Formulas.Score = "quality_per_active_byte = quality" },
			want: "formulas.score",
		},
		{
			name: "estimated provenance URL",
			edit: func(dataset *Dataset) {
				dataset.Candidates[0].MeasurementStatus = "estimated"
				dataset.Candidates[0].Provenance = Provenance{Kind: "literature_hypothesis", URL: "not-a-url"}
			},
			want: "absolute http(s) URL",
		},
		{
			name: "accepted provenance kind",
			edit: func(dataset *Dataset) {
				dataset.Candidates[0].Provenance = Provenance{Kind: "literature_hypothesis", URL: "https://example.com/paper"}
			},
			want: "accepted row requires measured provenance",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dataset := validDataset(measuredCandidate("candidate"))
			test.edit(&dataset)
			if err := dataset.Validate(); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Validate() error = %v, want containing %q", err, test.want)
			}
		})
	}
}

func TestValidationRejectsActiveByteOverflow(t *testing.T) {
	candidate := measuredCandidate("overflow")
	candidate.ActiveWeightBytes = math.MaxUint64
	candidate.StateBytes = 1
	dataset := validDataset(candidate)
	if err := dataset.Validate(); err == nil || !strings.Contains(err.Error(), "overflows") {
		t.Fatalf("Validate() error = %v, want overflow error", err)
	}
}

func measuredCandidate(id string) Candidate {
	return Candidate{
		ID:                id,
		Architecture:      "synthetic architecture",
		MigrationClass:    "method-witness-only",
		EnvelopeID:        "envelope",
		QualityMetric:     "quality",
		QualitySourceKind: "deterministic-test",
		MeasurementStatus: "accepted",
		Quality:           0.5,
		ActiveWeightBytes: 80,
		StateBytes:        10,
		KVBytesAtEnvelope: 10,
		Provenance: Provenance{
			Kind:    "synthetic_control_measurement",
			Locator: "rank_test.go#" + id,
		},
	}
}

func validDataset(candidates ...Candidate) Dataset {
	return Dataset{
		SchemaVersion: "archrank.candidates/v1",
		Formulas: Formulas{
			ActiveBytes: ActiveBytesFormula,
			Score:       ScoreFormula,
		},
		Candidates: candidates,
	}
}
