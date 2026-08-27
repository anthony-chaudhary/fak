package studymonitor

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestInventoryReportAcceptsValidatedStudyForgeReceipt(t *testing.T) {
	dir := t.TempDir()
	mapPath := writeInventoryMapForForgeTest(t, dir)
	receiptPath := writeStudyForgeReceiptForTest(t, dir, validStudyForgeReceiptForTest())
	registry := forgeInventoryRegistryForTest(mapPath, receiptPath)

	report := BuildInventoryReportWithMapFiles(registry, dir)
	if !report.OK {
		t.Fatalf("report = %+v, want validated receipt to satisfy forge source class", report)
	}
	row := report.Repositories[0]
	if row.ForgeReceiptPath != filepath.Base(receiptPath) {
		t.Fatalf("forge_receipt_path = %q", row.ForgeReceiptPath)
	}
	for _, evidence := range row.SourceEvidence {
		if evidence.Class == "open_closed_issues_prs_discussions" {
			t.Fatal("test accidentally satisfied forge class with free-form source_evidence")
		}
	}
}

func TestDecodeStudyForgeReceiptAcceptsStandaloneReceipt(t *testing.T) {
	want := validStudyForgeReceiptForTest()
	data, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeStudyForgeReceiptEvidence(data)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != want.Schema || got.Repository != want.Repository || len(got.Sources) != len(want.Sources) {
		t.Fatalf("decoded receipt = %+v", got)
	}
}

func TestValidateStudyForgeAcceptsAlreadyUpgradedExactLegacyProjection(t *testing.T) {
	receipt := validStudyForgeReceiptForTest()
	delta := receipt.NonAtomicDelta
	delta.EvidenceMode = ""
	delta.IdentityBasis = "legacy_checkpoint_projection"
	delta.MixedEvidence = ""
	delta.DedicatedEvidence = ""
	delta.RelationEvidence = ""
	delta.IdentitySetsAvailable = nil
	delta.ObservedEndpointCardinalityDelta = nil
	delta.Policy.Metric = ""
	delta.SymmetricDifferenceLowerBound = 0
	delta.SymmetricDifferenceUpperBound = 0
	delta.Verdict = ""
	if err := ValidateStudyForgeReceiptEvidence(receipt, "owner/repo", "abc"); err != nil {
		t.Fatalf("already-upgraded exact receipt rejected: %v", err)
	}
}

func TestValidateStudyForgeLegacyCountOnlyEvidenceObservedShape(t *testing.T) {
	receipt := legacyCountOnlyStudyForgeReceiptForTest()
	delta := receipt.NonAtomicDelta
	if err := ValidateStudyForgeReceiptEvidence(receipt, "owner/repo", "abc"); err != nil {
		t.Fatalf("count-only evidence = %v", err)
	}
	if delta.EvidenceMode != studyForgeEvidenceModeLegacyCountOnly || delta.RelationEvidence != studyForgeEvidenceUnavailable {
		t.Fatalf("count-only evidence is not typed: %+v", delta)
	}
	if delta.Overlap != nil || delta.OnlyInMixed != nil || delta.OnlyInDedicated != nil || delta.OverlapCount != nil || delta.OnlyInMixedCount != nil || delta.OnlyInDedicatedCount != nil {
		t.Fatalf("unrecoverable identity sets were fabricated: %+v", delta)
	}
	if delta.IdentitySetsAvailable == nil || *delta.IdentitySetsAvailable {
		t.Fatalf("legacy identity availability = %v, want explicit false", delta.IdentitySetsAvailable)
	}
	if delta.SymmetricDifferenceLowerBound != 23 || delta.SymmetricDifferenceUpperBound != 71079 {
		t.Fatalf("possible symmetric difference = %d..%d, want [23,71079]", delta.SymmetricDifferenceLowerBound, delta.SymmetricDifferenceUpperBound)
	}
	if delta.Policy.Metric != studyForgePolicyMetricCardinalityDelta || delta.ObservedEndpointCardinalityDelta == nil || *delta.ObservedEndpointCardinalityDelta != 23 {
		t.Fatalf("count-only policy metric = %+v", delta)
	}
	if delta.Verdict != studyForgePolicyEvaluationAccepted || !delta.Accepted {
		t.Fatalf("count-only verdict = %+v, want accepted cardinality delta", delta)
	}
}

func TestValidateStudyForgeLegacyCountOnlyRejectsContradictoryShapes(t *testing.T) {
	tests := []struct {
		name string
		edit func(*StudyForgeReceiptEvidence)
		want string
	}{
		{
			name: "fabricated empty overlap",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.Overlap = []StudyForgeCrossEndpointIdentity{}
			},
			want: "identity sets must be unavailable",
		},
		{
			name: "fabricated relation count",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				count := 0
				receipt.NonAtomicDelta.OverlapCount = &count
			},
			want: "identity sets must be unavailable",
		},
		{
			name: "forged bound",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.SymmetricDifferenceLowerBound++
			},
			want: "bounds contradict endpoint counts",
		},
		{
			name: "wrong verdict",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.Verdict = studyForgePolicyEvaluationRejected
				receipt.NonAtomicDelta.Accepted = false
			},
			want: "verdict contradicts its observed metric and policy",
		},
		{
			name: "wrong policy metric",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.Policy.Metric = studyForgePolicyMetricIdentityDelta
			},
			want: "acceptance policy is missing or unbounded",
		},
		{
			name: "missing policy metric",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.Policy.Metric = ""
			},
			want: "acceptance policy is missing or unbounded",
		},
		{
			name: "wrong observed cardinality delta",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				*receipt.NonAtomicDelta.ObservedEndpointCardinalityDelta++
			},
			want: "observed endpoint cardinality delta contradicts endpoint counts",
		},
		{
			name: "missing observed cardinality delta",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.ObservedEndpointCardinalityDelta = nil
			},
			want: "observed endpoint cardinality delta contradicts endpoint counts",
		},
		{
			name: "missing reason",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.EvidenceReason = ""
			},
			want: "legacy count-only reason",
		},
		{
			name: "identity availability missing",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.IdentitySetsAvailable = nil
			},
			want: "identity_sets_available must be false",
		},
		{
			name: "identity availability forged true",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				available := true
				receipt.NonAtomicDelta.IdentitySetsAvailable = &available
			},
			want: "identity_sets_available must be false",
		},
		{
			name: "projected basis",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.IdentityBasis = "legacy_checkpoint_projection"
			},
			want: "invalid identity_basis",
		},
		{
			name: "endpoint count contradiction",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta.MixedCount--
			},
			want: "counts contradict endpoint evidence",
		},
		{
			name: "overflow",
			edit: func(receipt *StudyForgeReceiptEvidence) {
				maxInt := int(^uint(0) >> 1)
				receipt.Sources[0].ClassifiedPullCount = maxInt
				receipt.Sources[1].NormalizedCount = 1
				receipt.NonAtomicDelta.MixedCount = maxInt
				receipt.NonAtomicDelta.DedicatedCount = 1
			},
			want: "overflow symmetric-difference bounds",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receipt := legacyCountOnlyStudyForgeReceiptForTest()
			tt.edit(&receipt)
			err := validateStudyForgeNonAtomicDelta(receipt.NonAtomicDelta, receipt.Sources[0], receipt.Sources[1])
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validation = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestValidateStudyForgeLegacyCountOnlyPolicyProofSemantics(t *testing.T) {
	t.Run("terminal cardinality delta within bound is accepted", func(t *testing.T) {
		receipt := legacyCountOnlyStudyForgeReceiptForTest()
		if err := ValidateStudyForgeReceiptEvidence(receipt, "owner/repo", "abc"); err != nil {
			t.Fatalf("within-bound count-only receipt = %v", err)
		}
	})

	t.Run("cardinality delta over bound is rejected", func(t *testing.T) {
		receipt := legacyCountOnlyStudyForgeReceiptForTest()
		receipt.Sources[1].FetchedCount = 36529
		receipt.Sources[1].NormalizedCount = 36529
		receipt.Sources[1].UniqueCount = 36529
		receipt.NonAtomicDelta = legacyCountOnlyDeltaForTest(35528, 36529, studyForgePolicyEvaluationRejected)
		if err := validateStudyForgeNonAtomicDelta(receipt.NonAtomicDelta, receipt.Sources[0], receipt.Sources[1]); err != nil {
			t.Fatalf("typed rejected evidence = %v", err)
		}
		err := ValidateStudyForgeReceiptEvidence(receipt, "owner/repo", "abc")
		if err == nil || !strings.Contains(err.Error(), "did not prove policy acceptance") {
			t.Fatalf("complete rejected receipt validation = %v", err)
		}
	})
}

func TestValidateStudyForgeExactIdentityPolicyMetricRemainsStrict(t *testing.T) {
	t.Run("exact identity evidence remains accepted", func(t *testing.T) {
		receipt := validStudyForgeReceiptForTest()
		if err := ValidateStudyForgeReceiptEvidence(receipt, "owner/repo", "abc"); err != nil {
			t.Fatalf("exact receipt rejected: %v", err)
		}
	})

	t.Run("legacy cardinality metric is rejected for exact evidence", func(t *testing.T) {
		receipt := validStudyForgeReceiptForTest()
		receipt.NonAtomicDelta.Policy.Metric = studyForgePolicyMetricCardinalityDelta
		err := ValidateStudyForgeReceiptEvidence(receipt, "owner/repo", "abc")
		if err == nil || !strings.Contains(err.Error(), "acceptance policy is missing or unbounded") {
			t.Fatalf("exact metric mismatch validation = %v", err)
		}
	})

	t.Run("identity symmetric difference still controls acceptance", func(t *testing.T) {
		receipt := validStudyForgeReceiptForTest()
		delta := receipt.NonAtomicDelta
		delta.Policy.MaxTotal = 0
		err := ValidateStudyForgeReceiptEvidence(receipt, "owner/repo", "abc")
		if err == nil || !strings.Contains(err.Error(), "policy did not accept the endpoint drift") {
			t.Fatalf("exact symmetric-difference validation = %v", err)
		}
	})
}

func TestInventoryReportRejectsPartialOrInconsistentStudyForgeReceipt(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*StudyForgeReceiptEvidence)
		want   string
	}{
		{
			name: "partial top level",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Status = "partial"
			},
			want: `status must be complete, got "partial"`,
		},
		{
			name: "missing discussions",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Sources = append(receipt.Sources[:2], receipt.Sources[3:]...)
			},
			want: "missing source discussions",
		},
		{
			name: "duplicate identity count",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Sources[0].UniqueCount--
			},
			want: "source issues normalized_count does not match unique_count",
		},
		{
			name: "partial source",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Sources[1].Status = "partial"
			},
			want: "source pulls status must be complete",
		},
		{
			name: "revision mismatch",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.Revision = "def"
			},
			want: "revision does not match checked_revision",
		},
		{
			name: "missing non atomic delta",
			mutate: func(receipt *StudyForgeReceiptEvidence) {
				receipt.NonAtomicDelta = nil
			},
			want: "non_atomic_delta is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			mapPath := writeInventoryMapForForgeTest(t, dir)
			receipt := validStudyForgeReceiptForTest()
			tt.mutate(&receipt)
			receiptPath := writeStudyForgeReceiptForTest(t, dir, receipt)

			report := BuildInventoryReportWithMapFiles(forgeInventoryRegistryForTest(mapPath, receiptPath), dir)
			if report.OK {
				t.Fatalf("report = %+v, want blocker", report)
			}
			reasons := strings.Join(report.Repositories[0].Reasons, "\n")
			if !strings.Contains(reasons, "studyforge receipt is invalid: "+tt.want) {
				t.Fatalf("reasons = %q, want %q", reasons, tt.want)
			}
			if !strings.Contains(reasons, "missing source classes: open_closed_issues_prs_discussions") {
				t.Fatalf("invalid receipt must not satisfy forge source class: %q", reasons)
			}
		})
	}
}

func validStudyForgeReceiptForTest() StudyForgeReceiptEvidence {
	sources := make([]StudyForgeSourceReceiptEvidence, 0, len(requiredStudyForgeSources))
	for _, name := range requiredStudyForgeSources {
		sources = append(sources, StudyForgeSourceReceiptEvidence{
			Name:            name,
			Status:          "complete",
			Pages:           []json.RawMessage{json.RawMessage(`{"number":1}`)},
			FetchedCount:    2,
			NormalizedCount: 2,
			UniqueCount:     2,
			Checksum:        "sha256:" + name,
		})
	}
	// The issues census legitimately fetches mixed pull-request rows and then
	// partitions them out of its normalized issue-only index.
	sources[0].FetchedCount = 3
	sources[0].ClassifiedPullCount = 1
	identity := StudyForgeCrossEndpointIdentity{ID: 1}
	overlapCount, onlyInMixedCount, onlyInDedicatedCount := 1, 0, 1
	return StudyForgeReceiptEvidence{
		Schema:     StudyForgeReceiptSchema,
		Repository: "owner/repo",
		Revision:   "abc",
		Cutoff:     "2026-08-26T22:00:00Z",
		Status:     "complete",
		Sources:    sources,
		NonAtomicDelta: &StudyForgeNonAtomicDeltaEvidence{
			Type: "non_atomic_delta", MixedSource: "issues", DedicatedSource: "pulls",
			EvidenceMode: studyForgeEvidenceModeExactIdentity, IdentityBasis: "captured_endpoint_rows",
			MixedEvidence: studyForgeEvidenceExactIdentities, DedicatedEvidence: studyForgeEvidenceExactIdentities, RelationEvidence: studyForgeEvidenceExactIdentities,
			MixedCrawl:     StudyForgeCrawlWindow{StartedAt: "2026-08-26T22:00:00Z", EndedAt: "2026-08-26T22:01:00Z"},
			DedicatedCrawl: StudyForgeCrawlWindow{StartedAt: "2026-08-26T22:01:00Z", EndedAt: "2026-08-26T22:02:00Z"},
			MixedCount:     1, DedicatedCount: 2, OverlapCount: &overlapCount, OnlyInMixedCount: &onlyInMixedCount, OnlyInDedicatedCount: &onlyInDedicatedCount,
			Overlap: []StudyForgeCrossEndpointIdentity{identity}, OnlyInMixed: []StudyForgeCrossEndpointIdentity{}, OnlyInDedicated: []StudyForgeCrossEndpointIdentity{{ID: 2}},
			SymmetricDifferenceLowerBound: 1, SymmetricDifferenceUpperBound: 1,
			Policy:  &StudyForgeNonAtomicDeltaPolicy{Type: "bounded_identity_delta", MaxOnlyInMixed: 1000, MaxOnlyInDedicated: 1000, MaxTotal: 1000},
			Verdict: studyForgePolicyEvaluationAccepted, Accepted: true,
		},
		IndexChecksum: "sha256:index",
	}
}

func legacyCountOnlyStudyForgeReceiptForTest() StudyForgeReceiptEvidence {
	receipt := validStudyForgeReceiptForTest()
	receipt.Sources[0].FetchedCount = 53134
	receipt.Sources[0].NormalizedCount = 17606
	receipt.Sources[0].UniqueCount = 17606
	receipt.Sources[0].ClassifiedPullCount = 35528
	receipt.Sources[1].FetchedCount = 35551
	receipt.Sources[1].NormalizedCount = 35551
	receipt.Sources[1].UniqueCount = 35551
	receipt.NonAtomicDelta = legacyCountOnlyDeltaForTest(35528, 35551, studyForgePolicyEvaluationAccepted)
	return receipt
}

func legacyCountOnlyDeltaForTest(mixed, dedicated int, evaluation string) *StudyForgeNonAtomicDeltaEvidence {
	lower := mixed - dedicated
	if lower < 0 {
		lower = -lower
	}
	identitySetsAvailable := false
	observedEndpointCardinalityDelta := lower
	return &StudyForgeNonAtomicDeltaEvidence{
		Type:                             "non_atomic_delta",
		MixedSource:                      "issues",
		DedicatedSource:                  "pulls",
		EvidenceMode:                     studyForgeEvidenceModeLegacyCountOnly,
		IdentitySetsAvailable:            &identitySetsAvailable,
		IdentityBasis:                    studyForgeLegacyCountOnlyBasis,
		EvidenceReason:                   studyForgeLegacyCountOnlyReason,
		MixedEvidence:                    studyForgeEvidenceExactCountOnly,
		DedicatedEvidence:                studyForgeEvidenceExactIdentities,
		RelationEvidence:                 studyForgeEvidenceUnavailable,
		MixedCrawl:                       StudyForgeCrawlWindow{StartedAt: "2026-08-26T22:00:00Z", EndedAt: "2026-08-26T22:01:00Z"},
		DedicatedCrawl:                   StudyForgeCrawlWindow{StartedAt: "2026-08-26T22:01:00Z", EndedAt: "2026-08-26T22:02:00Z"},
		MixedCount:                       mixed,
		DedicatedCount:                   dedicated,
		SymmetricDifferenceLowerBound:    lower,
		SymmetricDifferenceUpperBound:    mixed + dedicated,
		ObservedEndpointCardinalityDelta: &observedEndpointCardinalityDelta,
		Policy: &StudyForgeNonAtomicDeltaPolicy{
			Type:               "bounded_identity_delta",
			Metric:             studyForgePolicyMetricCardinalityDelta,
			MaxOnlyInMixed:     1000,
			MaxOnlyInDedicated: 1000,
			MaxTotal:           1000,
		},
		Verdict:  evaluation,
		Accepted: evaluation == studyForgePolicyEvaluationAccepted,
	}
}

func writeStudyForgeReceiptForTest(t *testing.T, dir string, receipt StudyForgeReceiptEvidence) string {
	t.Helper()
	// Capture writes the normalized corpus envelope; the monitor also accepts a
	// standalone receipt for content-addressed receipt-only artifacts.
	envelope := struct {
		Schema  string                    `json:"schema"`
		Receipt StudyForgeReceiptEvidence `json:"receipt"`
		Records []any                     `json:"records"`
	}{Schema: studyForgeCorpusSchema, Receipt: receipt, Records: []any{}}
	data, err := json.MarshalIndent(envelope, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "forge-receipt.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeInventoryMapForForgeTest(t *testing.T, dir string) string {
	t.Helper()
	var data bytes.Buffer
	if err := WriteInventoryMapJSON(&data, validInventoryMapForTest()); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "inventory-map.json")
	if err := os.WriteFile(path, data.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func forgeInventoryRegistryForTest(mapPath, receiptPath string) Registry {
	return Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{{
		Repository:      "owner/repo",
		URL:             "https://example/repo",
		Status:          "candidate",
		Priority:        1,
		Why:             "fresh",
		LastChecked:     "2026-08-26",
		CheckedRevision: "abc",
		Inventory: &InventoryContract{
			MapPath:          filepath.Base(mapPath),
			ForgeReceiptPath: filepath.Base(receiptPath),
			IndexedRevision:  "abc",
			SourceClasses:    append([]string(nil), RequiredInventorySourceClasses...),
			SourceEvidence: []InventorySourceEvidence{
				{Class: "fak_selfquery_witness", Evidence: []string{"fak capabilities owner/repo"}},
				{Class: "candidate_matrix", Evidence: []string{"docs/notes/STUDY.md#candidate-matrix"}},
				{Class: "issue_tracking", Evidence: []string{"#123"}},
			},
			SubsystemCount:     1,
			CompletenessCritic: "local map and external receipts reconciled",
		},
	}}}
}
