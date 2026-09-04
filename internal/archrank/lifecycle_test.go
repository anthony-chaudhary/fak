package archrank

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// TestArchitectureRankingLifecycle verifies the end-to-end lifecycle:
// JSON decoding, invariant validation, deterministic score sorting, and unranked tracking.
func TestArchitectureRankingLifecycle(t *testing.T) {
	rawDataset := `{
		"schema_version": "archrank.candidates/v1",
		"formulas": {
			"active_bytes": "active_bytes = active_weight_bytes + state_bytes + kv_bytes_at_envelope",
			"score": "quality_per_active_byte = quality / active_bytes"
		},
		"candidates": [
			{
				"id": "arch-alpha-7b",
				"architecture": "dense-transformer",
				"migration_class": "clean-room",
				"envelope_id": "tier-7b-ctx4k",
				"quality_metric": "mmlu-pro",
				"quality_source_kind": "measured-benchmark",
				"measurement_status": "accepted",
				"quality": 0.72,
				"active_weight_bytes": 14000000000,
				"state_bytes": 1000000,
				"kv_bytes_at_envelope": 500000000,
				"provenance": {
					"kind": "measured_artifact",
					"locator": "runbooks/eval_7b.log#alpha"
				}
			},
			{
				"id": "arch-beta-7b",
				"architecture": "mixture-of-experts",
				"migration_class": "clean-room",
				"envelope_id": "tier-7b-ctx4k",
				"quality_metric": "mmlu-pro",
				"quality_source_kind": "measured-benchmark",
				"measurement_status": "accepted",
				"quality": 0.78,
				"active_weight_bytes": 7000000000,
				"state_bytes": 2000000,
				"kv_bytes_at_envelope": 500000000,
				"provenance": {
					"kind": "synthetic_control_measurement",
					"locator": "runbooks/eval_7b.log#beta"
				}
			},
			{
				"id": "arch-gamma-7b-estimated",
				"architecture": "state-space-hybrid",
				"migration_class": "experimental",
				"envelope_id": "tier-7b-ctx4k",
				"quality_metric": "mmlu-pro",
				"quality_source_kind": "measured-benchmark",
				"measurement_status": "estimated",
				"quality": 0.81,
				"active_weight_bytes": 6000000000,
				"state_bytes": 50000000,
				"kv_bytes_at_envelope": 1000000,
				"provenance": {
					"kind": "literature_hypothesis",
					"url": "https://arxiv.org/abs/2401.00000"
				}
			},
			{
				"id": "arch-delta-singleton-14b",
				"architecture": "dense-transformer",
				"migration_class": "clean-room",
				"envelope_id": "tier-14b-ctx4k",
				"quality_metric": "mmlu-pro",
				"quality_source_kind": "measured-benchmark",
				"measurement_status": "accepted",
				"quality": 0.82,
				"active_weight_bytes": 28000000000,
				"state_bytes": 2000000,
				"kv_bytes_at_envelope": 1000000000,
				"provenance": {
					"kind": "measured_artifact",
					"locator": "runbooks/eval_14b.log#delta"
				}
			}
		]
	}`

	dataset, err := LoadJSON(strings.NewReader(rawDataset))
	if err != nil {
		t.Fatalf("LoadJSON failed in lifecycle: %v", err)
	}

	if err := dataset.Validate(); err != nil {
		t.Fatalf("Dataset.Validate failed in lifecycle: %v", err)
	}

	result, err := Rank(*dataset)
	if err != nil {
		t.Fatalf("Rank failed in lifecycle: %v", err)
	}

	if len(result.Groups) != 1 {
		t.Fatalf("got %d ranked groups, want 1", len(result.Groups))
	}
	group := result.Groups[0]
	if group.Key.EnvelopeID != "tier-7b-ctx4k" || group.Key.QualityMetric != "mmlu-pro" || group.Key.QualitySourceKind != "measured-benchmark" {
		t.Fatalf("unexpected group comparability key: %+v", group.Key)
	}
	if len(group.Rows) != 2 {
		t.Fatalf("got %d ranked rows in tier-7b, want 2", len(group.Rows))
	}

	if group.Rows[0].ID != "arch-beta-7b" || group.Rows[0].Rank != 1 {
		t.Errorf("row 0 expected arch-beta-7b rank 1, got %+v", group.Rows[0])
	}
	if group.Rows[1].ID != "arch-alpha-7b" || group.Rows[1].Rank != 2 {
		t.Errorf("row 1 expected arch-alpha-7b rank 2, got %+v", group.Rows[1])
	}

	if len(result.Unranked) != 2 {
		t.Fatalf("got %d unranked rows, want 2", len(result.Unranked))
	}
	unrankedMap := make(map[string]string)
	for _, u := range result.Unranked {
		unrankedMap[u.ID] = u.Reason
	}
	if reason, ok := unrankedMap["arch-gamma-7b-estimated"]; !ok || !strings.Contains(reason, "estimated row") {
		t.Errorf("unexpected unranked reason for estimated candidate: %q", reason)
	}
	if reason, ok := unrankedMap["arch-delta-singleton-14b"]; !ok || !strings.Contains(reason, "unmatched comparability key") {
		t.Errorf("unexpected unranked reason for singleton candidate: %q", reason)
	}

	repeatResult, err := Rank(*dataset)
	if err != nil {
		t.Fatalf("Rank repeat failed: %v", err)
	}
	if len(repeatResult.Groups) != len(result.Groups) || len(repeatResult.Unranked) != len(result.Unranked) {
		t.Fatal("Rank output is non-deterministic in group/unranked counts")
	}
	if repeatResult.Groups[0].Rows[0].ID != result.Groups[0].Rows[0].ID || repeatResult.Groups[0].Rows[1].ID != result.Groups[0].Rows[1].ID {
		t.Fatal("Rank output is non-deterministic in row ordering")
	}
}

// TestScoreEvaluationAndPrecision verifies arithmetic precision of quality per active byte,
// proper tie-breaking by ID, and sequential rank assignment.
func TestScoreEvaluationAndPrecision(t *testing.T) {
	c1 := Candidate{
		ID:                "cand-z-score",
		Architecture:      "arch-x",
		MigrationClass:    "class-a",
		EnvelopeID:        "env-tie",
		QualityMetric:     "metric-acc",
		QualitySourceKind: "source-audit",
		MeasurementStatus: "accepted",
		Quality:           0.50,
		ActiveWeightBytes: 50,
		StateBytes:        25,
		KVBytesAtEnvelope: 25,
		Provenance: Provenance{
			Kind:    "measured_artifact",
			Locator: "loc-z",
		},
	}
	c2 := Candidate{
		ID:                "cand-a-score",
		Architecture:      "arch-y",
		MigrationClass:    "class-a",
		EnvelopeID:        "env-tie",
		QualityMetric:     "metric-acc",
		QualitySourceKind: "source-audit",
		MeasurementStatus: "accepted",
		Quality:           1.00,
		ActiveWeightBytes: 150,
		StateBytes:        25,
		KVBytesAtEnvelope: 25,
		Provenance: Provenance{
			Kind:    "measured_artifact",
			Locator: "loc-a",
		},
	}
	c3 := Candidate{
		ID:                "cand-m-higher",
		Architecture:      "arch-z",
		MigrationClass:    "class-a",
		EnvelopeID:        "env-tie",
		QualityMetric:     "metric-acc",
		QualitySourceKind: "source-audit",
		MeasurementStatus: "accepted",
		Quality:           0.90,
		ActiveWeightBytes: 80,
		StateBytes:        10,
		KVBytesAtEnvelope: 10,
		Provenance: Provenance{
			Kind:    "measured_artifact",
			Locator: "loc-m",
		},
	}

	ds := validDataset(c1, c2, c3)
	res, err := Rank(ds)
	if err != nil {
		t.Fatalf("Rank failed: %v", err)
	}
	if len(res.Groups) != 1 {
		t.Fatalf("got %d groups, want 1", len(res.Groups))
	}
	rows := res.Groups[0].Rows
	if len(rows) != 3 {
		t.Fatalf("got %d rows, want 3", len(rows))
	}

	if rows[0].ID != "cand-m-higher" || rows[0].Rank != 1 {
		t.Errorf("expected cand-m-higher at rank 1, got %+v", rows[0])
	}
	if rows[1].ID != "cand-a-score" || rows[1].Rank != 2 {
		t.Errorf("expected cand-a-score at rank 2 (tie-break by ID), got %+v", rows[1])
	}
	if rows[2].ID != "cand-z-score" || rows[2].Rank != 3 {
		t.Errorf("expected cand-z-score at rank 3, got %+v", rows[2])
	}

	const eps = 1e-12
	if math.Abs(rows[0].QualityPerActiveByte-0.009) > eps {
		t.Errorf("score mismatch for row 0: got %v, want 0.009", rows[0].QualityPerActiveByte)
	}
	if math.Abs(rows[1].QualityPerActiveByte-0.005) > eps {
		t.Errorf("score mismatch for row 1: got %v, want 0.005", rows[1].QualityPerActiveByte)
	}
	if math.Abs(rows[2].QualityPerActiveByte-0.005) > eps {
		t.Errorf("score mismatch for row 2: got %v, want 0.005", rows[2].QualityPerActiveByte)
	}
}

// TestTierComparisonAndIsolation verifies that architecture candidates are strictly partitioned
// by their envelope tier, quality metric, and source kind. Cross-tier contamination must not occur.
func TestTierComparisonAndIsolation(t *testing.T) {
	edgeA := measuredCandidate("edge-arch-a")
	edgeA.EnvelopeID = "tier-edge-1b"
	edgeA.Quality = 0.60
	edgeA.ActiveWeightBytes = 1000
	edgeA.StateBytes = 0
	edgeA.KVBytesAtEnvelope = 0

	edgeB := measuredCandidate("edge-arch-b")
	edgeB.EnvelopeID = "tier-edge-1b"
	edgeB.Quality = 0.80
	edgeB.ActiveWeightBytes = 1000
	edgeB.StateBytes = 0
	edgeB.KVBytesAtEnvelope = 0

	dcA := measuredCandidate("dc-arch-a")
	dcA.EnvelopeID = "tier-dc-70b"
	dcA.Quality = 0.90
	dcA.ActiveWeightBytes = 70000
	dcA.StateBytes = 0
	dcA.KVBytesAtEnvelope = 0

	dcB := measuredCandidate("dc-arch-b")
	dcB.EnvelopeID = "tier-dc-70b"
	dcB.Quality = 0.85
	dcB.ActiveWeightBytes = 50000
	dcB.StateBytes = 0
	dcB.KVBytesAtEnvelope = 0

	cloudAlone := measuredCandidate("cloud-arch-lone")
	cloudAlone.EnvelopeID = "tier-cloud-405b"

	edgeDiffMetric := measuredCandidate("edge-different-metric")
	edgeDiffMetric.EnvelopeID = "tier-edge-1b"
	edgeDiffMetric.QualityMetric = "code-eval"

	dataset := validDataset(edgeA, edgeB, dcA, dcB, cloudAlone, edgeDiffMetric)
	result, err := Rank(dataset)
	if err != nil {
		t.Fatalf("Rank failed: %v", err)
	}

	if len(result.Groups) != 2 {
		t.Fatalf("got %d groups, want 2", len(result.Groups))
	}

	if result.Groups[0].Key.EnvelopeID != "tier-dc-70b" {
		t.Errorf("expected group 0 to be tier-dc-70b, got %q", result.Groups[0].Key.EnvelopeID)
	}
	if result.Groups[1].Key.EnvelopeID != "tier-edge-1b" {
		t.Errorf("expected group 1 to be tier-edge-1b, got %q", result.Groups[1].Key.EnvelopeID)
	}

	if result.Groups[0].Rows[0].ID != "dc-arch-b" || result.Groups[0].Rows[0].Rank != 1 {
		t.Errorf("expected dc-arch-b rank 1, got %+v", result.Groups[0].Rows[0])
	}
	if result.Groups[0].Rows[1].ID != "dc-arch-a" || result.Groups[0].Rows[1].Rank != 2 {
		t.Errorf("expected dc-arch-a rank 2, got %+v", result.Groups[0].Rows[1])
	}

	if result.Groups[1].Rows[0].ID != "edge-arch-b" || result.Groups[1].Rows[0].Rank != 1 {
		t.Errorf("expected edge-arch-b rank 1, got %+v", result.Groups[1].Rows[0])
	}
	if result.Groups[1].Rows[1].ID != "edge-arch-a" || result.Groups[1].Rows[1].Rank != 2 {
		t.Errorf("expected edge-arch-a rank 2, got %+v", result.Groups[1].Rows[1])
	}

	if len(result.Unranked) != 2 {
		t.Fatalf("got %d unranked, want 2", len(result.Unranked))
	}
	unrankedIDs := map[string]bool{
		result.Unranked[0].ID: true,
		result.Unranked[1].ID: true,
	}
	if !unrankedIDs["cloud-arch-lone"] || !unrankedIDs["edge-different-metric"] {
		t.Errorf("unexpected unranked items: %+v", result.Unranked)
	}
}

// TestActiveBytesArithmeticAndOverflow ensures arithmetic bounds and uint64 overflow protections
// are enforced fail-closed.
func TestActiveBytesArithmeticAndOverflow(t *testing.T) {
	c := Candidate{
		ActiveWeightBytes: 10,
		StateBytes:        20,
		KVBytesAtEnvelope: 30,
	}
	total, err := c.ActiveBytes()
	if err != nil {
		t.Fatalf("unexpected error on valid active bytes: %v", err)
	}
	if total != 60 {
		t.Errorf("got %d, want 60", total)
	}

	zeroC := Candidate{
		ActiveWeightBytes: 0,
		StateBytes:        0,
		KVBytesAtEnvelope: 0,
	}
	if _, err := zeroC.ActiveBytes(); err == nil || !strings.Contains(err.Error(), "greater than zero") {
		t.Errorf("expected greater than zero error, got %v", err)
	}

	overflowWS := Candidate{
		ActiveWeightBytes: math.MaxUint64 - 5,
		StateBytes:        10,
		KVBytesAtEnvelope: 0,
	}
	if _, err := overflowWS.ActiveBytes(); err == nil || !strings.Contains(err.Error(), "overflows uint64") {
		t.Errorf("expected overflow error on weight+state, got %v", err)
	}

	overflowKV := Candidate{
		ActiveWeightBytes: math.MaxUint64 / 2,
		StateBytes:        math.MaxUint64 / 2,
		KVBytesAtEnvelope: 10,
	}
	if _, err := overflowKV.ActiveBytes(); err == nil || !strings.Contains(err.Error(), "overflows uint64") {
		t.Errorf("expected overflow error on kv bytes, got %v", err)
	}

	maxC := Candidate{
		ActiveWeightBytes: math.MaxUint64 - 10,
		StateBytes:        6,
		KVBytesAtEnvelope: 4,
	}
	maxTotal, err := maxC.ActiveBytes()
	if err != nil {
		t.Fatalf("unexpected error on MaxUint64 sum: %v", err)
	}
	if maxTotal != math.MaxUint64 {
		t.Errorf("got %d, want %d", maxTotal, uint64(math.MaxUint64))
	}
}

// TestLoadJSONStrictDecoder verifies that LoadJSON disallows unknown fields and trailing data.
func TestLoadJSONStrictDecoder(t *testing.T) {
	unknownFieldJSON := `{
		"schema_version": "archrank.candidates/v1",
		"unknown_field": "unexpected",
		"formulas": {
			"active_bytes": "active_bytes = active_weight_bytes + state_bytes + kv_bytes_at_envelope",
			"score": "quality_per_active_byte = quality / active_bytes"
		},
		"candidates": []
	}`
	if _, err := LoadJSON(strings.NewReader(unknownFieldJSON)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Errorf("expected unknown field error, got %v", err)
	}

	trailingJSON := `{
		"schema_version": "archrank.candidates/v1",
		"formulas": {
			"active_bytes": "active_bytes = active_weight_bytes + state_bytes + kv_bytes_at_envelope",
			"score": "quality_per_active_byte = quality / active_bytes"
		},
		"candidates": [
			{
				"id": "c1",
				"architecture": "arch",
				"migration_class": "clean",
				"envelope_id": "env",
				"quality_metric": "qm",
				"quality_source_kind": "qsk",
				"measurement_status": "accepted",
				"quality": 1.0,
				"active_weight_bytes": 100,
				"state_bytes": 0,
				"kv_bytes_at_envelope": 0,
				"provenance": {
					"kind": "measured_artifact",
					"locator": "loc"
				}
			}
		]
	} {"extra": true}`
	if _, err := LoadJSON(strings.NewReader(trailingJSON)); err == nil || !strings.Contains(err.Error(), "multiple JSON values") {
		t.Errorf("expected multiple JSON values error, got %v", err)
	}

	if _, err := LoadJSON(strings.NewReader(`{malformed`)); err == nil {
		t.Error("expected error on malformed JSON, got nil")
	}

	if _, err := LoadJSON(strings.NewReader(``)); err == nil {
		t.Error("expected error on empty reader, got nil")
	}
}

// TestCandidateIntegrityInvariants verifies individual candidate field validation bounds.
func TestCandidateIntegrityInvariants(t *testing.T) {
	base := func() Candidate {
		return Candidate{
			ID:                "cand-valid",
			Architecture:      "transformer",
			MigrationClass:    "clean-room",
			EnvelopeID:        "env-1",
			QualityMetric:     "metric-1",
			QualitySourceKind: "source-1",
			MeasurementStatus: "accepted",
			Quality:           0.85,
			ActiveWeightBytes: 1000,
			StateBytes:        100,
			KVBytesAtEnvelope: 100,
			Provenance: Provenance{
				Kind:    "measured_artifact",
				Locator: "loc-1",
			},
		}
	}

	cases := []struct {
		name      string
		mutate    func(*Candidate)
		wantMatch string
	}{
		{
			name:      "empty id",
			mutate:    func(c *Candidate) { c.ID = "   " },
			wantMatch: "id is required",
		},
		{
			name:      "empty architecture",
			mutate:    func(c *Candidate) { c.Architecture = "" },
			wantMatch: "architecture is required",
		},
		{
			name:      "empty migration_class",
			mutate:    func(c *Candidate) { c.MigrationClass = "" },
			wantMatch: "migration_class is required",
		},
		{
			name:      "empty envelope_id",
			mutate:    func(c *Candidate) { c.EnvelopeID = "" },
			wantMatch: "envelope_id is required",
		},
		{
			name:      "empty quality_metric",
			mutate:    func(c *Candidate) { c.QualityMetric = "" },
			wantMatch: "quality_metric is required",
		},
		{
			name:      "empty quality_source_kind",
			mutate:    func(c *Candidate) { c.QualitySourceKind = "" },
			wantMatch: "quality_source_kind is required",
		},
		{
			name:      "empty provenance kind",
			mutate:    func(c *Candidate) { c.Provenance.Kind = "" },
			wantMatch: "provenance.kind is required",
		},
		{
			name:      "negative quality",
			mutate:    func(c *Candidate) { c.Quality = -0.1 },
			wantMatch: "quality must be finite and non-negative",
		},
		{
			name:      "NaN quality",
			mutate:    func(c *Candidate) { c.Quality = math.NaN() },
			wantMatch: "quality must be finite and non-negative",
		},
		{
			name:      "Inf quality",
			mutate:    func(c *Candidate) { c.Quality = math.Inf(1) },
			wantMatch: "quality must be finite and non-negative",
		},
		{
			name:      "accepted with empty locator",
			mutate:    func(c *Candidate) { c.Provenance.Locator = "  " },
			wantMatch: "accepted row provenance.locator is required",
		},
		{
			name: "estimated with non-url",
			mutate: func(c *Candidate) {
				c.MeasurementStatus = "estimated"
				c.Provenance.Kind = "literature_hypothesis"
				c.Provenance.URL = "ftp://invalid-scheme.org"
			},
			wantMatch: "must be an absolute http(s) URL",
		},
		{
			name: "unknown measurement status",
			mutate: func(c *Candidate) {
				c.MeasurementStatus = "provisional"
			},
			wantMatch: "measurement_status must be accepted or estimated",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cand := base()
			tc.mutate(&cand)
			ds := validDataset(cand)
			err := ds.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.wantMatch) {
				t.Fatalf("expected error containing %q, got %v", tc.wantMatch, err)
			}
		})
	}
}

// TestDatasetEmptyAndDuplicateCandidates verifies failure when dataset has 0 candidates
// or duplicate candidate IDs.
func TestDatasetEmptyAndDuplicateCandidates(t *testing.T) {
	emptyDS := validDataset()
	if err := emptyDS.Validate(); err == nil || !strings.Contains(err.Error(), "at least one row is required") {
		t.Errorf("expected error for empty candidates, got %v", err)
	}

	c1 := measuredCandidate("dup-id")
	c2 := measuredCandidate("dup-id")
	dupDS := validDataset(c1, c2)
	if err := dupDS.Validate(); err == nil || !strings.Contains(err.Error(), "duplicate id") {
		t.Errorf("expected error for duplicate candidate ID, got %v", err)
	}
}

// BenchmarkArchRank measures the throughput of ranking candidate datasets.
func BenchmarkArchRank(b *testing.B) {
	c1 := measuredCandidate("bench-cand-1")
	c2 := measuredCandidate("bench-cand-2")
	c3 := measuredCandidate("bench-cand-3")
	c4 := measuredCandidate("bench-cand-4")
	dataset := validDataset(c1, c2, c3, c4)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, err := Rank(dataset)
		if err != nil {
			b.Fatalf("Rank failed: %v", err)
		}
		if len(res.Groups) == 0 {
			b.Fatal("unexpected empty groups in benchmark")
		}
	}
}

// BenchmarkDatasetValidate measures the cost of schema, formula, and candidate invariant validation.
func BenchmarkDatasetValidate(b *testing.B) {
	c1 := measuredCandidate("bench-val-1")
	c2 := measuredCandidate("bench-val-2")
	dataset := validDataset(c1, c2)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := dataset.Validate(); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

// BenchmarkActiveBytes measures active-byte accounting formula calculation and overflow bounds.
func BenchmarkActiveBytes(b *testing.B) {
	cand := measuredCandidate("bench-active-bytes")

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bytes, err := cand.ActiveBytes()
		if err != nil || bytes == 0 {
			b.Fatalf("ActiveBytes failed: %v", err)
		}
	}
}

// BenchmarkLoadJSON measures JSON deserialization with strict unknown-field checking.
func BenchmarkLoadJSON(b *testing.B) {
	c1 := measuredCandidate("bench-load-1")
	c2 := measuredCandidate("bench-load-2")
	dataset := validDataset(c1, c2)
	data, err := json.Marshal(dataset)
	if err != nil {
		b.Fatalf("json.Marshal failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := LoadJSON(bytes.NewReader(data))
		if err != nil {
			b.Fatalf("LoadJSON failed: %v", err)
		}
	}
}
