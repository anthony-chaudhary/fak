package benchauthority

import (
	"errors"
	"strings"
	"testing"
)

// Helper to create float pointer for optional thresholds.
func floatPtr(v float64) *float64 {
	return &v
}

func TestParserCompleteness_ValidCompletelyParsedFixture(t *testing.T) {
	receipt := ParserCompletenessReceipt{
		Schema: ParserCompletenessSchema,
		Source: SourceIdentity{
			Path:   "benchmarks/agentic_eval.jsonl",
			SHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		},
		ParserVersion:          "1.4.0",
		ParserSchema:           "jsonl-agentic/v1",
		TotalBytes:             20480,
		TotalRecords:           150,
		ParsedRecords:          150,
		IgnoredByPolicyRecords: 0,
		UnknownFamilyRecords:   0,
		MalformedRecords:       0,
		TruncatedRecords:       0,
		DuplicateRecords:       0,
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("expected valid receipt to pass Validate(), got: %v", err)
	}

	if got := receipt.CompletenessRatio(); got != 1.0 {
		t.Errorf("expected CompletenessRatio = 1.0, got %f", got)
	}
	if got := receipt.MalformedRatio(); got != 0.0 {
		t.Errorf("expected MalformedRatio = 0.0, got %f", got)
	}
	if got := receipt.UnknownFamilyRatio(); got != 0.0 {
		t.Errorf("expected UnknownFamilyRatio = 0.0, got %f", got)
	}
	if got := receipt.TruncatedRatio(); got != 0.0 {
		t.Errorf("expected TruncatedRatio = 0.0, got %f", got)
	}
	if got := receipt.DuplicateRatio(); got != 0.0 {
		t.Errorf("expected DuplicateRatio = 0.0, got %f", got)
	}

	// Admission with strict thresholds (1.0 completeness, 0.0 malformed)
	verdict := receipt.EvaluateClaimAdmission(CompletenessThresholds{
		MinCompletenessRatio: 1.0,
		MaxMalformedRatio:    0.0,
	})
	if !verdict.Admitted() {
		t.Errorf("expected clean receipt to be admitted, got verdict: %+v", verdict)
	}
	if verdict.Verdict != Admit {
		t.Errorf("expected verdict Admit, got %s", verdict.Verdict)
	}
	if verdict.Reason != "" {
		t.Errorf("expected empty reason on admission, got %q", verdict.Reason)
	}
}

func TestParserCompleteness_ConservingMixedFixture(t *testing.T) {
	receipt := ParserCompletenessReceipt{
		Schema: ParserCompletenessSchema,
		Source: SourceIdentity{
			Artifact: "mixed_run_2026.jsonl",
			SHA256:   "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
		},
		ParserVersion:          "2.1.0-rev4",
		ParserSchema:           "jsonl-turn/v2",
		TotalBytes:             8192,
		TotalRecords:           100,
		ParsedRecords:          70,
		IgnoredByPolicyRecords: 10,
		UnknownFamilyRecords:   10,
		MalformedRecords:       5,
		TruncatedRecords:       3,
		DuplicateRecords:       2,
	}

	// Sum = 70 + 10 + 10 + 5 + 3 + 2 = 100 == TotalRecords
	receipt.AddUnknownSample(RecordSample{
		Line:    12,
		Offset:  1048,
		Family:  "legacy_v0_turn",
		Reason:  "unrecognized turn envelope v0",
		Snippet: `{"type": "legacy_turn", "id": 12}`,
	}, 10)

	receipt.AddMalformedSample(RecordSample{
		Line:    45,
		Offset:  3490,
		Family:  "tool_call",
		Reason:  "unexpected EOF while parsing string",
		Snippet: `{"call": "exec", "arg": "echo 'unclosed`,
	}, 10)

	if err := receipt.Validate(); err != nil {
		t.Fatalf("expected mixed conserving receipt to pass Validate(), got: %v", err)
	}

	if len(receipt.UnknownSamples) != 1 {
		t.Errorf("expected 1 unknown sample, got %d", len(receipt.UnknownSamples))
	}
	if len(receipt.MalformedSamples) != 1 {
		t.Errorf("expected 1 malformed sample, got %d", len(receipt.MalformedSamples))
	}

	// Verify ratios
	if got := receipt.CompletenessRatio(); got != 0.70 {
		t.Errorf("expected CompletenessRatio = 0.70, got %f", got)
	}
	if got := receipt.MalformedRatio(); got != 0.05 {
		t.Errorf("expected MalformedRatio = 0.05, got %f", got)
	}
	if got := receipt.UnknownFamilyRatio(); got != 0.10 {
		t.Errorf("expected UnknownFamilyRatio = 0.10, got %f", got)
	}
	if got := receipt.TruncatedRatio(); got != 0.03 {
		t.Errorf("expected TruncatedRatio = 0.03, got %f", got)
	}
	if got := receipt.DuplicateRatio(); got != 0.02 {
		t.Errorf("expected DuplicateRatio = 0.02, got %f", got)
	}

	// Verify JSON round-trip
	data, err := receipt.JSON()
	if err != nil {
		t.Fatalf("receipt.JSON() error: %v", err)
	}
	decoded, err := DecodeParserCompletenessReceipt(data)
	if err != nil {
		t.Fatalf("DecodeParserCompletenessReceipt error: %v", err)
	}
	if err := decoded.Validate(); err != nil {
		t.Fatalf("decoded receipt failed validation: %v", err)
	}
	if decoded.TotalRecords != receipt.TotalRecords || decoded.ParsedRecords != receipt.ParsedRecords {
		t.Errorf("round-trip mismatch: got %d parsed, want %d", decoded.ParsedRecords, receipt.ParsedRecords)
	}
}

func TestParserCompleteness_BrokenConservationFailsValidation(t *testing.T) {
	tests := []struct {
		name    string
		receipt ParserCompletenessReceipt
	}{
		{
			name: "under-accounted sum less than total",
			receipt: ParserCompletenessReceipt{
				Schema: ParserCompletenessSchema,
				Source: SourceIdentity{
					Path:   "test.jsonl",
					SHA256: "abc",
				},
				TotalRecords:  100,
				ParsedRecords: 90, // sum = 90 != 100
			},
		},
		{
			name: "over-accounted sum greater than total",
			receipt: ParserCompletenessReceipt{
				Schema: ParserCompletenessSchema,
				Source: SourceIdentity{
					Path:   "test.jsonl",
					SHA256: "abc",
				},
				TotalRecords:     100,
				ParsedRecords:    90,
				MalformedRecords: 15, // sum = 105 != 100
			},
		},
		{
			name: "zero total with non-zero parts",
			receipt: ParserCompletenessReceipt{
				Schema: ParserCompletenessSchema,
				Source: SourceIdentity{
					Path:   "test.jsonl",
					SHA256: "abc",
				},
				TotalRecords:  0,
				ParsedRecords: 1, // sum = 1 != 0
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.receipt.Validate()
			if err == nil {
				t.Fatal("expected Validate() to fail on broken conservation, got nil")
			}
			if !errors.Is(err, ErrAccountingIncongruent) {
				t.Errorf("expected ErrAccountingIncongruent, got: %v", err)
			}

			// Admission gate must also fail closed with ReasonAccountingIncongruent
			verdict := tc.receipt.EvaluateClaimAdmission(CompletenessThresholds{
				MinCompletenessRatio: 0.5,
			})
			if verdict.Admitted() {
				t.Error("expected broken conservation to be denied by admission gate")
			}
			if verdict.Verdict != Deny {
				t.Errorf("expected Deny verdict, got %s", verdict.Verdict)
			}
			if verdict.Reason != ReasonAccountingIncongruent {
				t.Errorf("expected ReasonAccountingIncongruent, got %s", verdict.Reason)
			}
		})
	}
}

func TestParserCompleteness_AdmissionEvaluation(t *testing.T) {
	baseReceipt := ParserCompletenessReceipt{
		Schema: ParserCompletenessSchema,
		Source: SourceIdentity{
			Path:   "bench.jsonl",
			SHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		},
		ParserVersion: "1.0.0",
		ParserSchema:  "jsonl/v1",
		TotalRecords:  100,
	}

	t.Run("clean passes strict 1.0 threshold", func(t *testing.T) {
		r := baseReceipt
		r.ParsedRecords = 100

		verdict := r.EvaluateClaimAdmission(CompletenessThresholds{
			MinCompletenessRatio: 1.0,
			MaxMalformedRatio:    0.0,
		})
		if !verdict.Admitted() {
			t.Fatalf("expected admission, got %s (%s)", verdict.Verdict, verdict.Reason)
		}
	})

	t.Run("clean passes tolerant 0.95 threshold", func(t *testing.T) {
		r := baseReceipt
		r.ParsedRecords = 100

		verdict := r.EvaluateClaimAdmission(CompletenessThresholds{
			MinCompletenessRatio: 0.95,
			MaxMalformedRatio:    0.0,
		})
		if !verdict.Admitted() {
			t.Fatalf("expected admission, got %s (%s)", verdict.Verdict, verdict.Reason)
		}
	})

	t.Run("lossy input denies on completeness deficit", func(t *testing.T) {
		r := baseReceipt
		r.ParsedRecords = 90
		r.IgnoredByPolicyRecords = 10 // 90% completeness

		verdict := r.EvaluateClaimAdmission(CompletenessThresholds{
			MinCompletenessRatio: 0.95,
			MaxMalformedRatio:    0.0,
		})
		if verdict.Admitted() {
			t.Fatal("expected denial on completeness deficit")
		}
		if verdict.Verdict != Deny {
			t.Errorf("expected Deny, got %s", verdict.Verdict)
		}
		if verdict.Reason != ReasonParserCompletenessDeficit {
			t.Errorf("expected %s, got %s", ReasonParserCompletenessDeficit, verdict.Reason)
		}
	})

	t.Run("lossy input denies on malformed record breach", func(t *testing.T) {
		r := baseReceipt
		r.ParsedRecords = 96
		r.MalformedRecords = 4 // 96% parsed, 4% malformed

		verdict := r.EvaluateClaimAdmission(CompletenessThresholds{
			MinCompletenessRatio: 0.95, // 96% >= 95% passes completeness
			MaxMalformedRatio:    0.0,  // but 4% > 0% breaches malformed threshold
		})
		if verdict.Admitted() {
			t.Fatal("expected denial on malformed breach")
		}
		if verdict.Verdict != Deny {
			t.Errorf("expected Deny, got %s", verdict.Verdict)
		}
		if verdict.Reason != ReasonMalformedRecordBreach {
			t.Errorf("expected %s, got %s", ReasonMalformedRecordBreach, verdict.Reason)
		}
	})

	t.Run("lossy input denies on unknown family breach", func(t *testing.T) {
		r := baseReceipt
		r.ParsedRecords = 92
		r.UnknownFamilyRecords = 8 // 92% parsed, 8% unknown

		verdict := r.EvaluateClaimAdmission(CompletenessThresholds{
			MinCompletenessRatio: 0.90,
			MaxMalformedRatio:    0.0,
			MaxUnknownRatio:      floatPtr(0.05), // max 5% unknown allowed, but got 8%
		})
		if verdict.Admitted() {
			t.Fatal("expected denial on unknown family breach")
		}
		if verdict.Verdict != Deny {
			t.Errorf("expected Deny, got %s", verdict.Verdict)
		}
		if verdict.Reason != ReasonUnknownFamilyBreach {
			t.Errorf("expected %s, got %s", ReasonUnknownFamilyBreach, verdict.Reason)
		}
	})

	t.Run("lossy input denies on truncated record breach", func(t *testing.T) {
		r := baseReceipt
		r.ParsedRecords = 95
		r.TruncatedRecords = 5

		verdict := r.EvaluateClaimAdmission(CompletenessThresholds{
			MinCompletenessRatio: 0.90,
			MaxMalformedRatio:    0.0,
			MaxTruncatedRatio:    floatPtr(0.02),
		})
		if verdict.Admitted() {
			t.Fatal("expected denial on truncated record breach")
		}
		if verdict.Reason != ReasonTruncatedRecordBreach {
			t.Errorf("expected %s, got %s", ReasonTruncatedRecordBreach, verdict.Reason)
		}
	})

	t.Run("lossy input denies on duplicate record breach", func(t *testing.T) {
		r := baseReceipt
		r.ParsedRecords = 95
		r.DuplicateRecords = 5

		verdict := r.EvaluateClaimAdmission(CompletenessThresholds{
			MinCompletenessRatio: 0.90,
			MaxMalformedRatio:    0.0,
			MaxDuplicateRatio:    floatPtr(0.01),
		})
		if verdict.Admitted() {
			t.Fatal("expected denial on duplicate record breach")
		}
		if verdict.Reason != ReasonDuplicateRecordBreach {
			t.Errorf("expected %s, got %s", ReasonDuplicateRecordBreach, verdict.Reason)
		}
	})
}

func TestParserCompleteness_ZeroDenominatorAndEmptyInput(t *testing.T) {
	emptyReceipt := ParserCompletenessReceipt{
		Schema: ParserCompletenessSchema,
		Source: SourceIdentity{
			Path:   "empty_bench.jsonl",
			SHA256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		},
		ParserVersion: "1.0.0",
		ParserSchema:  "jsonl/v1",
		TotalBytes:    0,
		TotalRecords:  0,
		ParsedRecords: 0,
	}

	// 1. Validate passes on 0 == 0 conservation
	if err := emptyReceipt.Validate(); err != nil {
		t.Fatalf("expected empty receipt (0==0) to pass Validate(), got: %v", err)
	}

	// 2. Ratios on zero records must be safe (0.0, not NaN or Inf)
	if r := emptyReceipt.CompletenessRatio(); r != 0.0 {
		t.Errorf("expected 0.0 completeness ratio on zero records, got %f", r)
	}
	if r := emptyReceipt.MalformedRatio(); r != 0.0 {
		t.Errorf("expected 0.0 malformed ratio on zero records, got %f", r)
	}
	if r := emptyReceipt.UnknownFamilyRatio(); r != 0.0 {
		t.Errorf("expected 0.0 unknown family ratio on zero records, got %f", r)
	}
	if r := emptyReceipt.TruncatedRatio(); r != 0.0 {
		t.Errorf("expected 0.0 truncated ratio on zero records, got %f", r)
	}
	if r := emptyReceipt.DuplicateRatio(); r != 0.0 {
		t.Errorf("expected 0.0 duplicate ratio on zero records, got %f", r)
	}

	// 3. Admission defaults to Deny on empty input
	defaultVerdict := emptyReceipt.EvaluateClaimAdmission(CompletenessThresholds{
		MinCompletenessRatio: 1.0,
		MaxMalformedRatio:    0.0,
	})
	if defaultVerdict.Admitted() {
		t.Error("expected default thresholds to deny empty input")
	}
	if defaultVerdict.Reason != ReasonEmptyInput {
		t.Errorf("expected %s for empty input, got %s", ReasonEmptyInput, defaultVerdict.Reason)
	}

	// 4. When AllowEmpty is true and MinCompletenessRatio == 0, admit
	permissiveVerdict := emptyReceipt.EvaluateClaimAdmission(CompletenessThresholds{
		MinCompletenessRatio: 0.0,
		MaxMalformedRatio:    0.0,
		AllowEmpty:           true,
	})
	if !permissiveVerdict.Admitted() {
		t.Errorf("expected permissive empty input to be admitted, got: %+v", permissiveVerdict)
	}

	// 5. When AllowEmpty is true but MinCompletenessRatio > 0, deny on completeness deficit
	deficitVerdict := emptyReceipt.EvaluateClaimAdmission(CompletenessThresholds{
		MinCompletenessRatio: 0.95,
		MaxMalformedRatio:    0.0,
		AllowEmpty:           true,
	})
	if deficitVerdict.Admitted() {
		t.Error("expected empty input with min completeness ratio 0.95 to deny")
	}
	if deficitVerdict.Reason != ReasonParserCompletenessDeficit {
		t.Errorf("expected %s, got %s", ReasonParserCompletenessDeficit, deficitVerdict.Reason)
	}
}

func TestParserCompleteness_ValidationRules(t *testing.T) {
	valid := ParserCompletenessReceipt{
		Schema: ParserCompletenessSchema,
		Source: SourceIdentity{
			Path:   "artifact.jsonl",
			SHA256: "deadbeef",
		},
		ParserVersion: "1.0.0",
		ParserSchema:  "jsonl/v1",
		TotalRecords:  10,
		ParsedRecords: 10,
	}

	t.Run("invalid schema", func(t *testing.T) {
		r := valid
		r.Schema = "fak.wrong-schema/v1"
		err := r.Validate()
		if !errors.Is(err, ErrInvalidSchema) {
			t.Errorf("expected ErrInvalidSchema, got: %v", err)
		}
	})

	t.Run("missing source identity", func(t *testing.T) {
		r := valid
		r.Source = SourceIdentity{SHA256: ""}
		err := r.Validate()
		if !errors.Is(err, ErrMissingSource) {
			t.Errorf("expected ErrMissingSource, got: %v", err)
		}

		r.Source = SourceIdentity{SHA256: "abc", Path: ""}
		err = r.Validate()
		if !errors.Is(err, ErrMissingSource) {
			t.Errorf("expected ErrMissingSource, got: %v", err)
		}
	})

	t.Run("negative total bytes", func(t *testing.T) {
		r := valid
		r.TotalBytes = -1
		err := r.Validate()
		if !errors.Is(err, ErrNegativeCount) {
			t.Errorf("expected ErrNegativeCount, got: %v", err)
		}
	})

	t.Run("negative record count", func(t *testing.T) {
		r := valid
		r.ParsedRecords = -5
		err := r.Validate()
		if !errors.Is(err, ErrNegativeCount) {
			t.Errorf("expected ErrNegativeCount, got: %v", err)
		}
	})

	t.Run("negative line or offset in sample", func(t *testing.T) {
		r := valid
		r.ParsedRecords = 9
		r.MalformedRecords = 1
		r.MalformedSamples = []RecordSample{
			{Line: -1, Offset: 10, Snippet: "err"},
		}
		if err := r.Validate(); err == nil {
			t.Error("expected validation failure for negative line")
		}
	})

	t.Run("samples present when category count is zero", func(t *testing.T) {
		r := valid
		r.UnknownFamilyRecords = 0
		r.UnknownSamples = []RecordSample{
			{Line: 1, Offset: 0, Snippet: "foo"},
		}
		if err := r.Validate(); err == nil {
			t.Error("expected validation failure for samples present when count is 0")
		}
	})
}

func TestParserCompleteness_SampleBoundingAndScrubbing(t *testing.T) {
	raw := "line 1\r\nline 2\nline 3\twith tabs"
	scrubbed := ScrubSnippet(raw, 50)
	if strings.Contains(scrubbed, "\n") || strings.Contains(scrubbed, "\r") || strings.Contains(scrubbed, "\t") {
		t.Errorf("scrubbed snippet contains raw control characters: %q", scrubbed)
	}

	longRaw := strings.Repeat("A", 300)
	shortScrubbed := ScrubSnippet(longRaw, 100)
	if len(shortScrubbed) > 103 { // 100 + "..."
		t.Errorf("scrubbed snippet exceeded max length: %d", len(shortScrubbed))
	}
	if !strings.HasSuffix(shortScrubbed, "...") {
		t.Errorf("expected snippet to end with '...', got: %q", shortScrubbed)
	}

	r := ParserCompletenessReceipt{
		Schema: ParserCompletenessSchema,
		Source: SourceIdentity{Path: "f.jsonl", SHA256: "abc"},
	}

	// Add more samples than max allowed (max = 2)
	r.AddUnknownSample(RecordSample{Line: 1, Snippet: "s1"}, 2)
	r.AddUnknownSample(RecordSample{Line: 2, Snippet: "s2"}, 2)
	r.AddUnknownSample(RecordSample{Line: 3, Snippet: "s3"}, 2)

	if len(r.UnknownSamples) != 2 {
		t.Errorf("expected exactly 2 samples due to bounding, got %d", len(r.UnknownSamples))
	}
}
