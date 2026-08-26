package ultracodetokenizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"testing"
)

func fixture(t *testing.T) (CanonicalInput, []Measurement) {
	t.Helper()
	var in CanonicalInput
	b, err := os.ReadFile("testdata/canonical.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &in); err != nil {
		t.Fatal(err)
	}
	var measurements []Measurement
	b, err = os.ReadFile("testdata/measurements.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &measurements); err != nil {
		t.Fatal(err)
	}
	for i := range measurements {
		measurements[i].CanonicalDigest = Digest(in)
	}
	return in, measurements
}

func TestEvaluateSeparatesTokenizerEffectsFromSemanticOmission(t *testing.T) {
	in, measurements := fixture(t)
	report, err := Evaluate(in, measurements)
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != Schema || len(report.Results) != 3 {
		t.Fatalf("unexpected report: %+v", report)
	}
	for _, result := range report.Results {
		if result.Provenance.OmittedBytes != len(in.FullMessages)-len(in.ScopedMessages) || result.Provenance.OmittedMessages != 2 {
			t.Fatalf("semantic omission changed for %s: %+v", result.Tokenizer.Family, result.Provenance)
		}
		if result.Provenance.ModelInputTokens == 0 || result.Provenance.RuntimePrefixReadTokens != 30 {
			t.Fatalf("token provenance collapsed for %s: %+v", result.Tokenizer.Family, result.Provenance)
		}
	}
	want := (54.0 / 84.0) - (49.0 / 79.0)
	if diff := report.TokenizerOnlyScopeShareMovement - want; diff < -1e-12 || diff > 1e-12 {
		t.Fatalf("movement=%v want %v", report.TokenizerOnlyScopeShareMovement, want)
	}
	if report.PromotionEvidence == "" || report.DemotionEvidence == "" || report.InvalidatingAssumption == "" {
		t.Fatal("generation evidence is incomplete")
	}
}

func TestEvaluateAbstainsOnDifferentCanonicalInputOrOutcome(t *testing.T) {
	in, measurements := fixture(t)
	measurements[1].CanonicalDigest = "sha256:different"
	if _, err := Evaluate(in, measurements); !errors.Is(err, ErrAbstain) {
		t.Fatalf("digest error=%v", err)
	}
	_, measurements = fixture(t)
	measurements[1].AcceptedOutcome = "different"
	if _, err := Evaluate(in, measurements); !errors.Is(err, ErrAbstain) {
		t.Fatalf("outcome error=%v", err)
	}
}

func TestEvaluateRequiresThreeDistinctFamilies(t *testing.T) {
	in, measurements := fixture(t)
	if _, err := Evaluate(in, measurements[:2]); !errors.Is(err, ErrAbstain) {
		t.Fatalf("two-family error=%v", err)
	}
	measurements[2].Tokenizer.Family = measurements[1].Tokenizer.Family
	if _, err := Evaluate(in, measurements); !errors.Is(err, ErrAbstain) {
		t.Fatalf("duplicate-family error=%v", err)
	}
}

func TestNormalizedReportFixture(t *testing.T) {
	in, measurements := fixture(t)
	report, err := Evaluate(in, measurements)
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, '\n')
	want, err := os.ReadFile("testdata/report.json")
	if err != nil {
		t.Fatal(err)
	}
	got = bytes.ReplaceAll(got, []byte("\r\n"), []byte("\n"))
	want = bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n"))
	if !bytes.Equal(bytes.TrimSpace(got), bytes.TrimSpace(want)) {
		t.Fatalf("normalized report drifted\ngot:\n%s\nwant:\n%s", got, want)
	}
}
