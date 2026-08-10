package quantprov

import (
	"os"
	"testing"
)

var support = Support{
	Quantizers: map[string][]string{"example-quantizer": {"1.2.0"}},
	Formats:    []string{"example-format"},
}

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestGoldenProvenanceConfirmed(t *testing.T) {
	got, err := ParseAndVerify(fixture(t, "confirmed.json"), support)
	if err != nil {
		t.Fatal(err)
	}
	if got.Confidence != ConfidenceConfirmed || got.Reason != ReasonConfirmed {
		t.Fatalf("got %s/%s (%s), want confirmed", got.Confidence, got.Reason, got.Detail)
	}
	if got.Record == nil || got.Record.License != "Apache-2.0" || got.Record.CalibrationIdentity == "" {
		t.Fatalf("provenance record not preserved: %#v", got.Record)
	}
}

func TestMissingFieldIsExplicitlyIncomplete(t *testing.T) {
	got, err := ParseAndVerify(fixture(t, "missing-calibration.json"), support)
	if err != nil {
		t.Fatal(err)
	}
	if got.Confidence != ConfidenceIncomplete || got.Reason != ReasonMissingField || got.Detail != "calibration_identity" {
		t.Fatalf("got %#v, want missing calibration identity", got)
	}
}

func TestTamperedConversionChainIsExplicit(t *testing.T) {
	got, err := ParseAndVerify(fixture(t, "tampered.json"), support)
	if err != nil {
		t.Fatal(err)
	}
	if got.Confidence != ConfidenceTampered || got.Reason != ReasonBrokenChain {
		t.Fatalf("got %#v, want broken-chain tamper result", got)
	}
}

func TestUnsupportedInputsNeverFallBack(t *testing.T) {
	base, err := ParseAndVerify(fixture(t, "confirmed.json"), support)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name   string
		mutate func(*Record)
		want   ReasonCode
	}{
		{"schema", func(r *Record) { r.Schema = "quantprov/v99" }, ReasonUnknownSchema},
		{"quantizer", func(r *Record) { r.Quantizer = "unknown" }, ReasonUnsupportedQuantizer},
		{"version", func(r *Record) { r.QuantizerVersion = "9" }, ReasonUnsupportedVersion},
		{"format", func(r *Record) { r.Format = "unknown" }, ReasonUnsupportedFormat},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			record := *base.Record
			tc.mutate(&record)
			got := Verify(record, support)
			if got.Confidence != ConfidenceUnsupported || got.Reason != tc.want {
				t.Fatalf("got %s/%s, want unsupported/%s", got.Confidence, got.Reason, tc.want)
			}
		})
	}
}

func TestMalformedJSONReturnsTypedInvalidResult(t *testing.T) {
	got, err := ParseAndVerify([]byte(`{"schema":`), support)
	if err == nil {
		t.Fatal("expected parse error")
	}
	if got.Confidence != ConfidenceInvalid || got.Reason != ReasonInvalidJSON {
		t.Fatalf("got %#v", got)
	}
}
