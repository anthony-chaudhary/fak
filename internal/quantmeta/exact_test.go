package quantmeta

import (
	"errors"
	"math"
	"testing"
)

// exact_test.go is the #14494 witness for the exact-quantization admission
// gate: a value that its target format would alter is refused, never silently
// rounded, and the check stays a pure function of (value, format) -- no format
// is ranked.

// TestExactRoundTrip proves the round-trip property: RoundTripExact is true iff
// encoding then decoding under the named format is bit-identical; it is false
// (never a silent pass) when storage would alter the value; and it refuses with
// an error on non-finite inputs and unknown or uncheckable formats.
func TestExactRoundTrip(t *testing.T) {
	cases := []struct {
		name    string
		value   float32
		format  Format
		want    bool
		wantErr error
	}{
		// fp8-e4m3: exact values survive.
		{"fp8 zero", 0.0, FormatFP8E4M3, true, nil},
		{"fp8 one", 1.0, FormatFP8E4M3, true, nil},
		{"fp8 max finite", 448.0, FormatFP8E4M3, true, nil},
		{"fp8 smallest normal", 0.015625, FormatFP8E4M3, true, nil},       // 2^-6
		{"fp8 smallest subnormal", 0.001953125, FormatFP8E4M3, true, nil}, // 2^-9
		{"fp8 negative", -1.0, FormatFP8E4M3, true, nil},
		// fp8-e4m3: altered values are refused (false, no error).
		{"fp8 out of range", 449.0, FormatFP8E4M3, false, nil},
		{"fp8 not representable", 0.1, FormatFP8E4M3, false, nil},
		{"fp8 between codes", 1.1, FormatFP8E4M3, false, nil},
		// fp4-e2m1: exact values survive.
		{"fp4 zero", 0.0, FormatFP4E2M1, true, nil},
		{"fp4 half", 0.5, FormatFP4E2M1, true, nil},
		{"fp4 one", 1.0, FormatFP4E2M1, true, nil},
		{"fp4 one half", 1.5, FormatFP4E2M1, true, nil},
		{"fp4 two", 2.0, FormatFP4E2M1, true, nil},
		{"fp4 three", 3.0, FormatFP4E2M1, true, nil},
		{"fp4 four", 4.0, FormatFP4E2M1, true, nil},
		{"fp4 six", 6.0, FormatFP4E2M1, true, nil},
		{"fp4 negative six", -6.0, FormatFP4E2M1, true, nil},
		// fp4-e2m1: altered values are refused.
		{"fp4 not in set", 0.75, FormatFP4E2M1, false, nil},
		{"fp4 above max", 7.0, FormatFP4E2M1, false, nil},
		{"fp4 between", 1.25, FormatFP4E2M1, false, nil},
		// mxfp4: exact values survive.
		{"mxfp4 zero", 0.0, FormatMXFP4, true, nil},
		{"mxfp4 one", 1.0, FormatMXFP4, true, nil},
		{"mxfp4 one half", 1.5, FormatMXFP4, true, nil},
		{"mxfp4 three", 3.0, FormatMXFP4, true, nil},
		{"mxfp4 six", 6.0, FormatMXFP4, true, nil},
		{"mxfp4 scaled", 0.75, FormatMXFP4, true, nil}, // 1.5 * 2^-1
		{"mxfp4 power of two", 64.0, FormatMXFP4, true, nil},
		{"mxfp4 one point five times power", 96.0, FormatMXFP4, true, nil}, // 1.5 * 2^6
		{"mxfp4 negative", -3.0, FormatMXFP4, true, nil},
		// mxfp4: altered values are refused.
		{"mxfp4 not representable", 0.3, FormatMXFP4, false, nil},
		{"mxfp4 hundred", 100.0, FormatMXFP4, false, nil},
		{"mxfp4 between", 1.25, FormatMXFP4, false, nil},
		// Non-finite inputs refuse with an error, not a false pass.
		{"nan", float32(math.NaN()), FormatFP8E4M3, false, ErrNonFiniteValue},
		{"+inf", float32(math.Inf(1)), FormatFP4E2M1, false, ErrNonFiniteValue},
		{"-inf", float32(math.Inf(-1)), FormatMXFP4, false, ErrNonFiniteValue},
		// Unknown formats refuse with an error, not a false pass.
		{"unknown format", 1.0, Format("some-2027-format"), false, ErrFormatNotCheckable},
		// A known format with no primitive refuses fail-closed.
		{"known no primitive", 1.0, FormatF16, false, ErrFormatNotCheckable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RoundTripExact(tc.value, tc.format)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want errors.Is %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected err: %v", err)
			}
			if got != tc.want {
				t.Errorf("RoundTripExact(%v, %q) = %v, want %v", tc.value, tc.format, got, tc.want)
			}
		})
	}
}

// TestExactAdmit proves the admission gate: SUPPORT only when every value
// round-trips exactly, REFUSE with ReasonExactnessViolation when any value
// would be altered, and fail-closed errors for uncheckable input.
func TestExactAdmit(t *testing.T) {
	t.Run("all exact supports", func(t *testing.T) {
		got, err := AdmitExact([]float32{0.0, 1.0, 1.5, -6.0}, FormatFP4E2M1)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if got.Outcome != OutcomeSupported {
			t.Errorf("Outcome = %q (reasons %v), want %q", got.Outcome, got.Reasons, OutcomeSupported)
		}
	})

	t.Run("one altered refuses", func(t *testing.T) {
		got, err := AdmitExact([]float32{1.0, 0.75, 2.0}, FormatFP4E2M1)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if got.Outcome != OutcomeRefuse {
			t.Errorf("Outcome = %q (reasons %v), want %q", got.Outcome, got.Reasons, OutcomeRefuse)
		}
		if !hasReason(got, ReasonExactnessViolation) {
			t.Errorf("Reasons = %v, want to include %q", got.Reasons, ReasonExactnessViolation)
		}
	})

	t.Run("empty set supports vacuously", func(t *testing.T) {
		got, err := AdmitExact(nil, FormatFP8E4M3)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if got.Outcome != OutcomeSupported {
			t.Errorf("Outcome = %q, want %q for an empty set", got.Outcome, OutcomeSupported)
		}
	})

	t.Run("unknown format errors", func(t *testing.T) {
		_, err := AdmitExact([]float32{1.0}, Format("some-2027-format"))
		if !errors.Is(err, ErrFormatNotCheckable) {
			t.Errorf("err = %v, want errors.Is %v", err, ErrFormatNotCheckable)
		}
	})

	t.Run("non-finite value errors", func(t *testing.T) {
		_, err := AdmitExact([]float32{1.0, float32(math.Inf(1))}, FormatFP8E4M3)
		if !errors.Is(err, ErrNonFiniteValue) {
			t.Errorf("err = %v, want errors.Is %v", err, ErrNonFiniteValue)
		}
	})
}

// TestExactReasonKnown proves ReasonExactnessViolation is a registered public
// reason code, so an Adjudicate-style caller can check Reason.Known() rather
// than inferring the refusal from prose.
func TestExactReasonKnown(t *testing.T) {
	if !ReasonExactnessViolation.Known() {
		t.Errorf("ReasonExactnessViolation.Known() = false, want true")
	}
	if ReasonExactnessViolation == ReasonFullyDeclared {
		t.Error("ReasonExactnessViolation must be distinct from the support reason")
	}
}
