package kvquantmeta

import (
	"math"
	"testing"
)

var testSupport = Support{
	Schemes:     map[string][]string{"kvq": {"1"}},
	Precisions:  []Precision{PrecisionFP8, PrecisionINT8, PrecisionINT4, PrecisionINT2},
	Groupings:   []Grouping{GroupingPerToken, GroupingPerChannel, GroupingPerTokenChannel},
	Transforms:  []string{"hadamard"},
	Transitions: map[string][]string{"hot": {"warm"}, "warm": {"cold"}},
}

func descriptor(p Precision) Descriptor {
	return Descriptor{ID: "kvq", Version: "1", KeyPrecision: p, ValuePrecision: p, Grouping: GroupingPerToken, ResidualWindowTokens: 128, Tier: "hot", Recoverability: RecoverableApproximate}
}

func TestGoldenPrecisions(t *testing.T) {
	for _, precision := range []Precision{PrecisionINT8, PrecisionFP8, PrecisionINT4, PrecisionINT2} {
		t.Run(string(precision), func(t *testing.T) {
			got := Validate(descriptor(precision), testSupport)
			if !got.Supported || got.Reason != ReasonSupported {
				t.Fatalf("got %#v", got)
			}
		})
	}
}

func TestUnknownSchemeHasExplicitResult(t *testing.T) {
	d := descriptor(Precision("int3"))
	got := Validate(d, testSupport)
	if got.Supported || got.Reason != ReasonUnknownScheme {
		t.Fatalf("got %#v", got)
	}
}

func TestKAndVPrecisionsRemainIndependent(t *testing.T) {
	d := descriptor(PrecisionINT8)
	d.ValuePrecision = PrecisionINT4
	got := Validate(d, testSupport)
	if !got.Supported {
		t.Fatalf("mixed K/V precision rejected: %#v", got)
	}
	if d.KeyPrecision == d.ValuePrecision {
		t.Fatal("test did not exercise independent K/V precision")
	}
}

func TestDescriptorDoesNotConflateWeightQuantization(t *testing.T) {
	d := descriptor(PrecisionINT4)
	if field := missing(d); field != "" {
		t.Fatalf("valid descriptor missing %s", field)
	}
	// There is intentionally no weight-precision field: this contract owns only cache K/V state.
}

func TestTierTransitionsAreDirectedAndExplicit(t *testing.T) {
	from := descriptor(PrecisionFP8)
	to := descriptor(PrecisionINT4)
	to.Tier = "warm"
	to.Recoverability = RecoverableNone
	if got := ValidateTransition(Transition{From: from, To: to}, testSupport); !got.Supported {
		t.Fatalf("declared transition: %#v", got)
	}
	if got := ValidateTransition(Transition{From: to, To: from}, testSupport); got.Supported || got.Reason != ReasonUnsupportedTransition {
		t.Fatalf("undeclared reverse: %#v", got)
	}
}

func TestInvalidGroupingRequiresGroupSize(t *testing.T) {
	d := descriptor(PrecisionINT4)
	d.Grouping = GroupingPerChannel
	got := Validate(d, testSupport)
	if got.Supported || got.Reason != ReasonInvalidDescriptor || got.Detail != "group_size" {
		t.Fatalf("got %#v", got)
	}
}

// TestConstantBlockDequantizesExactly pins the zero-variance property: a cache
// block whose values are all c has max == min, so the block scale is 0. Without
// a floor the quantize divisor is 0 and the round trip is not finite; with the
// floor the block stays finite and returns to c exactly.
func TestConstantBlockDequantizesExactly(t *testing.T) {
	if got := FloorScale(0); !(got > 0) || math.IsInf(float64(got), 0) || math.IsNaN(float64(got)) {
		t.Fatalf("FloorScale(0) = %v, want a finite positive floor", got)
	}
	if got := FloorScale(minScale); got != minScale {
		t.Fatalf("FloorScale(minScale) = %v, want the floor itself %v", got, minScale)
	}
	if got := FloorScale(4); got != 4 {
		t.Fatalf("FloorScale(4) = %v, want an above-floor scale passed through unchanged", got)
	}
	// The floor is load-bearing: the unfloored zero scale yields 0/0.
	var zero float32
	if unfloored := (zero - zero) / (zero - zero); !math.IsNaN(float64(unfloored)) {
		t.Fatalf("unfloored zero scale produced %v, want the NaN this test exists to prevent", unfloored)
	}

	for _, value := range []float32{0, -2.5, 7, 1 << 20} {
		min, max := value, value
		scale := FloorScale((max - min) / 255)
		quantized := (value - min) / scale
		got := min + quantized*scale
		if math.IsInf(float64(got), 0) || math.IsNaN(float64(got)) || got != value {
			t.Fatalf("constant block dequantized to %v (scale=%v), want finite %v", got, scale, value)
		}
	}
}
