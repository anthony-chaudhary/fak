package kvquantmeta

import "testing"

//enumlint:exempt FP16 and BF16 are unquantized controls, deliberately outside this quantized-support fixture.
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
