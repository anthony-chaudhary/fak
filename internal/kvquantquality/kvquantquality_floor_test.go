package kvquantquality

import (
	"encoding/json"
	"math"
	"testing"
)

// upstreamFloorFixtures are the three floors the pinned upstream study measured
// on a large MoE checkpoint: JigSawPT/deepseek-v41-flash-on-5090 (MIT),
// tools/kv_floor.py:63-81@21d29fdb275191bbd5d288c3729324c7b16d2599. They are
// quoted priors, not appliance measurements: every use keeps the
// upstream_quoted_unverified label visible.
var upstreamFloorFixtures = []struct {
	name          string
	scheme        FloorScheme
	relativeError float64
}{
	{"fp8 block-32 window KV", FloorScheme{Precision: "fp8", BlockSize: 32, ScaleFormat: "ue8m0", Role: FloorRoleWindowKV}, 0.0280},
	{"fp4 block-16 compressed latent", FloorScheme{Precision: "fp4", BlockSize: 16, ScaleFormat: "e4m3", Role: FloorRoleLatent}, 0.1039},
	{"fp4 block-32 indexer q/k", FloorScheme{Precision: "fp4", BlockSize: 32, Role: FloorRoleIndexer}, 0.1564},
}

func newUpstreamReceipt(t *testing.T, fx struct {
	name          string
	scheme        FloorScheme
	relativeError float64
}) FloorReceipt {
	t.Helper()
	receipt, err := NewFloorReceipt(fx.scheme, RoundTripStats{RelativeError: fx.relativeError}, FloorEvidenceUpstreamQuoted,
		"JigSawPT/deepseek-v41-flash-on-5090 tools/kv_floor.py:63-81@21d29fdb275191bbd5d288c3729324c7b16d2599", "MIT",
		"clean-room method extraction; upstream code neither vendored nor executed")
	if err != nil {
		t.Fatalf("NewFloorReceipt(%s): %v", fx.name, err)
	}
	return receipt
}

// TestNewFloorReceiptStampsTheVersionedContract pins the receipt shape for the
// three upstream-measured schemes: the floor contract version, the kept-visible
// upstream-quoted label, and the scheme/floor values themselves.
func TestNewFloorReceiptStampsTheVersionedContract(t *testing.T) {
	for _, fx := range upstreamFloorFixtures {
		t.Run(fx.name, func(t *testing.T) {
			receipt := newUpstreamReceipt(t, fx)
			if receipt.Contract != FloorContractVersion {
				t.Fatalf("receipt contract = %q, want %q", receipt.Contract, FloorContractVersion)
			}
			if receipt.Evidence != FloorEvidenceUpstreamQuoted {
				t.Fatalf("fixture evidence = %q, want the upstream-quoted label kept visible", receipt.Evidence)
			}
			if receipt.Scheme != fx.scheme || receipt.Stats.RelativeError != fx.relativeError {
				t.Fatalf("receipt = %+v, want scheme %+v floor %v", receipt, fx.scheme, fx.relativeError)
			}
		})
	}
}

// TestAdjudicateAgainstFloorSeparatesSchemeErrorFromDefect is the load-bearing
// witness: drift at or below the floor is the scheme paying its own round-trip
// cost (not a defect); drift above it is a regression signal the budget
// evaluation should see.
func TestAdjudicateAgainstFloorSeparatesSchemeErrorFromDefect(t *testing.T) {
	for _, fx := range upstreamFloorFixtures {
		t.Run(fx.name, func(t *testing.T) {
			receipt := newUpstreamReceipt(t, fx)

			at, err := AdjudicateAgainstFloor(receipt, fx.relativeError)
			if err != nil {
				t.Fatalf("at-floor observation refused: %v", err)
			}
			if at != FloorWithinFloor {
				t.Fatalf("observed == floor judged %q, want %q", at, FloorWithinFloor)
			}

			below, err := AdjudicateAgainstFloor(receipt, fx.relativeError/2)
			if err != nil {
				t.Fatalf("below-floor observation refused: %v", err)
			}
			if below != FloorWithinFloor {
				t.Fatalf("observed < floor judged %q, want %q", below, FloorWithinFloor)
			}

			above, err := AdjudicateAgainstFloor(receipt, fx.relativeError+0.001)
			if err != nil {
				t.Fatalf("above-floor observation refused: %v", err)
			}
			if above != FloorAboveFloor {
				t.Fatalf("observed > floor judged %q, want %q", above, FloorAboveFloor)
			}
		})
	}
}

// TestNewFloorReceiptFailsClosedOnMalformedInput pins the fail-closed edge of
// the constructor: unknown vocabulary, non-positive geometry, and non-finite
// statistics refuse rather than guess.
func TestNewFloorReceiptFailsClosedOnMalformedInput(t *testing.T) {
	base := RoundTripStats{RelativeError: 0.01}
	inf := math.Inf(1)
	cases := []struct {
		name   string
		scheme FloorScheme
		stats  RoundTripStats
		ev     FloorEvidenceKind
	}{
		{"blank precision", FloorScheme{BlockSize: 32, Role: FloorRoleWindowKV}, base, FloorEvidenceUpstreamQuoted},
		{"zero block size", FloorScheme{Precision: "fp8", BlockSize: 0, Role: FloorRoleWindowKV}, base, FloorEvidenceUpstreamQuoted},
		{"unknown role", FloorScheme{Precision: "fp8", BlockSize: 32, Role: "shared_expert"}, base, FloorEvidenceUpstreamQuoted},
		{"unclassified evidence", FloorScheme{Precision: "fp8", BlockSize: 32, Role: FloorRoleWindowKV}, base, "measured"},
		{"negative relative error", FloorScheme{Precision: "fp8", BlockSize: 32, Role: FloorRoleWindowKV}, RoundTripStats{RelativeError: -0.01}, FloorEvidenceUpstreamQuoted},
		{"NaN relative error", FloorScheme{Precision: "fp8", BlockSize: 32, Role: FloorRoleWindowKV}, RoundTripStats{RelativeError: math.NaN()}, FloorEvidenceUpstreamQuoted},
		{"non-finite optional correlation", FloorScheme{Precision: "fp8", BlockSize: 32, Role: FloorRoleWindowKV}, RoundTripStats{RelativeError: 0.01, Correlation: &inf}, FloorEvidenceUpstreamQuoted},
	}
	for _, tc := range cases {
		if _, err := NewFloorReceipt(tc.scheme, tc.stats, tc.ev, "", "", ""); err == nil {
			t.Errorf("%s was admitted; malformed floor input must fail closed", tc.name)
		}
	}
	// A lossless scheme owes nothing: a zero floor is a valid receipt, and any
	// positive observation then exceeds it.
	lossless, err := NewFloorReceipt(FloorScheme{Precision: "fp16", BlockSize: 1, Role: FloorRoleWindowKV},
		RoundTripStats{RelativeError: 0}, FloorEvidenceObservedHardware, "", "", "")
	if err != nil {
		t.Fatalf("zero floor refused: %v", err)
	}
	if verdict, err := AdjudicateAgainstFloor(lossless, 0.0001); err != nil || verdict != FloorAboveFloor {
		t.Fatalf("positive observation against a zero floor = (%q, %v), want (%q, nil)", verdict, err, FloorAboveFloor)
	}
}

// TestAdjudicateAgainstFloorRefusesUnusableComparisons pins the comparison's
// own fail-closed edges: the floor contract stamp is enforced, and non-finite
// or negative observations never reach a verdict.
func TestAdjudicateAgainstFloorRefusesUnusableComparisons(t *testing.T) {
	receipt, err := NewFloorReceipt(FloorScheme{Precision: "fp8", BlockSize: 32, Role: FloorRoleWindowKV},
		RoundTripStats{RelativeError: 0.028}, FloorEvidenceUpstreamQuoted, "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	stale := receipt
	stale.Contract = ContractVersion // the budget contract, not the floor contract
	if _, err := AdjudicateAgainstFloor(stale, 0.01); err == nil {
		t.Error("a budget-contract receipt was compared as a floor; the contract stamp must be enforced")
	}
	for _, obs := range []float64{math.NaN(), math.Inf(1), -0.001} {
		if _, err := AdjudicateAgainstFloor(receipt, obs); err == nil {
			t.Errorf("observed relative error %v was admitted; non-finite or negative observations must fail closed", obs)
		}
	}
}

// TestFloorReceiptSurvivesJSONRoundTrip proves the receipt is a durable
// artifact: marshaling and parsing back preserves every field, and an omitted
// optional axis comes back absent (nil), not fabricated as zero.
func TestFloorReceiptSurvivesJSONRoundTrip(t *testing.T) {
	corr := 0.997
	maxDev := 0.0042
	in, err := NewFloorReceipt(FloorScheme{Precision: "fp4", BlockSize: 16, ScaleFormat: "e4m3", Role: FloorRoleLatent},
		RoundTripStats{RelativeError: 0.1039, Correlation: &corr, MaxAbsDeviation: &maxDev},
		FloorEvidenceUpstreamQuoted, "pin", "MIT", "clean-room")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out FloorReceipt
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out.Contract != in.Contract || out.Scheme != in.Scheme || out.Evidence != in.Evidence ||
		out.SourcePin != in.SourcePin || out.License != in.License || out.Provenance != in.Provenance ||
		out.Stats.RelativeError != in.Stats.RelativeError {
		t.Fatalf("round trip changed the receipt:\n in=%+v\nout=%+v", in, out)
	}
	if out.Stats.Correlation == nil || *out.Stats.Correlation != *in.Stats.Correlation ||
		out.Stats.MaxAbsDeviation == nil || *out.Stats.MaxAbsDeviation != *in.Stats.MaxAbsDeviation {
		t.Fatalf("round trip lost the optional axes:\n in=%+v\nout=%+v", in.Stats, out.Stats)
	}

	minimal, err := NewFloorReceipt(FloorScheme{Precision: "fp8", BlockSize: 32, Role: FloorRoleWindowKV},
		RoundTripStats{RelativeError: 0.028}, FloorEvidenceUpstreamQuoted, "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	rawMin, err := json.Marshal(minimal)
	if err != nil {
		t.Fatalf("marshal minimal: %v", err)
	}
	var outMin FloorReceipt
	if err := json.Unmarshal(rawMin, &outMin); err != nil {
		t.Fatalf("unmarshal minimal: %v", err)
	}
	if outMin.Stats.Correlation != nil || outMin.Stats.MaxAbsDeviation != nil {
		t.Fatalf("omitted optional axes came back non-nil: %+v", outMin.Stats)
	}
}
