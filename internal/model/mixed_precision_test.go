package model

import (
	"math"
	"testing"
)

func TestMixedPrecisionPhysicalByteReversal(t *testing.T) {
	selection, err := SelectMixedPrecision([]MixedPrecisionObservation{
		{Name: "int2-codebook", NominalBits: 2, Sensitivity: 0.02, AttemptedTokens: 100, PhysicalBytes: 1800, UnpackBytes: 800, DecodeNanoseconds: 1000, EnergyJoules: 2, Occupancy: 0.9},
		{Name: "int4-grouped", NominalBits: 4, Sensitivity: 0.02, AttemptedTokens: 100, PhysicalBytes: 2200, UnpackBytes: 100, DecodeNanoseconds: 1000, EnergyJoules: 2, Occupancy: 0.9},
	}, 0.03, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	if got := selection.Selected.Observation.Name; got != "int4-grouped" {
		t.Fatalf("selected %q; nominally smaller format must lose when metadata/unpack bytes cost more", got)
	}
	if got := selection.Selected.Cost.BytesPerAccepted; got != 23 {
		t.Fatalf("bytes per accepted token = %v, want 23", got)
	}
}

func TestMixedPrecisionAcceptedTokenReversal(t *testing.T) {
	selection, err := SelectMixedPrecision([]MixedPrecisionObservation{
		{Name: "fragile-int2", NominalBits: 2, Sensitivity: 0.02, AttemptedTokens: 100, RejectedTokens: 60, PhysicalBytes: 1200, DecodeNanoseconds: 800, EnergyJoules: 1.2, Occupancy: 0.9},
		{Name: "stable-int4", NominalBits: 4, Sensitivity: 0.02, AttemptedTokens: 100, RejectedTokens: 0, PhysicalBytes: 2000, DecodeNanoseconds: 1200, EnergyJoules: 2, Occupancy: 0.9},
	}, 0.03, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	if got := selection.Selected.Observation.Name; got != "stable-int4" {
		t.Fatalf("selected %q; rejected output must count against accepted-token cost", got)
	}
	if got := selection.Candidates[1].Cost.AcceptedTokens; got != 40 {
		t.Fatalf("accepted tokens = %d, want 40", got)
	}
}

func TestMixedPrecisionSensitivityAndOccupancyReject(t *testing.T) {
	selection, err := SelectMixedPrecision([]MixedPrecisionObservation{
		{Name: "sensitive", NominalBits: 2, Sensitivity: 0.20, AttemptedTokens: 10, PhysicalBytes: 10, Occupancy: 1},
		{Name: "underfilled", NominalBits: 3, Sensitivity: 0.01, AttemptedTokens: 10, PhysicalBytes: 20, Occupancy: 0.2},
		{Name: "admitted", NominalBits: 4, Sensitivity: 0.01, AttemptedTokens: 10, PhysicalBytes: 30, Occupancy: 0.9},
	}, 0.05, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	if selection.Selected.Observation.Name != "admitted" {
		t.Fatalf("selected %q, want admitted", selection.Selected.Observation.Name)
	}
	if selection.Candidates[1].Reason == "" || selection.Candidates[2].Reason == "" {
		t.Fatalf("rejected candidates lack reasons: %+v", selection.Candidates)
	}
}

func TestMixedPrecisionAccountingPerAcceptedToken(t *testing.T) {
	selection, err := SelectMixedPrecision([]MixedPrecisionObservation{{
		Name: "measured", NominalBits: 4, Sensitivity: 0.01,
		AttemptedTokens: 12, RejectedTokens: 2,
		PhysicalBytes: 80, UnpackBytes: 20,
		SetupNanoseconds: 40, DecodeNanoseconds: 60,
		EnergyJoules: 5, Occupancy: 0.75,
	}}, 0.02, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	cost := selection.Selected.Cost
	if cost.AcceptedTokens != 10 || cost.NetBytes != 100 || cost.NetNanoseconds != 100 || cost.NetJoules != 5 {
		t.Fatalf("unexpected net accounting: %+v", cost)
	}
	if cost.BytesPerAccepted != 10 || cost.NanosecondsPerAccept != 10 || cost.JoulesPerAccepted != 0.5 {
		t.Fatalf("unexpected accepted-token accounting: %+v", cost)
	}
}

func TestMixedPrecisionDeterministicTie(t *testing.T) {
	observations := []MixedPrecisionObservation{
		{Name: "z-format", NominalBits: 2, Sensitivity: 0.01, AttemptedTokens: 10, PhysicalBytes: 100, DecodeNanoseconds: 100, EnergyJoules: 1, Occupancy: 1},
		{Name: "a-format", NominalBits: 8, Sensitivity: 0.01, AttemptedTokens: 10, PhysicalBytes: 100, DecodeNanoseconds: 100, EnergyJoules: 1, Occupancy: 1},
	}
	selection, err := SelectMixedPrecision(observations, 0.02, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if got := selection.Selected.Observation.Name; got != "a-format" {
		t.Fatalf("selected %q, want stable name tie-break a-format", got)
	}
}

func TestMixedPrecisionInvalidInputs(t *testing.T) {
	valid := MixedPrecisionObservation{Name: "valid", NominalBits: 4, Sensitivity: 0.01, AttemptedTokens: 10, PhysicalBytes: 10, Occupancy: 1}
	tests := []struct {
		name string
		obs  []MixedPrecisionObservation
		max  float64
		occ  float64
	}{
		{name: "empty"},
		{name: "no accepted tokens", obs: []MixedPrecisionObservation{{Name: "bad", NominalBits: 2, AttemptedTokens: 2, RejectedTokens: 2, Occupancy: 1}}, max: 1},
		{name: "nan sensitivity", obs: []MixedPrecisionObservation{{Name: "bad", NominalBits: 2, Sensitivity: math.NaN(), AttemptedTokens: 1, Occupancy: 1}}, max: 1},
		{name: "infinite energy", obs: []MixedPrecisionObservation{{Name: "bad", NominalBits: 2, AttemptedTokens: 1, EnergyJoules: math.Inf(1), Occupancy: 1}}, max: 1},
		{name: "bad envelope", obs: []MixedPrecisionObservation{valid}, max: 1, occ: 1.1},
		{name: "duplicate", obs: []MixedPrecisionObservation{valid, valid}, max: 1},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := SelectMixedPrecision(test.obs, test.max, test.occ); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}
