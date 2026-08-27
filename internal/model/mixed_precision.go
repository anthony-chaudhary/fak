package model

import (
	"fmt"
	"math"
	"sort"
)

// MixedPrecisionObservation is one measured fak-native format at a fixed
// workload envelope. PhysicalBytes must include weights, metadata, codebooks,
// residuals, padding, and any other resident representation bytes. UnpackBytes
// accounts for bytes materialized or moved while decoding the observation.
type MixedPrecisionObservation struct {
	Name              string  `json:"name"`
	NominalBits       int     `json:"nominal_bits"`
	Sensitivity       float64 `json:"sensitivity"`
	AttemptedTokens   int64   `json:"attempted_tokens"`
	RejectedTokens    int64   `json:"rejected_tokens"`
	PhysicalBytes     int64   `json:"physical_bytes"`
	UnpackBytes       int64   `json:"unpack_bytes"`
	SetupNanoseconds  int64   `json:"setup_nanoseconds"`
	DecodeNanoseconds int64   `json:"decode_nanoseconds"`
	EnergyJoules      float64 `json:"energy_joules"`
	Occupancy         float64 `json:"occupancy"`
}

// MixedPrecisionCost is measured net resource cost divided by accepted tokens.
type MixedPrecisionCost struct {
	AcceptedTokens       int64   `json:"accepted_tokens"`
	NetBytes             int64   `json:"net_bytes"`
	NetNanoseconds       int64   `json:"net_nanoseconds"`
	NetJoules            float64 `json:"net_joules"`
	BytesPerAccepted     float64 `json:"bytes_per_accepted_token"`
	NanosecondsPerAccept float64 `json:"nanoseconds_per_accepted_token"`
	JoulesPerAccepted    float64 `json:"joules_per_accepted_token"`
}

// MixedPrecisionCandidate records either comparable accounting or the reason
// an observation could not enter selection.
type MixedPrecisionCandidate struct {
	Observation MixedPrecisionObservation `json:"observation"`
	Cost        MixedPrecisionCost        `json:"cost"`
	Feasible    bool                      `json:"feasible"`
	Reason      string                    `json:"reason,omitempty"`
}

// MixedPrecisionSelection is an auditable selection receipt. Ranking is by
// physical bytes per accepted token, then time, energy, sensitivity, name, and
// nominal bits. Nominal width is deliberately only the final tie breaker.
type MixedPrecisionSelection struct {
	Schema           string                    `json:"schema"`
	Engine           string                    `json:"engine"`
	MaxSensitivity   float64                   `json:"max_sensitivity"`
	MinimumOccupancy float64                   `json:"minimum_occupancy"`
	Selected         MixedPrecisionCandidate   `json:"selected"`
	Candidates       []MixedPrecisionCandidate `json:"candidates"`
}

// SelectMixedPrecision selects the lowest measured physical-byte cost that
// satisfies the supplied sensitivity and occupancy envelope.
func SelectMixedPrecision(observations []MixedPrecisionObservation, maxSensitivity, minimumOccupancy float64) (MixedPrecisionSelection, error) {
	if len(observations) == 0 || !mixedPrecisionFinite(maxSensitivity) || maxSensitivity < 0 ||
		!mixedPrecisionFinite(minimumOccupancy) || minimumOccupancy < 0 || minimumOccupancy > 1 {
		return MixedPrecisionSelection{}, fmt.Errorf("model: invalid mixed-precision envelope")
	}

	receipt := MixedPrecisionSelection{
		Schema: "fak-mixed-precision-selection/1", Engine: "fak-native",
		MaxSensitivity: maxSensitivity, MinimumOccupancy: minimumOccupancy,
		Candidates: make([]MixedPrecisionCandidate, 0, len(observations)),
	}
	seen := make(map[string]struct{}, len(observations))
	for _, observation := range observations {
		if err := validateMixedPrecisionObservation(observation); err != nil {
			return MixedPrecisionSelection{}, err
		}
		if _, exists := seen[observation.Name]; exists {
			return MixedPrecisionSelection{}, fmt.Errorf("model: duplicate mixed-precision candidate %q", observation.Name)
		}
		seen[observation.Name] = struct{}{}

		candidate := MixedPrecisionCandidate{Observation: observation}
		candidate.Cost = mixedPrecisionCost(observation)
		switch {
		case observation.Sensitivity > maxSensitivity:
			candidate.Reason = "sensitivity exceeds envelope"
		case observation.Occupancy < minimumOccupancy:
			candidate.Reason = "occupancy below envelope"
		default:
			candidate.Feasible = true
		}
		receipt.Candidates = append(receipt.Candidates, candidate)
	}

	sort.Slice(receipt.Candidates, func(i, j int) bool {
		return mixedPrecisionLess(receipt.Candidates[i], receipt.Candidates[j])
	})
	for _, candidate := range receipt.Candidates {
		if candidate.Feasible {
			receipt.Selected = candidate
			return receipt, nil
		}
	}
	return MixedPrecisionSelection{}, fmt.Errorf("model: no mixed-precision candidate satisfies envelope")
}

func validateMixedPrecisionObservation(observation MixedPrecisionObservation) error {
	if observation.Name == "" || observation.NominalBits <= 0 || observation.AttemptedTokens <= 0 ||
		observation.RejectedTokens < 0 || observation.RejectedTokens >= observation.AttemptedTokens ||
		observation.PhysicalBytes < 0 || observation.UnpackBytes < 0 || observation.SetupNanoseconds < 0 ||
		observation.DecodeNanoseconds < 0 || !mixedPrecisionFinite(observation.Sensitivity) || observation.Sensitivity < 0 ||
		!mixedPrecisionFinite(observation.EnergyJoules) || observation.EnergyJoules < 0 ||
		!mixedPrecisionFinite(observation.Occupancy) || observation.Occupancy < 0 || observation.Occupancy > 1 {
		return fmt.Errorf("model: invalid mixed-precision candidate %q", observation.Name)
	}
	if observation.PhysicalBytes > math.MaxInt64-observation.UnpackBytes ||
		observation.SetupNanoseconds > math.MaxInt64-observation.DecodeNanoseconds {
		return fmt.Errorf("model: mixed-precision candidate %q overflows accounting", observation.Name)
	}
	return nil
}

func mixedPrecisionCost(observation MixedPrecisionObservation) MixedPrecisionCost {
	accepted := observation.AttemptedTokens - observation.RejectedTokens
	netBytes := observation.PhysicalBytes + observation.UnpackBytes
	netNanoseconds := observation.SetupNanoseconds + observation.DecodeNanoseconds
	denominator := float64(accepted)
	return MixedPrecisionCost{
		AcceptedTokens: accepted, NetBytes: netBytes, NetNanoseconds: netNanoseconds, NetJoules: observation.EnergyJoules,
		BytesPerAccepted:     float64(netBytes) / denominator,
		NanosecondsPerAccept: float64(netNanoseconds) / denominator,
		JoulesPerAccepted:    observation.EnergyJoules / denominator,
	}
}

func mixedPrecisionLess(a, b MixedPrecisionCandidate) bool {
	if a.Feasible != b.Feasible {
		return a.Feasible
	}
	if a.Cost.BytesPerAccepted != b.Cost.BytesPerAccepted {
		return a.Cost.BytesPerAccepted < b.Cost.BytesPerAccepted
	}
	if a.Cost.NanosecondsPerAccept != b.Cost.NanosecondsPerAccept {
		return a.Cost.NanosecondsPerAccept < b.Cost.NanosecondsPerAccept
	}
	if a.Cost.JoulesPerAccepted != b.Cost.JoulesPerAccepted {
		return a.Cost.JoulesPerAccepted < b.Cost.JoulesPerAccepted
	}
	if a.Observation.Sensitivity != b.Observation.Sensitivity {
		return a.Observation.Sensitivity < b.Observation.Sensitivity
	}
	if a.Observation.Name != b.Observation.Name {
		return a.Observation.Name < b.Observation.Name
	}
	return a.Observation.NominalBits < b.Observation.NominalBits
}

func mixedPrecisionFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
