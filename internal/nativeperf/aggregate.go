package nativeperf

import (
	"fmt"
	"math"
	"sort"
)

// DistributionSummary reports an explicit numerator and denominator for one
// repetition population. Dispersion is population standard deviation.
type DistributionSummary struct {
	Included int     `json:"included"`
	Total    int     `json:"total"`
	Mean     float64 `json:"mean_tokens_per_second"`
	Median   float64 `json:"median_tokens_per_second"`
	StdDev   float64 `json:"stddev_tokens_per_second"`
}

type RepetitionExclusion struct {
	AttestationDigest string `json:"attestation_digest"`
	Reason            string `json:"reason"`
}

type RepetitionSummaries struct {
	AllSample  DistributionSummary   `json:"all_sample"`
	CleanOnly  DistributionSummary   `json:"clean_only"`
	Policy     string                `json:"inclusion_policy"`
	Exclusions []RepetitionExclusion `json:"exclusions,omitempty"`
}

// SummarizeRepetitions reports both the complete distribution and the subset
// whose aligned ambient evidence is explicitly clean. Missing, investigate,
// invalid, and malformed evidence is excluded with one bounded digest/reason.
func SummarizeRepetitions(reps []Repetition, evidence []AmbientEvidence) RepetitionSummaries {
	all := make([]float64, len(reps))
	clean := make([]float64, 0, len(reps))
	excluded := make([]RepetitionExclusion, 0)
	for i, rep := range reps {
		all[i] = rep.TokensPerSecond
		if i >= len(evidence) {
			excluded = append(excluded, RepetitionExclusion{Reason: "missing"})
			continue
		}
		e := evidence[i]
		reason := string(e.Verdict)
		if err := ValidateAmbientEvidence(e); err != nil {
			reason = "invalid"
		} else if e.Verdict == AmbientClean {
			clean = append(clean, rep.TokensPerSecond)
			continue
		}
		excluded = append(excluded, RepetitionExclusion{AttestationDigest: e.Digest, Reason: reason})
	}
	return RepetitionSummaries{
		AllSample: distribution(all, len(reps)), CleanOnly: distribution(clean, len(reps)),
		Policy:     "clean_only requires validated ambient verdict=clean",
		Exclusions: excluded,
	}
}

func distribution(values []float64, total int) DistributionSummary {
	d := DistributionSummary{Included: len(values), Total: total}
	if len(values) == 0 {
		return d
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	for _, v := range sorted {
		d.Mean += v
	}
	d.Mean /= float64(len(sorted))
	if len(sorted)%2 == 1 {
		d.Median = sorted[len(sorted)/2]
	} else {
		d.Median = (sorted[len(sorted)/2-1] + sorted[len(sorted)/2]) / 2
	}
	for _, v := range sorted {
		delta := v - d.Mean
		d.StdDev += delta * delta
	}
	d.StdDev = math.Sqrt(d.StdDev / float64(len(sorted)))
	return d
}

func requireCleanSamples(name string, s RepetitionSummaries, minimum int) error {
	if minimum > 0 && s.CleanOnly.Included < minimum {
		return fmt.Errorf("%s has %d clean repetitions; policy requires %d; rerun with clean ambient attestations", name, s.CleanOnly.Included, minimum)
	}
	return nil
}

func cleanRepetitions(reps []Repetition, evidence []AmbientEvidence) []Repetition {
	clean := make([]Repetition, 0, len(reps))
	for i, rep := range reps {
		if i < len(evidence) && ValidateAmbientEvidence(evidence[i]) == nil && evidence[i].Verdict == AmbientClean {
			clean = append(clean, rep)
		}
	}
	return clean
}
