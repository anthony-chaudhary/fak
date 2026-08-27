package main

import (
	"encoding/json"
	"errors"
	"os"
	"sort"
)

type profileComparison struct {
	Schema               string    `json:"schema"`
	Control              string    `json:"control"`
	Candidate            string    `json:"candidate"`
	MinimumMedianPercent float64   `json:"minimum_median_percent"`
	RequireEveryImproves bool      `json:"require_every_improves"`
	Baseline             []float64 `json:"baseline"`
	Observed             []float64 `json:"observed"`
	Verdict              string    `json:"verdict"`
	MedianPercent        float64   `json:"median_percent"`
	Reason               string    `json:"reason"`
}

func compareProfiles(control, candidate []float64, minPercent float64, requireEvery bool) profileComparison {
	r := profileComparison{Schema: "fak.modelbench.profile-comparison/1", MinimumMedianPercent: minPercent, RequireEveryImproves: requireEvery, Baseline: control, Observed: candidate, Verdict: "REJECT"}
	if len(control) == 0 || len(control) != len(candidate) {
		r.Reason = "unmatched repetitions"
		return r
	}
	for i := range control {
		if control[i] <= 0 || candidate[i] <= 0 {
			r.Reason = "non-positive timing"
			return r
		}
		if requireEvery && candidate[i] >= control[i] {
			r.Reason = "not every repetition improved"
			return r
		}
	}
	cm := median(control)
	xm := median(candidate)
	r.MedianPercent = (cm - xm) / cm * 100
	if r.MedianPercent < minPercent {
		r.Reason = "median improvement below gate"
		return r
	}
	r.Verdict = "KEEP"
	r.Reason = "acceptance gate passed"
	return r
}
func median(xs []float64) float64 {
	v := append([]float64(nil), xs...)
	sort.Float64s(v)
	n := len(v)
	if n%2 == 1 {
		return v[n/2]
	}
	return (v[n/2-1] + v[n/2]) / 2
}
func loadPrefill(path string) (float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var v struct {
		Schema           string  `json:"schema"`
		PromptTokens     int     `json:"prompt_tokens"`
		CompletionTokens int     `json:"completion_tokens"`
		PrefillMS        float64 `json:"prefill_ms"`
		Engine           string  `json:"engine"`
		Fallback         string  `json:"fallback"`
		Artifact         string  `json:"artifact"`
	}
	if json.Unmarshal(raw, &v) != nil || v.Schema != "fak-native-performance-profile/1" || v.PromptTokens != 32 || v.CompletionTokens != 64 || v.PrefillMS <= 0 || v.Engine != "fak-native" || v.Fallback != "none" || v.Artifact == "" {
		return 0, errors.New("invalid exact P32/T64 profile")
	}
	return v.PrefillMS, nil
}
