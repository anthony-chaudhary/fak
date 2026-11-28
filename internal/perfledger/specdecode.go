// specdecode.go
//
// NOTICE / attribution:
//
// This file is a direct port of the speculative-decoding acceptance metrics
// from ai-dynamo/aiperf, pinned at ai-dynamo/aiperf@ab610add4e8876fca223197596c728e17365f381,
// files src/aiperf/metrics/types/spec_decode_metrics.py and
// src/aiperf/common/models/spec_decode_models.py
// (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES).
// Licensed under the Apache License, Version 2.0 (Apache-2.0).
// SPDX-License-Identifier: Apache-2.0
//
// The formulas are mirrored faithfully:
//   - mean acceptance length = 1 + num_accepted_draft_tokens / num_spec_steps
//     (the "j + 1" acceptance length; undefined when num_spec_steps == 0);
//   - draft acceptance rate = num_accepted_draft_tokens / num_draft_tokens as a
//     0..1 fraction, reported as a percentage (*100.0); undefined when
//     num_draft_tokens == 0;
//   - accepted-per-verified = (num_accepted + num_steps) / (num_draft +
//     num_steps) = (j + 1) / (l + 1); undefined when the denominator is 0;
//   - token-weighted acceptance length (run-level) = 1 + Sum(accepted) /
//     Sum(steps); undefined when Sum(steps) == 0;
//   - overall draft acceptance rate (run-level) = Sum(accepted) / Sum(draft) *
//     100.0; undefined when Sum(draft) == 0.
//
// The source raises NoMetricValue for undefined cases; this port returns the
// sentinel ErrNoMetricValue instead.
package perfledger

import (
	"encoding/json"
	"errors"
)

// ErrNoMetricValue mirrors aiperf's NoMetricValue: the metric is undefined for
// the given stats (zero verify steps, zero proposed draft tokens, or a zero
// accepted-per-verified denominator).
var ErrNoMetricValue = errors.New("no metric value")

// SpecDecodeStats is one request's speculative-decoding step stats, mirroring
// the engine-neutral SpecDecodeAcceptanceRecord fields the metrics read:
// accepted draft tokens, proposed draft tokens, and verify steps.
type SpecDecodeStats struct {
	NumAcceptedDraftTokens int
	NumDraftTokens         int
	NumSpecSteps           int
}

// MeanAcceptanceLength returns the per-request mean acceptance length: tokens
// emitted per verify step including the always-accepted bonus token
// (1 + accepted / steps, the "j + 1" acceptance length). It returns
// ErrNoMetricValue when NumSpecSteps == 0.
func MeanAcceptanceLength(s SpecDecodeStats) (float64, error) {
	if s.NumSpecSteps == 0 {
		return 0, ErrNoMetricValue
	}
	return 1.0 + float64(s.NumAcceptedDraftTokens)/float64(s.NumSpecSteps), nil
}

// DraftAcceptanceRate returns the per-request draft acceptance rate as a
// percentage: accepted draft tokens over proposed draft tokens
// (num_accepted_draft_tokens / num_draft_tokens * 100.0). Draft-only; excludes
// the bonus token. It returns ErrNoMetricValue when NumDraftTokens == 0.
func DraftAcceptanceRate(s SpecDecodeStats) (float64, error) {
	if s.NumDraftTokens == 0 {
		return 0, ErrNoMetricValue
	}
	return float64(s.NumAcceptedDraftTokens) / float64(s.NumDraftTokens) * 100.0, nil
}

// AcceptedPerVerified returns the per-request accepted-per-verified ratio:
// emitted tokens (accepted drafts plus one bonus per step) over verified
// tokens (proposed drafts plus one bonus per step) =
// (num_accepted + num_steps) / (num_draft + num_steps) = (j + 1) / (l + 1).
// It returns ErrNoMetricValue when the denominator is 0.
func AcceptedPerVerified(s SpecDecodeStats) (float64, error) {
	denominator := s.NumDraftTokens + s.NumSpecSteps
	if denominator == 0 {
		return 0, ErrNoMetricValue
	}
	return float64(s.NumAcceptedDraftTokens+s.NumSpecSteps) / float64(denominator), nil
}

// EffectiveSpeedup returns the effective speedup of speculative decoding over
// a single-token-per-step baseline: the mean acceptance length, i.e. tokens
// emitted per verify step including the always-accepted bonus token. It
// returns ErrNoMetricValue when NumSpecSteps == 0.
func EffectiveSpeedup(s SpecDecodeStats) (float64, error) {
	return MeanAcceptanceLength(s)
}

// TokenWeightedAcceptanceLength returns the run-level token-weighted mean
// acceptance length: 1 + Sum(accepted) / Sum(steps). It returns
// ErrNoMetricValue when Sum(steps) == 0.
func TokenWeightedAcceptanceLength(stats []SpecDecodeStats) (float64, error) {
	accepted, steps := 0, 0
	for _, s := range stats {
		accepted += s.NumAcceptedDraftTokens
		steps += s.NumSpecSteps
	}
	if steps == 0 {
		return 0, ErrNoMetricValue
	}
	return 1.0 + float64(accepted)/float64(steps), nil
}

// OverallDraftAcceptanceRate returns the run-level (token-weighted) draft
// acceptance rate as a percentage: Sum(accepted) / Sum(draft) * 100.0. It
// returns ErrNoMetricValue when Sum(draft) == 0.
func OverallDraftAcceptanceRate(stats []SpecDecodeStats) (float64, error) {
	accepted, draft := 0, 0
	for _, s := range stats {
		accepted += s.NumAcceptedDraftTokens
		draft += s.NumDraftTokens
	}
	if draft == 0 {
		return 0, ErrNoMetricValue
	}
	return float64(accepted) / float64(draft) * 100.0, nil
}

// specDecodeRawFields holds the JSON fields emitted by vLLM, TRT-LLM, and SGLang.
type specDecodeRawFields struct {
	NumAcceptedDraftTokens *int `json:"num_accepted_draft_tokens"`
	NumDraftTokens         *int `json:"num_draft_tokens"`
	NumSpecSteps           *int `json:"num_spec_steps"`
}

func (f *specDecodeRawFields) toStats() (SpecDecodeStats, bool) {
	if f == nil || (f.NumAcceptedDraftTokens == nil && f.NumDraftTokens == nil && f.NumSpecSteps == nil) {
		return SpecDecodeStats{}, false
	}
	var st SpecDecodeStats
	if f.NumAcceptedDraftTokens != nil {
		st.NumAcceptedDraftTokens = *f.NumAcceptedDraftTokens
	}
	if f.NumDraftTokens != nil {
		st.NumDraftTokens = *f.NumDraftTokens
	}
	if f.NumSpecSteps != nil {
		st.NumSpecSteps = *f.NumSpecSteps
	}
	return st, true
}

// CaptureSpecDecodeStats extracts per-request speculative-decoding stats from
// a raw completion/chat JSON response across the three upstream engine placements
// (aiperf base_endpoint.py):
//   - vLLM root (`spec_decode` or root-level counters)
//   - TRT-LLM per-choice (`choices[0].spec_decode`)
//   - SGLang extension (`sglext.spec_decode` or `sglext`)
//
// When n > 1 (or len(choices) > 1), per-request capture is suppressed and
// (SpecDecodeStats{}, false, nil) is returned.
func CaptureSpecDecodeStats(raw []byte, n int) (SpecDecodeStats, bool, error) {
	if n > 1 {
		return SpecDecodeStats{}, false, nil
	}
	if len(raw) == 0 {
		return SpecDecodeStats{}, false, nil
	}
	var payload struct {
		specDecodeRawFields
		SpecDecode *specDecodeRawFields `json:"spec_decode"`
		SglExt     *struct {
			specDecodeRawFields
			SpecDecode *specDecodeRawFields `json:"spec_decode"`
		} `json:"sglext"`
		Choices []struct {
			SpecDecode *specDecodeRawFields `json:"spec_decode"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return SpecDecodeStats{}, false, err
	}
	if len(payload.Choices) > 1 {
		return SpecDecodeStats{}, false, nil
	}
	// 1. vLLM root placement (nested spec_decode or direct root fields)
	if st, ok := payload.SpecDecode.toStats(); ok {
		return st, true, nil
	}
	if st, ok := payload.specDecodeRawFields.toStats(); ok {
		return st, true, nil
	}
	// 2. TRT-LLM per-choice placement
	if len(payload.Choices) == 1 {
		if st, ok := payload.Choices[0].SpecDecode.toStats(); ok {
			return st, true, nil
		}
	}
	// 3. SGLang sglext placement
	if payload.SglExt != nil {
		if st, ok := payload.SglExt.SpecDecode.toStats(); ok {
			return st, true, nil
		}
		if st, ok := payload.SglExt.specDecodeRawFields.toStats(); ok {
			return st, true, nil
		}
	}
	return SpecDecodeStats{}, false, nil
}
