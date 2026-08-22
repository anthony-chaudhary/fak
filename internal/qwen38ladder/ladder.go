// Package qwen38ladder defines the evidence-gated path from fast Qwen3.5
// experiments to an exact Qwen3.8-27B confirmation.
package qwen38ladder

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
)

const Schema = "fak.qwen38-ladder-evidence/1"

type Stage struct {
	ID              string   `json:"id"`
	Model           string   `json:"model"`
	Revision        string   `json:"revision"`
	ParametersB     float64  `json:"parameters_b"`
	MinimumPassRate float64  `json:"minimum_pass_rate"`
	Proves          []string `json:"proves"`
	DoesNotProve    []string `json:"does_not_prove,omitempty"`
}

// Stages is deliberately ordered by experiment cost. Revisions are immutable
// upstream Hugging Face commits observed on 2026-08-22.
var Stages = []Stage{
	{ID: "smoke", Model: "Qwen/Qwen3.5-0.8B", Revision: "2fc06364715b967f1860aea9cf38778875588b17", ParametersB: 0.8, MinimumPassRate: 1, Proves: []string{"loader", "tokenizer", "chat-template", "request-shape", "kernel-shape-genericity"}, DoesNotProve: []string{"quality", "27B-memory", "Qwen3.8-weights"}},
	{ID: "behavior", Model: "Qwen/Qwen3.5-2B", Revision: "15852e8c16360a2fea060d615a32b45270f8a8fc", ParametersB: 2, MinimumPassRate: .90, Proves: []string{"fast behavior iteration", "tool/json scoring"}, DoesNotProve: []string{"target quality", "27B-memory", "Qwen3.8-weights"}},
	{ID: "width", Model: "Qwen/Qwen3.5-4B", Revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", ParametersB: 4, MinimumPassRate: .90, Proves: []string{"wider-tensor kernels", "behavior trend"}, DoesNotProve: []string{"target quality", "27B-memory", "Qwen3.8-weights"}},
	{ID: "quality-proxy", Model: "Qwen/Qwen3.5-9B", Revision: "c202236235762e1c871ad0ccb60c8ee5ba337b9a", ParametersB: 9, MinimumPassRate: .90, Proves: []string{"medium-scale quality signal", "untied-embedding path"}, DoesNotProve: []string{"target quality", "27B-memory", "Qwen3.8-weights"}},
	{ID: "scale-rehearsal", Model: "Qwen/Qwen3.5-27B", Revision: "fc05daec18b0a78c049392ed2e771dde82bdf654", ParametersB: 27, MinimumPassRate: .95, Proves: []string{"exact 27B tensor geometry", "64-layer memory envelope", "24-head/4-kv-head kernels"}, DoesNotProve: []string{"Qwen3.8 weight quality", "Qwen3.8 exact identity"}},
	{ID: "target", Model: "Qwen/Qwen3.8-27B", Revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", ParametersB: 27, MinimumPassRate: .95, Proves: []string{"exact target behavior", "exact target performance", "release decision"}},
}

type Result struct {
	StageID         string  `json:"stage_id"`
	Model           string  `json:"model"`
	Revision        string  `json:"revision"`
	EnvironmentSHA  string  `json:"environment_sha256"`
	Trials          int     `json:"trials"`
	BaselinePassed  int     `json:"baseline_passed"`
	CandidatePassed int     `json:"candidate_passed"`
	BaselineMetric  float64 `json:"baseline_metric"`
	CandidateMetric float64 `json:"candidate_metric"`
}

type Evidence struct {
	Schema                string   `json:"schema"`
	Concept               string   `json:"concept"`
	CorpusSHA             string   `json:"corpus_sha256"`
	BaselineRuntimeSHA    string   `json:"baseline_runtime_sha"`
	CandidateRuntimeSHA   string   `json:"candidate_runtime_sha"`
	Metric                string   `json:"metric"`
	Direction             string   `json:"direction"`
	MinimumImprovementPct float64  `json:"minimum_improvement_pct"`
	Results               []Result `json:"results"`
}

type Decision struct {
	Verdict        string  `json:"verdict"`
	NextStage      *Stage  `json:"next_stage,omitempty"`
	ImprovementPct float64 `json:"improvement_pct,omitempty"`
	Reason         string  `json:"reason"`
}

func Decode(r io.Reader) (Evidence, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var evidence Evidence
	if err := dec.Decode(&evidence); err != nil {
		return Evidence{}, err
	}
	if evidence.Schema != Schema {
		return Evidence{}, fmt.Errorf("schema: got %q, want %q", evidence.Schema, Schema)
	}
	if evidence.Concept == "" || evidence.Metric == "" {
		return Evidence{}, errors.New("concept and metric are required")
	}
	if evidence.CorpusSHA == "" || evidence.BaselineRuntimeSHA == "" || evidence.CandidateRuntimeSHA == "" {
		return Evidence{}, errors.New("corpus and baseline/candidate runtime hashes are required")
	}
	if evidence.BaselineRuntimeSHA == evidence.CandidateRuntimeSHA {
		return Evidence{}, errors.New("baseline and candidate runtime hashes must differ")
	}
	if evidence.Direction != "lower" && evidence.Direction != "higher" {
		return Evidence{}, errors.New(`direction must be "lower" or "higher"`)
	}
	if evidence.MinimumImprovementPct < 0 || math.IsNaN(evidence.MinimumImprovementPct) || math.IsInf(evidence.MinimumImprovementPct, 0) {
		return Evidence{}, errors.New("minimum_improvement_pct must be finite and non-negative")
	}
	return evidence, nil
}

func Evaluate(e Evidence) (Decision, error) {
	if len(e.Results) > len(Stages) {
		return Decision{}, errors.New("more results than ladder stages")
	}
	for i, result := range e.Results {
		stage := Stages[i]
		if result.StageID != stage.ID || result.Model != stage.Model || result.Revision != stage.Revision {
			return Decision{}, fmt.Errorf("result %d identity mismatch: got %s %s@%s, want %s %s@%s", i, result.StageID, result.Model, result.Revision, stage.ID, stage.Model, stage.Revision)
		}
		if result.EnvironmentSHA == "" {
			return Decision{}, fmt.Errorf("result %d must bind environment_sha256", i)
		}
		if result.Trials <= 0 || result.BaselinePassed < 0 || result.BaselinePassed > result.Trials || result.CandidatePassed < 0 || result.CandidatePassed > result.Trials {
			return Decision{}, fmt.Errorf("result %d has invalid trial counts", i)
		}
		baselineRate := float64(result.BaselinePassed) / float64(result.Trials)
		if baselineRate < stage.MinimumPassRate {
			return Decision{Verdict: "HOLD", NextStage: &stage, Reason: fmt.Sprintf("%s baseline pass rate %.3f is below %.3f; repair the experiment before attributing a candidate gain", stage.ID, baselineRate, stage.MinimumPassRate)}, nil
		}
		candidateRate := float64(result.CandidatePassed) / float64(result.Trials)
		if candidateRate < stage.MinimumPassRate {
			return Decision{Verdict: "HOLD", NextStage: &stage, Reason: fmt.Sprintf("%s candidate pass rate %.3f is below %.3f; fix at the cheapest reproducing stage", stage.ID, candidateRate, stage.MinimumPassRate)}, nil
		}
		if result.CandidatePassed < result.BaselinePassed {
			return Decision{Verdict: "HOLD", NextStage: &stage, Reason: fmt.Sprintf("%s candidate regressed correctness from %d/%d to %d/%d", stage.ID, result.BaselinePassed, result.Trials, result.CandidatePassed, result.Trials)}, nil
		}
		improvement, err := improvementPercent(e.Direction, result.BaselineMetric, result.CandidateMetric)
		if err != nil {
			return Decision{}, fmt.Errorf("result %d: %w", i, err)
		}
		if improvement < e.MinimumImprovementPct {
			return Decision{Verdict: "HOLD", NextStage: &stage, ImprovementPct: improvement, Reason: fmt.Sprintf("%s %s improvement %.3f%% is below %.3f%%", stage.ID, e.Metric, improvement, e.MinimumImprovementPct)}, nil
		}
	}
	if len(e.Results) == len(Stages) {
		last := e.Results[len(e.Results)-1]
		improvement, _ := improvementPercent(e.Direction, last.BaselineMetric, last.CandidateMetric)
		return Decision{Verdict: "PASS", ImprovementPct: improvement, Reason: "exact Qwen3.8-27B target passed the paired baseline/candidate experiment"}, nil
	}
	next := Stages[len(e.Results)]
	return Decision{Verdict: "PROMOTE", NextStage: &next, Reason: "all cheaper stages passed; run the paired experiment at the next stage"}, nil
}

func improvementPercent(direction string, baseline, candidate float64) (float64, error) {
	if baseline <= 0 || candidate < 0 || math.IsNaN(baseline) || math.IsNaN(candidate) || math.IsInf(baseline, 0) || math.IsInf(candidate, 0) {
		return 0, errors.New("baseline_metric must be positive and candidate_metric must be finite and non-negative")
	}
	if direction == "lower" {
		return (baseline - candidate) / baseline * 100, nil
	}
	return (candidate - baseline) / baseline * 100, nil
}

func ValidateDefinition() error {
	if len(Stages) < 2 || Stages[len(Stages)-1].Model != "Qwen/Qwen3.8-27B" {
		return errors.New("ladder must terminate at exact Qwen3.8-27B")
	}
	ids := make([]string, 0, len(Stages))
	for _, stage := range Stages {
		if stage.ID == "" || stage.Model == "" || len(stage.Revision) != 40 || stage.MinimumPassRate <= 0 || stage.MinimumPassRate > 1 {
			return fmt.Errorf("invalid stage %q", stage.ID)
		}
		ids = append(ids, stage.ID)
	}
	sorted := slices.Clone(ids)
	slices.Sort(sorted)
	sorted = slices.Compact(sorted)
	if len(sorted) != len(ids) {
		return errors.New("duplicate stage ID")
	}
	return nil
}
