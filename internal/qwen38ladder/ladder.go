// Package qwen38ladder defines the evidence-gated path from fast Qwen3.5
// experiments to an exact Qwen3.8-27B confirmation.
package qwen38ladder

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
)

const Schema = "fak.qwen38-ladder-evidence/1"

const (
	ReasonMissingArm       = "MISSING_ARM"
	ReasonIdentityMismatch = "IDENTITY_MISMATCH"
	ReasonSamples          = "INSUFFICIENT_SAMPLES"
	ReasonInvalidBaseline  = "INVALID_BASELINE"
	ReasonQuality          = "QUALITY_REGRESSION"
	ReasonConfidence       = "CONFIDENCE_NOT_MET"
	ReasonProxyClaim       = "PROXY_ONLY_TARGET_CLAIM"
)

type Stage struct {
	ID              string   `json:"id"`
	Model           string   `json:"model"`
	Revision        string   `json:"revision"`
	ParametersB     float64  `json:"parameters_b"`
	MinimumPassRate float64  `json:"minimum_pass_rate"`
	Proves          []string `json:"proves"`
	DoesNotProve    []string `json:"does_not_prove,omitempty"`
}

// Stages is deliberately ordered by experiment cost. Revisions are immutable upstream commits.
var Stages = []Stage{
	{ID: "smoke", Model: "Qwen/Qwen3.5-0.8B", Revision: "2fc06364715b967f1860aea9cf38778875588b17", ParametersB: .8, MinimumPassRate: 1, Proves: []string{"loader", "tokenizer", "request-shape"}, DoesNotProve: []string{"quality", "Qwen3.8-weights"}},
	{ID: "behavior", Model: "Qwen/Qwen3.5-2B", Revision: "15852e8c16360a2fea060d615a32b45270f8a8fc", ParametersB: 2, MinimumPassRate: .9, Proves: []string{"fast behavior iteration", "tool/json scoring"}, DoesNotProve: []string{"target quality", "Qwen3.8-weights"}},
	{ID: "width", Model: "Qwen/Qwen3.5-4B", Revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", ParametersB: 4, MinimumPassRate: .9, Proves: []string{"wider-tensor kernels", "behavior trend"}, DoesNotProve: []string{"target quality", "Qwen3.8-weights"}},
	{ID: "quality-proxy", Model: "Qwen/Qwen3.5-9B", Revision: "c202236235762e1c871ad0ccb60c8ee5ba337b9a", ParametersB: 9, MinimumPassRate: .9, Proves: []string{"medium-scale quality signal", "untied-embedding path"}, DoesNotProve: []string{"target quality", "Qwen3.8-weights"}},
	{ID: "scale-rehearsal", Model: "Qwen/Qwen3.5-27B", Revision: "fc05daec18b0a78c049392ed2e771dde82bdf654", ParametersB: 27, MinimumPassRate: .95, Proves: []string{"exact 27B tensor geometry", "64-layer memory envelope"}, DoesNotProve: []string{"Qwen3.8 weight quality", "Qwen3.8 exact identity"}},
	{ID: "target", Model: "Qwen/Qwen3.8-27B", Revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", ParametersB: 27, MinimumPassRate: .95, Proves: []string{"exact target behavior", "exact target performance", "release decision"}},
}

type Corpus struct {
	Name            string  `json:"name"`
	TaskFamily      string  `json:"task_family"`
	SHA256          string  `json:"sha256"`
	MinimumPassRate float64 `json:"minimum_pass_rate"`
}
type ConfidenceRule struct {
	Method               string  `json:"method"`
	MinimumPairedSamples int     `json:"minimum_paired_samples"`
	MinimumWinRate       float64 `json:"minimum_win_rate"`
}

type Result struct {
	StageID                 string    `json:"stage_id"`
	Model                   string    `json:"model"`
	Revision                string    `json:"revision"`
	EnvironmentSHA          string    `json:"environment_sha256"`
	Trials                  int       `json:"trials"`
	BaselinePassed          int       `json:"baseline_passed"`
	CandidatePassed         int       `json:"candidate_passed"`
	BaselineMetric          float64   `json:"baseline_metric"`
	CandidateMetric         float64   `json:"candidate_metric"`
	Corpus                  string    `json:"corpus,omitempty"`
	TaskFamily              string    `json:"task_family,omitempty"`
	CorpusSHA               string    `json:"corpus_sha256,omitempty"`
	BaselineRuntimeSHA      string    `json:"baseline_runtime_sha,omitempty"`
	CandidateRuntimeSHA     string    `json:"candidate_runtime_sha,omitempty"`
	BaselineEnvironmentSHA  string    `json:"baseline_environment_sha256,omitempty"`
	CandidateEnvironmentSHA string    `json:"candidate_environment_sha256,omitempty"`
	BaselineSamples         []float64 `json:"baseline_samples,omitempty"`
	CandidateSamples        []float64 `json:"candidate_samples,omitempty"`
}

type Evidence struct {
	Schema                string          `json:"schema"`
	Concept               string          `json:"concept"`
	CorpusSHA             string          `json:"corpus_sha256"`
	BaselineRuntimeSHA    string          `json:"baseline_runtime_sha"`
	CandidateRuntimeSHA   string          `json:"candidate_runtime_sha"`
	Metric                string          `json:"metric"`
	Direction             string          `json:"direction"`
	MinimumImprovementPct float64         `json:"minimum_improvement_pct"`
	Results               []Result        `json:"results"`
	Corpora               []Corpus        `json:"corpora,omitempty"`
	Confidence            *ConfidenceRule `json:"confidence,omitempty"`
	PromotionClaim        string          `json:"promotion_claim,omitempty"`
}

type Decision struct {
	Verdict        string  `json:"verdict"`
	NextStage      *Stage  `json:"next_stage,omitempty"`
	ImprovementPct float64 `json:"improvement_pct,omitempty"`
	ReasonCode     string  `json:"reason_code,omitempty"`
	Reason         string  `json:"reason"`
}

func Decode(r io.Reader) (Evidence, error) {
	var e Evidence
	d := json.NewDecoder(r)
	d.DisallowUnknownFields()
	if err := d.Decode(&e); err != nil {
		return e, fmt.Errorf("decode evidence: %w", err)
	}
	if e.Schema != Schema {
		return e, fmt.Errorf("schema must be %q", Schema)
	}
	if e.Concept == "" || e.BaselineRuntimeSHA == "" || e.CandidateRuntimeSHA == "" || e.Metric == "" {
		return e, errors.New("concept, baseline_runtime_sha, candidate_runtime_sha, and metric are required")
	}
	if e.BaselineRuntimeSHA == e.CandidateRuntimeSHA {
		return e, errors.New("baseline_runtime_sha and candidate_runtime_sha must differ")
	}
	if e.Direction != "lower" && e.Direction != "higher" {
		return e, errors.New("direction must be lower or higher")
	}
	if len(e.Corpora) == 0 && e.CorpusSHA == "" {
		return e, errors.New("corpus_sha256 is required for legacy evidence")
	}
	if err := validateLegacyResultCorpusIdentities(e); err != nil {
		return e, err
	}
	if len(e.Corpora) > 0 {
		if e.Confidence == nil {
			return e, errors.New("confidence is required with corpora")
		}
		if e.Confidence.Method != "paired-win-rate" || e.Confidence.MinimumPairedSamples < 1 || e.Confidence.MinimumWinRate <= 0 || e.Confidence.MinimumWinRate > 1 {
			return e, errors.New("confidence must use paired-win-rate with valid minimums")
		}
		seen := map[string]bool{}
		for _, c := range e.Corpora {
			if c.Name == "" || c.TaskFamily == "" || c.SHA256 == "" || c.MinimumPassRate <= 0 || c.MinimumPassRate > 1 {
				return e, errors.New("each corpus requires name, task_family, sha256, and a valid minimum_pass_rate")
			}
			if seen[c.Name] {
				return e, fmt.Errorf("duplicate corpus %q", c.Name)
			}
			seen[c.Name] = true
		}
	}
	return e, nil
}

func hold(code, reason string, stage Stage) (Decision, error) {
	return Decision{Verdict: "HOLD", NextStage: &stage, ReasonCode: code, Reason: reason}, nil
}

func Evaluate(e Evidence) (Decision, error) {
	if len(e.Corpora) > 0 {
		return evaluateMulti(e)
	}
	if err := validateLegacyResultCorpusIdentities(e); err != nil {
		return Decision{}, err
	}
	if len(e.Results) > len(Stages) {
		return Decision{}, errors.New("more results than ladder stages")
	}
	for i, result := range e.Results {
		stage := Stages[i]
		if result.StageID != stage.ID || result.Model != stage.Model || result.Revision != stage.Revision {
			return Decision{}, fmt.Errorf("result %d identity mismatch", i)
		}
		if result.EnvironmentSHA == "" || result.Trials <= 0 || result.BaselinePassed < 0 || result.BaselinePassed > result.Trials || result.CandidatePassed < 0 || result.CandidatePassed > result.Trials {
			return Decision{}, fmt.Errorf("result %d has invalid trials, pass counts, or environment_sha256", i)
		}
		floor := stage.MinimumPassRate
		bp := float64(result.BaselinePassed) / float64(result.Trials)
		cp := float64(result.CandidatePassed) / float64(result.Trials)
		if bp < floor {
			return Decision{Verdict: "HOLD", NextStage: &stage, Reason: fmt.Sprintf("%s baseline pass rate %.3f is below %.3f", stage.ID, bp, floor)}, nil
		}
		if cp < floor {
			return Decision{Verdict: "HOLD", NextStage: &stage, Reason: fmt.Sprintf("%s candidate pass rate %.3f is below %.3f", stage.ID, cp, floor)}, nil
		}
		imp, err := improvementPercent(e.Direction, result.BaselineMetric, result.CandidateMetric)
		if err != nil {
			return Decision{}, fmt.Errorf("result %d: %w", i, err)
		}
		if imp < e.MinimumImprovementPct {
			return Decision{Verdict: "HOLD", NextStage: &stage, ImprovementPct: imp, Reason: fmt.Sprintf("%s %s improvement %.3f%% is below %.3f%%", stage.ID, e.Metric, imp, e.MinimumImprovementPct)}, nil
		}
	}
	if len(e.Results) == len(Stages) {
		last := e.Results[len(e.Results)-1]
		imp, _ := improvementPercent(e.Direction, last.BaselineMetric, last.CandidateMetric)
		return Decision{Verdict: "PASS", ImprovementPct: imp, Reason: "exact Qwen3.8-27B target passed the paired baseline/candidate experiment"}, nil
	}
	next := Stages[len(e.Results)]
	return Decision{Verdict: "PROMOTE", NextStage: &next, Reason: "all cheaper stages passed; run the paired experiment at the next stage"}, nil
}

func validateLegacyResultCorpusIdentities(e Evidence) error {
	if len(e.Corpora) > 0 || len(e.Results) == 0 {
		return nil
	}
	annotated := 0
	for i, result := range e.Results {
		if result.CorpusSHA == "" {
			continue
		}
		annotated++
		decoded, err := hex.DecodeString(result.CorpusSHA)
		if err != nil || len(decoded) != 32 {
			return fmt.Errorf("result %d corpus_sha256 must be exactly 64 hexadecimal characters", i)
		}
	}
	if annotated != 0 && annotated != len(e.Results) {
		return errors.New("legacy result corpus_sha256 must be present for every result or none")
	}
	return nil
}

func evaluateMulti(e Evidence) (Decision, error) {
	expected := len(Stages) * len(e.Corpora)
	if len(e.Results) > expected {
		return Decision{}, errors.New("more results than stage/corpus pairs")
	}
	for i, r := range e.Results {
		si := i / len(e.Corpora)
		ci := i % len(e.Corpora)
		stage, corpus := Stages[si], e.Corpora[ci]
		if r.StageID != stage.ID || r.Model != stage.Model || r.Revision != stage.Revision || r.Corpus != corpus.Name || r.TaskFamily != corpus.TaskFamily || r.CorpusSHA != corpus.SHA256 || r.BaselineRuntimeSHA != e.BaselineRuntimeSHA || r.CandidateRuntimeSHA != e.CandidateRuntimeSHA || r.BaselineEnvironmentSHA == "" || r.BaselineEnvironmentSHA != r.CandidateEnvironmentSHA {
			return hold(ReasonIdentityMismatch, fmt.Sprintf("%s/%s has mismatched corpus, runtime, revision, or environment identity", stage.ID, corpus.Name), stage)
		}
		if len(r.BaselineSamples) == 0 || len(r.CandidateSamples) == 0 || len(r.BaselineSamples) != len(r.CandidateSamples) {
			return hold(ReasonMissingArm, fmt.Sprintf("%s/%s requires equal baseline and candidate arms", stage.ID, corpus.Name), stage)
		}
		if len(r.BaselineSamples) < e.Confidence.MinimumPairedSamples {
			return hold(ReasonSamples, fmt.Sprintf("%s/%s has %d paired samples; need %d", stage.ID, corpus.Name, len(r.BaselineSamples), e.Confidence.MinimumPairedSamples), stage)
		}
		if r.BaselinePassed < 0 || r.BaselinePassed > len(r.BaselineSamples) || r.CandidatePassed < 0 || r.CandidatePassed > len(r.CandidateSamples) {
			return hold(ReasonMissingArm, fmt.Sprintf("%s/%s has invalid arm correctness counts", stage.ID, corpus.Name), stage)
		}
		floor := math.Max(stage.MinimumPassRate, corpus.MinimumPassRate)
		if float64(r.BaselinePassed)/float64(len(r.BaselineSamples)) < floor {
			return hold(ReasonInvalidBaseline, fmt.Sprintf("%s/%s baseline misses correctness floor %.3f", stage.ID, corpus.Name, floor), stage)
		}
		if float64(r.CandidatePassed)/float64(len(r.CandidateSamples)) < floor {
			return hold(ReasonQuality, fmt.Sprintf("%s/%s candidate misses correctness floor %.3f", stage.ID, corpus.Name, floor), stage)
		}
		wins := 0
		sumB, sumC := 0.0, 0.0
		for j, b := range r.BaselineSamples {
			c := r.CandidateSamples[j]
			if _, err := improvementPercent(e.Direction, b, c); err != nil {
				return hold(ReasonInvalidBaseline, fmt.Sprintf("%s/%s sample %d has invalid baseline", stage.ID, corpus.Name, j), stage)
			}
			imp, _ := improvementPercent(e.Direction, b, c)
			if imp >= e.MinimumImprovementPct {
				wins++
			}
			sumB += b
			sumC += c
		}
		winRate := float64(wins) / float64(len(r.BaselineSamples))
		if winRate < e.Confidence.MinimumWinRate {
			return hold(ReasonConfidence, fmt.Sprintf("%s/%s paired win rate %.3f is below %.3f", stage.ID, corpus.Name, winRate, e.Confidence.MinimumWinRate), stage)
		}
		if _, err := improvementPercent(e.Direction, sumB, sumC); err != nil {
			return hold(ReasonInvalidBaseline, fmt.Sprintf("%s/%s aggregate baseline is invalid", stage.ID, corpus.Name), stage)
		}
	}
	completeStages := len(e.Results) / len(e.Corpora)
	partial := len(e.Results) % len(e.Corpora)
	if partial != 0 {
		stage := Stages[completeStages]
		return hold(ReasonMissingArm, fmt.Sprintf("%s is missing one or more corpus arms", stage.ID), stage)
	}
	if (e.PromotionClaim == "default" || e.PromotionClaim == "release") && completeStages < len(Stages) {
		stage := Stages[completeStages]
		return hold(ReasonProxyClaim, "proxy evidence cannot authorize a default or release; exact target evidence is required", stage)
	}
	if completeStages == len(Stages) {
		return Decision{Verdict: "PASS", Reason: "every task family passed the declared paired-confidence rule on the exact Qwen3.8-27B target"}, nil
	}
	next := Stages[completeStages]
	return Decision{Verdict: "PROMOTE", NextStage: &next, Reason: "every task family passed; run the next paired stage"}, nil
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
	ids := []string{}
	for _, s := range Stages {
		if s.ID == "" || s.Model == "" || len(s.Revision) != 40 || s.MinimumPassRate <= 0 || s.MinimumPassRate > 1 {
			return fmt.Errorf("invalid stage %q", s.ID)
		}
		ids = append(ids, s.ID)
	}
	sorted := slices.Clone(ids)
	slices.Sort(sorted)
	sorted = slices.Compact(sorted)
	if len(sorted) != len(ids) {
		return errors.New("duplicate stage ID")
	}
	return nil
}
