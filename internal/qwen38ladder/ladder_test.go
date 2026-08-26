package qwen38ladder

import (
	"strings"
	"testing"
)

func TestDefinitionTerminatesAtExactTarget(t *testing.T) {
	if err := ValidateDefinition(); err != nil {
		t.Fatal(err)
	}
	got := Stages[len(Stages)-2]
	if got.Model != "Qwen/Qwen3.5-27B" || !strings.Contains(strings.Join(got.Proves, " "), "exact 27B tensor geometry") {
		t.Fatalf("scale rehearsal = %#v", got)
	}
}

func TestEvaluatePromotesOneStageAtATime(t *testing.T) {
	e := evidence(pass(Stages[0]))
	got, err := Evaluate(e)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "PROMOTE" || got.NextStage == nil || got.NextStage.ID != "behavior" {
		t.Fatalf("decision = %#v", got)
	}
}

func TestEvaluateHoldsWhenBaselineMissesCorrectnessFloor(t *testing.T) {
	e := evidence(pass(Stages[0]))
	e.Results[0].BaselinePassed = 0
	got, err := Evaluate(e)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "HOLD" || got.NextStage == nil || got.NextStage.ID != "smoke" || !strings.Contains(got.Reason, "baseline pass rate") {
		t.Fatalf("decision = %#v", got)
	}
}
func TestEvaluateHoldsAtCheapestCorrectnessFailure(t *testing.T) {
	e := evidence(pass(Stages[0]), pass(Stages[1]))
	e.Results[1].CandidatePassed = 8
	got, err := Evaluate(e)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "HOLD" || got.NextStage == nil || got.NextStage.ID != "behavior" {
		t.Fatalf("decision = %#v", got)
	}
}

func TestEvaluateHoldsWhenCandidateDoesNotBeatBaseline(t *testing.T) {
	e := evidence(pass(Stages[0]))
	e.Results[0].CandidateMetric = 99
	e.MinimumImprovementPct = 5
	got, err := Evaluate(e)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "HOLD" || got.ImprovementPct != 1 {
		t.Fatalf("decision = %#v", got)
	}
}

func TestEvaluateSupportsHigherIsBetterMetrics(t *testing.T) {
	e := evidence(pass(Stages[0]))
	e.Direction = "higher"
	e.Results[0].BaselineMetric = 100
	e.Results[0].CandidateMetric = 110
	e.MinimumImprovementPct = 9
	got, err := Evaluate(e)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "PROMOTE" {
		t.Fatalf("decision = %#v", got)
	}
}

func TestEvaluateRejectsSkippedStageAndInvalidExperiment(t *testing.T) {
	skipped := evidence(pass(Stages[1]))
	if _, err := Evaluate(skipped); err == nil || !strings.Contains(err.Error(), "identity mismatch") {
		t.Fatalf("error = %v", err)
	}

	invalid := evidence(pass(Stages[0]))
	invalid.Results[0].EnvironmentSHA = ""
	if _, err := Evaluate(invalid); err == nil || !strings.Contains(err.Error(), "environment_sha256") {
		t.Fatalf("error = %v", err)
	}
}

func TestEvaluateRequiresExactTargetBeforePass(t *testing.T) {
	e := evidence()
	for _, stage := range Stages {
		e.Results = append(e.Results, pass(stage))
	}
	got, err := Evaluate(e)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "PASS" || got.NextStage != nil || got.ImprovementPct != 10 {
		t.Fatalf("decision = %#v", got)
	}
}

func TestDecodeRequiresPairedImmutableExperiment(t *testing.T) {
	_, err := Decode(strings.NewReader(`{"schema":"fak.qwen38-ladder-evidence/1","concept":"kernel","corpus_sha256":"c","baseline_runtime_sha":"same","candidate_runtime_sha":"same","metric":"p95_ms","direction":"lower","results":[]}`))
	if err == nil || !strings.Contains(err.Error(), "must differ") {
		t.Fatalf("error = %v", err)
	}
}

func evidence(results ...Result) Evidence {
	return Evidence{Schema: Schema, Concept: "fused-gdn", CorpusSHA: "corpus", BaselineRuntimeSHA: "base", CandidateRuntimeSHA: "candidate", Metric: "p95_ms", Direction: "lower", MinimumImprovementPct: 5, Results: results}
}

func pass(stage Stage) Result {
	return Result{StageID: stage.ID, Model: stage.Model, Revision: stage.Revision, EnvironmentSHA: "environment", Trials: 20, BaselinePassed: 20, CandidatePassed: 20, BaselineMetric: 100, CandidateMetric: 90}
}

func TestEvaluateMultiCorpusRejectsWeakPromotionEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Evidence)
		code   string
	}{
		{"missing arm", func(e *Evidence) { e.Results = e.Results[:1] }, ReasonMissingArm},
		{"identity mismatch", func(e *Evidence) { e.Results[0].CorpusSHA = "other" }, ReasonIdentityMismatch},
		{"insufficient samples", func(e *Evidence) { e.Confidence.MinimumPairedSamples = 4 }, ReasonSamples},
		{"invalid baseline", func(e *Evidence) { e.Results[0].BaselineSamples[0] = 0 }, ReasonInvalidBaseline},
		{"quality regression", func(e *Evidence) { e.Results[0].CandidatePassed = 1 }, ReasonQuality},
		{"confidence", func(e *Evidence) { e.Results[0].CandidateSamples = []float64{90, 100, 100} }, ReasonConfidence},
		{"proxy target claim", func(e *Evidence) { e.PromotionClaim = "release" }, ReasonProxyClaim},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := multiEvidence(1)
			tt.mutate(&e)
			got, err := Evaluate(e)
			if err != nil {
				t.Fatal(err)
			}
			if got.Verdict != "HOLD" || got.ReasonCode != tt.code {
				t.Fatalf("decision = %#v", got)
			}
		})
	}
}

func TestEvaluateMultiCorpusRequiresEveryFamilyAtExactTargetForPass(t *testing.T) {
	e := multiEvidence(len(Stages))
	e.PromotionClaim = "release"
	got, err := Evaluate(e)
	if err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "PASS" || got.NextStage != nil {
		t.Fatalf("decision = %#v", got)
	}
}

func TestDecodeAcceptsLegacyAndRequiresExplicitMultiCorpusConfidence(t *testing.T) {
	legacy := `{"schema":"fak.qwen38-ladder-evidence/1","concept":"kernel","corpus_sha256":"c","baseline_runtime_sha":"base","candidate_runtime_sha":"candidate","metric":"p95_ms","direction":"lower","results":[]}`
	if _, err := Decode(strings.NewReader(legacy)); err != nil {
		t.Fatalf("legacy evidence: %v", err)
	}
	multi := `{"schema":"fak.qwen38-ladder-evidence/1","concept":"kernel","baseline_runtime_sha":"base","candidate_runtime_sha":"candidate","metric":"p95_ms","direction":"lower","corpora":[{"name":"arithmetic","task_family":"reasoning","sha256":"a","minimum_pass_rate":0.9}],"results":[]}`
	if _, err := Decode(strings.NewReader(multi)); err == nil || !strings.Contains(err.Error(), "confidence is required") {
		t.Fatalf("error = %v", err)
	}
}

func multiEvidence(stageCount int) Evidence {
	corpora := []Corpus{{Name: "arithmetic", TaskFamily: "reasoning", SHA256: "arith", MinimumPassRate: .9}, {Name: "tool-json", TaskFamily: "tool-use", SHA256: "tools", MinimumPassRate: .9}}
	e := Evidence{Schema: Schema, Concept: "fused-gdn", BaselineRuntimeSHA: "base", CandidateRuntimeSHA: "candidate", Metric: "p95_ms", Direction: "lower", MinimumImprovementPct: 5, Corpora: corpora, Confidence: &ConfidenceRule{Method: "paired-win-rate", MinimumPairedSamples: 3, MinimumWinRate: 2.0 / 3.0}}
	for _, stage := range Stages[:stageCount] {
		for _, corpus := range corpora {
			e.Results = append(e.Results, Result{StageID: stage.ID, Model: stage.Model, Revision: stage.Revision, Corpus: corpus.Name, TaskFamily: corpus.TaskFamily, CorpusSHA: corpus.SHA256, BaselineRuntimeSHA: e.BaselineRuntimeSHA, CandidateRuntimeSHA: e.CandidateRuntimeSHA, BaselineEnvironmentSHA: "env", CandidateEnvironmentSHA: "env", BaselinePassed: 3, CandidatePassed: 3, BaselineSamples: []float64{100, 100, 100}, CandidateSamples: []float64{90, 90, 90}})
		}
	}
	return e
}
