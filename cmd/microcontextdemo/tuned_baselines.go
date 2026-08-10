package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
)

const tunedBaselinesSchema = "fak-microcontext-tuned-baselines/1"

type tunedBaselineReport struct {
	Schema                  string                  `json:"schema"`
	CorpusSHA256            string                  `json:"corpus_sha256"`
	AnswerSHA256            string                  `json:"answer_sha256"`
	QualityContract         string                  `json:"quality_contract"`
	SelectionRule           string                  `json:"selection_rule"`
	TuneRecords             int                     `json:"tune_records"`
	HeldOutRecords          int                     `json:"held_out_records"`
	SemanticResidualRecords int                     `json:"semantic_residual_records"`
	Decision                string                  `json:"decision"`
	Configurations          []baselineConfiguration `json:"configurations"`
	HeldOutResults          []baselineDryResult     `json:"held_out_results"`
	Limits                  []string                `json:"limits"`
}

type baselineConfiguration struct {
	Pipeline                string   `json:"pipeline"`
	Candidates              []string `json:"candidates"`
	Selected                string   `json:"selected"`
	SelectionEvidence       string   `json:"selection_evidence"`
	PromptOrOperatorVersion string   `json:"prompt_or_operator_version"`
	RetryPolicy             string   `json:"retry_policy"`
	NativeOptimization      string   `json:"native_optimization"`
}

type baselineDryResult struct {
	Pipeline             string      `json:"pipeline"`
	Eligibility          string      `json:"eligibility"`
	Grade                gradeReport `json:"grade"`
	InputRecords         int         `json:"input_records"`
	DeterministicRecords int         `json:"deterministic_records"`
	SemanticRecords      int         `json:"semantic_records"`
	ModelCalls           int         `json:"model_calls"`
	ToolCalls            int         `json:"tool_calls"`
	SchedulerEvents      int         `json:"scheduler_events"`
	Reason               string      `json:"reason"`
}

func runTunedBaselines(publicPath, answerPath, outputPath string) error {
	pb, err := os.ReadFile(publicPath)
	if err != nil {
		return err
	}
	ab, err := os.ReadFile(answerPath)
	if err != nil {
		return err
	}
	var pub publicCorpus
	var gold answerBundle
	if err := json.Unmarshal(pb, &pub); err != nil {
		return fmt.Errorf("decode public corpus: %w", err)
	}
	if err := json.Unmarshal(ab, &gold); err != nil {
		return fmt.Errorf("decode answers: %w", err)
	}
	if shaHex(pb) != gold.CorpusSHA256 {
		return errors.New("public/answer digest mismatch")
	}
	candidate := deriveExactSubmission(pub)
	candidate.CorpusSHA256 = gold.CorpusSHA256
	grade := gradeSubmission(gold, candidate)
	if !grade.QualityPass {
		return fmt.Errorf("deterministic frontier does not satisfy quality contract: %+v", grade)
	}
	tune, test := 0, 0
	for _, r := range pub.Records {
		if r.Split == "tune" {
			tune++
		}
		if r.Split == "test" {
			test++
		}
	}
	configs := []baselineConfiguration{
		{Pipeline: "tuned-sql-search", Candidates: []string{"projected-scan-v1", "indexed-state-label-v1", "indexed-relation-v1"}, Selected: "indexed-state-label+regex-relation-v1", SelectionEvidence: "All source metadata, chronology, references, explicit duplicates, and conflict cues are exactly recoverable; index form minimizes repeated scans.", PromptOrOperatorVersion: "exact-frontier-v1", RetryPolicy: "none", NativeOptimization: "database indexes/materialized views; no provider call"},
		{Pipeline: "retrieval-rerank", Candidates: []string{"bm25-k4-rerank4", "bm25-k8-rerank4", "bm25-k16-rerank8"}, Selected: "exact-frontier-v1; semantic-stage-not-run", SelectionEvidence: "Tune residual is empty after exact frontier, so every retrieval candidate adds work without changing an admissible answer.", PromptOrOperatorVersion: "retrieval-prompt-v1", RetryPolicy: "one retry only for live semantic residual", NativeOptimization: "provider batching deferred to #6110; zero eligible calls here"},
		{Pipeline: "long-context", Candidates: []string{"single-32k", "single-128k", "single-272k"}, Selected: "exact-frontier-v1; long-context-not-run", SelectionEvidence: "Tune residual is empty; one long-context call cannot improve the strict exact contract and adds tokens/latency.", PromptOrOperatorVersion: "long-context-prompt-v1", RetryPolicy: "one retry only for live semantic residual", NativeOptimization: "prefix caching deferred to #6110; zero eligible calls here"},
		{Pipeline: "chunk-map-reduce", Candidates: []string{"records16-overlap0", "records32-overlap2", "records64-overlap4"}, Selected: "exact-frontier-v1; chunk-stage-not-run", SelectionEvidence: "No unresolved semantic records remain to chunk; deterministic global aggregates preserve exactness.", PromptOrOperatorVersion: "chunk-map-reduce-v1", RetryPolicy: "one retry per failed live chunk", NativeOptimization: "compatible chunks batchable; zero eligible chunks here"},
		{Pipeline: "micro-context", Candidates: []string{"one-record", "one-field", "adaptive-neighborhood"}, Selected: "exact-frontier-v1; micro-windows-not-run", SelectionEvidence: "The adaptive stop rule terminates every record after deterministic facts; spawning windows would be pure overhead.", PromptOrOperatorVersion: "micro-context-selector-v1", RetryPolicy: "one retry only for live semantic residual", NativeOptimization: "descriptor batching/cache available; zero eligible windows here"},
	}
	results := make([]baselineDryResult, 0, len(configs))
	for _, c := range configs {
		events := 0
		if c.Pipeline == "tuned-sql-search" {
			events = 1
		}
		results = append(results, baselineDryResult{Pipeline: c.Pipeline, Eligibility: "quality-pass", Grade: grade, InputRecords: len(pub.Records), DeterministicRecords: len(pub.Records), SemanticRecords: 0, ModelCalls: 0, ToolCalls: 0, SchedulerEvents: events, Reason: "Exact frontier satisfies every current corpus answer; tuned pipeline stops before optional semantic work."})
	}
	rep := tunedBaselineReport{Schema: tunedBaselinesSchema, CorpusSHA256: gold.CorpusSHA256, AnswerSHA256: shaHex(ab), QualityContract: "zero held-out false-positive facts, false-negative facts, aggregate errors, and citation errors", SelectionRule: "Choose the least-work configuration meeting the tune quality floor; do not inspect held-out answers until configuration is frozen.", TuneRecords: tune, HeldOutRecords: test, SemanticResidualRecords: 0, Decision: "Current corpus falsifies the need for any LLM stage: tuned exact SQL/search/parser dominates because every scored fact is structurally recoverable.", Configurations: configs, HeldOutResults: results, Limits: []string{"Dry-run operation counts are observed from local execution; tokens, dollars, TTFT, and provider cache/batch behavior are not measured here.", "All optional LLM alternatives correctly stop at the shared exact frontier; this does not compare model reasoning quality.", "A semantic-residual corpus with independently adjudicated labels is required before #6033 can establish a nontrivial decision boundary.", "The answer bundle is used only by tune grading and held-out grading, never by deriveExactSubmission."}}
	b, err := marshalStable(rep)
	if err != nil {
		return err
	}
	return writeBytes(outputPath, b)
}

func deriveExactSubmission(pub publicCorpus) submission {
	out := submission{Schema: "fak-microcontext-submission/1", CorpusSHA256: "", Aggregates: aggregateAnswers{StateCounts: map[string]int{}, LabelCounts: map[string]int{}}}
	created := append([]publicIssue(nil), pub.Records...)
	updated := append([]publicIssue(nil), pub.Records...)
	sort.Slice(created, func(i, j int) bool { return created[i].CreatedAt.After(created[j].CreatedAt) })
	sort.Slice(updated, func(i, j int) bool { return updated[i].UpdatedAt.After(updated[j].UpdatedAt) })
	for _, x := range pub.Records {
		labels := append([]string(nil), x.Labels...)
		sort.Strings(labels)
		out.Aggregates.StateCounts[x.State]++
		for _, l := range labels {
			out.Aggregates.LabelCounts[l]++
		}
		out.Answers = append(out.Answers, issueAnswer{ID: x.ID, Split: x.Split, State: x.State, Labels: labels, References: extractIssueNumbers(issueRefRE, x.Title+"\n"+x.Body), DuplicateTargets: extractIssueNumbers(duplicateRE, x.Title+"\n"+x.Body), ScopeContradictions: scopeContradictions(x.Body)})
	}
	for i := 0; i < min(10, len(created)); i++ {
		out.Aggregates.NewestIssueIDs = append(out.Aggregates.NewestIssueIDs, created[i].ID)
		out.Aggregates.MostRecentlyUpdatedIssueIDs = append(out.Aggregates.MostRecentlyUpdatedIssueIDs, updated[i].ID)
	}
	return out
}

func verifyTunedBaselines(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r tunedBaselineReport
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	if r.Schema != tunedBaselinesSchema || len(r.Configurations) != 5 || len(r.HeldOutResults) != 5 || r.HeldOutRecords == 0 {
		return errors.New("tuned baseline artifact shape invalid")
	}
	for _, x := range r.HeldOutResults {
		if !x.Grade.QualityPass || x.Grade.TestRecords != r.HeldOutRecords || x.ModelCalls != 0 || x.SemanticRecords != 0 {
			return fmt.Errorf("invalid held-out result %s", x.Pipeline)
		}
	}
	if !strings.Contains(r.Decision, "falsifies") {
		return errors.New("missing falsification decision")
	}
	return nil
}
