package qwen38quant

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const Schema = "fak.qwen38-quant-report/1"

var RequiredArms = []string{"bf16", "fp8", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs", "awq_int4", "gptq_int4", "exl2"}
var RequiredWorkloads = []string{"text", "json_schema", "correlated_tools", "coding_reasoning", "long_context_retrieval", "repeated_workflow_cache"}

type Fixture struct {
	ID                    string           `json:"id"`
	Workload              string           `json:"workload"`
	Prompt                string           `json:"prompt"`
	ExpectedExact         string           `json:"expected_exact,omitempty"`
	JSONSchema            map[string]any   `json:"json_schema,omitempty"`
	ExpectedJSON          map[string]any   `json:"expected_json,omitempty"`
	Tools                 []map[string]any `json:"tools,omitempty"`
	ExpectedTool          map[string]any   `json:"expected_tool,omitempty"`
	Generator             map[string]any   `json:"generator,omitempty"`
	MinimumContextTokens  int              `json:"minimum_context_tokens,omitempty"`
	CacheSequence         []string         `json:"cache_sequence,omitempty"`
	RequiredCacheEvidence []string         `json:"required_cache_evidence,omitempty"`
	MaxOutputTokens       int              `json:"max_output_tokens"`
}

type Corpus struct {
	Schema                     string    `json:"schema"`
	ID                         string    `json:"id"`
	ModelFamily                string    `json:"model_family"`
	Arms                       []string  `json:"arms"`
	Workloads                  []string  `json:"workloads"`
	MinimumRepetitions         int       `json:"minimum_repetitions_per_workload"`
	QualityPrecedesPerformance bool      `json:"quality_precedes_performance"`
	Fixtures                   []Fixture `json:"fixtures"`
}

type Identity struct {
	Model, CheckpointSHA256, ArtifactSHA256, TokenizerSHA256, TemplateSHA256 string
	QuantizerRevision, RuntimeRevision, FakModuleRev                         string
}

type Environment struct {
	Command       []string `json:"command"`
	Hardware      string   `json:"hardware"`
	Software      string   `json:"software"`
	ContextTokens int      `json:"context_tokens"`
	CacheMode     string   `json:"cache_mode"`
	RequireDevice string   `json:"require_device"`
	DenyFallback  bool     `json:"deny_fallback"`
}

type Trial struct {
	Workload         string  `json:"workload"`
	Repetition       int     `json:"repetition"`
	Quality          string  `json:"quality"`
	LatencyMS        float64 `json:"latency_ms,omitempty"`
	Failure          string  `json:"failure,omitempty"`
	CompletionTokens int     `json:"completion_tokens,omitempty"`
}

type Report struct {
	Schema, CorpusID, CorpusSHA256, Arm string
	Identity                            Identity    `json:"identity"`
	Environment                         Environment `json:"environment"`
	Trials                              []Trial     `json:"trials"`
	Verdict, EvidenceClass              string
	RawArchiveSHA256                    string `json:"raw_archive_sha256"`
	StaleAfter                          string `json:"stale_after"`
	RollbackThreshold                   string `json:"rollback_threshold"`
}

func DecodeCorpus(data []byte) (Corpus, error) {
	var c Corpus
	d := json.NewDecoder(strings.NewReader(string(data)))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return Corpus{}, err
	}
	if err := c.Validate(); err != nil {
		return Corpus{}, err
	}
	return c, nil
}

func (c Corpus) Validate() error {
	if c.Schema != "fak.qwen38-quant-corpus/1" || c.ID == "" || c.ModelFamily != "Qwen3.8-27B" || c.MinimumRepetitions < 3 || !c.QualityPrecedesPerformance {
		return errors.New("invalid corpus identity or policy")
	}
	if err := exactSet("arms", c.Arms, RequiredArms); err != nil {
		return err
	}
	if err := exactSet("workloads", c.Workloads, RequiredWorkloads); err != nil {
		return err
	}
	if len(c.Fixtures) != len(RequiredWorkloads) {
		return errors.New("each workload requires one fixture")
	}
	ids, families := map[string]bool{}, map[string]bool{}
	for _, f := range c.Fixtures {
		if f.ID == "" || ids[f.ID] || families[f.Workload] || !contains(RequiredWorkloads, f.Workload) || f.Prompt == "" || f.MaxOutputTokens <= 0 {
			return errors.New("invalid or duplicate fixture")
		}
		ids[f.ID], families[f.Workload] = true, true
		if f.ExpectedExact == "" && len(f.ExpectedJSON) == 0 && len(f.ExpectedTool) == 0 {
			return fmt.Errorf("fixture %s lacks expected effect", f.ID)
		}
		switch f.Workload {
		case "json_schema":
			if len(f.JSONSchema) == 0 || len(f.ExpectedJSON) == 0 {
				return errors.New("json fixture incomplete")
			}
		case "correlated_tools":
			if len(f.Tools) == 0 || len(f.ExpectedTool) == 0 {
				return errors.New("tool fixture incomplete")
			}
		case "long_context_retrieval":
			if len(f.Generator) == 0 || f.MinimumContextTokens <= 0 {
				return errors.New("long-context fixture incomplete")
			}
		case "repeated_workflow_cache":
			if len(f.CacheSequence) < 3 || len(f.RequiredCacheEvidence) == 0 {
				return errors.New("cache fixture incomplete")
			}
		}
	}
	return nil
}

func Validate(r Report, c Corpus) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if r.Schema != Schema {
		return fmt.Errorf("schema: got %q", r.Schema)
	}
	if r.CorpusID != c.ID || r.CorpusSHA256 != CorpusDigest(c) {
		return errors.New("corpus drift")
	}
	if !contains(RequiredArms, r.Arm) {
		return fmt.Errorf("unknown arm %q", r.Arm)
	}
	if missing := missingIdentity(r.Identity); len(missing) != 0 {
		return fmt.Errorf("missing immutable identity: %s", strings.Join(missing, ", "))
	}
	if len(r.Environment.Command) == 0 || r.Environment.Command[0] == "" {
		return errors.New("missing argv-form command")
	}
	if r.Environment.Hardware == "" || r.Environment.Software == "" || r.Environment.ContextTokens <= 0 || r.Environment.CacheMode == "" || r.Environment.RequireDevice == "" {
		return errors.New("incomplete environment")
	}
	if !r.Environment.DenyFallback {
		return errors.New("ambiguous fallback policy")
	}
	if !validHash(r.RawArchiveSHA256) {
		return errors.New("missing raw archive hash")
	}
	if r.StaleAfter == "" || r.RollbackThreshold == "" {
		return errors.New("missing lifecycle boundary")
	}
	if r.EvidenceClass != "CAMPAIGN" {
		return errors.New("acceptance evidence cannot satisfy campaign")
	}
	if r.Verdict != "PROMOTE" && r.Verdict != "HOLD" && r.Verdict != "EXCLUDE" {
		return errors.New("invalid verdict")
	}
	counts, failed, successfulCompletions := map[string]map[int]bool{}, false, 0
	for _, w := range c.Workloads {
		counts[w] = map[int]bool{}
	}
	for _, t := range r.Trials {
		if _, ok := counts[t.Workload]; !ok || t.Repetition <= 0 || counts[t.Workload][t.Repetition] {
			return errors.New("invalid or duplicate trial")
		}
		counts[t.Workload][t.Repetition] = true
		if t.CompletionTokens > 0 {
			successfulCompletions++
		}
		if t.Quality != "PASS" {
			failed = true
			if t.Failure == "" {
				return errors.New("failed trial not retained")
			}
		}
	}
	for w, reps := range counts {
		if len(reps) < c.MinimumRepetitions {
			return fmt.Errorf("workload %s has fewer than %d repetitions", w, c.MinimumRepetitions)
		}
	}
	if successfulCompletions == 0 {
		return errors.New("trials: no successful API completions; fix the serving contract before campaign analysis")
	}
	if failed && r.Verdict == "PROMOTE" {
		return errors.New("quality failure cannot be promoted")
	}
	return nil
}

func CorpusDigest(c Corpus) string {
	b, _ := json.Marshal(c)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func LegacyAcceptance(arm, corpusID, model string, latencies []float64) Report {
	r := Report{Schema: Schema, CorpusID: corpusID, Arm: arm, Identity: Identity{Model: model}, Verdict: "HOLD", EvidenceClass: "ACCEPTANCE_ONLY"}
	for n, ms := range latencies {
		r.Trials = append(r.Trials, Trial{Workload: RequiredWorkloads[n%len(RequiredWorkloads)], Repetition: 1, Quality: "PASS", LatencyMS: ms})
	}
	return r
}

func Selfcheck() error {
	c := testCorpus()
	good := validFixture(c)
	if err := Validate(good, c); err != nil {
		return fmt.Errorf("valid fixture: %w", err)
	}
	mutations := []func(*Report){func(r *Report) { r.CorpusID = "drift" }, func(r *Report) { r.Identity.ArtifactSHA256 = "" }, func(r *Report) { r.Environment.DenyFallback = false }, func(r *Report) { r.Trials = r.Trials[:len(r.Trials)-1] }, func(r *Report) {
		r.Trials[0].Quality = "FAIL"
		r.Trials[0].Failure = "quality regression"
		r.Verdict = "PROMOTE"
	}}
	for i, mutate := range mutations {
		r := validFixture(c)
		mutate(&r)
		if Validate(r, c) == nil {
			return fmt.Errorf("refusal fixture %d accepted", i)
		}
	}
	return nil
}

func testCorpus() Corpus {
	fixtures := make([]Fixture, 0, len(RequiredWorkloads))
	for _, w := range RequiredWorkloads {
		f := Fixture{ID: w + "-v1", Workload: w, Prompt: "fixture", ExpectedExact: "ok", MaxOutputTokens: 8}
		switch w {
		case "json_schema":
			f.JSONSchema = map[string]any{"type": "object"}
			f.ExpectedJSON = map[string]any{"ok": true}
			f.ExpectedExact = ""
		case "correlated_tools":
			f.Tools = []map[string]any{{"type": "function"}}
			f.ExpectedTool = map[string]any{"name": "x"}
			f.ExpectedExact = ""
		case "long_context_retrieval":
			f.Generator = map[string]any{"kind": "records"}
			f.MinimumContextTokens = 16
		case "repeated_workflow_cache":
			f.CacheSequence = []string{"cold", "warm", "restart"}
			f.RequiredCacheEvidence = []string{"saved"}
		}
		fixtures = append(fixtures, f)
	}
	return Corpus{Schema: "fak.qwen38-quant-corpus/1", ID: "qwen38-27b-agentic-v1", ModelFamily: "Qwen3.8-27B", Arms: append([]string(nil), RequiredArms...), Workloads: append([]string(nil), RequiredWorkloads...), MinimumRepetitions: 3, QualityPrecedesPerformance: true, Fixtures: fixtures}
}
func DefaultCorpus() Corpus { return testCorpus() }

func validFixture(c Corpus) Report {
	h := strings.Repeat("a", 64)
	var trials []Trial
	for _, w := range c.Workloads {
		for n := 1; n <= c.MinimumRepetitions; n++ {
			trials = append(trials, Trial{Workload: w, Repetition: n, Quality: "PASS", LatencyMS: float64(n), CompletionTokens: 1})
		}
	}
	return Report{Schema: Schema, CorpusID: c.ID, CorpusSHA256: CorpusDigest(c), Arm: "q4_k_m", Identity: Identity{Model: "Qwen/Qwen3.8-27B", CheckpointSHA256: h, ArtifactSHA256: h, TokenizerSHA256: h, TemplateSHA256: h, QuantizerRevision: "q@rev", RuntimeRevision: "r@rev", FakModuleRev: "internal/model@r1+gabc"}, Environment: Environment{Command: []string{"fak", "serve"}, Hardware: "A100", Software: "CUDA", ContextTokens: 16384, CacheMode: "on", RequireDevice: "cuda", DenyFallback: true}, Trials: trials, Verdict: "PROMOTE", EvidenceClass: "CAMPAIGN", RawArchiveSHA256: h, StaleAfter: "2026-11-20", RollbackThreshold: "quality pass rate below 100%"}
}

func missingIdentity(i Identity) []string {
	vals := map[string]string{"model": i.Model, "checkpoint_sha256": i.CheckpointSHA256, "artifact_sha256": i.ArtifactSHA256, "tokenizer_sha256": i.TokenizerSHA256, "template_sha256": i.TemplateSHA256, "quantizer_revision": i.QuantizerRevision, "runtime_revision": i.RuntimeRevision, "fak_module_rev": i.FakModuleRev}
	var out []string
	for k, v := range vals {
		if v == "" || strings.HasSuffix(k, "sha256") && !validHash(v) {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}
func validHash(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}
func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
func exactSet(name string, got, want []string) error {
	if len(got) != len(want) {
		return fmt.Errorf("%s: want %d got %d", name, len(want), len(got))
	}
	seen := map[string]bool{}
	for _, v := range got {
		if seen[v] || !contains(want, v) {
			return fmt.Errorf("%s mismatch", name)
		}
		seen[v] = true
	}
	return nil
}
