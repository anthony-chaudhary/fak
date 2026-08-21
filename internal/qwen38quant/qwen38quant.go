// Package qwen38quant defines the evidence contract for the Qwen3.8-27B
// quantization campaign. It validates evidence; it does not infer missing facts.
package qwen38quant

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const Schema = "fak.qwen38-quant-report/1"

var RequiredArms = []string{"bf16", "fp8", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs", "awq_int4", "gptq_int4", "exl2"}
var RequiredWorkloads = []string{"text", "json_schema", "correlated_tools", "coding_reasoning", "long_context_retrieval", "repeated_workflow_cache"}

type Corpus struct {
	Schema    string   `json:"schema"`
	ID        string   `json:"id"`
	Workloads []string `json:"workloads"`
}

type Identity struct {
	Model             string `json:"model"`
	CheckpointSHA256  string `json:"checkpoint_sha256"`
	ArtifactSHA256    string `json:"artifact_sha256"`
	TokenizerSHA256   string `json:"tokenizer_sha256"`
	TemplateSHA256    string `json:"template_sha256"`
	QuantizerRevision string `json:"quantizer_revision"`
	RuntimeRevision   string `json:"runtime_revision"`
	FakModuleRev      string `json:"fak_module_rev"`
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
	Workload   string  `json:"workload"`
	Repetition int     `json:"repetition"`
	Quality    string  `json:"quality"`
	LatencyMS  float64 `json:"latency_ms,omitempty"`
	Failure    string  `json:"failure,omitempty"`
}

type Report struct {
	Schema            string      `json:"schema"`
	CorpusID          string      `json:"corpus_id"`
	Arm               string      `json:"arm"`
	Identity          Identity    `json:"identity"`
	Environment       Environment `json:"environment"`
	Trials            []Trial     `json:"trials"`
	Verdict           string      `json:"verdict"`
	EvidenceClass     string      `json:"evidence_class"`
	RawArchiveSHA256  string      `json:"raw_archive_sha256"`
	StaleAfter        string      `json:"stale_after"`
	RollbackThreshold string      `json:"rollback_threshold"`
}

func DefaultCorpus() Corpus {
	return Corpus{Schema: "fak.qwen38-quant-corpus/1", ID: "qwen38-27b-agentic-v1", Workloads: append([]string(nil), RequiredWorkloads...)}
}

func (c Corpus) Validate() error {
	if c.Schema != "fak.qwen38-quant-corpus/1" || c.ID == "" {
		return errors.New("invalid corpus identity")
	}
	return exactSet("workloads", c.Workloads, RequiredWorkloads)
}

func Validate(r Report, c Corpus) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if r.Schema != Schema {
		return fmt.Errorf("schema: got %q", r.Schema)
	}
	if r.CorpusID != c.ID {
		return errors.New("corpus drift")
	}
	if !contains(RequiredArms, r.Arm) {
		return fmt.Errorf("unknown arm %q", r.Arm)
	}
	missing := missingIdentity(r.Identity)
	if len(missing) != 0 {
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
	counts := map[string]map[int]bool{}
	qualityPass := true
	for _, t := range r.Trials {
		if !contains(c.Workloads, t.Workload) {
			return fmt.Errorf("unknown workload %q", t.Workload)
		}
		if t.Repetition < 1 {
			return errors.New("invalid repetition")
		}
		if t.Quality != "PASS" && t.Quality != "FAIL" {
			return errors.New("quality must be PASS or FAIL")
		}
		if t.Quality == "FAIL" {
			qualityPass = false
			if t.Failure == "" {
				return errors.New("failed trial missing retained failure")
			}
		}
		if counts[t.Workload] == nil {
			counts[t.Workload] = map[int]bool{}
		}
		if counts[t.Workload][t.Repetition] {
			return errors.New("duplicate repetition")
		}
		counts[t.Workload][t.Repetition] = true
	}
	for _, w := range c.Workloads {
		if len(counts[w]) < 3 {
			return fmt.Errorf("fewer than three repeats for %s", w)
		}
	}
	if !qualityPass && r.Verdict == "PROMOTE" {
		return errors.New("performance promotion attached to failing quality")
	}
	if r.Verdict != "PROMOTE" && r.Verdict != "HOLD" && r.Verdict != "EXCLUDE" {
		return errors.New("invalid verdict")
	}
	if r.EvidenceClass != "CAMPAIGN" {
		return errors.New("campaign report must declare CAMPAIGN evidence")
	}
	return nil
}

func LegacyAcceptance(arm, corpusID, model string, latencies []float64) Report {
	trials := make([]Trial, 0, len(latencies))
	legacy := []string{"text", "json_schema", "correlated_tools"}
	for i, ms := range latencies {
		if i >= len(legacy) {
			break
		}
		trials = append(trials, Trial{Workload: legacy[i], Repetition: 1, Quality: "PASS", LatencyMS: ms})
	}
	return Report{Schema: Schema, CorpusID: corpusID, Arm: arm, Identity: Identity{Model: model}, Trials: trials, Verdict: "HOLD", EvidenceClass: "ACCEPTANCE_ONLY"}
}

func Selfcheck() error {
	c := DefaultCorpus()
	if err := c.Validate(); err != nil {
		return err
	}
	for _, arm := range []string{"fp8", "q4_k_m"} {
		r := LegacyAcceptance(arm, c.ID, "Qwen3.8-27B", []float64{1, 2, 3})
		if r.Verdict != "HOLD" || r.EvidenceClass != "ACCEPTANCE_ONLY" {
			return errors.New("legacy import promoted")
		}
		if Validate(r, c) == nil {
			return errors.New("acceptance-only evidence passed campaign validation")
		}
	}
	good := validFixture(c)
	if err := Validate(good, c); err != nil {
		return fmt.Errorf("valid fixture: %w", err)
	}
	mutations := []func(*Report){
		func(r *Report) { r.CorpusID = "drift" }, func(r *Report) { r.Identity.ArtifactSHA256 = "" },
		func(r *Report) { r.Environment.DenyFallback = false }, func(r *Report) { r.Trials = r.Trials[:len(r.Trials)-1] },
		func(r *Report) {
			r.Trials[0].Quality = "FAIL"
			r.Trials[0].Failure = "quality regression"
			r.Verdict = "PROMOTE"
		},
	}
	for i, mutate := range mutations {
		r := validFixture(c)
		mutate(&r)
		if Validate(r, c) == nil {
			return fmt.Errorf("refusal fixture %d accepted", i)
		}
	}
	return nil
}

func validFixture(c Corpus) Report {
	h := strings.Repeat("a", 64)
	trials := []Trial{}
	for _, w := range c.Workloads {
		for n := 1; n <= 3; n++ {
			trials = append(trials, Trial{Workload: w, Repetition: n, Quality: "PASS", LatencyMS: float64(n)})
		}
	}
	return Report{Schema: Schema, CorpusID: c.ID, Arm: "q4_k_m", Identity: Identity{Model: "Qwen/Qwen3.8-27B", CheckpointSHA256: h, ArtifactSHA256: h, TokenizerSHA256: h, TemplateSHA256: h, QuantizerRevision: "q@rev", RuntimeRevision: "r@rev", FakModuleRev: "internal/model@r1+gabc"}, Environment: Environment{Command: []string{"fak", "serve"}, Hardware: "A100", Software: "CUDA", ContextTokens: 16384, CacheMode: "on", RequireDevice: "cuda", DenyFallback: true}, Trials: trials, Verdict: "PROMOTE", EvidenceClass: "CAMPAIGN", RawArchiveSHA256: h, StaleAfter: "2026-11-20", RollbackThreshold: "quality pass rate below 100%"}
}

func CorpusDigest(c Corpus) string {
	h := sha256.Sum256([]byte(c.Schema + "\n" + c.ID + "\n" + strings.Join(c.Workloads, "\n")))
	return hex.EncodeToString(h[:])
}
func missingIdentity(i Identity) []string {
	vals := map[string]string{"model": i.Model, "checkpoint_sha256": i.CheckpointSHA256, "artifact_sha256": i.ArtifactSHA256, "tokenizer_sha256": i.TokenizerSHA256, "template_sha256": i.TemplateSHA256, "quantizer_revision": i.QuantizerRevision, "runtime_revision": i.RuntimeRevision, "fak_module_rev": i.FakModuleRev}
	out := []string{}
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
