package studyclass

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/studyforge"
)

type mechanismRule struct {
	mechanism    Mechanism
	titleSignals []string
	bodySignals  []string
	labels       []string
}

var mechanismRules = []mechanismRule{
	{MechanismArchitectureRuntime, []string{"architecture", "runtime engine", "executor", "engine core", "vllm ir", "worker process"}, []string{"runtime engine", "engine core", "vllm ir", "worker process"}, []string{"v1", "vllm-ir", "rust"}},
	{MechanismSchedulingBatching, []string{"scheduler", "scheduling", "continuous batching", "chunked prefill", "batch admission", "decode scheduling", "preemption"}, []string{"continuous batching", "chunked prefill", "batch admission", "decode scheduling"}, []string{"scheduler"}},
	{MechanismKVCache, []string{"kv cache", "kv-cache", "paged attention", "prefix caching", "prefix cache", "cache connector", "block manager"}, []string{"kv cache", "kv-cache", "paged attention", "prefix caching", "cache connector", "block manager"}, []string{"kv-connector", "kv-cache-manager"}},
	{MechanismKernelsCompilation, []string{"cuda kernel", "custom kernel", "triton", "torch.compile", "cuda graph", "compilation", "compiler", "gemm"}, []string{"cuda kernel", "custom kernel", "triton kernel", "torch.compile", "cuda graph", "gemm kernel"}, []string{"torch.compile", "vllm-ir"}},
	{MechanismSpeculativeDecoding, []string{"speculative decoding", "speculative-decoding", "draft model", "eagle", "dflash"}, []string{"speculative decoding", "speculative-decoding", "draft model", "eagle speculative", "dflash"}, []string{"speculative-decoding", "dflash"}},
	{MechanismDistributedParallelism, []string{"distributed", "tensor parallel", "pipeline parallel", "data parallel", "expert parallel", "all-reduce", "all reduce", "collective", "nccl"}, []string{"tensor parallel", "pipeline parallel", "data parallel", "expert parallel", "all-reduce", "all reduce", "nccl collective"}, []string{"ray"}},
	{MechanismMemoryResidency, []string{"out of memory", "oom", "memory residency", "memory allocation", "memory leak", "allocator", "fragmentation", "offload", "swapping"}, []string{"out of memory", "memory residency", "memory allocation", "memory leak", "allocator fragmentation", "cpu offload", "swapping"}, nil},
	{MechanismModelBackendHardware, []string{"model support", "backend", "hardware", "gpu", "cuda", "rocm", "quantization", "multimodal", "multi-modal"}, []string{"model support", "hardware backend", "gpu backend", "quantization backend", "multi-modal model"}, []string{"rocm", "nvidia", "cpu", "tpu", "intel-gpu", "new-model", "multi-modality", "quantization", "qwen", "deepseek", "llama", "mistral", "gpt-oss", "kimi", "minimax", "glm", "cohere"}},
	{MechanismAPIsToolCallingStructuredOutput, []string{"tool calling", "tool-calling", "structured output", "structured-output", "openai api", "api server", "frontend"}, []string{"tool calling", "tool-calling", "structured output", "structured-output", "openai api", "json schema", "guided decoding"}, []string{"frontend", "tool-calling", "structured-output", "streaming-input"}},
	{MechanismObservabilityOperations, []string{"observability", "logging", "metrics", "telemetry", "tracing", "profiling", "health check", "deployment", "kubernetes", "helm", "installation", "startup"}, []string{"metrics endpoint", "distributed tracing", "health check", "kubernetes deployment", "helm chart", "startup latency"}, []string{"installation", "usage", "startup-ux"}},
	{MechanismReliabilitySecurity, []string{"security", "reliability", "crash", "hang", "deadlock", "race condition", "retry", "fault tolerance", "data loss", "authentication", "authorization", "cve", "vulnerability"}, []string{"race condition", "retry storm", "fault tolerance", "data loss", "authentication failure", "security vulnerability", "deadlock"}, nil},
	{MechanismTestsCIDocs, []string{"unit test", "integration test", "test failure", "documentation", "github actions", "continuous integration", "ci failure", "build failure"}, []string{"unit test", "integration test", "test failure", "github actions", "continuous integration", "ci failure", "build failure"}, []string{"ci/build", "documentation", "ci-failure", "build-docs", "github_actions", "ready-run-all-tests"}},
}

var (
	duplicateTargetRE  = regexp.MustCompile(`(?i)\b(?:duplicate of|duplicates)\s+(?:#\d+|https://github\.com/[^/\s]+/[^/\s]+/(?:issues|pull)/\d+)`)
	supersededTargetRE = regexp.MustCompile(`(?i)(?:^\s*(?:\[[^\]]+\]\s*)?superseded by\s+(?:#\d+|https://github\.com/[^/\s]+/[^/\s]+/(?:issues|pull)/\d+)|^\s*\[superseded\])`)
)

func classifyRecord(r studyforge.Record) Classification {
	state := normalizeState(r.State)
	disposition, confidence, dispositionEvidence := classifyDisposition(r, state)
	c := Classification{
		Identity:            identity(r.Source, r.ID),
		Source:              r.Source,
		Kind:                r.Kind,
		ID:                  r.ID,
		NodeID:              r.NodeID,
		Number:              r.Number,
		URL:                 r.URL,
		Labels:              append([]string(nil), r.Labels...),
		State:               state,
		Dates:               recordDates(r),
		Merged:              r.Merged || r.MergedAt != "",
		Disposition:         disposition,
		Confidence:          confidence,
		DispositionEvidence: dispositionEvidence,
		Mechanisms:          classifyMechanisms(r, disposition),
	}
	return c
}

func classifyDisposition(r studyforge.Record, state string) (Disposition, Confidence, []Evidence) {
	if r.Source == "releases" || r.Source == "labels" || r.Source == "milestones" {
		return DispositionReleaseMetadataNoncandidate, ConfidenceHigh, []Evidence{{Rule: "disposition.metadata_source", Field: "source", Signal: r.Source}}
	}
	if r.Source == "pulls" && (r.Merged || r.MergedAt != "") {
		field, signal := "merged", "true"
		if r.MergedAt != "" {
			field, signal = "merged_at", "present"
		}
		return DispositionMergedLanded, ConfidenceHigh, []Evidence{{Rule: "disposition.merged_pull", Field: field, Signal: signal}}
	}
	if field, signal, ok := explicitRelationship(r, duplicateTargetRE); ok {
		return DispositionDuplicate, ConfidenceMedium, []Evidence{{Rule: "disposition.duplicate.explicit_target", Field: field, Signal: signal}}
	}
	if normalizedLabels(r.Labels)["stale"] {
		return DispositionStaleSuperseded, ConfidenceHigh, []Evidence{{Rule: "disposition.stale_superseded.label", Field: "labels", Signal: "stale"}}
	}
	if field, signal, ok := explicitRelationship(r, supersededTargetRE); ok {
		return DispositionStaleSuperseded, ConfidenceMedium, []Evidence{{Rule: "disposition.stale_superseded.explicit_target", Field: field, Signal: signal}}
	}
	category := strings.ToLower(strings.TrimSpace(r.Category))
	if category == "q&a" || category == "questions" {
		return DispositionSupportQuestion, ConfidenceHigh, []Evidence{{Rule: "disposition.question_category", Field: "category", Signal: category}}
	}
	if e, ok := dispositionTitleSignal(r, []string{"question", "support", "usage", "installation"}, []string{"how do i", "how to", "is it possible", "why does", "question"}, "support_question"); ok {
		return DispositionSupportQuestion, e.confidence, []Evidence{e.evidence}
	}
	if strings.HasSuffix(strings.TrimSpace(r.Title), "?") {
		return DispositionSupportQuestion, ConfidenceMedium, []Evidence{{Rule: "disposition.question_title", Field: "title", Signal: "terminal_question_mark"}}
	}
	if e, ok := dispositionTitleSignal(r, []string{"bug", "regression"}, []string{"regression", "segfault", "crash", "deadlock", "wrong result", "data corruption", "out of memory", "oom"}, "regression_bug"); ok {
		return DispositionRegressionBug, e.confidence, []Evidence{e.evidence}
	}
	if state == "open" {
		if e, ok := dispositionTitleSignal(r, []string{"feature request", "enhancement", "rfc", "proposal"}, []string{"feature request", "proposal", "rfc", "support for", "add support"}, "open_proposal"); ok {
			return DispositionOpenProposal, e.confidence, []Evidence{e.evidence}
		}
		return DispositionOpenProposal, ConfidenceLow, []Evidence{{Rule: "disposition.open_fallback", Field: "state", Signal: "open"}}
	}
	return DispositionClosedUnmerged, ConfidenceLow, []Evidence{{Rule: "disposition.closed_fallback", Field: "state", Signal: state}}
}

func explicitRelationship(r studyforge.Record, re *regexp.Regexp) (string, string, bool) {
	if match := re.FindString(r.Title); match != "" {
		return "title", strings.ToLower(strings.TrimSpace(match)), true
	}
	if match := re.FindString(r.Body); match != "" {
		return "body", strings.ToLower(strings.TrimSpace(match)), true
	}
	return "", "", false
}

func dispositionTitleSignal(r studyforge.Record, labels, titleSignals []string, rule string) (dispositionMatch, bool) {
	labelSet := normalizedLabels(r.Labels)
	for _, signal := range labels {
		if labelSet[strings.ToLower(signal)] {
			return dispositionMatch{ConfidenceHigh, Evidence{Rule: "disposition." + rule + ".label", Field: "labels", Signal: signal}}, true
		}
	}
	for _, signal := range titleSignals {
		if containsSignal(r.Title, signal) {
			return dispositionMatch{ConfidenceMedium, Evidence{Rule: "disposition." + rule + ".title", Field: "title", Signal: signal}}, true
		}
	}
	return dispositionMatch{}, false
}

type dispositionMatch struct {
	confidence Confidence
	evidence   Evidence
}

func classifyMechanisms(r studyforge.Record, disposition Disposition) []MechanismMatch {
	if disposition == DispositionReleaseMetadataNoncandidate {
		return []MechanismMatch{{
			Name:       MechanismExplicitNonCandidate,
			Confidence: ConfidenceHigh,
			Evidence:   []Evidence{{Rule: "mechanism.explicit_non_candidate.disposition", Field: "disposition", Signal: string(disposition)}},
		}}
	}

	labels := normalizedLabels(r.Labels)
	matches := make([]MechanismMatch, 0)
	for _, rule := range mechanismRules {
		var evidence []Evidence
		confidence := ConfidenceLow
		for _, signal := range rule.labels {
			if labels[strings.ToLower(signal)] {
				evidence = append(evidence, Evidence{Rule: "mechanism." + string(rule.mechanism) + ".label", Field: "labels", Signal: signal})
				confidence = higherConfidence(confidence, ConfidenceHigh)
			}
		}
		for _, signal := range rule.titleSignals {
			if containsSignal(r.Title, signal) {
				evidence = append(evidence, Evidence{Rule: "mechanism." + string(rule.mechanism) + ".title", Field: "title", Signal: signal})
				confidence = higherConfidence(confidence, ConfidenceMedium)
			}
		}
		for _, signal := range rule.bodySignals {
			if containsSignal(r.Body, signal) {
				evidence = append(evidence, Evidence{Rule: "mechanism." + string(rule.mechanism) + ".body", Field: "body", Signal: signal})
			}
		}
		if len(evidence) > 0 {
			sortEvidence(evidence)
			matches = append(matches, MechanismMatch{Name: rule.mechanism, Confidence: confidence, Evidence: evidence})
		}
	}
	sort.Slice(matches, func(i, j int) bool { return mechanismRank(matches[i].Name) < mechanismRank(matches[j].Name) })
	return matches
}

func normalizedLabels(labels []string) map[string]bool {
	out := make(map[string]bool, len(labels))
	for _, label := range labels {
		out[strings.ToLower(strings.TrimSpace(label))] = true
	}
	return out
}

func containsSignal(text, signal string) bool {
	text = strings.ToLower(text)
	signal = strings.ToLower(signal)
	for start := 0; ; {
		i := strings.Index(text[start:], signal)
		if i < 0 {
			return false
		}
		i += start
		beforeOK := i == 0 || !isWordRune(rune(text[i-1])) || !isWordRune(rune(signal[0]))
		after := i + len(signal)
		afterOK := after == len(text) || !isWordRune(rune(text[after])) || !isWordRune(rune(signal[len(signal)-1]))
		if beforeOK && afterOK {
			return true
		}
		start = i + 1
	}
}

func isWordRune(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' }

func higherConfidence(a, b Confidence) Confidence {
	if confidenceRank(b) < confidenceRank(a) {
		return b
	}
	return a
}

func normalizeState(state string) string {
	state = strings.ToLower(strings.TrimSpace(state))
	if state == "" {
		return "none"
	}
	return state
}

func identity(source string, id int64) string { return fmt.Sprintf("%s:%d", source, id) }

func recordDates(r studyforge.Record) Dates {
	return Dates{Created: r.CreatedAt, Updated: r.UpdatedAt, Closed: r.ClosedAt, Merged: r.MergedAt, Published: r.PublishedAt}
}

func mechanismRank(m Mechanism) int {
	for i, candidate := range Mechanisms {
		if candidate == m {
			return i
		}
	}
	return len(Mechanisms)
}

func confidenceRank(c Confidence) int {
	for i, candidate := range Confidences {
		if candidate == c {
			return i
		}
	}
	return len(Confidences)
}

func sortEvidence(e []Evidence) {
	sort.Slice(e, func(i, j int) bool {
		if e[i].Rule != e[j].Rule {
			return e[i].Rule < e[j].Rule
		}
		if e[i].Field != e[j].Field {
			return e[i].Field < e[j].Field
		}
		return e[i].Signal < e[j].Signal
	})
}
