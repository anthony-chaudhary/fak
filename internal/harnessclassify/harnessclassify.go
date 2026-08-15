package harnessclassify

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const Schema = "fak.harness-classification/v1alpha1"

type Input struct {
	Path          string            `json:"path,omitempty"`
	Task          string            `json:"task,omitempty"`
	TaskDomain    string            `json:"task_domain,omitempty"`
	ProjectDomain string            `json:"project_domain,omitempty"`
	Signals       map[string]string `json:"signals,omitempty"`
	Choice        *Choice           `json:"choice,omitempty"`
	Now           time.Time         `json:"-"`
}

type Choice struct {
	Domain    string    `json:"domain"`
	Scope     string    `json:"scope"`
	ExpiresAt time.Time `json:"expires_at"`
	Reason    string    `json:"reason"`
}

type Evidence struct {
	Kind   string `json:"kind"`
	Source string `json:"source"`
	Value  string `json:"value"`
	Weight int    `json:"weight,omitempty"`
}

type Candidate struct {
	Domain string `json:"domain"`
	Score  int    `json:"score"`
}

type DecisionRequest struct {
	Reason  string   `json:"reason"`
	Choices []string `json:"choices"`
	Scope   string   `json:"scope"`
}

type Result struct {
	Schema          string           `json:"schema"`
	Domain          string           `json:"domain,omitempty"`
	Source          string           `json:"source"`
	Confidence      float64          `json:"confidence"`
	Evidence        []Evidence       `json:"evidence,omitempty"`
	Candidates      []Candidate      `json:"candidates,omitempty"`
	NeedsDecision   bool             `json:"needs_decision"`
	DecisionRequest *DecisionRequest `json:"decision_request,omitempty"`
	ContextKey      string           `json:"context_key"`
}

type rule struct {
	domain, kind, needle string
	weight               int
}

var rules = []rule{
	{"legal", "extension", ".docx", 2}, {"legal", "extension", ".pdf", 1},
	{"legal", "task", "brief", 2}, {"legal", "task", "contract", 2}, {"legal", "task", "citation", 2}, {"legal", "task", "deposition", 3}, {"legal", "task", "matter", 1},
	{"coding", "extension", ".go", 2}, {"coding", "extension", ".py", 2}, {"coding", "extension", ".ts", 2}, {"coding", "extension", ".rs", 2},
	{"coding", "task", "compile", 2}, {"coding", "task", "implement", 2}, {"coding", "task", "test", 1}, {"coding", "task", "refactor", 2}, {"coding", "task", "bug", 2},
	{"integrated", "task", "incident", 2}, {"integrated", "task", "deploy", 2}, {"integrated", "task", "customer", 1}, {"integrated", "task", "runbook", 2}, {"integrated", "task", "production", 2},
}

var domains = map[string]bool{"legal": true, "coding": true, "integrated": true}

func Classify(in Input) (Result, error) {
	now := in.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	key := contextKey(in.Path, in.Task)
	if in.TaskDomain != "" {
		return explicit(in.TaskDomain, "task-declaration", key)
	}
	if in.ProjectDomain != "" {
		return explicit(in.ProjectDomain, "project-declaration", key)
	}
	if in.Choice != nil {
		if err := validateDomain(in.Choice.Domain); err != nil {
			return Result{}, err
		}
		if in.Choice.Scope != key {
			return Result{}, fmt.Errorf("remembered choice scope %q does not match context %q", in.Choice.Scope, key)
		}
		if !in.Choice.ExpiresAt.After(now) {
			return Result{}, fmt.Errorf("remembered choice expired at %s", in.Choice.ExpiresAt.UTC().Format(time.RFC3339))
		}
		if strings.TrimSpace(in.Choice.Reason) == "" {
			return Result{}, fmt.Errorf("remembered choice reason is required")
		}
		return Result{Schema: Schema, Domain: in.Choice.Domain, Source: "remembered-choice", Confidence: 1, ContextKey: key, Evidence: []Evidence{{Kind: "operator-choice", Source: in.Choice.Scope, Value: in.Choice.Reason}}}, nil
	}

	scores := map[string]int{}
	evidence := make([]Evidence, 0)
	task := strings.ToLower(in.Task)
	ext := strings.ToLower(filepath.Ext(in.Path))
	for _, r := range rules {
		matched := (r.kind == "task" && tokenContains(task, r.needle)) || (r.kind == "extension" && ext == r.needle)
		if matched {
			scores[r.domain] += r.weight
			evidence = append(evidence, Evidence{Kind: r.kind, Source: r.needle, Value: r.domain, Weight: r.weight})
		}
	}
	keys := make([]string, 0, len(in.Signals))
	for k := range in.Signals {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		domain := strings.ToLower(strings.TrimSpace(in.Signals[k]))
		if !domains[domain] {
			return Result{}, fmt.Errorf("signal %q names unknown domain %q", k, domain)
		}
		scores[domain] += 2
		evidence = append(evidence, Evidence{Kind: "signal", Source: k, Value: domain, Weight: 2})
	}
	candidates := make([]Candidate, 0, len(scores))
	for domain, score := range scores {
		candidates = append(candidates, Candidate{Domain: domain, Score: score})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].Score != candidates[j].Score {
			return candidates[i].Score > candidates[j].Score
		}
		return candidates[i].Domain < candidates[j].Domain
	})
	result := Result{Schema: Schema, Source: "inference", Evidence: evidence, Candidates: candidates, ContextKey: key}
	if len(candidates) == 0 {
		return decision(result, "no domain signal", []string{"legal", "coding", "integrated"}), nil
	}
	margin := candidates[0].Score
	if len(candidates) > 1 {
		margin -= candidates[1].Score
	}
	independent := independentEvidence(evidence, candidates[0].Domain)
	if candidates[0].Score >= 4 && margin >= 2 && independent >= 2 {
		result.Domain = candidates[0].Domain
		result.Confidence = min(0.99, 0.55+float64(candidates[0].Score)*0.07+float64(margin)*0.03)
		return result, nil
	}
	choices := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		choices = append(choices, candidate.Domain)
	}
	if len(choices) == 1 {
		for _, d := range []string{"legal", "coding", "integrated"} {
			if d != choices[0] {
				choices = append(choices, d)
			}
		}
	}
	return decision(result, "low confidence or conflicting signals", choices), nil
}

func explicit(domain, source, key string) (Result, error) {
	domain = strings.ToLower(strings.TrimSpace(domain))
	if err := validateDomain(domain); err != nil {
		return Result{}, err
	}
	return Result{Schema: Schema, Domain: domain, Source: source, Confidence: 1, ContextKey: key, Evidence: []Evidence{{Kind: "declaration", Source: source, Value: domain}}}, nil
}

func decision(result Result, reason string, choices []string) Result {
	result.NeedsDecision = true
	result.DecisionRequest = &DecisionRequest{Reason: reason, Choices: choices, Scope: result.ContextKey}
	return result
}

func validateDomain(domain string) error {
	if !domains[strings.ToLower(strings.TrimSpace(domain))] {
		return fmt.Errorf("unknown domain %q", domain)
	}
	return nil
}

func contextKey(path, task string) string {
	path = filepath.ToSlash(filepath.Clean(strings.TrimSpace(path)))
	sum := sha256.Sum256([]byte(path + "\x00" + strings.TrimSpace(task)))
	return "ctx:" + hex.EncodeToString(sum[:8])
}

func tokenContains(text, needle string) bool {
	for _, token := range strings.FieldsFunc(text, func(r rune) bool { return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9') }) {
		if token == needle {
			return true
		}
	}
	return false
}

func independentEvidence(evidence []Evidence, domain string) int {
	kinds := map[string]bool{}
	for _, e := range evidence {
		if e.Value == domain {
			kinds[e.Kind+"/"+e.Source] = true
		}
	}
	return len(kinds)
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func EncodeChoice(choice Choice) ([]byte, error) { return json.MarshalIndent(choice, "", "  ") }
