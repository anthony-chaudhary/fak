package guardcompile

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/stablejson"
)

const PromptSchema = "fak.guard.compile.v1"

// Request is the bounded authoring input sent to exactly one model extractor.
type Request struct {
	Transcript string
	Intent     string
	Tool       string
	Field      string
}

// Extraction is the model-authored candidate. It is never installed directly.
type Extraction struct {
	DenyRegex string `json:"deny_regex"`
	Reason    string `json:"reason"`
	Severity  string `json:"severity"`
	Fix       string `json:"fix"`
}

// Candidate is the validated, review-only artifact and proposed manifest bytes.
type Candidate struct {
	Schema   string          `json:"schema"`
	Tool     string          `json:"tool"`
	Field    string          `json:"field"`
	Rule     Extraction      `json:"rule"`
	Manifest json.RawMessage `json:"manifest"`
}

// Extractor is the sole model seam. Compile calls it exactly once.
type Extractor interface {
	Extract(prompt string) ([]byte, error)
}

type ExtractorFunc func(string) ([]byte, error)

func (f ExtractorFunc) Extract(prompt string) ([]byte, error) { return f(prompt) }

// Prompt renders the complete extraction contract. The model returns JSON only.
func Prompt(req Request) (string, error) {
	if strings.TrimSpace(req.Transcript) == "" || strings.TrimSpace(req.Intent) == "" {
		return "", fmt.Errorf("transcript and intent are required")
	}
	if strings.ContainsAny(req.Intent, "\r\n") {
		return "", fmt.Errorf("intent must be one line")
	}
	if strings.TrimSpace(req.Tool) == "" || strings.TrimSpace(req.Field) == "" {
		return "", fmt.Errorf("tool and field are required")
	}
	return fmt.Sprintf("schema=%s\nExtract ONE literal RE2 guard rule as strict JSON with keys deny_regex, reason, severity, fix. severity is block or warn. Do not generalize beyond witnessed text.\ntool=%s\nfield=%s\nintent=%s\ntranscript:\n%s", PromptSchema, req.Tool, req.Field, req.Intent, req.Transcript), nil
}

// Compile invokes extractor exactly once, validates its fixed artifact, and
// appends it to the root arg_rules array in a proposed manifest. It never writes
// the policy file or mutates manifest.
func Compile(req Request, manifest []byte, extractor Extractor) (Candidate, error) {
	if extractor == nil {
		return Candidate{}, fmt.Errorf("extractor is required")
	}
	prompt, err := Prompt(req)
	if err != nil {
		return Candidate{}, err
	}
	raw, err := extractor.Extract(prompt)
	if err != nil {
		return Candidate{}, fmt.Errorf("extract rule: %w", err)
	}

	var rule Extraction
	dec := json.NewDecoder(bytes.NewReader(stripFence(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&rule); err != nil {
		return Candidate{}, fmt.Errorf("invalid extraction: %w", err)
	}
	if err := requireJSONEOF(dec); err != nil {
		return Candidate{}, fmt.Errorf("invalid extraction: %w", err)
	}
	rule.DenyRegex = strings.TrimSpace(rule.DenyRegex)
	if rule.DenyRegex == "" {
		return Candidate{}, fmt.Errorf("invalid deny_regex: empty")
	}
	if _, err := regexp.Compile(rule.DenyRegex); err != nil {
		return Candidate{}, fmt.Errorf("invalid deny_regex: %w", err)
	}
	rule.Reason = strings.TrimSpace(rule.Reason)
	rule.Severity = strings.ToLower(strings.TrimSpace(rule.Severity))
	rule.Fix = strings.TrimSpace(rule.Fix)
	if rule.Reason == "" || rule.Fix == "" || (rule.Severity != "block" && rule.Severity != "warn") {
		return Candidate{}, fmt.Errorf("reason, fix, and severity block|warn are required")
	}

	var doc map[string]any
	manifestDecoder := json.NewDecoder(bytes.NewReader(manifest))
	if err := manifestDecoder.Decode(&doc); err != nil {
		return Candidate{}, fmt.Errorf("manifest: %w", err)
	}
	if err := requireJSONEOF(manifestDecoder); err != nil {
		return Candidate{}, fmt.Errorf("manifest: %w", err)
	}
	if _, ok := doc["version"].(string); !ok {
		return Candidate{}, fmt.Errorf("manifest has no version")
	}
	rules, ok := doc["arg_rules"].([]any)
	if !ok && doc["arg_rules"] != nil {
		return Candidate{}, fmt.Errorf("manifest arg_rules is not an array")
	}

	tool := strings.TrimSpace(req.Tool)
	field := strings.TrimSpace(req.Field)
	proposed := map[string]any{
		"tool": tool, "arg": field, "deny_regex": rule.DenyRegex,
		"reason": rule.Reason, "fix": rule.Fix,
	}
	if rule.Severity == "warn" {
		proposed["advisory"] = true
	}
	doc["arg_rules"] = append(rules, proposed)
	out, err := stablejson.Marshal(doc)
	if err != nil {
		return Candidate{}, err
	}
	if _, err := policy.Parse(out); err != nil {
		return Candidate{}, fmt.Errorf("proposed policy: %w", err)
	}
	return Candidate{Schema: PromptSchema, Tool: tool, Field: field, Rule: rule, Manifest: out}, nil
}

func requireJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}

func stripFence(b []byte) []byte {
	s := strings.TrimSpace(string(b))
	if strings.HasPrefix(s, "```") {
		if i := strings.IndexByte(s, '\n'); i >= 0 {
			s = s[i+1:]
		}
		s = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "```"))
	}
	return []byte(s)
}

// ProposedDiff is a bounded human-review surface; no patch is applied automatically.
func ProposedDiff(path string, before, after []byte) string {
	if bytes.Equal(before, after) {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "--- %s\n+++ %s (proposed; not applied)\n", path, path)
	oldLines := strings.Split(strings.TrimSuffix(string(before), "\n"), "\n")
	newLines := strings.Split(strings.TrimSuffix(string(after), "\n"), "\n")
	for _, line := range oldLines {
		fmt.Fprintf(&b, "-%s\n", line)
	}
	for _, line := range newLines {
		fmt.Fprintf(&b, "+%s\n", line)
	}
	return b.String()
}
