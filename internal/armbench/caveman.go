package armbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	CavemanRevision   = "c72984e4392c7a154e55c11dbf445f01ce5c35d4"
	CavemanModel      = "claude-sonnet-4-20250514"
	cavemanPromptsSHA = "773e557f9187363c44e7e5aae2d27268720bcd8772865e119825078b06da93d7"
	cavemanSkillSHA   = "daf9cec496ebd039809d8236f99f17fa1b4beaadf8ce4e2d532d0da51d70afce"
	cavemanRunSHA     = "530a387918418713e64ded97794f41a1ffe6a01e833a69d2cb447bf4640facce"
)

type CavemanOptions struct {
	InputDir, OutDir, BaseURL, APIKey, Model, Label string
	Trials                                          int
}
type cavemanCorpus struct {
	Version int             `json:"version"`
	Prompts []CavemanPrompt `json:"prompts"`
}
type CavemanPrompt struct{ ID, Category, Prompt string }
type CavemanUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
type CavemanCall struct {
	PromptID, Arm      string
	Trial              int
	Text, FinishReason string
	Usage              CavemanUsage
	SemanticPass       bool
	Missing            []string `json:",omitempty"`
	Raw                json.RawMessage
}
type CavemanSummary struct {
	Arm                           string
	MedianOutputByPrompt          map[string]int
	AverageMedian                 float64
	SemanticPassed, SemanticTotal int
}
type CavemanPacket struct {
	Schema, Source, Revision, RunLabel, ProviderEndpoint, RequestedModel, ResolvedModel string
	ExactModel                                                                          bool
	Temperature                                                                         int
	MaxOutputTokens, Trials                                                             int
	Hashes                                                                              map[string]string
	Calls                                                                               []CavemanCall
	Summary                                                                             []CavemanSummary
	Upstream                                                                            map[string]any
	GeneratedAt                                                                         string
}

func RunCaveman(ctx context.Context, o CavemanOptions) (CavemanPacket, error) {
	if o.Trials == 0 {
		o.Trials = 3
	}
	if o.Trials != 3 {
		return CavemanPacket{}, errors.New("caveman control requires exactly three trials")
	}
	if o.Model == "" {
		o.Model = CavemanModel
	}
	if o.Label == "" {
		o.Label = "exact-model"
	}
	hashes, err := verifyCavemanInputs(o.InputDir)
	if err != nil {
		return CavemanPacket{}, err
	}
	var corpus cavemanCorpus
	b, _ := os.ReadFile(filepath.Join(o.InputDir, "prompts.json"))
	if err = json.Unmarshal(b, &corpus); err != nil {
		return CavemanPacket{}, err
	}
	skill, _ := os.ReadFile(filepath.Join(o.InputDir, "SKILL.md"))
	p := CavemanPacket{Schema: "fak/armbench-caveman-native/1", Source: "JuliusBrussee/caveman", Revision: CavemanRevision, RunLabel: o.Label, ProviderEndpoint: sanitizeEndpoint(o.BaseURL), RequestedModel: CavemanModel, ResolvedModel: o.Model, ExactModel: o.Model == CavemanModel, Temperature: 0, MaxOutputTokens: 4096, Trials: 3, Hashes: hashes, Upstream: map[string]any{"average_normal": 1214, "average_caveman": 294, "saved_percent": 65, "quality": "unevaluated"}, GeneratedAt: time.Now().UTC().Format(time.RFC3339)}
	client := &http.Client{Timeout: 5 * time.Minute}
	for _, q := range corpus.Prompts {
		for _, arm := range []string{"normal", "caveman"} {
			sys := "You are a helpful assistant."
			if arm == "caveman" {
				sys = string(skill)
			}
			for trial := 1; trial <= 3; trial++ {
				c, e := callOpenAI(ctx, client, o, sys, q, arm, trial)
				if e != nil {
					return p, fmt.Errorf("%s %s trial %d: %w", q.ID, arm, trial, e)
				}
				c.SemanticPass, c.Missing = semanticGate(q.ID, c.Text)
				p.Calls = append(p.Calls, c)
			}
		}
	}
	p.Summary = summarizeCaveman(p.Calls)
	if err = os.MkdirAll(o.OutDir, 0755); err != nil {
		return p, err
	}
	out, _ := json.MarshalIndent(p, "", "  ")
	err = os.WriteFile(filepath.Join(o.OutDir, "manifest.json"), append(out, '\n'), 0644)
	return p, err
}
func verifyCavemanInputs(dir string) (map[string]string, error) {
	expected := map[string]string{"prompts.json": cavemanPromptsSHA, "SKILL.md": cavemanSkillSHA, "run.py": cavemanRunSHA}
	got := map[string]string{}
	for n, want := range expected {
		b, e := os.ReadFile(filepath.Join(dir, n))
		if e != nil {
			return nil, e
		}
		s := sha256.Sum256(b)
		h := hex.EncodeToString(s[:])
		if h != want {
			return nil, fmt.Errorf("%s hash %s, want %s", n, h, want)
		}
		got[n] = h
	}
	return got, nil
}
func callOpenAI(ctx context.Context, h *http.Client, o CavemanOptions, system string, q CavemanPrompt, arm string, trial int) (CavemanCall, error) {
	body := map[string]any{"model": o.Model, "temperature": 0, "max_completion_tokens": 4096, "messages": []map[string]string{{"role": "system", "content": system}, {"role": "user", "content": q.Prompt}}}
	b, _ := json.Marshal(body)
	req, _ := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(o.BaseURL, "/")+"/chat/completions", bytes.NewReader(b))
	req.Header.Set("Authorization", "Bearer "+o.APIKey)
	req.Header.Set("Content-Type", "application/json")
	r, e := h.Do(req)
	if e != nil {
		return CavemanCall{}, e
	}
	defer r.Body.Close()
	raw, _ := io.ReadAll(r.Body)
	if r.StatusCode/100 != 2 {
		return CavemanCall{}, fmt.Errorf("provider status %d: %s", r.StatusCode, strings.TrimSpace(string(raw)))
	}
	var x struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Finish string `json:"finish_reason"`
		} `json:"choices"`
		Usage CavemanUsage `json:"usage"`
	}
	if e = json.Unmarshal(raw, &x); e != nil {
		return CavemanCall{}, e
	}
	if len(x.Choices) != 1 {
		return CavemanCall{}, errors.New("provider returned no single choice")
	}
	return CavemanCall{PromptID: q.ID, Arm: arm, Trial: trial, Text: x.Choices[0].Message.Content, FinishReason: x.Choices[0].Finish, Usage: x.Usage, Raw: raw}, nil
}
func semanticGate(id, text string) (bool, []string) {
	rules := map[string][][]string{
		"react-rerender": {{"referential", "reference", "identity"}, {"memo", "usememo"}}, "auth-middleware-fix": {{"seconds"}, {"date.now", "1000"}}, "postgres-pool": {{"pool"}, {"timeout"}, {"error"}}, "git-rebase-merge": {{"history"}, {"rebase"}, {"merge"}}, "async-refactor": {{"async"}, {"await"}, {"not found"}}, "microservices-monolith": {{"measure", "profile"}, {"complexity", "operational"}}, "pr-security-review": {{"sql injection"}, {"parameter", "placeholder", "where id = $1", "where id = ?"}, {"error"}}, "docker-multi-stage": {{"from"}, {"as build", "as builder"}, {"npm ci"}, {"cmd"}}, "race-condition-debug": {{"atomic", "transaction", "returning"}, {"update"}}, "error-boundary": {{"componentdidcatch"}, {"getderivedstatefromerror"}, {"retry"}, {"log", "console.error", "onerror"}}}
	low := strings.ToLower(text)
	missing := []string{}
	for _, alternatives := range rules[id] {
		ok := false
		for _, s := range alternatives {
			if strings.Contains(low, s) {
				ok = true
				break
			}
		}
		if !ok {
			missing = append(missing, strings.Join(alternatives, "|"))
		}
	}
	return len(missing) == 0, missing
}
func summarizeCaveman(cs []CavemanCall) []CavemanSummary {
	out := []CavemanSummary{}
	for _, arm := range []string{"normal", "caveman"} {
		by := map[string][]int{}
		pass, total := 0, 0
		for _, c := range cs {
			if c.Arm == arm {
				by[c.PromptID] = append(by[c.PromptID], c.Usage.CompletionTokens)
				total++
				if c.SemanticPass {
					pass++
				}
			}
		}
		med := map[string]int{}
		sum := 0
		for id, v := range by {
			sort.Ints(v)
			med[id] = v[len(v)/2]
			sum += med[id]
		}
		out = append(out, CavemanSummary{arm, med, float64(sum) / float64(len(med)), pass, total})
	}
	return out
}
func sanitizeEndpoint(s string) string {
	if s == "" {
		return ""
	}
	return strings.TrimRight(s, "/")
}
