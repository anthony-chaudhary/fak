package toolpriorbench

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
	"sort"
	"strings"
	"time"
)

const LedgerVersion = "fak.tool-prior-ledger/v1"

type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}
type Variant struct {
	ID        string `json:"id"`
	Class     string `json:"class"`
	Name      string `json:"name"`
	Semantics string `json:"semantics"`
	Accepted  bool   `json:"accepted"`
	Reason    string `json:"reason,omitempty"`
}
type CorpusCase struct {
	ID     string `json:"id"`
	Prompt string `json:"prompt"`
	Query  string `json:"query"`
}
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}
type Trial struct {
	Variant     string          `json:"variant"`
	Case        string          `json:"case"`
	Attempt     int             `json:"attempt"`
	Selected    string          `json:"selected,omitempty"`
	Arguments   json.RawMessage `json:"arguments,omitempty"`
	SelectionOK bool            `json:"selection_ok"`
	ArgumentsOK bool            `json:"arguments_ok"`
	LatencyMS   int64           `json:"latency_ms"`
	Usage       Usage           `json:"usage"`
	Request     json.RawMessage `json:"request"`
	Response    json.RawMessage `json:"response"`
	Error       string          `json:"error,omitempty"`
}
type Metrics struct {
	Trials            int     `json:"trials"`
	SelectionAccuracy float64 `json:"selection_accuracy"`
	ArgumentValidity  float64 `json:"argument_validity"`
	CorrectionTurns   int     `json:"correction_turns"`
	MeanLatencyMS     float64 `json:"mean_latency_ms"`
	PromptTokens      int     `json:"prompt_tokens"`
	CompletionTokens  int     `json:"completion_tokens"`
}
type Compatibility struct {
	Variant string   `json:"variant"`
	Model   string   `json:"model"`
	Harness string   `json:"harness"`
	Verdict string   `json:"verdict"`
	Reason  string   `json:"reason,omitempty"`
	Metrics *Metrics `json:"metrics,omitempty"`
}
type Ledger struct {
	Version          string          `json:"version"`
	Date             string          `json:"date"`
	Provider         string          `json:"provider"`
	Endpoint         string          `json:"endpoint"`
	Model            string          `json:"model"`
	SnapshotDigest   string          `json:"snapshot_digest"`
	SystemPrompt     string          `json:"system_prompt"`
	Schema           json.RawMessage `json:"schema"`
	Variants         []Variant       `json:"variants"`
	Corpus           []CorpusCase    `json:"corpus"`
	Trials           []Trial         `json:"trials"`
	Compatibility    []Compatibility `json:"compatibility"`
	Baseline         string          `json:"tuned_no_alias_baseline"`
	HonestConclusion string          `json:"honest_conclusion"`
}
type Config struct {
	Endpoint, Model string
	Timeout         time.Duration
	Client          *http.Client
	Now             func() time.Time
}

var schema = json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","description":"Exact repository search query from the user."}},"required":["query"],"additionalProperties":false}`)
var variants = []Variant{{"private", "private_canonical", "fak_repo_lookup_7c", "repository_text_search_v1", true, ""}, {"provider", "provider_native_style", "search_repository", "repository_text_search_v1", true, ""}, {"api", "popular_api_style", "github_search_code", "repository_text_search_v1", true, ""}, {"command", "popular_command_style", "grep", "repository_text_search_v1", true, ""}, {"shell", "harness_builtin", "functions.shell_command", "arbitrary_shell_v1", false, "rejected: shell_command has argv/shell side effects and result semantics that do not match repository_text_search_v1"}}
var corpus = []CorpusCase{{"literal", "Find the exact text registration_digest in the repository. Use the repository search tool once; do not answer from memory.", "registration_digest"}, {"path", "Find the exact text internal/toolcatalog in the repository. Use the repository search tool once; do not answer from memory.", "internal/toolcatalog"}, {"flaglike", "Find the exact text --expose in the repository. Use the repository search tool once; do not answer from memory.", "--expose"}}

const systemPrompt = "You are evaluating tool selection. Call exactly one appropriate tool. Copy the user's requested search query exactly into the query field. Do not answer in prose."

func Schema() json.RawMessage { return append(json.RawMessage(nil), schema...) }
func Variants() []Variant     { return append([]Variant(nil), variants...) }
func Corpus() []CorpusCase    { return append([]CorpusCase(nil), corpus...) }

// Digest identifies the exact accepted variants and schema used by the corpus.
func Digest() string { d, _ := snapshotDigest(variants, schema); return d }
func Run(ctx context.Context, cfg Config) (Ledger, error) {
	if cfg.Endpoint == "" || cfg.Model == "" {
		return Ledger{}, errors.New("endpoint and model are required")
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 90 * time.Second
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{Timeout: cfg.Timeout}
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	digest, err := snapshotDigest(variants, schema)
	if err != nil {
		return Ledger{}, err
	}
	l := Ledger{LedgerVersion, cfg.Now().UTC().Format("2006-01-02"), "ollama", cfg.Endpoint, cfg.Model, digest, systemPrompt, Schema(), Variants(), Corpus(), nil, nil, "private", "not yet: measured results have not been evaluated"}
	for _, v := range variants {
		if !v.Accepted {
			l.Compatibility = append(l.Compatibility, Compatibility{v.ID, cfg.Model, "codex", "rejected_semantic_mismatch", v.Reason, nil})
			continue
		}
		for _, c := range corpus {
			t := call(ctx, cfg, v, c, 1, nil)
			l.Trials = append(l.Trials, t)
			if !t.SelectionOK || !t.ArgumentsOK {
				x := "Your prior call was invalid. Call the repository search tool now with the exact query requested by the user."
				l.Trials = append(l.Trials, call(ctx, cfg, v, c, 2, &x))
			}
		}
	}
	ms := summarize(l.Trials)
	b := ms["private"]
	for _, v := range variants {
		if !v.Accepted {
			continue
		}
		m := ms[v.ID]
		verdict, reason := "compatible", "same semantics; measured against tuned private-name baseline"
		if m.SelectionAccuracy < b.SelectionAccuracy || m.ArgumentValidity < b.ArgumentValidity {
			verdict, reason = "not_preferred", "quality is below the tuned private-name baseline"
		}
		l.Compatibility = append(l.Compatibility, Compatibility{v.ID, cfg.Model, "ollama_openai_tool_shape", verdict, reason, &m})
	}
	l.HonestConclusion = conclusion(ms)
	return l, nil
}
func call(ctx context.Context, cfg Config, v Variant, c CorpusCase, attempt int, correction *string) Trial {
	messages := []map[string]string{{"role": "system", "content": systemPrompt}, {"role": "user", "content": c.Prompt}}
	if correction != nil {
		messages = append(messages, map[string]string{"role": "user", "content": *correction})
	}
	decoy := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`)
	tools := []Tool{{"function", Function{v.Name, "Search repository text for an exact query. Use for requests to find text in a repository.", Schema()}}, {"function", Function{"read_repository_file", "Read a known repository file by exact path. Do not use to search for text.", decoy}}, {"function", Function{"list_repository_tree", "List repository paths. Do not use to search file contents.", json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`)}}}
	req := map[string]any{"model": cfg.Model, "messages": messages, "tools": tools, "stream": false, "options": map[string]any{"temperature": 0, "seed": 6820}}
	rawReq, _ := json.Marshal(req)
	tr := Trial{Variant: v.ID, Case: c.ID, Attempt: attempt, Request: rawReq}
	started := time.Now()
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.Endpoint, "/")+"/api/chat", bytes.NewReader(rawReq))
	if err != nil {
		tr.Error = err.Error()
		return tr
	}
	r.Header.Set("Content-Type", "application/json")
	resp, err := cfg.Client.Do(r)
	tr.LatencyMS = time.Since(started).Milliseconds()
	if err != nil {
		tr.Error = err.Error()
		return tr
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	tr.Response = append(json.RawMessage(nil), raw...)
	if err != nil {
		tr.Error = err.Error()
		return tr
	}
	if resp.StatusCode/100 != 2 {
		tr.Error = fmt.Sprintf("http %d", resp.StatusCode)
		return tr
	}
	var out struct {
		Message struct {
			ToolCalls []struct {
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
		Prompt int `json:"prompt_eval_count"`
		Eval   int `json:"eval_count"`
	}
	if err = json.Unmarshal(raw, &out); err != nil {
		tr.Error = err.Error()
		return tr
	}
	tr.Usage = Usage{out.Prompt, out.Eval}
	if len(out.Message.ToolCalls) == 0 {
		return tr
	}
	tr.Selected = out.Message.ToolCalls[0].Function.Name
	tr.Arguments = out.Message.ToolCalls[0].Function.Arguments
	tr.SelectionOK = tr.Selected == v.Name
	var a struct {
		Query string `json:"query"`
	}
	if json.Unmarshal(tr.Arguments, &a) == nil && a.Query == c.Query {
		tr.ArgumentsOK = true
	}
	return tr
}
func snapshotDigest(vs []Variant, s json.RawMessage) (string, error) {
	a := []Variant{}
	for _, v := range vs {
		if v.Accepted {
			a = append(a, v)
		}
	}
	sort.Slice(a, func(i, j int) bool { return a[i].ID < a[j].ID })
	b, e := json.Marshal(struct {
		Version  string          `json:"version"`
		Variants []Variant       `json:"variants"`
		Schema   json.RawMessage `json:"schema"`
	}{LedgerVersion, a, s})
	if e != nil {
		return "", e
	}
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:]), nil
}
func summarize(ts []Trial) map[string]Metrics {
	o := map[string]Metrics{}
	for _, t := range ts {
		m := o[t.Variant]
		m.Trials++
		if t.SelectionOK {
			m.SelectionAccuracy++
		}
		if t.ArgumentsOK {
			m.ArgumentValidity++
		}
		if t.Attempt > 1 {
			m.CorrectionTurns++
		}
		m.MeanLatencyMS += float64(t.LatencyMS)
		m.PromptTokens += t.Usage.PromptTokens
		m.CompletionTokens += t.Usage.CompletionTokens
		o[t.Variant] = m
	}
	for k, m := range o {
		if m.Trials > 0 {
			m.SelectionAccuracy /= float64(m.Trials)
			m.ArgumentValidity /= float64(m.Trials)
			m.MeanLatencyMS /= float64(m.Trials)
		}
		o[k] = m
	}
	return o
}
func conclusion(ms map[string]Metrics) string {
	b := ms["private"]
	best, score := "private", b.SelectionAccuracy+b.ArgumentValidity
	for k, m := range ms {
		if x := m.SelectionAccuracy + m.ArgumentValidity; x > score {
			best, score = k, x
		}
	}
	if best == "private" {
		return "No measured alias improves selection plus argument validity over the tuned private-name baseline; keep canonical names unless a model-specific table says otherwise."
	}
	return fmt.Sprintf("Variant %s improves measured selection plus argument validity over the tuned private-name baseline for this exact model and corpus only.", best)
}
