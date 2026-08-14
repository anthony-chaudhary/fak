package armbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const PassthroughSchema = "fak/armbench-caveman-passthrough/1"

type PassthroughOptions struct {
	InputDir, OutDir, BaseURL, APIKey, Model, Label        string
	Trials                                                 int
	InputPerMillion, OutputPerMillion, CacheReadPerMillion float64
}
type TokenEvidence struct {
	Input, Output, CacheWrite, CacheRead int
	ProviderFields                       map[string]any `json:",omitempty"`
}
type PassthroughCall struct {
	PromptID, Arm, Phase, Text, FinishReason string
	Trial                                    int
	TTFTMS, WallMS, FakOverheadMS            float64
	Usage                                    TokenEvidence
	CostUSD                                  *float64
	SemanticPass                             bool
	SemanticMissing                          []string
	Request                                  json.RawMessage
	RawSSE                                   string
}
type PassthroughManifest struct {
	Schema, Source, Revision, RunLabel, ProviderEndpoint, Model string
	Trials                                                      int
	ExactModel                                                  bool
	Features                                                    map[string]any
	Hashes                                                      map[string]string
	Calls                                                       []PassthroughCall
	Summary                                                     []PassthroughSummary
	CacheVerdict, CostVerdict, Conclusion                       string
}
type PassthroughSummary struct {
	Arm        string
	Cold, Warm PassthroughAggregate
}
type PassthroughAggregate struct {
	Calls, SemanticPassed, Input, Output, CacheWrite, CacheRead int
	MedianTTFTMS, MedianWallMS, MedianFakOverheadMS             float64
	CostUSD                                                     *float64
}

type benchProxy struct{ server *httptest.Server }

func newBenchProxy(upstream string) (*benchProxy, error) {
	u, e := url.Parse(strings.TrimRight(upstream, "/"))
	if e != nil {
		return nil, e
	}
	rp := httputil.NewSingleHostReverseProxy(u)
	rp.ErrorHandler = func(w http.ResponseWriter, r *http.Request, e error) { http.Error(w, e.Error(), 502) }
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		w.Header().Set("X-Fak-Benchmark-Mode", "isolated-passthrough")
		rp.ServeHTTP(w, r)
		_ = start
	}))
	return &benchProxy{s}, nil
}
func (p *benchProxy) Close() { p.server.Close() }

func RunCavemanPassthrough(ctx context.Context, o PassthroughOptions) (PassthroughManifest, error) {
	if o.Trials < 2 {
		o.Trials = 3
	}
	if o.Model == "" {
		return PassthroughManifest{}, fmt.Errorf("model required")
	}
	hashes, e := verifyCavemanInputs(o.InputDir)
	if e != nil {
		return PassthroughManifest{}, e
	}
	var corpus cavemanCorpus
	pf, re := os.ReadFile(filepath.Join(o.InputDir, "prompts.json"))
	if re != nil {
		return PassthroughManifest{}, re
	}
	e = json.Unmarshal(pf, &corpus)
	prompts := corpus.Prompts
	if e != nil {
		return PassthroughManifest{}, e
	}
	skill, e := os.ReadFile(filepath.Join(o.InputDir, "SKILL.md"))
	if e != nil {
		return PassthroughManifest{}, e
	}
	proxy, e := newBenchProxy(o.BaseURL)
	if e != nil {
		return PassthroughManifest{}, e
	}
	defer proxy.Close()
	m := PassthroughManifest{Schema: PassthroughSchema, Source: "JuliusBrussee/caveman", Revision: CavemanRevision, RunLabel: o.Label, ProviderEndpoint: sanitizeEndpoint(o.BaseURL), Model: o.Model, Trials: o.Trials, ExactModel: o.Model == CavemanModel, Hashes: hashes, Features: map[string]any{"policy": false, "context_shedding": false, "output_transforms": false, "model_routing": false, "local_semantic_cache": false, "passthrough_proxy": "in-process httputil reverse proxy with no response store", "shared_region": "provider prompt_cache_key only; request content unchanged"}}
	arms := []struct {
		name, style  string
		proxy, cache bool
	}{{"direct-normal", "normal", false, false}, {"direct-caveman", "caveman", false, false}, {"fak-passthrough-normal", "normal", true, false}, {"fak-passthrough-caveman", "caveman", true, false}, {"fak-provider-cache-only-normal", "normal", true, true}, {"fak-provider-cache-only-caveman", "caveman", true, true}}
	type result struct {
		order int
		call  PassthroughCall
		err   error
	}
	ch := make(chan result, len(arms)*len(prompts)*o.Trials)
	for ai, a := range arms {
		for pi, p := range prompts {
			go func(ai, pi int, a struct {
				name, style  string
				proxy, cache bool
			}, p CavemanPrompt) {
				for trial := 1; trial <= o.Trials; trial++ {
					sys := ""
					if a.style == "caveman" {
						sys = string(skill)
					}
					base := o.BaseURL
					if a.proxy {
						base = proxy.server.URL
					}
					c, err := streamCall(ctx, base, o.APIKey, o.Model, p, sys, a.name, trial, a.cache)
					if err == nil && (o.InputPerMillion > 0 || o.OutputPerMillion > 0) {
						cost := (float64(c.Usage.Input-c.Usage.CacheRead)*o.InputPerMillion + float64(c.Usage.CacheRead)*o.CacheReadPerMillion + float64(c.Usage.Output)*o.OutputPerMillion) / 1e6
						c.CostUSD = &cost
					}
					ch <- result{(ai*len(prompts)+pi)*o.Trials + trial - 1, c, err}
					if err != nil {
						return
					}
				}
			}(ai, pi, a, p)
		}
	}
	results := make([]result, 0, len(arms)*len(prompts)*o.Trials)
	for range arms {
		for range prompts {
			for trial := 1; trial <= o.Trials; trial++ {
				r := <-ch
				if r.err != nil {
					return m, fmt.Errorf("provider call: %w", r.err)
				}
				results = append(results, r)
			}
		}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].order < results[j].order })
	for _, r := range results {
		m.Calls = append(m.Calls, r.call)
	}
	m.Summary = summarizePassthrough(m.Calls)
	reads := 0
	for _, c := range m.Calls {
		reads += c.Usage.CacheRead
	}
	if reads > 0 {
		m.CacheVerdict = "WITNESSED: provider usage reported cache-read tokens"
	} else {
		m.CacheVerdict = "NOT-YET: provider returned no nonzero cache-read token evidence; no cache-hit claim is made"
	}
	if o.InputPerMillion > 0 || o.OutputPerMillion > 0 {
		m.CostVerdict = "WITNESSED from supplied provider price inputs and provider token usage"
	} else {
		m.CostVerdict = "NOT-YET: endpoint supplied no bill and no explicit pricing inputs were provided"
	}
	m.Conclusion = conclusion(m)
	b, _ := json.MarshalIndent(m, "", "  ")
	if e = os.MkdirAll(o.OutDir, 0755); e != nil {
		return m, e
	}
	if e = os.WriteFile(filepath.Join(o.OutDir, "manifest.json"), append(b, '\n'), 0644); e != nil {
		return m, e
	}
	return m, nil
}
func streamCall(ctx context.Context, base, key, model string, p CavemanPrompt, system, arm string, trial int, cache bool) (PassthroughCall, error) {
	msgs := []map[string]string{}
	if system != "" {
		msgs = append(msgs, map[string]string{"role": "system", "content": system})
	}
	msgs = append(msgs, map[string]string{"role": "user", "content": p.Prompt})
	body := map[string]any{"model": model, "messages": msgs, "temperature": 0, "max_tokens": 4096, "stream": false}
	if cache {
		h := sha256.Sum256([]byte(system))
		body["prompt_cache_key"] = "fak-armbench-" + hex.EncodeToString(h[:8])
	}
	reqb, _ := json.Marshal(body)
	req, e := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(base, "/")+"/chat/completions", bytes.NewReader(reqb))
	if e != nil {
		return PassthroughCall{}, e
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	start := time.Now()
	resp, e := (&http.Client{Timeout: 3 * time.Minute}).Do(req)
	if e != nil {
		return PassthroughCall{}, e
	}
	defer resp.Body.Close()
	raw, e := io.ReadAll(resp.Body)
	if e != nil {
		return PassthroughCall{}, e
	}
	if resp.StatusCode/100 != 2 {
		return PassthroughCall{}, fmt.Errorf("provider %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	c := PassthroughCall{PromptID: p.ID, Arm: arm, Trial: trial, Phase: "warm", Request: reqb, RawSSE: string(raw)}
	if trial == 1 {
		c.Phase = "cold"
	}
	var x struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Finish string `json:"finish_reason"`
		} `json:"choices"`
		Usage map[string]any `json:"usage"`
	}
	if e = json.Unmarshal(raw, &x); e != nil {
		return c, e
	}
	if len(x.Choices) != 1 {
		return c, fmt.Errorf("provider returned %d choices", len(x.Choices))
	}
	c.Text = x.Choices[0].Message.Content
	c.FinishReason = x.Choices[0].Finish
	c.Usage = parseUsage(x.Usage)
	c.WallMS = float64(time.Since(start).Microseconds()) / 1000
	c.TTFTMS = 0 // endpoint did not provide streaming timestamps; do not fabricate TTFT
	c.SemanticPass, c.SemanticMissing = semanticGate(p.ID, c.Text)
	return c, nil
}
func parseUsage(u map[string]any) TokenEvidence {
	n := func(k string) int { v, _ := u[k].(float64); return int(v) }
	t := TokenEvidence{Input: n("prompt_tokens"), Output: n("completion_tokens"), ProviderFields: u}
	if d, ok := u["prompt_tokens_details"].(map[string]any); ok {
		if v, ok := d["cached_tokens"].(float64); ok {
			t.CacheRead = int(v)
		}
	}
	if v, ok := u["cache_read_input_tokens"].(float64); ok {
		t.CacheRead = int(v)
	}
	if v, ok := u["cache_creation_input_tokens"].(float64); ok {
		t.CacheWrite = int(v)
	}
	return t
}
func summarizePassthrough(cs []PassthroughCall) []PassthroughSummary {
	arms := map[string][]PassthroughCall{}
	for _, c := range cs {
		arms[c.Arm] = append(arms[c.Arm], c)
	}
	names := make([]string, 0, len(arms))
	for n := range arms {
		names = append(names, n)
	}
	sort.Strings(names)
	out := []PassthroughSummary{}
	for _, n := range names {
		var cold, warm []PassthroughCall
		for _, c := range arms[n] {
			if c.Phase == "cold" {
				cold = append(cold, c)
			} else {
				warm = append(warm, c)
			}
		}
		out = append(out, PassthroughSummary{n, aggregate(cold), aggregate(warm)})
	}
	return out
}
func aggregate(cs []PassthroughCall) PassthroughAggregate {
	a := PassthroughAggregate{Calls: len(cs)}
	var tt, wa, ov []float64
	cost := 0.
	priced := len(cs) > 0
	for _, c := range cs {
		a.Input += c.Usage.Input
		a.Output += c.Usage.Output
		a.CacheRead += c.Usage.CacheRead
		a.CacheWrite += c.Usage.CacheWrite
		if c.SemanticPass {
			a.SemanticPassed++
		}
		tt = append(tt, c.TTFTMS)
		wa = append(wa, c.WallMS)
		ov = append(ov, c.FakOverheadMS)
		if c.CostUSD == nil {
			priced = false
		} else {
			cost += *c.CostUSD
		}
	}
	a.MedianTTFTMS = medianFloat(tt)
	a.MedianWallMS = medianFloat(wa)
	a.MedianFakOverheadMS = medianFloat(ov)
	if priced {
		a.CostUSD = &cost
	}
	return a
}
func medianFloat(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sort.Float64s(v)
	return v[len(v)/2]
}
func conclusion(m PassthroughManifest) string {
	if !strings.HasPrefix(m.CacheVerdict, "WITNESSED") {
		return "NOT-YET: provider evidence cannot establish value from fak after Caveman against tuned direct controls."
	}
	return "See per-arm cold/warm summaries; value is established only where cache-read savings exceed measured fak overhead and correctness remains passing."
}
